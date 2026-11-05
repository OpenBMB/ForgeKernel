# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl.py
# Integrated production implementation for swi_glu (SwiGLU activation).
#
# L0 path (module_count=1): single kernel covers the entire algorithm.
# Pipeline: input.chunk(2,dim) -> cast FP16->FP32 -> sigmoid(x0) ->
#           mul(x0*sig) -> mul(silu*x1) -> cast FP32->FP16 -> assemble
#
# Follows Layer A-L design from DESIGN.md.
# Template: pypto-op-develop/templates/impl_template.py
# Tiling:   (T0, T1) = (128, 128); Stage 7 tuned from (32, 64)
# =============================================================================

# ─── Layer A/G: Imports ────────────────────────────────────────
import pypto
import torch
import torch_npu  # noqa: F401  (NPU device init required before JIT)

# ─── Layer B: Tile constants (compile-time known, OL48) ────────
T0 = 128  # view tile rows (Stage 7: 32→128, reduce loop iterations 16×)
T1 = 128  # view tile cols (Stage 7: 64→128, reduce loop iterations 2×; FP16 16-elem aligned)

# ─── Layer J: JIT entry (with Layer I body inlined) ───────────
#
# @pypto.frontend.jit decorator (OL01 strict literal) with NPU run_mode.
# Tensor args precede non-tensor args (OL26). All dynamic axes annotated
# with pypto.DYNAMIC (OL29, OL43). Body inlines the Layer I loop/view/
# compute/assemble recipe directly.
#
# The kernel tiles the output shape with (T0, T1) windows. For each tile
# it reads two chunks from the input (x0 and x1, split along split_dim),
# computes SwiGLU in FP32, and assembles the FP16 result back.
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
def swi_glu_kernel_npu(
    input: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
    split_dim: int,  # 0 or 1 (compile-time constant)
):
    """SwiGLU elementwise kernel with tiled chunk + FP32 compute."""
    # Layer I — Tile setup
    pypto.set_vec_tile_shapes(T0, T1)

    # SymbolicScalar from dynamic output axes (OL29 / OL43)
    M_out = output.shape[0]   # DYNAMIC axis 0
    N_out = output.shape[1]   # DYNAMIC axis 1

    # half_offset = offset of x1 relative to x0 in input storage,
    # along the split dimension.
    if split_dim == 0:
        half_offset = M_out        # SymbolicScalar: rows axis
    else:
        half_offset = N_out        # SymbolicScalar: cols axis

    # Outer / inner loop bounds (SymbolicScalar, ceiling division)
    rows_loop = (M_out + T0 - 1) // T0
    cols_loop = (N_out + T1 - 1) // T1

    # Layer I — Outer row loop (no unroll_list — OL49)
    for r in pypto.loop(rows_loop, name="rows"):
        # Layer I — Inner col loop (Stage 7: unroll_list=[4] for parallel batching)
        for c in pypto.loop(cols_loop, name="cols", unroll_list=[4]):
            r_off = r * T0   # SymbolicScalar
            c_off = c * T1   # SymbolicScalar

            # Valid shape clamping for tail tiles (per-axis <= tile size)
            v_r = (M_out - r_off).min(T0)   # v_r <= T0 always
            v_c = (N_out - c_off).min(T1)   # v_c <= T1 always

            # ── Step 1-2: view x0 from input (same offset as output tile) ──
            x0_tile = pypto.view(
                input, shape=[T0, T1], offsets=[r_off, c_off],
                valid_shape=[v_r, v_c],
            )                                              # [T0,T1] FP16
            x0_fp32 = pypto.cast(x0_tile, pypto.DT_FP32)  # [T0,T1] FP32

            # ── Step 1-2: view x1 (shifted by half_offset along split_dim) ──
            if split_dim == 0:
                x1_tile = pypto.view(
                    input, shape=[T0, T1],
                    offsets=[half_offset + r_off, c_off],
                    valid_shape=[v_r, v_c],
                )                                          # [T0,T1] FP16
            else:
                x1_tile = pypto.view(
                    input, shape=[T0, T1],
                    offsets=[r_off, half_offset + c_off],
                    valid_shape=[v_r, v_c],
                )                                          # [T0,T1] FP16
            x1_fp32 = pypto.cast(x1_tile, pypto.DT_FP32)   # [T0,T1] FP32

            # ── Step 3: pypto.sigmoid (FP32 only) ──
            sig = pypto.sigmoid(x0_fp32)                   # [T0,T1] FP32

            # ── Step 4: SiLU = x0 * sigmoid(x0) ──
            silu = x0_fp32 * sig                           # [T0,T1] FP32

            # ── Step 5: silu * x1 ──
            result_fp32 = silu * x1_fp32                   # [T0,T1] FP32

            # ── Step 6: cast back to FP16 ──
            result = pypto.cast(result_fp32, pypto.DT_FP16)  # [T0,T1] FP16

            # ── Writeback (Layer I + Layer H) ──
            # Padding region guaranteed all-0.0 through the arithmetic:
            #   x0 pad = 0.0 -> sigmoid(0.0) = 0.5 -> silu = 0.0*0.5 = 0.0
            #   x1 pad = 0.0 -> silu*0.0 = 0.0
            # Full-shape assemble over tail-tile boundary cells with zeros,
            # which are either still inside output bounds (no adjacent tile
            # to collide with) or the first iteration of an adjacent tile
            # re-covers via its own valid writeback.
            pypto.assemble(result, [r_off, c_off], output)

# ─── Layer K: Host wrapper ────────────────────────────────────
#
# Only 4 responsibilities:
#   1. Normalize dim -> split_dim (0 or 1)
#   2. Compute output shape (input shape with split dim halved)
#   3. Allocate output via torch.empty with explicit dtype / device
#   4. Invoke JIT exactly ONCE — NO Python `for ... in range(...)` (OL45)
#
# Public entry point: swi_glu_impl(input, dim=-1) -> torch.Tensor
# Also aliased as swi_glu_module1_wrapper for backward compat with
# per-phase test harness (test_swi_glu.py imports from module1_impl).
def swi_glu_impl(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """SwiGLU host wrapper — production entry point.

    Computes output = SiLU(x0) * x1, where x0, x1 = input.chunk(2, dim=dim).

    Args:
        input: FP16 2D tensor [M, N] on NPU device; split dim must be even.
        dim:   split dimension (-1 default, normalised to 0 or 1).

    Returns:
        output: FP16 [M_out, N_out] with dim-dim halved.
    """
    M, N = input.shape
    ndim = input.ndim  # 2 for our cases

    # Normalize dim into [0, ndim)
    split_dim_normalized = dim % ndim  # -1 -> 1 for 2D; 0 -> 0; 1 -> 1

    if split_dim_normalized == 0:
        M_out, N_out = M // 2, N
        split_dim = 0
    else:   # dim = -1 or 1 (last axis for 2D)
        M_out, N_out = M, N // 2
        split_dim = 1

    # Allocate output buffer via torch (NOT pypto.zeros / etc.)
    output = torch.empty(
        M_out, N_out,
        dtype=input.dtype,
        device=input.device,
    )

    # Single JIT call (Layer K contract, OL45)
    swi_glu_kernel_npu(input, output, split_dim)

    return output

# Backward-compatible wrapper so that per-phase test harness can also
# import from this file under the module1 wrapper name.
def swi_glu_module1_wrapper(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Module-scoped wrapper (delegates to swi_glu_impl) for per-phase test harness."""
    return swi_glu_impl(input, dim=dim)

def swi_glu_wrapper(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    return swi_glu_impl(input, dim=dim)
