# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl.py
# Integrated production implementation for swi_glu (SwiGLU activation).
#
# Pipeline: view(x0, x1) → sigmoid(x0) → mul(x0*sig) → mul(silu*x1) → assemble
#
# Layer A-L design from DESIGN.md §4.2 pseudocode.
# Template: pypto-op-develop/templates/impl_template.py
# Tiling: (T0, T1) = (64, 64) — Stage 7 Round 1 optimization (from baseline (32,32))
# FP32 direct path — no cast needed (c2 all cases are float32)
# =============================================================================

# ─── Layer A: Imports ──────────────────────────────────────────
import pypto
import torch
import torch_npu  # noqa: F401  NPU device init required before JIT

# ─── Layer B: Tile constants (compile-time known, OL48) ───────
T0 = 64   # view tile rows — Round 1: 32→64 (2×, UB budget 49,152 B ≤ 65,536 B)
T1 = 64   # view tile cols — Round 1: 32→64 (2×, UB budget 49,152 B ≤ 65,536 B)

# ─── Layer I + J: JIT entry with kernel body inlined ──────────
#
# @pypto.frontend.jit (OL01 strict literal) with NPU run_mode.
# Tensor args precede non-tensor args (OL26). All dynamic axes annotated
# with pypto.DYNAMIC (OL29, OL43). Body inlines the Layer I loop/view/
# compute/assemble recipe directly.
#
# The kernel tiles the output shape [M_out, N_out] with (T0, T1) windows.
# For each tile it reads two chunks from the input (x0, x1 split along
# split_dim), computes SwiGLU in FP32, and assembles the FP32 result back.
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
def swi_glu_kernel_npu(
    input: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),   # [M, N]
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),  # [M_out, N_out]
    split_dim: int,  # 0 or 1 (compile-time constant)
):
    """SwiGLU FP32 elementwise kernel with tiled chunk + FP32 compute."""
    # ── Layer I — Tile setup ──
    pypto.set_vec_tile_shapes(T0, T1)                         # vec tile (64, 64)

    # SymbolicScalar from dynamic output axes (OL29 / OL43)
    M_out = output.shape[0]                                    # SymbolicScalar
    N_out = output.shape[1]                                    # SymbolicScalar

    # half_offset: where x1 starts in input along split_dim
    if split_dim == 0:                                         # compile-time branch
        half_offset = M_out                                    # SymbolicScalar
    else:
        half_offset = N_out                                    # SymbolicScalar

    # Outer / inner loop bounds (SymbolicScalar, ceiling division)
    rows_loop = (M_out + T0 - 1) // T0                         # SymbolicScalar
    cols_loop = (N_out + T1 - 1) // T1                         # SymbolicScalar

    # ── Layer I — Nested tile loops ──
    for r in pypto.loop(rows_loop, name="rows"):               # outer row loop (no unroll_list, OL49)
        for c in pypto.loop(cols_loop, name="cols", unroll_list=[8]):  # inner col loop — Round 3: [4]→[8]
            r_off = r * T0                                     # SymbolicScalar
            c_off = c * T1                                     # SymbolicScalar

            # Valid shape clamping for tail tiles (.min() ensures ≤ tile)
            v_r = (M_out - r_off).min(T0)                      # SymbolicScalar, ≤ T0
            v_c = (N_out - c_off).min(T1)                      # SymbolicScalar, ≤ T1

            # ── Step 1a: view x0 from input (same offset as output tile) ──
            x0_tile = pypto.view(
                input, shape=[T0, T1],
                offsets=[r_off, c_off],
                valid_shape=[v_r, v_c],
            )                                                  # [T0,T1] FP32

            # ── Step 1b: view x1 (shifted by half_offset along split_dim) ──
            if split_dim == 0:                                 # compile-time branch
                x1_tile = pypto.view(
                    input, shape=[T0, T1],
                    offsets=[half_offset + r_off, c_off],
                    valid_shape=[v_r, v_c],
                )                                              # [T0,T1] FP32
            else:                                              # split_dim == 1
                x1_tile = pypto.view(
                    input, shape=[T0, T1],
                    offsets=[r_off, half_offset + c_off],
                    valid_shape=[v_r, v_c],
                )                                              # [T0,T1] FP32

            # ── Step 2: sigmoid (FP32 only, input already FP32) ──
            sig = pypto.sigmoid(x0_tile)                       # [T0,T1] FP32

            # ── Step 3: SiLU = x0 * sigmoid(x0) ──
            silu = x0_tile * sig                               # [T0,T1] FP32

            # ── Step 4: result = silu * x1 ──
            result = silu * x1_tile                            # [T0,T1] FP32

            # ── Step 5: writeback ──
            # Padding region arithmetic guarantees zero:
            #   x0_pad=0 → sigmoid(0)=0.5 → silu=0*0.5=0 → result=0*x1=0
            pypto.assemble(result, [r_off, c_off], output)

# ─── Layer K: Host wrapper ─────────────────────────────────────
#
# Only 4 responsibilities:
#   1. Normalize dim → split_dim (0 or 1)
#   2. Compute output shape (input shape with split dim halved)
#   3. Allocate output via torch.empty with explicit dtype / device (OL58)
#   4. Invoke JIT exactly ONCE — NO Python `for ... in range(...)` (OL45)
#
# Public entry point: swi_glu_impl(input, dim=-1) → torch.Tensor
def swi_glu_impl(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """SwiGLU host wrapper — production entry point.

    Computes output = SiLU(x0) * x1, where x0, x1 = input.chunk(2, dim=dim).

    Args:
        input: FP32 2D tensor [M, N] on NPU device; split dim size must be even.
        dim:   split dimension (-1 default, normalized to 0 or 1).

    Returns:
        output: FP32 [M_out, N_out] with split dim halved.
    """
    M, N = input.shape
    ndim = input.ndim                                          # 2 for our cases

    # Normalize dim into [0, ndim)
    split_dim_normalized = dim % ndim                          # -1→1 for 2D; 0→0; 1→1

    if split_dim_normalized == 0:
        M_out, N_out = M // 2, N
        split_dim = 0
    else:
        M_out, N_out = M, N // 2
        split_dim = 1

    # Allocate output buffer via torch (NOT pypto.zeros — Layer K contract, OL58)
    output = torch.empty(
        M_out, N_out,
        dtype=input.dtype,
        device=input.device,
    )

    # Single JIT call (Layer K contract, OL45)
    swi_glu_kernel_npu(input, output, split_dim)

    return output

# ─── Backward-compatible wrapper alias ────────────────────────
# Required by lint OL08: module-level function ending with _wrapper.
# Production entry point is swi_glu_impl; this alias satisfies the
# naming convention for test harness and import compatibility.
def swi_glu_wrapper(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Production wrapper alias (delegates to swi_glu_impl)."""
    return swi_glu_impl(input, dim=dim)
