# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl.py — Integrated production implementation
# =============================================================================
#
# SwiGLU activation (Shazeer 2020 "GLU Variants Improve Transformer"):
#   output = SiLU(x0) * x1 = (x0 * sigmoid(x0)) * x1
#   where (x0, x1) = input.chunk(2, dim)
#
# Pipeline: view(input) -> cast(BF16->FP32) -> sigmoid(x0) -> mul(x0*sig)
#           -> mul(silu*x1) -> cast(FP32->BF16) -> assemble(output)
#
# Decomposition: L0 (module_count=1), vector-only (no matmul/reduction).
#
# Layer A/G: Imports
# Layer B:   Tile constants (T0, T1, T2 = 16, 16, 16)
# Layer J:   JIT entry with inlined Layer I body
# Layer K:   Host wrapper (swi_glu_impl) + swi_glu_wrapper alias
#
# Template: pypto-op-develop/templates/impl_template.py
# Tiling:   vec-only set_vec_tile_shapes(16, 16, 16) from DESIGN.md 3.2.5
# Loop:     3-level nested (rows -> cols -> depth), unroll_list=[1] on innermost
# =============================================================================

# --- Layer A/G: Imports ----------------------------------------------------
import pypto
import torch
import torch_npu  # noqa: F401  (NPU device init required before JIT)

# --- Layer B: Tile constants (compile-time known, OL48) --------------------
T0 = 16  # vec tile outer (rows)
T1 = 48  # vec tile mid (cols) — perf-tuned (+21% vs T1=32)
T2 = 16  # vec tile inner (depth, BF16 16-element alignment)

# --- Layer J: JIT entry (Layer I body inlined) ----------------------------
#
# @pypto.frontend.jit decorator (OL01 strict literal) with NPU run_mode.
# Tensor args first, then non-tensor args (OL26).
# All dynamic axes annotated with pypto.DYNAMIC (OL29, OL43).
# Body inlines the Layer I loop/view/compute/assemble recipe.
#
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
def swi_glu_kernel_npu(
    input: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    split_dim: int,  # compile-time constant: 0, 1, or 2
):
    """SwiGLU 3D elementwise kernel: output = SiLU(x0) * x1.

    Tiles the output shape with (T0, T1, T2) windows.
    For each tile: views x0/x1 from input (offset by half along split_dim),
    casts to FP32, computes sigmoid + 2x mul, casts back to BF16, assembles.
    """
    # Layer I - Tile setup
    pypto.set_vec_tile_shapes(T0, T1, T2)

    # SymbolicScalars from output shape (OL29 / OL43)
    D0 = output.shape[0]  # SymbolicScalar
    D1 = output.shape[1]  # SymbolicScalar
    D2 = output.shape[2]  # SymbolicScalar

    # half_offset = output.shape[split_dim] (SymbolicScalar)
    # Equals input.shape[split_dim] // 2.
    # Used as the x1 view offset along the split dimension.
    # split_dim is a compile-time Python int, safe for if/else (OL06 compliant).
    if split_dim == 0:
        half = D0  # SymbolicScalar
    elif split_dim == 1:
        half = D1  # SymbolicScalar
    else:  # split_dim == 2
        half = D2  # SymbolicScalar

    # Loop bounds (SymbolicScalar, ceiling division)
    rows_loop = (D0 + T0 - 1) // T0   # SymbolicScalar
    cols_loop = (D1 + T1 - 1) // T1   # SymbolicScalar
    depth_loop = (D2 + T2 - 1) // T2  # SymbolicScalar

    # Layer I - Three-level nested loop
    # OL49: unroll_list only on innermost pypto.loop
    # OL56: unroll_list single value [1] (Stage 7 baseline)
    for r in pypto.loop(rows_loop, name="rows"):
        r_off = r * T0  # SymbolicScalar
        v_r = (D0 - r_off).min(T0)  # SymbolicScalar, always <= T0

        for c in pypto.loop(cols_loop, name="cols"):
            c_off = c * T1  # SymbolicScalar
            v_c = (D1 - c_off).min(T1)  # SymbolicScalar, always <= T1

            for d in pypto.loop(depth_loop, name="depth", unroll_list=[16]):
                d_off = d * T2  # SymbolicScalar
                v_d = (D2 - d_off).min(T2)  # SymbolicScalar, always <= T2
                vs = [v_r, v_c, v_d]  # valid_shape list

                # View x0: same offset as output tile
                # x0_tile: [T0,T1,T2] BF16
                x0_tile = pypto.view(
                    input, shape=[T0, T1, T2],
                    offsets=[r_off, c_off, d_off],
                    valid_shape=vs,
                )
                # x0_fp32: [T0,T1,T2] FP32
                x0_fp32 = pypto.cast(x0_tile, pypto.DT_FP32)

                # View x1: offset + half along split_dim
                # split_dim is compile-time const, safe for Python if/else
                if split_dim == 0:
                    x1_off = [half + r_off, c_off, d_off]
                elif split_dim == 1:
                    x1_off = [r_off, half + c_off, d_off]
                else:  # split_dim == 2
                    x1_off = [r_off, c_off, half + d_off]

                # x1_tile: [T0,T1,T2] BF16
                x1_tile = pypto.view(
                    input, shape=[T0, T1, T2],
                    offsets=x1_off,
                    valid_shape=vs,
                )
                # x1_fp32: [T0,T1,T2] FP32
                x1_fp32 = pypto.cast(x1_tile, pypto.DT_FP32)

                # Compute SwiGLU in FP32
                # sig: [T0,T1,T2] FP32
                sig = pypto.sigmoid(x0_fp32)
                # silu: [T0,T1,T2] FP32 (SiLU = x * sigmoid(x))
                silu = x0_fp32 * sig
                # result_fp32: [T0,T1,T2] FP32 (gated output)
                result_fp32 = silu * x1_fp32

                # Cast back to BF16
                # result: [T0,T1,T2] BF16
                result = pypto.cast(result_fp32, pypto.DT_BF16)

                # Writeback (Layer I + Layer H)
                # Padding region auto-zeros through arithmetic:
                #   x0_pad=0 -> sigmoid(0)=0.5 -> silu=0*0.5=0 -> result=0*x1_pad=0
                pypto.assemble(result, [r_off, c_off, d_off], output)

# --- Layer K: Host wrapper ------------------------------------------------
#
# Only 4 responsibilities:
#   1. Normalize dim -> split_dim (0, 1, or 2)
#   2. Compute output shape (split dim halved)
#   3. Allocate output via torch.empty with explicit dtype / device (OL58)
#   4. Invoke JIT exactly ONCE - NO Python `for ... in range(...)` (OL45)
#
def swi_glu_impl(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """SwiGLU host wrapper - production entry point.

    Computes output = SiLU(x0) * x1, where (x0, x1) = input.chunk(2, dim).

    Args:
        input: BF16 3D tensor [D0, D1, D2] on NPU device; split dim size must be even.
        dim:   split dimension (default -1).

    Returns:
        output: BF16 [D0, D1, D2] with dim-axis halved.
    """
    ndim = input.ndim  # 3 for our cases

    # Normalize dim into [0, ndim)
    split_dim_normalized = dim % ndim  # -1 -> 2 for 3D; Python int, not SymbolicScalar

    # Compute output shape (input shape with split dim halved)
    out_shape = list(input.shape)  # Python list
    out_shape[split_dim_normalized] //= 2

    # Allocate output buffer via torch (Layer K contract: torch.* only, OL58)
    output = torch.empty(
        *out_shape,
        dtype=input.dtype,
        device=input.device,
    )

    # Single JIT call (Layer K contract: no `for ... in range(...)`, OL45)
    swi_glu_kernel_npu(input, output, split_dim_normalized)

    return output

# Layer K alias for E2E test harness
# (test_swi_glu.py imports swi_glu_wrapper)
def swi_glu_wrapper(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Op-scoped wrapper (delegates to swi_glu_impl) for E2E test harness."""
    return swi_glu_impl(input, dim=dim)

def swi_glu_wrapper(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    return swi_glu_impl(input, dim=dim)
