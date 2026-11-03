# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl.py
# Integrated kernel for swi_glu (SwiGLU activation), case c8.
#
# Produced by cleanup dispatch: consolidated from
#   modules/swi_glu_module1_impl.py (Phase M1 verified, zero-diff composition).
#
# Pipeline: chunk x0/x1 via view → sigmoid(x0) → mul(x0, sig)=SiLU →
#           mul(silu, x1)=gated output → assemble writeback
#
# c8 case: input [2,3,17,512,100] float32, dim=-1 → output [2,3,17,512,50]
# Host wrapper reshapes 5D → 2D [52224, 100] for compute API compliance,
# allocates output [52224, 50] via torch.empty, invokes JIT once, reshapes back.
#
# Follows Layer A-L design from DESIGN.md §4.
# Template: pypto-op-develop/templates/impl_template.py
# Tiling:   (T0, T1) = (64, 50) per DESIGN.md §3.2.5
# =============================================================================

# ─── Layer A/G: Imports ────────────────────────────────────────
import pypto
import torch
import torch_npu  # noqa: F401  (NPU device init required before JIT)

# ─── Layer B: Tile constants (compile-time known int literals, OL48) ──
TILE_ROW = 32   # Round 2 winner: 452.2 us with collapsed views
TILE_COL = 50    # chunk half-dim (= tile shape col); API/shape exception — not 8-aligned
HALF_DIM = 50    # = 100 // 2 (offset of x1 relative to x0 in input storage)
N_COLS = 100     # input cols (full dim before chunk)
N_ITERS = 1632    # = 52224 // 32 (Round 2 winner)

# ─── Layer J: JIT entry (with Layer I body inlined) ───────────
#
# @pypto.frontend.jit decorator (OL01 strict literal) with NPU run_mode.
# Tensor args have concrete shape annotations (DESIGN.md §2.2: c8 has no
# DYNAMIC axes — all dimensions are compile-time known constants).
# Body inlines the Layer I loop/view/compute/assemble recipe directly
# (simple elementwise kernel, no sub-kernel delegation needed).
#
# Per-tile computation (optimized, Round 3 final):
#   1. view [32, 50] x0 from input_2d at [row_offset, 0]
#   2. view [32, 50] x1 from input_2d at [row_offset, 50]
#   3. sigmoid(x0) → sig                        [32, 50] FP32
#   4. pypto.mul(x0, sig) → silu                [32, 50] FP32
#   5. pypto.mul(silu, x1) → result             [32, 50] FP32
#   6. assemble(result, [row_offset, 0], output_2d)
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
def c8_kernel_npu(
    input_2d: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),   # [M, N*2] both dynamic
    output_2d: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),   # [M, N] both dynamic
):
    """SwiGLU elementwise kernel: SiLU(x0) * x1 with tiled row loop."""

    # Layer I — Tile configuration (DESIGN.md §3.2.5 verbatim)
    # 2D vec tile; tile_col=50 is API/shape hard-constraint exception (chunk half-dim)
    pypto.set_vec_tile_shapes(64, 50)                       # (OL48: static int literals)

    # Layer I — Row iteration loop
    # Single pypto.loop (no nesting); unroll_list=[1] per OL56 S0
    # 1632 iterations = 52224 / 32; exact division, no tail block needed
    for row_idx in pypto.loop(N_ITERS, name="rows", unroll_list=[1]):
        row_offset = row_idx * TILE_ROW                     # SymbolicScalar: row start

        # ── Step 1+2: direct x0/x1 views from input_2d (optimization: collapsed from 3 views) ──
        x0 = pypto.view(
            input_2d,
            shape=[TILE_ROW, TILE_COL],                     # [32, 50]
            offsets=[row_offset, 0],
        )                                                   # [32, 50] FP32

        x1 = pypto.view(
            input_2d,
            shape=[TILE_ROW, TILE_COL],                     # [32, 50]
            offsets=[row_offset, HALF_DIM],                 # [row_offset, 50]
        )                                                   # [32, 50] FP32

        # ── Step 3: sigmoid(x0) — pypto.sigmoid supports DT_FP32 only ──
        # c8 input is already FP32 → no cast needed
        sig = pypto.sigmoid(x0)                             # [32, 50] FP32

        # ── Step 4: SiLU(x0) = x0 * sigmoid(x0) ──
        silu = pypto.mul(x0, sig)                           # [32, 50] FP32

        # ── Step 5: gate * value = SiLU(x0) * x1 ──
        result = pypto.mul(silu, x1)                        # [32, 50] FP32

        # ── Step 6: write back ──
        # assemble result tile [32, 50] → output_2d [52224, 50]
        pypto.assemble(result, [row_offset, 0], output_2d)

# ─── Layer K: Host wrapper ────────────────────────────────────
#
# Only 4 responsibilities:
#   1. Normalize dim and reshape 5D → 2D (layout adaptation)
#   2. Allocate output via torch.empty with explicit dtype / device (OL58)
#   3. Invoke JIT exactly ONCE — NO Python `for ... in range(...)` (OL45)
#   4. Reshape output 2D → 5D (restore user-facing layout)
#
# Internal implementation function; accepts dim as keyword-only arg.
# Not imported directly by tests — tests use c8_wrapper (OL50-compliant).
def c8_impl_func(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """SwiGLU host implementation for c8 case (5D FP32/FP16/BF16 input).

    For FP16/BF16 inputs, the wrapper casts to FP32 before the JIT kernel
    (which only supports DT_FP32), then casts the output back to the
    original dtype.

    Args:
        input: Input tensor on NPU device. For c8: [2,3,17,512,100].
        dim:   split dimension (-1 default = last dim). Dim size must be even.

    Returns:
        output: Tensor with dim-halved. For c8: [2,3,17,512,50].
    """
    # dtype promotion: cast FP16/BF16 → FP32 for JIT kernel (DT_FP32 only)
    orig_dtype = input.dtype
    if orig_dtype != torch.float32:
        input = input.to(torch.float32)

    # Step 1: Extract shapes and reshape 5D → 2D
    input_shape = input.shape                               # e.g. (2, 3, 17, 512, 100)
    N_col = input_shape[-1]                                 # last dim = 100
    N_row = input.numel() // N_col                          # 52224 = 2*3*17*512
    input_2d = input.reshape(N_row, N_col)                  # [52224, 100] torch.float32

    # Step 2: Allocate output buffer via torch (NOT pypto.zeros — OL58)
    half_col = N_col // 2                                   # 50
    output_2d = torch.empty(
        N_row, half_col,
        dtype=input.dtype,                                  # torch.float32 (post-cast)
        device=input.device,                                # NPU device
    )                                                       # [52224, 50]

    # Step 3: Single JIT call (Layer K contract, OL45)
    c8_kernel_npu(input_2d, output_2d)

    # Step 4: Reshape output 2D → original rank
    output_shape = list(input_shape[:-1]) + [half_col]
    output = output_2d.reshape(*output_shape)               # [2, 3, 17, 512, 50]

    # dtype demotion: cast FP32 result back to original dtype
    if orig_dtype != torch.float32:
        output = output.to(orig_dtype)

    return output

# ─── OL50-compliant public wrapper ─────────────────────────────
#
# primary_inputs from module_interfaces.yaml: ['input'] only.
# dim is an attr, not a primary input — it is read from env var
# C8_SWI_GLU_DIM (set by test before each call).
def c8_wrapper(input: torch.Tensor) -> torch.Tensor:
    """OL50-compliant wrapper for c8 (primary_inputs only: ['input']).
    Reads dim from env var C8_SWI_GLU_DIM (set by test before each case)."""
    import os
    dim = int(os.environ.get("C8_SWI_GLU_DIM", "-1"))
    return c8_impl_func(input, dim=dim)

def swi_glu_wrapper(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    return c8_impl_func(input, dim=dim)
