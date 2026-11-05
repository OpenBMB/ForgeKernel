# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl.py  —  Integrated production implementation of SwiGLU (c5)
#
# Layer A–L: SwiGLU activation kernel for 4D FP32 input (c5 benchmark).
#
# Pipeline: pypto.view(x0/x1, offsets, valid_shape)
#         → pypto.sigmoid(x0)
#         → mul(x0, sig)          [SiLU]
#         → mul(silu, x1)         [gate]
#         → pypto.assemble(result, offsets, output)
#
# Tiling:   pypto.set_vec_tile_shapes(1, 16, 16, 32)  (DESIGN.md §3.2.5)
# Loop:     4-level nested pypto.loop over [D0, D1, D2, D3]
#           unroll_list=[1] on innermost (OL56 single-value, Stage 6 default)
# Split:    split_dim is Python int from host normalization; if/else at
#           compile-time selects which axis gets the half_offset shift.
# C5 FP32-only path — no cast required (sigmoid FP32-only matches input dtype).
#
# Template: pypto-op-develop/templates/impl_template.py
# Design:   custom/swi_glu/c5/DESIGN.md
# Contract: custom/swi_glu/c5/eval/module_interfaces.yaml
# =============================================================================

# ─── Layer A/G: Imports ────────────────────────────────────────
import pypto
import torch
import torch_npu  # noqa: F401  (NPU device init required before JIT)

# ─── Layer B: Tile Constants (compile-time known, OL48) ────────
# From DESIGN.md §3.2.5 (Stage 7 前 Tile shape 基线)
# Vec tile = (1, 16, 16, 32): T0=1 (batch-collapsed), T1/T2=16, T3=32
# FP32 alignment: T3=32 is 8-elem aligned (32/8=4)
# UB per-op: binary mul = 1*16*16*32*4*3 = 96 KB ≤ 192 KB ✓
T0 = 1   # tile dim 0 (batch-collapsed)
T1 = 16  # tile dim 1
T2 = 16  # tile dim 2
T3 = 32  # tile dim 3 (FP32 8-elem aligned)

# ─── Layer H + I + J: JIT entry (body inlined) ───────────────
#
# @pypto.frontend.jit decorator (OL01 strict literal) with NPU run_mode.
# Tensor args precede non-tensor args (OL26). All dynamic axes annotated
# with pypto.DYNAMIC (OL29, OL43). Body inlines the Layer I loop/view/
# compute/assemble recipe directly (no pypto.is_loop_begin/end — no need
# for @pypto.frontend.function or helper separation).
#
# The kernel tiles the OUTPUT shape with [T0, T1, T2, T3] windows. For each
# tile it reads two chunks from the input (x0 at same offset, x1 shifted by
# half_size along split_dim), computes SwiGLU in FP32, and assembles the
# result back.
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
def swi_glu_kernel_npu(
    input: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),   # [D0_in, D1_in, D2_in, D3_in]
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),  # [D0, D1, D2, D3]
    split_dim: int,  # compile-time constant: 0/1/2/3 from host normalization
):
    """SwiGLU: output = SiLU(x0) * x1 where x0, x1 = chunk(input, 2, split_dim)."""

    # ─── Layer I: Tile setup ───────────────────────────────────
    pypto.set_vec_tile_shapes(T0, T1, T2, T3)      # tile = [1, 16, 16, 32]

    # SymbolicScalar from dynamic output axes (OL29/OL43)
    D0 = output.shape[0]   # DYNAMIC axis 0
    D1 = output.shape[1]   # DYNAMIC axis 1
    D2 = output.shape[2]   # DYNAMIC axis 2
    D3 = output.shape[3]   # DYNAMIC axis 3

    # half_size: x1 offset relative to x0 in input storage, along split_dim.
    # split_dim is Python int → safe for compile-time if/else (OL-safe).
    if split_dim == 0:
        half_size = D0        # SymbolicScalar
    elif split_dim == 1:
        half_size = D1        # SymbolicScalar
    elif split_dim == 2:
        half_size = D2        # SymbolicScalar
    else:  # split_dim == 3
        half_size = D3        # SymbolicScalar

    # Loop bounds: ceiling division (SymbolicScalar)
    N0 = (D0 + T0 - 1) // T0   # = D0 (since T0=1)
    N1 = (D1 + T1 - 1) // T1
    N2 = (D2 + T2 - 1) // T2
    N3 = (D3 + T3 - 1) // T3

    # ─── Layer I: 4D nested loop ──────────────────────────────
    # Outer loops: no unroll_list (OL49)
    # Innermost: unroll_list=[1] (OL56 single-value, Stage 6 default)
    for i0 in pypto.loop(N0, name="d0"):                 # SymbolicScalar i0
        for i1 in pypto.loop(N1, name="d1"):             # SymbolicScalar i1
            for i2 in pypto.loop(N2, name="d2"):         # SymbolicScalar i2
                for i3 in pypto.loop(N3, name="d3", unroll_list=[1]):  # innermost

                    # Tile offsets (SymbolicScalar)
                    off0 = i0 * T0                       # = i0 (T0=1)
                    off1 = i1 * T1
                    off2 = i2 * T2
                    off3 = i3 * T3

                    # Valid shape per axis (SymbolicScalar, clamped ≤ tile size)
                    v0 = (D0 - off0).min(T0)             # ≤ T0=1 always
                    v1 = (D1 - off1).min(T1)             # ≤ T1
                    v2 = (D2 - off2).min(T2)             # ≤ T2
                    v3 = (D3 - off3).min(T3)             # ≤ T3

                    # ── Step 1: view x0 from input ──
                    # x0 chunk at same offset as the output tile (no shift)
                    x0_tile = pypto.view(
                        input, shape=[T0, T1, T2, T3],
                        offsets=[off0, off1, off2, off3],
                        valid_shape=[v0, v1, v2, v3],
                    )                                                  # [T0,T1,T2,T3] FP32

                    # ── Step 2: view x1 from input ──
                    # x1 chunk shifted by half_size along split_dim
                    if split_dim == 0:
                        x1_tile = pypto.view(
                            input, shape=[T0, T1, T2, T3],
                            offsets=[half_size + off0, off1, off2, off3],
                            valid_shape=[v0, v1, v2, v3],
                        )                                              # [T0,T1,T2,T3] FP32
                    elif split_dim == 1:
                        x1_tile = pypto.view(
                            input, shape=[T0, T1, T2, T3],
                            offsets=[off0, half_size + off1, off2, off3],
                            valid_shape=[v0, v1, v2, v3],
                        )                                              # [T0,T1,T2,T3] FP32
                    elif split_dim == 2:
                        x1_tile = pypto.view(
                            input, shape=[T0, T1, T2, T3],
                            offsets=[off0, off1, half_size + off2, off3],
                            valid_shape=[v0, v1, v2, v3],
                        )                                              # [T0,T1,T2,T3] FP32
                    else:  # split_dim == 3
                        x1_tile = pypto.view(
                            input, shape=[T0, T1, T2, T3],
                            offsets=[off0, off1, off2, half_size + off3],
                            valid_shape=[v0, v1, v2, v3],
                        )                                              # [T0,T1,T2,T3] FP32

                    # ── Step 3: sigmoid(x0) — FP32 only ──
                    sig = pypto.sigmoid(x0_tile)                       # [T0,T1,T2,T3] FP32

                    # ── Step 4: SiLU = x0 * sigmoid(x0) ──
                    silu = x0_tile * sig                               # [T0,T1,T2,T3] FP32

                    # ── Step 5: gate = silu * x1 ──
                    result = silu * x1_tile                            # [T0,T1,T2,T3] FP32

                    # ── Step 6: assemble writeback ──
                    # Padding through arithmetic: x0 pad=0 → sigmoid(0)=0.5
                    # → SiLU=0*0.5=0 → 0*x1=0 → safe to assemble full tile
                    pypto.assemble(
                        result,
                        [off0, off1, off2, off3],
                        output,
                    )

# ─── Layer K: Host wrapper ─────────────────────────────────────
#
# Only 4 responsibilities:
#   1. Normalize dim → split_dim (0/1/2/3 for 4D)
#   2. Compute output shape (input shape with split dim halved)
#   3. Allocate output via torch.empty with explicit dtype / device
#   4. Invoke JIT exactly ONCE — NO Python `for ... in range(...)` (OL45)
def swi_glu_wrapper(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """SwiGLU production entry point — host wrapper.

    Computes: output = SiLU(x0) * x1, where x0, x1 = input.chunk(2, dim=dim).

    Args:
        input: FP32 4D tensor [D0, D1, D2, D3] on NPU device.
        dim:   split dimension (default -1). Size along dim must be even.

    Returns:
        output: FP32 [D0, ..., dim_size/2, ..., D3] with split dim halved.
    """
    ndim = input.ndim  # 4 for c5 cases

    # Normalize dim into [0, ndim)
    split_dim = dim % ndim  # e.g. -1 → 3 for 4D

    # Compute output shape: split_dim axis halved
    output_shape = list(input.shape)
    output_shape[split_dim] //= 2

    # Allocate output buffer via torch (OL58 requirement)
    output = torch.empty(*output_shape, dtype=input.dtype, device=input.device)

    # Single JIT call (Layer K contract, OL45 — NO Python for loop)
    swi_glu_kernel_npu(input, output, split_dim)

    return output

# ─── Public API alias ──────────────────────────────────────────
# Primary production entry point (README / test usage: from swi_glu_impl import swi_glu_impl).
swi_glu_impl = swi_glu_wrapper
