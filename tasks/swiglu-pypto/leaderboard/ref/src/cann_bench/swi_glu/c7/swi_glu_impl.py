# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl.py  —  SwiGLU activation kernel (c7 — 3D FP32)
#
# Layer A–L: SwiGLU activation kernel for 3D FP32 input (c7 case).
#
# Pipeline: pypto.view(x0/x1, offsets, valid_shape)
#         → pypto.sigmoid(x0)
#         → mul(x0, sig)          [SiLU]
#         → mul(silu, x1)         [gate]
#         → pypto.assemble(result, offsets, output)
#
# Tiling:   pypto.set_vec_tile_shapes(1, 64, 128)  (Stage 7 R2: from (1,16,32))
# Loop:     3-level nested pypto.loop over [D0, D1, D2]
#           unroll_list=[4] on innermost (OL56 Stage 7 multi-value, R3)
# Split:    split_dim is Python int from host normalization; if/elif/else at
#           compile-time selects which axis gets the half_offset shift.
# C7 FP32-only path — no cast required (sigmoid FP32-only matches input dtype).
# Non-aligned dim1=1023 handled by valid_shape clamping on tail tiles.
#
# Template: pypto-op-develop/templates/impl_template.py
# Design:   custom/swi_glu/c7/DESIGN.md
# =============================================================================

# ─── Imports ───────────────────────────────────────────────────
import pypto
import torch
import torch_npu  # noqa: F401  (NPU device init required before JIT)

# ─── Tile Constants (compile-time known, OL48) ─────────────────
# Stage 7 R2: (1,16,32)→(1,64,128), 16x fewer iterations (8192→512)
# T0=1 (batch-collapsed), T1=64 (seq), T2=128 (embed/2)
# FP32 alignment: T2=128 is 8-elem aligned (128/8=16)
# UB per-op: binary mul = 1*64*128*4*3 = 96 KB ≤ 196 KB
T0 = 1   # tile dim 0 (batch-collapsed)
T1 = 64  # tile dim 1 (sequence) — Stage 7 Round 1: 16→64
T2 = 128 # tile dim 2 (embed/2, FP32 8-elem aligned) — Stage 7 Round 2: 64→128, halve D2 iterations

# ─── JIT Kernel ────────────────────────────────────────────────
#
# @pypto.frontend.jit decorator (OL01 strict literal) with NPU run_mode.
# Tensor args precede non-tensor args (OL26). All dynamic axes annotated
# with pypto.DYNAMIC (OL29, OL43). Body inlines the loop/view/compute/
# assemble recipe directly (no separate _kernel_impl helper).
#
# The kernel tiles the OUTPUT shape with [T0, T1, T2] windows. For each
# tile it reads two chunks from the input (x0 at same offset, x1 shifted by
# half_size along split_dim), computes SwiGLU in FP32, and assembles the
# result back into the output buffer.
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
def swi_glu_kernel_npu(
    input: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),   # [D0, D1, D2_in]
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),  # [D0, D1, D2]
    split_dim: int,  # compile-time constant: 0/1/2 from host normalization
):
    """SwiGLU: output = SiLU(x0) * x1 where x0, x1 = chunk(input, 2, split_dim)."""

    # ─── Tile setup ────────────────────────────────────────────
    pypto.set_vec_tile_shapes(T0, T1, T2)              # tile = [1, 64, 128]

    # SymbolicScalar from dynamic output axes (OL29/OL43)
    D0 = output.shape[0]   # DYNAMIC axis 0
    D1 = output.shape[1]   # DYNAMIC axis 1
    D2 = output.shape[2]   # DYNAMIC axis 2

    # half_size: x1 offset relative to x0 in input storage, along split_dim.
    # split_dim is Python int → safe for compile-time if/else (OL-safe).
    if split_dim == 0:
        half_size = D0        # SymbolicScalar
    elif split_dim == 1:
        half_size = D1        # SymbolicScalar
    else:  # split_dim == 2
        half_size = D2        # SymbolicScalar

    # Loop bounds: ceiling division (SymbolicScalar)
    N0 = (D0 + T0 - 1) // T0   # = D0 (since T0=1)
    N1 = (D1 + T1 - 1) // T1   # e.g. ceil(1023/64) = 16
    N2 = (D2 + T2 - 1) // T2   # e.g. ceil(2048/128) = 16

    # ─── 3D nested loop ────────────────────────────────────────
    # Outer loops: no unroll_list (OL49)
    # Innermost: unroll_list=[4] (OL56 Stage 7 multi-value, R3)
    for i0 in pypto.loop(N0, name="d0"):                 # SymbolicScalar i0
        for i1 in pypto.loop(N1, name="d1"):             # SymbolicScalar i1
            for i2 in pypto.loop(N2, name="d2", unroll_list=[4]):  # innermost (Stage 7 R3)

                # Tile offsets (SymbolicScalar)
                off0 = i0 * T0                           # = i0 (T0=1)
                off1 = i1 * T1
                off2 = i2 * T2

                # Valid shape per axis (SymbolicScalar, clamped ≤ tile size)
                v0 = (D0 - off0).min(T0)                 # ≤ T0=1 always
                v1 = (D1 - off1).min(T1)                 # ≤ T1 (critical for 1023 tail)
                v2 = (D2 - off2).min(T2)                 # ≤ T2

                # ── Step 1: view x0 from input ──
                # x0 chunk at same offset as the output tile (no shift)
                x0_tile = pypto.view(
                    input, shape=[T0, T1, T2],
                    offsets=[off0, off1, off2],
                    valid_shape=[v0, v1, v2],
                )                                                  # [T0,T1,T2] FP32

                # ── Step 2: view x1 from input ──
                # x1 chunk shifted by half_size along split_dim
                if split_dim == 0:
                    x1_tile = pypto.view(
                        input, shape=[T0, T1, T2],
                        offsets=[half_size + off0, off1, off2],
                        valid_shape=[v0, v1, v2],
                    )                                              # [T0,T1,T2] FP32
                elif split_dim == 1:
                    x1_tile = pypto.view(
                        input, shape=[T0, T1, T2],
                        offsets=[off0, half_size + off1, off2],
                        valid_shape=[v0, v1, v2],
                    )                                              # [T0,T1,T2] FP32
                else:  # split_dim == 2
                    x1_tile = pypto.view(
                        input, shape=[T0, T1, T2],
                        offsets=[off0, off1, half_size + off2],
                        valid_shape=[v0, v1, v2],
                    )                                              # [T0,T1,T2] FP32

                # ── Step 3: sigmoid(x0) — FP32 only ──
                sig = pypto.sigmoid(x0_tile)                       # [T0,T1,T2] FP32

                # ── Step 4: SiLU = x0 * sigmoid(x0) ──
                silu = x0_tile * sig                               # [T0,T1,T2] FP32

                # ── Step 5: gate = silu * x1 ──
                result = silu * x1_tile                            # [T0,T1,T2] FP32

                # ── Step 6: assemble writeback ──
                # Padding through arithmetic: x0 pad=0 → sigmoid(0)=0.5
                # → SiLU=0*0.5=0 → 0*x1=0 → safe to assemble full tile
                pypto.assemble(
                    result,
                    [off0, off1, off2],
                    output,
                )

# ─── Host Wrapper ──────────────────────────────────────────────
#
# Only 4 responsibilities:
#   1. Normalize dim → split_dim (0/1/2 for 3D)
#   2. Compute output shape (input shape with split dim halved)
#   3. Allocate output via torch.empty with explicit dtype / device
#   4. Invoke JIT exactly ONCE — NO Python `for ... in range(...)` (OL45)
def swi_glu_wrapper(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """SwiGLU production entry point.

    Computes: output = SiLU(x0) * x1, where x0, x1 = input.chunk(2, dim=dim).

    Args:
        input: FP32 3D tensor [D0, D1, D2_in] on NPU device.
        dim:   split dimension (default -1). Size along dim must be even.

    Returns:
        output: FP32 [D0, ..., dim_size/2, ..., D2_in] with split dim halved.
    """
    ndim = input.ndim  # 3 for c7 case

    # Normalize dim into [0, ndim)
    split_dim = dim % ndim  # e.g. -1 → 2 for 3D

    # Compute output shape: split_dim axis halved
    output_shape = list(input.shape)
    output_shape[split_dim] //= 2

    # Allocate output buffer via torch (OL58 requirement)
    output = torch.empty(*output_shape, dtype=input.dtype, device=input.device)

    # Single JIT call (Layer K contract, OL45 — NO Python for loop)
    swi_glu_kernel_npu(input, output, split_dim)

    return output

# ─── Public API alias ──────────────────────────────────────────
# Primary production entry point (README / test usage).
swi_glu_impl = swi_glu_wrapper
