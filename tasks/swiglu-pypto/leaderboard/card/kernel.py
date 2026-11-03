# Consolidated from CANN-Bench's public PyPTO SwiGlu c1..c8 classes.

# The initial candidate is public code; subsequent changes are agent-made.


# ===== Public reference class c1 =====
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl_c1.py
# Integrated production implementation for swi_glu (SwiGLU activation).
#
# L0 path (module_count=1): single kernel covers the entire algorithm.
# Pipeline: input.chunk(2,dim) -> cast FP16->FP32 -> sigmoid(x0) ->
#           mul(x0*sig) -> mul(silu*x1) -> cast FP32->FP16 -> assemble
#
# Follows Layer A-L design from DESIGN.md.
# Template: pypto-op-develop/templates/impl_template.py
# Tiling:   (T0_C1, T1_C1) = (128, 128); Stage 7 tuned from (32, 64)
# =============================================================================

# ─── Layer A/G: Imports ────────────────────────────────────────
import pypto
import torch
import torch_npu  # noqa: F401  (NPU device init required before JIT)

# ─── Layer B: Tile constants (compile-time known, OL48) ────────
T0_C1 = 128  # view tile rows (Stage 7: 32→128, reduce loop iterations 16×)
T1_C1 = 128  # view tile cols (Stage 7: 64→128, reduce loop iterations 2×; FP16 16-elem aligned)

# ─── Layer J: JIT entry (with Layer I body inlined) ───────────
#
# @pypto.frontend.jit decorator (OL01 strict literal) with NPU run_mode.
# Tensor args precede non-tensor args (OL26). All dynamic axes annotated
# with pypto.DYNAMIC (OL29, OL43). Body inlines the Layer I loop/view/
# compute/assemble recipe directly.
#
# The kernel tiles the output shape with (T0_C1, T1_C1) windows. For each tile
# it reads two chunks from the input (x0 and x1, split along split_dim),
# computes SwiGLU in FP32, and assembles the FP16 result back.
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
def swi_glu_kernel_npu_c1(
    input: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
    split_dim: int,  # 0 or 1 (compile-time constant)
):
    """SwiGLU elementwise kernel with tiled chunk + FP32 compute."""
    # Layer I — Tile setup
    pypto.set_vec_tile_shapes(T0_C1, T1_C1)

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
    rows_loop = (M_out + T0_C1 - 1) // T0_C1
    cols_loop = (N_out + T1_C1 - 1) // T1_C1

    # Layer I — Outer row loop (no unroll_list — OL49)
    for r in pypto.loop(rows_loop, name="rows"):
        # Layer I — Inner col loop (Stage 7: unroll_list=[4] for parallel batching)
        for c in pypto.loop(cols_loop, name="cols", unroll_list=[4]):
            r_off = r * T0_C1   # SymbolicScalar
            c_off = c * T1_C1   # SymbolicScalar

            # Valid shape clamping for tail tiles (per-axis <= tile size)
            v_r = (M_out - r_off).min(T0_C1)   # v_r <= T0_C1 always
            v_c = (N_out - c_off).min(T1_C1)   # v_c <= T1_C1 always

            # ── Step 1-2: view x0 from input (same offset as output tile) ──
            x0_tile = pypto.view(
                input, shape=[T0_C1, T1_C1], offsets=[r_off, c_off],
                valid_shape=[v_r, v_c],
            )                                              # [T0_C1,T1_C1] FP16
            x0_fp32 = pypto.cast(x0_tile, pypto.DT_FP32)  # [T0_C1,T1_C1] FP32

            # ── Step 1-2: view x1 (shifted by half_offset along split_dim) ──
            if split_dim == 0:
                x1_tile = pypto.view(
                    input, shape=[T0_C1, T1_C1],
                    offsets=[half_offset + r_off, c_off],
                    valid_shape=[v_r, v_c],
                )                                          # [T0_C1,T1_C1] FP16
            else:
                x1_tile = pypto.view(
                    input, shape=[T0_C1, T1_C1],
                    offsets=[r_off, half_offset + c_off],
                    valid_shape=[v_r, v_c],
                )                                          # [T0_C1,T1_C1] FP16
            x1_fp32 = pypto.cast(x1_tile, pypto.DT_FP32)   # [T0_C1,T1_C1] FP32

            # ── Step 3: pypto.sigmoid (FP32 only) ──
            sig = pypto.sigmoid(x0_fp32)                   # [T0_C1,T1_C1] FP32

            # ── Step 4: SiLU = x0 * sigmoid(x0) ──
            silu = x0_fp32 * sig                           # [T0_C1,T1_C1] FP32

            # ── Step 5: silu * x1 ──
            result_fp32 = silu * x1_fp32                   # [T0_C1,T1_C1] FP32

            # ── Step 6: cast back to FP16 ──
            result = pypto.cast(result_fp32, pypto.DT_FP16)  # [T0_C1,T1_C1] FP16

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
# Public entry point: swi_glu_impl_c1(input, dim=-1) -> torch.Tensor
# Also aliased as swi_glu_module1_wrapper_c1 for backward compat with
# per-phase test harness (test_swi_glu.py imports from module1_impl).
def swi_glu_impl_c1(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
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
    swi_glu_kernel_npu_c1(input, output, split_dim)

    return output

# Backward-compatible wrapper so that per-phase test harness can also
# import from this file under the module1 wrapper name.
def swi_glu_module1_wrapper_c1(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Module-scoped wrapper (delegates to swi_glu_impl_c1) for per-phase test harness."""
    return swi_glu_impl_c1(input, dim=dim)

def swi_glu_wrapper_c1(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    return swi_glu_impl_c1(input, dim=dim)


# ===== Public reference class c2 =====
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl_c2.py
# Integrated production implementation for swi_glu (SwiGLU activation).
#
# Pipeline: view(x0, x1) → sigmoid(x0) → mul(x0*sig) → mul(silu*x1) → assemble
#
# Layer A-L design from DESIGN.md §4.2 pseudocode.
# Template: pypto-op-develop/templates/impl_template.py
# Tiling: (T0_C2, T1_C2) = (64, 64) — Stage 7 Round 1 optimization (from baseline (32,32))
# FP32 direct path — no cast needed (c2 all cases are float32)
# =============================================================================

# ─── Layer A: Imports ──────────────────────────────────────────
import pypto
import torch
import torch_npu  # noqa: F401  NPU device init required before JIT

# ─── Layer B: Tile constants (compile-time known, OL48) ───────
T0_C2 = 64   # view tile rows — Round 1: 32→64 (2×, UB budget 49,152 B ≤ 65,536 B)
T1_C2 = 64   # view tile cols — Round 1: 32→64 (2×, UB budget 49,152 B ≤ 65,536 B)

# ─── Layer I + J: JIT entry with kernel body inlined ──────────
#
# @pypto.frontend.jit (OL01 strict literal) with NPU run_mode.
# Tensor args precede non-tensor args (OL26). All dynamic axes annotated
# with pypto.DYNAMIC (OL29, OL43). Body inlines the Layer I loop/view/
# compute/assemble recipe directly.
#
# The kernel tiles the output shape [M_out, N_out] with (T0_C2, T1_C2) windows.
# For each tile it reads two chunks from the input (x0, x1 split along
# split_dim), computes SwiGLU in FP32, and assembles the FP32 result back.
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
def swi_glu_kernel_npu_c2(
    input: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),   # [M, N]
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),  # [M_out, N_out]
    split_dim: int,  # 0 or 1 (compile-time constant)
):
    """SwiGLU FP32 elementwise kernel with tiled chunk + FP32 compute."""
    # ── Layer I — Tile setup ──
    pypto.set_vec_tile_shapes(T0_C2, T1_C2)                         # vec tile (64, 64)

    # SymbolicScalar from dynamic output axes (OL29 / OL43)
    M_out = output.shape[0]                                    # SymbolicScalar
    N_out = output.shape[1]                                    # SymbolicScalar

    # half_offset: where x1 starts in input along split_dim
    if split_dim == 0:                                         # compile-time branch
        half_offset = M_out                                    # SymbolicScalar
    else:
        half_offset = N_out                                    # SymbolicScalar

    # Outer / inner loop bounds (SymbolicScalar, ceiling division)
    rows_loop = (M_out + T0_C2 - 1) // T0_C2                         # SymbolicScalar
    cols_loop = (N_out + T1_C2 - 1) // T1_C2                         # SymbolicScalar

    # ── Layer I — Nested tile loops ──
    for r in pypto.loop(rows_loop, name="rows"):               # outer row loop (no unroll_list, OL49)
        for c in pypto.loop(cols_loop, name="cols", unroll_list=[8]):  # inner col loop — Round 3: [4]→[8]
            r_off = r * T0_C2                                     # SymbolicScalar
            c_off = c * T1_C2                                     # SymbolicScalar

            # Valid shape clamping for tail tiles (.min() ensures ≤ tile)
            v_r = (M_out - r_off).min(T0_C2)                      # SymbolicScalar, ≤ T0_C2
            v_c = (N_out - c_off).min(T1_C2)                      # SymbolicScalar, ≤ T1_C2

            # ── Step 1a: view x0 from input (same offset as output tile) ──
            x0_tile = pypto.view(
                input, shape=[T0_C2, T1_C2],
                offsets=[r_off, c_off],
                valid_shape=[v_r, v_c],
            )                                                  # [T0_C2,T1_C2] FP32

            # ── Step 1b: view x1 (shifted by half_offset along split_dim) ──
            if split_dim == 0:                                 # compile-time branch
                x1_tile = pypto.view(
                    input, shape=[T0_C2, T1_C2],
                    offsets=[half_offset + r_off, c_off],
                    valid_shape=[v_r, v_c],
                )                                              # [T0_C2,T1_C2] FP32
            else:                                              # split_dim == 1
                x1_tile = pypto.view(
                    input, shape=[T0_C2, T1_C2],
                    offsets=[r_off, half_offset + c_off],
                    valid_shape=[v_r, v_c],
                )                                              # [T0_C2,T1_C2] FP32

            # ── Step 2: sigmoid (FP32 only, input already FP32) ──
            sig = pypto.sigmoid(x0_tile)                       # [T0_C2,T1_C2] FP32

            # ── Step 3: SiLU = x0 * sigmoid(x0) ──
            silu = x0_tile * sig                               # [T0_C2,T1_C2] FP32

            # ── Step 4: result = silu * x1 ──
            result = silu * x1_tile                            # [T0_C2,T1_C2] FP32

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
# Public entry point: swi_glu_impl_c2(input, dim=-1) → torch.Tensor
def swi_glu_impl_c2(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
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
    swi_glu_kernel_npu_c2(input, output, split_dim)

    return output

# ─── Backward-compatible wrapper alias ────────────────────────
# Required by lint OL08: module-level function ending with _wrapper.
# Production entry point is swi_glu_impl_c2; this alias satisfies the
# naming convention for test harness and import compatibility.
def swi_glu_wrapper_c2(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Production wrapper alias (delegates to swi_glu_impl_c2)."""
    return swi_glu_impl_c2(input, dim=dim)


# ===== Public reference class c3 =====
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl_c3.py
# Integrated kernel implementation for swi_glu c3 (all bfloat16 cases).
#
# L0 path (module_count=1): single module covers the entire algorithm.
# Pipeline: view(chunk) -> cast BF16->FP32 -> sigmoid(x0) ->
#           mul(x0*sig) -> mul(silu*x1) -> cast FP32->BF16 -> assemble
#
# Follows Layer A-L design from DESIGN.md.
# Template: pypto-op-develop/templates/impl_template.py
# Tiling:   (T0_C3, T1_C3) = (256, 32); Stage 7 best config: R1 unroll+[4], R2 T0_C3 128->256.
#           T1_C3=32 optimal for BF16 thin-output cases (T1_C3=128 regressed).
# unroll_list: [4] (parallel batching on inner cols loop).
# =============================================================================

# ─── Layer A/G: Imports ────────────────────────────────────────
import pypto
import torch
import torch_npu  # noqa: F401  (NPU device init required before JIT)

# ─── Layer B: Tile constants (compile-time known, OL48) ────────
T0_C3 = 256  # view tile rows (Stage 7 R2: 128->256, halve case12 rows_loop; T0_C3=512 tested but regressed)
T1_C3 = 32   # view tile cols (BF16 16-elem aligned; kept 32 — T1_C3=128 regressed case6/12/16)

# ─── Layer J: JIT entry (with Layer I body inlined) ───────────
#
# @pypto.frontend.jit decorator (OL01 strict literal) with NPU run_mode.
# Tensor args precede non-tensor args (OL26). All dynamic axes annotated
# with pypto.DYNAMIC (OL29, OL43). Body inlines the Layer I loop/view/
# compute/assemble recipe directly — no separate _kernel_impl helper,
# so no pypto.is_loop_begin/end parser caveat applies.
#
# The kernel tiles the output shape with (T0_C3, T1_C3) windows. For each tile
# it reads two chunks from the input (x0 and x1, split along split_dim),
# computes SwiGLU in FP32, and assembles the BF16 result back.
#
# Snapshot markers: SIG_IMPL, SIG_JIT, CALL_IMPL embedded at
# key points for the snapshot generator (pypto-op-verify).
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
def swi_glu_kernel_npu_c3(
    input: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    split_dim: int,  # 0 or 1 (compile-time constant)
):
    """SwiGLU elementwise kernel with tiled chunk + FP32 compute (c3 — BF16).

    Formula: output = SiLU(x0) * x1, where (x0, x1) = input.chunk(2, dim).
    All ops are tile-internal (no cross-tile state). BF16 inputs are cast
    to FP32 for intermediate compute (pypto.sigmoid requires DT_FP32);
    result is cast back to BF16.
    """
    # ── Layer I — Tile setup ──
    # SIG_IMPL
    pypto.set_vec_tile_shapes(T0_C3, T1_C3)

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
    rows_loop = (M_out + T0_C3 - 1) // T0_C3
    cols_loop = (N_out + T1_C3 - 1) // T1_C3

    # ── Layer I — Outer row loop (no unroll_list — OL49) ──
    for r in pypto.loop(rows_loop, name="rows"):
        # ── Layer I — Inner col loop (Stage 7 best: [4], optimal vs [2] and [8]) ──
        for c in pypto.loop(cols_loop, name="cols", unroll_list=[4]):
            r_off = r * T0_C3   # SymbolicScalar: row byte offset
            c_off = c * T1_C3   # SymbolicScalar: col byte offset

            # Valid shape clamping for tail tiles (per-axis <= tile size)
            v_r = (M_out - r_off).min(T0_C3)   # v_r <= T0_C3 always
            v_c = (N_out - c_off).min(T1_C3)   # v_c <= T1_C3 always

            # ── Step 1: view x0 from input (same offset as output tile) ──
            x0_tile = pypto.view(
                input, shape=[T0_C3, T1_C3], offsets=[r_off, c_off],
                valid_shape=[v_r, v_c],
            )                                              # [T0_C3,T1_C3] BF16
            x0_fp32 = pypto.cast(x0_tile, pypto.DT_FP32)  # [T0_C3,T1_C3] FP32

            # ── Step 2: view x1 (shifted by half_offset along split_dim) ──
            if split_dim == 0:
                x1_tile = pypto.view(
                    input, shape=[T0_C3, T1_C3],
                    offsets=[half_offset + r_off, c_off],
                    valid_shape=[v_r, v_c],
                )                                          # [T0_C3,T1_C3] BF16
            else:
                x1_tile = pypto.view(
                    input, shape=[T0_C3, T1_C3],
                    offsets=[r_off, half_offset + c_off],
                    valid_shape=[v_r, v_c],
                )                                          # [T0_C3,T1_C3] BF16
            x1_fp32 = pypto.cast(x1_tile, pypto.DT_FP32)   # [T0_C3,T1_C3] FP32

            # ── Step 3: pypto.sigmoid (FP32 only) ──
            sig = pypto.sigmoid(x0_fp32)                   # [T0_C3,T1_C3] FP32

            # ── Step 4: SiLU = x0 * sigmoid(x0) ──
            silu = x0_fp32 * sig                           # [T0_C3,T1_C3] FP32

            # ── Step 5: silu * x1 ──
            result_fp32 = silu * x1_fp32                   # [T0_C3,T1_C3] FP32

            # ── Step 6: cast back to BF16 ──
            result = pypto.cast(result_fp32, pypto.DT_BF16)  # [T0_C3,T1_C3] BF16

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
# Public entry point: swi_glu_impl_c3(input, dim=-1) -> torch.Tensor
def swi_glu_impl_c3(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """SwiGLU host wrapper — production entry point (c3 — bfloat16).

    Computes output = SiLU(x0) * x1, where x0, x1 = input.chunk(2, dim=dim).

    Args:
        input: BF16 2D tensor [M, N] on NPU device; split dim must be even.
        dim:   split dimension (-1 default, normalised to 0 or 1).

    Returns:
        output: BF16 [M_out, N_out] with dim-dim halved.
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

    # Allocate output buffer via torch (NOT pypto.zeros / etc., OL58)
    output = torch.empty(
        M_out, N_out,
        dtype=torch.bfloat16,
        device=input.device,
    )

    # Single JIT call (Layer K contract, OL45)
    swi_glu_kernel_npu_c3(input, output, split_dim)

    return output

# Production wrapper matching module_interfaces.yaml host_wrapper signature.
# Satisfies OL08 requirement for a *_wrapper module-level function.
def swi_glu_wrapper_c3(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """SwiGLU production wrapper (c3 — bfloat16, per-phase test compat)."""
    return swi_glu_impl_c3(input, dim=dim)


# ===== Public reference class c4 =====
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl_c4.py — Integrated production implementation
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
# Layer B:   Tile constants (T0_C4, T1_C4, T2_C4 = 16, 16, 16)
# Layer J:   JIT entry with inlined Layer I body
# Layer K:   Host wrapper (swi_glu_impl_c4) + swi_glu_wrapper_c4 alias
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
T0_C4 = 16  # vec tile outer (rows)
T1_C4 = 48  # vec tile mid (cols) — perf-tuned (+21% vs T1_C4=32)
T2_C4 = 16  # vec tile inner (depth, BF16 16-element alignment)

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
def swi_glu_kernel_npu_c4(
    input: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    split_dim: int,  # compile-time constant: 0, 1, or 2
):
    """SwiGLU 3D elementwise kernel: output = SiLU(x0) * x1.

    Tiles the output shape with (T0_C4, T1_C4, T2_C4) windows.
    For each tile: views x0/x1 from input (offset by half along split_dim),
    casts to FP32, computes sigmoid + 2x mul, casts back to BF16, assembles.
    """
    # Layer I - Tile setup
    pypto.set_vec_tile_shapes(T0_C4, T1_C4, T2_C4)

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
    rows_loop = (D0 + T0_C4 - 1) // T0_C4   # SymbolicScalar
    cols_loop = (D1 + T1_C4 - 1) // T1_C4   # SymbolicScalar
    depth_loop = (D2 + T2_C4 - 1) // T2_C4  # SymbolicScalar

    # Layer I - Three-level nested loop
    # OL49: unroll_list only on innermost pypto.loop
    # OL56: unroll_list single value [1] (Stage 7 baseline)
    for r in pypto.loop(rows_loop, name="rows"):
        r_off = r * T0_C4  # SymbolicScalar
        v_r = (D0 - r_off).min(T0_C4)  # SymbolicScalar, always <= T0_C4

        for c in pypto.loop(cols_loop, name="cols"):
            c_off = c * T1_C4  # SymbolicScalar
            v_c = (D1 - c_off).min(T1_C4)  # SymbolicScalar, always <= T1_C4

            for d in pypto.loop(depth_loop, name="depth", unroll_list=[16]):
                d_off = d * T2_C4  # SymbolicScalar
                v_d = (D2 - d_off).min(T2_C4)  # SymbolicScalar, always <= T2_C4
                vs = [v_r, v_c, v_d]  # valid_shape list

                # View x0: same offset as output tile
                # x0_tile: [T0_C4,T1_C4,T2_C4] BF16
                x0_tile = pypto.view(
                    input, shape=[T0_C4, T1_C4, T2_C4],
                    offsets=[r_off, c_off, d_off],
                    valid_shape=vs,
                )
                # x0_fp32: [T0_C4,T1_C4,T2_C4] FP32
                x0_fp32 = pypto.cast(x0_tile, pypto.DT_FP32)

                # View x1: offset + half along split_dim
                # split_dim is compile-time const, safe for Python if/else
                if split_dim == 0:
                    x1_off = [half + r_off, c_off, d_off]
                elif split_dim == 1:
                    x1_off = [r_off, half + c_off, d_off]
                else:  # split_dim == 2
                    x1_off = [r_off, c_off, half + d_off]

                # x1_tile: [T0_C4,T1_C4,T2_C4] BF16
                x1_tile = pypto.view(
                    input, shape=[T0_C4, T1_C4, T2_C4],
                    offsets=x1_off,
                    valid_shape=vs,
                )
                # x1_fp32: [T0_C4,T1_C4,T2_C4] FP32
                x1_fp32 = pypto.cast(x1_tile, pypto.DT_FP32)

                # Compute SwiGLU in FP32
                # sig: [T0_C4,T1_C4,T2_C4] FP32
                sig = pypto.sigmoid(x0_fp32)
                # silu: [T0_C4,T1_C4,T2_C4] FP32 (SiLU = x * sigmoid(x))
                silu = x0_fp32 * sig
                # result_fp32: [T0_C4,T1_C4,T2_C4] FP32 (gated output)
                result_fp32 = silu * x1_fp32

                # Cast back to BF16
                # result: [T0_C4,T1_C4,T2_C4] BF16
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
def swi_glu_impl_c4(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
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
    swi_glu_kernel_npu_c4(input, output, split_dim_normalized)

    return output

# Layer K alias for E2E test harness
# (test_swi_glu.py imports swi_glu_wrapper_c4)
def swi_glu_wrapper_c4(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Op-scoped wrapper (delegates to swi_glu_impl_c4) for E2E test harness."""
    return swi_glu_impl_c4(input, dim=dim)

def swi_glu_wrapper_c4(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    return swi_glu_impl_c4(input, dim=dim)


# ===== Public reference class c5 =====
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl_c5.py  —  Integrated production implementation of SwiGLU (c5)
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
# Vec tile = (1, 16, 16, 32): T0_C5=1 (batch-collapsed), T1_C5/T2_C5=16, T3_C5=32
# FP32 alignment: T3_C5=32 is 8-elem aligned (32/8=4)
# UB per-op: binary mul = 1*16*16*32*4*3 = 96 KB ≤ 192 KB ✓
T0_C5 = 1   # tile dim 0 (batch-collapsed)
T1_C5 = 16  # tile dim 1
T2_C5 = 16  # tile dim 2
T3_C5 = 32  # tile dim 3 (FP32 8-elem aligned)

# ─── Layer H + I + J: JIT entry (body inlined) ───────────────
#
# @pypto.frontend.jit decorator (OL01 strict literal) with NPU run_mode.
# Tensor args precede non-tensor args (OL26). All dynamic axes annotated
# with pypto.DYNAMIC (OL29, OL43). Body inlines the Layer I loop/view/
# compute/assemble recipe directly (no pypto.is_loop_begin/end — no need
# for @pypto.frontend.function or helper separation).
#
# The kernel tiles the OUTPUT shape with [T0_C5, T1_C5, T2_C5, T3_C5] windows. For each
# tile it reads two chunks from the input (x0 at same offset, x1 shifted by
# half_size along split_dim), computes SwiGLU in FP32, and assembles the
# result back.
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
def swi_glu_kernel_npu_c5(
    input: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),   # [D0_in, D1_in, D2_in, D3_in]
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),  # [D0, D1, D2, D3]
    split_dim: int,  # compile-time constant: 0/1/2/3 from host normalization
):
    """SwiGLU: output = SiLU(x0) * x1 where x0, x1 = chunk(input, 2, split_dim)."""

    # ─── Layer I: Tile setup ───────────────────────────────────
    pypto.set_vec_tile_shapes(T0_C5, T1_C5, T2_C5, T3_C5)      # tile = [1, 16, 16, 32]

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
    N0 = (D0 + T0_C5 - 1) // T0_C5   # = D0 (since T0_C5=1)
    N1 = (D1 + T1_C5 - 1) // T1_C5
    N2 = (D2 + T2_C5 - 1) // T2_C5
    N3 = (D3 + T3_C5 - 1) // T3_C5

    # ─── Layer I: 4D nested loop ──────────────────────────────
    # Outer loops: no unroll_list (OL49)
    # Innermost: unroll_list=[1] (OL56 single-value, Stage 6 default)
    for i0 in pypto.loop(N0, name="d0"):                 # SymbolicScalar i0
        for i1 in pypto.loop(N1, name="d1"):             # SymbolicScalar i1
            for i2 in pypto.loop(N2, name="d2"):         # SymbolicScalar i2
                for i3 in pypto.loop(N3, name="d3", unroll_list=[1]):  # innermost

                    # Tile offsets (SymbolicScalar)
                    off0 = i0 * T0_C5                       # = i0 (T0_C5=1)
                    off1 = i1 * T1_C5
                    off2 = i2 * T2_C5
                    off3 = i3 * T3_C5

                    # Valid shape per axis (SymbolicScalar, clamped ≤ tile size)
                    v0 = (D0 - off0).min(T0_C5)             # ≤ T0_C5=1 always
                    v1 = (D1 - off1).min(T1_C5)             # ≤ T1_C5
                    v2 = (D2 - off2).min(T2_C5)             # ≤ T2_C5
                    v3 = (D3 - off3).min(T3_C5)             # ≤ T3_C5

                    # ── Step 1: view x0 from input ──
                    # x0 chunk at same offset as the output tile (no shift)
                    x0_tile = pypto.view(
                        input, shape=[T0_C5, T1_C5, T2_C5, T3_C5],
                        offsets=[off0, off1, off2, off3],
                        valid_shape=[v0, v1, v2, v3],
                    )                                                  # [T0_C5,T1_C5,T2_C5,T3_C5] FP32

                    # ── Step 2: view x1 from input ──
                    # x1 chunk shifted by half_size along split_dim
                    if split_dim == 0:
                        x1_tile = pypto.view(
                            input, shape=[T0_C5, T1_C5, T2_C5, T3_C5],
                            offsets=[half_size + off0, off1, off2, off3],
                            valid_shape=[v0, v1, v2, v3],
                        )                                              # [T0_C5,T1_C5,T2_C5,T3_C5] FP32
                    elif split_dim == 1:
                        x1_tile = pypto.view(
                            input, shape=[T0_C5, T1_C5, T2_C5, T3_C5],
                            offsets=[off0, half_size + off1, off2, off3],
                            valid_shape=[v0, v1, v2, v3],
                        )                                              # [T0_C5,T1_C5,T2_C5,T3_C5] FP32
                    elif split_dim == 2:
                        x1_tile = pypto.view(
                            input, shape=[T0_C5, T1_C5, T2_C5, T3_C5],
                            offsets=[off0, off1, half_size + off2, off3],
                            valid_shape=[v0, v1, v2, v3],
                        )                                              # [T0_C5,T1_C5,T2_C5,T3_C5] FP32
                    else:  # split_dim == 3
                        x1_tile = pypto.view(
                            input, shape=[T0_C5, T1_C5, T2_C5, T3_C5],
                            offsets=[off0, off1, off2, half_size + off3],
                            valid_shape=[v0, v1, v2, v3],
                        )                                              # [T0_C5,T1_C5,T2_C5,T3_C5] FP32

                    # ── Step 3: sigmoid(x0) — FP32 only ──
                    sig = pypto.sigmoid(x0_tile)                       # [T0_C5,T1_C5,T2_C5,T3_C5] FP32

                    # ── Step 4: SiLU = x0 * sigmoid(x0) ──
                    silu = x0_tile * sig                               # [T0_C5,T1_C5,T2_C5,T3_C5] FP32

                    # ── Step 5: gate = silu * x1 ──
                    result = silu * x1_tile                            # [T0_C5,T1_C5,T2_C5,T3_C5] FP32

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
def swi_glu_wrapper_c5(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
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
    swi_glu_kernel_npu_c5(input, output, split_dim)

    return output

# ─── Public API alias ──────────────────────────────────────────
# Primary production entry point (README / test usage: from swi_glu_impl_c5 import swi_glu_impl_c5).
swi_glu_impl_c5 = swi_glu_wrapper_c5


# ===== Public reference class c6 =====
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl_c6.py
# Integrated production implementation for swi_glu (c6: 5D tensor variant).
#
# L0 path (module_count=1): single kernel covers the entire algorithm.
# Pipeline: input.chunk(2,dim) -> cast FP16->FP32 -> sigmoid(x0) ->
#           mul(x0*sig) -> mul(silu*x1) -> cast FP32->FP16 -> assemble
#
# Follows Layer A-L design from DESIGN.md.
# Template: pypto-op-develop/templates/impl_template.py
# Tiling:   (T0_C6, T1_C6) = (64, 128); Stage 7 R2 optimized tile (smaller T0_C6 reduces per-tile cost).
# Host wrapper reshapes N-D -> 2D before kernel, 2D -> N-D after.
# =============================================================================

# ─── Layer A/G: Imports ────────────────────────────────────────
import pypto
import torch
import torch_npu  # noqa: F401  (NPU device init required before JIT)

# ─── Layer B: Tile constants (compile-time known, OL48) ────────
T0_C6 = 64   # view tile rows (Stage 7 R2: smaller tile → lower regs/compile, 47 iter)
T1_C6 = 128  # view tile cols (proven: FP16 16-elem aligned)

# ─── Layer J: JIT entry (with Layer I body inlined) ───────────
#
# @pypto.frontend.jit decorator (OL01 strict literal) with NPU run_mode.
# Tensor args precede non-tensor args (OL26). All dynamic axes annotated
# with pypto.DYNAMIC (OL29, OL43). Body inlines the Layer I loop/view/
# compute/assemble recipe directly.
#
# The kernel tiles the output shape with (T0_C6, T1_C6) windows. For each tile
# it reads two chunks from the input (x0 and x1, split along split_dim),
# computes SwiGLU in FP32, and assembles the FP16 result back.
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
def swi_glu_kernel_npu_c6(
    input: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
    split_dim: int,  # 0 or 1 (compile-time constant)
):
    """SwiGLU elementwise kernel with tiled chunk + FP32 compute."""
    # Layer I — Tile setup
    pypto.set_vec_tile_shapes(T0_C6, T1_C6)

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
    rows_loop = (M_out + T0_C6 - 1) // T0_C6
    cols_loop = (N_out + T1_C6 - 1) // T1_C6

    # Layer I — Outer row loop (no unroll_list — OL49)
    for r in pypto.loop(rows_loop, name="rows"):
        # Layer I — Inner col loop (unroll_list=[4], Stage 7 R1: parallel batch 4 iter)
        for c in pypto.loop(cols_loop, name="cols", unroll_list=[4]):
            r_off = r * T0_C6   # SymbolicScalar
            c_off = c * T1_C6   # SymbolicScalar

            # Valid shape clamping for tail tiles (per-axis <= tile size)
            v_r = (M_out - r_off).min(T0_C6)   # v_r <= T0_C6 always
            v_c = (N_out - c_off).min(T1_C6)   # v_c <= T1_C6 always

            # ── Step 1-2: view x0 from input (same offset as output tile) ──
            x0_tile = pypto.view(
                input, shape=[T0_C6, T1_C6], offsets=[r_off, c_off],
                valid_shape=[v_r, v_c],
            )                                              # [T0_C6,T1_C6] FP16
            x0_fp32 = pypto.cast(x0_tile, pypto.DT_FP32)  # [T0_C6,T1_C6] FP32

            # ── Step 1-2: view x1 (shifted by half_offset along split_dim) ──
            if split_dim == 0:
                x1_tile = pypto.view(
                    input, shape=[T0_C6, T1_C6],
                    offsets=[half_offset + r_off, c_off],
                    valid_shape=[v_r, v_c],
                )                                          # [T0_C6,T1_C6] FP16
            else:
                x1_tile = pypto.view(
                    input, shape=[T0_C6, T1_C6],
                    offsets=[r_off, half_offset + c_off],
                    valid_shape=[v_r, v_c],
                )                                          # [T0_C6,T1_C6] FP16
            x1_fp32 = pypto.cast(x1_tile, pypto.DT_FP32)   # [T0_C6,T1_C6] FP32

            # ── Step 3: pypto.sigmoid (FP32 only) ──
            sig = pypto.sigmoid(x0_fp32)                   # [T0_C6,T1_C6] FP32

            # ── Step 4: SiLU = x0 * sigmoid(x0) ──
            silu = x0_fp32 * sig                           # [T0_C6,T1_C6] FP32

            # ── Step 5: silu * x1 ──
            result_fp32 = silu * x1_fp32                   # [T0_C6,T1_C6] FP32

            # ── Step 6: cast back to FP16 ──
            result = pypto.cast(result_fp32, pypto.DT_FP16)  # [T0_C6,T1_C6] FP16

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
# Only 5 responsibilities:
#   1. Normalize dim -> split_dim_normalized (0 to ndim-1)
#   2. Reshape N-D input to 2D (collapse non-split dims into M)
#   3. Compute output shape; allocate output buffer via torch.empty
#   4. Invoke JIT exactly ONCE — NO Python for...in range (OL45)
#   5. Reshape output back to N-D target shape
#
# Public entry point: swi_glu_impl_c6(input, dim=-1) -> torch.Tensor
def swi_glu_impl_c6(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """SwiGLU host wrapper with N-D -> 2D reshape for production use.

    Computes output = SiLU(x0) * x1, where x0, x1 = input.chunk(2, dim=dim).

    For c6 case: input [3,7,11,13,1012] dim=-1
      -> reshape [3003, 1012] -> kernel -> [3003, 506]
      -> reshape back [3,7,11,13,506]

    Args:
        input: N-D tensor on NPU device; split dim must be even.
        dim:   split dimension (-1 default).

    Returns:
        output tensor, same dtype as input, dim-dim halved.
    """
    orig_shape = input.shape
    ndim = input.ndim

    # Normalize dim into [0, ndim)
    split_dim_normalized = dim % ndim

    # Compute 2D-equivalent shape: flatten all dims except split_dim
    split_size = orig_shape[split_dim_normalized]
    M = input.numel() // split_size  # product of all dims except split
    N = split_size

    # Reshape N-D -> 2D (no-copy for contiguous)
    input_2d = input.reshape(M, N)

    # Determine 2D output shape and split_dim_mapped
    if split_dim_normalized == 0:
        M_out_2d, N_out_2d = M // 2, N
        split_dim_mapped = 0   # split along rows
    else:
        M_out_2d, N_out_2d = M, N // 2
        split_dim_mapped = 1   # split along cols (or inner)

    # Allocate output buffer via torch (NOT pypto.zeros — host-side, OL58)
    output_2d = torch.empty(
        M_out_2d, N_out_2d,
        dtype=input.dtype,
        device=input.device,
    )

    # Single JIT call (Layer K contract, OL45)
    swi_glu_kernel_npu_c6(input_2d, output_2d, split_dim_mapped)

    # Reshape output back to N-D target shape (no-copy view when possible)
    out_shape = list(orig_shape)
    out_shape[split_dim_normalized] //= 2
    output = output_2d.reshape(out_shape)

    return output

# Backward-compatible wrapper so that per-phase test harness can also
# import from this file under the module1 wrapper name.
def swi_glu_module1_wrapper_c6(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Module-scoped wrapper (delegates to swi_glu_impl_c6) for backward compat."""
    return swi_glu_impl_c6(input, dim=dim)

def swi_glu_wrapper_c6(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    return swi_glu_impl_c6(input, dim=dim)


# ===== Public reference class c7 =====
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl_c7.py  —  SwiGLU activation kernel (c7 — 3D FP32)
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
# T0_C7=1 (batch-collapsed), T1_C7=64 (seq), T2_C7=128 (embed/2)
# FP32 alignment: T2_C7=128 is 8-elem aligned (128/8=16)
# UB per-op: binary mul = 1*64*128*4*3 = 96 KB ≤ 196 KB
T0_C7 = 1   # tile dim 0 (batch-collapsed)
T1_C7 = 64  # tile dim 1 (sequence) — Stage 7 Round 1: 16→64
T2_C7 = 128 # tile dim 2 (embed/2, FP32 8-elem aligned) — Stage 7 Round 2: 64→128, halve D2 iterations

# ─── JIT Kernel ────────────────────────────────────────────────
#
# @pypto.frontend.jit decorator (OL01 strict literal) with NPU run_mode.
# Tensor args precede non-tensor args (OL26). All dynamic axes annotated
# with pypto.DYNAMIC (OL29, OL43). Body inlines the loop/view/compute/
# assemble recipe directly (no separate _kernel_impl helper).
#
# The kernel tiles the OUTPUT shape with [T0_C7, T1_C7, T2_C7] windows. For each
# tile it reads two chunks from the input (x0 at same offset, x1 shifted by
# half_size along split_dim), computes SwiGLU in FP32, and assembles the
# result back into the output buffer.
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU, "valid_shape_optimize": 1},
    pass_options={"vec_nbuffer_setting": {"DEFAULT": 8}},
)
def swi_glu_kernel_npu_c7(
    input: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),   # [D0, D1, D2_in]
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),  # [D0, D1, D2]
    split_dim: int,  # compile-time constant: 0/1/2 from host normalization
):
    """SwiGLU: output = SiLU(x0) * x1 where x0, x1 = chunk(input, 2, split_dim)."""

    # ─── Tile setup ────────────────────────────────────────────
    pypto.set_vec_tile_shapes(T0_C7, T1_C7, T2_C7)              # tile = [1, 64, 128]

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
    N0 = (D0 + T0_C7 - 1) // T0_C7   # = D0 (since T0_C7=1)
    N1 = (D1 + T1_C7 - 1) // T1_C7   # e.g. ceil(1023/64) = 16
    N2 = (D2 + T2_C7 - 1) // T2_C7   # e.g. ceil(2048/128) = 16

    # ─── 3D nested loop ────────────────────────────────────────
    # Outer loops: no unroll_list (OL49)
    # Innermost: unroll_list=[4] (OL56 Stage 7 multi-value, R3)
    for i0 in pypto.loop(N0, name="d0"):                 # SymbolicScalar i0
        for i1 in pypto.loop(N1, name="d1"):             # SymbolicScalar i1
            for i2 in pypto.loop(N2, name="d2", unroll_list=[4]):  # innermost (Stage 7 R3)

                # Tile offsets (SymbolicScalar)
                off0 = i0 * T0_C7                           # = i0 (T0_C7=1)
                off1 = i1 * T1_C7
                off2 = i2 * T2_C7

                # Valid shape per axis (SymbolicScalar, clamped ≤ tile size)
                v0 = (D0 - off0).min(T0_C7)                 # ≤ T0_C7=1 always
                v1 = (D1 - off1).min(T1_C7)                 # ≤ T1_C7 (critical for 1023 tail)
                v2 = (D2 - off2).min(T2_C7)                 # ≤ T2_C7

                # ── Step 1: view x0 from input ──
                # x0 chunk at same offset as the output tile (no shift)
                x0_tile = pypto.view(
                    input, shape=[T0_C7, T1_C7, T2_C7],
                    offsets=[off0, off1, off2],
                    valid_shape=[v0, v1, v2],
                )                                                  # [T0_C7,T1_C7,T2_C7] FP32

                # ── Step 2: view x1 from input ──
                # x1 chunk shifted by half_size along split_dim
                if split_dim == 0:
                    x1_tile = pypto.view(
                        input, shape=[T0_C7, T1_C7, T2_C7],
                        offsets=[half_size + off0, off1, off2],
                        valid_shape=[v0, v1, v2],
                    )                                              # [T0_C7,T1_C7,T2_C7] FP32
                elif split_dim == 1:
                    x1_tile = pypto.view(
                        input, shape=[T0_C7, T1_C7, T2_C7],
                        offsets=[off0, half_size + off1, off2],
                        valid_shape=[v0, v1, v2],
                    )                                              # [T0_C7,T1_C7,T2_C7] FP32
                else:  # split_dim == 2
                    x1_tile = pypto.view(
                        input, shape=[T0_C7, T1_C7, T2_C7],
                        offsets=[off0, off1, half_size + off2],
                        valid_shape=[v0, v1, v2],
                    )                                              # [T0_C7,T1_C7,T2_C7] FP32

                # ── Step 3: sigmoid(x0) — FP32 only ──
                sig = pypto.sigmoid(x0_tile)                       # [T0_C7,T1_C7,T2_C7] FP32

                # ── Step 4: SiLU = x0 * sigmoid(x0) ──
                silu = x0_tile * sig                               # [T0_C7,T1_C7,T2_C7] FP32

                # ── Step 5: gate = silu * x1 ──
                result = silu * x1_tile                            # [T0_C7,T1_C7,T2_C7] FP32

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
def swi_glu_wrapper_c7(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
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
    swi_glu_kernel_npu_c7(input, output, split_dim)

    return output

# ─── Public API alias ──────────────────────────────────────────
# Primary production entry point (README / test usage).
swi_glu_impl_c7 = swi_glu_wrapper_c7


# ===== Public reference class c8 =====
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# swi_glu_impl_c8.py
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
# Tiling:   (T0_C8, T1_C8) = (64, 50) per DESIGN.md §3.2.5
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

def swi_glu_wrapper_c8(input: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    return c8_impl_func(input, dim=dim)



# Dispatch contract from the public reference: (ndim, dtype) -> implementation class.
_CLASSES = {
    (2, "float16"): swi_glu_impl_c1,
    (2, "float32"): swi_glu_impl_c2,
    (2, "bfloat16"): swi_glu_impl_c3,
    (3, "bfloat16"): swi_glu_impl_c4,
    (4, "float32"): swi_glu_impl_c5,
    (5, "float16"): swi_glu_impl_c6,
    (3, "float32"): swi_glu_impl_c7,
    (5, "float32"): swi_glu_wrapper_c8,
}


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    signature = (int(input.ndim), str(input.dtype).rsplit(".", 1)[-1])
    try:
        implementation = _CLASSES[signature]
    except KeyError as exc:
        raise ValueError(f"no PyPTO SwiGlu class for {signature}") from exc
    return implementation(input, dim=dim)
