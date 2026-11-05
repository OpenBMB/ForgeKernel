# PyPTO on Ascend 950

This KB is for operator-agnostic, measured PyPTO/Ascend facts. It intentionally starts small: a candidate recipe is promoted only after an NPU probe verifies it.

## Local interfaces

- `pypto.frontend.jit` lowers a function for `pypto.RunMode.NPU`.
- Tensor annotations fix rank and dtype; dynamic extents use `pypto.DYNAMIC`.
- Use `pypto.view` to read tiles, `pypto.assemble` to write results, and `pypto.cast` for FP32 intermediate arithmetic. A `view` is a **per-axis slice** (each extent is consumed along the same source axis), not a reshape; for a strided grouped operand read one index of an inner axis as `view(x, [N, 1, D], [r0, g, 0])` then `reshape(..., [1, N, D])` — see [pypto-operand-tile-construction](pypto-operand-tile-construction.md).
- `pypto.set_vec_tile_shapes` configures vector tile extents. `unroll_list` belongs on the innermost `pypto.loop` only.
- Host code allocates output storage with `torch.empty`; numerical work stays in the JIT kernel.

## Measurement

The official per-case time follows the card's declared `timing_basis` and the
run's `--perf-metric-strategy`. Under `cann_kernel_details` the per-case
`elapsed_us` is the median of the five repeat per-iteration sums of the PyPTO
task rows, and that window includes the `AI_CPU` control tax — read the
`MIX_AIC` compute row for kernel-quality comparisons and never mix bases across
a ladder. Under `trace_view` the candidate metric is AI-Core E2E, aggregated as
a geometric mean over every public case; that basis requires `op_times.trace_view`
to actually be present (it is absent under a `kernel_details` invocation
`[MEASURED, job 1793833012]`). See
[pypto-dynamic-execution](pypto-dynamic-execution.md).

## Probe-certified recipes

- [valid_shape_assemble](valid_shape_assemble.md) — ✅ Verified on Ascend 950PR
  (job `1793761707`, probe `notes/tile_valid_shape_probe.py`, `PROBE PASS max_diff=0.0`):
  valid-only assembly into a sentinel destination. **No-go:** raw-view invalid
  storage and multi-output padding compositions must not be assumed to be zero;
  retry requires a new independent materialization probe.
- [device_floor_sandwich](device_floor_sandwich.md) — ✅ Verified on Ascend 950PR
  (job `1793791124`, probe `notes/recon_round5.py` P3): device-side per-launch floor
  ≈ 0.5–1.2 µs (vs ~11.9 µs host-paced wall floor); device-event sandwich is the
  certified msprof-free timing path. **Dead channels in torch_npu 2.13.0.rc1:**
  kineto chrome export carries 0 kernel events; in-process torch_npu.profiler msprof
  backend lacks the export API.
- [pypto-operand-tile-construction](pypto-operand-tile-construction.md) — ✅ Verified
  on Ascend 950PR (`PROBE PASS max_diff=5.91913e-08, job 1793825465`, probe
  `notes/pypto-operand-tile-construction.py`): `pypto.view` is a per-axis slice, so a
  strided rank-3 operand is `view(x, [N, 1, D], [r0, g, 0])` (element-exact at
  fp16/bf16/fp32) and `reshape(..., [1, N, D])` preserves element order; the result is
  a legal batched cube operand (`b_trans=True` fp16/bf16, `b_trans=False` fp16).
  **No-go:** a middle-axis run length `view(x, [1, N, D], [r0, g, 0])` selects a
  different window (measured max_abs 1.105e+01); partial last-axis boxes, N≠D cube
  operands, 4-D sources and materialise-then-reshape are not certified — each retry
  needs a new independent probe.
- [pypto-dynamic-execution](pypto-dynamic-execution.md) — ✅ Verified on Ascend
  950PR (`PROBE PASS max_diff=6.14766e-06, job 1793835682`, probe
  `notes/pypto-dynamic-execution.py`): one `@pypto.frontend.jit` call records 3
  msprof task rows sharing one kernel name — 2 `AI_CPU` control rows
  (`Block Num 0`) + 1 compute row (`MIX_AIC`, `Block Num 28`, `Mix Block Num 56`
  on the probed kernel) — that overlap in time, so the per-call 3-row sum
  exceeds the compute row. **No-go:** treating the 3-row sum as kernel time,
  reading the row count as a call count, or assuming the row model / control
  tax on another core mix or stack version; each retry needs a new independent
  probe.

## Current gaps

The msprof launch/task-row model of a PyPTO call is mapped by
[pypto-dynamic-execution](pypto-dynamic-execution.md). NCU-style launch
footprints, register/SMEM equivalents, stall taxonomies and instruction-level
inspection are not yet mapped. Do not invent Ascend equivalents for unavailable
NCU metrics; mark them unavailable until a real probe exists.
