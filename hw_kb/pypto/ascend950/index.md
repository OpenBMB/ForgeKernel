# PyPTO on Ascend 950

This KB is for operator-agnostic, measured PyPTO/Ascend facts. It intentionally starts small: a candidate recipe is promoted only after an NPU probe verifies it.

## Local interfaces

- `pypto.frontend.jit` lowers a function for `pypto.RunMode.NPU`.
- Tensor annotations fix rank and dtype; dynamic extents use `pypto.DYNAMIC`.
- Use `pypto.view` to read tiles, `pypto.assemble` to write results, and `pypto.cast` for FP32 intermediate arithmetic.
- `pypto.set_vec_tile_shapes` configures vector tile extents. `unroll_list` belongs on the innermost `pypto.loop` only.
- Host code allocates output storage with `torch.empty`; numerical work stays in the JIT kernel.

## Measurement

CANN-Bench PyPTO tasks use `--perf-metric-strategy trace_view`. The candidate metric is AI-Core E2E time, aggregated as a geometric mean over every public case. Default `kernel_details.csv` host timings are not the PyPTO device-time currency.

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

## Current gaps

Launch footprints, register/SMEM equivalents, stall taxonomies and instruction-level inspection are not yet mapped from NCU. Do not invent Ascend equivalents for unavailable NCU metrics; mark them unavailable until a real probe exists.
