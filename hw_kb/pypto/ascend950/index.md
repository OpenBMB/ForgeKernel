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

## Current gaps

Launch footprints, register/SMEM equivalents, stall taxonomies and instruction-level inspection are not yet mapped from NCU. Do not invent Ascend equivalents for unavailable NCU metrics; mark them unavailable until a real probe exists.
