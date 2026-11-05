# PyPTO dynamic-execution task rows in Ascend msprof

- Architecture: Ascend 950PR, PyPTO 0.3.0, CANN 9.2, torch_npu 2.13.0.rc1
- Status: ✅ Verified on Ascend 950PR — `PROBE PASS max_diff=6.14766e-06, job
  1793835682` (round 1)
- Certification evidence: probe `notes/pypto-dynamic-execution.py`. Scope: 8
  profiled calls of a generic synthetic cube+vector kernel record 24 task rows
  under one kernel name — per call 2 `AI_CPU` control rows (`Block Num 0`) + 1
  `MIX_AIC` compute row (`Block Num 28`, `Mix Block Num 56`); the three rows
  overlap in time and every contract C1-C7 passed. Not a speed claim.

## Purpose

A `@pypto.frontend.jit` call is carried onto the device by PyPTO's dynamic
execution, and the msprof task tables record it as **three device tasks that
share one kernel name**: two `AI_CPU` control tasks and one compute task. The
control tasks carry `Block Num 0`; the compute task carries the kernel's block
decomposition (its `Block Num`, with `Mix Block Num` equal to twice it on a mix
kernel). The control tasks run concurrently with the compute task, so the sum
of the three per-call durations is larger than the compute duration — the
difference is runtime control overhead, not kernel work.

Row reading rule: the compute row (`Task Type` of a cube+vector kernel is
`MIX_AIC`, `Block Num > 0`) is the kernel's device time; the control rows
(`AI_CPU`, `Block Num 0`) are launch overhead. Three rows are **one call**, not
three.

## When needed

Use this when accounting PyPTO time from `kernel_details.csv` /
`op_summary_*.csv`, or when an official CANN-Bench number taken with
`--perf-metric-strategy kernel_details` must be interpreted. Observable
symptoms: three rows per call, a per-call sum much larger than any single row,
and `AI_CPU` rows sitting next to one compute row.

Not applicable to non-PyPTO kernels (a single-launch library kernel records one
task row) or to a PyPTO kernel with a different core mix. Verify the compute
row's `Task Type` and `Block Num` on the new kernel before reusing the
accounting; a different row model is a new certification.

## Minimal runnable example

```python
# Generic synthetic kernel: 64 independent batched cube products + a vector
# epilogue (a[256,8,128] fp16 x b[64,128,128] fp16 -> y[256,8,128] fp32).
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def dyn_exec_kernel(a: pypto.Tensor([256, 8, 128], pypto.DT_FP16),
                    b: pypto.Tensor([64, 128, 128], pypto.DT_FP16),
                    y: pypto.Tensor([256, 8, 128], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(4, 8, 128)
    pypto.set_cube_tile_shapes([16, 16], [128, 128], [128, 128])
    for u in pypto.loop(0, 64, name="units", parallel=True):
        av = pypto.view(a, [4, 8, 128], [u * 4, 0, 0])
        bv = pypto.view(b, [1, 128, 128], [u, 0, 0])
        c = pypto.matmul(av, bv, pypto.DT_FP32, b_trans=True)
        pypto.assemble(pypto.add(c, 1.0), [u * 4, 0, 0], y)

# Warm up (JIT compile), then profile a fixed number of calls.
exp = torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False, data_simplification=False)
sched = torch_npu.profiler.schedule(wait=0, warmup=2, active=8, repeat=1,
                                    skip_first=2)
with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        schedule=sched,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(outdir),
        experimental_config=exp) as prof:
    for _ in range(12):
        dyn_exec_kernel(a_npu, b_npu, y_npu)
        torch_npu.npu.synchronize()
        prof.step()

# Export the raw session, then read the task table:
#   msprof --export=on --output=<PROF_* session dir>
#   -> <session>/mindstudio_profiler_output/op_summary_*.csv
# Columns used: Op Name, Task Type, Task Duration(us),
#               Task Start Time(us), Block Num, Mix Block Num, Step Id.
```

Group the candidate rows by `Step Id` (or by nearest `Task Start Time(us)`
when the export has no step column). Per call: one compute row
(`Task Type != AI_CPU`, `Block Num > 0`) plus two `AI_CPU` rows (`Block Num 0`).
The compute duration is kernel device time; the three-row sum is the runtime's
per-call device window.

## Gotchas

- **Do not sum the three rows as kernel time.** The control tasks overlap the
  compute task; their durations are runtime overhead. Read the compute row for
  kernel time and the sum only when reproducing a metric that is defined on the
  full per-call window.
- **Do not read the row count as a call count.** 480 candidate rows over 160
  calls are 160 calls x 3 task rows, not 480 calls `[MEASURED, job 1793834140]`.
  The extra rows are control tasks of the same kernel name.
- **The official `kernel_details` metric includes the control tax.** Under
  `device_window_basis=cann_kernel_details` the per-case `elapsed_us` is the
  median of the five repeat per-iteration sums of the three rows. On the
  20-shape campaign that sum measured 2.05x–3.05x the compute row (geomean
  2.353x) `[MEASURED, job 1793834140]`; the certification probe measured
  2.540x–2.684x (median 2.615x) on its synthetic kernel `[PROBE, job
  1793835682]`. The ratio depends on the kernel duration, so never hard-code
  it, and read an optimization ladder on the compute row (or on `trace_view`
  AI-Core E2E only when the run actually produces it).
- **`op_times.trace_view.aicore_e2e` may not exist under a `kernel_details`
  invocation.** A job that asked for `timing_basis=cann_trace_view` while the
  invocation used `--perf-metric-strategy kernel_details` failed with
  `trace_view.aicore_e2e required` `[MEASURED, job 1793833012]`. Check that the
  report actually carries `op_times.trace_view` before treating it as the
  currency.
- **Export the raw session before reading Task Type.** In this stack the
  msprof export (`op_summary_*.csv`) is what carries `Task Type`, `Block Num`
  and `Mix Block Num`; `kernel_details.csv` alone may not separate the control
  rows from the compute row.
- **`Block Num` is the kernel's own decomposition.** Only the control rows'
  `0` is constant. A unit-rich kernel read `Block Num 28` with `Mix Block Num
  56` `[MEASURED, job 1793834140]`; report the value measured on your kernel
  instead of assuming 28.
- **No performance claim.** The rows report device task durations and say
  nothing about kernel quality or speed.

## Retry preconditions

If a PyPTO / CANN upgrade or a different kernel form changes the row model —
a different compute `Task Type`, a different control-row count, or no
`op_summary` export — do not reuse this accounting. Rerun an independent
synthetic probe and record the new model. If the msprof export fails, fix the
export path (msprof on `PATH`, a raw `PROF_*` session present) before retrying;
without it the task type cannot be separated from the kernel name in
`kernel_details.csv`.

## Sources

- Public API symbols: `pypto.frontend.jit`, `pypto.loop`, `pypto.view`,
  `pypto.matmul`, `pypto.set_cube_tile_shapes`, `pypto.set_vec_tile_shapes`,
  `pypto.assemble` (local PyPTO DSL tree); `torch_npu.profiler.profile`,
  `torch_npu.profiler._ExperimentalConfig`,
  `torch_npu.profiler.ProfilerLevel.Level1`,
  `torch_npu.profiler.schedule`,
  `torch_npu.profiler.tensorboard_trace_handler`;
  `msprof --export=on`; CANN msprof `kernel_details` / `op_summary` columns.
- Probe verdict: job `1793835682` — `PROBE PASS max_diff=6.14766e-06`;
  contracts C1, C1b, C2a–C2c, C3–C6 (row model / overlap) and C7 (host
  float64 numeric check) all PASS.
- Detailed provenance archive:
  `docs/kb-provenance/pypto/pypto-dynamic-execution.md`

## Certification scope

Job `1793835682` (probe `notes/pypto-dynamic-execution.py`, round 1) ran the
mechanism on the real NPU with generic synthetic inputs and operator-agnostic
PyPTO NPU operations; the device was synchronised before the host-side float64
comparison. Every contract line passed:

- C1/C1b: one kernel name (`PYPTO_dyn_exec_kernel`); 24 task rows = 16
  `AI_CPU` control rows + 8 compute rows, grouped into 8 calls (3 rows per
  call) by nearest `Task Start Time(us)`; no other kernel name.
- C2a–C2c: control `Block Num` = 0; compute `Block Num` = 28 (> 0); compute
  `Task Type` = `MIX_AIC`.
- C3: `Mix Block Num` = 2 x `Block Num` on all 8 compute rows (28 -> 56).
- C4: per-call sum / compute > 1 on all 8 calls; tax 2.540x–2.684x, median
  2.615x.
- C5: the three rows overlap in time on 8/8 calls; overlap 88.534–104.800 us.
- C6: last-5 median per-call sum 196.707 us vs median compute 75.384 us
  (ratio 2.609x).
- C7: kernel output vs an independent host float64 reference,
  `max_abs = 6.14766e-06` (the printed `max_diff`).

Not covered: non-PyPTO kernels, a PyPTO kernel with a different core mix, a
different PyPTO/CANN/torch_npu version, host-side framework marks, and any
performance or speed claim.
