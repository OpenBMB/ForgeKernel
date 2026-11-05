# Provenance: `pypto-dynamic-execution`

Human-side evidence for the recipe
`hw_kb/pypto/ascend950/pypto-dynamic-execution.md`. Not in any agent read
set.

## Why this entry exists

The M0 audit of this campaign reconciled the gate contract against the raw
msprof columns of the KEEP job (job `1793834140`) instead of the summary table,
and found an operator-agnostic interface fact missing from `hw_kb/`: a PyPTO
call is recorded by msprof as three device tasks sharing one kernel name, and
the official `kernel_details` per-case time is the median of the five repeat
per-iteration sums of those three rows. Reading that as kernel work inflated
the raw candidate-to-campaign-baseline ratio; the compute row alone is the
kernel-quality currency.

The diagnosis class was `G3` with deficient locus `KB:pypto-dynamic-execution`.
This entry is that KB recipe; the operator's milestone was not reopened.

## Row model measured in the campaign

`[MEASURED, job 1793834140, official_report + cann kernel_details]`

- 640 `kernel_details` rows in the job: 160 harness cache-clean rows
  (`_Z19CannBenchCacheCleanIDhEvPhS0_llj`, Block Num 56, outside the candidate
  timing window) and 480 candidate rows.
- The 480 candidate rows are 160 calls x 3 task rows: per call, two `AI_CPU`
  control rows with `Block Num 0` and one `MIX_AIC` compute row with
  `Block Num 28`, `Mix Block Num 56` (= 2 x 28). All three carry the same
  candidate kernel name; there is no second candidate kernel name.
- The 160 calls are 20 public cases x 8 recorded calls (3 warm-up + 5 repeat).
- Re-deriving the official `elapsed_us` from the CSV as the median of the five
  repeat per-iteration sums reproduces all 20 official case values and the
  device median 40.239995907 ms.
- MIX_AIC-row-only geomean over the 20 cases: 17.105 ms. Control-task tax:
  geomean 2.353x, per-case 2.05x–3.05x.
- Campaign baseline `[MEASURED, job 1793797379]`: 0.061163533474975 ms, one
  `MIX_AIC` row per call (`FusedInferAttentionScore`, no control rows). Raw
  ratio 657.9x; tax-credited ratio 279.7x.
- The measured control rows overlap the compute row, so the tax is
  double-counted runtime overhead, not extra compute work. The certification
  probe tests the overlap directly on a synthetic kernel.

## Timing-basis channel failure

`[MEASURED, job 1793833012]` A job whose tree carried
`timing_basis=cann_trace_view` while the invocation used
`--perf-metric-strategy kernel_details` failed the adapter with
`trace_view.aicore_e2e required`: under that strategy the official report's
`op_times` carries `device_kernels` only, and no `op_times.trace_view` was
banked under the card's basis. The card's `eval/cann_task.json` used in the
KEEP job therefore declares `timing_basis=cann_kernel_details`, and the
`kernel_details` window is the currency this entry documents.

## Certification probe (verdict)

Job `1793835682`, round 1: `PROBE PASS max_diff=6.14766e-06`.

`notes/pypto-dynamic-execution.py` (workspace) builds a generic synthetic PyPTO
kernel — 64 parallel units of a batch-4 8x128 K=128 fp16 cube product with a
vector epilogue — and profiles twelve calls with the torch_npu profiler at
`ProfilerLevel.Level1`, `skip_first=2, warmup=2, active=8, repeat=1`, syncing
between calls. It then exports the raw `PROF_*` session with
`msprof --export=on` and reads `op_summary_*.csv`. All contract lines passed:

- C1: one kernel name (`PYPTO_dyn_exec_kernel`), 24 candidate rows = 16
  `AI_CPU` control rows + 8 compute rows (3 rows per call); no other kernel
  name. C1b grouped the rows by nearest `Task Start Time(us)` into 8 calls.
- C2a: control `Block Num` values `[0]`. C2b: compute `Block Num` values
  `[28]`. C2c: compute `Task Type` `['MIX_AIC']`.
- C3: `Mix Block Num == 2 x Block Num` on all 8 compute rows; observed
  `[(28, 56)]`.
- C4: per-call sum / compute > 1 on 8 calls; tax range 2.540..2.684x, median
  2.615x.
- C5: the three rows overlap in time on 8/8 calls; overlap 88.534..104.800 us.
- C6: last-5 median per-call sum 196.707 us vs compute median 75.384 us
  (ratio 2.609x).
- C7: output vs an independent host float64 reference,
  `max_abs = 6.147660e-06` (probe tolerance 5e-2 absolute / 1e-3 relative L2).

The probe makes no speed claim and certifies no operator-level number.

## What this entry does not certify

- The control-task share is not a constant: the 2.05x–3.05x range is a
  campaign measurement on 20 shapes, not a hardware invariant, and the
  certification probe measured 2.540x–2.684x (median 2.615x) on its own
  synthetic kernel. A different kernel duration moves it.
- No claim is made about a kernel whose core mix differs from cube+vector, or
  about a PyPTO/CANN version other than the one in the Architecture line.
- No claim is made that `op_times.trace_view` exists under any strategy; the
  measured environment did not produce it under `kernel_details`.
- The row model is from the device task tables; host-side framework marks are
  out of scope.

## Reuse requirement

The recipe is operator-agnostic. Its reuse must be visible in the campaign's
student record (the milestone/student may cite
`KB:pypto-dynamic-execution`). Certification of the row model does not imply a
performance gain, and the delivered operator is still judged by CANN-Bench.
