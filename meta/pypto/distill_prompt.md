# PyPTO architecture agent — initial curriculum

Read `{MILESTONES_TEMPLATE_PATH}` and `{META_MILESTONE_PATH}` for the common
milestone skeleton, five-tuples, DAG, no-code boundary and evidence discipline.
The following Ascend rules replace their NVIDIA-specific instrumentation rules.
Operator facts come only from the task card, reference and measurements.

You may read the declared source oracle `{ORACLE}` in this source-mode run.
Do not copy, rename or transliterate its implementation into the milestone or
KB. The coding agent implements every decision independently.

## Measured inputs
Architecture/campaign reference (the ONLY stop anchor):
{ARCH_REF}

Reference baseline archive, including CANN official data and profiler:
{REF_BASELINE}

Timed-window contract:
{WINDOW_CONTRACT}

Hardware KB: `{KB_READ_DIR}/index.md`, architecture `{KB_ARCH}`.
Write milestone v1 to `{MILESTONE_PATH}`.

## Curriculum
Use the exact common headings `## 0.` through `## 6.` and `## Worked Example`
from the policy. In §3 every D<x>.<y> has What / Why / Trigger / Cost / Depends on.
Use explicit `⇐` prerequisites and `⊗` benefit-coupled groups. Every phase has a
`### M<k> Gate` block and a literal `mechanism-verification line`.

M0 is the final architecture, not a naive warm-up: decomposition, dataflow,
intermediate ABI, precision, tiles, core roles, UB/L1 storage, transfer/compute
ordering, synchronization and tail handling. Explain each buffer's producer,
consumer, lifetime, capacity and ownership. Distinguish compiler-managed
synchronization from explicit synchronization. Do not infer CUDA warp roles,
register counts or bank conflicts from PyPTO source.
M0 gate is correctness plus measurable architecture/mechanism predicates,
without an absolute latency threshold. Starting tile/stage/unroll parameters
that remain tunable need explicit M1+ decisions with candidate ranges, triggers
and measured tradeoffs. No preplanned placeholders. Uncertain mechanisms stay
open questions, not acceptance lines. A topology limit reopens the earliest
M0 dependency; it is not disguised as late tuning.

## Ascend evidence channels
- `[VERIFY: correctness]`: official CANN all-case judge.
- `[VERIFY: cann-trace-view]`: case `op_times.trace_view` AI-Core time, jitter,
  AICPU gap. Timing alone does NOT prove overlap, memory traffic or stalls.
- `[VERIFY: msprof-direct]`: actual available CSV table columns (task type,
  duration, block dimensions and observed launches). Cite file, case, job and
  repeat scope. `kernel_details` and `op_summary` are distinct sources; never
  add their durations together or divide totals by an assumed repeat count.
- `[VERIFY: msprof-derived]`: explicit derivation using those actual columns.
- `[VERIFY: probe]`: a targeted NPU experiment with job and measured verdict.
- `[VERIFY: inferred]` / `[VERIFY: none]`: hypotheses/build guidance only, never
  hard gate lines. Missing msprof data is unavailable, not a fabricated metric.

NCU/SASS channels and their SYNC/GATHER/PHASE vocabulary are NVIDIA-specific;
do not use them here. State Ascend synchronization/feed/phase contracts in plain
language with verifiable evidence instead. Every numeric performance statement
must carry `[MEASURED, job N]` or `[ESTIMATE]` with its derivation. Every gate
predicate needs a real channel; do not label source inspection as a probe.

`target_ms` derives only from the campaign architecture reference in the
`{GATE_WINDOW}` window and the injected campaign target. Preserve
`target_window: {GATE_WINDOW}`. CANN `baseline_perf_us`, `speedup`, `t_hw_us`,
case/per-operator/overall scores are benchmark metadata, not `ref_*`. Preserve
the official scoring version; do not recalculate the paper's 1.1.2 score from
2.0.0 metadata. Separate-job reference comparisons are not paired same-job.

Use certified KB recipes by name where relevant. Record dead ends with a
symptom, evidence and retry precondition. Do not manufacture certification.
Write the complete milestone file; final reply is a one-line summary.
