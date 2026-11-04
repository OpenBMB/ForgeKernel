# PyPTO / Ascend validation ledger

This document separates harness evidence, real-device evidence and the paper's
hosted experiment. It is not a claim that CANN-Bench 2.0.0 on Ascend950PR
reproduces the paper's CANN-Bench 1.1.2 scores on Ascend910C.

## Execution facts

- Controller: current `feat/pypto-ascend` worktree. Teacher resolves through
  `TEACHER_*` to `k3.1-pro`; student uses the task's model. Secrets stay in `.env`
  and are not written to task snapshots, commands, logs or commits.
- NPU: Ascend950PR_9579, CANN 9.2, PyPTO 0.3.0, torch_npu 2.13.0.rc1.
- Official evaluator: CANN-Bench 2.0.0 at
  `059cb61d41b279ad021160517bd2a5f6b7a05e01`; public cases, official golden
  correctness. Local public scores are not the paper's hidden hosted scores.
- Timing is cross-job median-vs-median, not CUDA same-job paired timing. CANN
  benchmark baseline and `t_hw_us` are preserved separately from the campaign
  architecture reference. Three student and three fresh-reference jobs are
  required for a final target/promotion decision.
- Official JSON, evaluator logs, msprof export records and raw CSVs travel via
  bounded, SHA-256-checked job artifacts. Per-iteration trajectory rows include
  official scores, baseline/bound, jitter, AICPU gap and memory fields.

## Real-device evidence

### M1 + outer loop

Run: `.artifacts/swiglu-pypto_261104-002946-c108/`.

The run executed M0, M1 and an outer M0 audit/patch. `meta_status.jsonl` records
`KEEP_PATCH`; the M1 measurement records a KEEP. It did not make a meaningless
NCU request. Both reference and student produced real `msprof` exports.

**Protocol finding:** the optimistic in-loop stop reached the 1.05 target, but
final three-plus-three confirmation measured only 1.0448916506×. The initial
shared `stop=target` path marked this run done too early. The code now requires
the later complete confirmation to meet the target. The historical artifact is
kept as an example of why optimistic stop readings are not final claims.

### Sealed cross-generation reference

Run: `.artifacts/swiglu-pypto-g2_261104-103528-6867/`.

The parent run's confirmed best is the next reference. Its source is sealed at
run installation; the student starts from a signature-only stub. Recon uses the
actual reference job and both `kernel_details` / `op_summary` tables, explicitly
distinguishing measured facts, inferred candidates and unavailable mechanisms.
The student produced a real M0 implementation and an NPU smoke PASS. The first
full M0 evaluation was REVERTed at the old 1800-second job ceiling after several
cases had passed, before any eval JSON existed. This exposed a harness budget
bug, not a numerical verdict: `[loop] eval_timeout_s` now configures the full
20-case job (7200 s here), distinct from CANN's operator timeout. The failed
row remains in the trajectory. Final confirmation outcomes are recorded below
when complete.

### Level-4 strong-library reference

Run: `.artifacts/gqa-pypto_261104-105938-4599/` (an earlier generated package
failure is retained in `.artifacts/gqa-pypto_261104-105255-0bee/`).

The reference is the Huawei `FusedInferAttentionScore` library path, sealed
independently of a PyPTO-from-stub student. Reference: 20/20 public cases,
geomean 0.0613804073 ms, official compilation/function/performance scores and
hardware bounds preserved. Its msprof profile independently identifies the
fused top-level attention launch. The candidate is still under evaluation;
no claim of reaching the strong-library target is made here.

### Probe-certified KB

- Job `1793761442`: valid-only assembly PASS; raw-view automatic zero padding
  FAIL. These are separate claims, not a single result to average away.
- Job `1793761537`: the attempted multi-output explicit-padding composition
  failed its independent expectation. It remains a no-go, with a retry requiring
  a new single-purpose materialization probe.
- Job `1793761707`: single-output valid-only `assemble` probe PASS,
  `max_diff=0.0`. `hw_kb/pypto/ascend950/valid_shape_assemble.md` records only
  this narrow verified contract, the failed padding hypotheses and retry
  preconditions. `docs/kb-provenance/pypto/valid_shape_assemble.md` preserves
  the experiment history. The recipe passed KB lint. The GQA milestone then
  explicitly prescribes `valid_shape_assemble` for valid-only output tails,
  while requiring fresh probes for the new K/V-tail layouts. This is the
  measured cross-operator reuse path, not merely a file added to the KB.

## CPU regression

The suite includes report conversion, trajectory/baseline round trips, artifact
path/checksum handling, no-NCU routing, verification-channel validation, teacher
model selection, sealed promotion, final-target rejection, cross-job noise,
Level-4 task generation and independent experiment edit checks.

Latest full run: **2083 passed**. The two initial failures were controller-only
missing `torch`; installing the documented test dependency did not weaken or
skip any test.

## Remaining distinction from the paper

This work validates mechanisms and harness behavior on a different NPU and
benchmark version. It does not reproduce the paper's eight-operator 910C hosted
score table, hidden cases or model-controlled experimental comparison. Those
are explicit protocol differences, not gaps that can be closed by re-labeling a
local score.
