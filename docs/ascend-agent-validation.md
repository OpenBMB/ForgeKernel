# PyPTO / Ascend validation ledger

This document separates harness evidence, real-device evidence and the paper's
hosted experiment. It is not a claim that CANN-Bench 2.0.0 on Ascend950PR
reproduces the paper's CANN-Bench 1.1.2 scores on Ascend910C.

## Execution facts

- Controller: current `feat/pypto-ascend` worktree. Current PyPTO tasks explicitly
  select provider `deepseek` for BOTH roles, endpoint
  `https://api.deepseek.com/anthropic`, model `deepseek-flash[1m]`, effort `max`.
  The provider reads `API_KEY`; native `TEACHER_*` overrides cannot redirect these
  tasks. Earlier experiments used Kimi for the teacher (see routing incident below).
  Secrets remain in gitignored `.env`, never in task snapshots or command lines.
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

## Provider-routing incident and corrected configuration

Changing `TEACHER_MODEL` alone did **not** switch providers: `.env` retained
`TEACHER_BASEURL=https://api.kimi.com/coding` and the old teacher credential.
Native teacher routing in `agents.py` therefore sent requests with a DeepSeek
model string to Kimi. A successful `READY` reply proved connectivity, NOT the
identity of the served model. The later Kimi weekly-quota response exposed this
configuration error; the claimed all-DeepSeek experiment was not valid.

Affected runs (retain their original snapshots/logs; do not relabel them):

- `.artifacts/swiglu-pypto_261104-172829-bdf3/`: hardware measurements remain
  valid, but the teacher provider was Kimi. Final 3+3 confirmation: student
  **0.0557052830 ms**, speedup **1.2103149052×** vs the public PyPTO reference.
  This is not an all-DeepSeek or blackbox result.
- `.artifacts/gqa-pypto_261104-172829-55be/`: reference **0.0612754942 ms**;
  five recon rounds completed, then teacher distillation hit Kimi's weekly
  quota. The retrying controller was stopped when the routing error was found.
  No student result from this run establishes parity with the strong library.

Fix: `global_config/backends.toml` now declares an explicit `deepseek` provider.
All three PyPTO task cards and both task generators use it; teacher-specific
model overrides that removed `[1m]` were removed. Local Kimi `TEACHER_*`
credentials/endpoint were removed. The CLI launcher replaces both Anthropic
API credentials with the selected provider key and clears inherited OAuth.
The isolated Claude configuration remains in use; no personal global settings
were overwritten.

Verified with **both actual task launch commands**: the exported endpoint is
DeepSeek's official Anthropic API, the selected credential variable is
`API_KEY`, and teacher/student each returned `DEEPSEEK_ROUTE_OK` with rc=0.
Regression tests also inject stale Kimi endpoint/model/key and OAuth values,
and verify that neither role falls back to them. Local evidence:
`.artifacts/ascend-gap-work/deepseek-official-routing/preflight.json`.

Configuration for future launches:

```bash
set -a
source local.env
source .env
set +a
unset TEACHER_API_KEY TEACHER_BASEURL TEACHER_MODEL CLAUDE_CODE_OAUTH_TOKEN
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_API_KEY="$API_KEY" ANTHROPIC_AUTH_TOKEN="$API_KEY"
export ANTHROPIC_MODEL='deepseek-flash[1m]'
export ANTHROPIC_DEFAULT_OPUS_MODEL='deepseek-flash[1m]'
export ANTHROPIC_DEFAULT_SONNET_MODEL='deepseek-flash[1m]'
export ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-flash
export CLAUDE_CODE_SUBAGENT_MODEL=deepseek-flash
export CLAUDE_CODE_EFFORT_LEVEL=max
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=786432
```

Old runs freeze `task.toml`: editing `tasks/<op>/` does not rewrite their routing.
For a genuinely all-DeepSeek experiment use `forge new tasks/<op>`, not a resume
of an affected native-provider snapshot. Runtime-only source-mode measurements
must not be presented as independently rebuilt blackbox performance.

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

Last full run before the routing correction: **2085 passed**. The two initial failures were controller-only
missing `torch`; installing the documented test dependency did not weaken or
skip any test.

## Remaining distinction from the paper

This work validates mechanisms and harness behavior on a different NPU and
benchmark version. It does not reproduce the paper's eight-operator 910C hosted
score table, hidden cases or model-controlled experimental comparison. Those
are explicit protocol differences, not gaps that can be closed by re-labeling a
local score.
