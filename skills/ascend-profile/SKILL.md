---
name: ascend-profile
description: Read Ascend trace_view/msprof evidence and run a targeted full-case PyPTO ablation.
---

# Ascend measurement and verdict experiments

Use only for PyPTO NPU tasks. No NVIDIA metrics are available here.

- `cann_trace_view` is the task's AI-Core E2E timing currency. Its jitter and
  AICPU gap are not per-instruction stall counters.
- `msprof` exports the CANN collector's raw PROF sessions. `kernel_details.csv`
  and `op_summary_*.csv` are separate views of those sessions, NOT independent
  experiments. Do not add their times. Digests explicitly cover all recorded
  repetitions; do not read launch count as one invocation without establishing
  the repetition/window convention.
- Raw official reports, export logs and CSVs are checksummed in each controller
  job's `.jobs/<job>/artifacts/`. Missing executable, absent raw session,
  failed export, unsupported schema and empty table are unavailable evidence,
  not zero-valued readings. A partial export cannot certify a missing channel.
- `baseline_perf_us` / `t_hw_us` / official scores are benchmark metadata. They
  never replace the declared campaign reference or produce a `vs_ref_*` value.

For a mechanism hypothesis write an isolated `notes/<name>.py` with synthetic
inputs, an independent expected result and a finite error norm. Request it via
PROBE_REQUEST; only a completed NPU job certifies a recipe.

For a topology/lever price, the driver seeds `notes/ascend_probe.py`. Write an
experiment JSON with `hypothesis`, `edits: [{old, new}]`, `repeat: 3` and
`min_gain_pct: 1.5`. A tiny notes script may call
`run_experiment(Path.cwd(), json.loads(Path('notes/experiment.json').read_text()))`.
Request that script via PROBE_REQUEST. The frozen runner builds base and variant
in isolated copies, evaluates the full official case set, alternates their order,
archives raw evidence, and computes a conservative two-sided noise floor.
It never edits the working candidate and never replaces the gate clock with an
agent-written benchmark. At 20 JIT-heavy cases this can be expensive; ask one
falsifiable question per experiment. A null/negative result becomes a no-go with
explicit retry preconditions, not an invitation to repeat the same experiment.
