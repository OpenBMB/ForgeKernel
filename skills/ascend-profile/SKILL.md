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
in isolated copies, evaluates the official case set, alternates their order,
archives raw evidence, and computes a conservative two-sided noise floor.
It never edits the working candidate and never replaces the gate clock with an
agent-written benchmark. At 20 JIT-heavy cases this can be expensive; ask one
falsifiable question per experiment. A null/negative result becomes a no-go with
explicit retry preconditions, not an invitation to repeat the same experiment.

## Narrowing a job's case set (the iteration-cost lever)

Every case of a shape-specialized PyPTO kernel is its own JIT compile, so a
20-case job pays 20 compiles whatever the question was. A scout job evaluates the
card's declared representative subset instead: same official task files, same case
ids, same golden inputs, same published baselines, one build, 5 compiles.

Three ways to ask for it, all landing on one adapter input:

- **experiment spec** (the teacher/recon path, and the one to use):
  `{"hypothesis": ..., "edits": [...], "repeat": 3, "cases": "scout"}`.
  `"cases": "gqa_8"` is a single worst-case probe; `"gqa_1,gqa_8"` is an explicit
  list; leaving `cases` out runs the whole declared set, as before.
- **gpu_entry task**: `bash gpu_entry.sh scout` (or `scout gqa_1,gqa_8`).
- **environment**: `FK_CANN_CASES=scout` (`all` / `full` / `gate` / `ref` /
  `reference` / `confirm` mean the full set; unset means the full set).

The subset is built as an isolated official task root under `CANN_BENCH_DIR`
(`cases.yaml` filtered, everything else symlinked to the official files, and the
official `metadata/` linked in), never as a vendor flag: the pinned CANN-Bench has
no case-set option. Unknown ids, duplicate ids and a missing `scout_cases` are
refused, and the gate tasks (`all`, `correctness`, `perf`, ...) clear any inherited
`FK_CANN_CASES`, so a scout can never be the row a KEEP was read from.

A scout result always carries `workload_scope`, which the loop's eval parser
refuses as a gate reading, and the probe runner prints `PROBE SCOUT` rather than
`PROBE PASS`:

```
PROBE SCOUT max_diff=0.0 gain_pct=12.3 noise_pct=1.8 cases=5 (FK_CANN_CASES=scout)
ASCEND-PRICE-JSON: {"verdict": "SCOUT-PRICE", "certifies": false, ...}
```

That is deliberate. Cite the numbers as a pricing reading that buys or kills a
full-set gate; a scout cannot certify a lever, and a probe that returns no
`PROBE PASS`/`FAIL` line is the expected outcome here, not a broken probe. Cost is
recorded on every job (`cann_bench.cost`: wall seconds, budget, requested cases,
and `interrupted` when the adapter stopped the evaluator itself), which is what
makes "this campaign is compile-bound" a readable measurement.
