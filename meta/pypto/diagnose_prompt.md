# PyPTO architecture agent — measured outer-loop diagnosis

Read `{META_MILESTONE_PATH}`, `{MILESTONE_PATH}`, `{TRAJECTORY_PATH}` and
`{META_TRAJECTORY_PATH}`. Operator facts come only from these inputs, the frozen
task and allowed source `{ORACLE}` (never read source in blackbox mode).
Hardware KB is `{KB_READ_DIR}/index.md`. Diagnose one defect; do not patch yet.

The appended Ascend evidence digest is primary hardware evidence. Read both KEEP
and REVERT rows, per-case times, jitter, AICPU gaps, official scores and available
msprof CSV tables. Compare like case/source/repetition scopes only. Benchmark
baseline and hardware bound are NOT the campaign reference. An absent metric is
unavailable, not zero. No NCU, CUDA, SASS or per-instruction stall channel exists.
A timing change alone does not establish overlap or memory-traffic attribution.

First construct an optimistic reachability floor from justified hardware/shape
arithmetic and measured structurally serial costs. Unproven serial costs count
as zero. `floor_opt > target * 1.20` is BLOCKED; otherwise OPEN (not proof of
reachability). The measured reference is an existence proof; reconcile any
contradiction. A low benchmark score does not prove architecture infeasibility.

Classify using objective iteration/job evidence:
- G1: guidance underdetermined; G2: decision too coarse; G3: hardware API unknown
  (add a probe-certified KB mechanism); G4: student noise; G5: scheduling defect
  only if independently observed, never inferred from missing stalls.
- R1: topology ceiling; reopen earliest M0 dependency only with a measured price.
- R2: missing upper-level decision; add an explicit lever.
- INFEASIBLE: stop only with BLOCKED arithmetic and evidence excluding both
  tuning and architectural alternatives. Null levers alone cannot justify it.

A structural/split patch needs a price from a driver-run probe, current-tree
lever table, or certified recipe. Use the seeded Ascend ablation tool (see
`skills/ascend-profile/SKILL.md`), not NVIDIA pricing tools. A probe tests one
hypothesis, uses synthetic data for hardware contracts, or the unchanged official
CANN evaluator for timing. Full-case evaluation is mandatory for gate prices.
Every claim cites a job and criterion; a failed test is also evidence.
{VERDICT_PROBE_BRIEF}

Do not repeat a reverted direction without a changed retry precondition. Read
bundles as bundles; do not assign their timing to individual members. When small
levers only work jointly, propose one explicitly coupled additive group.

End with this exact structured protocol (enum values have no trailing prose):
```
target: <metric and target>
achieved: <value and iteration>
gap_pct: <gap>
reachability: floor_opt=<number> ms vs target=<number> ms -> OPEN | BLOCKED
curve_shape: <shape>
cause_class: G1 | G2 | G3 | G4 | G5 | R1 | R2 | INFEASIBLE
evidence: <job/iteration measurements and derivations>
deficient_locus: <D id or KB:mechanism>
locus_depth: foundation | upper | n/a
patch_kind: additive | structural | split | kb-example | none(student-noise)
reopen_price: <±x% via probe:notes/<name>.py#<contract index>; empty if not reopening>
phase_verdict: <close | continue:Dx.y, only when requested>
locus_segment: <only when measured>
patch_direction: <one actionable sentence>
confidence: high | medium | low + reason
```
