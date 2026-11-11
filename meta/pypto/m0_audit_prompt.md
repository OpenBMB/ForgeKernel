# PyPTO M0 architecture audit

The first whole-operator correctness PASS has landed. Audit structure, not
whether its speed already meets the final target. Read `{MILESTONE_PATH}`,
`{TRAJECTORY_PATH}`, `{META_TRAJECTORY_PATH}`, `{META_MILESTONE_PATH}` and the
appended backend-labelled Ascend evidence. Use `{KB_READ_DIR}` only for this
architecture. Campaign reference:
{ARCH_REF}
Window contract:
{WINDOW_CONTRACT}

Reconcile per-case correctness, observed launch/task types, block dimensions,
repetition-scoped durations and trace_view AICPU gaps with the M0 contracts.
Check UB/L1 buffer budgets, producer/consumer ownership, intermediate ABI,
precision, DMA/compute ordering, tails and compiler-managed synchronization.
Only actual msprof columns or NPU probes verify hardware mechanisms; unavailable
stalls/counters are not NCU-equivalent. Do not substitute benchmark baseline or
hardware bound for the campaign reference. State uncertainty explicitly.

Rule every discrepancy: expected and assigned to a reasonably priced M1 lever;
student/API defect (G3 recipe); or milestone defect (G1/G2/R1/R2). Choose the
shallowest defect. A structural reopen needs a current-tree measured probe price;
source inference alone is not a price. If all measurable contracts reconcile and
unknowns are explicitly discharged, return OK. Do not repeat a clean audit.
{VERDICT_PROBE_BRIEF_AUDIT}

```
cause_class: OK | G1 | G2 | G3 | G4 | R1 | R2
deficient_locus: <D id / KB:mechanism; empty when OK>
evidence: <measured source, case and job; unresolved channels>
reopen_price: <±x% via probe:notes/<name>.py#<index>; required for R1>
phase_verdict: <only when requested>
locus_segment: <only when measured>
patch_direction: <single defect, or what reconciles when OK>
```
