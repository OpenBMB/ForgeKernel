# PyPTO coding agent — one measured iteration

Implement the next milestone decisions, not an optimization of your own choosing.
This prompt is operator-agnostic. The frozen task card's README and
`eval/leaderboard/problem.json` define entry points, shapes, dtypes, dispatcher
classes and packaging. Preserve all of them. Edit only `{KERNEL_FILE}` in
`{WORKSPACE}`; standalone probes may be written under `notes/`.

## Inputs
- Milestone: `{MILESTONE_PATH}`
- Objective trajectory: `{STATUS_JSONL}`
- Hardware recipes: `{KB_READ_DIR}/index.md` (only `{KB_ARCH}`)
- Implementation language: `{LANGUAGE}`

Read the milestone, trajectory and KB index first. Cite any recipe used in your
change description. Never read a reference tree, another branch, another run or
an equivalent operator implementation. Only the current workspace, milestone,
KB and permitted local PyPTO/PTO API sources are readable. You are offline;
do not use networking, package installation, or reference-library calls.

## Implementation and measurement
Use PyPTO NPU JIT and the API supported by the installed version. PyTorch is for
host allocation/input plumbing only, not for computing the operator. Do not
change `build.sh`, `setup.py`, `cann_bench/`, `eval/`, `gpu_entry.sh`, the judge,
clock, input set or tolerances. The card builds your single file into its wheel.
The driver evaluates every public case with CANN-Bench. Hidden cases are not
available locally. `cann_trace_view` AI-Core E2E is device time, not host time.

The trajectory includes official `cann_bench` scores, benchmark baselines,
hardware bounds, jitter and `profiler` evidence. These are three distinct
quantities: benchmark baseline, campaign architecture reference, hardware bound.
Do not treat a benchmark speedup as a campaign speedup. `msprof` CSV summaries
are hardware observations over all recorded repetitions, not a per-call timing
ratio. Missing counters are unavailable, not zero. No NCU, SASS, PTX or CUDA
sanitizer is available on this device.

## Decision discipline
1. Follow §4 phase order and §3 dependencies (`⇐`, `⇒`, `⊗`). M0 lands the real
   architecture as a unit: decomposition, tiling, dataflow, dtype, UB/L1 buffer
   budgets, core roles and synchronization. M1+ tunes only declared levers inside
   that topology. Only the teacher may reopen M0.
2. Bundle compatible decisions with `+`; state shared buffer/tile/sync budgets.
   Gate rows (`M0-gate`, etc.) must be separate, never bundled. For stepwise
   benefit-coupled groups, start each change with `part k of n` (n ≤ 4).
3. Before first correctness PASS, build substeps are NEUTRAL. A whole-kernel
   smoke probe must prove all launches complete before submitting the gate.
   Mount risky memory/synchronization mechanisms one at a time. After two
   failures of one bundle, unbundle and isolate the first failed contract.
4. A temporary conservative form must start with
   `[PLACEHOLDER: D<x>.<y> → required form]`; its replacement starts with
   `[RESTORED: D<x>.<y>]`. No gate closes with unresolved placeholders.
5. After green, a final phase delivery may include `[CLOSES: M<k>-gate]`; the
   driver closes it with the same measurement if it KEEPs. Unchanged code is
   not remeasured. REOPEN invalidates the listed decisions and dependents only;
   REOPEN-REVERT restores the earlier ledger. Never rebuild unaffected work.
6. A REVERT bans that combination under its recorded conditions. Retry only
   with a changed precondition. Once the DAG is exhausted, return EXHAUSTED;
   the outer loop adds decisions. Do not invent a free-play phase. A
   measurement-unavailable stop (the card's declared timing window was not
   reported) is a harness/configuration defect, not a REVERT of your kernel:
   that combination is not banned, and the run stops for the card to be fixed.

## Targeted NPU probes
Write a small independent `notes/<name>.py` to test one hardware contract with
known synthetic inputs. Use `torch_npu.npu.synchronize()` before reporting.
Numerical probes compute a real finite error and print
`PROBE PASS max_diff=<error>` or `PROBE FAIL max_diff=<error>` (error ≤ 1e3 for
certification). Non-numerical compile/launch probes use `max_diff=n/a`.
Every printed contract must pass. A crash/no verdict is not an accuracy error.
For runtime failures, minimize shape, check bounds/assembly offsets and isolate
DMA/sync stages using CANN logs; never blindly repeat a failed whole-operator run.
Request the driver to run it by ending with only:
`PROBE_REQUEST: notes/<name>.py`

Otherwise commit on the current branch, naming decision IDs, then end with:
```
decision_id: D0.1+D0.2
change: objective change, shared budgets and any recipe used
files_touched: {KERNEL_FILE}
```
The driver supplies correctness, timing, profiler and KEEP/REVERT. Never
self-report those as measured facts. When no decision remains:
`EXHAUSTED: <reason>`
