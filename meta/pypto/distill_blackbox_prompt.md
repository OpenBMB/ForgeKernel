# PyPTO architecture agent — sealed-reference curriculum

The reference source is NOT readable. Never inspect, decode, disassemble or
unpack its payload. Only use its measured behavior and the recon report below.
The milestone must contain decisions and contracts, never implementation code.

Read the common schema `{META_MILESTONE_PATH}` and
`{MILESTONES_TEMPLATE_PATH}`. Write `{MILESTONE_PATH}` with their exact §0–§6 and
Worked Example headings, five-tuples (What/Why/Trigger/Cost/Depends on), explicit
`⇐` DAG edges, `⊗` companions, ordered M0→M1+ phases and per-phase Gate blocks.
Each phase needs a literal mechanism-verification line. No free-play phase.

Architecture reference (sole stop anchor):
{ARCH_REF}

Reference measurements (including official CANN metadata and Ascend profiler):
{REF_BASELINE}

Window contract:
{WINDOW_CONTRACT}

Recon report: `{RECON_REPORT}`. Hardware KB: `{KB_READ_DIR}/index.md` (`{KB_ARCH}`).

M0 settles the real decomposition, dataflow, dtype, intermediate ABI, tiling,
core roles, UB/L1 buffer/lifetime budgets and DMA/compute ordering. Explicitly
state what evidence establishes synchronization, feed and phase relations.
Unknown topology is an open hypothesis, not a fake measured fact. M0 gate is
correctness + measured architecture/mechanism, not an absolute time threshold.
M1+ only tunes declared tile/stage/unroll/buffer levers with ranges and
tradeoffs inside that topology. Topology changes reopen M0 dependencies.
Never schedule a placeholder as the initial architecture.

Ascend channels replace NVIDIA rules in the common template:
`[VERIFY: correctness]`, `[VERIFY: cann-trace-view]`, `[VERIFY: msprof-direct]`,
`[VERIFY: msprof-derived]`, `[VERIFY: probe]`. NCU/SASS and their warp-specific
SYNC/GATHER/PHASE tags are inapplicable. Cite available CSV columns, case, file,
job and repetition scope; absent counters remain unavailable. Do not add
kernel_details and op_summary timings together. Timings cannot prove a stall
or overlap claim. Inferred/none rows stay outside gates. Every performance
number carries `[MEASURED, job N]` or `[ESTIMATE]` plus its derivation.

Preserve the injected target and `target_window: {GATE_WINDOW}`. CANN benchmark
baseline, hardware bound and scores never replace the campaign reference.
Cross-job ratios are not paired same-job. Public local evaluation is not the
paper's hidden hosted scoring. Cite reusable certified KB recipes by name;
no-go entries need evidence and a retry precondition.

Write the whole milestone. Final reply: one-line summary.
