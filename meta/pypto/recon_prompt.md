# PyPTO / Ascend reference reconnaissance

The reference is runnable through the teacher-only tree `{RECON_DIR}`. Do not
read or decode sealed payloads, keys, installed reference packages, compiler
source caches or hidden task inputs. In blackbox mode the source is unavailable;
record uncertainty rather than reconstructing source. In source mode only the
explicitly allowed oracle may be read, and source intent is not measurement.

Use these measured reference archives first; no repeat job is needed merely to
copy a number:
{ARCH_REF}
{REF_BASELINE}
Window: {WINDOW_CONTRACT}

Read `{KB_READ_DIR}/index.md`. Write both `{RECON_REPORT}` and
`{CANDIDATES_PATH}`. The report records:
1. Task/hardware/version/case scope and job provenance; timing currency and
   separation of campaign reference, CANN benchmark baseline and hardware bound.
2. Casewise trace_view times, jitter and AICPU gaps; msprof kernel_details and
   op_summary table observations with exact CSV source/repetition scope.
3. Inferred architecture candidates (kernel split, dataflow, tile/core/buffer
   choices), each separated from the actual measured facts. List competing
   interpretations and a falsifiable experiment distinguishing them.
4. Explicit unavailable channels/open questions and targeted NPU-probe needs.

Do not translate missing Ascend counters into NCU/SASS metrics. Do not add two
profiler views' durations or infer per-call kernel count from aggregate repeats.
Use `[MEASURED, job N]` and `[ESTIMATE]`; use only Ascend verification labels.
A single candidate is acceptable if evidence distinguishes it; source-free
inference must not claim exact hidden buffer topology without a probe.

Optional measurement: write a real `notes/<name>.py` inside `{RECON_DIR}` and
end only with `RECON_REQUEST: notes/<name>.py`. The driver commits/runs it.
It may invoke `bash gpu_entry.sh ref` for the fixed reference evaluation or run
an independent synthetic NPU experiment, but must not access the reference's
implementation. At most {MAX_RECON_ROUNDS} rounds. After both reports are
written, end with a brief summary; never fabricate a completed measurement.
