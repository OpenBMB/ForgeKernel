# Device-side per-launch floor via device-event sandwich

✅ Verified on Ascend 950PR (job `1793791124`, probe
`gqa-pypto_261104-172829-55be/operators/_gqa-pyptoref/notes/recon_round5.py` P3).

**Fact.** The universal *device-side* per-launch floor on Ascend950PR is ≈ 0.5–1.2 µs
(trivial 256-element fp16 kernel, median sandwich spacing over 5 trials at M ∈ {4,8,16}).
Real tiny kernels land higher: a 4 MB fp16 reduction ≈ 4.7–5.7 µs, a small
[16,512,128]×[16,128,128] matmul ≈ 2.6–3.3 µs. This is ~10× below the ~11.9 µs
host-enqueue-paced *wall-clock* floor measured separately — do not use wall-clock floors
to reason about device-time metrics (kernel_details / trace_view AI-Core E2E).

**Method (the certified msprof-free path).** Enqueue
[start-event, LONG-kernel, tiny × M, end-event]; all M tiny ops must be enqueued while
LONG still runs on-device (LONG ≈ 200–400 µs ≫ M × ~12 µs host enqueue), so host pacing
cannot hide device time. (combo − solo_LONG)/M = tiny device duration + inter-kernel gap
(gap ≥ 0), i.e. an **upper bound** on the tiny kernel's device Duration. Use medians —
per-trial mins can be garbage (one −27.9 ms outlier observed at M=4). Sanity-check the
LONG kernel against a known bandwidth number.

**Dead channels in torch_npu 2.13.0.rc1 (do not retry blindly).**
- kineto chrome-trace export contains **0 kernel events** (139 events, all host
  `cpu_op`/`Trace`) — device durations are absent from the export, not mis-filtered.
- `torch_npu.profiler` msprof backend: `export_type` ctor kw rejected, no `.export()`
  attribute; produces only framework-level op marks. Per-kernel durations exist only via
  the harness/standalone msprof path.

Related: [[valid_shape_assemble]]
