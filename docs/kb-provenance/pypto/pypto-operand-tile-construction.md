# Provenance: `pypto-operand-tile-construction`

Human-side evidence for the recipe
`hw_kb/pypto/ascend950pr/pypto-operand-tile-construction.md`. Not in any agent
read set.

## Mechanism and public contract

The recipe is abstracted from the operator campaign that needed it. The public
PyPTO DSL source states the contract it rests on:

- `pypto.operation.view` — the docstring says a view is "more like slice" than
  a reshape; `shape` is the per-axis extent and `offsets` is the per-axis start,
  so an extent is consumed along the same source axis.
- `pypto.op.matmul.matmul` — 3-D/4-D operands are batched over their leading
  axes with broadcasting; with `b_trans=True` the right operand is read as
  `[batch, N, K]`, with `b_trans=False` as `[batch, K, N]`.

Consequently, an operand that is "N consecutive rows at one index of the middle
axis of a rank-3 tensor" must be written `view(x, [N, 1, D], [r0, g, 0])` and
then moved to `[1, N, D]` by `pypto.reshape`; the unit leading axis is the
batch axis the cube broadcasts.

## Probe history (verdicts only)

- Job `1793822998` (operator-side probe K1): a strided 3-D Q-operand view fed
  to the cube is exact (C5 `rel_l2 = 5.8e-08`), while the legacy KV construction
  `view(k, [1, NT, D], [row, kh, 0])` is a wrong window (C3 `max_abs =
  1.076e+01`) and stays wrong through the cube (C6 `max_abs = 4.535e+01`).
  Probe-harness defects on other rungs (destination dtype, wrong host shapes)
  produced no verdict for those rungs.
- Job `1793823806` (view-semantics probe): H1 `view(k, [NT, 1, D], [R0, KH, 0])`
  is exact against host indexing (`max_abs = 0.0`); H2 the legacy
  `view(k, [1, NT, D], [R0, KH, 0])` differs from the intended box by
  `max_abs = 1.073e+01` (and matches neither a flat walk nor a zero fill);
  H3 the 3-D view reshaped to 2-D is exact (`max_abs = 0.0`); H4/H5/H6/H7
  constructions that reshape a 4-D view or mix tile ranks were rejected at
  compile time (`REGISTER_COPY` / `CAST` tile-shape mismatch); H8 the 2-D
  matmul control is sound (`rel_l2 = 5.583e-08`).
- Job `1793824430` (operand-fix probe): P1 the token-first view reshaped to
  `[1, NT, D]` is exact through the cube (`rel_l2 = 5.836e-08`); P2 a two-tile
  non-causal chained consumption is exact (`worst rel_l2 = 2.077e-04`); P4-P7
  full operator bodies at four geometries pass with this construction
  (`1.150e-05`, `1.624e-03`, `2.808e-04`, `5.201e-05`).
- Job `1793817015` (operator-side whole-case probe, legacy construction): only
  the single case whose grouped axis has extent 1 — the one geometry where the
  legacy box happens to be contiguous — passes (`4.41e-05`); the strided cases
  fail (`1.155e-01`, `6.657e-02`) or die on dtype/harness errors.
- Job `1793825465` (this certification, round 1): mechanism-only contracts
  C1-C8 in `notes/pypto-operand-tile-construction.py`, every contract line
  PASS with the device synchronised before each host-side float64 comparison.
  Readings: C1 fp16 view exact (`max_abs = 0`), C2 bf16 view exact (`0`),
  C3 fp32 view exact (`0`), C4 reshape element order exact (`0`),
  C5 fp16 `b_trans=True` cube (`rel_l2 = 5.919e-08`, tol 2e-3),
  C6 bf16 `b_trans=True` cube (`rel_l2 = 5.590e-08`, tol 2e-2),
  C7 fp16 `b_trans=False` cube (`rel_l2 = 5.919e-08`, tol 2e-3),
  C8 negative control: a middle-axis run length reads a different window
  (`max_abs = 1.105e+01`; the line is non-numeric by design, `max_diff=n/a`).
  Whole-probe verdict: `PROBE PASS max_diff=5.91913e-08`.

## What this entry does not certify

- Cube operands with N != D, D > 256, partial last-axis boxes, 4-D sources and
  fused multi-consumer chains. Each needs its own probe.
- The materialise-then-reshape fallback. It was attempted as H5 at job
  `1793823806` and rejected at compile time in that form; a differently shaped
  attempt would be a new experiment.
- Performance. Nothing in the certification measures time.

## Open risk carried outside this entry

The causal chain experiment P3 (job `1793824430`) failed at an isolating
`(S_q=4, G=16, D=128, S_kv=256)` geometry (`worst rel_l2 = 8.668e-02`), while
the non-causal chain P2 and the full bodies P4/P6/P7 — including multi-tile
bottom-right causal decode — passed. The anomaly involves the masking rung of
that hand-written chain, not the operand construction that P1 isolates, and it
sits outside the delivered body's envelope. It remains unexplained; it is
carried as risk on the operator side and must not be attributed to this recipe
without a new isolating probe.

## Reuse requirement

The recipe is operator-agnostic. Its reuse must be visible in the campaign's
student record (the milestone/student may cite `KB:pypto-operand-tile-construction`).
Probe certification does not imply a performance gain, and the actual delivered
operator is still judged by CANN-Bench.
