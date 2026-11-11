# `valid_shape` masks assemble writeback on Ascend 950

- Architecture: Ascend 950PR, PyPTO 0.3.0, CANN 9.2
- Status: ✅ Verified
- Certification evidence: job `1793761707`, probe `notes/tile_valid_shape_probe.py`, `PROBE PASS max_diff=0.0`

## Purpose

On a two-dimensional tile with `shape=[8, 8]`, `offsets=[4, 4]` and
`valid_shape=[4, 4]`, `pypto.assemble` transfers the valid rectangle into a
sentinel-initialized destination and leaves cells outside the valid rectangle
unchanged. This certifies valid-only writeback, not any broader padding rule.

## When needed

Use this contract when tail tiles must not overwrite neighboring valid data or
memory outside the logical tensor. It is relevant whenever a fixed tile shape
exceeds the remaining valid extent. Verify each new tile layout and output
composition independently; this probe is not a general out-of-bounds proof.

## Minimal runnable example

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def masked_assemble(
    x: pypto.Tensor([16, 16], pypto.DT_FP32),
    y: pypto.Tensor([16, 16], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8, 8)
    tile = pypto.view(x, shape=[8, 8], offsets=[4, 4], valid_shape=[4, 4])
    pypto.assemble(tile, [4, 4], y)
```

Allocate `x` with known distinct values and `y` with a sentinel. Compare every
cell against independent host-side indexing: valid cells come from the same
coordinates in `x`; all other cells keep the sentinel. Synchronize before reading.

## Gotchas

- **Do not assume raw view invalid storage is zero.** The falsifying experiment
  in job `1793761442` left sentinel values in invalid storage even though valid
  writeback passed. Materialize/read-check padding explicitly when a subsequent
  operation needs defined padding values.
- Full-shape arithmetic after a view may observe invalid cells. Never use that
  observation as a substitute for valid-only assembly.
- Multi-output explicit-padding compositions are not certified by this entry.
  The attempted combination in job `1793761537` did not meet its independent
  expectation. Do not infer padding semantics by chaining calls.

## Retry preconditions

A proposed padding composition needs a new single-purpose NPU probe with a
known sentinel, independently derived expected values and a materialization
step that actually observes the intermediate. Different rank, dtype, overlap or
writeback order requires another probe. Do not retry an identical multi-output
composition expecting a different answer.

## Sources

- Public API: `pypto.view`, `pypto.assemble`, `pypto.set_vec_tile_shapes`.
- Probe verdicts: jobs `1793761442` and `1793761707` (valid-only writeback);
  jobs `1793761442` and `1793761537` (failed padding-composition hypotheses).
