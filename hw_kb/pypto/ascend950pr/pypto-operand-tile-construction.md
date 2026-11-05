# Operand-tile construction: per-axis `view` + `reshape` for a batched cube operand

- Architecture: Ascend 950PR, PyPTO 0.3.0, CANN 9.2
- Status: ✅ Verified
- Certification evidence: `PROBE PASS max_diff=5.91913e-08, job 1793825465`,
  probe `notes/pypto-operand-tile-construction.py` (round 1). Scope: rank-3
  per-axis view exactness at fp16/bf16/fp32, `reshape([N,1,D] -> [1,N,D])`
  element order, the constructed operand through the batched cube matmul
  (`b_trans=True` fp16 and bf16, `b_trans=False` fp16), and the negative
  control that a middle-axis run length selects a different window.

## Purpose

`pypto.view` is a per-axis slice (its public DSL docstring says it is "more like
slice" than a reshape), so each extent of a view is consumed along that same
source axis. On a rank-3 GM tensor `x[T, G, D]`, the operand that spans `N`
consecutive rows at one index `g` of the middle axis, over the full last-axis
extent, is therefore built with every extent on its own source axis:

    v  = pypto.view(x, [N, 1, D], [r0, g, 0])
    op = pypto.reshape(v, [1, N, D])

The view reproduces exactly the strided box `x[r0:r0+N, g, :]` (bit-exact
against host-side indexing), and the reshape preserves element order, so the
result is a legal operand for `pypto.matmul`. The leading unit axis is the batch
axis that `pypto.matmul` broadcasts when the other operand carries a larger
batch extent. The certification probe covers both cube orientations:
`b_trans=True` (B is `[batch, N, K]`) and `b_trans=False` (B is
`[batch, K, N]`).

## When needed

Use this construction whenever an operand is a strided window of a higher-rank
GM tensor and a batched `pypto.matmul` consumes it — in particular when the
operand is one index of an inner grouped axis and the run of rows must become
the matmul's token/N axis.

Not applicable when the source is already 2-D and contiguous at the needed
orientation (a plain 2-D `view` is enough), or when the operand needs only a
partial last-axis slice rather than the source's full last extent (not
certified here; see Gotchas).

## Minimal runnable example

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def strided_b_operand(
    a: pypto.Tensor([32, 8, 128], pypto.DT_FP16),
    x: pypto.Tensor([512, 8, 128], pypto.DT_FP16),
    y: pypto.Tensor([4, 8, 128], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(4, 8, 128)
    pypto.set_cube_tile_shapes([16, 16], [128, 128], [128, 128])
    av = pypto.view(a, [4, 8, 128], [0, 0, 0])
    op = pypto.reshape(pypto.view(x, [128, 1, 128], [128, 3, 0]), [1, 128, 128])
    pypto.assemble(pypto.matmul(av, op, pypto.DT_FP32, b_trans=True), [0, 0, 0], y)
```

Feed synthetic values, synchronize the device, then compare `y` against a
float64 host-side computation over the same strided box
(`a[0:4, 0:8, :]` times `x[128:256, 3, :]` transposed). This is the shape of
the fp16 cube contract exercised by the certification probe (relative L2 <=
2e-3); the same construction with `b_trans=False` is its K x N orientation.

## Gotchas

- **Do not put the run length on the middle axis.**
  `pypto.view(x, [1, N, D], [r0, g, 0])` asks for `N` values along the middle
  axis at one row. It is a different window: measured to differ from the
  intended box by `max_abs = 1.073e+01` on fp16 synthetic data (job
  `1793823806`). The middle extent belongs to the source's middle axis and must
  be `1`.
- **The reshape preserves element order, so it is exact only for the
  contiguous box.** Keep the last-axis extent equal to the source's full last
  extent. The probed form covers `D` in {64, 128, 256}; partial last-axis
  boxes are not covered.
- **Match `set_vec_tile_shapes` rank and extents to the tensors the kernel
  produces.** Reshaping a view whose tile rank differs from the configured
  vector tile rank was rejected at compile time (`REGISTER_COPY` / `CAST`
  tile-shape mismatch, job `1793823806`); a 4-D source with a 4-D view reshaped
  to 3-D failed the same way. The rank-3 form in this entry is the probed one.
- **The unit leading extent is what makes the cube broadcast.** Keep it `1`; a
  larger leading extent multiplies the operand instead of sharing it.
- **No performance claim.** This entry is about construction exactness and
  cube-path numerics only; it says nothing about speed.

## Retry preconditions

If a new rank, dtype or geometry fails to lower: change the construct (rank,
tile shapes, or the source layout) and run a new independent probe that
compares against host-side indexing. Do not retry an identical construct
expecting a different answer. The 4-D-source form and the materialise-then-
reshape variant measured at job `1793823806` are not certified; retrying either
requires its own probe. A host-side fold to a 2-D `[rows, G*D]` source followed
by a contiguous 2-D slice is a candidate fallback but is not certified here.

## Certification scope

Job `1793825465` (probe `notes/pypto-operand-tile-construction.py`) ran eight
contracts; every contract line passed at round 1, with device synchronisation
before each host-side float64 comparison.

- `pypto.view` on a rank-3 source with middle extent 1 and full last-axis
  extent: element-exact (`max_abs = 0`) at fp16 `[512, 8, 128]` (N=64, g=3,
  r0=128), bf16 `[256, 4, 256]` (N=32, g=2, r0=64), and fp32 `[128, 16, 64]`
  (N=16, g=5, r0=8).
- `pypto.reshape` from `[N, 1, D]` to `[1, N, D]`: element-exact
  (`max_abs = 0`) at the fp16 geometry above.
- `pypto.matmul` with the constructed operand: fp16 `b_trans=True` at
  D=N=128 (`rel_l2 = 5.919e-08`), fp16 `b_trans=False` at K=N=D=128
  (`rel_l2 = 5.919e-08`), and bf16 `b_trans=True` at D=N=128
  (`rel_l2 = 5.590e-08`), each within its stated tolerance against a float64
  host-side computation.
- Negative control (same job): `view(x, [1, N, D], [r0, g, 0])` reads a
  different window, `max_abs = 1.105e+01` from the intended box.
- Not covered: N != D cube operands, D > 256, partial last-axis boxes, 4-D
  sources, fused multi-consumer chains, and any performance claim.

## Sources

- Public API symbols: `pypto.view`, `pypto.reshape`, `pypto.matmul`,
  `pypto.assemble`, `pypto.set_vec_tile_shapes`, `pypto.set_cube_tile_shapes`
  (local PyPTO DSL source tree).
- Probe verdicts: job `1793825465` (this entry: C1-C8, every contract PASS);
  job `1793823806` (per-axis view exactness, reshape-of-view, 4-D compile
  rejects, negative control); job `1793824430` (cube path with the reshaped
  token-first operand).
- Detailed provenance archive:
  `docs/kb-provenance/pypto/pypto-operand-tile-construction.md`
