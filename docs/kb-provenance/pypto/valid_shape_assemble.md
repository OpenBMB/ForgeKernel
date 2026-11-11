# Provenance: `valid_shape_assemble`

## Hardware and probe history

- Execution: SSH `9502`, physical NPU 1 mapped to logical device 0; Ascend950PR_9579,
  CANN 9.2, PyPTO 0.3.0, Python 3.12.15.
- The initial combined expectation was rejected (`1793761261`): invalid storage
  retained sentinel values, contrary to the automatic-padding hypothesis.
- Separated verdicts (`1793761442`): raw-view zero-padding FAIL (`max_diff=1`),
  valid-only assembly PASS (`max_diff=0`). This separated a real mechanism from
  an incorrect generalization rather than weakening the numerical criterion.
- Multi-output explicit-padding composition (`1793761537`) failed its independent
  expectation (`max_diff=119`). No KB claim of a working multi-output padding
  composition is admitted. The mechanism needs a new isolated materialization probe.
- Final single-output certification (`1793761707`): `PROBE PASS max_diff=0.0`.
  The destination is sentinel-initialized. Valid cells match synthetic range
  inputs at identical coordinates; every other cell remains the sentinel.

## Reuse requirement

The recipe is operator-agnostic. It was read by the Level-4 GQA campaign after
entry, not only authored by hand. Its reuse must be visible in the campaign
milestone/student record, with the actual final implementation still judged by
CANN-Bench. Probe certification does not imply a performance gain.

Original logs are in the local `.artifacts/ascend-gap-work/tile-probe-*.log`
files and the controller job's `.jobs/` archive.
