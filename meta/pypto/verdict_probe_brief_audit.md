## Optional NPU contract probe
Up to {MAX_VERDICT_PROBE_ROUNDS} driver-run probes may resolve uncertain M0
contracts. Write `notes/<name>.py` with a falsifiable criterion and print
`PROBE PASS/FAIL max_diff=<finite error>` (or `n/a` for compile/launch only).
End the turn with `PROBE_REQUEST: notes/<name>.py`. No NCU/CUDA tooling.
A structural reopen requires a measured base/variant price, not an inference
from a source layout or a coincidental timing improvement. The seeded
`notes/ascend_probe.py` prices one exact-edit variant; adding `"cases": "scout"`
to its spec narrows that price to the card's representative cases (never a gate
reading, `PROBE SCOUT` instead of `PROBE PASS`), and a full-case run is still
what certifies anything.
