# Hardware knowledge-base recipe template

Every recipe describes one operator-agnostic hardware/API mechanism on one
architecture. Source code may appear here only as a minimal independent example.
No operator algorithms, reference implementations, oracle symbols or copied
experiment narratives. Use the same section titles as this template.

# <mechanism>

- Architecture: <target hardware and software range>
- Status: Verified only after a completed NPU probe; otherwise not admissible
- Certification evidence: <job, probe path, exact verdict and scope>

## Purpose
What hardware/API behavior the entry establishes, without operator-specific claims.

## When needed
The observable trigger symptom. Include inapplicable cases and boundaries.

## Minimal runnable example
An independent synthetic-input example tested by the certification probe.

## Gotchas
Limits, negative controls and negative results. No invented counters or speedups.

## Retry preconditions
What must change before retrying a failed/inapplicable mechanism.

## Sources
Public documentation/API symbols and the probe verdict/job only. Narrative and
reference-specific material belongs in docs/kb-provenance/pypto/, never here.
