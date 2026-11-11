# PyPTO on Ascend (generic fallback)

This tree is the fallback for an Ascend variant without its own exact KB tree.
It contains only variant-neutral interface notes. It must not contain a
hardware-specific certified recipe, and a recipe certified here is not evidence
for a particular 910B/C or 950PR/DT SKU.

## Local interfaces

- Use the installed PyPTO/PTO version's public API.
- Prefer \`pypto.view\` for per-axis slices; do not treat it as a reshape.
- Keep dtype and rank annotations consistent with the dispatcher contract.
- Host code allocates storage; numerical work stays in the JIT kernel.

## Measurement

The task card declares the timing basis. Missing profiler fields are
unavailable, not zero. Record the exact hardware variant and software versions
in a certification probe.

## Current gaps

Add an exact \`hw_kb/pypto/<variant>/\` tree before recording a recipe measured
on that variant.
