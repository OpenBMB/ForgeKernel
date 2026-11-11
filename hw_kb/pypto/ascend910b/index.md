# PyPTO on Ascend 910B

This is the exact knowledge-base tree for Ascend 910B. It starts empty on purpose:
no recipe in another architecture's tree is evidence about this one.

## Local interfaces

- Use the task's frozen PyPTO/PTO SDK and the official CANN-Bench adapter.
- Start from `../TEMPLATE.md` when adding a recipe.
- Confirm the installed PyPTO version and platform configuration before relying on
  API details from a different Ascend variant.

## Measurement

Use the card's declared `timing_basis`; do not mix `cann_trace_view`,
`cann_kernel_details` or msprof task rows. A probe must run on the target
hardware before a recipe is marked Verified.

## Current gaps

No architecture-specific recipe has been certified on Ascend 910B in this branch.
Do not copy a 950PR certification into this tree merely because the API looks
similar.
