# GQA — PyPTO on Ascend, CANN-Bench tasks/level4/gqa
Frozen public cases are judged by official golden.py, not the library reference.
`kernel.py` is student-owned and starts from a signature stub. No vendor operator
call is permitted as its implementation. The separate strong library tree is sealed
in blackbox mode. Public local scoring is not the paper's 910C / CANN 1.1.2 hosted result.
The library uses kernel_details kernel-only time. PyPTO records both that and trace_view
AI-Core E2E. Reference comparisons are cross-job median-vs-median, not paired timing.
