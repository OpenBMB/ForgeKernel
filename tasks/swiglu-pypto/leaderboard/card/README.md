# SwiGlu — PyPTO / CANN-Bench

The official 20 public cases are the only judge. `kernel.py` is the student-owned candidate;
`build.sh` packages it as `cann_bench.swi_glu`. `eval/cann_bench.py` converts the official
`trace_view` report into ForgeKernel's correctness + geomean eval JSON. The reference runs in
the separate `_<op>ref` tree and is never an in-job comparator.
