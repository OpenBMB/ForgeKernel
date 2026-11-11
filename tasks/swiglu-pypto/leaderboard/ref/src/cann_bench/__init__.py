from cann_bench.swi_glu import swi_glu as _swi_glu_impl

def swi_glu(input, dim=-1):
    return _swi_glu_impl(input, dim=dim)
