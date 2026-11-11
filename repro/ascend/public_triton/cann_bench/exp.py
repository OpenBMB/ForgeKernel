"""Triton-Ascend implementation of the CANN Bench Exp interface."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_BLOCK_SIZE = 4096


@triton.jit
def _exp_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    scale,
    shift,
    log_base,
    HAS_BASE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    value = value * scale + shift
    if HAS_BASE:
        value = value * log_base
    tl.store(output_ptr + offsets, tl.exp(value), mask=mask)


def exp(
    x: torch.Tensor,
    base: float = -1.0,
    scale: float = 1.0,
    shift: float = 0.0,
) -> torch.Tensor:
    """Return ``exp((x * scale + shift) * log(base))`` on an NPU tensor.

    ``base <= 0`` selects the natural-base form used by the task schema. The
    wrapper only prepares metadata and output storage; the numerical work runs
    in ``_exp_kernel``.
    """
    if not x.is_contiguous():
        x = x.contiguous()

    output = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    _exp_kernel[grid](
        x,
        output,
        n_elements,
        scale,
        shift,
        math.log(base) if base > 0 else 0.0,
        HAS_BASE=base > 0,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output
