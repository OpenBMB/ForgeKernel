"""Triton-Ascend implementation of the CANN Bench SwiGLU interface."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_MAX_BLOCK_SIZE = 4096


@triton.jit
def _swi_glu_kernel(
    input_ptr,
    output_ptr,
    half_chunk_elements,
    BLOCK_SIZE: tl.constexpr,
):
    inner = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    outer = tl.program_id(1)
    mask = inner < half_chunk_elements

    # The 2D grid removes division and modulo from every element's address path.
    x0_offsets = outer * (2 * half_chunk_elements) + inner
    x1_offsets = x0_offsets + half_chunk_elements
    output_offsets = outer * half_chunk_elements + inner

    x0 = tl.load(input_ptr + x0_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(input_ptr + x1_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + output_offsets, x0 * tl.sigmoid(x0) * x1, mask=mask)


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Split ``input`` on ``dim`` and return ``silu(x0) * x1``."""
    if input.ndim == 0:
        raise ValueError("swi_glu expects an input with at least one dimension")
    dim = dim + input.ndim if dim < 0 else dim
    if dim < 0 or dim >= input.ndim:
        raise IndexError(f"dimension out of range: {dim}")
    if input.shape[dim] % 2 != 0:
        raise ValueError("the SwiGLU split dimension must have an even size")
    if not input.is_contiguous():
        input = input.contiguous()

    output_shape = list(input.shape)
    output_shape[dim] //= 2
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    n_elements = output.numel()
    if n_elements == 0:
        return output

    half_chunk_elements = output_shape[dim]
    for trailing_size in output_shape[dim + 1 :]:
        half_chunk_elements *= trailing_size

    outer_slices = n_elements // half_chunk_elements
    block_size = min(triton.next_power_of_2(half_chunk_elements), _MAX_BLOCK_SIZE)
    grid = (triton.cdiv(half_chunk_elements, block_size), outer_slices)
    _swi_glu_kernel[grid](
        input,
        output,
        half_chunk_elements,
        BLOCK_SIZE=block_size,
    )
    return output
