"""NPU contract: valid_shape masks writeback to a sentinel output.

Single-output experiment isolates the one claim being certified. No claim is
made about raw view padding or multi-output explicit-padding compositions:
the no-go records for those hypotheses remain in the experiment archive.
"""
import pypto
import torch
import torch_npu


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def masked_assemble(
    x: pypto.Tensor([16, 16], pypto.DT_FP32),
    y: pypto.Tensor([16, 16], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8, 8)
    tile = pypto.view(x, shape=[8, 8], offsets=[4, 4], valid_shape=[4, 4])
    pypto.assemble(tile, [4, 4], y)


x = torch.arange(256, dtype=torch.float32, device="npu").reshape(16, 16)
y = torch.full((16, 16), -1, dtype=torch.float32, device="npu")
masked_assemble(x, y)
torch_npu.npu.synchronize()
actual = y.cpu().numpy()
expected = torch.full((16, 16), -1, dtype=torch.float32).numpy()
expected[4:8, 4:8] = x.cpu().numpy()[4:8, 4:8]
error = float(torch.max(torch.abs(torch.from_numpy(actual - expected))))
print(f"PROBE {'PASS' if error == 0 else 'FAIL'} max_diff={error}", flush=True)
