"""Independent NPU valid_shape read-padding and valid-only writeback contract.

Synthetic tensors and CPU arithmetic are independent of any operator/reference.
No performance claim. A view is materialized separately to observe read padding;
assemble writes into a sentinel tensor to observe valid-only writeback.
"""
import pypto
import torch
import torch_npu


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def tile_contract(
    x: pypto.Tensor([16, 16], pypto.DT_FP32),
    y: pypto.Tensor([16, 16], pypto.DT_FP32),
    padded: pypto.Tensor([8, 8], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8, 8)
    tile = pypto.view(x, shape=[8, 8], offsets=[4, 4], valid_shape=[4, 4])
    pypto.assemble(tile, [4, 4], y)
    pypto.assemble(tile, [0, 0], padded)


x = torch.arange(256, dtype=torch.float32, device="npu").reshape(16, 16)
y = torch.full((16, 16), -1, dtype=torch.float32, device="npu")
padded = torch.full((8, 8), -1, dtype=torch.float32, device="npu")
tile_contract(x, y, padded)
torch_npu.npu.synchronize()
actual = y.cpu().numpy()
padded_actual = padded.cpu().numpy()
expected = torch.full((16, 16), -1, dtype=torch.float32).numpy()
expected[4:8, 4:8] = x.cpu().numpy()[4:8, 4:8]
expected_padded = torch.zeros((8, 8), dtype=torch.float32).numpy()
expected_padded[:4, :4] = x.cpu().numpy()[4:8, 4:8]
read_error = float(torch.max(torch.abs(torch.from_numpy(padded_actual - expected_padded))))
write_error = float(torch.max(torch.abs(torch.from_numpy(actual - expected))))
print(f"PROBE {'PASS' if read_error == 0 else 'FAIL'} max_diff={read_error}", flush=True)
print(f"PROBE {'PASS' if write_error == 0 else 'FAIL'} max_diff={write_error}", flush=True)
if read_error or write_error:
    print('PADDED', padded_actual.tolist(), flush=True)
    print('WRITEBACK', actual.tolist(), flush=True)
error = max(read_error, write_error)
