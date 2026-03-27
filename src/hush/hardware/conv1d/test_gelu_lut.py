import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
whisper_path = project_root / "whisper"
sys.path.insert(0, str(whisper_path))

import cocotb
import torch
import torch.nn.functional as F
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly

SCALE_FACTOR = 100.0

def quantise_to_int32(tensor, scale=SCALE_FACTOR):
    quantised = (tensor * scale * scale).clamp(-2147483648, 2147483647).round().int()
    return quantised

def dequantise_from_int32(tensor, scale=SCALE_FACTOR):
    return tensor.float() / (scale * scale)

@cocotb.test()
async def test_gelu_lut_wrapper(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    dut.rst_n.value = 0
    dut.data_in.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1

    for _ in range(5):
        await RisingEdge(dut.clk)

    dut.rst_n.value = 1

    for _ in range(5):
        await RisingEdge(dut.clk)

    print("\n" + "="*80)
    print("Testing GELU LUT Wrapper with various inputs")
    print("="*80)

    test_vals = [-2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5, 2.0]
    max_diff = 0.0
    mean_diff = 0.0

    for val in test_vals:

        val_q = quantise_to_int32(torch.tensor([val]))
        val_int = int(val_q.item())

        await FallingEdge(dut.clk)
        dut.data_in.value = val_int
        dut.data_in_valid.value = 1
        await RisingEdge(dut.clk)

        await FallingEdge(dut.clk)
        dut.data_in_valid.value = 0

        cycles = 0
        while int(dut.data_out_valid.value) == 0 and cycles < 20:
            await RisingEdge(dut.clk)
            cycles += 1
            await ReadOnly()

        if int(dut.data_out_valid.value):
            hw_out_int = int(dut.data_out.value)

            if hw_out_int & (1 << 31):
                hw_out_int -= (1 << 32)

            hw_out = dequantise_from_int32(torch.tensor([hw_out_int]))
            sw_out = F.gelu(torch.tensor([val]))

            diff = abs(hw_out.item() - sw_out.item())
            max_diff = max(max_diff, diff)
            mean_diff += diff

            status = "✓" if diff < 0.1 else "✗"
            print(f"{status} Input: {val:6.2f} (q={val_int:8d})  |  HW GELU: {hw_out.item():8.4f} (q={hw_out_int:8d})  |  SW GELU: {sw_out.item():8.4f}  |  Diff: {diff:7.4f}")
        else:
            print(f"ERROR: No valid output for input {val}")

        await RisingEdge(dut.clk)

    mean_diff /= len(test_vals)

    print("="*80)
    print(f"Max diff: {max_diff:.4f}")
    print(f"Mean diff: {mean_diff:.4f}")
    print("="*80)

    assert max_diff < 0.1, f"Max diff too large: {max_diff}"
    assert mean_diff < 0.05, f"Mean diff too large: {mean_diff}"

    print("GELU LUT test completed successfully!")
