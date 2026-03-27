import cocotb
import torch
import torch.nn.functional as F
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly, ClockCycles

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
whisper_path = project_root / "whisper"
sys.path.insert(0, str(whisper_path))

from whisper.quantise import LinearInteger

SCALE = 100.0
DATA_WIDTH = 16
ACC_WIDTH = 2 * DATA_WIDTH

# Helpers


def q_input(t):
    return (t * SCALE).clamp(-32768, 32767).round().int()


def q_weight(t):
    return (t * SCALE).clamp(-32768, 32767).round().int()


def q_bias(t):
    return (t * SCALE * SCALE).clamp(-(2**31), 2**31 - 1).round().int()


def deq(t):
    return t.float() / (SCALE * SCALE)


def sign_ext(val, bits=ACC_WIDTH):
    if val & (1 << (bits - 1)):
        val -= 1 << bits
    return val


async def init(dut, in_ch, out_ch, seq_len):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.weight_valid.value = 0
    dut.bias_valid.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    dut.in_channels_cfg.value = in_ch
    dut.out_channels_cfg.value = out_ch
    dut.input_length.value = seq_len
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 3)


async def start(dut):
    await FallingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    while not int(dut.busy.value):
        await RisingEdge(dut.clk)


async def load_weights(dut, w_q):
    """w_q: [out_ch, in_ch] int tensor"""
    out_ch, in_ch = w_q.shape
    for oc in range(out_ch):
        for ic in range(in_ch):
            await FallingEdge(dut.clk)
            dut.weight_in.value = int(w_q[oc, ic].item())
            dut.weight_valid.value = 1
            dut.weight_out_ch.value = oc
            dut.weight_in_ch.value = ic
            await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.weight_valid.value = 0
    await RisingEdge(dut.clk)


async def load_biases(dut, b_q):
    """b_q: [out_ch] int tensor"""
    for oc in range(b_q.shape[0]):
        await FallingEdge(dut.clk)
        dut.bias_in.value = int(b_q[oc].item())
        dut.bias_valid.value = 1
        dut.bias_out_ch.value = oc
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.bias_valid.value = 0
    await RisingEdge(dut.clk)


async def load_input(dut, x_q):
    """x_q: [1, in_ch, seq_len] int tensor"""
    _, in_ch, seq_len = x_q.shape
    for ch in range(in_ch):
        for pos in range(seq_len):
            await FallingEdge(dut.clk)
            dut.data_in.value = int(x_q[0, ch, pos].item())
            dut.data_in_valid.value = 1
            dut.in_channel_idx.value = ch
            dut.in_pos_idx.value = pos
            await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.data_in_valid.value = 0
    await RisingEdge(dut.clk)


async def read_output(dut, out_ch, seq_len, timeout=500000):
    """Returns [1, out_ch, seq_len] int tensor"""
    out = torch.zeros(1, out_ch, seq_len, dtype=torch.int64)
    total = out_ch * seq_len
    count = 0
    cycles = 0
    while count < total and cycles < timeout:
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.data_out_valid.value):
            ch = int(dut.out_channel_idx.value)
            pos = int(dut.out_pos_idx.value)
            val = sign_ext(int(dut.data_out.value))
            out[0, ch, pos] = val
            count += 1
        cycles += 1
    assert count == total, f"only got {count}/{total} outputs"
    return out


def sw_linear(x, w, b):
    """Reference: quantise, integer matmul, return int and float"""
    x_q = q_input(x)
    w_q = q_weight(w)
    b_q = q_bias(b) if b is not None else None
    # F.linear on the quantised values (simulates integer MAC)
    out_q = F.linear(x_q.float(), w_q.float(), b_q.float() if b_q is not None else None)
    return out_q.round().long(), deq(out_q)


def check(hw, sw, label, atol=1):
    # atol so we can handle slight delta due to rounding etc
    diff = (hw - sw).abs()
    max_diff = diff.max().item()
    assert max_diff <= atol, f"FAIL {label}: max_diff={max_diff} > {atol}"
    cocotb.log.info(f"PASS {label}: max_diff={max_diff}")


# Tests


@cocotb.test()
async def test_small_linear(dut):
    """4x4 linear, 2 timesteps"""
    in_ch, out_ch, seq_len = 4, 4, 2
    await init(dut, in_ch, out_ch, seq_len)

    torch.manual_seed(42)
    w = torch.randn(out_ch, in_ch)
    b = torch.randn(out_ch)
    x = torch.randn(1, seq_len, in_ch)  # [batch, seq, features]

    # x needs to be [1, in_ch, seq_len] for the hardware (channel-first)
    x_hw = x.permute(0, 2, 1)

    ref_q, ref_float = sw_linear(x, w, b)

    await start(dut)
    await load_weights(dut, q_weight(w))
    await load_biases(dut, q_bias(b))
    await load_input(dut, q_input(x_hw))
    hw_out = await read_output(dut, out_ch, seq_len)

    # hw_out is [1, out_ch, seq_len], ref_q is [1, seq_len, out_ch]
    hw_compare = hw_out.permute(0, 2, 1)  # -> [1, seq_len, out_ch]
    check(hw_compare, ref_q, "small 4x4")


@cocotb.test()
async def test_no_bias(dut):
    """Linear with bias=False (key projection case)"""
    in_ch, out_ch, seq_len = 4, 4, 2
    await init(dut, in_ch, out_ch, seq_len)

    torch.manual_seed(7)
    w = torch.randn(out_ch, in_ch)
    b = torch.zeros(out_ch)  # zero bias = no bias
    x = torch.randn(1, seq_len, in_ch)
    x_hw = x.permute(0, 2, 1)

    ref_q, _ = sw_linear(x, w, b)

    await start(dut)
    await load_weights(dut, q_weight(w))
    await load_biases(dut, q_bias(b))
    await load_input(dut, q_input(x_hw))
    hw_out = await read_output(dut, out_ch, seq_len)

    hw_compare = hw_out.permute(0, 2, 1)
    check(hw_compare, ref_q, "no bias")


@cocotb.test()
async def test_single_timestep(dut):
    """Single position"""
    in_ch, out_ch, seq_len = 4, 4, 1
    await init(dut, in_ch, out_ch, seq_len)

    torch.manual_seed(99)
    w = torch.randn(out_ch, in_ch)
    b = torch.randn(out_ch)
    x = torch.randn(1, 1, in_ch)
    x_hw = x.permute(0, 2, 1)

    ref_q, _ = sw_linear(x, w, b)

    await start(dut)
    await load_weights(dut, q_weight(w))
    await load_biases(dut, q_bias(b))
    await load_input(dut, q_input(x_hw))
    hw_out = await read_output(dut, out_ch, seq_len)

    hw_compare = hw_out.permute(0, 2, 1)
    check(hw_compare, ref_q, "single timestep")


@cocotb.test()
async def test_whisper_dims(dut):
    """384x384 linear, 3 timesteps, same as Whisper dims"""
    in_ch, out_ch, seq_len = 384, 384, 3
    await init(dut, in_ch, out_ch, seq_len)

    torch.manual_seed(0)
    w = torch.randn(out_ch, in_ch) * 0.05  # Whisper-scale weights
    b = torch.randn(out_ch) * 0.01
    x = torch.randn(1, seq_len, in_ch)
    x_hw = x.permute(0, 2, 1)

    ref_q, _ = sw_linear(x, w, b)

    await start(dut)
    await load_weights(dut, q_weight(w))
    await load_biases(dut, q_bias(b))
    await load_input(dut, q_input(x_hw))
    hw_out = await read_output(dut, out_ch, seq_len)

    hw_compare = hw_out.permute(0, 2, 1)
    check(hw_compare, ref_q, "whisper 384x384")


@cocotb.test()
async def test_matches_linearinteger(dut):
    """RTL output matches the actual LinearInteger class used in Whisper"""
    in_ch, out_ch, seq_len = 8, 8, 2
    await init(dut, in_ch, out_ch, seq_len)

    torch.manual_seed(55)
    config = {
        "data_in_width": 16,
        "data_in_frac_width": 8,
        "weight_width": 16,
        "weight_frac_width": 8,
        "bias_width": 16,
        "bias_frac_width": 8,
    }

    layer = LinearInteger(in_ch, out_ch, bias=True, config=config)
    x = torch.randn(1, seq_len, in_ch)

    with torch.no_grad():
        py_out = layer(x)

    # Use LinearInteger's own quantizers
    x_q = layer.x_quantizer(x)
    w_q = layer.w_quantizer(layer.weight.data)
    b_q = layer.b_quantizer(layer.bias.data)

    # Convert to integers for the RTL (by multiplying by scale = 2^frac_width)
    frac = config["data_in_frac_width"]
    scale = 2**frac
    x_int = (x_q * scale).round().int()
    w_int = (w_q * scale).round().int()
    b_int = (b_q * scale * scale).round().int()  # bias in accumulator domain

    x_hw = x_int.permute(0, 2, 1)  # [1, features, seq_len] for hardware

    await start(dut)
    await load_weights(dut, w_int)
    await load_biases(dut, b_int)
    await load_input(dut, x_hw)
    hw_out = await read_output(dut, out_ch, seq_len)

    # Dequantise from accumulator domain (scale^2)
    hw_float = hw_out.permute(0, 2, 1).float() / (scale * scale)

    diff = (hw_float - py_out).abs()
    max_diff = diff.max().item()
    cocotb.log.info(f"LinearInteger vs RTL max_diff: {max_diff}")
    assert max_diff < 0.001, f"fail: max_diff={max_diff} :("
