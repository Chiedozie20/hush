import cocotb
import torch

import sys
from pathlib import Path

from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


project_root = Path(__file__).parent.parent.parent.parent.parent
whisper_path = project_root / "whisper" / "whisper"
sys.path.insert(0, str(whisper_path))

from quantise import LayerNormInteger


FRAME_SIZE = 32
FRAC_WIDTH = 17
SCALE = 1 << FRAC_WIDTH
TEST_TOKENS = 3
ATOL = 1e-3
RTOL = 1e-3
RANDOM_SWEEP_SEEDS = (0, 3, 7, 11)


def quantise_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor * SCALE).to(torch.int32)


def dequantise_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float32) / SCALE


def build_random_inputs(seed: int = 7, tokens: int = TEST_TOKENS):
    torch.manual_seed(seed)
    return torch.randn(tokens, FRAME_SIZE, dtype=torch.float32) * 1.5 + 0.2


def layernorm_reference(x: torch.Tensor):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + 1e-5)


class LayerNormDriver:
    def __init__(self, dut):
        self.dut = dut

    async def reset(self):
        self.dut.rst_n.value = 0
        self.dut.data_in.value = 0
        self.dut.data_in_valid.value = 0

        for _ in range(4):
            await RisingEdge(self.dut.clk)

        self.dut.rst_n.value = 1
        await RisingEdge(self.dut.clk)

    async def wait_until_ready(self):
        while True:
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            if int(self.dut.data_in_ready.value):
                return

    async def send_frame(self, frame_q: torch.Tensor):
        await self.wait_until_ready()

        for value in frame_q.tolist():
            await FallingEdge(self.dut.clk)
            self.dut.data_in.value = int(value)
            self.dut.data_in_valid.value = 1
            await RisingEdge(self.dut.clk)

        await FallingEdge(self.dut.clk)
        self.dut.data_in.value = 0
        self.dut.data_in_valid.value = 0

    async def recv_frame(self) -> torch.Tensor:
        received = []
        idle_cycles = 0

        while len(received) < FRAME_SIZE:
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            if int(self.dut.data_out_valid.value):
                received.append(int(self.dut.data_out.value.to_signed()))
                idle_cycles = 0
            else:
                idle_cycles += 1
                if idle_cycles > (FRAME_SIZE * 6):
                    cocotb.log.error(
                        "timeout state=%s sample_count=%s output_idx=%s ready=%s valid=%s",
                        self.dut.state.value,
                        self.dut.sample_count.value,
                        self.dut.output_idx.value,
                        self.dut.data_in_ready.value,
                        self.dut.data_out_valid.value,
                    )
                    raise TimeoutError("timed out waiting for layernorm output")

        return torch.tensor(received, dtype=torch.int32)

    async def run_tensor(self, x_q: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token_idx in range(x_q.shape[0]):
            cocotb.log.info(f"Sending token {token_idx}")
            await self.send_frame(x_q[token_idx])
            cocotb.log.info(f"Receiving token {token_idx}")
            outputs.append(await self.recv_frame())
        return torch.stack(outputs, dim=0)


async def run_case(dut, x: torch.Tensor, label: str):
    driver = LayerNormDriver(dut)
    await driver.reset()

    x_q = quantise_tensor(x)
    y_q = await driver.run_tensor(x_q)
    await ClockCycles(dut.clk, 2)

    y_hw = dequantise_tensor(y_q)
    y_ref = layernorm_reference(dequantise_tensor(x_q))

    max_diff = torch.max(torch.abs(y_hw - y_ref)).item()
    cocotb.log.info(f"{label} max abs diff: {max_diff:.6f}")

    assert y_hw.shape == y_ref.shape
    assert torch.allclose(y_hw, y_ref, atol=ATOL, rtol=RTOL), (
        f"{label} mismatch: max abs diff {max_diff:.6f}"
    )


@cocotb.test()
async def test_layernorm_random_tokens(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    x = build_random_inputs(seed=7, tokens=TEST_TOKENS)
    await run_case(dut, x, "random_tokens")


@cocotb.test()
async def test_layernorm_near_constant_inputs(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    base = torch.full((2, FRAME_SIZE), 0.375, dtype=torch.float32)
    ramp = torch.linspace(-2e-3, 2e-3, FRAME_SIZE, dtype=torch.float32)
    x = torch.stack((base[0] + ramp, base[1] - ramp), dim=0)
    await run_case(dut, x, "near_constant_inputs")


@cocotb.test()
async def test_layernorm_exact_constant_inputs(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    x = torch.full((2, FRAME_SIZE), -0.625, dtype=torch.float32)
    await run_case(dut, x, "exact_constant_inputs")


@cocotb.test()
async def test_layernorm_matches_reference_on_shifted_input(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    x = build_random_inputs(seed=19, tokens=2) + 3.5
    await run_case(dut, x, "shifted_input")


@cocotb.test()
async def test_layernorm_random_sweep(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    x = torch.cat(
        [build_random_inputs(seed=seed, tokens=2) for seed in RANDOM_SWEEP_SEEDS],
        dim=0,
    )
    await run_case(dut, x, "random_sweep")


@cocotb.test()
async def test_layernorm_back_to_back_runs(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    driver = LayerNormDriver(dut)
    await driver.reset()

    x1 = build_random_inputs(seed=31, tokens=1)
    x2 = build_random_inputs(seed=37, tokens=1) + 1.25

    x1_q = quantise_tensor(x1)
    y1_q = await driver.run_tensor(x1_q)

    x2_q = quantise_tensor(x2)
    y2_q = await driver.run_tensor(x2_q)
    await ClockCycles(dut.clk, 2)

    y1_hw = dequantise_tensor(y1_q)
    y1_ref = layernorm_reference(dequantise_tensor(x1_q))
    y2_hw = dequantise_tensor(y2_q)
    y2_ref = layernorm_reference(dequantise_tensor(x2_q))

    max_diff_1 = torch.max(torch.abs(y1_hw - y1_ref)).item()
    max_diff_2 = torch.max(torch.abs(y2_hw - y2_ref)).item()
    cocotb.log.info(f"back_to_back first max abs diff: {max_diff_1:.6f}")
    cocotb.log.info(f"back_to_back second max abs diff: {max_diff_2:.6f}")

    assert torch.allclose(y1_hw, y1_ref, atol=ATOL, rtol=RTOL), (
        f"back_to_back first mismatch: max abs diff {max_diff_1:.6f}"
    )
    assert torch.allclose(y2_hw, y2_ref, atol=ATOL, rtol=RTOL), (
        f"back_to_back second mismatch: max abs diff {max_diff_2:.6f}"
    )


@cocotb.test()
async def test_matches_layernorminteger(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    driver = LayerNormDriver(dut)
    await driver.reset()

    x = build_random_inputs(seed=53, tokens=2)
    x_q = quantise_tensor(x)

    config = {
        "data_in_width": 28,
        "data_in_frac_width": FRAC_WIDTH,
        "weight_width": 28,
        "weight_frac_width": FRAC_WIDTH,
        "bias_width": 28,
        "bias_frac_width": FRAC_WIDTH,
    }

    layer = LayerNormInteger(FRAME_SIZE, elementwise_affine=False, config=config)
    sw_out = layer(x).detach()

    y_q = await driver.run_tensor(x_q)
    await ClockCycles(dut.clk, 2)

    y_hw = dequantise_tensor(y_q)
    max_diff = torch.max(torch.abs(y_hw - sw_out)).item()
    cocotb.log.info(f"matches_layernorminteger max abs diff: {max_diff:.6f}")

    assert torch.allclose(y_hw, sw_out, atol=ATOL, rtol=RTOL), (
        f"matches_layernorminteger mismatch: max abs diff {max_diff:.6f}"
    )
