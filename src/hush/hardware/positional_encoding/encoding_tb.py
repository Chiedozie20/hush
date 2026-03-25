import cocotb
import numpy as np
import torch
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


TENSOR_SHAPE = (5, 384)
WIDTH = 16
FRAC_BITS = 12
MAX_TIMESCALE = 10000
ABS_TOL = 32


def sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(
        -log_timescale_increment * torch.arange(channels // 2, dtype=torch.float32)
    )
    scaled_time = torch.arange(length, dtype=torch.float32)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def quantize_q4_12(values: torch.Tensor, width: int = WIDTH, frac_bits: int = FRAC_BITS) -> torch.Tensor:
    scale = 1 << frac_bits
    qmin = -(1 << (width - 1))
    qmax = (1 << (width - 1)) - 1
    quantized = torch.round(values * scale).to(torch.int32)
    return torch.clamp(quantized, qmin, qmax)


def to_twos(value: int, width: int = WIDTH) -> int:
    return value & ((1 << width) - 1)


def from_twos(value: int, width: int = WIDTH) -> int:
    if value >= (1 << (width - 1)):
        return value - (1 << width)
    return value


def build_test_tensor() -> torch.Tensor:
    total_values = TENSOR_SHAPE[0] * TENSOR_SHAPE[1]
    base = torch.linspace(-0.75, 0.75, steps=total_values, dtype=torch.float32)
    return quantize_q4_12(base.reshape(TENSOR_SHAPE))


def format_position_comparison(
    hw_output: torch.Tensor,
    reference_output: torch.Tensor,
    diff: torch.Tensor,
    position: int,
) -> str:
    header = "idx | received | expected | abs_diff"
    rows = [header]
    for idx, (received, expected, abs_diff) in enumerate(
        zip(hw_output[position].tolist(), reference_output[position].tolist(), diff[position].tolist())
    ):
        rows.append(f"{idx:3d} | {received:8d} | {expected:8d} | {abs_diff:8d}")
    return "\n".join(rows)


class PositionalEncodingDriver:
    def __init__(self, dut):
        self.dut = dut
        self.n_state = TENSOR_SHAPE[1]

    async def reset(self):
        self.dut.i_rst.value = 1
        self.dut.i_valid.value = 0
        self.dut.i_position.value = 0

        for idx in range(self.n_state):
            self.dut.i_x[idx].value = 0

        for _ in range(3):
            await RisingEdge(self.dut.i_clk)

        self.dut.i_rst.value = 0

        for _ in range(3):
            await RisingEdge(self.dut.i_clk)
            await ReadOnly()
            assert int(self.dut.o_valid.value) == 0, "reset should clear o_valid"

    async def encode_position(self, x_slice: torch.Tensor, position: int) -> torch.Tensor:
        assert x_slice.shape == (self.n_state,)

        await FallingEdge(self.dut.i_clk)
        self.dut.i_position.value = position
        self.dut.i_valid.value = 1
        for idx, value in enumerate(x_slice.tolist()):
            self.dut.i_x[idx].value = to_twos(int(value))

        await RisingEdge(self.dut.i_clk)

        await FallingEdge(self.dut.i_clk)
        self.dut.i_valid.value = 0

        while True:
            await RisingEdge(self.dut.i_clk)
            await ReadOnly()
            if int(self.dut.o_valid.value):
                values = [
                    from_twos(int(self.dut.o_x[idx].value))
                    for idx in range(self.n_state)
                ]
                return torch.tensor(values, dtype=torch.int32)

    async def encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape == TENSOR_SHAPE
        outputs = []
        for position in range(tensor.shape[0]):
            outputs.append(await self.encode_position(tensor[position], position))
        return torch.stack(outputs, dim=0)


@cocotb.test()
async def test_positional_encoding_tensor(dut):
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    driver = PositionalEncodingDriver(dut)

    await driver.reset()

    input_tensor = build_test_tensor()
    hw_output = await driver.encode_tensor(input_tensor)

    reference_encoding = quantize_q4_12(
        sinusoids(TENSOR_SHAPE[0], TENSOR_SHAPE[1], max_timescale=MAX_TIMESCALE)
    )
    reference_output = input_tensor + reference_encoding

    diff = (hw_output - reference_output).abs()
    max_diff = int(diff.max().item())
    failing_positions = torch.nonzero(torch.any(diff > ABS_TOL, dim=1), as_tuple=False)
    first_failing_position = int(failing_positions[0].item()) if len(failing_positions) else 0

    assert torch.all(diff <= ABS_TOL), (
        f"hardware positional encoding deviated from reference: max_diff={max_diff}, "
        f"tolerance={ABS_TOL}\n"
        f"first failing position={first_failing_position}\n"
        f"{format_position_comparison(hw_output, reference_output, diff, position=first_failing_position)}"
    )
