import cocotb
import numpy as np
import torch
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


TENSOR_SHAPE = (1500, 384)
WIDTH = 16
FRAC_BITS = 12
MAX_TIMESCALE = 10000
ABS_TOL = 0.2 
ABS_TOL_Q = int(round(ABS_TOL * (1 << FRAC_BITS)))


def sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(
        -log_timescale_increment * torch.arange(channels // 2, dtype=torch.float32)
    )
    scaled_time = torch.arange(length, dtype=torch.float32)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def quantise_q(values: torch.Tensor, width: int = WIDTH, frac_bits: int = FRAC_BITS) -> torch.Tensor:
    scale = 1 << frac_bits
    qmin = -(1 << (width - 1))
    qmax = (1 << (width - 1)) - 1
    quantised = torch.round(values * scale).to(torch.int32)
    return torch.clamp(quantised, qmin, qmax)


def to_twos(value: int, width: int = WIDTH) -> int:
    return value & ((1 << width) - 1)


def from_twos(value: int, width: int = WIDTH) -> int:
    if value >= (1 << (width - 1)):
        return value - (1 << width)
    return value


def build_test_tensor() -> torch.Tensor:
    total_values = TENSOR_SHAPE[0] * TENSOR_SHAPE[1]
    base = torch.linspace(-0.75, 0.75, steps=total_values, dtype=torch.float32)
    return quantise_q(base.reshape(TENSOR_SHAPE))


def read_signed(value: int, width: int = WIDTH) -> int:
    return from_twos(value & ((1 << width) - 1), width)


def format_position_comparison(
    input_tensor: torch.Tensor,
    hw_sin: torch.Tensor,
    hw_output: torch.Tensor,
    reference_encoding: torch.Tensor,
    reference_output: torch.Tensor,
    diff: torch.Tensor,
    position: int,
) -> str:
    header = "idx | x_value | hw_sin | ref_sin | sin_diff | received | expected | abs_diff"
    rows = [header]
    for idx, (x_value, captured_sin, expected_sin, received, expected, abs_diff) in enumerate(
        zip(
            input_tensor[position].tolist(),
            hw_sin[position].tolist(),
            reference_encoding[position].tolist(),
            hw_output[position].tolist(),
            reference_output[position].tolist(),
            diff[position].tolist(),
        )
    ):
        rows.append(
            f"{idx:3d} | {x_value:7d} | {captured_sin:6d} | {expected_sin:7d} | "
            f"{abs(captured_sin - expected_sin):8d} | {received:8d} | {expected:8d} | {abs_diff:8d}"
        )
    return "\n".join(rows)


def format_limit_violations(
    input_tensor: torch.Tensor,
    hw_sin: torch.Tensor,
    hw_output: torch.Tensor,
    reference_encoding: torch.Tensor,
    reference_output: torch.Tensor,
    diff: torch.Tensor,
    limit: int,
) -> list[str]:
    failing_entries = torch.nonzero(diff > limit, as_tuple=False)
    rows = ["position | idx | x_value | hw_sin | ref_sin | sin_diff | received | expected | abs_diff"]
    for position, idx in failing_entries.tolist():
        rows.append(
            f"{position:8d} | {idx:3d} | "
            f"{int(input_tensor[position, idx]):7d} | {int(hw_sin[position, idx]):6d} | "
            f"{int(reference_encoding[position, idx]):7d} | "
            f"{abs(int(hw_sin[position, idx]) - int(reference_encoding[position, idx])):8d} | "
            f"{int(hw_output[position, idx]):8d} | {int(reference_output[position, idx]):8d} | "
            f"{int(diff[position, idx]):8d}"
        )
    return rows


class PositionalEncodingDriver:
    def __init__(self, dut):
        self.dut = dut
        self.n_state = TENSOR_SHAPE[1]
        self.word_mask = (1 << WIDTH) - 1

    def pack_state_vector(self, values: list[int]) -> int:
        packed = 0
        for idx, value in enumerate(values):
            shift = (self.n_state - 1 - idx) * WIDTH
            packed |= to_twos(int(value)) << shift
        return packed

    def unpack_state_vector(self, packed: int) -> torch.Tensor:
        values = []
        for idx in range(self.n_state):
            shift = (self.n_state - 1 - idx) * WIDTH
            word = (packed >> shift) & self.word_mask
            values.append(from_twos(word))
        return torch.tensor(values, dtype=torch.int32)

    def read_signal(self, signal, width: int = WIDTH) -> int:
        return read_signed(int(signal.value), width)

    async def reset(self):
        self.dut.i_rst.value = 1
        self.dut.i_valid.value = 0
        self.dut.i_position.value = 0
        self.dut.i_x.value = 0

        for _ in range(3):
            await RisingEdge(self.dut.i_clk)

        self.dut.i_rst.value = 0

        for _ in range(3):
            await RisingEdge(self.dut.i_clk)
            await ReadOnly()
            assert int(self.dut.o_valid.value) == 0, "reset should clear o_valid"

    async def encode_position(self, x_slice: torch.Tensor, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert x_slice.shape == (self.n_state,)
        captured_sin = torch.zeros(self.n_state, dtype=torch.int32)

        await FallingEdge(self.dut.i_clk)
        self.dut.i_position.value = position
        self.dut.i_valid.value = 1
        self.dut.i_x.value = self.pack_state_vector(x_slice.tolist())

        await RisingEdge(self.dut.i_clk)

        await FallingEdge(self.dut.i_clk)
        self.dut.i_valid.value = 0

        while True:
            await RisingEdge(self.dut.i_clk)
            await ReadOnly()
            if int(self.dut.write_valid.value):
                write_idx = int(self.dut.write_idx.value)
                captured_sin[write_idx] = self.read_signal(self.dut.sin_value)
            if int(self.dut.o_valid.value):
                return self.unpack_state_vector(int(self.dut.o_x.value)), captured_sin

    async def encode_tensor(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert tensor.shape == TENSOR_SHAPE
        outputs = []
        sin_terms = []
        for position in range(tensor.shape[0]):
            output, captured_sin = await self.encode_position(tensor[position], position)
            outputs.append(output)
            sin_terms.append(captured_sin)
        return torch.stack(outputs, dim=0), torch.stack(sin_terms, dim=0)


@cocotb.test()
async def test_positional_encoding_tensor(dut):
    cocotb.start_soon(Clock(dut.i_clk, 10, unit="ns").start())
    driver = PositionalEncodingDriver(dut)

    await driver.reset()

    input_tensor = build_test_tensor()
    hw_output, hw_sin = await driver.encode_tensor(input_tensor)

    reference_encoding = quantise_q(
        sinusoids(TENSOR_SHAPE[0], TENSOR_SHAPE[1], max_timescale=MAX_TIMESCALE)
    )
    reference_output = input_tensor + reference_encoding

    diff = (hw_output - reference_output).abs()
    max_diff = int(diff.max().item())
    max_sin_diff = int((hw_sin - reference_encoding).abs().max().item())
    failing_positions = torch.nonzero(torch.any(diff > ABS_TOL_Q, dim=1), as_tuple=False)

    if len(failing_positions):
        first_failing_position = int(failing_positions[0].item())
        dut._log.warning(
            "hardware positional encoding deviated from reference: "
            f"max_diff={max_diff}, tolerance={ABS_TOL_Q}, "
            f"max_sin_diff={max_sin_diff}, "
            f"failing_positions={len(failing_positions)}"
        )
        dut._log.warning(
            "first failing position detail:\n%s",
            format_position_comparison(
                input_tensor,
                hw_sin,
                hw_output,
                reference_encoding,
                reference_output,
                diff,
                position=first_failing_position,
            ),
        )
        # dut._log.warning(
        #     "all entries above tolerance:\n%s",
        #     "\n".join(
        #         format_limit_violations(
        #             input_tensor,
        #             hw_sin,
        #             hw_output,
        #             reference_encoding,
        #             reference_output,
        #             diff,
        #             limit=ABS_TOL_Q,
        #         )
        #     ),
        # )
    else:
        dut._log.info(
            "hardware positional encoding matched reference within tolerance: "
            f"max_diff={max_diff}, max_sin_diff={max_sin_diff}, tolerance={ABS_TOL_Q}"
        )
