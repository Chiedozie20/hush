import math
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


DATA_WIDTH = 32
INT_WIDTH = 16
FRAC_WIDTH = 16
ROOT_FRAC_WIDTH = 16
INPUT_SCALE = 1 << FRAC_WIDTH
ROOT_INT_WIDTH = (INT_WIDTH + 1) // 2
OUTPUT_SCALE = 1 << ROOT_FRAC_WIDTH
NUM_TESTS = 64
# The BRAM-backed implementation is an approximation, so the tolerance is
# looser than the original iterative exact sqrt core.
ATOL = 4.0e-4


def quantise_input(value: float) -> int:
    max_value = ((1 << INT_WIDTH) - 1) + ((INPUT_SCALE - 1) / INPUT_SCALE)
    clipped = min(max(value, 0.0), max_value)
    return min(round(clipped * INPUT_SCALE), (1 << DATA_WIDTH) - 1)


def quantise_output(value: float) -> int:
    max_value = ((1 << ROOT_INT_WIDTH) - 1) + ((OUTPUT_SCALE - 1) / OUTPUT_SCALE)
    clipped = min(max(value, 0.0), max_value)
    return min(round(clipped * OUTPUT_SCALE), (1 << (ROOT_INT_WIDTH + ROOT_FRAC_WIDTH)) - 1)


class FixedISqrtDriver:
    def __init__(self, dut):
        self.dut = dut

    async def reset(self):
        self.dut.rst_n.value = 0
        self.dut.start.value = 0
        self.dut.radicand.value = 0

        for _ in range(4):
            await RisingEdge(self.dut.clk)

        self.dut.rst_n.value = 1
        await RisingEdge(self.dut.clk)

    async def run_once(self, radicand_q: int) -> int:
        await FallingEdge(self.dut.clk)
        self.dut.radicand.value = radicand_q
        self.dut.start.value = 1
        await RisingEdge(self.dut.clk)

        await FallingEdge(self.dut.clk)
        self.dut.start.value = 0

        while True:
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            if int(self.dut.done.value):
                return int(self.dut.root.value)


@cocotb.test()
async def test_fixed_isqrt_random(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    driver = FixedISqrtDriver(dut)
    random.seed(17)

    await driver.reset()

    max_abs_error = 0.0

    for idx in range(NUM_TESTS):
        value = random.uniform(0.0, (1 << INT_WIDTH) - 1.0)
        radicand_q = quantise_input(value)
        radicand = radicand_q / INPUT_SCALE
        cocotb.log.info(f"fixed_isqrt radicand: {radicand:.6f}")

        root_q = await driver.run_once(radicand_q)
        root = root_q / OUTPUT_SCALE
        cocotb.log.info(f"fixed_isqrt root: {root:.6f}")


        golden_q = quantise_output(math.sqrt(radicand))
        golden = golden_q / OUTPUT_SCALE
        cocotb.log.info(f"fixed_isqrt golden: {golden:.6f}")

        abs_error = abs(root - golden)
        max_abs_error = max(max_abs_error, abs_error)

        assert abs_error <= ATOL, (
            f"sqrt mismatch sample={idx} input={radicand:.6f} hw={root:.6f} golden={golden:.6f} "
            f"hw_q={root_q} golden_q={golden_q}"
        )

    cocotb.log.info(f"fixed_isqrt max abs error: {max_abs_error:.6f}")
