import cocotb
import torch
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


LATENCY = 3
TENSOR_SHAPE = (1, 1500, 384)


def build_tensor():
    total_values = TENSOR_SHAPE[0] * TENSOR_SHAPE[1] * TENSOR_SHAPE[2]
    return (torch.arange(total_values, dtype=torch.int32) % 256).reshape(TENSOR_SHAPE)


class PassthroughDriver:
    def __init__(self, dut):
        self.dut = dut

    async def reset(self):
        self.dut.rst_n.value = 0
        self.dut.data_in.value = 0
        self.dut.data_in_valid.value = 0

        for _ in range(2):
            await RisingEdge(self.dut.clk)

        self.dut.rst_n.value = 1

        for _ in range(LATENCY + 1):
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            assert int(self.dut.data_out_valid.value) == 0, "reset should clear the pipeline"

    async def passthrough_tensor(self, tensor):
        expected = [int(value) for value in tensor.flatten().tolist()]
        received = []
        first_output_cycle = None
        cycle = 0

        for value in expected:
            await FallingEdge(self.dut.clk)
            self.dut.data_in.value = value
            self.dut.data_in_valid.value = 1
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            cycle += 1

            if int(self.dut.data_out_valid.value):
                if first_output_cycle is None:
                    first_output_cycle = cycle
                received.append(int(self.dut.data_out.value))

        await FallingEdge(self.dut.clk)
        self.dut.data_in.value = 0
        self.dut.data_in_valid.value = 0

        while len(received) < len(expected):
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            cycle += 1
            if int(self.dut.data_out_valid.value):
                if first_output_cycle is None:
                    first_output_cycle = cycle
                received.append(int(self.dut.data_out.value))

        assert first_output_cycle == LATENCY, (
            f"expected first output after {LATENCY} cycles, got {first_output_cycle}"
        )
        assert len(received) == len(expected), "output count did not match input count"

        return torch.tensor(received, dtype=tensor.dtype).reshape(tensor.shape)


@cocotb.test()
async def test_passthrough_tensor(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    driver = PassthroughDriver(dut)

    await driver.reset()

    tensor = build_tensor()
    passthrough_tensor = await driver.passthrough_tensor(tensor)

    assert torch.equal(
        passthrough_tensor, tensor
    ), "passthrough output differed from tensor input"
