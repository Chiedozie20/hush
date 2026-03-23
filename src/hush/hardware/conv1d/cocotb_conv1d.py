import cocotb
import torch
import torch.nn.functional as F
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


DATA_WIDTH = 16
ACC_WIDTH = 2 * DATA_WIDTH  # 32-bit accumulator output
IN_CHANNELS = 4
OUT_CHANNELS = 8
KERNEL_SIZE = 3
STRIDE = 1
MAX_SEQ_LEN = 100
INPUT_LENGTH = 50


class Conv1dDriver:
    def __init__(self, dut):
        self.dut = dut
        self.data_width = DATA_WIDTH
        self.acc_width = ACC_WIDTH
        self.in_channels = IN_CHANNELS
        self.out_channels = OUT_CHANNELS
        self.kernel_size = KERNEL_SIZE
        self.stride = STRIDE

    async def reset(self):
        self.dut.rst_n.value = 0
        self.dut.start.value = 0
        self.dut.data_in.value = 0
        self.dut.data_in_valid.value = 0
        self.dut.in_channel_idx.value = 0
        self.dut.in_pos_idx.value = 0
        self.dut.weight_in.value = 0
        self.dut.weight_valid.value = 0
        self.dut.weight_out_ch.value = 0
        self.dut.weight_in_ch.value = 0
        self.dut.weight_k_idx.value = 0
        self.dut.bias_in.value = 0
        self.dut.bias_valid.value = 0
        self.dut.bias_out_ch.value = 0
        self.dut.data_out_ready.value = 1
        self.dut.input_length.value = INPUT_LENGTH

        for _ in range(5):
            await RisingEdge(self.dut.clk)

        self.dut.rst_n.value = 1

        for _ in range(5):
            await RisingEdge(self.dut.clk)

    async def _load_weights(self, weights):
        await RisingEdge(self.dut.clk)
        self.dut.start.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.start.value = 0

        while int(self.dut.busy.value) == 0:
            await RisingEdge(self.dut.clk)

        for out_ch in range(self.out_channels):
            for in_ch in range(self.in_channels):
                for k in range(self.kernel_size):
                    await FallingEdge(self.dut.clk)
                    weight_val = int(weights[out_ch, in_ch, k].item())
                    self.dut.weight_in.value = weight_val
                    self.dut.weight_valid.value = 1
                    self.dut.weight_out_ch.value = out_ch
                    self.dut.weight_in_ch.value = in_ch
                    self.dut.weight_k_idx.value = k
                    await RisingEdge(self.dut.clk)

        await FallingEdge(self.dut.clk)
        self.dut.weight_valid.value = 0
        await RisingEdge(self.dut.clk)

    async def _load_biases(self, biases):
        for out_ch in range(self.out_channels):
            await FallingEdge(self.dut.clk)
            bias_val = int(biases[out_ch].item())
            self.dut.bias_in.value = bias_val
            self.dut.bias_valid.value = 1
            self.dut.bias_out_ch.value = out_ch
            await RisingEdge(self.dut.clk)

        await FallingEdge(self.dut.clk)
        self.dut.bias_valid.value = 0
        await RisingEdge(self.dut.clk)

    async def _load_input(self, input_tensor):
        batch, in_ch, seq_len = input_tensor.shape
        assert batch == 1
        assert in_ch == self.in_channels
        assert seq_len <= MAX_SEQ_LEN

        for ch in range(in_ch):
            for pos in range(seq_len):
                await FallingEdge(self.dut.clk)
                data_val = int(input_tensor[0, ch, pos].item())
                self.dut.data_in.value = data_val
                self.dut.data_in_valid.value = 1
                self.dut.in_channel_idx.value = ch
                self.dut.in_pos_idx.value = pos
                await RisingEdge(self.dut.clk)

        await FallingEdge(self.dut.clk)
        self.dut.data_in_valid.value = 0
        await RisingEdge(self.dut.clk)

    async def _read_output(self, output_length):
        outputs = []
        total = self.out_channels * output_length

        while len(outputs) < total:
            await RisingEdge(self.dut.clk)
            await ReadOnly()

            if int(self.dut.data_out_valid.value):
                data_val = int(self.dut.data_out.value)
                out_ch = int(self.dut.out_channel_idx.value)
                out_pos = int(self.dut.out_pos_idx.value)
                if data_val & (1 << (self.acc_width - 1)):
                    data_val -= (1 << self.acc_width)

                outputs.append((out_ch, out_pos, data_val))

        output_tensor = torch.zeros(1, self.out_channels, output_length, dtype=torch.int64)
        for out_ch, out_pos, val in outputs:
            output_tensor[0, out_ch, out_pos] = val

        return output_tensor

    async def conv1d_tensor(self, input_tensor, weights, biases):
        input_length = input_tensor.shape[2]
        output_length = (input_length + 2 - self.kernel_size) // self.stride + 1

        self.dut.input_length.value = input_length

        await self._load_weights(weights)
        await self._load_biases(biases)
        await self._load_input(input_tensor)

        output = await self._read_output(output_length)
        while int(self.dut.done.value) == 0:
            await RisingEdge(self.dut.clk)

        await FallingEdge(self.dut.clk)
        self.dut.start.value = 0
        await RisingEdge(self.dut.clk)

        return output


@cocotb.test()
async def test_conv1d(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    driver = Conv1dDriver(dut)

    await driver.reset()

    input_tensor = torch.randint(-10, 10, (1, IN_CHANNELS, INPUT_LENGTH), dtype=torch.int32)
    weights = torch.randint(-5, 5, (OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE), dtype=torch.int32)
    biases = torch.randint(-5, 5, (OUT_CHANNELS,), dtype=torch.int32)

    hw_output = await driver.conv1d_tensor(input_tensor, weights, biases)

    input_float = input_tensor.float()
    weights_float = weights.float()
    biases_float = biases.float()
    ref_output = F.conv1d(input_float, weights_float, biases_float, stride=STRIDE, padding=1)
    ref_output_int = ref_output.long()

    print(f"Input shape: {input_tensor.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"HW output shape: {hw_output.shape}")
    print(f"Ref output shape: {ref_output_int.shape}")
    print(f"HW output sample: {hw_output[0, 0, :10]}")
    print(f"Ref output sample: {ref_output_int[0, 0, :10]}")

    diff = torch.abs(hw_output - ref_output_int)
    max_diff = diff.max().item()
    mean_diff = diff.float().mean().item()
    percent_diff = (diff != 0).float().mean().item() * 100
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff:.4f}")
    print(f"Percentage error: {percent_diff:.2f}%")

    assert torch.equal(hw_output, ref_output_int), \
        f"HW output does not match reference! Max diff: {max_diff}"
    print("PASS: HW output matches reference exactly")