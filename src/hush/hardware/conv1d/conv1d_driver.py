import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
whisper_path = project_root / "whisper"
sys.path.insert(0, str(whisper_path))

import cocotb
import torch
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly

DATA_WIDTH = 16
SCALE_FACTOR = 100.0

def quantise_to_int16(tensor, scale=SCALE_FACTOR):
    quantised = (tensor * scale).clamp(-32768, 32767).round().int()
    return quantised


def quantise_bias_to_int16(tensor, scale=SCALE_FACTOR):
    quantised = (tensor * scale * scale).clamp(-2147483648, 2147483647).round().int()
    return quantised


def dequantise_from_int16(tensor, scale=SCALE_FACTOR):
    return tensor.float() / (scale * scale)


class Conv1dDriver:
    def __init__(self, dut, in_channels, out_channels, kernel_size, stride, max_seq_len):
        self.dut = dut
        self.data_width = DATA_WIDTH
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_seq_len = max_seq_len

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
        self.dut.input_length.value = 0

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

        total = self.out_channels * self.in_channels * self.kernel_size
        count = 0
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
                    count += 1
                    if count % 10000 == 0:
                        print(f"Loading weights: {count}/{total} ({100*count/total:.1f}%)")

        await FallingEdge(self.dut.clk)
        self.dut.weight_valid.value = 0
        await RisingEdge(self.dut.clk)
        print(f"Weights loaded: {total}/{total} (100.0%)")

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
        assert seq_len <= self.max_seq_len

        total = in_ch * seq_len
        count = 0
        for ch in range(in_ch):
            for pos in range(seq_len):
                await FallingEdge(self.dut.clk)
                data_val = int(input_tensor[0, ch, pos].item())
                self.dut.data_in.value = data_val
                self.dut.data_in_valid.value = 1
                self.dut.in_channel_idx.value = ch
                self.dut.in_pos_idx.value = pos
                await RisingEdge(self.dut.clk)
                count += 1
                if count % 10000 == 0:
                    print(f"Loading input: {count}/{total} ({100*count/total:.1f}%)")

        await FallingEdge(self.dut.clk)
        self.dut.data_in_valid.value = 0
        await RisingEdge(self.dut.clk)
        print(f"Input loaded: {total}/{total} (100.0%)")

    async def _read_output(self, output_length):
        acc_width = 2 * self.data_width  # 32 bits — full accumulator

        outputs = []
        total = self.out_channels * output_length
        last_percent = 0

        while len(outputs) < total and int(self.dut.done.value) == 0:
            await RisingEdge(self.dut.clk)
            await ReadOnly()

            if int(self.dut.data_out_valid.value):
                data_val = int(self.dut.data_out.value)
                out_ch = int(self.dut.out_channel_idx.value)
                out_pos = int(self.dut.out_pos_idx.value)

                # Sign-extend from accumulator width (32 bits)
                if data_val & (1 << (acc_width - 1)):
                    data_val -= (1 << acc_width)

                outputs.append((out_ch, out_pos, data_val))

                percent = int(100 * len(outputs) / total)
                if percent >= last_percent + 10:
                    print(f"Reading output: {len(outputs)}/{total} ({percent}%)")
                    last_percent = percent

        output_tensor = torch.zeros(1, self.out_channels, output_length, dtype=torch.int64)
        max_out_pos = -1
        skipped = 0
        for out_ch, out_pos, val in outputs:
            if out_pos >= output_length:
                print(f"WARNING: Skipping out_pos={out_pos} >= output_length={output_length}")
                skipped += 1
                continue
            max_out_pos = max(max_out_pos, out_pos)
            output_tensor[0, out_ch, out_pos] = val

        print(f"Output read: {len(outputs)-skipped}/{total} ({100*(len(outputs)-skipped)/total:.1f}%), max_out_pos={max_out_pos}, skipped={skipped}")
        return output_tensor

    async def conv1d_tensor(self, input_tensor, weights, biases):
        input_length = input_tensor.shape[2]
        output_length = (input_length + 2 - self.kernel_size) // self.stride + 1

        self.dut.input_length.value = input_length

        print("Starting weight loading...")
        await self._load_weights(weights)

        await RisingEdge(self.dut.clk)
        await ReadOnly()
        w_loaded = [int(self.dut.weights[0][i][0].value) for i in range(min(5, self.in_channels))]
        print(f"DEBUG: After weight load - weights[0][0:5][0]={w_loaded}")

        print("Starting bias loading...")
        await self._load_biases(biases)

        await RisingEdge(self.dut.clk)
        await ReadOnly()
        b_loaded = [int(self.dut.biases[i].value) for i in range(min(5, self.out_channels))]
        print(f"DEBUG: After bias load - biases[0:5]={b_loaded}")

        print("Starting input loading...")
        await self._load_input(input_tensor)

        await RisingEdge(self.dut.clk)
        await ReadOnly()
        inp_loaded = [int(self.dut.input_data[0][i].value) for i in range(min(5, input_length))]
        input_loaded_sig = int(self.dut.input_loaded.value)
        input_count_sig = int(self.dut.input_count.value)
        print(f"DEBUG: After input load - input_data[0][0:5]={inp_loaded}")
        print(f"DEBUG: input_loaded={input_loaded_sig}, input_count={input_count_sig}, expected={self.in_channels * input_length}")

        print("Starting concurrent computation and output read...")

        async def monitor_computation():
            cycles = 0
            first_output_seen = False
            while int(self.dut.done.value) == 0:
                await RisingEdge(self.dut.clk)
                cycles += 1
                await ReadOnly()

                if cycles == 10:
                    mac_comp = int(self.dut.mac_computing.value)
                    mac_cnt = int(self.dut.mac_cycle_counter.value)
                    acc0 = int(self.dut.mac_accumulators[0].value)
                    print(f"DEBUG: cycle 10 - mac_computing={mac_comp}, mac_cycle_counter={mac_cnt}, mac_accumulators[0]={acc0}")

                if int(self.dut.data_out_valid.value) and not first_output_seen:
                    first_output_seen = True
                    out_val = int(self.dut.data_out.value)
                    out_ch = int(self.dut.out_channel_idx.value)
                    out_pos = int(self.dut.out_pos_idx.value)
                    acc_val = int(self.dut.mac_accumulators[0].value)
                    print(f"DEBUG: First output - cycle={cycles}, out_ch={out_ch}, out_pos={out_pos}, data_out={out_val}, mac_acc[0]={acc_val}")

                if cycles % 100000 == 0:
                    out_pos_val = int(self.dut.out_pos.value)
                    print(f"Computing... {cycles} cycles elapsed, out_pos={out_pos_val}")
            final_out_pos = int(self.dut.out_pos.value)
            print(f"Computation done in {cycles} cycles, final out_pos={final_out_pos}, expected output_length={output_length}")
        compute_task = cocotb.start_soon(monitor_computation())
        output = await self._read_output(output_length)
        await compute_task
        await FallingEdge(self.dut.clk)
        self.dut.start.value = 0
        await RisingEdge(self.dut.clk)

        return output
