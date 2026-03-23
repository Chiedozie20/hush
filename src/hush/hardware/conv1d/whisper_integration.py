import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
whisper_path = project_root / "whisper"
sys.path.insert(0, str(whisper_path))

import cocotb
import torch
import torch.nn.functional as F
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from whisper import load_model
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram

from conv1d_driver import Conv1dDriver, quantise_to_int16, quantise_bias_to_int16, dequantise_from_int16


@cocotb.test()
async def test_whisper_conv1_integration(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    device = "cpu"
    model = load_model("tiny.en", device=device)

    n_mels = model.dims.n_mels
    n_state = model.dims.n_audio_state

    conv1_weight = model.encoder.conv1.weight.data
    conv1_bias = model.encoder.conv1.bias.data

    out_channels, in_channels, kernel_size = conv1_weight.shape
    stride = 1
    max_seq_len = 3000

    driver = Conv1dDriver(dut, in_channels, out_channels, kernel_size, stride, max_seq_len)
    await driver.reset()

    audio_path = whisper_path / "tests" / "jfk.flac"
    audio = load_audio(str(audio_path))
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio, n_mels=n_mels)

    mel_input = mel.unsqueeze(0)

    weights_quantised = quantise_to_int16(conv1_weight)
    biases_quantised = quantise_bias_to_int16(conv1_bias)
    mel_quantised = quantise_to_int16(mel_input)

    print(f"DEBUG: Sample weights: {weights_quantised[0, 0, :3]}")
    print(f"DEBUG: Sample biases: {biases_quantised[:5]}")
    print(f"DEBUG: Sample input: {mel_quantised[0, 0, :5]}")

    hw_output_quantised = await driver.conv1d_tensor(mel_quantised, weights_quantised, biases_quantised)
    hw_output = dequantise_from_int16(hw_output_quantised)

    with torch.no_grad():
        sw_output = F.conv1d(mel_input, conv1_weight, conv1_bias, stride=stride, padding=1)

    diff = torch.abs(hw_output - sw_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    percent_diff = diff.float().mean().item() * 100

    print(f"HW output shape: {hw_output.shape}")
    print(f"SW output shape: {sw_output.shape}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Percentage diff: {percent_diff:.2f}%")
    print(f"HW output sample [0,0,:5]: {hw_output[0, 0, :5]}")
    print(f"SW output sample [0,0,:5]: {sw_output[0, 0, :5]}")
    print(f"HW output nonzero count: {torch.count_nonzero(hw_output)}")
    print(f"HW output sample [0,0,495:505]: {hw_output[0, 0, 495:505]}")
    print(f"SW output sample [0,0,495:505]: {sw_output[0, 0, 495:505]}")

    assert hw_output.shape == sw_output.shape
    assert max_diff < 1.0