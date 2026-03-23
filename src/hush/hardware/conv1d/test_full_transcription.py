import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent.parent
whisper_path = project_root / "whisper"
sys.path.insert(0, str(whisper_path))

import torch
import torch.nn.functional as F
from whisper import load_model
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
from whisper.decoding import decode as whisper_decode, DecodingOptions
import subprocess
import os


SCALE_FACTOR = 100.0


def quantise_to_int16(tensor, scale=SCALE_FACTOR):
    quantised = (tensor * scale).clamp(-32768, 32767).round().int()
    return quantised


def quantise_bias_to_int16(tensor, scale=SCALE_FACTOR):
    quantised = (tensor * scale * scale).clamp(-2147483648, 2147483647).round().int()
    return quantised


def dequantise_from_int16(tensor, scale=SCALE_FACTOR):
    return tensor.float() / (scale * scale)


def run_full_whisper_with_custom_conv1(model, audio, conv1_output):
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    mel_input = mel.unsqueeze(0)

    with torch.no_grad():
        x = conv1_output
        x = F.gelu(x)
        x = model.encoder.conv2(x)
        x = F.gelu(x)
        x = x.permute(0, 2, 1)
        x = (x + model.encoder.positional_embedding).to(x.dtype)

        for block in model.encoder.blocks:
            x = block(x)

        x = model.encoder.ln_post(x)
        options = DecodingOptions(language="en", without_timestamps=True)
        result = whisper_decode(model, mel_input, options)

    return result


def run_hardware_simulation_via_cocotb():
    """
    Run the CocoTB hardware simulation and retrieve the hardware output.
    This function launches the full CocoTB simulation and saves the HW output to a file.
    """
    print("\n" + "="*80)
    print("RUNNING HARDWARE SIMULATION VIA COCOTB")
    print("="*80)

    # Create a temporary test module that runs the simulation and saves output
    test_module_code = '''
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent.parent
whisper_path = project_root / "whisper"
sys.path.insert(0, str(whisper_path))

import cocotb
import torch
import torch.nn.functional as F
from cocotb.clock import Clock
from whisper import load_model
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
from conv1d_driver import Conv1dDriver, quantise_to_int16, quantise_bias_to_int16, dequantise_from_int16


@cocotb.test()
async def test_hw_and_save(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    device = "cpu"
    model = load_model("tiny.en", device=device)

    n_mels = model.dims.n_mels
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

    print("\\nRunning hardware convolution...")
    hw_output_quantised = await driver.conv1d_tensor(mel_quantised, weights_quantised, biases_quantised)
    hw_output = dequantise_from_int16(hw_output_quantised)

    # Save the hardware output
    torch.save(hw_output, "hw_conv1_output.pt")
    print(f"\\nSaved hardware output to hw_conv1_output.pt")
    print(f"HW output shape: {hw_output.shape}")
'''

    temp_test_path = Path(__file__).parent / "temp_hw_test.py"
    with open(temp_test_path, "w") as f:
        f.write(test_module_code)

    try:
        env = os.environ.copy()
        env["MODULE"] = "temp_hw_test"

        print("\nStarting hardware simulation (this may take 1-2 minutes)...")

        result = subprocess.run(
            ["make", "-f", "Makefile.cocotb", "SIM=verilator", "sim"],
            cwd=Path(__file__).parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print("\nCocoTB simulation output:")
            print(result.stdout)
            if result.stderr:
                print("Errors/warnings:")
                print(result.stderr)
            raise RuntimeError(f"CocoTB simulation failed with return code {result.returncode}")

        print("Hardware simulation completed successfully")

        # Load the saved hardware output
        hw_output_path = Path(__file__).parent / "hw_conv1_output.pt"
        if not hw_output_path.exists():
            raise RuntimeError("Hardware output file was not created by simulation")

        print("Loading hardware output from simulation...")
        hw_output = torch.load(hw_output_path)
        print(f"Loaded hardware output: shape={hw_output.shape}")

        # Clean up temporary files
        hw_output_path.unlink()

        return hw_output

    finally:
        # Clean up temporary test module
        if temp_test_path.exists():
            temp_test_path.unlink()


def main():
    print("Loading Whisper tiny.en model...")
    device = "cpu"
    model = load_model("tiny.en", device=device)

    print("Loading JFK audio example...")
    audio_path = whisper_path / "tests" / "jfk.flac"
    audio = load_audio(str(audio_path))
    audio = pad_or_trim(audio)

    print("Generating mel spectrogram...")
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    mel_input = mel.unsqueeze(0)
    conv1_weight = model.encoder.conv1.weight.data
    conv1_bias = model.encoder.conv1.bias.data

    # Run software baseline
    print("\n" + "="*80)
    print("RUNNING SOFTWARE BASELINE")
    print("="*80)
    with torch.no_grad():
        sw_conv1_output = F.conv1d(mel_input, conv1_weight, conv1_bias, stride=1, padding=1)

    sw_result = run_full_whisper_with_custom_conv1(model, audio, sw_conv1_output)
    sw_text = sw_result[0].text

    # Run hardware simulation
    hw_conv1_output = run_hardware_simulation_via_cocotb()

    print("\n" + "="*80)
    print("RUNNING FULL WHISPER WITH HARDWARE CONV1 OUTPUT")
    print("="*80)
    hw_result = run_full_whisper_with_custom_conv1(model, audio, hw_conv1_output)
    hw_text = hw_result[0].text

    # Compare results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"SW Text: {sw_text}")
    print(f"HW Text: {hw_text}")
    print(f"\nTexts match: {sw_text == hw_text}")

    diff = torch.abs(hw_conv1_output - sw_conv1_output)
    print(f"\nConv1 output max difference: {diff.max().item():.6f}")
    print(f"Conv1 output mean difference: {diff.mean().item():.6f}")
    print(f"Conv1 output percentage diff: {diff.float().mean().item() * 100:.2f}%")

    if sw_text == hw_text:
        print("\n" + "="*80)
        print("SUCCESS: Hardware implementation produces identical transcription!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("WARNING: Transcriptions differ!")
        print("="*80)


if __name__ == "__main__":
    main()
