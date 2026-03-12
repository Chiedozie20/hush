import wave
from pathlib import Path

import cocotb
import numpy as np
import torch
import torch.nn.functional as F
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from . import load_model
from .decoding import DecodingOptions
from .model import AudioEncoder, Whisper


LATENCY = 3
TENSOR_SHAPE = (1, 384, 3000)


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


def load_fixed_wav(path: Path) -> np.ndarray:
    from .audio import N_SAMPLES, SAMPLE_RATE

    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        num_frames = wf.getnframes()
        comp_type = wf.getcomptype()

        if channels != 1:
            raise ValueError(f"Expected mono WAV, got {channels} channels")
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit WAV, got {sample_width * 8}-bit")
        if frame_rate != SAMPLE_RATE:
            raise ValueError(
                f"Expected {SAMPLE_RATE} Hz WAV, got {frame_rate} Hz. "
                "Resample to 16 kHz first."
            )
        if comp_type != "NONE":
            raise ValueError("Expected uncompressed PCM WAV")
        if num_frames > N_SAMPLES:
            raise ValueError("Expected audio <= 30 seconds")
        pcm_bytes = wf.readframes(num_frames)

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def prepare_mel(path: Path, n_mels: int, device: torch.device) -> torch.Tensor:
    from .audio import N_FRAMES, log_mel_spectrogram, pad_or_trim

    audio = load_fixed_wav(path)
    mel = log_mel_spectrogram(audio, n_mels=n_mels)
    mel = pad_or_trim(mel, N_FRAMES).to(device)
    return mel.float()


def get_config():
    return {
        "conv1d": "quantised",
        "conv1d_config": {
            "name": "integer",
            "data_in_width": 12,
            "data_in_frac_width": 4,
            "weight_width": 12,
            "weight_frac_width": 4,
            "bias_width": 12,
            "bias_frac_width": 4,
        },
    }


def encode_audio_with_optional_passthrough(
    encoder: AudioEncoder, mel: torch.Tensor, passthrough_tensor=None
) -> torch.Tensor:
    x = F.gelu(encoder.conv1(mel))
    if passthrough_tensor is not None:
        x = passthrough_tensor
    x = F.gelu(encoder.conv2(x))
    x = x.permute(0, 2, 1)
    assert x.shape[1:] == encoder.positional_embedding.shape, "incorrect audio shape"
    x = (x + encoder.positional_embedding).to(x.dtype)

    for block in encoder.blocks:
        x = block(x)

    return encoder.ln_post(x)


@cocotb.test()
async def test_passthrough_tensor(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    driver = PassthroughDriver(dut)

    await driver.reset()

    path = "/home/chiedozie/projects/ADLS/hush/whisper/daveL.wav"
    encoder_config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    model: Whisper = load_model(
        "tiny.en", device=device, encoder_config=encoder_config, driver=driver
    )

    mel = prepare_mel(path, model.dims.n_mels, device_obj)
    fp16 = model.device.type == "cuda"
    decode_mel = mel.half() if fp16 else mel.float()

    options = DecodingOptions(
        task="transcribe",
        language="en",
        temperature=0.0,
        fp16=fp16,
        without_timestamps=True,
    )
    with torch.inference_mode():
        conv1_output = F.gelu(model.encoder.conv1(decode_mel.unsqueeze(0)))
        passthrough_output = await driver.passthrough_tensor(conv1_output)
        encoded_audio_features = encode_audio_with_optional_passthrough(
            model.encoder, decode_mel.unsqueeze(0), passthrough_output
        )
        result = model.decode(encoded_audio_features[0], options)
    print(result.text)

    tensor = build_tensor()
    passthrough_tensor = await driver.passthrough_tensor(tensor)

    assert torch.equal(
        passthrough_tensor, tensor
    ), "passthrough output differed from tensor input"
