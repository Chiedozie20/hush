import argparse
import wave
from pathlib import Path

import numpy as np
import torch

from whisper import load_model
from whisper.audio import N_FRAMES, N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim
from whisper.decoding import DecodingOptions
from whisper.model import Whisper


def load_fixed_wav(path: Path) -> np.ndarray:
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
    audio = load_fixed_wav(path)
    mel = log_mel_spectrogram(audio, n_mels=n_mels)
    mel = pad_or_trim(mel, N_FRAMES).to(device)
    return mel.float()


def simple_whisper_inference(path: Path) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    model: Whisper = load_model("tiny.en", device=device)
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
    encoded_audio_features = model.encoder(decode_mel.unsqueeze(0))
    result = model.decode(encoded_audio_features[0], options)
    return result.text


def main():
    parser = argparse.ArgumentParser(
        description="Minimal one-file Whisper pipeline (fixed WAV -> text)"
    )
    parser.add_argument("audio_wav", type=Path, help="Path to mono 16-bit 16kHz WAV")
    args = parser.parse_args()

    
    text = simple_whisper_inference(args.audio_wav)
    print(text)


if __name__ == "__main__":
    main()
