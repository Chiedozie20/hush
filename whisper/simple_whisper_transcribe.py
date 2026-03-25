import argparse
import os
import wave
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from whisper.audio import (
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from whisper.decoding import DecodingOptions
from whisper.model import Whisper
from whisper.normalizers import EnglishTextNormalizer

import whisper
from whisper import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    import soundfile as sf

    def _patched_load(filepath, *args, **kwargs):
        data, samplerate = sf.read(filepath, dtype="float32")
        waveform = torch.from_numpy(data).unsqueeze(0)
        return waveform, samplerate

    torchaudio.load = _patched_load

except ImportError:
    pass



class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """

    def __init__(self, split="test-clean", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)

        return (mel, text)


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


def get_config():
    encoder_config = {
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
        "attention": "quantised",
        "attention_config": {
            "name": "integer",
            "data_in_width": 16,
            "data_in_frac_width": 8,
            "weight_width": 16,
            "weight_frac_width": 8,
            "bias_width": 16,
            "bias_frac_width": 8,
        },
    }
    return encoder_config


def simple_whisper_inference(path: Path) -> str:
    encoder_config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    model: Whisper = load_model("tiny.en", device=device, encoder_config=encoder_config)

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
        encoded_audio_features = model.encoder(decode_mel.unsqueeze(0))
        result = model.decode(encoded_audio_features[0], options)
    return result.text


def benchmark_WER(model):
    dataset = LibriSpeech("test-clean")
    dataset = torch.utils.data.Subset(dataset, range(10,20))
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
    options = whisper.DecodingOptions(language="en", without_timestamps=True)

    hypotheses = []
    references = []

    for mels, texts in tqdm(loader):
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)
    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))

    normalizer = EnglishTextNormalizer()
    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]
    wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    return wer


def main():
    parser = argparse.ArgumentParser(
        description="Minimal one-file Whisper pipeline (fixed WAV -> text)"
    )
    parser.add_argument(
        "audio_wav",
        nargs="?",
        type=Path,
        help="Path to mono 16-bit 16kHz WAV",
    )
    parser.add_argument(
        "--test-wer",
        action="store_true",
        help="Run the LibriSpeech WER benchmark instead of single-file transcription",
    )
    args = parser.parse_args()

    if args.test_wer:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder_config = get_config()
        model: Whisper = load_model(
            "tiny.en", device=device, encoder_config=encoder_config
        )
        wer = benchmark_WER(model)
        print(f"WER: {wer * 100:.2f} %")
        return

    if args.audio_wav is None:
        parser.error("audio_wav is required unless --test-wer is set")

    text = simple_whisper_inference(args.audio_wav)
    print(text)


if __name__ == "__main__":
    main()
