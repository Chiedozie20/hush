"""
Simulated Annealing (SA) over Whisper encoder quantisation configs!
"""

import argparse
import json
import math
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import jiwer
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from whisper.model import Whisper
from whisper.normalizers import EnglishTextNormalizer

import whisper
from whisper import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T_START = 1.5
T_END = 0.005
STEPS = 150
RESTARTS = 5
HW_WEIGHT = 0.05
SUBSET_SIZE = 100
SEED = 42
LOG_PATH = Path("anneal_log.json")

WIDTH_MIN = 4
WIDTH_MAX = 32

_FRAC_TO_WIDTH = {
    "conv1d_data_in_frac_width": "conv1d_data_in_width",
    "conv1d_weight_frac_width": "conv1d_weight_width",
    "conv1d_bias_frac_width": "conv1d_bias_width",
    "attn_data_in_frac_width": "attn_data_in_width",
    "attn_weight_frac_width": "attn_weight_width",
    "attn_bias_frac_width": "attn_bias_width",
    "layernorm_data_in_frac_width": "layernorm_data_in_width",
    "layernorm_weight_frac_width": "layernorm_weight_width",
    "layernorm_bias_frac_width": "layernorm_bias_width",
    "positional_embedding_data_in_frac_width": "positional_embedding_data_in_width",
}

_original_load = torchaudio.load


def _patched_load(filepath, *args, **kwargs):
    data, samplerate = sf.read(filepath, dtype="float32")
    waveform = torch.from_numpy(data).unsqueeze(0)
    return waveform, samplerate


torchaudio.load = _patched_load


class LibriSpeech(torch.utils.data.Dataset):
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


def benchmark_WER(model: Whisper) -> float:
    dataset = LibriSpeech("test-clean")
    dataset = torch.utils.data.Subset(dataset, range(SUBSET_SIZE))
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    hypotheses = []
    references = []

    for mels, texts in tqdm(loader, desc="eval", leave=False):
        results = model.decode(mels, options)
        hypotheses.extend([r.text for r in results])
        references.extend(texts)

    normalizer = EnglishTextNormalizer()
    hyp_clean = [normalizer(t) for t in hypotheses]
    ref_clean = [normalizer(t) for t in references]
    return jiwer.wer(ref_clean, hyp_clean)


@dataclass
class QuantState:
    conv1d_mode: str = "quantised"
    conv1d_data_in_width: int = 16
    conv1d_data_in_frac_width: int = 8
    conv1d_weight_width: int = 16
    conv1d_weight_frac_width: int = 8
    conv1d_bias_width: int = 16
    conv1d_bias_frac_width: int = 8

    attn_mode: str = "quantised"
    attn_data_in_width: int = 16
    attn_data_in_frac_width: int = 8
    attn_weight_width: int = 16
    attn_weight_frac_width: int = 8
    attn_bias_width: int = 16
    attn_bias_frac_width: int = 8

    layernorm_mode: str = "quantised"
    layernorm_data_in_width: int = 32
    layernorm_data_in_frac_width: int = 16
    layernorm_weight_width: int = 32
    layernorm_weight_frac_width: int = 16
    layernorm_bias_width: int = 32
    layernorm_bias_frac_width: int = 16

    positional_embedding_mode: str = "quantised"
    positional_embedding_data_in_width: int = 16
    positional_embedding_data_in_frac_width: int = 12

    def to_encoder_config(self) -> dict:
        def _block(mode: str, prefix: str):
            if mode != "quantised":
                return mode, {}
            return "quantised", {
                "name": "integer",
                "data_in_width": getattr(self, f"{prefix}_data_in_width"),
                "data_in_frac_width": getattr(self, f"{prefix}_data_in_frac_width"),
                "weight_width": getattr(self, f"{prefix}_weight_width"),
                "weight_frac_width": getattr(self, f"{prefix}_weight_frac_width"),
                "bias_width": getattr(self, f"{prefix}_bias_width"),
                "bias_frac_width": getattr(self, f"{prefix}_bias_frac_width"),
            }

        def _data_only_block(mode: str, prefix: str):
            if mode != "quantised":
                return mode, {}
            return "quantised", {
                "name": "integer",
                "data_in_width": getattr(self, f"{prefix}_data_in_width"),
                "data_in_frac_width": getattr(self, f"{prefix}_data_in_frac_width"),
            }

        c_mode, c_cfg = _block(self.conv1d_mode, "conv1d")
        a_mode, a_cfg = _block(self.attn_mode, "attn")
        l_mode, l_cfg = _block(self.layernorm_mode, "layernorm")
        p_mode, p_cfg = _data_only_block(
            self.positional_embedding_mode, "positional_embedding"
        )

        cfg: dict = {
            "conv1d": c_mode,
            "attention": a_mode,
            "layernorm": l_mode,
            "positional_embedding": p_mode,
        }

        if c_cfg:
            cfg["conv1d_config"] = c_cfg
        if a_cfg:
            cfg["attention_config"] = a_cfg
        if l_cfg:
            cfg["layernorm_config"] = l_cfg
        if p_cfg:
            cfg["positional_embedding_config"] = p_cfg

        return cfg

    @classmethod
    def from_encoder_config(cls, cfg: dict) -> "QuantState":
        kw: dict = {}
        for block, prefix in [
            ("conv1d", "conv1d"),
            ("attention", "attn"),
            ("layernorm", "layernorm"),
        ]:
            kw[f"{prefix}_mode"] = cfg.get(block, "float")
            sub = cfg.get(f"{block}_config", {})
            for suffix in (
                "data_in_width",
                "data_in_frac_width",
                "weight_width",
                "weight_frac_width",
                "bias_width",
                "bias_frac_width",
            ):
                kw[f"{prefix}_{suffix}"] = sub.get(suffix, 16)

        pos_sub = cfg.get("positional_embedding_config", {})
        kw["positional_embedding_mode"] = cfg.get("positional_embedding", "quantised")
        kw["positional_embedding_data_in_width"] = pos_sub.get("data_in_width", 16)
        kw["positional_embedding_data_in_frac_width"] = pos_sub.get(
            "data_in_frac_width", 12
        )

        return cls(**kw)

    def total_bits(self) -> int:
        total = 0
        for prefix, mode in [
            ("conv1d", self.conv1d_mode),
            ("attn", self.attn_mode),
            ("layernorm", self.layernorm_mode),
        ]:
            if mode == "quantised":
                for s in ("data_in_width", "weight_width", "bias_width"):
                    total += getattr(self, f"{prefix}_{s}")
            else:
                total += 32 * 3

        if self.positional_embedding_mode == "quantised":
            total += self.positional_embedding_data_in_width
        else:
            total += 32

        return total

    def _int_fields(self) -> list[str]:
        return [
            f for f in self.__dataclass_fields__ if isinstance(getattr(self, f), int)
        ]

    def _cat_fields(self) -> list[str]:
        return [
            "conv1d_mode",
            "attn_mode",
            "layernorm_mode",
            "positional_embedding_mode",
        ]


def neighbour(state: QuantState) -> QuantState:
    s = deepcopy(state)
    choices = s._int_fields() + s._cat_fields()
    field_name = random.choice(choices)

    if field_name in s._cat_fields():
        old = getattr(s, field_name)
        setattr(s, field_name, "float" if old == "quantised" else "quantised")
    else:
        delta = random.choice([-2, -1, 1, 2])
        new_val = max(WIDTH_MIN, min(WIDTH_MAX, getattr(s, field_name) + delta))
        setattr(s, field_name, new_val)

        if field_name in _FRAC_TO_WIDTH:
            parent_val = getattr(s, _FRAC_TO_WIDTH[field_name])
            if new_val > parent_val:
                setattr(s, field_name, parent_val)
        for frac, parent in _FRAC_TO_WIDTH.items():
            if parent == field_name and getattr(s, frac) > new_val:
                setattr(s, frac, new_val)

    return s


def cost_fn(state: QuantState) -> tuple[float, float, float]:
    encoder_config = state.to_encoder_config()
    model = load_model("tiny.en", device=DEVICE, encoder_config=encoder_config)
    wer = benchmark_WER(model)
    hw_cost = state.total_bits() / (32 * 10)
    total_cost = wer + HW_WEIGHT * hw_cost
    return total_cost, wer, hw_cost


def _single_schedule(
    initial: QuantState,
    t_start: float,
    restart_idx: int,
    log: list[dict],
    global_step: int,
) -> tuple[QuantState, float, float, float, int]:
    state = deepcopy(initial)
    cost, wer, hw_cost = cost_fn(state)
    best_state, best_cost = deepcopy(state), cost
    best_wer, best_hw_cost = wer, hw_cost

    for step in range(STEPS):
        T = t_start * (T_END / t_start) ** (step / max(STEPS - 1, 1))

        candidate = neighbour(state)
        candidate_cost, candidate_wer, candidate_hw_cost = cost_fn(candidate)

        delta = candidate_cost - cost
        accepted = delta < 0 or random.random() < math.exp(-delta / T)
        if accepted:
            state, cost = candidate, candidate_cost
            wer, hw_cost = candidate_wer, candidate_hw_cost

        if cost < best_cost:
            best_state, best_cost = deepcopy(state), cost
            best_wer, best_hw_cost = wer, hw_cost

        print(
            f"[{restart_idx}/{RESTARTS} step {step:3d}] T={T:.4f}  cost={cost:.4f}  "
            f"best={best_cost:.4f}  bits={state.total_bits():4d}  "
            f"conv1d={state.conv1d_mode:<9s}  "
            f"attn={state.attn_mode:<9s}  "
            f"layernorm={state.layernorm_mode:<9s}  "
            f"positional_embedding={state.positional_embedding_mode:<9s}"
        )

        log.append(
            {
                "restart": restart_idx,
                "step": global_step,
                "T": round(T, 6),
                "cost": round(cost, 6),
                "best_cost": round(best_cost, 6),
                "wer": round(wer, 6),
                "hw_cost": round(hw_cost, 6),
                "best_wer": round(best_wer, 6),
                "best_hw_cost": round(best_hw_cost, 6),
                "total_bits": state.total_bits(),
                "accepted": accepted,
                "config": state.to_encoder_config(),
            }
        )
        global_step += 1

    return best_state, best_cost, best_wer, best_hw_cost, global_step


def simulated_annealing(
    initial: QuantState | None = None,
) -> tuple[QuantState, float, float, float]:
    random.seed(SEED)

    best_state = initial or QuantState()
    best_cost = float("inf")
    best_wer = float("inf")
    best_hw_cost = float("inf")
    log: list[dict] = []
    global_step = 0

    sa_start = time.perf_counter()

    for r in range(RESTARTS):
        t_start = T_START / (r + 1)
        print(f"\n--- restart {r}/{RESTARTS}  T_start={t_start:.4f} --\n")

        run_best, run_cost, run_best_wer, run_best_hw_cost, global_step = (
            _single_schedule(best_state, t_start, r, log, global_step)
        )

        if run_cost < best_cost:
            best_state, best_cost = run_best, run_cost
            best_wer, best_hw_cost = run_best_wer, run_best_hw_cost

        print(
            f"\n--- restart {r}/{RESTARTS} done  "
            f"run_best={run_cost:.4f}  global_best={best_cost:.4f} ---"
        )

    sa_elapsed = time.perf_counter() - sa_start
    total_steps = RESTARTS * STEPS
    avg_per_restart = sa_elapsed / RESTARTS if RESTARTS else 0.0
    avg_per_step = sa_elapsed / total_steps if total_steps else 0.0

    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"\nLog saved to {LOG_PATH}")
    print(
        f"SA took {sa_elapsed:.2f}s total  "
        f"({RESTARTS} restarts, {total_steps} steps, "
        f"avg {avg_per_restart:.2f}s/restart, {avg_per_step:.2f}s/step)"
    )

    return best_state, best_cost, best_wer, best_hw_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    initial = None
    if args.resume:
        with open(args.resume) as f:
            prior_log = json.load(f)
        best_entry = min(prior_log, key=lambda e: e["best_cost"])
        initial = QuantState.from_encoder_config(best_entry["config"])
        print(
            f"Resuming from step {best_entry['step']} "
            f"(cost={best_entry['best_cost']:.4f})"
        )

    best_state, best_cost, best_wer, best_hw_cost = simulated_annealing(initial=initial)

    print(f"\n{'=' * 60}")
    print(
        f"Best cost: {best_cost:.4f}  "
        f"(wer={best_wer:.4f}, hw_cost={best_hw_cost:.4f}, bits={best_state.total_bits()})"
    )
    print(f"Config:\n{json.dumps(best_state.to_encoder_config(), indent=2)}")


if __name__ == "__main__":
    main()
