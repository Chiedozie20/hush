#!/usr/bin/env python3

import math
from pathlib import Path
import torch
import numpy as np
WIDTH = 16
FRAC_BITS = 12
MAX_TIMESCALE = 10000
HALF_STATE = 192
SIN_LUT_DEPTH = 1024

FREQ_FILE = "freq_lut.mem"
SIN_FILE = "sin_lut.mem"


def quantize_q4_12(value: float, width: int = WIDTH, frac_bits: int = FRAC_BITS) -> int:
    scale = 1 << frac_bits
    qmin = -(1 << (width - 1))
    qmax = (1 << (width - 1)) - 1
    quantized = int(round(value * scale))
    return max(qmin, min(qmax, quantized))


def to_hex_word(value: int, width: int = WIDTH) -> str:
    mask = (1 << width) - 1
    digits = width // 4
    return f"{value & mask:0{digits}x}"


def build_frequency_lut(depth: int = HALF_STATE) -> list[int]:
    log_timescale_increment = math.log(MAX_TIMESCALE) / (depth - 1)
    inv_timescales = [quantize_q4_12(math.exp(-log_timescale_increment * index)) for index in range(depth)]
    max_timescale=10000
    channels = 2 * depth
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales2 = torch.exp(
        -log_timescale_increment * torch.arange(channels // 2, dtype=torch.float32)
    )
    print(inv_timescales)
    print(inv_timescales2)
    return inv_timescales
    


def build_sine_lut(depth: int = SIN_LUT_DEPTH) -> list[int]:
    return [
        quantize_q4_12(math.sin(math.pi * index / (depth - 1)))
        for index in range(depth)
    ]


def write_mem(path: Path, values: list[int]) -> None:
    path.write_text("\n".join(to_hex_word(value) for value in values) + "\n", encoding="ascii")


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    write_mem(out_dir / FREQ_FILE, build_frequency_lut())
    write_mem(out_dir / SIN_FILE, build_sine_lut())


if __name__ == "__main__":
    main()
