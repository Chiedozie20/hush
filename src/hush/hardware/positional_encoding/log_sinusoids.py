import argparse

import numpy as np
import torch


N_CTX = 1500
N_STATE = 384
WIDTH = 16
FRAC_BITS = 12
WIDTH = 16
FRAC_BITS = 12

def quantize_q4_12(values: torch.Tensor, width: int = WIDTH, frac_bits: int = FRAC_BITS) -> torch.Tensor:
    scale = 1 << frac_bits
    qmin = -(1 << (width - 1))
    qmax = (1 << (width - 1)) - 1
    quantized = torch.round(values * scale).to(torch.int32)
    return torch.clamp(quantized, qmin, qmax)

def sinusoids(length: int, channels: int, max_timescale: int = 10000):
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(
        -log_timescale_increment * torch.arange(channels // 2, dtype=torch.float32)
    )
    scaled_time = (
        torch.arange(length, dtype=torch.float32)[:, np.newaxis]
        * inv_timescales[np.newaxis, :]
    )
    result = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    return inv_timescales, scaled_time, result


def quantize_q4_12(value: float, width: int = WIDTH, frac_bits: int = FRAC_BITS) -> int:
    scale = 1 << frac_bits
    qmin = -(1 << (width - 1))
    qmax = (1 << (width - 1)) - 1
    quantized = int(round(value * scale))
    return max(qmin, min(qmax, quantized))


def print_position_debug(position: int, inv_timescales: torch.Tensor, scaled_time: torch.Tensor, result: torch.Tensor):
    half_channels = inv_timescales.shape[0]
    print(f"Position {position}")
    print("chan_idx | kind | freq_idx | inv_timescale      | scaled_time        | value              | q4_12")
    for chan_idx in range(result.shape[1]):
        freq_idx = chan_idx if chan_idx < half_channels else chan_idx - half_channels
        kind = "sin" if chan_idx < half_channels else "cos"
        inv_time = inv_timescales[freq_idx].item()
        phase = scaled_time[position, freq_idx].item()
        value = result[position, chan_idx].item()
        quantized = quantize_q4_12(value)
        print(
            f"{chan_idx:8d} | {kind:4s} | {freq_idx:8d} | {inv_time:18.10f} | {phase:18.10f} | {value:18.10f} | {quantized:5d}"
        )


def main():
    parser = argparse.ArgumentParser(description="Inspect sinusoid values for a single position.")
    parser.add_argument("--length", type=int, default=N_CTX)
    parser.add_argument("--channels", type=int, default=N_STATE)
    parser.add_argument("--max-timescale", type=int, default=10000)
    parser.add_argument("--position", type=int, default=0)
    args = parser.parse_args()

    if not 0 <= args.position < args.length:
        raise ValueError(f"position must be in [0, {args.length - 1}], got {args.position}")

    inv_timescales, scaled_time, result = sinusoids(
        args.length, args.channels, max_timescale=args.max_timescale
    )
    print_position_debug(args.position, inv_timescales, scaled_time, result)


if __name__ == "__main__":
    main()
