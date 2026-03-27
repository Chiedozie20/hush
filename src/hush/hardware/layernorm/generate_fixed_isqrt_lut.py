#!/usr/bin/env python3
"""
Generate the normalized sqrt lookup table used by fixed_isqrt.sv.

Each ROM entry stores:
- even_base : sqrt(1 + f)
- even_delta: next even_base - even_base
- odd_base  : sqrt(2 * (1 + f))
- odd_delta : next odd_base - odd_base

where f steps across [0, 1) in LUT_ADDR_WIDTH segments. The hardware then uses
the low interpolation bits of the normalized mantissa to linearly interpolate
within the selected segment and applies a power-of-two scale factor.
"""

from __future__ import annotations

import math
from pathlib import Path


LUT_ADDR_WIDTH = 10
ROOT_FRAC_WIDTH = 16
LUT_GUARD_BITS = 10
OUTPUT_FILE = Path(__file__).with_name("fixed_isqrt_lut.mem")


def quantise(value: float, frac_width: int) -> int:
    return int(round(value * (1 << frac_width)))


def main() -> None:
    depth = 1 << LUT_ADDR_WIDTH
    lut_frac_width = ROOT_FRAC_WIDTH + LUT_GUARD_BITS
    value_width = lut_frac_width + 2
    mask = (1 << value_width) - 1

    lines = []
    for idx in range(depth):
        frac0 = idx / depth
        frac1 = (idx + 1) / depth

        even0 = quantise(math.sqrt(1.0 + frac0), lut_frac_width)
        even1 = quantise(math.sqrt(1.0 + frac1), lut_frac_width)
        odd0 = quantise(math.sqrt(2.0 * (1.0 + frac0)), lut_frac_width)
        odd1 = quantise(math.sqrt(2.0 * (1.0 + frac1)), lut_frac_width)

        even_delta = even1 - even0
        odd_delta = odd1 - odd0

        word = (
            ((odd_delta & mask) << (3 * value_width))
            | ((odd0 & mask) << (2 * value_width))
            | ((even_delta & mask) << value_width)
            | (even0 & mask)
        )
        lines.append(f"{word:0{(4 * value_width + 3) // 4}x}")

    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {OUTPUT_FILE} with {depth} entries")


if __name__ == "__main__":
    main()
