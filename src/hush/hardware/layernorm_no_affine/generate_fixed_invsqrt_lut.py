#!/usr/bin/env python3
"""
Generate a pure normalized inverse-square-root lookup table for fixed_invsqrt.sv.
"""

from __future__ import annotations

import math
from pathlib import Path


LUT_ADDR_WIDTH = 16
OUT_WIDTH = 28
OUT_FRAC_WIDTH = 17
LUT_GUARD_BITS = 8
OUTPUT_FILE = Path(__file__).with_name("fixed_invsqrt_lut.mem")


def quantise(value: float, frac_width: int) -> int:
    quantised = int(round(value * (1 << frac_width)))
    max_value = (1 << (OUT_WIDTH + LUT_GUARD_BITS)) - 1
    return min(max(quantised, 0), max_value)


def main() -> None:
    depth = 1 << LUT_ADDR_WIDTH
    frac_width = OUT_FRAC_WIDTH + LUT_GUARD_BITS
    word_width = 2 * (OUT_WIDTH + LUT_GUARD_BITS)

    lines = []
    for idx in range(depth):
        mantissa = 1.0 + (idx / depth)
        even = quantise(1.0 / math.sqrt(mantissa), frac_width)
        odd = quantise(1.0 / math.sqrt(2.0 * mantissa), frac_width)
        word = (odd << (OUT_WIDTH + LUT_GUARD_BITS)) | even
        lines.append(f"{word:0{(word_width + 3) // 4}x}")

    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {OUTPUT_FILE} with {len(lines)} entries")


if __name__ == "__main__":
    main()
