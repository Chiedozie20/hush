#!/usr/bin/env python3
"""
Simple GELU LUT generator without requiring mase/chop dependencies.
Generates a SystemVerilog case-based LUT for GELU activation.
"""

import torch
import torch.nn.functional as F

def generate_gelu_lut_sv(
    data_width: int,
    frac_width: int,
    output_file: str
):
    """
    Generate a SystemVerilog GELU LUT module.

    Args:
        data_width: Total bit width (e.g., 16)
        frac_width: Number of fractional bits (e.g., 14 means divide by 2^14 = 16384)
        output_file: Path to output .sv file
    """

    # Calculate range
    min_val = -(2 ** (data_width - 1))
    max_val = (2 ** (data_width - 1)) - 1
    scale = 2 ** frac_width

    print(f"Generating GELU LUT:")
    print(f"  Data width: {data_width} bits")
    print(f"  Frac width: {frac_width} bits (scale = {scale})")
    print(f"  Integer range: [{min_val}, {max_val}]")
    print(f"  Float range: [{min_val/scale:.4f}, {max_val/scale:.4f}]")
    print(f"  Total entries: {max_val - min_val + 1}")

    # Generate LUT
    lut = {}
    for int_val in range(min_val, max_val + 1):
        # Convert to float
        float_val = int_val / scale

        # Apply GELU
        gelu_out = F.gelu(torch.tensor([float_val])).item()

        # Quantize back to fixed point
        gelu_int = int(round(gelu_out * scale))

        # Clamp to valid range
        gelu_int = max(min_val, min(max_val, gelu_int))

        lut[int_val] = gelu_int

    # Generate SystemVerilog module
    sv_code = f"""// Auto-generated GELU LUT
// Data width: {data_width} bits, Fractional width: {frac_width} bits
// Scale factor: {scale} (divide by this to get float value)

`timescale 1ns / 1ps

module gelu_lut #(
    parameter int DATA_WIDTH = {data_width}
) (
    input  logic signed [DATA_WIDTH-1:0] data_in,
    output logic signed [DATA_WIDTH-1:0] data_out
);

    always_comb begin
        case (data_in)
"""

    # Add case entries (in chunks to avoid huge file)
    # For large LUTs, we can use ranges or default
    count = 0
    for int_val, gelu_val in lut.items():
        # Convert to binary string with proper width
        # For negative numbers, use unary minus on positive value
        if int_val < 0:
            input_str = f"-{data_width}'sd{-int_val}"
        else:
            input_str = f"{data_width}'sd{int_val}"

        if gelu_val < 0:
            output_str = f"-{data_width}'sd{-gelu_val}"
        else:
            output_str = f"{data_width}'sd{gelu_val}"

        sv_code += f"            {input_str}: data_out = {output_str};\n"
        count += 1

        if count % 1000 == 0:
            print(f"  Generated {count}/{len(lut)} entries...")

    sv_code += f"""            default: data_out = {data_width}'sd0;  // Should never happen
        endcase
    end

endmodule
"""

    # Write to file
    with open(output_file, 'w') as f:
        f.write(sv_code)

    print(f"\nGenerated {output_file} with {len(lut)} LUT entries")

    # Print some sample values for verification
    print("\nSample GELU values:")
    test_vals = [-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    for val in test_vals:
        int_in = int(round(val * scale))
        if int_in >= min_val and int_in <= max_val:
            int_out = lut[int_in]
            float_out = int_out / scale
            sw_out = F.gelu(torch.tensor([val])).item()
            print(f"  GELU({val:5.1f}) = {float_out:7.4f}  (SW: {sw_out:7.4f}, diff: {abs(float_out-sw_out):.4f})")


if __name__ == "__main__":
    # Generate 16-bit GELU LUT with 14 fractional bits
    # This matches our conv1d quantization: scale = 100^2 = 10000 ≈ 2^14 = 16384
    generate_gelu_lut_sv(
        data_width=16,
        frac_width=14,  # 2^14 = 16384 ≈ 10000 (our scale factor)
        output_file="gelu_lut.sv"
    )
