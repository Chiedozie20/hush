
import math

def gelu(value: float) -> float:
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))

def generate_gelu_lut_sv(
    data_width: int,
    frac_width: int,
    output_file: str
):

    min_val = -(2 ** (data_width - 1))
    max_val = (2 ** (data_width - 1)) - 1
    scale = 2 ** frac_width

    print(f"Generating GELU LUT:")
    print(f"  Data width: {data_width} bits")
    print(f"  Frac width: {frac_width} bits (scale = {scale})")
    print(f"  Integer range: [{min_val}, {max_val}]")
    print(f"  Float range: [{min_val/scale:.4f}, {max_val/scale:.4f}]")
    print(f"  Total entries: {max_val - min_val + 1}")

    lut = {}
    for int_val in range(min_val, max_val + 1):

        float_val = int_val / scale

        gelu_out = gelu(float_val)

        gelu_int = int(round(gelu_out * scale))

        gelu_int = max(min_val, min(max_val, gelu_int))

        lut[int_val] = gelu_int

    addr_width = data_width
    depth = 2 ** addr_width
    offset = 2 ** (data_width - 1)

    count = 0
    for int_val, gelu_val in lut.items():
        if gelu_val < 0:
            output_str = f"-{data_width}'sd{-gelu_val}"
        else:
            output_str = f"{data_width}'sd{gelu_val}"

        addr = int_val + offset
        sv_code += f"        lut_mem[{addr}] = {output_str};\n"
        count += 1

        if count % 1000 == 0:
            print(f"  Generated {count}/{len(lut)} entries...")

    with open(output_file, 'w') as f:
        f.write(sv_code)

    print(f"\nGenerated {output_file} with {len(lut)} LUT entries")

    print("\nSample GELU values:")
    test_vals = [-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    for val in test_vals:
        int_in = int(round(val * scale))
        if int_in >= min_val and int_in <= max_val:
            int_out = lut[int_in]
            float_out = int_out / scale
            sw_out = gelu(val)
            print(f"  GELU({val:5.1f}) = {float_out:7.4f}  (SW: {sw_out:7.4f}, diff: {abs(float_out-sw_out):.4f})")

if __name__ == "__main__":

    generate_gelu_lut_sv(
        data_width=16,
        frac_width=14,
        output_file="gelu_lut.sv"
    )
