module fixed_isqrt #(
    parameter DATA_WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16,
    parameter ROOT_FRAC_WIDTH = FRAC_WIDTH,
    parameter LUT_ADDR_WIDTH = 10,
    parameter LUT_INTERP_BITS = 8,
    parameter LUT_GUARD_BITS = 10
) (
    input logic clk,
    input logic rst_n,
    input logic start,
    input logic [DATA_WIDTH-1:0] radicand,
    output logic [(((INT_WIDTH + 1) / 2) + ROOT_FRAC_WIDTH)-1:0] root,
    output logic busy,
    output logic done
);

  localparam ROOT_INT_WIDTH = (INT_WIDTH + 1) / 2;
  localparam ROOT_WIDTH = ROOT_INT_WIDTH + ROOT_FRAC_WIDTH;
  localparam LUT_VALUE_WIDTH = ROOT_FRAC_WIDTH + LUT_GUARD_BITS + 2;
  localparam LUT_WORD_WIDTH = 4 * LUT_VALUE_WIDTH;
  localparam LUT_DEPTH = 1 << LUT_ADDR_WIDTH;
  localparam TOTAL_MANT_BITS = LUT_ADDR_WIDTH + LUT_INTERP_BITS;
  localparam MSB_INDEX_WIDTH = (DATA_WIDTH > 1) ? $clog2(DATA_WIDTH) : 1;
  localparam SHIFT_WIDTH = $clog2(DATA_WIDTH + FRAC_WIDTH + ROOT_FRAC_WIDTH + 1) + 1;
  localparam INTERP_MULT_WIDTH = LUT_VALUE_WIDTH + LUT_INTERP_BITS;
  localparam SCALED_WIDTH = ROOT_WIDTH + LUT_GUARD_BITS;
  localparam [SHIFT_WIDTH-1:0] FRAC_WIDTH_CONST = SHIFT_WIDTH'(FRAC_WIDTH);
  localparam FRAC_WIDTH_ODD = FRAC_WIDTH_CONST[0];
  localparam [INTERP_MULT_WIDTH-1:0] INTERP_ROUND = ({{(INTERP_MULT_WIDTH-1){1'b0}}, 1'b1}) << (LUT_INTERP_BITS - 1);

  localparam [1:0] STATE_IDLE = 2'd0;
  localparam [1:0] STATE_LOOKUP = 2'd1;
  localparam [1:0] STATE_OUTPUT = 2'd2;

  (* rom_style = "block", ram_style = "block" *)
  logic [LUT_WORD_WIDTH-1:0] lut_mem[0:LUT_DEPTH-1];

  logic [1:0] state;
  logic [LUT_ADDR_WIDTH-1:0] lut_addr_reg;
  logic [LUT_INTERP_BITS-1:0] frac_reg;
  logic odd_exp_reg;
  logic signed [SHIFT_WIDTH-1:0] scale_shift_reg;
  logic [LUT_WORD_WIDTH-1:0] lut_word_reg;

  logic [MSB_INDEX_WIDTH-1:0] lead_one_idx_in;
  logic [TOTAL_MANT_BITS-1:0] mantissa_bits_in;
  logic [LUT_ADDR_WIDTH-1:0] lut_addr_in;
  logic [LUT_INTERP_BITS-1:0] frac_in;
  logic odd_exp_in;
  logic signed [SHIFT_WIDTH-1:0] scale_shift_in;
  logic radicand_nonzero;

  logic [LUT_VALUE_WIDTH-1:0] even_base;
  logic [LUT_VALUE_WIDTH-1:0] even_delta;
  logic [LUT_VALUE_WIDTH-1:0] odd_base;
  logic [LUT_VALUE_WIDTH-1:0] odd_delta;
  logic [LUT_VALUE_WIDTH-1:0] even_interp;
  logic [LUT_VALUE_WIDTH-1:0] odd_interp;
  logic [LUT_VALUE_WIDTH-1:0] interp_value;
  logic [SCALED_WIDTH-1:0] scaled_value;
  logic [ROOT_WIDTH-1:0] scaled_root;
  logic signed [SHIFT_WIDTH-1:0] lead_one_idx_signed;

  integer bit_idx;

  function automatic [TOTAL_MANT_BITS-1:0] extract_mantissa_bits(
      input logic [DATA_WIDTH-1:0] value, input logic [MSB_INDEX_WIDTH-1:0] lead_idx);
    logic [DATA_WIDTH-1:0] normalized_value;
    begin
      if (lead_idx < TOTAL_MANT_BITS) begin
        normalized_value = value << (TOTAL_MANT_BITS - lead_idx);
      end else begin
        normalized_value = value >> (lead_idx - TOTAL_MANT_BITS);
      end
      extract_mantissa_bits = normalized_value[TOTAL_MANT_BITS-1:0];
    end
  endfunction

  initial begin
    $readmemh("fixed_isqrt_lut.mem", lut_mem);
  end

  always_comb begin
    radicand_nonzero = (radicand != '0);
    lead_one_idx_in = '0;
    for (bit_idx = 0; bit_idx < DATA_WIDTH; bit_idx = bit_idx + 1) begin
      if (radicand[bit_idx]) begin
        lead_one_idx_in = bit_idx[MSB_INDEX_WIDTH-1:0];
      end
    end

    mantissa_bits_in = extract_mantissa_bits(radicand, lead_one_idx_in);
    lut_addr_in = mantissa_bits_in[TOTAL_MANT_BITS-1-:LUT_ADDR_WIDTH];
    frac_in = mantissa_bits_in[LUT_INTERP_BITS-1:0];
    odd_exp_in = lead_one_idx_in[0] ^ FRAC_WIDTH_ODD;
    lead_one_idx_signed = $signed({{(SHIFT_WIDTH - MSB_INDEX_WIDTH) {1'b0}}, lead_one_idx_in});
    scale_shift_in = (lead_one_idx_signed - $signed(FRAC_WIDTH_CONST)) >>> 1;
  end

  always_comb begin
    logic [INTERP_MULT_WIDTH-1:0] even_mult;
    logic [INTERP_MULT_WIDTH-1:0] odd_mult;
    logic [INTERP_MULT_WIDTH-1:0] even_interp_wide;
    logic [INTERP_MULT_WIDTH-1:0] odd_interp_wide;
    logic [SCALED_WIDTH-1:0] rounded_scaled_value;
    even_base = lut_word_reg[(1*LUT_VALUE_WIDTH)-1-:LUT_VALUE_WIDTH];
    even_delta = lut_word_reg[(2*LUT_VALUE_WIDTH)-1-:LUT_VALUE_WIDTH];
    odd_base = lut_word_reg[(3*LUT_VALUE_WIDTH)-1-:LUT_VALUE_WIDTH];
    odd_delta = lut_word_reg[(4*LUT_VALUE_WIDTH)-1-:LUT_VALUE_WIDTH];

    even_mult = even_delta * frac_reg;
    odd_mult = odd_delta * frac_reg;
    even_interp_wide = {{(INTERP_MULT_WIDTH-LUT_VALUE_WIDTH){1'b0}}, even_base}
            + ((even_mult + INTERP_ROUND) >> LUT_INTERP_BITS);
    odd_interp_wide = {{(INTERP_MULT_WIDTH-LUT_VALUE_WIDTH){1'b0}}, odd_base}
            + ((odd_mult + INTERP_ROUND) >> LUT_INTERP_BITS);
    even_interp = even_interp_wide[LUT_VALUE_WIDTH-1:0];
    odd_interp = odd_interp_wide[LUT_VALUE_WIDTH-1:0];
    interp_value = odd_exp_reg ? odd_interp : even_interp;

    scaled_value = {{(SCALED_WIDTH - LUT_VALUE_WIDTH) {1'b0}}, interp_value};
    if (scale_shift_reg > 0) begin
      scaled_value = scaled_value << scale_shift_reg;
    end else if (scale_shift_reg < 0) begin
      scaled_value = scaled_value >> (-scale_shift_reg);
    end

    rounded_scaled_value = scaled_value + ({{(SCALED_WIDTH-1){1'b0}}, 1'b1} << (LUT_GUARD_BITS - 1));
    scaled_root = rounded_scaled_value[SCALED_WIDTH-1:LUT_GUARD_BITS];
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= STATE_IDLE;
      lut_addr_reg <= '0;
      frac_reg <= '0;
      odd_exp_reg <= 1'b0;
      scale_shift_reg <= '0;
      lut_word_reg <= '0;
      root <= '0;
      busy <= 1'b0;
      done <= 1'b0;
    end else begin
      done <= 1'b0;

      case (state)
        STATE_IDLE: begin
          busy <= 1'b0;

          if (start) begin
            if (radicand_nonzero) begin
              lut_addr_reg <= lut_addr_in;
              frac_reg <= frac_in;
              odd_exp_reg <= odd_exp_in;
              scale_shift_reg <= scale_shift_in;
              busy <= 1'b1;
              state <= STATE_LOOKUP;
            end else begin
              root <= '0;
              done <= 1'b1;
            end
          end
        end

        STATE_LOOKUP: begin
          lut_word_reg <= lut_mem[lut_addr_reg];
          state <= STATE_OUTPUT;
        end

        STATE_OUTPUT: begin
          root <= scaled_root;
          busy <= 1'b0;
          done <= 1'b1;
          state <= STATE_IDLE;
        end

        default: begin
          state <= STATE_IDLE;
          busy <= 1'b0;
        end
      endcase
    end
  end

endmodule
