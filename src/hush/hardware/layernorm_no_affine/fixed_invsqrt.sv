module fixed_invsqrt #(
    parameter DATA_WIDTH = 28,
    parameter INT_WIDTH = 11,
    parameter FRAC_WIDTH = 17,
    parameter OUT_WIDTH = 28,
    parameter OUT_FRAC_WIDTH = 17,
    parameter LUT_ADDR_WIDTH = 16,
    parameter LUT_GUARD_BITS = 8
) (
    input logic clk,
    input logic rst_n,
    input logic start,
    input logic [DATA_WIDTH-1:0] radicand,
    output logic [OUT_WIDTH-1:0] root,
    output logic busy,
    output logic done
);

  localparam LUT_VALUE_WIDTH = OUT_WIDTH + LUT_GUARD_BITS;  // table value width
  localparam LUT_WORD_WIDTH = 2 * LUT_VALUE_WIDTH;  // packed entry width
  localparam LUT_DEPTH = 1 << LUT_ADDR_WIDTH;  // table depth
  localparam MSB_INDEX_WIDTH = (DATA_WIDTH > 1) ? $clog2(DATA_WIDTH) : 1;  // lead-one index
  localparam SHIFT_WIDTH = $clog2(DATA_WIDTH + FRAC_WIDTH + OUT_FRAC_WIDTH + 1) + 2;  // shift width
  localparam SCALED_WIDTH = LUT_VALUE_WIDTH + DATA_WIDTH;  // rescale width
  localparam signed [SHIFT_WIDTH-1:0] FRAC_WIDTH_CONST = SHIFT_WIDTH'(FRAC_WIDTH);  // frac constant
  localparam FRAC_WIDTH_ODD = FRAC_WIDTH_CONST[0];  // frac parity

  typedef enum logic [1:0] {
    STATE_IDLE,
    STATE_LOOKUP,
    STATE_SCALE,
    STATE_OUTPUT
  } state_t;

  (* rom_style = "block", ram_style = "block" *)
  logic [LUT_WORD_WIDTH-1:0] lut_mem[0:LUT_DEPTH-1];

  state_t current_state;  // FSM state
  state_t next_state;  // FSM next
  (* max_fanout = 64 *) logic [LUT_ADDR_WIDTH-1:0] lut_addr_reg;  // table address
  logic odd_exp_reg;  // odd exponent
  logic radicand_nonzero_reg;  // zero flag
  logic signed [SHIFT_WIDTH-1:0] scale_shift_reg;  // exponent shift
  logic [LUT_WORD_WIDTH-1:0] lut_word_reg;  // table word
  logic signed [SCALED_WIDTH-1:0] scaled_value_reg;  // scaled result

  logic radicand_nonzero;  // input nonzero
  logic [MSB_INDEX_WIDTH-1:0] lead_one_idx_in;  // leading one
  logic [LUT_ADDR_WIDTH-1:0] lut_addr_in;  // next address
  logic odd_exp_in;  // next parity
  logic signed [SHIFT_WIDTH-1:0] scale_shift_in;  // next shift
  logic signed [SHIFT_WIDTH-1:0] lead_one_idx_signed;  // signed index
  logic [DATA_WIDTH+LUT_ADDR_WIDTH-1:0] normalized_value;  // normalized input
  logic [DATA_WIDTH+LUT_ADDR_WIDTH-1:0] radicand_ext;  // widened input

  logic [LUT_VALUE_WIDTH-1:0] even_value;  // even lookup
  logic [LUT_VALUE_WIDTH-1:0] odd_value;  // odd lookup
  logic [LUT_VALUE_WIDTH-1:0] lut_value_selected;  // selected lookup
  logic signed [SCALED_WIDTH-1:0] scaled_value_next;  // next scaled value
  logic [OUT_WIDTH-1:0] scaled_root;  // output root

  integer bit_idx;

  initial begin
    $readmemh("fixed_invsqrt_lut.mem", lut_mem);
  end

  always_comb begin
    next_state = current_state;

    unique case (current_state)
      STATE_IDLE: begin
        if (start) begin
          next_state = STATE_LOOKUP;
        end
      end

      STATE_LOOKUP: begin
        next_state = STATE_SCALE;
      end

      STATE_SCALE: begin
        next_state = STATE_OUTPUT;
      end

      STATE_OUTPUT: begin
        next_state = STATE_IDLE;
      end

      default: begin
        next_state = STATE_IDLE;
      end
    endcase
  end

  always_comb begin
    // Normalize input for lookup
    radicand_ext = {{LUT_ADDR_WIDTH{1'b0}}, radicand};
    radicand_nonzero = (radicand != '0);
    lead_one_idx_in = '0;
    for (bit_idx = 0; bit_idx < DATA_WIDTH; bit_idx = bit_idx + 1) begin
      if (radicand[bit_idx]) begin
        lead_one_idx_in = bit_idx[MSB_INDEX_WIDTH-1:0];
      end
    end

    if (lead_one_idx_in < LUT_ADDR_WIDTH) begin
      normalized_value = radicand_ext << (LUT_ADDR_WIDTH - lead_one_idx_in);
    end else begin
      normalized_value = radicand_ext >> (lead_one_idx_in - LUT_ADDR_WIDTH);
    end

    lut_addr_in = normalized_value[LUT_ADDR_WIDTH-1:0];
    odd_exp_in = lead_one_idx_in[0] ^ FRAC_WIDTH_ODD;
    lead_one_idx_signed = $signed({{(SHIFT_WIDTH - MSB_INDEX_WIDTH) {1'b0}}, lead_one_idx_in});
    scale_shift_in = -(($signed(lead_one_idx_signed) - $signed(FRAC_WIDTH_CONST)) >>> 1);
  end

  always_comb begin
    // Lookup decode and rescale
    even_value = lut_word_reg[LUT_VALUE_WIDTH-1:0];
    odd_value = lut_word_reg[LUT_WORD_WIDTH-1:LUT_VALUE_WIDTH];
    lut_value_selected = odd_exp_reg ? odd_value : even_value;

    scaled_value_next = {{(SCALED_WIDTH - LUT_VALUE_WIDTH) {1'b0}}, lut_value_selected};
    if (scale_shift_reg > 0) begin
      scaled_value_next = scaled_value_next <<< scale_shift_reg;
    end else if (scale_shift_reg < 0) begin
      scaled_value_next = scaled_value_next >>> (-scale_shift_reg);
    end

    if (!radicand_nonzero_reg) begin
      scaled_root = {OUT_WIDTH{1'b1}};
    end else begin
      scaled_root = scaled_value_reg[OUT_WIDTH+LUT_GUARD_BITS-1:LUT_GUARD_BITS];
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      current_state <= STATE_IDLE;
      lut_addr_reg <= '0;
      odd_exp_reg <= 1'b0;
      radicand_nonzero_reg <= 1'b0;
      scale_shift_reg <= '0;
      lut_word_reg <= '0;
      scaled_value_reg <= '0;
      root <= '0;
      busy <= 1'b0;
      done <= 1'b0;
    end else begin
      current_state <= next_state;
      done <= 1'b0;

      unique case (current_state)
        STATE_IDLE: begin
          busy <= 1'b0;
          if (start) begin
            lut_addr_reg <= lut_addr_in;
            odd_exp_reg <= odd_exp_in;
            radicand_nonzero_reg <= radicand_nonzero;
            scale_shift_reg <= scale_shift_in;
            busy <= 1'b1;
          end
        end

        STATE_LOOKUP: begin
          lut_word_reg <= lut_mem[lut_addr_reg];
        end

        STATE_SCALE: begin
          scaled_value_reg <= scaled_value_next;
        end

        STATE_OUTPUT: begin
          root <= scaled_root;
          busy <= 1'b0;
          done <= 1'b1;
        end
      endcase
    end
  end

endmodule
