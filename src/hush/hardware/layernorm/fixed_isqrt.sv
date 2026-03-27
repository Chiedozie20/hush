module fixed_isqrt #(
    parameter DATA_WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16,
    parameter ROOT_FRAC_WIDTH = FRAC_WIDTH
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
  localparam RADICAND_FRAC_WIDTH = 2 * ROOT_FRAC_WIDTH;
  localparam LEFT_SHIFT = (RADICAND_FRAC_WIDTH >= FRAC_WIDTH)
        ? (RADICAND_FRAC_WIDTH - FRAC_WIDTH) : 0;
  localparam RIGHT_SHIFT = (RADICAND_FRAC_WIDTH < FRAC_WIDTH)
        ? (FRAC_WIDTH - RADICAND_FRAC_WIDTH) : 0;
  localparam PRESCALE_WIDTH = DATA_WIDTH + LEFT_SHIFT;
  localparam VALUE_WIDTH = 2 * ((PRESCALE_WIDTH + 1) / 2);
  localparam INTERNAL_ROOT_WIDTH = VALUE_WIDTH / 2;
  localparam COUNT_WIDTH = $clog2(INTERNAL_ROOT_WIDTH + 1);
  localparam REM_WIDTH = INTERNAL_ROOT_WIDTH + 2;

  logic [PRESCALE_WIDTH-1:0] scaled_radicand;
  logic [VALUE_WIDTH-1:0] radicand_aligned;
  logic [VALUE_WIDTH-1:0] radicand_shift;
  logic [INTERNAL_ROOT_WIDTH-1:0] root_reg;
  logic [REM_WIDTH-1:0] remainder_reg;
  logic [COUNT_WIDTH-1:0] iter_count;

  logic [1:0] pair_bits;
  logic [REM_WIDTH-1:0] remainder_shift;
  logic [REM_WIDTH-1:0] trial_value;

  generate
    if (RADICAND_FRAC_WIDTH >= FRAC_WIDTH) begin : gen_left_shift
      assign scaled_radicand = {radicand, {LEFT_SHIFT{1'b0}}};
    end else begin : gen_right_shift
      assign scaled_radicand = radicand >> RIGHT_SHIFT;
    end
  endgenerate

  assign radicand_aligned = {{(VALUE_WIDTH - PRESCALE_WIDTH) {1'b0}}, scaled_radicand};
  assign pair_bits = radicand_shift[VALUE_WIDTH-1-:2];
  assign remainder_shift = {remainder_reg[REM_WIDTH-3:0], pair_bits};
  assign trial_value = {{(REM_WIDTH - INTERNAL_ROOT_WIDTH - 2) {1'b0}}, root_reg, 2'b01};
  assign root = root_reg[ROOT_WIDTH-1:0];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      radicand_shift <= '0;
      root_reg <= '0;
      remainder_reg <= '0;
      iter_count <= '0;
      busy <= 1'b0;
      done <= 1'b0;
    end else begin
      done <= 1'b0;

      if (start && !busy) begin
        radicand_shift <= radicand_aligned;
        root_reg <= '0;
        remainder_reg <= '0;
        iter_count <= INTERNAL_ROOT_WIDTH[COUNT_WIDTH-1:0];
        busy <= 1'b1;
      end else if (busy) begin
        radicand_shift <= {radicand_shift[VALUE_WIDTH-3:0], 2'b00};

        if (remainder_shift >= trial_value) begin
          remainder_reg <= remainder_shift - trial_value;
          root_reg <= {root_reg[INTERNAL_ROOT_WIDTH-2:0], 1'b1};
        end else begin
          remainder_reg <= remainder_shift;
          root_reg <= {root_reg[INTERNAL_ROOT_WIDTH-2:0], 1'b0};
        end

        if (iter_count == {{(COUNT_WIDTH - 1) {1'b0}}, 1'b1}) begin
          iter_count <= '0;
          busy <= 1'b0;
          done <= 1'b1;
        end else begin
          iter_count <= iter_count - {{(COUNT_WIDTH - 1) {1'b0}}, 1'b1};
        end
      end
    end
  end

endmodule
