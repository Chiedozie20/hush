module layernorm #(
    parameter DATA_WIDTH = 32,
    parameter FRAC_WIDTH = 16,
    parameter FRAME_SIZE = 384,
    parameter EPSILON = 32'd42950
) (
    input logic clk,
    input logic rst_n,
    input logic signed [DATA_WIDTH-1:0] data_in,
    input logic data_in_valid,
    output logic data_in_ready,
    input logic gamma_wr_en,
    input logic [$clog2(FRAME_SIZE)-1:0] gamma_wr_addr,
    input logic signed [DATA_WIDTH-1:0] gamma_wr_data,
    input logic beta_wr_en,
    input logic [$clog2(FRAME_SIZE)-1:0] beta_wr_addr,
    input logic signed [DATA_WIDTH-1:0] beta_wr_data,
    output logic signed [DATA_WIDTH-1:0] data_out,
    output logic data_out_valid
);

  localparam INDEX_WIDTH = $clog2(FRAME_SIZE);
  localparam COUNT_WIDTH = $clog2(FRAME_SIZE + 1);
  localparam SUM_WIDTH = DATA_WIDTH + COUNT_WIDTH;
  localparam SUMSQ_WIDTH = (2 * DATA_WIDTH) + COUNT_WIDTH;
  localparam VAR_WIDTH = 2 * ((SUMSQ_WIDTH + 1) / 2);
  localparam VAR_INT_WIDTH = VAR_WIDTH - (2 * FRAC_WIDTH);
  localparam STDEV_WIDTH = ((VAR_INT_WIDTH + 1) / 2) + FRAC_WIDTH;
  localparam CENTER_WIDTH = DATA_WIDTH + FRAC_WIDTH + 1;
  localparam NORM_WIDTH = CENTER_WIDTH;
  localparam SCALED_WIDTH = NORM_WIDTH + DATA_WIDTH;
  localparam AFFINE_WIDTH = SCALED_WIDTH + 1;
  localparam STATE_WIDTH = 2;

  localparam [STATE_WIDTH-1:0] STATE_CAPTURE = 2'd0;
  localparam [STATE_WIDTH-1:0] STATE_START_SQRT = 2'd1;
  localparam [STATE_WIDTH-1:0] STATE_WAIT_SQRT = 2'd2;
  localparam [STATE_WIDTH-1:0] STATE_OUTPUT = 2'd3;

  logic [STATE_WIDTH-1:0] state;

  logic signed [DATA_WIDTH-1:0] gamma_mem[0:FRAME_SIZE-1];
  logic signed [DATA_WIDTH-1:0] beta_mem[0:FRAME_SIZE-1];
  logic signed [DATA_WIDTH-1:0] sample_buf[0:FRAME_SIZE-1];

  logic [COUNT_WIDTH-1:0] sample_count;
  logic [INDEX_WIDTH-1:0] output_idx;

  logic signed [SUM_WIDTH-1:0] sum_acc;
  logic [SUMSQ_WIDTH-1:0] sumsq_acc;

  logic signed [SUM_WIDTH-1:0] data_in_ext;
  logic [2*DATA_WIDTH-1:0] square_term;
  logic signed [SUM_WIDTH-1:0] sum_next;
  logic [SUMSQ_WIDTH-1:0] sumsq_next;

  logic signed [DATA_WIDTH-1:0] mean_reg;
  logic signed [2*DATA_WIDTH-1:0] mean_sq_reg;
  logic [VAR_WIDTH-1:0] ex2_reg;
  logic [VAR_WIDTH-1:0] variance_reg;
  logic [VAR_WIDTH-1:0] sqrt_radicand;

  logic sqrt_start;
  logic [STDEV_WIDTH-1:0] sqrt_root;
  logic sqrt_busy;
  logic sqrt_done;

  logic signed [DATA_WIDTH-1:0] sample_cur;
  logic signed [DATA_WIDTH-1:0] gamma_cur;
  logic signed [DATA_WIDTH-1:0] beta_cur;

  logic signed [DATA_WIDTH:0] centered_value;
  logic signed [CENTER_WIDTH-1:0] centered_shifted;
  logic [STDEV_WIDTH-1:0] stddev_value;
  logic signed [CENTER_WIDTH-1:0] centered_ext;
  logic signed [NORM_WIDTH-1:0] norm_value;
  logic signed [NORM_WIDTH-1:0] stddev_divisor_ext;
  logic signed [SCALED_WIDTH-1:0] scaled_value;
  logic signed [AFFINE_WIDTH-1:0] scaled_ext;
  logic signed [AFFINE_WIDTH-1:0] affine_value;

  logic signed [SUM_WIDTH-1:0] mean_next_full;
  logic signed [(2*SUM_WIDTH)-1:0] mean_sq_full;
  logic [SUMSQ_WIDTH-1:0] ex2_next_full;
  logic [VAR_WIDTH-1:0] ex2_next_var;
  logic [VAR_WIDTH-1:0] mean_sq_next_var;
  logic [VAR_WIDTH-1:0] epsilon_ext;
  logic signed [AFFINE_WIDTH-1:0] beta_ext;

  assign data_in_ext = {{COUNT_WIDTH{data_in[DATA_WIDTH-1]}}, data_in};
  assign square_term = data_in * data_in;
  assign sum_next = sum_acc + data_in_ext;
  assign sumsq_next = sumsq_acc + {{COUNT_WIDTH{1'b0}}, square_term};

  assign sample_cur = sample_buf[output_idx];
  assign gamma_cur = gamma_mem[output_idx];
  assign beta_cur = beta_mem[output_idx];

  assign mean_next_full = sum_next / FRAME_SIZE;
  assign mean_sq_full = mean_next_full * mean_next_full;
  assign ex2_next_full = sumsq_next / FRAME_SIZE;
  assign ex2_next_var = {{(VAR_WIDTH - SUMSQ_WIDTH) {1'b0}}, ex2_next_full};
  assign mean_sq_next_var = mean_sq_full[VAR_WIDTH-1:0];
  assign epsilon_ext = {{(VAR_WIDTH - 32) {1'b0}}, EPSILON};

  assign centered_value = sample_cur - mean_reg;
  assign centered_ext = {
    {(CENTER_WIDTH - DATA_WIDTH - 1) {centered_value[DATA_WIDTH]}}, centered_value
  };
  assign centered_shifted = centered_ext <<< FRAC_WIDTH;
  assign stddev_value = (sqrt_root == '0) ? {{(STDEV_WIDTH - 1) {1'b0}}, 1'b1} : sqrt_root;
  assign stddev_divisor_ext = {{(NORM_WIDTH - STDEV_WIDTH) {1'b0}}, stddev_value};
  assign norm_value = centered_shifted / stddev_divisor_ext;
  assign scaled_value = norm_value * gamma_cur;
  assign beta_ext = {{(AFFINE_WIDTH - DATA_WIDTH) {beta_cur[DATA_WIDTH-1]}}, beta_cur};
  assign scaled_ext = {
    {(AFFINE_WIDTH - SCALED_WIDTH) {scaled_value[SCALED_WIDTH-1]}}, scaled_value
  };
  assign affine_value = (scaled_ext >>> FRAC_WIDTH) + beta_ext;

  assign sqrt_radicand = variance_reg;
  assign data_in_ready = (state == STATE_CAPTURE);

  fixed_isqrt #(
      .DATA_WIDTH(VAR_WIDTH),
      .INT_WIDTH(VAR_INT_WIDTH),
      .FRAC_WIDTH(2 * FRAC_WIDTH),
      .ROOT_FRAC_WIDTH(FRAC_WIDTH)
  ) sqrt_core (
      .clk(clk),
      .rst_n(rst_n),
      .start(sqrt_start),
      .radicand(sqrt_radicand),
      .root(sqrt_root),
      .busy(sqrt_busy),
      .done(sqrt_done)
  );

  initial begin
    $dumpfile("layernorm.vcd");
    $dumpvars(0, layernorm);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= STATE_CAPTURE;
      sample_count <= '0;
      output_idx <= '0;
      sum_acc <= '0;
      sumsq_acc <= '0;
      mean_reg <= '0;
      mean_sq_reg <= '0;
      ex2_reg <= '0;
      variance_reg <= '0;
      sqrt_start <= 1'b0;
      data_out <= '0;
      data_out_valid <= 1'b0;
    end else begin
      sqrt_start <= 1'b0;
      data_out_valid <= 1'b0;

      if (gamma_wr_en) begin
        gamma_mem[gamma_wr_addr] <= gamma_wr_data;
      end

      if (beta_wr_en) begin
        beta_mem[beta_wr_addr] <= beta_wr_data;
      end

      case (state)
        STATE_CAPTURE: begin
          if (data_in_valid && data_in_ready) begin
            sample_buf[sample_count] <= data_in;
            sum_acc <= sum_next;
            sumsq_acc <= sumsq_next;

            if (sample_count == FRAME_SIZE - 1) begin
              mean_reg <= mean_next_full[DATA_WIDTH-1:0];
              mean_sq_reg <= mean_sq_full[2*DATA_WIDTH-1:0];
              ex2_reg <= ex2_next_var;

              if (ex2_next_var > mean_sq_next_var) begin
                variance_reg <= ex2_next_var - mean_sq_next_var + epsilon_ext;
              end else begin
                variance_reg <= epsilon_ext;
              end

              sample_count <= '0;
              output_idx <= '0;
              sum_acc <= '0;
              sumsq_acc <= '0;
              state <= STATE_START_SQRT;
            end else begin
              sample_count <= sample_count + {{(COUNT_WIDTH - 1) {1'b0}}, 1'b1};
            end
          end
        end

        STATE_START_SQRT: begin
          sqrt_start <= 1'b1;
          state <= STATE_WAIT_SQRT;
        end

        STATE_WAIT_SQRT: begin
          if (sqrt_done) begin
            output_idx <= '0;
            state <= STATE_OUTPUT;
          end
        end

        STATE_OUTPUT: begin
          data_out <= affine_value[DATA_WIDTH-1:0];
          data_out_valid <= 1'b1;

          if (output_idx == FRAME_SIZE - 1) begin
            output_idx <= '0;
            state <= STATE_CAPTURE;
          end else begin
            output_idx <= output_idx + {{(INDEX_WIDTH - 1) {1'b0}}, 1'b1};
          end
        end

        default: begin
          state <= STATE_CAPTURE;
        end
      endcase
    end
  end

endmodule


// ==============================================
// === yosys statistics ===
// ==============================================

// === layernorm ===

//         +----------Local Count, excluding submodules.
//         | 
//     12552 wires
//    208600 wire bits
//      1202 public wires
//     38835 public wire bits
//        13 ports
//       153 port bits
//      8965 cells
//         6   $add
//        18   $adff
//      1152   $dff
//         3   $div
//      2314   $eq
//         1   $gt
//         1   $logic_and
//       406   $logic_not
//         3   $mul
//      3882   $mux
//      1170   $not
//         7   $pmux
//         2   $sub
//         1 submodules
//         1   $paramod$e01021525ae5a5435257777fb8e448dbab22288a\fixed_isqrt

// === $paramod$e01021525ae5a5435257777fb8e448dbab22288a\fixed_isqrt ===

//         +----------Local Count, excluding submodules.
//         | 
//        75 wires
//      1137 wire bits
//        16 public wires
//       500 public wire bits
//         7 ports
//       116 port bits
//        36 cells
//         6   $adff
//         1   $eq
//         1   $ge
//         1   $logic_and
//         2   $logic_not
//        17   $mux
//         6   $not
//         2   $sub

// === design hierarchy ===

//         +----------Count including submodules.
//         | 
//      9001 layernorm
//        36 $paramod$e01021525ae5a5435257777fb8e448dbab22288a\fixed_isqrt

//         +----------Count including submodules.
//         | 
//     12627 wires
//    209737 wire bits
//      1218 public wires
//     39335 public wire bits
//        20 ports
//       269 port bits
//         - memories
//         - memory bits
//         - processes
//      9001 cells
//         6   $add
//        24   $adff
//      1152   $dff
//         3   $div
//      2315   $eq
//         1   $ge
//         1   $gt
//         2   $logic_and
//       408   $logic_not
//         3   $mul
//      3899   $mux
//      1176   $not
//         7   $pmux
//         4   $sub
//         1 submodules
//         1   $paramod$e01021525ae5a5435257777fb8e448dbab22288a\fixed_isqrt
