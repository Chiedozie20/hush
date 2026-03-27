module layernorm #(
    parameter DATA_WIDTH = 28,
    parameter FRAC_WIDTH = 17,
    parameter FRAME_SIZE = 32,
    parameter EPSILON = 32'd171799
) (
    input logic clk,
    input logic rst_n,
    input logic signed [DATA_WIDTH-1:0] data_in,
    input logic data_in_valid,
    output logic data_in_ready,
    output logic signed [DATA_WIDTH-1:0] data_out,
    output logic data_out_valid
);

  localparam INDEX_WIDTH = $clog2(FRAME_SIZE);  // frame index
  localparam COUNT_WIDTH = $clog2(FRAME_SIZE + 1);  // sample count
  localparam FRAME_SHIFT = $clog2(FRAME_SIZE);  // divide by frame
  localparam [INDEX_WIDTH-1:0] LAST_INDEX = INDEX_WIDTH'(FRAME_SIZE - 1);  // final sample
  localparam SUM_WIDTH = DATA_WIDTH + COUNT_WIDTH;  // sum width
  localparam SUMSQ_WIDTH = (2 * DATA_WIDTH) + COUNT_WIDTH;  // sumsq width
  localparam VAR_WIDTH = SUMSQ_WIDTH;  // variance width
  localparam INV_STD_WIDTH = DATA_WIDTH;  // inv-std width
  localparam CENTER_WIDTH = DATA_WIDTH + 1;  // centered width
  localparam PRODUCT_WIDTH = CENTER_WIDTH + INV_STD_WIDTH;  // multiply width

  typedef enum logic [3:0] {
    STATE_CAPTURE,
    STATE_STATS_MEAN,
    STATE_STATS_MOMENTS,
    STATE_STATS_VAR,
    STATE_START_INV,
    STATE_WAIT_INV,
    STATE_PREP_OUTPUT,
    STATE_READ_SAMPLE,
    STATE_CENTER,
    STATE_SCALE,
    STATE_OUTPUT
  } state_t;

  state_t current_state;  // FSM state
  state_t next_state;  // FSM next

  (* ram_style = "block", ramstyle = "M20K" *)
  logic signed [DATA_WIDTH-1:0] sample_buf[0:FRAME_SIZE-1];  // frame buffer

  logic [COUNT_WIDTH-1:0] sample_count;  // input count
  logic [INDEX_WIDTH-1:0] output_idx;  // output index
  logic [INDEX_WIDTH-1:0] sample_rd_addr;  // buffer address

  logic signed [SUM_WIDTH-1:0] sum_acc;  // running sum
  logic [SUMSQ_WIDTH-1:0] sumsq_acc;  // running sumsq

  logic signed [SUM_WIDTH-1:0] data_in_ext;  // sign-extended input
  logic signed [SUM_WIDTH-1:0] sum_next;  // next sum
  logic [2*DATA_WIDTH-1:0] square_term;  // input squared
  logic [SUMSQ_WIDTH-1:0] sumsq_next;  // next sumsq

  logic signed [SUM_WIDTH-1:0] frame_sum_reg;  // latched sum
  logic [SUMSQ_WIDTH-1:0] frame_sumsq_reg;  // latched sumsq
  logic signed [DATA_WIDTH-1:0] mean_reg;  // frame mean
  logic signed [DATA_WIDTH-1:0] mean_out_reg;  // output mean
  logic [VAR_WIDTH-1:0] ex2_reg;  // E[x^2]
  logic [VAR_WIDTH-1:0] mean_sq_reg;  // mean squared
  logic [VAR_WIDTH-1:0] variance_reg;  // variance
  logic [VAR_WIDTH-1:0] inv_radicand;  // invsqrt input

  logic inv_start;  // invsqrt start
  logic [INV_STD_WIDTH-1:0] inv_stddev;  // invsqrt result
  logic inv_done;  // invsqrt done

  logic signed [SUM_WIDTH-1:0] mean_frame_full;  // shifted mean
  logic signed [(2*SUM_WIDTH)-1:0] mean_sq_full;  // full mean^2
  logic [SUMSQ_WIDTH-1:0] ex2_frame_full;  // shifted E[x^2]
  logic [VAR_WIDTH-1:0] ex2_frame_var;  // aligned E[x^2]
  logic [VAR_WIDTH-1:0] mean_sq_var;  // aligned mean^2
  logic [VAR_WIDTH-1:0] epsilon_ext;  // aligned epsilon

  logic signed [DATA_WIDTH-1:0] sample_reg;  // output sample
  logic output_last_reg;  // final output
  logic [INV_STD_WIDTH-1:0] inv_stddev_reg;  // latched inv-std
  logic signed [CENTER_WIDTH-1:0] centered_reg;  // sample minus mean
  logic signed [PRODUCT_WIDTH-1:0] scaled_value_reg;  // normalized product
  logic signed [DATA_WIDTH-1:0] norm_out_next;  // output value

  logic signed [DATA_WIDTH:0] centered_value_next;  // next centered
  logic signed [CENTER_WIDTH-1:0] centered_next;  // resized centered
  logic signed [PRODUCT_WIDTH-1:0] scaled_value_next;  // next product

  always_comb begin
    // Capture datapath
    data_in_ext = {{COUNT_WIDTH{data_in[DATA_WIDTH-1]}}, data_in};
    sum_next = sum_acc + data_in_ext;
    square_term = data_in * data_in;
    sumsq_next = sumsq_acc + {{COUNT_WIDTH{1'b0}}, square_term};

    // Statistics datapath
    mean_frame_full = frame_sum_reg >>> FRAME_SHIFT;
    mean_sq_full = mean_frame_full * mean_frame_full;
    ex2_frame_full = frame_sumsq_reg >> FRAME_SHIFT;
    ex2_frame_var = {{(VAR_WIDTH - SUMSQ_WIDTH) {1'b0}}, ex2_frame_full};
    mean_sq_var = mean_sq_full[VAR_WIDTH-1:0];
    epsilon_ext = {{(VAR_WIDTH - 32) {1'b0}}, EPSILON};

    // Output datapath
    centered_value_next = sample_reg - mean_out_reg;
    centered_next = centered_value_next;
    scaled_value_next = centered_reg * $signed(inv_stddev_reg);
    norm_out_next = DATA_WIDTH'($signed(scaled_value_reg >>> FRAC_WIDTH));

    // Control-facing signals
    inv_radicand = variance_reg;
    data_in_ready = (current_state == STATE_CAPTURE);
  end

  always_comb begin
    next_state = current_state;

    unique case (current_state)
      STATE_CAPTURE: begin
        if (data_in_valid && data_in_ready) begin
          if (sample_count == FRAME_SIZE - 1) begin
            next_state = STATE_STATS_MEAN;
          end
        end
      end

      STATE_STATS_MEAN: begin
        next_state = STATE_STATS_MOMENTS;
      end

      STATE_STATS_MOMENTS: begin
        next_state = STATE_STATS_VAR;
      end

      STATE_STATS_VAR: begin
        next_state = STATE_START_INV;
      end

      STATE_START_INV: begin
        next_state = STATE_WAIT_INV;
      end

      STATE_WAIT_INV: begin
        if (inv_done) begin
          next_state = STATE_PREP_OUTPUT;
        end
      end

      STATE_PREP_OUTPUT: begin
        next_state = STATE_READ_SAMPLE;
      end

      STATE_READ_SAMPLE: begin
        next_state = STATE_CENTER;
      end

      STATE_CENTER: begin
        next_state = STATE_SCALE;
      end

      STATE_SCALE: begin
        next_state = STATE_OUTPUT;
      end

      STATE_OUTPUT: begin
        if (output_last_reg) begin
          next_state = STATE_CAPTURE;
        end else begin
          next_state = STATE_PREP_OUTPUT;
        end
      end

      default: begin
        next_state = STATE_CAPTURE;
      end
    endcase
  end

  initial begin
    $dumpfile("layernorm.vcd");
    $dumpvars(0, layernorm);
  end

  fixed_invsqrt #(
      .DATA_WIDTH(VAR_WIDTH),
      .INT_WIDTH(VAR_WIDTH - (2 * FRAC_WIDTH)),
      .FRAC_WIDTH(2 * FRAC_WIDTH),
      .OUT_WIDTH(INV_STD_WIDTH),
      .OUT_FRAC_WIDTH(FRAC_WIDTH)
  ) invsqrt_core (
      .clk(clk),
      .rst_n(rst_n),
      .start(inv_start),
      .radicand(inv_radicand),
      .root(inv_stddev),
      .busy(),
      .done(inv_done)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      current_state <= STATE_CAPTURE;
      sample_count <= '0;
      output_idx <= '0;
      sample_rd_addr <= '0;
      sum_acc <= '0;
      sumsq_acc <= '0;
      frame_sum_reg <= '0;
      frame_sumsq_reg <= '0;
      mean_reg <= '0;
      mean_out_reg <= '0;
      ex2_reg <= '0;
      mean_sq_reg <= '0;
      variance_reg <= '0;
      sample_reg <= '0;
      output_last_reg <= 1'b0;
      inv_stddev_reg <= '0;
      centered_reg <= '0;
      scaled_value_reg <= '0;
      inv_start <= 1'b0;
      data_out <= '0;
      data_out_valid <= 1'b0;
    end else begin
      current_state <= next_state;
      inv_start <= 1'b0;
      data_out_valid <= 1'b0;

      unique case (current_state)
        STATE_CAPTURE: begin
          if (data_in_valid && data_in_ready) begin
            sum_acc <= sum_next;
            sumsq_acc <= sumsq_next;

            if (sample_count == FRAME_SIZE - 1) begin
              frame_sum_reg <= sum_next;
              frame_sumsq_reg <= sumsq_next;
              sample_count <= '0;
              output_idx <= '0;
              sum_acc <= '0;
              sumsq_acc <= '0;
            end else begin
              sample_count <= sample_count + {{(COUNT_WIDTH - 1) {1'b0}}, 1'b1};
            end
          end
        end

        STATE_STATS_MEAN: begin
          mean_reg <= mean_frame_full[DATA_WIDTH-1:0];
        end

        STATE_STATS_MOMENTS: begin
          ex2_reg <= ex2_frame_var;
          mean_sq_reg <= mean_sq_var;
        end

        STATE_STATS_VAR: begin
          if (ex2_reg > mean_sq_reg) begin
            variance_reg <= ex2_reg - mean_sq_reg + epsilon_ext;
          end else begin
            variance_reg <= epsilon_ext;
          end
        end

        STATE_START_INV: begin
          inv_start <= 1'b1;
        end

        STATE_WAIT_INV: begin
          if (inv_done) begin
            mean_out_reg <= mean_reg;
            inv_stddev_reg <= inv_stddev;
            output_idx <= '0;
          end
        end

        STATE_PREP_OUTPUT: begin
          sample_rd_addr <= output_idx;
          output_last_reg <= (output_idx == LAST_INDEX);

          if (output_idx == LAST_INDEX) begin
            output_idx <= '0;
          end else begin
            output_idx <= output_idx + {{(INDEX_WIDTH - 1) {1'b0}}, 1'b1};
          end
        end

        STATE_READ_SAMPLE: begin
          sample_reg <= sample_buf[sample_rd_addr];
        end

        STATE_CENTER: begin
          centered_reg <= centered_next;
        end

        STATE_SCALE: begin
          scaled_value_reg <= scaled_value_next;
        end

        STATE_OUTPUT: begin
          data_out <= norm_out_next;
          data_out_valid <= 1'b1;
        end
      endcase
    end
  end

  always_ff @(posedge clk) begin
    if (data_in_valid && data_in_ready) begin
      sample_buf[sample_count[INDEX_WIDTH-1:0]] <= data_in;
    end
  end

endmodule
