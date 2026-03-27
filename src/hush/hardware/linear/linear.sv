module linear #(
    parameter int DATA_WIDTH = 16,
    parameter int MAX_IN_CHANNELS = 32,
    parameter int MAX_OUT_CHANNELS = 32,
    parameter int MAX_SEQ_LEN = 16,
    parameter int NUM_MAC_UNITS = 32
) (
    input logic clk,
    input logic rst_n,
    input logic start,
    output logic done,
    output logic busy,

    input logic [15:0] in_channels_cfg,  // Runtime: actual input channels
    input logic [15:0] out_channels_cfg,  // Runtime: actual output channels

    input logic [DATA_WIDTH-1:0] weight_in,
    input logic [2*DATA_WIDTH-1:0] bias_in,
    input logic weight_valid,
    input logic bias_valid,
    input logic [15:0] weight_out_ch,
    input logic [15:0] weight_in_ch,
    input logic [15:0] bias_out_ch,

    // Data input interface
    input logic [DATA_WIDTH-1:0] data_in,
    input logic data_in_valid,
    output logic data_in_ready,
    input logic [15:0] in_channel_idx,
    input logic [15:0] in_pos_idx,

    output logic [2*DATA_WIDTH-1:0] data_out,
    output logic data_out_valid,
    input logic data_out_ready,
    output logic [15:0] out_channel_idx,
    output logic [15:0] out_pos_idx,

    input logic [15:0] input_length
);

  localparam int MAX_MAC_CYCLES = MAX_IN_CHANNELS;
  logic [31:0] mac_cycles;
  assign mac_cycles = {16'd0, in_channels_cfg};

  typedef enum logic [2:0] {
    IDLE,
    LOAD_WEIGHTS,
    LOAD_BIAS,
    COMPUTE,
    OUTPUT,  // dedicated output-drain state
    DONE_STATE
  } state_t;

  state_t state, next_state;

  // Storage
  logic [DATA_WIDTH-1:0] weights[MAX_OUT_CHANNELS][MAX_IN_CHANNELS];
  logic [2*DATA_WIDTH-1:0] biases[MAX_OUT_CHANNELS];  // 32-bit to match accumulator domain
  logic [DATA_WIDTH-1:0] input_data[MAX_IN_CHANNELS][MAX_SEQ_LEN];

  // Control signals
  logic weights_loaded;
  logic bias_loaded;
  logic input_loaded;
  logic [31:0] input_count;  // must hold IN_CHANNELS * MAX_SEQ_LEN (e.g. 240000)
  logic [15:0] out_pos;
  logic [15:0] output_ch_counter;

  logic [15:0] output_length;
  assign output_length = input_length;

  // MAC engine signals
  logic signed [2*DATA_WIDTH-1:0] mac_accumulators[NUM_MAC_UNITS];
  logic [15:0] mac_in_ch[NUM_MAC_UNITS];
  logic mac_computing;
  logic [15:0] mac_cycle_counter;
  logic mac_done;  // pulses high for one cycle when MAC finishes

  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n) state <= IDLE;
    else state <= next_state;
  end

  always_comb begin
    next_state = state;
    case (state)
      IDLE: begin
        if (start) next_state = LOAD_WEIGHTS;
      end

      LOAD_WEIGHTS: begin
        if (weights_loaded) next_state = LOAD_BIAS;
      end

      LOAD_BIAS: begin
        if (bias_loaded && input_loaded) next_state = COMPUTE;
      end

      COMPUTE: begin
        // MAC just finished — move to OUTPUT to drain accumulators
        if (mac_done) next_state = OUTPUT;
      end

      OUTPUT: begin
        // All channels for current position drained
        // Use runtime out_channels_cfg instead of compile-time OUT_CHANNELS
        if (data_out_valid && data_out_ready && output_ch_counter == out_channels_cfg - 1) begin
          if (out_pos + 1 >= output_length) next_state = DONE_STATE;
          else next_state = COMPUTE;  // next position
        end
      end

      DONE_STATE: begin
        if (!start) next_state = IDLE;
      end

      default: next_state = IDLE;
    endcase
  end

  assign busy = (state != IDLE) && (state != DONE_STATE);
  assign done = (state == DONE_STATE);
  assign data_in_ready = (state == LOAD_WEIGHTS || state == LOAD_BIAS || state == COMPUTE);

  // Weight loading
  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n) begin
      weights_loaded <= '0;
      for (int oc = 0; oc < MAX_OUT_CHANNELS; oc++) begin
        for (int ic = 0; ic < MAX_IN_CHANNELS; ic++) begin
          weights[oc][ic] <= '0;
        end
      end
    end else if (state == LOAD_WEIGHTS && weight_valid) begin
      if (weight_out_ch == out_channels_cfg - 1 && weight_in_ch == in_channels_cfg - 1) begin
        weights_loaded <= 1'b1;
      end
      weights[weight_out_ch][weight_in_ch] <= weight_in;
    end else if (state == IDLE) begin
      weights_loaded <= '0;
    end
  end

  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n) begin
      bias_loaded <= '0;
      for (int oc = 0; oc < MAX_OUT_CHANNELS; oc++) begin
        biases[oc] <= '0;
      end
    end else if (state == LOAD_BIAS && bias_valid) begin
      biases[bias_out_ch] <= bias_in;
      if (bias_out_ch == out_channels_cfg - 1) bias_loaded <= 1'b1;
    end else if (state == IDLE) begin
      bias_loaded <= '0;
    end
  end

  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n) begin
      input_loaded <= '0;
      input_count <= '0;
      for (int ic = 0; ic < MAX_IN_CHANNELS; ic++) begin
        for (int pos = 0; pos < MAX_SEQ_LEN; pos++) begin
          input_data[ic][pos] <= '0;
        end
      end
    end else if (state == IDLE) begin
      input_loaded <= '0;
      input_count <= '0;
    end else if (data_in_valid && data_in_ready) begin
      if (!input_loaded) begin
        input_count <= input_count + 32'd1;
        if (input_count + 32'd1 >= 32'(in_channels_cfg) * 32'(input_length)) begin
          input_loaded <= 1'b1;
        end
      end
      input_data[in_channel_idx][in_pos_idx] <= data_in;
    end
  end

  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < NUM_MAC_UNITS; i++) begin
        mac_accumulators[i] <= '0;
        mac_in_ch[i] <= '0;
      end
      mac_computing <= '0;
      mac_cycle_counter <= '0;
      mac_done <= '0;
    end else if (state == COMPUTE) begin
      mac_done <= '0;  // default: clear pulse

      if (mac_computing) begin
        for (int i = 0; i < NUM_MAC_UNITS; i++) begin
          if (i < out_channels_cfg) begin
            mac_accumulators[i] <= mac_accumulators[i] + $signed(
                input_data[mac_in_ch[i]][out_pos]
            ) * $signed(
                weights[i][mac_in_ch[i]]
            );

            // Advance input feature counter
            if (mac_in_ch[i] < in_channels_cfg - 1) mac_in_ch[i] <= mac_in_ch[i] + 16'd1;
          end
        end

        mac_cycle_counter <= mac_cycle_counter + 16'd1;
        if (mac_cycle_counter == mac_cycles[15:0] - 1) begin
          mac_computing <= '0;
          mac_done <= 1'b1;
        end

      end else if (!mac_done) begin
        for (int i = 0; i < NUM_MAC_UNITS; i++) begin
          if (i < out_channels_cfg) mac_accumulators[i] <= $signed(biases[i]);
          else mac_accumulators[i] <= '0;
          mac_in_ch[i] <= '0;
        end
        mac_computing <= 1'b1;
        mac_cycle_counter <= '0;
      end
      // else: mac_done is high, mac_computing is low means hold accumulators
      // stable for one cycle while FSM transitions to OUTPUT

    end else begin  // IDLE
      mac_computing <= '0;
      mac_cycle_counter <= '0;
      mac_done <= '0;
    end
  end

  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n) begin
      data_out <= '0;
      data_out_valid <= '0;
      out_channel_idx <= '0;
      out_pos_idx <= '0;
      out_pos <= '0;
      output_ch_counter <= '0;
    end else begin
      case (state)
        OUTPUT: begin
          if (!data_out_valid) begin
            // Present first (or next) channel
            data_out <= mac_accumulators[output_ch_counter];
            out_channel_idx <= output_ch_counter;
            out_pos_idx <= out_pos;
            data_out_valid <= 1'b1;
          end else if (data_out_ready) begin
            // Use runtime out_channels_cfg instead of OUT_CHANNELS
            if (output_ch_counter < out_channels_cfg - 1) begin
              // Next channel, same position
              output_ch_counter <= output_ch_counter + 16'd1;
              data_out <= mac_accumulators[output_ch_counter+1];
              out_channel_idx <= output_ch_counter + 16'd1;
              out_pos_idx <= out_pos;
              // data_out_valid stays high, back-to-back handshake
            end else begin
              // All channels drained for this position
              output_ch_counter <= '0;
              out_pos <= out_pos + 16'd1;
              data_out_valid <= '0;
            end
          end
        end

        IDLE: begin
          // Full reset on new operation
          data_out <= '0;
          data_out_valid <= '0;
          out_channel_idx <= '0;
          out_pos_idx <= '0;
          out_pos <= '0;
          output_ch_counter <= '0;
        end

        // In any other state, output is not valid
        default: begin
          data_out_valid <= '0;
        end
      endcase
    end
  end

endmodule
