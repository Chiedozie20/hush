// parameterised 1d convolutional layer.
// 6 state fsm - IDLE, LOAD_WEIGHTS, LOAD_BIAS, COMPUTE, OUTPUT, DONE_STATE
// weights and biases are loaded in parallel, then computation starts once both are ready
// during COMPUTE, NUM_MAC_UNITS are processed in parallel
// each MAC unit handles one output channel, iterating through input channels 
// and kernel positions sequentially


module conv1d #(
    parameter int DATA_WIDTH      = 16,
    parameter int MAX_IN_CHANNELS = 384,
    parameter int MAX_OUT_CHANNELS = 384,
    parameter int KERNEL_SIZE     = 3,
    parameter int MAX_SEQ_LEN     = 3000,
    parameter int NUM_MAC_UNITS   = 384,
    parameter int PAD             = (KERNEL_SIZE - 1) / 2
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,
    output logic busy,


    input  logic [15:0] in_channels_cfg,
    input  logic [15:0] out_channels_cfg,
    input  logic [15:0] stride_cfg,

    input  logic [DATA_WIDTH-1:0] weight_in,
    input  logic [2*DATA_WIDTH-1:0] bias_in,
    input  logic weight_valid,
    input  logic bias_valid,
    input  logic [15:0] weight_out_ch,
    input  logic [15:0] weight_in_ch,
    input  logic [15:0] weight_k_idx,
    input  logic [15:0] bias_out_ch,


    input  logic [DATA_WIDTH-1:0] data_in,
    input  logic data_in_valid,
    output logic data_in_ready,
    input  logic [15:0] in_channel_idx,
    input  logic [15:0] in_pos_idx,

    output logic [2*DATA_WIDTH-1:0] data_out,
    output logic data_out_valid,
    input  logic data_out_ready,
    output logic [15:0] out_channel_idx,
    output logic [15:0] out_pos_idx,

    input  logic [15:0] input_length
);

    localparam int MAX_MAC_CYCLES = MAX_IN_CHANNELS * KERNEL_SIZE;
    logic [31:0] mac_cycles;
    assign mac_cycles = {16'd0, in_channels_cfg} * KERNEL_SIZE;



    typedef enum logic [2:0] {
        IDLE,
        LOAD_WEIGHTS,
        LOAD_BIAS,
        COMPUTE,
        OUTPUT,
        DONE_STATE
    } state_t;

    state_t state, next_state;





    logic [DATA_WIDTH-1:0] weights    [MAX_OUT_CHANNELS-1:0][MAX_IN_CHANNELS-1:0][KERNEL_SIZE-1:0];
    logic [2*DATA_WIDTH-1:0] biases   [MAX_OUT_CHANNELS-1:0];
    logic [DATA_WIDTH-1:0] input_data [MAX_IN_CHANNELS-1:0][MAX_SEQ_LEN-1:0];
    logic        weights_loaded;
    logic        bias_loaded;
    logic        input_loaded;
    logic [31:0] input_count;
    logic [15:0] out_pos;
    logic [15:0] output_ch_counter;

    logic [15:0] output_length;

    assign output_length = (input_length + 2 * PAD - KERNEL_SIZE) / stride_cfg + 1;
    logic signed [2*DATA_WIDTH-1:0] mac_accumulators [NUM_MAC_UNITS-1:0];
    logic [15:0] mac_in_ch         [NUM_MAC_UNITS-1:0];
    logic [15:0] mac_k_idx         [NUM_MAC_UNITS-1:0];
    logic        mac_computing;
    logic [15:0] mac_cycle_counter;
    logic        mac_done;
    logic signed [16:0] input_pos [NUM_MAC_UNITS-1:0];
    genvar gi;
    generate
        for (gi = 0; gi < NUM_MAC_UNITS; gi++) begin : gen_input_pos
            assign input_pos[gi] = $signed({1'b0, out_pos}) * $signed({1'b0, stride_cfg})
                                 + $signed({1'b0, mac_k_idx[gi]})
                                 - $signed(PAD[16:0]);
        end
    endgenerate
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
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
                if (mac_done) next_state = OUTPUT;
            end
            OUTPUT: begin
                if (data_out_valid && data_out_ready &&
                    output_ch_counter == out_channels_cfg - 1) begin
                    if (out_pos + 1 >= output_length)
                        next_state = DONE_STATE;
                    else
                        next_state = COMPUTE;
                end
            end
            DONE_STATE: begin
                if (!start) next_state = IDLE;
            end

            default: next_state = IDLE;
        endcase
    end

    assign busy        = (state != IDLE) && (state != DONE_STATE);
    assign done        = (state == DONE_STATE);
    assign data_in_ready = (state == LOAD_WEIGHTS || state == LOAD_BIAS || state == COMPUTE);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weights_loaded <= 1'b0;
            for (int oc = 0; oc < MAX_OUT_CHANNELS; oc++)
                for (int ic = 0; ic < MAX_IN_CHANNELS; ic++)
                    for (int k = 0; k < KERNEL_SIZE; k++)
                        weights[oc][ic][k] <= '0;
        end else if (state == LOAD_WEIGHTS && weight_valid) begin
            weights[weight_out_ch][weight_in_ch][weight_k_idx] <= weight_in;
            if (weight_out_ch == out_channels_cfg-1 &&
                weight_in_ch  == in_channels_cfg-1  &&
                weight_k_idx  == KERNEL_SIZE-1)
                weights_loaded <= 1'b1;
        end else if (state == IDLE) begin
            weights_loaded <= 1'b0;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bias_loaded <= 1'b0;
            for (int oc = 0; oc < MAX_OUT_CHANNELS; oc++)
                biases[oc] <= '0;
        end else if (state == LOAD_BIAS && bias_valid) begin
            biases[bias_out_ch] <= bias_in;

            if (bias_out_ch == out_channels_cfg-1)
                bias_loaded <= 1'b1;
        end else if (state == IDLE) begin
            bias_loaded <= 1'b0;
        end
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int ic = 0; ic < MAX_IN_CHANNELS; ic++)
                for (int pos = 0; pos < MAX_SEQ_LEN; pos++)
                    input_data[ic][pos] <= '0;
            input_loaded <= 1'b0;
            input_count  <= 32'd0;
        end else if (state == IDLE) begin
            input_loaded <= 1'b0;
            input_count  <= 32'd0;
        end else if (data_in_valid && data_in_ready) begin
            input_data[in_channel_idx][in_pos_idx] <= data_in;
            if (!input_loaded) begin
                input_count <= input_count + 32'd1;

                if (input_count + 32'd1 >= 32'(in_channels_cfg) * 32'(input_length))
                    input_loaded <= 1'b1;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_MAC_UNITS; i++) begin
                mac_accumulators[i] <= '0;
                mac_in_ch[i]        <= 16'd0;
                mac_k_idx[i]        <= 16'd0;
            end
            mac_computing     <= 1'b0;
            mac_cycle_counter <= 16'd0;
            mac_done          <= 1'b0;
        end else if (state == COMPUTE) begin
            mac_done <= 1'b0;

            if (mac_computing) begin
                for (int i = 0; i < NUM_MAC_UNITS; i++) begin
                    if (i < out_channels_cfg) begin
                        if (input_pos[i] >= 0 &&
                            input_pos[i] < $signed({1'b0, input_length})) begin
                            mac_accumulators[i] <= mac_accumulators[i]
                                + $signed(input_data[mac_in_ch[i]][input_pos[i][15:0]])
                                * $signed(weights[i][mac_in_ch[i]][mac_k_idx[i]]);
                        end


                        if (mac_k_idx[i] < KERNEL_SIZE - 1) begin
                            mac_k_idx[i] <= mac_k_idx[i] + 16'd1;
                        end else begin
                            mac_k_idx[i] <= 16'd0;
                            if (mac_in_ch[i] < in_channels_cfg - 1)
                                mac_in_ch[i] <= mac_in_ch[i] + 16'd1;
                        end
                    end
                end

                mac_cycle_counter <= mac_cycle_counter + 16'd1;
                if (mac_cycle_counter == mac_cycles[15:0] - 1) begin
                    mac_computing <= 1'b0;
                    mac_done      <= 1'b1;
                end

            end else if (!mac_done) begin
                for (int i = 0; i < NUM_MAC_UNITS; i++) begin
                    if (i < out_channels_cfg)
                        mac_accumulators[i] <= $signed(biases[i]);
                    else
                        mac_accumulators[i] <= '0;
                    mac_in_ch[i]  <= 16'd0;
                    mac_k_idx[i]  <= 16'd0;
                end
                mac_computing     <= 1'b1;
                mac_cycle_counter <= 16'd0;
            end
        end else begin
            mac_computing     <= 1'b0;
            mac_cycle_counter <= 16'd0;
            mac_done          <= 1'b0;
        end
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out         <= '0;
            data_out_valid   <= 1'b0;
            out_channel_idx  <= 16'd0;
            out_pos_idx      <= 16'd0;
            out_pos          <= 16'd0;
            output_ch_counter <= 16'd0;
        end else if (state == OUTPUT) begin
            if (!data_out_valid) begin

                data_out        <= mac_accumulators[output_ch_counter];
                out_channel_idx <= output_ch_counter;
                out_pos_idx     <= out_pos;
                data_out_valid  <= 1'b1;
            end else if (data_out_ready) begin

                if (output_ch_counter < out_channels_cfg - 1) begin

                    output_ch_counter <= output_ch_counter + 16'd1;
                    data_out          <= mac_accumulators[output_ch_counter + 1];
                    out_channel_idx   <= output_ch_counter + 16'd1;
                    out_pos_idx       <= out_pos;

                end else begin

                    output_ch_counter <= 16'd0;
                    out_pos           <= out_pos + 16'd1;
                    data_out_valid    <= 1'b0;
                end
            end
        end else if (state == IDLE) begin
            data_out         <= '0;
            data_out_valid   <= 1'b0;
            out_channel_idx  <= 16'd0;
            out_pos_idx      <= 16'd0;
            out_pos          <= 16'd0;
            output_ch_counter <= 16'd0;
        end else begin

            data_out_valid <= 1'b0;
        end
    end

endmodule