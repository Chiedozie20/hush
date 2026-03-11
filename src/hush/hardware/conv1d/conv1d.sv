module conv1d #(
    parameter int DATA_WIDTH = 16,      
    parameter int IN_CHANNELS = 4,     
    parameter int OUT_CHANNELS = 8,    
    parameter int KERNEL_SIZE = 3,
    parameter int STRIDE = 1,
    parameter int MAX_SEQ_LEN = 100    
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,
    output logic busy,
    input  logic [DATA_WIDTH-1:0] data_in,
    input  logic data_in_valid,
    output logic data_in_ready,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [7:0] in_channel_idx,  // which input channel this data belongs to
    input  logic [15:0] in_pos_idx,     // position in sequence
    /* verilator lint_on UNUSEDSIGNAL */


    input  logic [DATA_WIDTH-1:0] weight_in,
    input  logic [DATA_WIDTH-1:0] bias_in,
    input  logic weight_valid,
    input  logic bias_valid,
    input  logic [7:0] weight_out_ch,   
    input  logic [7:0] weight_in_ch,    
    input  logic [7:0] weight_k_idx,    
    input  logic [7:0] bias_out_ch,   

    output logic [DATA_WIDTH-1:0] data_out,
    output logic data_out_valid,
    input  logic data_out_ready,
    output logic [7:0] out_channel_idx, 
    output logic [15:0] out_pos_idx,   

    input  logic [15:0] input_length 
);

    typedef enum logic [2:0] {
        IDLE,
        LOAD_WEIGHTS,
        LOAD_BIAS,
        COMPUTE,
        DONE_STATE
    } state_t;

    state_t state, next_state;
    logic [DATA_WIDTH-1:0] weights [OUT_CHANNELS-1:0][IN_CHANNELS-1:0][KERNEL_SIZE-1:0];
    logic [DATA_WIDTH-1:0] biases [OUT_CHANNELS-1:0];
    logic [DATA_WIDTH-1:0] input_data [IN_CHANNELS-1:0][MAX_SEQ_LEN-1:0];

    logic weights_loaded;
    logic bias_loaded;
    logic [15:0] out_pos;      
    logic [7:0]  out_ch;
    logic [7:0]  in_ch;
    logic [7:0]  k_idx;

    logic [2*DATA_WIDTH-1:0] mac_product;
    logic [2*DATA_WIDTH-1:0] mac_accumulator;
    logic mac_done;

    logic [15:0] output_length;
    assign output_length = (input_length + 2 - KERNEL_SIZE) / STRIDE + 1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
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
                if (bias_loaded) next_state = COMPUTE;
            end

            COMPUTE: begin
                if (mac_done) begin
                    next_state = DONE_STATE;
                end
            end

            DONE_STATE: begin
                if (!start) next_state = IDLE;
            end

            default: begin
                next_state = IDLE;
            end
        endcase
    end


    assign busy = (state != IDLE) && (state != DONE_STATE);
    assign done = (state == DONE_STATE);
    assign data_in_ready = (state == LOAD_WEIGHTS || state == LOAD_BIAS || state == COMPUTE);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weights_loaded <= 0;
            for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                for (int ic = 0; ic < IN_CHANNELS; ic++) begin
                    for (int k = 0; k < KERNEL_SIZE; k++) begin
                        weights[oc][ic][k] <= 0;
                    end
                end
            end
        end else if (state == LOAD_WEIGHTS && weight_valid) begin
            weights[weight_out_ch][weight_in_ch][weight_k_idx] <= weight_in;
            if (weight_out_ch == OUT_CHANNELS-1 &&
                weight_in_ch == IN_CHANNELS-1 &&
                weight_k_idx == KERNEL_SIZE-1) begin
                weights_loaded <= 1;
            end
        end else if (state == IDLE) begin
            weights_loaded <= 0;
        end
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bias_loaded <= 0;
            for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                biases[oc] <= 0;
            end
        end else if (state == LOAD_BIAS && bias_valid) begin
            biases[bias_out_ch] <= bias_in;
            if (bias_out_ch == OUT_CHANNELS-1) begin
                bias_loaded <= 1;
            end
        end else if (state == IDLE) begin
            bias_loaded <= 0;
        end
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int ic = 0; ic < IN_CHANNELS; ic++) begin
                for (int pos = 0; pos < MAX_SEQ_LEN; pos++) begin
                    input_data[ic][pos] <= 0;
                end
            end
        end else if (data_in_valid && data_in_ready) begin
            input_data[in_channel_idx][in_pos_idx] <= data_in;
        end
    end
    typedef enum logic [2:0] {
        MAC_IDLE,
        MAC_INIT,
        MAC_MULTIPLY,
        MAC_ACCUMULATE,
        MAC_OUTPUT,
        MAC_WAIT
    } mac_state_t;

    mac_state_t mac_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_state <= MAC_IDLE;
            out_pos <= 0;
            out_ch <= 0;
            in_ch <= 0;
            k_idx <= 0;
            mac_accumulator <= 0;
            mac_product <= 0;
            data_out_valid <= 0;
            mac_done <= 0;
        end else begin
            case (mac_state)
                MAC_IDLE: begin
                    mac_done <= 0;
                    if (state == COMPUTE) begin
                        mac_state <= MAC_INIT;
                        out_pos <= 0;
                        out_ch <= 0;
                    end
                end

                MAC_INIT: begin
                    mac_accumulator <= {{DATA_WIDTH{biases[out_ch][DATA_WIDTH-1]}}, biases[out_ch]};
                    in_ch <= 0;
                    k_idx <= 0;
                    mac_state <= MAC_MULTIPLY;
                end

                MAC_MULTIPLY: begin
                    logic [15:0] input_pos;
                    input_pos = out_pos * STRIDE + k_idx;
                    if (input_pos < 1 || input_pos > input_length) begin
                        mac_product <= 0;
                    end else begin
                        mac_product <= $signed(input_data[in_ch][input_pos-1]) *
                                     $signed(weights[out_ch][in_ch][k_idx]);
                    end
                    mac_state <= MAC_ACCUMULATE;
                end

                MAC_ACCUMULATE: begin
                    mac_accumulator <= mac_accumulator + mac_product;
                    if (k_idx < KERNEL_SIZE - 1) begin
                        k_idx <= k_idx + 1;
                        mac_state <= MAC_MULTIPLY;
                    end else begin
                        k_idx <= 0;
                        if (in_ch < IN_CHANNELS - 1) begin
                            in_ch <= in_ch + 1;
                            mac_state <= MAC_MULTIPLY;
                        end else begin
                            in_ch <= 0;
                            mac_state <= MAC_OUTPUT;
                        end
                    end
                end

                MAC_OUTPUT: begin
                    data_out <= mac_accumulator[DATA_WIDTH-1:0];  // Simplified: should handle scaling
                    out_channel_idx <= out_ch;
                    out_pos_idx <= out_pos;
                    data_out_valid <= 1;
                    mac_state <= MAC_WAIT;
                end

                MAC_WAIT: begin
                    if (data_out_ready) begin
                        data_out_valid <= 0;
                        if (out_pos < output_length - 1) begin
                            out_pos <= out_pos + 1;
                            mac_state <= MAC_INIT;
                        end else begin
                            out_pos <= 0;
                            if (out_ch < OUT_CHANNELS - 1) begin
                                out_ch <= out_ch + 1;
                                mac_state <= MAC_INIT;
                            end else begin
                                mac_done <= 1;
                                mac_state <= MAC_IDLE;
                            end
                        end
                    end
                end

                default: begin
                    mac_state <= MAC_IDLE;
                end
            endcase
        end
    end

endmodule
