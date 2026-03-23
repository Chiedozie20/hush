module conv1d #(
    parameter int DATA_WIDTH    = 16,
    parameter int IN_CHANNELS   = 80,
    parameter int OUT_CHANNELS  = 384,
    parameter int KERNEL_SIZE   = 3,
    parameter int STRIDE        = 1,
    parameter int MAX_SEQ_LEN   = 3000,
    parameter int NUM_MAC_UNITS = 384,
    parameter int PAD           = (KERNEL_SIZE - 1) / 2 
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,
    output logic busy,

    input  logic [DATA_WIDTH-1:0] weight_in,
    input  logic [2*DATA_WIDTH-1:0] bias_in, 
    input  logic weight_valid,
    input  logic bias_valid,
    input  logic [15:0] weight_out_ch,
    input  logic [15:0] weight_in_ch,
    input  logic [15:0] weight_k_idx,
    input  logic [15:0] bias_out_ch,

    // Data input interface
    input  logic [DATA_WIDTH-1:0] data_in,
    input  logic data_in_valid,
    output logic data_in_ready,
    input  logic [15:0] in_channel_idx,
    input  logic [15:0] in_pos_idx,

    // Data output interface — full accumulator width to preserve fixed-point precision
    output logic [2*DATA_WIDTH-1:0] data_out,
    output logic data_out_valid,
    input  logic data_out_ready,
    output logic [15:0] out_channel_idx,
    output logic [15:0] out_pos_idx,

    input  logic [15:0] input_length
);

    // -------------------------------------------------------------------------
    // Local parameters
    // -------------------------------------------------------------------------
    localparam int MAC_CYCLES = IN_CHANNELS * KERNEL_SIZE;

    // -------------------------------------------------------------------------
    // FSM states
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        IDLE,
        LOAD_WEIGHTS,
        LOAD_BIAS,
        COMPUTE,
        OUTPUT,       // dedicated output-drain state
        DONE_STATE
    } state_t;

    state_t state, next_state;

    // -------------------------------------------------------------------------
    // Storage (kept as regs per user request — will migrate to DDR later)
    // -------------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] weights    [OUT_CHANNELS-1:0][IN_CHANNELS-1:0][KERNEL_SIZE-1:0];
    logic [2*DATA_WIDTH-1:0] biases   [OUT_CHANNELS-1:0];  // 32-bit to match accumulator domain
    logic [DATA_WIDTH-1:0] input_data [IN_CHANNELS-1:0][MAX_SEQ_LEN-1:0];

    // -------------------------------------------------------------------------
    // Control signals
    // -------------------------------------------------------------------------
    logic        weights_loaded;
    logic        bias_loaded;
    logic        input_loaded;
    logic [31:0] input_count;       // must hold IN_CHANNELS * MAX_SEQ_LEN (e.g. 240000)
    logic [15:0] out_pos;
    logic [15:0] output_ch_counter;

    logic [15:0] output_length;
    assign output_length = (input_length + 2 * PAD - KERNEL_SIZE) / STRIDE + 1;

    // -------------------------------------------------------------------------
    // MAC engine signals
    // -------------------------------------------------------------------------
    logic signed [2*DATA_WIDTH-1:0] mac_accumulators [NUM_MAC_UNITS-1:0];
    logic [15:0] mac_in_ch         [NUM_MAC_UNITS-1:0];
    logic [15:0] mac_k_idx         [NUM_MAC_UNITS-1:0];
    logic        mac_computing;
    logic [15:0] mac_cycle_counter;
    logic        mac_done;           // pulses high for one cycle when MAC finishes

    // Pre-computed input position per MAC unit (moved outside always_ff for portability)
    logic signed [16:0] input_pos [NUM_MAC_UNITS-1:0];
    genvar gi;
    generate
        for (gi = 0; gi < NUM_MAC_UNITS; gi++) begin : gen_input_pos
            assign input_pos[gi] = $signed({1'b0, out_pos}) * STRIDE
                                 + $signed({1'b0, mac_k_idx[gi]})
                                 - $signed(PAD[16:0]);
        end
    endgenerate

    // =========================================================================
    // FSM — next state
    // =========================================================================
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
                // MAC just finished — move to OUTPUT to drain accumulators
                if (mac_done) next_state = OUTPUT;
            end

            OUTPUT: begin
                // All channels for current position drained
                if (data_out_valid && data_out_ready &&
                    output_ch_counter == OUT_CHANNELS - 1) begin
                    if (out_pos + 1 >= output_length)
                        next_state = DONE_STATE;
                    else
                        next_state = COMPUTE;  // next position
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
    assign data_in_ready = (state == LOAD_WEIGHTS || state == LOAD_BIAS ||
                            state == COMPUTE);

    // =========================================================================
    // Weight loading
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weights_loaded <= 1'b0;
            for (int oc = 0; oc < OUT_CHANNELS; oc++)
                for (int ic = 0; ic < IN_CHANNELS; ic++)
                    for (int k = 0; k < KERNEL_SIZE; k++)
                        weights[oc][ic][k] <= '0;
        end else if (state == LOAD_WEIGHTS && weight_valid) begin
            weights[weight_out_ch][weight_in_ch][weight_k_idx] <= weight_in;
            if (weight_out_ch == OUT_CHANNELS-1 &&
                weight_in_ch  == IN_CHANNELS-1  &&
                weight_k_idx  == KERNEL_SIZE-1)
                weights_loaded <= 1'b1;
        end else if (state == IDLE) begin
            weights_loaded <= 1'b0;
        end
    end

    // =========================================================================
    // Bias loading
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bias_loaded <= 1'b0;
            for (int oc = 0; oc < OUT_CHANNELS; oc++)
                biases[oc] <= '0;
        end else if (state == LOAD_BIAS && bias_valid) begin
            biases[bias_out_ch] <= bias_in;
            if (bias_out_ch == OUT_CHANNELS-1)
                bias_loaded <= 1'b1;
        end else if (state == IDLE) begin
            bias_loaded <= 1'b0;
        end
    end

    // =========================================================================
    // Input data loading
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int ic = 0; ic < IN_CHANNELS; ic++)
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
                if (input_count + 32'd1 >= 32'(IN_CHANNELS) * 32'(input_length))
                    input_loaded <= 1'b1;
            end
        end
    end

    // =========================================================================
    // MAC engine (COMPUTE state only)
    //
    // FIX: The original read accumulators on the same cycle as the final
    //      multiply-accumulate, returning a stale value. Now we let the MAC
    //      run to completion (mac_done pulses one cycle AFTER the last product
    //      is accumulated) and drain in a separate OUTPUT state.
    // =========================================================================
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
            mac_done <= 1'b0;  // default: clear pulse

            if (mac_computing) begin
                // --- MAC step ---
                for (int i = 0; i < NUM_MAC_UNITS; i++) begin
                    if (i < OUT_CHANNELS) begin
                        // Generalized padding: only accumulate when input_pos
                        // is within [0, input_length-1]
                        if (input_pos[i] >= 0 &&
                            input_pos[i] < $signed({1'b0, input_length})) begin
                            mac_accumulators[i] <= mac_accumulators[i]
                                + $signed(input_data[mac_in_ch[i]][input_pos[i][15:0]])
                                * $signed(weights[i][mac_in_ch[i]][mac_k_idx[i]]);
                        end

                        // Advance (k, ic) counters
                        if (mac_k_idx[i] < KERNEL_SIZE - 1) begin
                            mac_k_idx[i] <= mac_k_idx[i] + 16'd1;
                        end else begin
                            mac_k_idx[i] <= 16'd0;
                            if (mac_in_ch[i] < IN_CHANNELS - 1)
                                mac_in_ch[i] <= mac_in_ch[i] + 16'd1;
                        end
                    end
                end

                mac_cycle_counter <= mac_cycle_counter + 16'd1;

                // Last MAC cycle: the product above is being written NOW.
                // Signal mac_done so the FSM transitions to OUTPUT next cycle.
                // The accumulators will be stable when OUTPUT reads them.
                if (mac_cycle_counter == MAC_CYCLES - 1) begin
                    mac_computing <= 1'b0;
                    mac_done      <= 1'b1;
                end

            end else if (!mac_done) begin
                // --- Initialise accumulators with bias ---
                // Guard: only re-init when we are genuinely starting a new
                // position, NOT on the cycle after mac_done when accumulators
                // hold completed results waiting for the OUTPUT state.
                // Bias is already 32-bit (quantized by scale^2), so no sign extension needed.
                for (int i = 0; i < NUM_MAC_UNITS; i++) begin
                    if (i < OUT_CHANNELS)
                        mac_accumulators[i] <= $signed(biases[i]);
                    else
                        mac_accumulators[i] <= '0;
                    mac_in_ch[i]  <= 16'd0;
                    mac_k_idx[i]  <= 16'd0;
                end
                mac_computing     <= 1'b1;
                mac_cycle_counter <= 16'd0;
            end
            // else: mac_done is high, mac_computing is low — hold accumulators
            // stable for one cycle while FSM transitions to OUTPUT

        end else begin
            // Not in COMPUTE — hold MAC idle
            mac_computing     <= 1'b0;
            mac_cycle_counter <= 16'd0;
            mac_done          <= 1'b0;
        end
    end

    // =========================================================================
    // Output drain (OUTPUT state only)
    //
    // FIX: Separated into its own state so there is no race between the last
    //      MAC accumulation and the output read. Accumulators are stable here.
    // =========================================================================
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
                // Present first (or next) channel
                data_out        <= mac_accumulators[output_ch_counter];
                out_channel_idx <= output_ch_counter;
                out_pos_idx     <= out_pos;
                data_out_valid  <= 1'b1;
            end else if (data_out_ready) begin
                if (output_ch_counter < OUT_CHANNELS - 1) begin
                    // Next channel, same position
                    output_ch_counter <= output_ch_counter + 16'd1;
                    data_out          <= mac_accumulators[output_ch_counter + 1];
                    out_channel_idx   <= output_ch_counter + 16'd1;
                    out_pos_idx       <= out_pos;
                    // data_out_valid stays high — back-to-back handshake
                end else begin
                    // All channels drained for this position
                    output_ch_counter <= 16'd0;
                    out_pos           <= out_pos + 16'd1;
                    data_out_valid    <= 1'b0;
                end
            end
        end else if (state == IDLE) begin
            // Full reset on new operation
            data_out         <= '0;
            data_out_valid   <= 1'b0;
            out_channel_idx  <= 16'd0;
            out_pos_idx      <= 16'd0;
            out_pos          <= 16'd0;
            output_ch_counter <= 16'd0;
        end else begin
            // In any other state, output is not valid
            data_out_valid <= 1'b0;
        end
    end

endmodule