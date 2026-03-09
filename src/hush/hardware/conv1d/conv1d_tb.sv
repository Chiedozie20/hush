/**
 * Testbench for Conv1d Hardware Accelerator
 *
 * Tests the 1D convolution module with simple known values
 * to verify correct computation.
 */

`timescale 1ns/1ps

module conv1d_tb;

    // Parameters - small for testing
    localparam int DATA_WIDTH = 16;
    localparam int IN_CHANNELS = 2;
    localparam int OUT_CHANNELS = 2;
    localparam int KERNEL_SIZE = 3;
    localparam int STRIDE = 1;
    localparam int MAX_SEQ_LEN = 8;
    localparam int INPUT_LEN = 5;

    // Clock and reset
    logic clk;
    logic rst_n;

    // Control signals
    logic start;
    logic done;
    logic busy;

    // Input data interface
    logic [DATA_WIDTH-1:0] data_in;
    logic data_in_valid;
    logic data_in_ready;
    logic [7:0] in_channel_idx;
    logic [15:0] in_pos_idx;

    // Weight/bias interface
    logic [DATA_WIDTH-1:0] weight_in;
    logic [DATA_WIDTH-1:0] bias_in;
    logic weight_valid;
    logic bias_valid;
    logic [7:0] weight_out_ch;
    logic [7:0] weight_in_ch;
    logic [7:0] weight_k_idx;
    logic [7:0] bias_out_ch;

    // Output data interface
    logic [DATA_WIDTH-1:0] data_out;
    logic data_out_valid;
    logic data_out_ready;
    logic [7:0] out_channel_idx;
    logic [15:0] out_pos_idx;

    // Configuration
    logic [15:0] input_length;

    // DUT instantiation
    conv1d #(
        .DATA_WIDTH(DATA_WIDTH),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .KERNEL_SIZE(KERNEL_SIZE),
        .STRIDE(STRIDE),
        .MAX_SEQ_LEN(MAX_SEQ_LEN)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .busy(busy),
        .data_in(data_in),
        .data_in_valid(data_in_valid),
        .data_in_ready(data_in_ready),
        .in_channel_idx(in_channel_idx),
        .in_pos_idx(in_pos_idx),
        .weight_in(weight_in),
        .bias_in(bias_in),
        .weight_valid(weight_valid),
        .bias_valid(bias_valid),
        .weight_out_ch(weight_out_ch),
        .weight_in_ch(weight_in_ch),
        .weight_k_idx(weight_k_idx),
        .bias_out_ch(bias_out_ch),
        .data_out(data_out),
        .data_out_valid(data_out_valid),
        .data_out_ready(data_out_ready),
        .out_channel_idx(out_channel_idx),
        .out_pos_idx(out_pos_idx),
        .input_length(input_length)
    );

    // Clock generation (100 MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test data storage
    logic [DATA_WIDTH-1:0] test_input [IN_CHANNELS-1:0][INPUT_LEN-1:0];
    logic [DATA_WIDTH-1:0] test_weights [OUT_CHANNELS-1:0][IN_CHANNELS-1:0][KERNEL_SIZE-1:0];
    logic [DATA_WIDTH-1:0] test_biases [OUT_CHANNELS-1:0];
    logic [DATA_WIDTH-1:0] expected_output [OUT_CHANNELS-1:0][INPUT_LEN-1:0];

    // Output capture
    logic [DATA_WIDTH-1:0] actual_output [OUT_CHANNELS-1:0][INPUT_LEN-1:0];
    int output_count;

    // Task to load weights
    task load_weights();
        $display("[%0t] Loading weights...", $time);
        weight_valid = 1;
        for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
            for (int ic = 0; ic < IN_CHANNELS; ic++) begin
                for (int k = 0; k < KERNEL_SIZE; k++) begin
                    @(posedge clk);
                    weight_in = test_weights[oc][ic][k];
                    weight_out_ch = oc;
                    weight_in_ch = ic;
                    weight_k_idx = k;
                end
            end
        end
        @(posedge clk);
        weight_valid = 0;
        $display("[%0t] Weights loaded", $time);
    endtask

    // Task to load biases
    task load_biases();
        $display("[%0t] Loading biases...", $time);
        bias_valid = 1;
        for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
            @(posedge clk);
            bias_in = test_biases[oc];
            bias_out_ch = oc;
        end
        @(posedge clk);
        bias_valid = 0;
        $display("[%0t] Biases loaded", $time);
    endtask

    // Task to load input data
    task load_input_data();
        $display("[%0t] Loading input data...", $time);
        data_in_valid = 1;
        for (int ic = 0; ic < IN_CHANNELS; ic++) begin
            for (int pos = 0; pos < INPUT_LEN; pos++) begin
                @(posedge clk);
                while (!data_in_ready) @(posedge clk);
                data_in = test_input[ic][pos];
                in_channel_idx = ic;
                in_pos_idx = pos;
            end
        end
        @(posedge clk);
        data_in_valid = 0;
        $display("[%0t] Input data loaded", $time);
    endtask

    // Task to compute expected output (golden model)
    task compute_expected_output();
        int input_pos;
        logic signed [2*DATA_WIDTH-1:0] acc;

        $display("[%0t] Computing expected output...", $time);

        for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
            for (int pos = 0; pos < INPUT_LEN; pos++) begin
                acc = {{DATA_WIDTH{test_biases[oc][DATA_WIDTH-1]}}, test_biases[oc]};

                for (int ic = 0; ic < IN_CHANNELS; ic++) begin
                    for (int k = 0; k < KERNEL_SIZE; k++) begin
                        input_pos = pos * STRIDE + k;

                        // Handle padding (zero padding)
                        if (input_pos >= 1 && input_pos <= INPUT_LEN) begin
                            acc += $signed(test_input[ic][input_pos-1]) *
                                   $signed(test_weights[oc][ic][k]);
                        end
                    end
                end

                expected_output[oc][pos] = acc[DATA_WIDTH-1:0];
                $display("  Expected[%0d][%0d] = %0d (0x%0h)", oc, pos,
                         $signed(expected_output[oc][pos]), expected_output[oc][pos]);
            end
        end
    endtask

    // Monitor output
    always @(posedge clk) begin
        if (data_out_valid && data_out_ready) begin
            actual_output[out_channel_idx][out_pos_idx] = data_out;
            $display("[%0t] Output[%0d][%0d] = %0d (0x%0h)",
                     $time, out_channel_idx, out_pos_idx,
                     $signed(data_out), data_out);
            output_count++;
        end
    end

    // Main test sequence
    initial begin
        // Initialize signals
        rst_n = 0;
        start = 0;
        data_in_valid = 0;
        weight_valid = 0;
        bias_valid = 0;
        data_out_ready = 1;  // Always ready to receive
        input_length = INPUT_LEN;
        output_count = 0;

        // Initialize test data with simple values
        // Input: channel 0 = [1, 2, 3, 4, 5], channel 1 = [6, 7, 8, 9, 10]
        test_input[0][0] = 16'd1;
        test_input[0][1] = 16'd2;
        test_input[0][2] = 16'd3;
        test_input[0][3] = 16'd4;
        test_input[0][4] = 16'd5;

        test_input[1][0] = 16'd6;
        test_input[1][1] = 16'd7;
        test_input[1][2] = 16'd8;
        test_input[1][3] = 16'd9;
        test_input[1][4] = 16'd10;

        // Weights: simple 1x3 kernels
        // Output channel 0: weights = [[1,1,1], [0,0,0]]
        test_weights[0][0][0] = 16'd1;
        test_weights[0][0][1] = 16'd1;
        test_weights[0][0][2] = 16'd1;
        test_weights[0][1][0] = 16'd0;
        test_weights[0][1][1] = 16'd0;
        test_weights[0][1][2] = 16'd0;

        // Output channel 1: weights = [[0,0,0], [1,1,1]]
        test_weights[1][0][0] = 16'd0;
        test_weights[1][0][1] = 16'd0;
        test_weights[1][0][2] = 16'd0;
        test_weights[1][1][0] = 16'd1;
        test_weights[1][1][1] = 16'd1;
        test_weights[1][1][2] = 16'd1;

        // Biases: zero
        test_biases[0] = 16'd0;
        test_biases[1] = 16'd0;

        $display("========================================");
        $display("  Conv1d Testbench");
        $display("========================================");
        $display("Parameters:");
        $display("  IN_CHANNELS=%0d, OUT_CHANNELS=%0d", IN_CHANNELS, OUT_CHANNELS);
        $display("  KERNEL_SIZE=%0d, STRIDE=%0d", KERNEL_SIZE, STRIDE);
        $display("  INPUT_LEN=%0d", INPUT_LEN);
        $display("========================================\n");

        // Compute expected output
        compute_expected_output();

        // Reset
        $display("[%0t] Applying reset...", $time);
        #20;
        rst_n = 1;
        #20;

        // Start the computation
        $display("[%0t] Starting computation...", $time);
        start = 1;
        @(posedge clk);

        // Load weights
        load_weights();

        // Load biases
        load_biases();

        // Load input data
        load_input_data();

        // Wait for done
        start = 0;
        wait(done);
        $display("[%0t] Computation done!", $time);

        // Wait a bit more for any remaining outputs
        repeat(10) @(posedge clk);

        // Check results
        $display("\n========================================");
        $display("  Results Comparison");
        $display("========================================");

        int errors = 0;
        for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
            for (int pos = 0; pos < INPUT_LEN; pos++) begin
                if (actual_output[oc][pos] !== expected_output[oc][pos]) begin
                    $display("ERROR: Mismatch at [%0d][%0d]", oc, pos);
                    $display("  Expected: %0d (0x%0h)", $signed(expected_output[oc][pos]), expected_output[oc][pos]);
                    $display("  Actual:   %0d (0x%0h)", $signed(actual_output[oc][pos]), actual_output[oc][pos]);
                    errors++;
                end else begin
                    $display("OK: [%0d][%0d] = %0d", oc, pos, $signed(actual_output[oc][pos]));
                end
            end
        end

        $display("\n========================================");
        if (errors == 0) begin
            $display("  TEST PASSED!");
        end else begin
            $display("  TEST FAILED with %0d errors", errors);
        end
        $display("========================================\n");

        $finish;
    end

    // Timeout
    initial begin
        #100000;  // 100us timeout
        $display("ERROR: Testbench timeout!");
        $finish;
    end

    // Waveform dumping
    initial begin
        $dumpfile("conv1d_tb.vcd");
        $dumpvars(0, conv1d_tb);
    end

endmodule
