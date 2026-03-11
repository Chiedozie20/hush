/**
 * Testbench for Conv1d Hardware Accelerator (Verilator compatible)
 *
 * Simplified testbench that works with Verilator's C++ backend
 */

module conv1d_tb_verilator;

    localparam int DATA_WIDTH = 16;
    localparam int IN_CHANNELS = 2;
    localparam int OUT_CHANNELS = 2;
    localparam int KERNEL_SIZE = 3;
    localparam int STRIDE = 1;
    localparam int MAX_SEQ_LEN = 8;
    localparam int INPUT_LEN = 5;

    /* verilator lint_off PROCASSINIT */
    logic clk = 0;
    /* verilator lint_on PROCASSINIT */
    logic rst_n;
    logic start;
    logic done;
    /* verilator lint_off UNUSEDSIGNAL */
    logic busy;
    /* verilator lint_on UNUSEDSIGNAL */

    logic [DATA_WIDTH-1:0] data_in;
    logic data_in_valid;
    logic data_in_ready;
    logic [7:0] in_channel_idx;
    logic [15:0] in_pos_idx;
    logic [DATA_WIDTH-1:0] weight_in;
    logic [DATA_WIDTH-1:0] bias_in;
    logic weight_valid;
    logic bias_valid;
    logic [7:0] weight_out_ch;
    logic [7:0] weight_in_ch;
    logic [7:0] weight_k_idx;
    logic [7:0] bias_out_ch;
    logic [DATA_WIDTH-1:0] data_out;
    logic data_out_valid;
    logic data_out_ready;
    logic [7:0] out_channel_idx;
    logic [15:0] out_pos_idx;
    logic [15:0] input_length;
    conv1d #(
        .DATA_WIDTH(DATA_WIDTH),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .KERNEL_SIZE(KERNEL_SIZE),
        .STRIDE(STRIDE),
        .MAX_SEQ_LEN(MAX_SEQ_LEN)
    ) dut (
        .*
    );
    always #5 clk = ~clk;
    /* verilator lint_off PROCASSINIT */
    int cycle_count = 0;
    logic test_done = 0;
    /* verilator lint_on PROCASSINIT */
    int max_cycles = 10000;

    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (cycle_count >= max_cycles) begin
            $display("ERROR: Timeout after %0d cycles", cycle_count);
            $finish;
        end
        if (test_done) begin
            $display("\nTest completed in %0d cycles", cycle_count);
            $finish;
        end
    end

    // Test sequence
    initial begin
        $display("========================================");
        $display("  Conv1d Verilator Testbench");
        $display("========================================");
        $display("Parameters:");
        $display("  IN_CHANNELS=%0d, OUT_CHANNELS=%0d", IN_CHANNELS, OUT_CHANNELS);
        $display("  KERNEL_SIZE=%0d, STRIDE=%0d", KERNEL_SIZE, STRIDE);
        $display("  INPUT_LEN=%0d", INPUT_LEN);
        $display("========================================\n");

        rst_n = 0;
        start = 0;
        data_in_valid = 0;
        weight_valid = 0;
        bias_valid = 0;
        data_out_ready = 1;
        input_length = INPUT_LEN;
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        $display("[%0t] Starting computation...", $time);
        start = 1;
        @(posedge clk);

        // Load weights - simple identity-like pattern
        $display("[%0t] Loading weights...", $time);
        weight_valid = 1;
        for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
            for (int ic = 0; ic < IN_CHANNELS; ic++) begin
                for (int k = 0; k < KERNEL_SIZE; k++) begin
                    weight_in = (oc == ic) ? 16'd1 : 16'd0;  // Identity pattern
                    weight_out_ch = oc[7:0];
                    weight_in_ch = ic[7:0];
                    weight_k_idx = k[7:0];
                    @(posedge clk);
                end
            end
        end
        weight_valid = 0;

        // Load biases - zero
        $display("[%0t] Loading biases...", $time);
        bias_valid = 1;
        for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
            bias_in = 16'd0;
            bias_out_ch = oc[7:0];
            @(posedge clk);
        end
        bias_valid = 0;

        // Load input data
        $display("[%0t] Loading input data...", $time);
        data_in_valid = 1;
        for (int ic = 0; ic < IN_CHANNELS; ic++) begin
            for (int pos = 0; pos < INPUT_LEN; pos++) begin
                while (!data_in_ready) @(posedge clk);
                data_in = 16'(ic * 10 + pos + 1);  // Simple test pattern
                in_channel_idx = ic[7:0];
                in_pos_idx = pos[15:0];
                $display("  Input[%0d][%0d] = %0d", ic, pos, data_in);
                @(posedge clk);
            end
        end
        data_in_valid = 0;

        start = 0;

        // Wait for done
        $display("[%0t] Waiting for computation...", $time);
        wait(done);
        $display("[%0t] Computation done!", $time);

        repeat(10) @(posedge clk);
        test_done = 1;
    end

    // Track previous done state manually
    logic prev_done = 0;

    // Monitor outputs
    always @(posedge clk) begin
        if (data_out_valid) begin
            $display("[%0t] Output[%0d][%0d] = %0d (0x%04h) [valid=%b, ready=%b]",
                     $time, out_channel_idx, out_pos_idx,
                     $signed(data_out), data_out, data_out_valid, data_out_ready);
        end

        // Debug: show when done signal goes high
        if (done && !prev_done) begin
            $display("[%0t] DONE signal asserted", $time);
        end
        prev_done <= done;
    end

endmodule
