// GELU LUT Wrapper
// Converts 32-bit conv1d outputs to 16-bit for LUT lookup,
// then expands back to 32-bit for conv2 input

`timescale 1ns / 1ps

module gelu_lut_wrapper #(
    parameter int DATA_WIDTH = 32  // Input/output width from/to conv layers
) (
    input  logic clk,
    input  logic rst_n,

    input  logic signed [DATA_WIDTH-1:0] data_in,
    input  logic data_in_valid,
    output logic data_in_ready,

    output logic signed [DATA_WIDTH-1:0] data_out,
    output logic data_out_valid,
    input  logic data_out_ready
);

    // Pipeline stage 1: Truncate 32-bit to 16-bit
    logic signed [15:0] stage1_data_16;
    logic stage1_valid;

    // Pipeline stage 2: LUT lookup
    logic signed [15:0] stage2_data_16;
    logic stage2_valid;

    // Pipeline stage 3: Expand back to 32-bit
    logic signed [DATA_WIDTH-1:0] stage3_data_32;
    logic stage3_valid;

    // Pipeline enable - advance when output is consumed or pipeline not full
    logic pipeline_enable;
    assign pipeline_enable = !stage3_valid || data_out_ready;

    // Stage 1 intermediate signals
    logic signed [DATA_WIDTH-1:0] stage1_scaled;
    logic signed [47:0] stage1_temp;

    // =========================================================================
    // Stage 1: Truncate 32-bit to 16-bit
    // =========================================================================
    // Conv1d outputs are scaled by 100^2 = 10000
    // We need to convert to 16-bit with fractional width 14 (scale 2^14 = 16384)
    // Conversion: 32bit_value / 10000 * 16384 = 32bit_value * 1.6384

    always_comb begin
        // Multiply by 16384/10000 = 26214/16000 ≈ 1.6384
        // Use fixed-point: multiply by 1.6384 * 2^14 = 26830
        stage1_temp = data_in * 48'sd26830;
        stage1_scaled = stage1_temp[45:14];  // Extract result (divide by 2^14)
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_data_16 <= '0;
            stage1_valid   <= 1'b0;
        end else if (pipeline_enable) begin
            // Pipeline shifts: stage1 gets new data (or becomes invalid if no new data)
            stage1_valid <= data_in_valid;

            if (data_in_valid) begin
                // Clamp to 16-bit range
                if (stage1_scaled > 32767)
                    stage1_data_16 <= 16'sd32767;
                else if (stage1_scaled < -32768)
                    stage1_data_16 <= -16'sd32768;
                else
                    stage1_data_16 <= stage1_scaled[15:0];
            end
        end
    end

    // =========================================================================
    // Stage 2: LUT Lookup
    // =========================================================================
    gelu_lut gelu_inst (
        .data_in(stage1_data_16),
        .data_out(stage2_data_16)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage2_valid <= 1'b0;
        end else if (pipeline_enable) begin
            stage2_valid <= stage1_valid;
        end
    end

    // =========================================================================
    // Stage 3: Expand 16-bit back to 32-bit
    // =========================================================================
    // Reverse the scaling: LUT output is in 16-bit with frac_width=14
    // Convert back to 32-bit with scale 10000
    // Conversion: 16bit_value * 10000 / 16384 = 16bit_value * 0.6104

    // Stage 3 intermediate signals
    logic signed [47:0] stage3_temp;

    always_comb begin
        // Convert from scale 16384 to scale 10000
        // Formula: output = input * 10000 / 16384
        // Multiply by 10000, result is 16+14=30 bits max
        stage3_temp = $signed(stage2_data_16) * 48'sd10000;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage3_data_32 <= '0;
            stage3_valid   <= 1'b0;
        end else if (pipeline_enable) begin
            stage3_valid <= stage2_valid;
            if (stage2_valid) begin
                // Divide by 16384 (shift right by 14)
                stage3_data_32 <= stage3_temp[45:14];
            end
        end
    end

    // =========================================================================
    // Output
    // =========================================================================
    assign data_out = stage3_data_32;
    assign data_out_valid = stage3_valid;
    assign data_in_ready = pipeline_enable;

endmodule
