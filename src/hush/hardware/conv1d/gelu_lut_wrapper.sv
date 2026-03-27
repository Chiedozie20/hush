`timescale 1ns / 1ps

module gelu_lut_wrapper #(
    parameter int DATA_WIDTH = 32
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
    logic signed [15:0] stage1_data_16;
    logic stage1_valid;
    logic signed [15:0] stage2_data_16;
    logic stage2_valid;
    logic signed [DATA_WIDTH-1:0] stage3_data_32;
    logic stage3_valid;
    logic pipeline_enable;
    assign pipeline_enable = !stage3_valid || data_out_ready;
    logic signed [DATA_WIDTH-1:0] stage1_scaled;
    logic signed [47:0] stage1_temp;
    always_comb begin
        stage1_temp = data_in * 48'sd26830;
        stage1_scaled = stage1_temp[45:14];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_data_16 <= '0;
            stage1_valid   <= 1'b0;
        end else if (pipeline_enable) begin

            stage1_valid <= data_in_valid;

            if (data_in_valid) begin

                if (stage1_scaled > 32767)
                    stage1_data_16 <= 16'sd32767;
                else if (stage1_scaled < -32768)
                    stage1_data_16 <= -16'sd32768;
                else
                    stage1_data_16 <= stage1_scaled[15:0];
            end
        end
    end
    gelu_lut gelu_inst (
        .clk(clk),
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
    logic signed [47:0] stage3_temp;
    always_comb begin
        stage3_temp = $signed(stage2_data_16) * 48'sd10000;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage3_data_32 <= '0;
            stage3_valid   <= 1'b0;
        end else if (pipeline_enable) begin
            stage3_valid <= stage2_valid;
            if (stage2_valid) begin

                stage3_data_32 <= stage3_temp[45:14];
            end
        end
    end
    assign data_out = stage3_data_32;
    assign data_out_valid = stage3_valid;
    assign data_in_ready = pipeline_enable;

endmodule
