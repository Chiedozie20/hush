// Testbench wrapper that instantiates both conv1d and gelu_lut_wrapper
// This allows CocoTB to access both modules in a single test

`timescale 1ns / 1ps

module conv_gelu_testbench #(
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

    // Conv1d interface
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

    input  logic [15:0] input_length,

    // GELU interface
    input  logic [2*DATA_WIDTH-1:0] gelu_data_in,
    input  logic gelu_data_in_valid,
    output logic gelu_data_in_ready,

    output logic [2*DATA_WIDTH-1:0] gelu_data_out,
    output logic gelu_data_out_valid,
    input  logic gelu_data_out_ready
);

    // Conv1d instance
    conv1d #(
        .DATA_WIDTH(DATA_WIDTH),
        .MAX_IN_CHANNELS(MAX_IN_CHANNELS),
        .MAX_OUT_CHANNELS(MAX_OUT_CHANNELS),
        .KERNEL_SIZE(KERNEL_SIZE),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .NUM_MAC_UNITS(NUM_MAC_UNITS),
        .PAD(PAD)
    ) conv_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .busy(busy),

        .in_channels_cfg(in_channels_cfg),
        .out_channels_cfg(out_channels_cfg),
        .stride_cfg(stride_cfg),

        .weight_in(weight_in),
        .bias_in(bias_in),
        .weight_valid(weight_valid),
        .bias_valid(bias_valid),
        .weight_out_ch(weight_out_ch),
        .weight_in_ch(weight_in_ch),
        .weight_k_idx(weight_k_idx),
        .bias_out_ch(bias_out_ch),

        .data_in(data_in),
        .data_in_valid(data_in_valid),
        .data_in_ready(data_in_ready),
        .in_channel_idx(in_channel_idx),
        .in_pos_idx(in_pos_idx),

        .data_out(data_out),
        .data_out_valid(data_out_valid),
        .data_out_ready(data_out_ready),
        .out_channel_idx(out_channel_idx),
        .out_pos_idx(out_pos_idx),

        .input_length(input_length)
    );

    // GELU LUT wrapper instance
    gelu_lut_wrapper #(
        .DATA_WIDTH(2*DATA_WIDTH)
    ) gelu_inst (
        .clk(clk),
        .rst_n(rst_n),

        .data_in(gelu_data_in),
        .data_in_valid(gelu_data_in_valid),
        .data_in_ready(gelu_data_in_ready),

        .data_out(gelu_data_out),
        .data_out_valid(gelu_data_out_valid),
        .data_out_ready(gelu_data_out_ready)
    );

endmodule
