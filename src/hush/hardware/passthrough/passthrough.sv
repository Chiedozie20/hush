module passthrough (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [31:0] data_in,
    input  logic        data_in_valid,
    output logic [31:0] data_out,
    output logic        data_out_valid
);

    localparam int DATA_WIDTH = 32;
    localparam int LATENCY = 3;

    logic [DATA_WIDTH-1:0] pipeline_data [0:LATENCY-1];
    logic                  pipeline_valid[0:LATENCY-1];

    initial begin
        $dumpfile("passthrough.vcd");
        $dumpvars(0, passthrough);
    end

    integer i;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < LATENCY; i = i + 1) begin
                pipeline_data[i] <= '0;
                pipeline_valid[i] <= 1'b0;
            end
        end else begin
            pipeline_data[0] <= data_in;
            pipeline_valid[0] <= data_in_valid;

            for (i = 1; i < LATENCY; i = i + 1) begin
                pipeline_data[i] <= pipeline_data[i-1];
                pipeline_valid[i] <= pipeline_valid[i-1];
            end
        end
    end

    assign data_out = pipeline_data[LATENCY-1];
    assign data_out_valid = pipeline_valid[LATENCY-1];

endmodule
