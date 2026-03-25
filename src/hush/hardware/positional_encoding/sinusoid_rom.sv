module sinusoid_rom #(
    parameter int WIDTH = 16,
    parameter int PHASE_WIDTH = 32,
    parameter int INT_BITS = 4,
    parameter int FRAC_BITS = 12,
    parameter int LUT_DEPTH = 1024,
    parameter string INIT_FILE = "sin_lut.mem"
)(
    input  logic                           i_clk,
    input  logic signed [PHASE_WIDTH-1:0]  i_phase,
    input  logic                           i_cos,
    output logic signed [WIDTH-1:0]        o_sin
);

    localparam int ADDR_W = $clog2(LUT_DEPTH);
    localparam logic signed [WIDTH-1:0] PI_Q = 16'sd12868;
    localparam logic signed [WIDTH-1:0] TWO_PI_Q = 16'sd25736;
    localparam logic signed [WIDTH-1:0] HALF_PI_Q = 16'sd6434;

    logic signed [WIDTH-1:0] rom [0:LUT_DEPTH-1];

    logic [ADDR_W-1:0]       addr_q;
    logic                    negate_q;
    logic signed [WIDTH-1:0] rom_data_q;

    logic signed [PHASE_WIDTH-1:0] phase_adj;
    logic signed [PHASE_WIDTH-1:0] phase_mod;
    logic signed [PHASE_WIDTH-1:0] phase_fold;
    logic                          negate_d;
    logic signed [PHASE_WIDTH+ADDR_W-1:0] mult_tmp;
    logic signed [PHASE_WIDTH+ADDR_W-1:0] div_tmp;
    logic [ADDR_W-1:0] addr_d;

    initial begin
        $readmemh(INIT_FILE, rom);
    end

    always_comb begin
        phase_adj = i_cos ? ($signed(HALF_PI_Q) - i_phase) : i_phase;
        phase_mod = phase_adj % $signed(TWO_PI_Q);

        if (phase_mod < 0)
            phase_mod = phase_mod + $signed(TWO_PI_Q);

        if (phase_mod > $signed(PI_Q)) begin
            phase_fold = $signed(TWO_PI_Q) - phase_mod;
            negate_d = 1'b1;
        end else begin
            phase_fold = phase_mod;
            negate_d = 1'b0;
        end

        mult_tmp = phase_fold * (LUT_DEPTH - 1);
        div_tmp = mult_tmp / $signed(PI_Q);

        if (div_tmp < 0)
            addr_d = '0;
        else if (div_tmp > (LUT_DEPTH - 1))
            addr_d = LUT_DEPTH - 1;
        else
            addr_d = div_tmp[ADDR_W-1:0];
    end

    always_ff @(posedge i_clk) begin
        addr_q <= addr_d;
        negate_q <= negate_d;
        rom_data_q <= rom[addr_q];
        o_sin <= negate_q ? -rom_data_q : rom_data_q;
    end

endmodule
