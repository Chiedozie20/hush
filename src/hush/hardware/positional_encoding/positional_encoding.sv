module positional_encoding #(
    parameter int WIDTH = 16,
    parameter int N_STATE = 384,
    parameter int WPOS = $clog2(1500)
) (
    input  logic                     i_clk,
    input  logic                     i_rst,
    input  logic [0:N_STATE-1][WIDTH-1:0] i_x,
    input  logic [WPOS-1:0]          i_position,
    input  logic                     i_valid,
    output logic [0:N_STATE-1][WIDTH-1:0] o_x,
    output logic                     o_valid
);

    localparam int HALF_STATE = N_STATE / 2;
    localparam int STATE_W = (N_STATE > 1) ? $clog2(N_STATE) : 1;
    localparam int FREQ_IDX_W = (HALF_STATE > 1) ? $clog2(HALF_STATE) : 1;
    localparam int PHASE_WIDTH = WIDTH + WPOS + 1;
    localparam int SIN_ROM_LATENCY = 2;

    logic [0:N_STATE-1][WIDTH-1:0] x_latched;
    logic [WPOS-1:0]  position_latched;

    logic                    busy;
    logic [STATE_W-1:0]      issue_ctr;
    logic                    issue_done;

    logic [FREQ_IDX_W-1:0]   freq_index_in;
    logic signed [WIDTH-1:0] freq_out;

    logic                    issue_valid_d;
    logic [STATE_W-1:0]      state_idx_d;
    logic [WIDTH-1:0]        x_value_d;
    logic                    cos_d;

    logic signed [PHASE_WIDTH-1:0] phase_q;
    logic                          phase_valid_q;
    logic [STATE_W-1:0]            state_idx_q;
    logic [WIDTH-1:0]              x_value_q;

    logic [0:SIN_ROM_LATENCY-1][STATE_W-1:0] state_pipe;
    logic [0:SIN_ROM_LATENCY-1][WIDTH-1:0]   x_pipe;
    logic [0:SIN_ROM_LATENCY-1]              valid_pipe;

    logic                          cos_q;
    logic signed [WIDTH-1:0]       sin_value;
    logic                          write_valid;
    logic [STATE_W-1:0]            write_idx;
    logic [WIDTH-1:0]              write_x;
    logic signed [WIDTH:0]         sum_ext;

    integer i;

    assign freq_index_in = (issue_ctr < HALF_STATE) ? issue_ctr
                                                    : (issue_ctr - HALF_STATE);
    assign write_valid = valid_pipe[SIN_ROM_LATENCY-1];
    assign write_idx = state_pipe[SIN_ROM_LATENCY-1];
    assign write_x = x_pipe[SIN_ROM_LATENCY-1];
    assign sum_ext = $signed({1'b0, write_x}) + $signed(sin_value);

    frequency_index_rom #(
        .DEPTH(HALF_STATE),
        .WIDTH(WIDTH),
        .INIT_FILE("./freq_lut.mem")
    ) fi_lut (
        .i_clk(i_clk),
        .i_index(freq_index_in),
        .o_frequency(freq_out)
    );

    sinusoid_rom #(
        .WIDTH(WIDTH),
        .PHASE_WIDTH(PHASE_WIDTH),
        .INT_BITS(4),
        .FRAC_BITS(12),
        .LUT_DEPTH(1024),
        .INIT_FILE("./sin_lut.mem")
    ) u_sinusoid_rom (
        .i_clk(i_clk),
        .i_phase(phase_q),
        .i_cos(cos_q),
        .o_sin(sin_value)
    );

    always_ff @(posedge i_clk) begin
        if (i_rst) begin
            busy <= 1'b0;
            issue_ctr <= '0;
            issue_done <= 1'b0;
            issue_valid_d <= 1'b0;
            state_idx_d <= '0;
            x_value_d <= '0;
            cos_d <= 1'b0;
            phase_q <= '0;
            phase_valid_q <= 1'b0;
            state_idx_q <= '0;
            x_value_q <= '0;
            cos_q <= 1'b0;
            o_valid <= 1'b0;

            for (i = 0; i < N_STATE; i = i + 1) begin
                x_latched[i] <= '0;
                o_x[i] <= '0;
            end

            for (i = 0; i < SIN_ROM_LATENCY; i = i + 1) begin
                state_pipe[i] <= '0;
                x_pipe[i] <= '0;
                valid_pipe[i] <= 1'b0;
            end

            position_latched <= '0;
        end else begin
            o_valid <= 1'b0;

            issue_valid_d <= 1'b0;
            phase_valid_q <= 1'b0;

            if (!busy && i_valid) begin
                busy <= 1'b1;
                issue_ctr <= '0;
                issue_done <= 1'b0;
                position_latched <= i_position;

                for (i = 0; i < N_STATE; i = i + 1) begin
                    x_latched[i] <= i_x[i];
                end
            end else if (busy && !issue_done) begin
                issue_valid_d <= 1'b1;
                state_idx_d <= issue_ctr;
                x_value_d <= x_latched[issue_ctr];
                cos_d <= (issue_ctr >= (HALF_STATE - 1));

                if (issue_ctr == N_STATE - 1) begin
                    issue_done <= 1'b1;
                end else begin
                    issue_ctr <= issue_ctr + 1'b1;
                end
            end

            if (issue_valid_d) begin
                phase_q <= $signed({1'b0, position_latched}) * $signed(freq_out);
                phase_valid_q <= 1'b1;
                state_idx_q <= state_idx_d;
                x_value_q <= x_value_d;
                cos_q <= cos_d;
            end

            state_pipe[0] <= state_idx_q;
            x_pipe[0] <= x_value_q;
            valid_pipe[0] <= phase_valid_q;

            for (i = 1; i < SIN_ROM_LATENCY; i = i + 1) begin
                state_pipe[i] <= state_pipe[i-1];
                x_pipe[i] <= x_pipe[i-1];
                valid_pipe[i] <= valid_pipe[i-1];
            end

            if (write_valid) begin
                o_x[write_idx] <= sum_ext[WIDTH-1:0];

                if (write_idx == N_STATE - 1) begin
                    busy <= 1'b0;
                    o_valid <= 1'b1;
                end
            end
        end
    end

endmodule
