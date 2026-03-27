module frequency_index_rom #(
    parameter int DEPTH = 192,  // number of indexes
    parameter int WIDTH = 16,  // data width
    parameter string INIT_FILE = ""  // optional memory init file
) (
    input logic i_clk,
    input logic [$clog2(DEPTH)-1:0] i_index,
    output logic signed [WIDTH-1:0] o_frequency
);

  logic signed [WIDTH-1:0] rom[0:DEPTH-1];

  initial begin
    if (INIT_FILE != "") begin
      $readmemh(INIT_FILE, rom);
    end
  end

  always_ff @(posedge i_clk) begin
    o_frequency <= rom[i_index];
  end

endmodule
