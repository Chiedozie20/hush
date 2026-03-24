// ----- THIS IS JUST HERE FOR DEMONSTRATION SAKE -----

// Weight read port, one per block
// Block drives req, arbiter drives rd
module block #(
  parameter int AddrWidth = 32,
  parameter int DataWidth = 16,
  parameter int BurstWidth = 8 // Max burst count is 2^BurstWidth - 1
)(
  input logic clk,
  input logic rst_n,
  // weight read request (block -> arbiter)
  output logic req_valid,
  input logic req_ready,
  output logic [AddrWidth-1:0] req_addr,
  output logic [BurstWidth-1:0] req_len,
  // weight read data (arbiter -> block)
  input logic [DataWidth-1:0] rd_data,
  input logic rd_valid,
  output logic rd_ready,
  // activation in (previous block -> this block)
  input logic [DataWidth-1:0] act_in_data,
  input logic act_in_valid,
  output logic act_in_ready,
  // activation out (this block -> next block)
  output logic [DataWidth-1:0] act_out_data,
  output logic act_out_valid,
  input logic act_out_ready
);

endmodule

// Round robin arbiter
// N block-facing ports on the front, one memory port on the back
module arbiter #(
  parameter int NumPorts = 4,
  parameter int AddrWidth = 32,
  parameter int DataWidth = 16,
  parameter int BurstWidth = 8
)(
  input logic clk,
  input logic rst_n,
  // block side: arrays of the same port, one per block
  input logic [NumPorts-1:0] blk_req_valid,
  output logic [NumPorts-1:0] blk_req_ready,
  input logic [AddrWidth-1:0] blk_req_addr [NumPorts],
  input logic [BurstWidth-1:0] blk_req_len [NumPorts],
  output logic [DataWidth-1:0] blk_rd_data, // All blocks can share the same response
  output logic [NumPorts-1:0] blk_rd_valid,
  input logic [NumPorts-1:0] blk_rd_ready,
  // memory side: single port to ddr controller
  output logic mem_req_valid,
  input logic mem_req_ready,
  output logic [AddrWidth-1:0] mem_req_addr,
  output logic [BurstWidth-1:0] mem_req_len,
  input logic [DataWidth-1:0] mem_rd_data,
  input logic mem_rd_valid,
  output logic mem_rd_ready
);

endmodule

// Fake DDR controller
// Just a big array with configurable random latency
// Cocotb loads weights via $readmemh before sim start
module ddr_controller #(
  parameter int AddrWidth = 32,
  parameter int DataWidth = 16,
  parameter int BurstWidth = 8,
  parameter int Depth = 65536, // number of words
  parameter int MaxLatency = 20 // max random delay in cycles
)(
  input logic clk,
  input logic rst_n,
  // from arbiter, same interface as the memory side above
  input logic req_valid,
  output logic req_ready,
  input logic [AddrWidth-1:0] req_addr,
  input logic [BurstWidth-1:0] req_len,
  output logic [DataWidth-1:0] rd_data,
  output logic rd_valid,
  input logic rd_ready,
  // cocotb can poke this to change the latency during sim, or, maybe use LFSR, will see
  input logic [7:0] latency_cfg
);

endmodule
