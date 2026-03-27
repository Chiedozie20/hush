module ddr_controller #(
    parameter int AddrWidth = 32,
    parameter int DataWidth = 16,
    parameter int BurstWidth = 8,
    parameter int Depth = 65536,  // Number of words
    parameter int Latency = 20  // Max random delay in cycles, not impl yet
) (
    input logic clk,
    input logic rst_n,
    input logic req_valid,
    output logic req_ready,
    input logic [AddrWidth-1:0] req_addr,
    input logic [BurstWidth-1:0] req_len,
    output logic [DataWidth-1:0] rd_data,
    output logic rd_valid,
    input logic rd_ready
);

  typedef enum logic [1:0] {
    IDLE,
    WAIT,  // Simulates DDR latency
    READ
  } state_t;

  localparam int DepthWidth = $clog2(Depth);

  state_t state, n_state;

  logic [DataWidth-1:0] mem[Depth]  /*verilator public*/;

  logic [AddrWidth-1:0] req_addr_latched, n_req_addr_latched;
  logic [BurstWidth-1:0] req_len_latched, n_req_len_latched;
  logic [BurstWidth-1:0] burst_cntr, n_burst_cntr;
  logic [AddrWidth-1:0] rd_addr, n_rd_addr;
  logic [DataWidth-1:0] n_rd_data;

  logic start_transaction;
  logic transaction;
  logic done_transaction;

  logic n_req_ready, n_rd_valid;

  // Internal logic
  assign start_transaction = req_valid && req_ready;
  assign transaction = rd_valid && rd_ready;
  assign done_transaction = (burst_cntr == req_len_latched - 1'b1);

  initial begin
    $readmemh("weights.hex", mem);
  end

  // State register & output registration
  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      req_addr_latched <= '0;
      req_len_latched <= '0;
      burst_cntr <= '0;
      rd_addr <= '0;
      rd_data <= '0;
      rd_valid <= '0;
      req_ready <= '0;
    end else begin
      state <= n_state;
      req_addr_latched <= n_req_addr_latched;
      req_len_latched <= n_req_len_latched;
      burst_cntr <= n_burst_cntr;
      rd_addr <= n_rd_addr;
      rd_data <= n_rd_data;
      rd_valid <= n_rd_valid;
      req_ready <= n_req_ready;
    end
  end

  // Next state logic
  always_comb begin
    n_state = state;
    unique case (state)
      IDLE: begin
        if (start_transaction) n_state = WAIT;
      end
      WAIT: begin
        n_state = READ;
      end
      READ: begin
        if (transaction && done_transaction) n_state = IDLE;
      end
    endcase
  end

  // Combinational data path
  always_comb begin
    n_rd_data = rd_data;
    n_req_addr_latched = req_addr_latched;
    n_req_len_latched = req_len_latched;
    n_rd_addr = rd_addr;
    n_burst_cntr = burst_cntr;
    n_rd_valid = '0;
    n_req_ready = '0;
    unique case (state)
      IDLE: begin
        n_req_ready = 1'b1;
        if (start_transaction) begin
          n_req_addr_latched = req_addr;
          // Defensive, ensures the invariant of nonzero len
          n_req_len_latched = (req_len == '0) ? 1'b1 : req_len;
          n_rd_addr = req_addr;
          n_burst_cntr = '0;
          n_rd_data = '0;
        end
      end
      WAIT: begin
      end
      READ: begin
        n_rd_valid = 1'b1;
        n_rd_data = mem[rd_addr[DepthWidth-1:0]];
        if (transaction) begin
          n_rd_addr = rd_addr + 1'd1;
          n_burst_cntr = burst_cntr + 1'd1;
          if (done_transaction) begin
            n_rd_valid = '0;
            n_rd_data = '0;
            n_req_ready = 1'b1;  // On the next cycle we are ready for a new txn
          end else begin
            n_rd_data = mem[rd_addr[DepthWidth-1:0]+1'd1];
          end
        end
      end
    endcase
  end

endmodule
