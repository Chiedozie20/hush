import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import random


def conv1d_out_len(L_in, K=3, S=2, P=1):
    return ((L_in + 2*P - K) // S) + 1


@cocotb.test()
async def measure_tensor_time(dut):
    """Measure latency + total cycles until full tensor output"""

    # -------------------------
    # Config
    # -------------------------
    CLK_PERIOD_NS = 10
    INPUT_LENGTH = 128   # <-- change this
    CHANNELS = 384       # not directly used unless you stream per-channel

    EXPECTED_OUTPUTS = conv1d_out_len(INPUT_LENGTH)

    # -------------------------
    # Clock
    # -------------------------
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    # -------------------------
    # Reset
    # -------------------------
    dut.rst_n.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1

    for _ in range(5):
        await RisingEdge(dut.clk)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # -------------------------
    # Variables for timing
    # -------------------------
    cycle = 0
    start_cycle = None
    first_output_cycle = None
    last_output_cycle = None

    inputs_sent = 0
    outputs_received = 0

    # -------------------------
    # Main loop
    # -------------------------
    while outputs_received < EXPECTED_OUTPUTS:

        await RisingEdge(dut.clk)
        cycle += 1

        # ---------------------
        # Drive input
        # ---------------------
        if inputs_sent < INPUT_LENGTH and dut.data_in_ready.value:
            dut.data_in_valid.value = 1
            dut.data_in.value = random.randint(-1000, 1000)

            if start_cycle is None:
                start_cycle = cycle

            inputs_sent += 1
        else:
            dut.data_in_valid.value = 0

        # ---------------------
        # Check output
        # ---------------------
        if dut.data_out_valid.value and dut.data_out_ready.value:

            if first_output_cycle is None:
                first_output_cycle = cycle

            outputs_received += 1

            if outputs_received == EXPECTED_OUTPUTS:
                last_output_cycle = cycle

    # -------------------------
    # Results
    # -------------------------
    latency = first_output_cycle - start_cycle
    total_time = last_output_cycle - start_cycle

    dut._log.info(f"Input length: {INPUT_LENGTH}")
    dut._log.info(f"Expected outputs: {EXPECTED_OUTPUTS}")
    dut._log.info(f"Latency (cycles): {latency}")
    dut._log.info(f"Total time (cycles): {total_time}")
    dut._log.info(f"Throughput (cycles/output): {total_time / EXPECTED_OUTPUTS:.2f}")

    print("\n========== TIMING RESULTS ==========")
    print(f"Latency (cycles): {latency}")
    print(f"Total cycles: {total_time}")
    print(f"Cycles per output: {total_time / EXPECTED_OUTPUTS:.2f}")
    print("===================================\n")