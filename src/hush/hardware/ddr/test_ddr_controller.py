import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Helpers


async def init(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    dut.rst_n.value = 0
    dut.req_valid.value = 0
    dut.req_addr.value = 0
    dut.req_len.value = 0
    dut.rd_ready.value = 0

    # Reset assert / deassert
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def fill_mem(dut, base, count, offset=0xA000):
    for i in range(count):
        dut.mem[base + i].value = offset + i


async def await_idle(dut, timeout=50):
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.req_ready.value == 1:
            return
    raise TimeoutError("req_ready never asserted")


async def send_req(dut, addr, length):
    dut.req_valid.value = 1
    dut.req_addr.value = addr
    dut.req_len.value = length
    await await_idle(dut)
    dut.req_valid.value = 0


async def recv(dut, count, stall_every=0):
    """Collect count beats. stall_ever > 0 deasserts rd_ready every N beats."""
    data = []
    dut.rd_ready.value = 1
    while len(data) < count:
        await RisingEdge(dut.clk)
        if dut.rd_valid.value == 1 and dut.rd_ready.value == 1:
            data.append(int(dut.rd_data.value))
            if stall_every and len(data) % stall_every == 0 and len(data) < count:
                dut.rd_ready.value = 0
                await ClockCycles(dut.clk, 2)
                dut.rd_ready.value = 1
    dut.rd_ready.value = 0
    return data


async def read(dut, addr, length, stall_every=0):
    await send_req(dut, addr, length)
    return await recv(dut, length, stall_every)


def expected_seq(base, count, offset=0xA000):
    return [offset + base + i for i in range(count)]


def check(data, expected, label):
    def format_hex(d):
        return [hex(x) for x in d]

    assert data == expected, (
        f"FAIL {label}: got {format_hex(data)}, exp {format_hex(expected)}"
    )
    cocotb.log.info(f"PASS {label}: {format_hex(data)}")


# Tests


@cocotb.test()
async def test_single_read(dut):
    await init(dut)
    await fill_mem(dut, 0, 16)
    await await_idle(dut)

    data = await read(dut, 0, 1)
    check(data, expected_seq(0, 1), "single read addr=0")


@cocotb.test()
async def test_zero_len_read(dut):
    """Check the controller is robust."""
    await init(dut)
    await fill_mem(dut, 0, 16)
    await await_idle(dut)

    await send_req(dut, 0, 0)
    data = await recv(dut, 1)  # DUT clamps 0 -> 1 (hopefully)
    check(data, expected_seq(0, 1), "zero len clamped to 1")


@cocotb.test()
async def test_burst_read(dut):
    await init(dut)
    await fill_mem(dut, 0, 16)
    await await_idle(dut)

    data = await read(dut, 4, 4)
    check(data, expected_seq(4, 4), "burst len=4 addr=4")


@cocotb.test()
async def test_back_to_back(dut):
    await init(dut)
    await fill_mem(dut, 0, 16)
    await await_idle(dut)

    d1 = await read(dut, 0, 2)
    check(d1, expected_seq(0, 2), "b2b first")
    await await_idle(dut)
    d2 = await read(dut, 8, 3)
    check(d2, expected_seq(8, 3), "b2b second")


@cocotb.test()
async def test_backpressure(dut):
    """Stall rd_ready after every beat, data must still be correct."""
    await init(dut)
    await fill_mem(dut, 0, 16)
    await await_idle(dut)

    data = await read(dut, 0, 4, stall_every=1)
    check(data, expected_seq(0, 4), "backpressure stall_every=1")


@cocotb.test()
async def test_len_one_boundary(dut):
    """len=1 at several addresses, no burst continuation."""
    await init(dut)
    await fill_mem(dut, 0, 16)

    for addr in [0, 7, 15]:
        await await_idle(dut)
        data = await read(dut, addr, 1)
        check(data, expected_seq(addr, 1), f"len=1 addr={addr}")


@cocotb.test()
async def test_valid_before_ready(dut):
    """Assert req_valid while controller is still busy, must wait!"""
    # Basically checking here that an interruption does not corrupt
    # the current txn in flight
    await init(dut)
    await fill_mem(dut, 0, 16)
    await await_idle(dut)

    # Start first transaction
    await send_req(dut, 0, 2)

    # Immediately assert next request (controller is in WAIT/READ, not IDLE)
    dut.req_valid.value = 1
    dut.req_addr.value = 4
    dut.req_len.value = 3

    # Drain first burst
    d1 = await recv(dut, 2)
    check(d1, expected_seq(0, 2), "overlap first")

    # Controller returns to IDLE, handshakes second request
    d2 = await recv(dut, 3)
    dut.req_valid.value = 0
    check(d2, expected_seq(4, 3), "overlap second")
