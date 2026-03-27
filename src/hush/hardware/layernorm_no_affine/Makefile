SIM ?= verilator
TOPLEVEL_LANG = verilog
TOPLEVEL = layernorm
COCOTB_TEST_MODULES = cocotb_layernorm
VERILOG_SOURCES = $(PWD)/fixed_invsqrt.sv $(PWD)/layernorm.sv

export PYTHONPATH := $(PWD):$(PYTHONPATH)
export CCACHE_DISABLE ?= 1
WAVEFILE = layernorm.vcd

.PHONY: all test test_isqrt wave clean

all: test
test: sim

test_isqrt:
	$(MAKE) sim SIM_BUILD=sim_build_isqrt TOPLEVEL=fixed_isqrt COCOTB_TEST_MODULES=cocotb_fixed_isqrt VERILOG_SOURCES="$(PWD)/fixed_isqrt.sv"

wave: test
	gtkwave $(WAVEFILE)

clean::
	rm -f $(WAVEFILE)

include $(shell python3 -m cocotb_tools.config --makefiles)/Makefile.sim
