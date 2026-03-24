TOPLEVEL = ddr_controller
MODULE = test_ddr_controller

SIM = verilator
TOPLEVEL_LANG = verilog

VERILOG_SOURCES = ../../rtl/ddr_controller.sv

# Waveforms on by default, disabled with WAVES=0
WAVES ?= 1
ifeq ($(WAVES),1)
  TRACE_ARGS = --trace --trace-fst --trace-structs
endif
EXTRA_ARGS += $(TRACE_ARGS) -Wno-fatal

include $(shell cocotb-config --makefiles)/Makefile.sim

