TOPLEVEL = linear
MODULE = test_linear

SIM = verilator
TOPLEVEL_LANG = verilog

VERILOG_SOURCES = linear.sv

WAVES ?= 1
ifeq ($(WAVES),1)
  TRACE_ARGS = --trace --trace-fst --trace-structs
endif
EXTRA_ARGS += $(TRACE_ARGS) -Wno-fatal

include $(shell cocotb-config --makefiles)/Makefile.sim

