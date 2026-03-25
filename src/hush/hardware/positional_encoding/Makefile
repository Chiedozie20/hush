SIM ?= verilator

TOPLEVEL_LANG := verilog
TOPLEVEL := positional_encoding
MODULE := encoding_tb

VERILOG_SOURCES := \
	$(PWD)/positional_encoding.sv \
	$(PWD)/frequency_index_rom.sv \
	$(PWD)/sinusoid_rom.sv

export PYTHONPATH := $(PWD):$(PYTHONPATH)

WAVES ?= 1
ifeq ($(WAVES),1)
  TRACE_ARGS = --trace --trace-fst --trace-structs
endif
EXTRA_ARGS += $(TRACE_ARGS) -Wno-fatal

include $(shell cocotb-config --makefiles)/Makefile.sim