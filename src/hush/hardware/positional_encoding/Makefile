SIM ?= icarus
TOPLEVEL_LANG := verilog
TOPLEVEL := positional_encoding
MODULE := encoding_tb

VERILOG_SOURCES := \
	$(PWD)/possitional_encoding.sv \
	$(PWD)/frequency_index_rom.sv \
	$(PWD)/sinusoid_rom.sv

export PYTHONPATH := $(PWD):$(PYTHONPATH)
WAVEFILE ?= positional_encoding.vcd

.PHONY: all test wave clean

all: test

test: sim

wave: test
	gtkwave $(WAVEFILE)

clean::
	rm -rf sim_build __pycache__
	rm -f $(WAVEFILE)

include $(shell cocotb-config --makefiles)/Makefile.sim
