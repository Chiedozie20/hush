SIM ?= icarus
TOPLEVEL_LANG = verilog
TOPLEVEL = passthrough
MODULE = cocotb_passthrough
VERILOG_SOURCES = $(PWD)/passthrough.sv

export PYTHONPATH := $(PWD):$(PYTHONPATH)
WAVEFILE = passthrough.vcd

.PHONY: all test wave clean

all: test
test: sim

wave: test
	gtkwave $(WAVEFILE)

clean::
	rm -f $(WAVEFILE)

include $(shell cocotb-config --makefiles)/Makefile.sim
