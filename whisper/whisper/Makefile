SIM ?= icarus
TOPLEVEL_LANG := verilog
TOPLEVEL := passthrough

MAKEFILE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PACKAGE_ROOT := $(abspath $(MAKEFILE_DIR)/..)

# The cocotb test lives in whisper/model.py and relies on package-relative imports.
MODULE := whisper.model
VERILOG_SOURCES := $(MAKEFILE_DIR)/passthrough.sv

export PYTHONPATH := $(PACKAGE_ROOT):$(PYTHONPATH)
WAVEFILE := passthrough.vcd

COCOTB_MAKEFILES := $(shell cocotb-config --makefiles 2>/dev/null)


.PHONY: all test wave clean

all: test
test: sim

wave: test
	gtkwave $(WAVEFILE)

clean::
	rm -f $(WAVEFILE)

include $(COCOTB_MAKEFILES)/Makefile.sim
