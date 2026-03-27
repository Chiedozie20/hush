SIM ?= verilator
.DEFAULT_GOAL := sim
.PHONY: waves

TOPLEVEL_LANG := verilog
TOPLEVEL := positional_encoding
MODULE := encoding_tb
WAVEFILE := dump.fst

VERILOG_SOURCES := \
	$(PWD)/positional_encoding.sv \
	$(PWD)/frequency_index_rom.sv \
	$(PWD)/sinusoid_rom.sv

ROM_GEN := $(PWD)/generate_encoding_mem.py
ROM_FILES := $(PWD)/freq_lut.mem $(PWD)/sin_lut.mem
CUSTOM_COMPILE_DEPS += $(ROM_FILES)

export PYTHONPATH := $(PWD):$(PYTHONPATH)

$(ROM_FILES): $(ROM_GEN)
	python3 $(ROM_GEN)

WAVES ?= 1
ifeq ($(WAVES),1)
  TRACE_ARGS = --trace --trace-fst --trace-structs
endif
EXTRA_ARGS += $(TRACE_ARGS) -Wno-fatal

waves: $(WAVEFILE) 
	gtkwave $(WAVEFILE) &

include $(shell cocotb-config --makefiles)/Makefile.sim
