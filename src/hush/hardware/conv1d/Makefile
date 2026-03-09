# Makefile for Conv1d Hardware Simulation

# Simulator choice: verilator or iverilog
SIM ?= iverilog

# Source files
RTL_SRC = conv1d_complete.sv
TB_SRC = conv1d_tb.sv

# Output files
VVP_OUT = conv1d_tb.vvp
VCD_OUT = conv1d_tb.vcd

# Verilator settings
VERILATOR = verilator
VERILATOR_FLAGS = --binary --timing --trace --build -Wall

# Icarus Verilog settings
IVERILOG = iverilog
IVERILOG_FLAGS = -g2012 -o $(VVP_OUT)
VVP = vvp

.PHONY: all clean sim wave help

# Default target
all: sim

# Help target
help:
	@echo "Available targets:"
	@echo "  make sim        - Run simulation with iverilog (default)"
	@echo "  make wave       - Open waveform viewer (GTKWave)"
	@echo "  make clean      - Clean generated files"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  SIM=iverilog    - Use Icarus Verilog (default)"
	@echo "  SIM=verilator   - Use Verilator (if installed)"

# Simulate with Icarus Verilog
sim_iverilog: $(RTL_SRC) $(TB_SRC)
	@echo "Compiling with Icarus Verilog..."
	$(IVERILOG) $(IVERILOG_FLAGS) $(RTL_SRC) $(TB_SRC)
	@echo "Running simulation..."
	$(VVP) $(VVP_OUT)
	@echo "Simulation complete! Waveform saved to $(VCD_OUT)"

# Simulate with Verilator
sim_verilator: $(RTL_SRC) $(TB_SRC)
	@echo "Compiling with Verilator..."
	$(VERILATOR) $(VERILATOR_FLAGS) --top-module conv1d_tb $(RTL_SRC) $(TB_SRC)
	@echo "Running simulation..."
	./obj_dir/Vconv1d_tb
	@echo "Simulation complete!"

# Auto-detect simulator
sim:
ifeq ($(SIM),verilator)
	@make sim_verilator
else
	@make sim_iverilog
endif

# Open waveform viewer
wave: $(VCD_OUT)
	@if command -v gtkwave >/dev/null 2>&1; then \
		echo "Opening waveform in GTKWave..."; \
		gtkwave $(VCD_OUT) &; \
	else \
		echo "GTKWave not found. Install with: sudo apt install gtkwave"; \
		exit 1; \
	fi

# Clean generated files
clean:
	rm -rf $(VVP_OUT) $(VCD_OUT) obj_dir *.log *.vcd
	@echo "Cleaned generated files"

# Check if simulators are installed
check:
	@echo "Checking for installed simulators..."
	@command -v $(IVERILOG) >/dev/null 2>&1 && echo "  ✓ Icarus Verilog found" || echo "  ✗ Icarus Verilog not found"
	@command -v $(VERILATOR) >/dev/null 2>&1 && echo "  ✓ Verilator found" || echo "  ✗ Verilator not found"
	@command -v gtkwave >/dev/null 2>&1 && echo "  ✓ GTKWave found" || echo "  ✗ GTKWave not found"
