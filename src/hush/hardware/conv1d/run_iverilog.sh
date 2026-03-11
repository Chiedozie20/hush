#!/bin/bash
# Script to run Icarus Verilog simulation

set -e

echo "========================================"
echo "  Conv1d Icarus Verilog Simulation"
echo "========================================"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf *.vvp *.vcd

# Compile with Icarus Verilog
echo "Compiling with iverilog..."
iverilog -g2012 \
    -o conv1d_tb.vvp \
    conv1d_complete.sv \
    conv1d_tb_verilator.sv

# Run simulation
echo "Running simulation..."
vvp conv1d_tb.vvp

echo ""
echo "========================================"
echo "  Simulation Complete!"
echo "========================================"
echo ""
echo "To view waveforms:"
echo "  gtkwave conv1d_tb_verilator.vcd"
echo ""
