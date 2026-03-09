#!/bin/bash
# Script to run Verilator simulation

set -e

echo "========================================"
echo "  Conv1d Verilator Simulation"
echo "========================================"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf obj_dir *.vcd

# Run Verilator
echo "Running Verilator..."
verilator --cc \
    --trace \
    -Wno-fatal \
    -Wno-WIDTHTRUNC \
    -Wno-WIDTHEXPAND \
    -Wno-PROCASSINIT \
    -Wno-UNUSEDSIGNAL \
    -Wno-CASEINCOMPLETE \
    --top-module conv1d_tb_verilator \
    --exe sim_main.cpp \
    conv1d_complete.sv \
    conv1d_tb_verilator.sv

# Build
echo "Building..."
make -C obj_dir -f Vconv1d_tb_verilator.mk

# Run simulation
echo "Running simulation..."
./obj_dir/Vconv1d_tb_verilator

echo ""
echo "========================================"
echo "  Simulation Complete!"
echo "========================================"
echo ""
echo "To view waveforms:"
echo "  gtkwave conv1d_tb_verilator.vcd"
echo ""
