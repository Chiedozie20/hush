#!/usr/bin/env bash
# Helper script to run the GELU + Conv2 test
# Works around nix GLIBC compatibility issues

cd "$(dirname "$0")"
source ../../../../../.venv/bin/activate

# Use absolute path to nix to avoid GLIBC issues
/nix/var/nix/profiles/default/bin/nix develop /home/jmitch/adls/hush --command bash -c "make -f Makefile.cocotb SIM=verilator test_gelu_conv2"
