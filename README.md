# Hush - Hardware-Accelerated Whisper Encoder

Hush is a hardware quantisation framework for OpenAI's Whisper speech recognition model. It provides configurable fixed-point implementations of encoder layers in both Python (bit-accurate models) and SystemVerilog (RTL), along with an automated design space exploration tool using Simulated Annealing.

## Project Structure

```
hush/
├── src/hush/hardware/  # RTL implementations & cocotb testbenches
│   ├── conv1d/         # Conv1d MAC array
│   ├── linear/         # Linear projection MAC array
│   ├── layernorm/      # LayerNorm (affine)
│   ├── layernorm_no_affine/
│   ├── positional_encoding/
│   ├── passthrough/    # Layer-to-Layer testing
│   └── ddr/            # DDR controller (experimental)
├── whisper/            # Forked OpenAI Whisper with quantisation support
│   ├── whisper/
│   │   ├── model.py    # Config-driven encoder (hardware/software switching)
│   │   └── quantise.py # Conv1dInteger, LinearInteger, quantisation utilities
│   ├── simple_whisper_transcribe.py # Inference & WER benchmarking
│   └── anneal.py       # Simulated annealing over quantisation configs
└── docs/               # Research notes
```

## Quick Start

### Transcription
```bash
python whisper/simple_whisper_transcribe.py whisper/daveL.wav
```

### WER Benchmark (LibriSpeech test-clean)
```bash
python whisper/simple_whisper_transcribe.py --test-wer
```

### Quantisation Design Space Exploration
```bash
python whisper/anneal.py
```

### Hardware Tests
Each RTL module has a Makefile that runs cocotb tests via Verilator:
```bash
cd src/hush/hardware/linear && make
cd src/hush/hardware/conv1d && make
cd src/hush/hardware/layernorm && make
cd src/hush/hardware/positional_encoding && make
cd src/hush/hardware/ddr && make # experimental controller
```

### Synthesis (Vivado)
```bash
vivado -mode batch -source fmax.tcl
```

## Quantisation Configuration

The encoder is configured via a single Python dictionary that controls which blocks use fixed-point arithmetic and at what precision:

```python
encoder_config = {
    "conv1d": "quantised",
    "conv1d_config": {
        "data_in_width": 10, "data_in_frac_width": 4,
        "weight_width": 14, "weight_frac_width": 14,
        "bias_width": 15, "bias_frac_width": 8,
    },
    "attention": "quantised",
    "attention_config": { ... },
    "layernorm": "quantised",
    "layernorm_config": { ... },
}
model = load_model("tiny.en", encoder_config=encoder_config)
```

Setting any block to "float" (or omitting it entirely) falls back to standard PyTorch 32-bit floating-point precision.

## Dev Setup

### Prerequisites
- [Nix](https://nixos.org/download/) with flakes enabled
- [direnv](https://direnv.net/) (recommended)
- Verilator (for RTL simulation)
- Vivado (optional, for synthesis)

### Getting started
```bash
direnv allow
```
First run sets up a Python venv with PyTorch and dependencies. To rebuild from scratch, delete `.venv/` and re-enter the directory!

### Without direnv
```bash
nix develop
```

## Authors
Oskar Bushrod, Chiedozie Ihebuzor, Jamie Mitchell, Ronit Ravi
