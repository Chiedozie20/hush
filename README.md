# hush

This project aims to port OpenAI's Whisper model onto FPGA hardware

Initial research in [docs/research.md](./docs/research.md)


## Running instructions 
for an example, input a 16 bit 16 kHz .wav file 
``` bash 
python whisper/simple_whisper_transcribe.py whisper/daveL.wav 
```
The corect output should show 
> Being able to communicate positions within a room is critical to our ability to focus light in a certain area or play subjects in their proper location on stage, but it goes even deeper than that. This proficiency provides the basic vocabulary in a common language that is spoken by production and staging professionals around the world.

## Dev setup

### Prerequisites
- [Nix](https://nixos.org/download/) with flakes enabled
- [direnv](https://direnv.net/) (recommended for not wasting time)

### Getting started

1. Clone [mase](https://github.com/DeepWok/mase) alongside this repo:
   ```
   Top/
   ├── mase/
   └── hush/
   ```

2. Create a `.envrc.local` in this repo:
   ```bash
   export MASE_PATH=../mase
   ```

3. Allow direnv:
   ```bash
   direnv allow
   ```
   This sets up a Python venv, installs PyTorch (with CUDA on Linux), and installs MASE if `MASE_PATH` is set. First run takes a while :(

4. To rebuild the environment from scratch, delete `.venv/` and re-enter the directory

### Without direnv
```bash
nix develop
```
