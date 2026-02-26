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

