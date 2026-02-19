# Research

## Contents
[Introduction](#introduction)  
[Useful links](#useful-links)  
[Benchmarking](#benchmarking)  
[Sections](#sections)

## Introduction

## Useful links

### Open AI Whisper 
[Github](https://github.com/openai/whisper)

[Paper](https://arxiv.org/abs/2212.04356)

### Moonshine
[Moonshine Github](https://github.com/moonshine-ai/moonshine)

[Moonshine paper](https://arxiv.org/abs/2410.15608)

Moonshine is speach recognition tool tailored for live translation, smaller run faster than whispers smaller models

### ggml whisper
[Repo](https://github.com/ggml-org/whisper.cpp)
C++ zero dependancy implementation


### Other links
[Low power fpga techniques paper](https://www.doc.ic.ac.uk/~wl/papers/08/ahs08jl.pdf)

## Benchmarking

## Sections

Open AI whisper model
```
WhisperForConditionalGeneration(
  (model): WhisperModel(
    (encoder): WhisperEncoder(
      (conv1): Conv1d(80, 384, kernel_size=(3,), stride=(1,), padding=(1,))
      (conv2): Conv1d(384, 384, kernel_size=(3,), stride=(2,), padding=(1,))
      (embed_positions): Embedding(1500, 384)
      (layers): ModuleList(
        (0-3): 4 x WhisperEncoderLayer(
          (self_attn): WhisperSdpaAttention(
            (k_proj): Linear(in_features=384, out_features=384, bias=False)
            (v_proj): Linear(in_features=384, out_features=384, bias=True)
            (q_proj): Linear(in_features=384, out_features=384, bias=True)
            (out_proj): Linear(in_features=384, out_features=384, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (activation_fn): GELUActivation()
          (fc1): Linear(in_features=384, out_features=1536, bias=True)
          (fc2): Linear(in_features=1536, out_features=384, bias=True)
          (final_layer_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layer_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
    )
    (decoder): WhisperDecoder(
      (embed_tokens): Embedding(51865, 384, padding_idx=50257)
      (embed_positions): WhisperPositionalEmbedding(448, 384)
      (layers): ModuleList(
        (0-3): 4 x WhisperDecoderLayer(
          (self_attn): WhisperSdpaAttention(
            (k_proj): Linear(in_features=384, out_features=384, bias=False)
            (v_proj): Linear(in_features=384, out_features=384, bias=True)
            (q_proj): Linear(in_features=384, out_features=384, bias=True)
            (out_proj): Linear(in_features=384, out_features=384, bias=True)
          )
          (activation_fn): GELUActivation()
          (self_attn_layer_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (encoder_attn): WhisperSdpaAttention(
            (k_proj): Linear(in_features=384, out_features=384, bias=False)
            (v_proj): Linear(in_features=384, out_features=384, bias=True)
            (q_proj): Linear(in_features=384, out_features=384, bias=True)
            (out_proj): Linear(in_features=384, out_features=384, bias=True)
          )
          (encoder_attn_layer_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=384, out_features=1536, bias=True)
          (fc2): Linear(in_features=1536, out_features=384, bias=True)
          (final_layer_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layer_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
    )
  )
  (proj_out): Linear(in_features=384, out_features=51865, bias=False)
)
```
