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

## Computational flow broken down 

**main()**

**load_weights()**

**prepare_mel()**

**_get_audio_features(mel)**
	model.encoder(mel) [AudioEncoder]
  
  conv1d -> gelu -> conv1d ->gelu -> quantise -> LayerNorm -> return audio_features

get start of transcript tokens

**decode(sot_tokens, audio features)**
``` python
# get start of sequence embeddings from dictionary
x = self.token_embedding(tokens) + self.pos_embedding[:t]
# mask makes matrix with -inf in top half to stop future attention
mask = self.causal_mask(t, x.device)

# begin blocks (x, audio_features, mask)
# loop though the folowing 4 times
start
# Layer norm -> self attn
x = x + self.self_attn(self.ln1(x), causal_mask=causal_mask)
# Layer norm -> cross attn
x = x + self.cross_attn(self.ln2(x), xa=audio_features)
# Layer norm -> mlp 
x = x + self.mlp(self.ln3(x))
GOTO start
# end blocks

x = self.ln(x)
logits = x @ self.token_embedding.weight.T
return logits
```

arg_max the logits to get the token

decode the tokens ->
	simple dictionary 

### What is layer norm 
$$y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta$$

### what is self_attn

input (x, audio_features)
nn.Linear = $y = xA^T + b$

``` python
# q, k, v are nn.Linear(in_features = n_state, out_features = n_state)
q = self._split_heads(self.q(x))
k = self._split_heads(self.k(audio_features))
v = self._split_heads(self.v(audio_features))

scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

attn = F.softmax(scores, dim=-1)
y = torch.matmul(attn, v)
y = self._merge_heads(y)

self.out = nn.Linear(n_state, n_state)
y = self._merge_heads(y)
return self.out(y)

```


### what is cross_attn
``` python
# q, k, v are nn.Linear(in_features = n_state, out_features = n_state)
q = self._split_heads(self.q(x))
k = self._split_heads(self.k(source))
v = self._split_heads(self.v(source))

scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

scores = scores + causal_mask
attn = F.softmax(scores, dim=-1)
y = torch.matmul(attn, v)
y = self._merge_heads(y)

self.out = nn.Linear(n_state, n_state)
y = self._merge_heads(y)
return self.out(y)

```

### what is MLP
``` python 
self.mlp = nn.Sequential(

nn.Linear(n_state, 4 * n_state),

nn.GELU(),

nn.Linear(4 * n_state, n_state),

)
```

## Memory usage analysis
- look into how much memory each stage uses
- figure out how many layers can be held on fpga at a time 
- e.g hold x and q  and store new x - if we have more memory we can fetch k in this time and double-buffer


### GPT Model Size Analysis

For `tiny.en`

**Core dims (`tiny.en`)**
- `n_mels=80`
- `n_audio_ctx=1500`
- `n_audio_state=384`
- `n_audio_head=6` → `head_dim=64`
- `n_audio_layer=4`
- `n_text_ctx=448`
- `n_text_state=384`
- `n_text_head=6` → `head_dim=64`
- `n_text_layer=4`
- `n_vocab=51864`

**Mel / audio pipeline**
- 30s audio samples: `480000`
- Mel frames: `3000`
- Mel filterbank: `80 x 201` (`whisper/assets/mel_filters.npz`)
- Log-mel input to encoder: `[B, 80, 3000]`
- After `conv1 (80→384, k=3)`: `[B, 384, 3000]`
- After `conv2 (384→384, k=3, stride=2)`: `[B, 384, 1500]`
- Encoder sequence output: `[B, 1500, 384]`

**Q/K/V matrices (all attention modules)**
Each attention module in Whisper uses:
- `Q`: weight `[384, 384]` + bias `[384]`
- `K`: weight `[384, 384]` (no bias)
- `V`: weight `[384, 384]` + bias `[384]`
- `Out`: weight `[384, 384]` + bias `[384]`

Per matrix sizes:
- One `384x384` weight = `147,456` params
- One such weight storage:
  - FP16: `0.28125 MiB`
  - FP32: `0.5625 MiB`

Per attention module total params:
- `590,976` params (`Q+K+V+Out` + biases where present)

Runtime attention tensor shapes:
- Encoder self-attn: `q,k,v` projected as `[B,1500,384]`, reshaped `[B,6,1500,64]`
- Decoder self-attn (token length `T`): `[B,6,T,64]`
- Decoder cross-attn:
  - `q`: `[B,6,T,64]`
  - `k,v` from audio: `[B,6,1500,64]`
  - attention scores: `[B,6,T,1500]`

**Layer parameter counts**
- `encoder.conv1`: `92,544`
- `encoder.conv2`: `442,752`
- One encoder block: `1,774,080`
- All 4 encoder blocks: `7,096,320`
- Encoder total: `7,632,384`

- `decoder.token_embedding`: `19,915,776`
- `decoder.positional_embedding`: `172,032`
- One decoder block (self-attn + cross-attn + MLP + norms): `2,365,824`
- All 4 decoder blocks: `9,463,296`
- Decoder total: `29,551,872`

**Total model params**
- `37,184,256` params

Approx weight size:
- FP16: `70.92 MiB`
- FP32: `141.85 MiB`




## Components In MASE 
- conv1d ❌
- gelu ✅
- quantiser ✅ (fixed_cast)
- layer norm ✅
- embedding hash table ❌
- self attention ✅
- cross attention 
- linear ✅
- softmax ✅
- mat mul ✅
- log-mel unit ❌


### Log-Mel Generation
NFFT = 400
HOP_LENGTH = 160
1. Convert input audio into an array of samples 
2. Hann-Window $w[n] = \frac{1}{2}\ \left[1 - \cos \left( \frac{2 \pi n}{N - 1} \right)\right] =\sin^2 \left( \frac{\pi n}{N - 1} \right)$ 
	$y[n] = x[n] \cdot w[n]$
	- Lookup table memory with window coefficients 
	- Simple element wise multiplication block can 
3. STFT (Short time Fourier transform ) 
	 Compute the FFT on different overlapping windows
4. Mat Mul With mel filter-bank 
5. Take the log of each element



