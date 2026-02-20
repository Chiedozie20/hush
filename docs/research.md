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


