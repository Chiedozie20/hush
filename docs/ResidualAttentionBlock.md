Breakdown of the ResidualAttentionBlock (RAB)

### Input
The RAB takes in the mel spectrogram of the audio + the sinusoidal positional embedding, which encode the position of each mel block

x.shape = torch.Size([1, 80, 3000]) 
80 frequency bins
30 seconds split into 3000 chunks 

after filters, this is torch.Size([1, 1500, 384])

### layers

No cross attention in the encoder, (it is used in the decoder only)

```python 
input x
	↓
LayerNorm	
	↓
MHA "with mask" 
	↓ + x  # residual skip
LayerNorm
	↓
  Linear
	↓
   GELU
	↓
  Linear
	↓ + x  # residual skip
  Output 
```

#### Layer Norm
