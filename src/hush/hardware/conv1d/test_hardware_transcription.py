import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent.parent
whisper_path = project_root / "whisper"
sys.path.insert(0, str(whisper_path))
import cocotb
import torch
import torch.nn.functional as F
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly
from whisper import load_model
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
from whisper.decoding import DecodingOptions, DecodingResult
SCALE_FACTOR = 100.0
DATA_WIDTH = 16
def quantise_to_int16(tensor, scale=SCALE_FACTOR):
    quantised = (tensor * scale).clamp(-32768, 32767).round().int()
    return quantised

def quantise_bias_to_int16(tensor, scale=SCALE_FACTOR):
    quantised = (tensor * scale * scale).clamp(-2147483648, 2147483647).round().int()
    return quantised

def dequantise_from_int32(tensor, scale=SCALE_FACTOR):
    return tensor.float() / (scale * scale)

def benchmark_WER(model):
    dataset = LibriSpeech("test-clean")
    dataset = torch.utils.data.Subset(dataset, range(10))
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    hypotheses = []
    references = []
    for mels, texts in tqdm(loader):
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)
    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
    normalizer = EnglishTextNormalizer()
    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]
    wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    return wer

async def load_weights(dut, weights, in_channels, out_channels, kernel_size):
    total = out_channels * in_channels * kernel_size
    for out_ch in range(out_channels):
        for in_ch in range(in_channels):
            for k in range(kernel_size):
                await FallingEdge(dut.clk)
                weight_val = int(weights[out_ch, in_ch, k].item())
                dut.weight_in.value = weight_val
                dut.weight_valid.value = 1
                dut.weight_out_ch.value = out_ch
                dut.weight_in_ch.value = in_ch
                dut.weight_k_idx.value = k
                await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.weight_valid.value = 0
    await RisingEdge(dut.clk)
    print(f"  Weights loaded: {total}/{total} (100.0%)")

async def load_biases(dut, biases, out_channels):
    for out_ch in range(out_channels):
        await FallingEdge(dut.clk)
        bias_val = int(biases[out_ch].item())
        dut.bias_in.value = bias_val
        dut.bias_valid.value = 1
        dut.bias_out_ch.value = out_ch
        await RisingEdge(dut.clk)

    await FallingEdge(dut.clk)
    dut.bias_valid.value = 0
    await RisingEdge(dut.clk)
    print(f"  Biases loaded: {out_channels}/{out_channels}")

async def load_input(dut, input_tensor, in_channels):
    batch, in_ch, seq_len = input_tensor.shape
    assert batch == 1
    assert in_ch == in_channels

    total = in_ch * seq_len
    count = 0
    for ch in range(in_ch):
        for pos in range(seq_len):
            await FallingEdge(dut.clk)
            data_val = int(input_tensor[0, ch, pos].item())
            dut.data_in.value = data_val
            dut.data_in_valid.value = 1
            dut.in_channel_idx.value = ch
            dut.in_pos_idx.value = pos
            await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.data_in_valid.value = 0
    await RisingEdge(dut.clk)
    print(f"  Input loaded: {total}/{total} (100.0%)")

async def read_output(dut, out_channels, output_length):
    acc_width = 2 * DATA_WIDTH

    outputs = []
    total = out_channels * output_length
    last_percent = 0

    while len(outputs) < total and int(dut.done.value) == 0:
        await RisingEdge(dut.clk)
        await ReadOnly()

        if int(dut.data_out_valid.value):
            data_val = int(dut.data_out.value)
            out_ch = int(dut.out_channel_idx.value)
            out_pos = int(dut.out_pos_idx.value)
            if data_val & (1 << (acc_width - 1)):
                data_val -= (1 << acc_width)

            outputs.append((out_ch, out_pos, data_val))

    output_tensor = torch.zeros(1, out_channels, output_length, dtype=torch.int64)
    skipped = 0
    for out_ch, out_pos, val in outputs:
        if out_pos >= output_length:
            skipped += 1
            continue
        output_tensor[0, out_ch, out_pos] = val

    print(f"  Output read: {len(outputs)-skipped}/{total} ({100*(len(outputs)-skipped)/total:.1f}%)")
    return output_tensor

async def apply_gelu_hardware(dut, conv_output_q):
    batch, channels, length = conv_output_q.shape
    gelu_output_q = torch.zeros_like(conv_output_q)

    print(f"  Applying hardware GELU to {batch}x{channels}x{length} tensor...")

    total_values = batch * channels * length
    processed = 0

    for b in range(batch):
        for c in range(channels):
            for l in range(length):
                val_int = int(conv_output_q[b, c, l].item())

                await FallingEdge(dut.clk)
                dut.gelu_data_in.value = val_int
                dut.gelu_data_in_valid.value = 1
                await RisingEdge(dut.clk)

                await FallingEdge(dut.clk)
                dut.gelu_data_in_valid.value = 0

                cycles = 0
                while int(dut.gelu_data_out_valid.value) == 0 and cycles < 20:
                    await RisingEdge(dut.clk)
                    cycles += 1
                    await ReadOnly()

                if int(dut.gelu_data_out_valid.value):
                    hw_out_int = int(dut.gelu_data_out.value)

                    if hw_out_int & (1 << 31):
                        hw_out_int -= (1 << 32)

                    gelu_output_q[b, c, l] = hw_out_int
                else:
                    print(f"  ERROR: No valid GELU output for value {val_int}")
                    gelu_output_q[b, c, l] = 0

                await RisingEdge(dut.clk)

                processed += 1
                if processed % 100000 == 0:
                    print(f"    Processed {processed}/{total_values} values through GELU...")

    print(f"  Hardware GELU complete: {processed} values processed")
    return gelu_output_q

@cocotb.test()
async def test_hardware_full_transcription(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    dut.rst_n.value = 0
    dut.start.value = 0
    dut.in_channels_cfg.value = 0
    dut.out_channels_cfg.value = 0
    dut.stride_cfg.value = 0
    dut.data_in.value = 0
    dut.data_in_valid.value = 0
    dut.in_channel_idx.value = 0
    dut.in_pos_idx.value = 0
    dut.weight_in.value = 0
    dut.weight_valid.value = 0
    dut.weight_out_ch.value = 0
    dut.weight_in_ch.value = 0
    dut.weight_k_idx.value = 0
    dut.bias_in.value = 0
    dut.bias_valid.value = 0
    dut.bias_out_ch.value = 0
    dut.data_out_ready.value = 1
    dut.input_length.value = 0
    dut.gelu_data_in.value = 0
    dut.gelu_data_in_valid.value = 0
    dut.gelu_data_out_ready.value = 1

    for _ in range(5):
        await RisingEdge(dut.clk)

    dut.rst_n.value = 1

    for _ in range(5):
        await RisingEdge(dut.clk)

    device = "cpu"
    model = load_model("tiny.en", device=device)

    conv1_weight = model.encoder.conv1.weight.data
    conv1_bias = model.encoder.conv1.bias.data
    conv2_weight = model.encoder.conv2.weight.data
    conv2_bias = model.encoder.conv2.bias.data

    audio_path = whisper_path / "tests" / "jfk.flac"
    audio = load_audio(str(audio_path))
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    mel_input = mel.unsqueeze(0)

    print("HARDWARE PIPELINE: Conv1→GELU→Conv2→GELU")
    print("\n[1/4] Running Conv1 in hardware...")
    conv1_out_ch, conv1_in_ch, kernel_size = conv1_weight.shape
    stride1 = 1
    input_length = mel_input.shape[2]
    output_length = (input_length + 2 - kernel_size) // stride1 + 1

    dut.in_channels_cfg.value = conv1_in_ch
    dut.out_channels_cfg.value = conv1_out_ch
    dut.stride_cfg.value = stride1
    dut.input_length.value = input_length

    conv1_weights_q = quantise_to_int16(conv1_weight)
    conv1_biases_q = quantise_bias_to_int16(conv1_bias)
    mel_q = quantise_to_int16(mel_input)

    await FallingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    while int(dut.busy.value) == 0:
        await RisingEdge(dut.clk)

    await load_weights(dut, conv1_weights_q, conv1_in_ch, conv1_out_ch, kernel_size)
    await load_biases(dut, conv1_biases_q, conv1_out_ch)
    await load_input(dut, mel_q, conv1_in_ch)

    hw_conv1_out_q = await read_output(dut, conv1_out_ch, output_length)

    print("\n[2/4] Applying first GELU in hardware...")
    hw_gelu1_out_q = await apply_gelu_hardware(dut, hw_conv1_out_q)

    print("\n[3/4] Running Conv2 in hardware...")
    dut.rst_n.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)

    conv2_out_ch, conv2_in_ch, kernel_size2 = conv2_weight.shape
    stride2 = 2
    conv2_input_length = output_length
    conv2_output_length = (conv2_input_length + 2 - kernel_size2) // stride2 + 1

    dut.in_channels_cfg.value = conv2_in_ch
    dut.out_channels_cfg.value = conv2_out_ch
    dut.stride_cfg.value = stride2
    dut.input_length.value = conv2_input_length

    hw_gelu1_out_float = dequantise_from_int32(hw_gelu1_out_q)
    hw_gelu1_out_16 = quantise_to_int16(hw_gelu1_out_float)

    conv2_weights_q = quantise_to_int16(conv2_weight)
    conv2_biases_q = quantise_bias_to_int16(conv2_bias)

    await FallingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    while int(dut.busy.value) == 0:
        await RisingEdge(dut.clk)

    await load_weights(dut, conv2_weights_q, conv2_in_ch, conv2_out_ch, kernel_size2)
    await load_biases(dut, conv2_biases_q, conv2_out_ch)
    await load_input(dut, hw_gelu1_out_16, conv2_in_ch)

    hw_conv2_out_q = await read_output(dut, conv2_out_ch, conv2_output_length)

    print("\n[4/4] Applying second GELU in hardware...")
    hw_gelu2_out_q = await apply_gelu_hardware(dut, hw_conv2_out_q)
    hw_gelu2_out = dequantise_from_int32(hw_gelu2_out_q)

    print(f"\nHardware frontend output shape: {hw_gelu2_out.shape}")
    print(f"Hardware frontend output range: [{hw_gelu2_out.min():.4f}, {hw_gelu2_out.max():.4f}]")

    print("RUNNING FULL WHISPER TRANSCRIPTION WITH HARDWARE FRONTEND")
    hw_output_transposed = hw_gelu2_out.permute(0, 2, 1)

    print(f"\nHardware output (transposed): {hw_output_transposed.shape}")
    positional_embedding = model.encoder.positional_embedding[:hw_output_transposed.shape[1]]
    encoder_input = (hw_output_transposed + positional_embedding).to(hw_output_transposed.dtype)

    print(f"After positional embedding: {encoder_input.shape}")
    print("\nRunning through transformer blocks...")
    x = encoder_input
    for block in model.encoder.blocks:
        x = block(x)

    encoder_output_hw = model.encoder.ln_post(x)
    with torch.no_grad():
        encoder_output_sw = model.encoder(mel_input)

    encoder_diff = (encoder_output_hw - encoder_output_sw).abs()
    print(f"\nEncoder output comparison (Hardware vs Software):")
    print(f"  Shape: {encoder_output_hw.shape}")
    print(f"  Max diff:    {encoder_diff.max():.6f}")
    print(f"  Mean diff:   {encoder_diff.mean():.6f}")
    print(f"  Median diff: {encoder_diff.median():.6f}")

    print("RUNNING DECODER FOR TRANSCRIPTION")

    print("\nDecoding with HARDWARE encoder output...")
    options = DecodingOptions(language="en", without_timestamps=True)

    from whisper.decoding import decode as whisper_decode

    original_embed_audio = model.embed_audio
    def custom_embed_audio(mel):
        return encoder_output_hw

    model.embed_audio = custom_embed_audio
    result_hw = whisper_decode(model, mel_input, options)[0]
    model.embed_audio = original_embed_audio

    result_sw = whisper_decode(model, mel_input, options)[0]
    print("TRANSCRIPTION COMPARISON")

    print(f"\nHardware: {result_hw.text}")
    print(f"Software: {result_sw.text}")

    hw_words = result_hw.text.strip().lower().split()
    sw_words = result_sw.text.strip().lower().split()

    matching_words = sum(1 for hw, sw in zip(hw_words, sw_words) if hw == sw)
    total_words = max(len(hw_words), len(sw_words))
    word_accuracy = matching_words / total_words if total_words > 0 else 0.0

    print(f"\nWord accuracy: {word_accuracy*100:.1f}% ({matching_words}/{total_words} words match)")
    print(f"Exact match: {'YES' if result_hw.text == result_sw.text else 'NO'}")

    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    print(f"\nEncoder output accuracy:")
    print(f"  Max diff:  {encoder_diff.max():.6f}")
    print(f"  Mean diff: {encoder_diff.mean():.6f}")
    print(f"  Percentage diff < 0.1: {(encoder_diff < 0.1).float().mean()*100:.2f}%")

    print(f"\nTranscription accuracy:")
    print(f"  Word accuracy: {word_accuracy*100:.1f}%")
    print(f"  Exact match: {result_hw.text == result_sw.text}")

    assert encoder_diff.max() < 5.0, f"Encoder max diff too large: {encoder_diff.max()}"
    assert encoder_diff.mean() < 0.1, f"Encoder mean diff too large: {encoder_diff.mean()}"
    assert word_accuracy >= 0.9, f"Word accuracy too low: {word_accuracy*100:.1f}%"

    print("HARDWARE TRANSCRIPTION TEST PASSED!")
    print(f"Hardware Conv1→GELU→Conv2→GELU frontend works end-to-end")
    print(f"Full transcription pipeline validated")
    print(f"Transcription: '{result_hw.text}'")
