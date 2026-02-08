# Bitwav Encoder

A high-performance, GPU-optimized audio encoder package for Bitwav TTS. This package handles the conversion of raw audio waveforms into discrete tokens (via FSQ) and global style embeddings (SSL-based).

## 🚀 Features

- **Multiple Encoder Modes**:
    - `StandaloneEncoder`: Base implementation for straightforward inference.
    - `OptimizedStandaloneEncoder`: Leverages `torch.compile` and mixed precision (FP16/BF16) for maximum throughput.
    - `PipelinedEncoder`: Optimized for overlapping chunk inference.
    - `LongAudioEncoder`: Handles extremely long audio (> 100s) using smart chunking and overlap strategies.
- **WavLM Integration**: Built-in support for WavLM Base Plus as the SSL feature extractor.
- **Automatic Weights Management**: Transparently handles local weights or automatic downloads from Hugging Face (`humair025/enc`).
- **Precision Verified**: Negligible loss conversion (FP32 -> FP16) with 98.3%+ token parity.

## 📦 Installation

### Option 1: Direct Pip Install (Recommended)
You can install the package directly from GitHub:
```bash
pip install git+https://github.com/humair-m/enc.git
```

### Option 2: Clone and Install
```bash
git clone https://github.com/humair-m/enc.git
cd enc
pip install -e .
```

## 🛠 Usage

### 1. Putting Audio (Inputs)
The `encode` method is flexible and accepts multiple input formats:

- **File Path**: Just pass the string path to your `.wav` file.
- **Torch Tensor**: A tensor of shape `(channels, samples)` or `(samples,)`.
- **Numpy Array**: A numpy array of the audio waveform.

### 2. Basic Example

```python
from enc import create_standalone_encoder
import torchaudio

# 1. Initialize (Downloads weights automatically if needed)
encoder = create_standalone_encoder(device="cuda")

# 2. Encode from File Path
tokens, style = encoder.encode("path/to/audio.wav")

# 3. Encode from Tensor (e.g., already loaded)
waveform, sr = torchaudio.load("audio.wav")
tokens, style = encoder.encode(waveform, sr=sr)
```

### High-Performance (Optimized)

```python
from enc import create_optimized_encoder

encoder = create_optimized_encoder(
    device="cuda",
    compile_mode="max-autotune"
)

# First run will trigger torch.compile warmup
tokens, style_emb = encoder.encode(waveform, sr=24000)
```

### Long Audio Processing

```python
from enc import create_long_audio_encoder

encoder = create_long_audio_encoder(device="cuda")
tokens, style_emb = encoder.encode_long(waveform, sr=24000)
```

## 📊 Precision & Performance

Detailed precision analysis is available in [REPORT.md](./REPORT.md).

| Metric | Baseline (FP32) | Optimized (FP16) |
| --- | --- | --- |
| **Model Size** | ~380MB | ~190MB |
| **Style Similarity** | 1.0 | 0.99999 |
| **Token Match** | 100% | 98.35% |

## 🔗 Related Repositories

- **Model Weights**: [humair025/enc (Hugging Face)](https://huggingface.co/humair025/enc)
- **Main Project**: [humair-m/bitwav (GitHub)](https://github.com/humair-m/bitwav)

## License
MIT
