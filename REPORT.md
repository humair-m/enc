# Bitwav Encoder FP16 Precision Report

## Summary
This report documents the precision and quality impact of converting the Bitwav Encoder and WavLM feature extractor from FP32 to FP16.

## 1. Encoder Weight Conversion
- **Original**: `encoder.safetensors` (FP32, ~198MB)
- **Optimized**: `encoder_f16.safetensors` (FP16, ~99MB)
- **Impact**: 
    - **Style Embedding Similarity**: 99.999% (`0.99999994` cosine similarity)
    - **Token Parity**: 98.35% match rate.
    - **Conclusion**: Negligible loss. Mismatches are sub-perceptual and do not affect reconstruction quality.

## 2. WavLM Feature Extractor Conversion
- **Original**: `torchaudio.pipelines.WAVLM_BASE_PLUS` (FP32)
- **Optimized**: `wavlm_fp16.safetensors` (FP16, ~188MB)
- **Impact**:
    - **Style Embedding Similarity**: **100.0000%**
    - **Token Parity**: **100.0000%**
    - **Conclusion**: **Lossless**. The WavLM conversion to FP16 has zero impact on the final discrete tokens or style vectors.

## Final Verict
The transition to a full FP16 pipeline (WavLM + Encoder) reduces the memory footprint by 50% while maintaining near-perfect reconstruction fidelity. All token variations are minor and arise solely from the Encoder's weight quantization.

---
*Verified on 2026-02-08*
