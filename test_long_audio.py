"""
Test Long Audio Encoding and Reconstruction

Process long.wav with 15s chunks, save tokens/embeddings, then reconstruct.
"""

import torch
import soundfile as sf
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from encoder_long_audio import create_long_audio_encoder
from bitwav_api import BitwavDecoder, BitwavVocoder


def test_long_audio_reconstruction():
    """Test encoding and reconstruction of long audio."""
    
    input_file = "long.wav"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*70)
    print("Long Audio Reconstruction Test")
    print("="*70)
    
    # Load audio
    print(f"\n[*] Loading {input_file}...")
    audio, sr = sf.read(input_file)
    audio_tensor = torch.from_numpy(audio).float()
    duration = len(audio) / sr
    
    print(f"[+] Loaded: {duration:.2f}s, {sr}Hz, {audio.shape}")
    
    # Create encoder with 15s chunks
    print(f"\n[*] Creating long audio encoder (15s chunks, 2s overlap)...")
    encoder = create_long_audio_encoder(
        weights_path=None,
        device=device,
        compile_mode="max-autotune",
        dtype=torch.float16,
        chunk_duration=15.0,  # 15 second chunks
        overlap_duration=2.0,  # 2 second overlap
        warmup=True
    )
    
    # Encode
    print(f"\n[*] Encoding with chunking...")
    tokens, global_emb = encoder.encode(
        audio_tensor, 
        sr=sr, 
        merge_strategy="weighted"
    )
    
    print(f"[+] Encoded to {len(tokens)} tokens, embedding: {global_emb.shape}")
    
    # Save tokens and embedding
    print(f"\n[*] Saving tokens and embedding...")
    torch.save({
        'tokens': tokens.cpu(),
        'global_embedding': global_emb.cpu(),
        'sample_rate': sr,
        'original_duration': duration,
        'chunk_duration': 15.0,
        'overlap_duration': 2.0,
        'num_tokens': len(tokens),
        'encoder_version': 'long_audio_v1'
    }, "long_encoded.pt")
    print(f"[+] Saved to long_encoded.pt")
    
    # Decode to mel
    print(f"\n[*] Decoding to mel spectrogram...")
    decoder = BitwavDecoder("decoder_fp16.onnx", device="cpu")
    mel = decoder.decode(tokens.cpu().numpy(), global_emb.cpu().numpy())
    print(f"[+] Mel shape: {mel.shape}")
    
    # Vocode to audio
    print(f"\n[*] Vocoding to audio...")
    vocoder = BitwavVocoder("hift.onnx", device="cpu")
    reconstructed = vocoder.infer(mel)
    
    # Save reconstructed
    output_file = "long_reconstructed.wav"
    sf.write(output_file, reconstructed, 24000)
    print(f"[+] Saved reconstructed audio to {output_file}")
    
    # Stats
    recon_duration = len(reconstructed) / 24000
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Original duration:      {duration:.2f}s")
    print(f"Reconstructed duration: {recon_duration:.2f}s")
    print(f"Token count:            {len(tokens)}")
    print(f"Expected tokens:        ~{int(duration * 25)} (at 25 Hz)")
    print(f"Saved artifacts:")
    print(f"  - long_encoded.pt      (tokens + embedding)")
    print(f"  - {output_file}    (reconstructed audio)")
    print(f"{'='*70}")
    
    # Quality check
    duration_diff = abs(duration - recon_duration)
    if duration_diff < 1.0:
        print(f"\n✅ Duration match: {duration_diff:.2f}s difference (good)")
    else:
        print(f"\n⚠️  Duration mismatch: {duration_diff:.2f}s difference")
    
    token_rate = len(tokens) / duration
    if 24 <= token_rate <= 26:
        print(f"✅ Token rate: {token_rate:.2f} tokens/sec (expected ~25)")
    else:
        print(f"⚠️  Token rate: {token_rate:.2f} tokens/sec (expected ~25)")


if __name__ == "__main__":
    test_long_audio_reconstruction()
