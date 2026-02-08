"""
Example: GPU-Optimized Encoder Usage

Demonstrates various usage patterns for the optimized encoder.
"""

import torch
import soundfile as sf
import time
from pathlib import Path
from encoder_optimized import create_optimized_encoder


def example_basic_usage():
    """Basic single-file encoding."""
    print("\n" + "="*70)
    print("Example 1: Basic Usage")
    print("="*70)
    
    # Create encoder (automatically compiles and warms up)
    encoder = create_optimized_encoder(
        weights_path=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_mode="max-autotune",
        dtype=torch.float16,
        warmup=True
    )
    
    # Load and encode audio
    audio, sr = sf.read("test_audio/comparison_llama.wav")
    audio_tensor = torch.from_numpy(audio).float()
    
    start = time.time()
    tokens, embedding = encoder.encode(audio_tensor, sr=sr)
    elapsed = time.time() - start
    
    print(f"Encoded in {elapsed:.4f}s")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Embedding shape: {embedding.shape}")
    
    return encoder


def example_batch_processing(encoder):
    """Batch encoding multiple files."""
    print("\n" + "="*70)
    print("Example 2: Batch Processing")
    print("="*70)
    
    # Load multiple audio files
    audio_dir = Path("test_audio")
    audio_files = list(audio_dir.glob("*.wav"))[:5]  # First 5 files
    
    waveforms = []
    for f in audio_files:
        audio, sr = sf.read(f)
        waveforms.append(torch.from_numpy(audio).float())
    
    print(f"Processing {len(waveforms)} files...")
    
    start = time.time()
    results = encoder.encode_batch(waveforms, sr=24000)
    elapsed = time.time() - start
    
    print(f"Batch encoded in {elapsed:.4f}s ({elapsed/len(waveforms):.4f}s per file)")
    
    for i, (tokens, emb) in enumerate(results):
        print(f"  File {i+1}: {tokens.shape[0]} tokens")


def example_different_dtypes():
    """Compare different precision modes."""
    print("\n" + "="*70)
    print("Example 3: Precision Comparison")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return
    
    audio, sr = sf.read("test_audio/comparison_llama.wav")
    audio_tensor = torch.from_numpy(audio).float()
    
    for dtype_name, dtype in [("FP32", torch.float32), ("FP16", torch.float16)]:
        print(f"\nTesting {dtype_name}...")
        
        encoder = create_optimized_encoder(
            weights_path=None,
            device="cuda",
            compile_mode="default",
            dtype=dtype,
            warmup=True
        )
        
        # Benchmark
        iterations = 5
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(iterations):
            tokens, emb = encoder.encode(audio_tensor, sr=sr)
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        avg_latency = elapsed / iterations
        mem_allocated = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"  Latency: {avg_latency:.4f}s")
        print(f"  Memory: {mem_allocated:.1f}MB")
        
        del encoder
        torch.cuda.empty_cache()


def example_streaming_simulation():
    """Simulate streaming processing."""
    print("\n" + "="*70)
    print("Example 4: Streaming Simulation")
    print("="*70)
    
    encoder = create_optimized_encoder(
        weights_path=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_mode="reduce-overhead",  # Best for low latency
        dtype=torch.float16,
        warmup=True
    )
    
    # Simulate 5-second chunks
    chunk_duration = 5.0
    sample_rate = 24000
    chunk_samples = int(chunk_duration * sample_rate)
    
    print(f"Processing {chunk_duration}s chunks in streaming mode...")
    
    for i in range(3):
        # Generate or load chunk
        chunk = torch.randn(chunk_samples)
        
        start = time.time()
        tokens, emb = encoder.encode(chunk, sr=sample_rate)
        elapsed = time.time() - start
        
        rtf = elapsed / chunk_duration
        print(f"Chunk {i+1}: {elapsed:.4f}s (RTF: {rtf:.4f})")


def example_save_tokens():
    """Encode and save tokens for later use."""
    print("\n" + "="*70)
    print("Example 5: Save Tokens")
    print("="*70)
    
    encoder = create_optimized_encoder(
        weights_path=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_mode="max-autotune",
        dtype=torch.float16,
        warmup=True
    )
    
    # Encode
    audio, sr = sf.read("test_audio/comparison_llama.wav")
    audio_tensor = torch.from_numpy(audio).float()
    tokens, embedding = encoder.encode(audio_tensor, sr=sr)
    
    # Save
    output_path = "encoded_tokens.pt"
    torch.save({
        'tokens': tokens.cpu(),
        'embedding': embedding.cpu(),
        'sample_rate': sr,
        'encoder_version': 'optimized_v1'
    }, output_path)
    
    print(f"Saved tokens to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.2f}KB")
    
    # Load and verify
    loaded = torch.load(output_path)
    print(f"Loaded {loaded['tokens'].shape[0]} tokens")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GPU-Optimized Encoder Examples")
    print("="*70)
    
    # Run examples
    encoder = example_basic_usage()
    example_batch_processing(encoder)
    example_different_dtypes()
    example_streaming_simulation()
    example_save_tokens()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
