"""
Pipelined Encoder with Parallel WavLM and Encoder Processing

Architecture:
- CUDA Stream 1: WavLM feature extraction
- CUDA Stream 2: Encoder processing (transformer + quantization)
- Overlap: While encoder processes batch N, WavLM extracts features for batch N+1

This maximizes GPU utilization by keeping both models busy simultaneously.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from queue import Queue
from threading import Thread
import soundfile as sf
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from encoder_optimized import OptimizedStandaloneEncoder
from encoder_standalone import ensure_weights_path


class PipelinedEncoder:
    """
    Pipelined encoder with parallel WavLM and encoder stages.
    
    Pipeline stages:
    1. WavLM (Stream 1): Extract SSL features
    2. Encoder (Stream 2): Transform + quantize + global embed
    
    Batches flow through stages with overlap for maximum throughput.
    """
    
    def __init__(
        self,
        weights_path: str = "encoder.safetensors",
        device: str = "cuda",
        compile_mode: str = "max-autotune",
        dtype: torch.dtype = torch.float16,
        warmup_both: bool = True,
        max_queue_size: int = 2
    ):
        self.device = device
        self.dtype = dtype
        self.max_queue_size = max_queue_size
        
        # Initialize base encoder
        self.encoder = OptimizedStandaloneEncoder(
            device=device,
            compile_mode=compile_mode,
            use_compile=True,
            dtype=dtype
        )
        self.encoder.load_weights(weights_path)
        self.encoder.ensure_ssl_extractor()
        
        # Create CUDA streams for parallel execution
        if torch.cuda.is_available():
            self.stream_wavlm = torch.cuda.Stream()
            self.stream_encoder = torch.cuda.Stream()
        else:
            self.stream_wavlm = None
            self.stream_encoder = None
        
        # Warmup
        if warmup_both:
            self.warmup()
    
    def warmup(self, num_iterations: int = 3):
        """Warmup both WavLM and encoder stages."""
        print("[*] Warming up pipelined encoder (both stages)...")
        
        # Create dummy batches
        dummy_audio = torch.randn(2, 24000 * 5).to(self.device)  # 2 samples, 5s each
        dummy_sr = 24000
        
        for i in range(num_iterations):
            # Warmup WavLM
            with torch.cuda.stream(self.stream_wavlm) if self.stream_wavlm else torch.no_grad():
                waveform_ssl = self._resample_for_ssl(dummy_audio, dummy_sr)
                all_ssl, _ = self.encoder.ssl_model.extract_features(waveform_ssl)
            
            # Warmup Encoder
            with torch.cuda.stream(self.stream_encoder) if self.stream_encoder else torch.no_grad():
                local_ssl = self.encoder._process_ssl_features(all_ssl, [6, 9])
                local_ssl = self.encoder._normalize_ssl_features(local_ssl)
                
                if self.encoder._compiled_encoder:
                    encoded = self.encoder._compiled_encoder(local_ssl)
                else:
                    encoded = self.encoder.local_encoder(local_ssl)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        print("[+] Warmup complete (both stages)")
    
    def _resample_for_ssl(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Resample audio to 16kHz for SSL model."""
        target_sr = 16000
        if sr != target_sr:
            if self.encoder._resampler is None or self.encoder._resampler_sr != sr:
                import torchaudio
                self.encoder._resampler = torchaudio.transforms.Resample(sr, target_sr).to(self.device)
                self.encoder._resampler_sr = sr
            return self.encoder._resampler(waveform)
        return waveform
    
    @torch.inference_mode()
    def encode_batch_pipelined(
        self,
        waveforms: List[torch.Tensor],
        sr: int = 24000,
        batch_size: int = 8
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode multiple audio samples with pipelined parallelism.
        
        Args:
            waveforms: List of audio tensors
            sr: Sample rate
            batch_size: Batch size for processing
            
        Returns:
            List of (tokens, global_embedding) tuples
        """
        if not torch.cuda.is_available():
            # Fallback to sequential processing
            return self._encode_sequential(waveforms, sr, batch_size)
        
        # Prepare batches
        batches = self._prepare_batches(waveforms, batch_size)
        results = [None] * len(waveforms)
        
        # Pipeline state
        ssl_features_queue = []
        
        for batch_idx, (batch_indices, padded_batch) in enumerate(batches):
            # Stage 1: WavLM feature extraction (Stream 1)
            with torch.cuda.stream(self.stream_wavlm):
                waveform_ssl = self._resample_for_ssl(padded_batch, sr)
                
                if self.encoder._compiled_ssl:
                    all_ssl, _ = self.encoder._compiled_ssl(waveform_ssl)
                else:
                    all_ssl, _ = self.encoder.ssl_model.extract_features(waveform_ssl)
                
                # Process SSL features
                local_ssl = self.encoder._process_ssl_features(all_ssl, [6, 9])
                local_ssl = self.encoder._normalize_ssl_features(local_ssl)
                global_ssl = self.encoder._process_ssl_features(all_ssl, [1, 2, 3, 4])
                
                # Store for next stage
                ssl_features_queue.append((batch_idx, batch_indices, local_ssl, global_ssl))
            
            # Stage 2: Encoder processing (Stream 2)
            # Process previous batch while WavLM is working on current batch
            if len(ssl_features_queue) > 1:
                self._process_encoder_stage(ssl_features_queue[0], results)
                ssl_features_queue.pop(0)
        
        # Process remaining batches in queue
        for ssl_features in ssl_features_queue:
            self._process_encoder_stage(ssl_features, results)
        
        # Synchronize all streams
        torch.cuda.synchronize()
        
        return results
    
    def _process_encoder_stage(
        self,
        ssl_features_data: Tuple,
        results: List
    ):
        """Process encoder stage on Stream 2."""
        batch_idx, batch_indices, local_ssl, global_ssl = ssl_features_data
        
        with torch.cuda.stream(self.stream_encoder):
            # Encode through transformer
            if self.encoder._compiled_encoder:
                encoded = self.encoder._compiled_encoder(local_ssl)
            else:
                encoded = self.encoder.local_encoder(local_ssl)
            
            # Downsample
            downsampled = self.encoder.conv_downsample(
                encoded.transpose(1, 2)
            ).transpose(1, 2)
            
            # Quantize
            _, indices = self.encoder.local_quantizer.encode(downsampled)
            
            # Global embedding
            global_emb = self.encoder.global_encoder(global_ssl)
            
            # Store results
            for i, orig_idx in enumerate(batch_indices):
                results[orig_idx] = (indices[i], global_emb[i])
    
    def _prepare_batches(
        self,
        waveforms: List[torch.Tensor],
        batch_size: int
    ) -> List[Tuple[List[int], torch.Tensor]]:
        """Prepare batched and padded waveforms."""
        import torch.nn.functional as F
        
        # Sort by length for efficient batching
        sorted_wavs = sorted(enumerate(waveforms), key=lambda x: x[1].shape[-1])
        
        batches = []
        for i in range(0, len(sorted_wavs), batch_size):
            batch = sorted_wavs[i:i+batch_size]
            indices, wavs = zip(*batch)
            
            # Pad to same length
            max_len = max(w.shape[-1] for w in wavs)
            padded = torch.stack([
                F.pad(w, (0, max_len - w.shape[-1])) for w in wavs
            ]).to(self.device)
            
            batches.append((list(indices), padded))
        
        return batches
    
    def _encode_sequential(
        self,
        waveforms: List[torch.Tensor],
        sr: int,
        batch_size: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Fallback sequential encoding for CPU."""
        return self.encoder.encode_batch(waveforms, sr=sr)
    
    @torch.inference_mode()
    def encode_single(
        self,
        waveform: torch.Tensor,
        sr: int = 24000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single audio sample (uses standard path)."""
        return self.encoder.encode(waveform, sr=sr)


def create_pipelined_encoder(
    weights_path: str = "encoder.safetensors",
    device: str = "cuda",
    compile_mode: str = "max-autotune",
    dtype: torch.dtype = torch.float16,
    warmup_both: bool = True
) -> PipelinedEncoder:
    """
    Factory function to create pipelined encoder.
    
    Args:
        weights_path: Path to encoder weights
        device: Device to use
        compile_mode: torch.compile mode
        dtype: Data type for inference
        warmup_both: Warmup both WavLM and encoder stages
        
    Returns:
        Initialized pipelined encoder
    """
    weights_path = ensure_weights_path(weights_path)
    
    return PipelinedEncoder(
        weights_path=weights_path,
        device=device,
        compile_mode=compile_mode,
        dtype=dtype,
        warmup_both=warmup_both
    )


if __name__ == "__main__":
    import time
    import numpy as np
    
    print("="*70)
    print("Pipelined Encoder Benchmark")
    print("="*70)
    
    # Create encoder
    encoder = create_pipelined_encoder(
        weights_path="encoder.safetensors",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_mode="max-autotune",
        dtype=torch.float16,
        warmup_both=True
    )
    
    # Load test audio files
    audio_files = [
        "test_audio/comparison_llama.wav",
        "test_audio/comparison_gguf_f16.wav",
        "test_audio/comparison_gguf_q4.wav",
        "test_audio/comparison_gguf_q8.wav",
    ]
    
    waveforms = []
    total_duration = 0
    
    for f in audio_files:
        try:
            audio, sr = sf.read(f)
            waveforms.append(torch.from_numpy(audio).float())
            total_duration += len(audio) / sr
        except:
            pass
    
    if not waveforms:
        print("[!] No test audio found, using synthetic data")
        waveforms = [torch.randn(24000 * 5) for _ in range(4)]
        total_duration = 20.0
    
    print(f"[*] Processing {len(waveforms)} files ({total_duration:.1f}s total audio)")
    
    # Benchmark pipelined encoding
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    results = encoder.encode_batch_pipelined(waveforms, sr=24000, batch_size=2)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    
    print(f"\n[RESULTS]")
    print(f"Total time: {elapsed:.4f}s")
    print(f"Throughput: {total_duration/elapsed:.2f}x real-time")
    print(f"Avg per file: {elapsed/len(waveforms):.4f}s")
    
    for i, (tokens, emb) in enumerate(results):
        print(f"  File {i+1}: {tokens.shape[0]} tokens, emb: {emb.shape}")
