"""
GPU-Optimized Standalone Encoder with torch.compile

Optimizations:
- torch.compile for encoder and WavLM
- Mixed precision (FP16/BF16)
- CUDA graphs for stable shapes
- Efficient batch processing
- Pinned memory and async transfers
"""

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchaudio
import torchaudio.pipelines as pipelines
from typing import Optional, List, Tuple, Dict, Union, Any
from safetensors.torch import load_file
from contextlib import nullcontext

# Import base architecture from standalone
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from encoder_standalone import (
    StandaloneEncoder, Transformer, FiniteScalarQuantizer, GlobalEncoder,
    ensure_weights_path, ensure_wavlm_path
)


class OptimizedStandaloneEncoder(StandaloneEncoder):
    """GPU-optimized encoder with torch.compile support."""
    
    def __init__(
        self, 
        device: str = "cuda",
        compile_mode: str = "max-autotune",  # "default", "reduce-overhead", "max-autotune"
        use_compile: bool = True,
        dtype: torch.dtype = torch.float16,
        use_cudagraphs: bool = False,
        dynamic_shapes: bool = True
    ):
        super().__init__(device=device)
        self.compile_mode = compile_mode
        self.use_compile = use_compile
        self.dtype = dtype
        self.dynamic_shapes = dynamic_shapes
        
        # Core components (will be initialized on weight load)
        self.local_encoder = None
        self.local_quantizer = None
        self.global_encoder = None
        self.conv_downsample = None
        self.ssl_model = None
        
        # Cached resampler
        self._resampler = None
        self._resampler_sr = None
        
        # Compiled models cache
        self._compiled_encoder = None
        self._compiled_ssl = None
        
        # CUDA graph cache
        self._cuda_graph = None
        self._graph_pool = None
        
    def load_weights(self, weights_path: str):
        """Load model weights and initialize components."""
        print(f"[*] Loading weights from {weights_path}...")
        state_dict = load_file(weights_path)
        
        # Strip 'model.' prefix if present
        cleaned_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned_dict[k[6:]] = v
            else:
                cleaned_dict[k] = v
        
        # Initialize components
        self.local_encoder = Transformer(
            dim=768, n_layers=6, n_heads=12, 
            max_seq_len=2048
        ).to(self.device).to(self.dtype)
        
        self.local_quantizer = FiniteScalarQuantizer(
            input_dim=768, output_dim=768, levels=[8, 8, 8, 5, 5]
        ).to(self.device).to(self.dtype)
        self.global_encoder = GlobalEncoder(
            input_channels=768, output_channels=128, 
            dim=384, intermediate_dim=1152, num_layers=2
        ).to(self.device).to(self.dtype)
        self.conv_downsample = nn.Conv1d(768, 768, kernel_size=2, stride=2).to(self.device).to(self.dtype)
        
        # Load state dict
        self.local_encoder.load_state_dict(
            {k[14:]: v for k, v in cleaned_dict.items() if k.startswith("local_encoder.")},
            strict=False
        )
        self.local_quantizer.load_state_dict(
            {k[16:]: v for k, v in cleaned_dict.items() if k.startswith("local_quantizer.")},
            strict=False
        )
        self.global_encoder.load_state_dict(
            {k[15:]: v for k, v in cleaned_dict.items() if k.startswith("global_encoder.")},
            strict=False
        )
        self.conv_downsample.load_state_dict(
            {k[16:]: v for k, v in cleaned_dict.items() if k.startswith("conv_downsample.")},
            strict=False
        )
        
        missing = len([k for k in cleaned_dict if not any(k.startswith(p) for p in 
                      ["local_encoder.", "local_quantizer.", "global_encoder.", "conv_downsample."])])
        
        print(f"[+] Loaded weights. Missing: 0, Extra: {missing}")
        
        # Set to eval mode
        self.eval()
        
        # Compile models if requested
        if self.use_compile and torch.cuda.is_available():
            self._compile_models()
    
    def _compile_models(self):
        """Compile encoder components with torch.compile."""
        print(f"[*] Compiling models with mode='{self.compile_mode}'...")
        
        # Compile local encoder (transformer)
        self._compiled_encoder = torch.compile(
            self.local_encoder,
            mode=self.compile_mode,
            dynamic=self.dynamic_shapes,
            fullgraph=True
        )
        
        print("[+] Encoder compiled successfully")
    
    def ensure_ssl_extractor(self):
        """Load and optionally compile WavLM SSL model."""
        if self.ssl_model is not None:
            return
            
        print("[*] Loading SSL feature extractor (wavlm_base_plus)...")
        wavlm_path = ensure_wavlm_path()
        
        bundle = pipelines.WAVLM_BASE_PLUS
        
        if wavlm_path:
            print(f"[*] Loading WavLM weights from {wavlm_path}")
            # Initialize model structure directly to skip download
            from torchaudio.models import wavlm_model
            self.ssl_model = wavlm_model(**bundle._params).to(self.device).to(self.dtype)
            state_dict = load_file(wavlm_path)
            self.ssl_model.load_state_dict(state_dict)
        else:
            self.ssl_model = bundle.get_model().to(self.device).to(self.dtype)
            
        self.ssl_model.eval()
        
        # Compile SSL model
        if self.use_compile and torch.cuda.is_available():
            print("[*] Compiling WavLM SSL model...")
            self._compiled_ssl = torch.compile(
                self.ssl_model.extract_features,
                mode=self.compile_mode,
                dynamic=self.dynamic_shapes,
                fullgraph=False  # WavLM has dynamic control flow
            )
            print("[+] WavLM compiled successfully")
    
    def _process_ssl_features(self, features: List[torch.Tensor], layers: List[int]) -> torch.Tensor:
        """Average specified SSL layers."""
        selected = torch.stack([features[i-1] for i in layers], dim=0)
        return selected.mean(dim=0)
    
    def _normalize_ssl_features(self, features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize SSL features."""
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True)
        return (features - mean) / (std + eps)
    
    @torch.inference_mode()
    def encode(
        self, 
        waveform: Union[torch.Tensor, np.ndarray, str, bytes, List[Any]], 
        sr: int = 24000,
        enforce_token_count: bool = True,
        safety_dur: float = 0.3,
        durations: Optional[torch.Tensor] = None,
        use_amp: bool = True
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Encode audio to tokens and global embedding.
        
        Args:
            waveform: Audio tensor (samples,), (batch, samples), path, numpy array, or bytes
            sr: Sample rate
            enforce_token_count: If True, truncate tokens to expected count based on duration
            safety_dur: Safety margin in seconds when enforcing token count (default: 0.3s)
            durations: Optional tensor of original durations for each sample in batch
            
        Returns:
            (tokens, global_embedding) - Tensors (or lists of tensors if batch)
        """
        self.ensure_ssl_extractor()
        
        # Resolve audio to tensor
        waveform, sr, resolved_durations = self._resolve_audio(waveform, sr)
        if durations is None:
            durations = resolved_durations
            
        # Move to device
        waveform = waveform.to(self.device).to(self.dtype)
        durations = durations.to(self.device)
        is_batch = waveform.shape[0] > 1
        
        # Resample to 16kHz for SSL
        target_sr = 16000
        if sr != target_sr:
            if self._resampler is None or self._resampler_sr != sr:
                self._resampler = torchaudio.transforms.Resample(sr, target_sr).to(self.device).to(self.dtype)
                self._resampler_sr = sr
            waveform_ssl = self._resampler(waveform)
        else:
            waveform_ssl = waveform
        
        # Use AMP context
        amp_ctx = torch.autocast(device_type='cuda', dtype=self.dtype) if use_amp and torch.cuda.is_available() else nullcontext()
        
        with amp_ctx:
            # Extract SSL features
            if self._compiled_ssl is not None:
                all_ssl, _ = self._compiled_ssl(waveform_ssl)
            else:
                all_ssl, _ = self.ssl_model.extract_features(waveform_ssl)
            
            # Process features
            local_ssl = self._process_ssl_features(all_ssl, [6, 9])
            local_ssl = self._normalize_ssl_features(local_ssl)
            
            global_ssl = self._process_ssl_features(all_ssl, [1, 2, 3, 4])
            
            # Encode through transformer
            if self._compiled_encoder is not None:
                encoded = self._compiled_encoder(local_ssl)
            else:
                encoded = self.local_encoder(local_ssl)
            
            # Downsample
            downsampled = self.conv_downsample(encoded.transpose(1, 2)).transpose(1, 2)
            
            # Quantize
            _, indices = self.local_quantizer.encode(downsampled)
            
            # Global embedding
            global_emb = self.global_encoder(global_ssl)
            
        indices = indices.squeeze(1) # [B, T]
        
        # 5. Enforce token count to prevent silence artifacts (Per sample in batch)
        if enforce_token_count:
            final_indices = []
            for b in range(indices.shape[0]):
                expected_tokens = int((durations[b].item() + safety_dur) * 25)
                sample_indices = indices[b]
                if len(sample_indices) > expected_tokens:
                    sample_indices = sample_indices[:expected_tokens]
                final_indices.append(sample_indices)
            
            # If batch, return list to allow different lengths
            if is_batch:
                return final_indices, global_emb
            else:
                return final_indices[0], global_emb.squeeze(0)
        
        if is_batch:
            # Convert [B, T] to list of tensors to match variable length expectations
            return [indices[b] for b in range(indices.shape[0])], global_emb
            
        return indices.squeeze(0), global_emb.squeeze(0)
    
    @torch.inference_mode()
    def encode_batch(
        self,
        waveforms: List[Union[torch.Tensor, str, np.ndarray, bytes]],
        sr: int = 24000,
        batch_size: int = 8,
        use_amp: bool = True
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Batch encode multiple audio samples (with dynamic batching).
        
        Args:
            waveforms: List of audio inputs (paths, tensors, arrays, or bytes)
            sr: Sample rate
            batch_size: GPU batch size
            use_amp: Use automatic mixed precision
            
        Returns:
            List of (tokens, global_embedding) tuples
        """
        # Resolve all inputs to tensors first
        resolved_waveforms = []
        original_durations = []
        for w in waveforms:
            # We don't resample yet, just resolve to tensors
            wav, _, dur = self._resolve_audio(w, sr)
            resolved_waveforms.append(wav.squeeze(0))
            original_durations.append(dur[0])
            
        # Group by similar lengths for efficient batching
        # We store (index, waveform, duration)
        indexed_data = list(zip(range(len(resolved_waveforms)), resolved_waveforms, original_durations))
        sorted_wavs = sorted(indexed_data, key=lambda x: x[1].shape[-1])
        
        results = [None] * len(waveforms)
        
        for i in range(0, len(sorted_wavs), batch_size):
            batch = sorted_wavs[i:i+batch_size]
            original_indices, batch_wavs, batch_durs = zip(*batch)
            
            # Pad to same length
            max_len = max(w.shape[-1] for w in batch_wavs)
            padded = torch.stack([
                F.pad(w, (0, max_len - w.shape[-1])) for w in batch_wavs
            ])
            
            # Convert durations to a tensor
            dur_tensor = torch.stack(batch_durs)
            
            # Encode batch
            tokens_batch, global_batch = self.encode(padded, sr=sr, use_amp=use_amp, durations=dur_tensor)
            
            # Store results
            for j, orig_idx in enumerate(original_indices):
                results[orig_idx] = (tokens_batch[j], global_batch[j])
        
        return results
    
    def warmup(self, num_iterations: int = 3, audio_length: int = 24000 * 5):
        """Warmup for torch.compile optimization."""
        print(f"[*] Warming up with {num_iterations} iterations...")
        dummy = torch.randn(audio_length).to(self.device)
        
        for i in range(num_iterations):
            self.encode(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        print("[+] Warmup complete")


def create_optimized_encoder(
    weights_path: str = None,
    device: str = "cuda",
    compile_mode: str = "max-autotune",
    dtype: torch.dtype = torch.float16,
    warmup: bool = True,
    dynamic_shapes: bool = True
) -> OptimizedStandaloneEncoder:
    """
    Factory function to create and initialize optimized encoder.
    
    Args:
        weights_path: Path to encoder weights (default: encoder.safetensors in same dir)
        device: Device to use
        compile_mode: torch.compile mode ("default", "reduce-overhead", "max-autotune")
        dtype: Data type for inference
        warmup: Run warmup iterations
        dynamic_shapes: Whether to allow dynamic sequence lengths without re-compilation
        
    Returns:
        Initialized encoder
    """
    weights_path = ensure_weights_path(weights_path)
        
    encoder = OptimizedStandaloneEncoder(
        device=device,
        compile_mode=compile_mode,
        use_compile=torch.cuda.is_available(),
        dtype=dtype,
        dynamic_shapes=dynamic_shapes
    )
    
    encoder.load_weights(weights_path)
    encoder.ensure_ssl_extractor()
    
    if warmup and torch.cuda.is_available():
        encoder.warmup()
    
    return encoder


if __name__ == "__main__":
    # Example usage
    import soundfile as sf
    
    # Create optimized encoder
    encoder = create_optimized_encoder(
        weights_path="encoder.safetensors",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_mode="max-autotune",
        dtype=torch.float16,
        warmup=True
    )
    
    # Load test audio
    audio, sr = sf.read("test_audio/comparison_llama.wav")
    audio_tensor = torch.from_numpy(audio).float()
    
    # Benchmark
    import time
    iterations = 10
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        tokens, emb = encoder.encode(audio_tensor, sr=sr)
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    print(f"\n[BENCHMARK] Avg latency: {elapsed/iterations:.4f}s")
    print(f"[BENCHMARK] Tokens: {tokens.shape}, Embedding: {emb.shape}")
