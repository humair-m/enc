"""
Long Audio Encoder with Overlap Chunking

Handles audio longer than the 80s transformer limit by:
1. Splitting into overlapping chunks (60s + 5s overlap)
2. Encoding each chunk independently
3. Concatenating tokens (discarding overlap regions)
4. Merging global embeddings

Quality preservation:
- Overlap prevents boundary artifacts
- Global embedding captures full audio context
- No information loss at chunk boundaries
"""

import torch
import torch.nn as nn
from typing import Tuple, List
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import soundfile as sf
from encoder_optimized import create_optimized_encoder
from encoder_standalone import ensure_weights_path


class LongAudioEncoder:
    """
    Encoder for audio longer than 80 seconds.
    
    Uses overlapping chunks to maintain quality at boundaries.
    """
    
    def __init__(
        self,
        weights_path: str = "encoder.safetensors",
        device: str = "cuda",
        compile_mode: str = "max-autotune",
        dtype: torch.dtype = torch.float16,
        chunk_duration: float = 60.0,  # seconds
        overlap_duration: float = 5.0,  # seconds overlap
        warmup: bool = True
    ):
        self.device = device
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        
        # Create base encoder (use StandaloneEncoder for compatibility)
        from encoder_standalone import StandaloneEncoder
        self.encoder = StandaloneEncoder(device=device)
        self.encoder.load_weights(weights_path)
        self.encoder.ensure_ssl_extractor()
        self.encoder.eval()
    
    def _split_into_chunks(
        self,
        waveform: torch.Tensor,
        sr: int
    ) -> List[Tuple[torch.Tensor, int, int]]:
        """
        Split audio into overlapping chunks.
        
        Returns:
            List of (chunk_audio, start_sample, end_sample)
        """
        chunk_samples = int(self.chunk_duration * sr)
        overlap_samples = int(self.overlap_duration * sr)
        stride_samples = chunk_samples - overlap_samples
        
        chunks = []
        start = 0
        
        while start < len(waveform):
            end = min(start + chunk_samples, len(waveform))
            chunk = waveform[start:end]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_samples and start > 0:
                chunk = torch.nn.functional.pad(
                    chunk, 
                    (0, chunk_samples - len(chunk))
                )
            
            chunks.append((chunk, start, end))
            
            # If we've reached the end, break
            if end >= len(waveform):
                break
            
            start += stride_samples
        
        return chunks
    
    def _calculate_valid_token_range(
        self,
        chunk_idx: int,
        total_chunks: int,
        tokens_per_chunk: int,
        overlap_tokens: int
    ) -> Tuple[int, int]:
        """
        Calculate which tokens to keep from each chunk (excluding overlap).
        
        Returns:
            (start_idx, end_idx) for valid tokens in this chunk
        """
        if total_chunks == 1:
            # Single chunk - keep everything
            return 0, tokens_per_chunk
        
        half_overlap = overlap_tokens // 2
        
        if chunk_idx == 0:
            # First chunk: keep all except end overlap
            return 0, tokens_per_chunk - half_overlap
        elif chunk_idx == total_chunks - 1:
            # Last chunk: keep all except start overlap
            return half_overlap, tokens_per_chunk
        else:
            # Middle chunks: remove both overlaps
            return half_overlap, tokens_per_chunk - half_overlap
    
    def _merge_global_embeddings(
        self,
        embeddings: List[torch.Tensor],
        chunk_weights: List[float] = None
    ) -> torch.Tensor:
        """
        Merge global embeddings from multiple chunks.
        
        Strategies:
        - Weighted average (default): longer chunks get more weight
        - First chunk: use only first chunk's embedding
        - Average: simple average of all embeddings
        """
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Weighted average (gives more weight to longer chunks)
        if chunk_weights is None:
            chunk_weights = [1.0] * len(embeddings)
        
        total_weight = sum(chunk_weights)
        weighted_sum = sum(
            emb * (w / total_weight) 
            for emb, w in zip(embeddings, chunk_weights)
        )
        
        return weighted_sum
    
    @torch.inference_mode()
    def encode(
        self,
        waveform: torch.Tensor,
        sr: int = 24000,
        merge_strategy: str = "weighted"  # "weighted", "first", "average"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode long audio file.
        
        Args:
            waveform: Audio tensor (samples,)
            sr: Sample rate
            merge_strategy: How to merge global embeddings
                - "weighted": weight by chunk duration (recommended)
                - "first": use first chunk only
                - "average": simple average
        
        Returns:
            (all_tokens, merged_global_embedding)
        """
        # Handle string path or numpy array
        if isinstance(waveform, str):
            import torchaudio
            waveform, sr = torchaudio.load(waveform)
        elif isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
            
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        audio_duration = waveform.shape[-1] / sr
        
        # Short audio - use direct encoding
        if audio_duration <= self.chunk_duration:
            return self.encoder.encode(waveform, sr=sr)
        
        # Long audio - use chunking
        print(f"[*] Processing {audio_duration:.1f}s audio with chunking...")
        
        chunks = self._split_into_chunks(waveform, sr)
        print(f"[*] Split into {len(chunks)} chunks ({self.chunk_duration}s chunks, {self.overlap_duration}s overlap)")
        
        all_tokens = []
        all_embeddings = []
        chunk_weights = []
        
        # Encode each chunk
        for i, (chunk_audio, start, end) in enumerate(chunks):
            tokens, global_emb = self.encoder.encode(chunk_audio, sr=sr)
            
            # Calculate overlap in tokens (25 Hz token rate)
            token_rate = 25  # tokens per second after downsampling
            overlap_tokens = int(self.overlap_duration * token_rate)
            
            # Determine valid token range for this chunk
            start_idx, end_idx = self._calculate_valid_token_range(
                i, len(chunks), len(tokens), overlap_tokens
            )
            
            # Keep only valid tokens (excluding overlap regions)
            valid_tokens = tokens[start_idx:end_idx]
            all_tokens.append(valid_tokens)
            
            # Store embedding and weight
            all_embeddings.append(global_emb)
            chunk_duration = (end - start) / sr
            chunk_weights.append(chunk_duration)
            
            print(f"  Chunk {i+1}/{len(chunks)}: {len(valid_tokens)} tokens (kept {start_idx}:{end_idx})")
        
        # Concatenate all tokens
        final_tokens = torch.cat(all_tokens, dim=0)
        
        # Merge global embeddings
        if merge_strategy == "first":
            final_embedding = all_embeddings[0]
        elif merge_strategy == "average":
            final_embedding = torch.stack(all_embeddings).mean(dim=0)
        else:  # weighted
            final_embedding = self._merge_global_embeddings(all_embeddings, chunk_weights)
        
        print(f"[+] Encoded {audio_duration:.1f}s → {len(final_tokens)} tokens")
        
        return final_tokens, final_embedding


def create_long_audio_encoder(
    weights_path: str = None,
    device: str = "cuda",
    compile_mode: str = "max-autotune",
    dtype: torch.dtype = torch.float16,
    chunk_duration: float = 60.0,
    overlap_duration: float = 5.0,
    warmup: bool = True
) -> LongAudioEncoder:
    """
    Factory function to create long audio encoder.
    
    Args:
        weights_path: Path to encoder weights (default: encoder.safetensors in same dir)
        device: Device to use
        compile_mode: torch.compile mode
        dtype: Data type
        chunk_duration: Duration of each chunk (seconds)
        overlap_duration: Overlap between chunks (seconds)
        warmup: Warmup encoder
        
    Returns:
        Initialized long audio encoder
    """
    weights_path = ensure_weights_path(weights_path)
        
    return LongAudioEncoder(
        weights_path=weights_path,
        device=device,
        compile_mode=compile_mode,
        dtype=dtype,
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration,
        warmup=warmup
    )


if __name__ == "__main__":
    import time
    
    print("="*70)
    print("Long Audio Encoder Test")
    print("="*70)
    
    # Create encoder
    encoder = create_long_audio_encoder(
        weights_path="encoder.safetensors",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_mode="max-autotune",
        dtype=torch.float16,
        chunk_duration=60.0,
        overlap_duration=5.0,
        warmup=True
    )
    
    # Test with synthetic long audio (2 minutes)
    print("\n[*] Testing with 2-minute synthetic audio...")
    long_audio = torch.randn(24000 * 120)  # 2 minutes
    
    start = time.time()
    tokens, embedding = encoder.encode(long_audio, sr=24000, merge_strategy="weighted")
    elapsed = time.time() - start
    
    print(f"\n[RESULTS]")
    print(f"Audio duration: 120.0s")
    print(f"Processing time: {elapsed:.4f}s")
    print(f"RTF: {elapsed/120.0:.4f}")
    print(f"Total tokens: {len(tokens)}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Expected tokens (120s × 25Hz): ~{120*25} (actual: {len(tokens)})")
