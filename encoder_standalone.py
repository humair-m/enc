import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.pipelines as pipelines
from typing import Optional, List, Tuple, Dict
from safetensors.torch import load_file
import os
import numpy as np

def ensure_weights_path(idx: str = None):
    """
    Resolve weights path. 
    1. If explicit path provided, use it.
    2. If None, check local 'encoder.safetensors'.
    3. If local missing, download 'encoder_f16.safetensors' from HF.
    """
    if idx is not None:
        return idx
        
    # Check default local path
    default_path = os.path.join(os.path.dirname(__file__), "encoder.safetensors")
    if os.path.exists(default_path):
        return default_path
        
    print(f"[*] Local weights not found at {default_path}")
    print("[*] Attempting download from Hugging Face (humair025/enc)...")
    
    try:
        from huggingface_hub import hf_hub_download
        # Download format: encoder_f16.safetensors
        cached_path = hf_hub_download(
            repo_id="humair025/enc", 
            filename="encoder_f16.safetensors"
        )
        print(f"[+] Downloaded/Found in cache: {cached_path}")
        return cached_path
    except ImportError:
        print("[!] huggingface_hub not installed. Please install it or provide weights_path.")
    except Exception as e:
        print(f"[!] Failed to download weights: {e}")
        
    return default_path


def ensure_wavlm_path(idx: str = None):
    """
    Resolve WavLM path.
    1. Check local 'wavlm_fp16.safetensors'.
    2. If missing, download from HF 'humair025/enc'.
    3. Return None if not found (fallback to torchaudio download).
    """
    if idx is not None:
        return idx
    
    default_path = os.path.join(os.path.dirname(__file__), "wavlm_fp16.safetensors")
    if os.path.exists(default_path):
        return default_path
        
    print(f"[*] Local WavLM not found at {default_path}")
    print("[*] Attempting download from Hugging Face (humair025/enc/wavlm_fp16.safetensors)...")
    
    try:
        from huggingface_hub import hf_hub_download
        cached_path = hf_hub_download(
            repo_id="humair025/enc", 
            filename="wavlm_fp16.safetensors"
        )
        print(f"[+] Downloaded/Found in cache: {cached_path}")
        return cached_path
    except ImportError:
        pass
    except Exception as e:
        print(f"[!] Failed to download WavLM: {e}")
        
    return None


# -----------------------------------------------------------------------------
# Utils & RoPE
# -----------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device="cpu"):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)

# -----------------------------------------------------------------------------
# Transformer Modules
# -----------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, window_size: Optional[int] = None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        self.window_per_side = window_size // 2 if window_size else None

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        if freqs_cis is not None:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)
            
        # Transpose for SDPA: (bsz, n_heads, seqlen, head_dim)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        
        mask = None
        if self.window_per_side is not None:
            mask = torch.ones((seqlen, seqlen), dtype=torch.bool, device=x.device)
            mask = torch.triu(mask, diagonal=-self.window_per_side) & torch.tril(mask, diagonal=self.window_per_side)
            
        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, multiple_of: int, window_size: Optional[int] = None):
        super().__init__()
        self.attention = Attention(dim, n_heads, window_size)
        self.feed_forward = FeedForward(dim, int(2/3 * 4 * dim)) # Default hidden dim ratio
        self.attention_norm = nn.LayerNorm(dim, eps=1e-5)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor]) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, dim: int, n_layers: int, n_heads: int, window_size: Optional[int] = None, max_seq_len: int = 2048):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads, 256, window_size) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.freqs_cis = None # Computed on first forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        if self.freqs_cis is None or self.freqs_cis.shape[0] < seqlen:
            self.freqs_cis = precompute_freqs_cis(dim // self.layers[0].attention.n_heads, seqlen, device=x.device)
            
        freqs_cis_step = self.freqs_cis[:seqlen]
        
        for layer in self.layers:
            x = layer(x, freqs_cis_step)
            
        return self.norm(x)

# -----------------------------------------------------------------------------
# Global Encoder (ConvNeXt + AttentiveStatsPool)
# -----------------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = residual + x.transpose(1, 2)
        return x

class ConvNextBackbone(nn.Module):
    def __init__(self, input_channels: int, dim: int, intermediate_dim: int, num_layers: int):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList([ConvNeXtBlock(dim, intermediate_dim) for _ in range(num_layers)])
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        x = self.final_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return x

class AttentiveStatsPool(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, attention_channels: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(input_channels, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, input_channels, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.proj = nn.Linear(input_channels * 2, output_channels)
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.attn(x)
        mean = torch.sum(alpha * x, dim=2)
        std = torch.sqrt(torch.sum(alpha * (x**2), dim=2) - mean**2 + 1e-6)
        return self.norm(self.proj(torch.cat([mean, std], dim=1)))

class GlobalEncoder(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dim: int, intermediate_dim: int, num_layers: int):
        super().__init__()
        self.backbone = ConvNextBackbone(input_channels, dim, intermediate_dim, num_layers)
        self.pooling = AttentiveStatsPool(dim, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2) # (B, T, C) -> (B, C, T)
        x = self.backbone(x)
        return self.pooling(x)

# -----------------------------------------------------------------------------
# Quantizer (FSQ)
# -----------------------------------------------------------------------------

class FSQ(nn.Module):
    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)

        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.long), persistent=False)
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.long)
        self.register_buffer("_basis", _basis, persistent=False)

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = torch.round(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        # (B, T, C) -> (B, T)
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis.to(torch.float64)).to(torch.long).sum(dim=-1)

    def encode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_q = self.quantize(z)
        indices = self.codes_to_indices(z_q)  # (B, T)
        return z_q, indices

class FiniteScalarQuantizer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, levels: List[int]) -> None:
        super().__init__()
        self.fsq = FSQ(levels)
        self.proj_in = nn.Linear(input_dim, len(levels))
        self.proj_out = nn.Linear(len(levels), output_dim)

    def encode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.proj_in(z)
        z_q, indices = self.fsq.encode(z)
        z_q = self.proj_out(z_q)
        return z_q, indices

# -----------------------------------------------------------------------------
# Main Standalone Encoder
# -----------------------------------------------------------------------------

class StandaloneEncoder(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        # Initialize sub-modules with known architecture from 25hz_clean.yaml
        self.ssl_model = None # Loaded on demand
        
        self.local_encoder = Transformer(dim=768, n_layers=6, n_heads=12, window_size=125)
        self.local_quantizer = FiniteScalarQuantizer(input_dim=768, output_dim=768, levels=[8, 8, 8, 5, 5])
        self.global_encoder = GlobalEncoder(input_channels=768, output_channels=128, dim=384, intermediate_dim=1152, num_layers=4)
        
        # 25hz_clean downsample factor is 2
        self.conv_downsample = nn.Conv1d(768, 768, kernel_size=2, stride=2)
        self.device = device
        self.to(device)

    def load_weights(self, path: str):
        print(f"[*] Loading weights from {path}...")
        weights = load_file(path)
        
        # Mapping logic: The model might have prefixes like 'model.' or be nested
        new_state_dict = {}
        for k, v in weights.items():
            # Strip common prefixes if they exist
            if k.startswith("model."):
                k = k[6:]
            new_state_dict[k] = v
            
        # We also need to handle FSQ buffers which might be missing in some state dicts
        # because they are computed from levels
        current_model_dict = self.state_dict()
        for k in current_model_dict.keys():
            if k not in new_state_dict and any(x in k for x in ["_levels", "_basis"]):
                # Keep our current buffer
                new_state_dict[k] = current_model_dict[k]

        miss, extra = self.load_state_dict(new_state_dict, strict=False)
        print(f"[+] Loaded weights. Missing: {len(miss)}, Extra: {len(extra)}")
        if len(miss) > 0:
            print(f"[!] Warning: Missing keys: {miss[:10]}...")

    def ensure_ssl_extractor(self, model_name="wavlm_base_plus"):
        if self.ssl_model is not None:
            return
            
        print(f"[*] Loading SSL feature extractor ({model_name})...")
        wavlm_path = ensure_wavlm_path()
        
        bundle = pipelines.WAVLM_BASE_PLUS
        
        if wavlm_path:
            print(f"[*] Loading WavLM weights from {wavlm_path}")
            # Initialize using bundle structure but overwrite weights
            self.ssl_model = bundle.get_model().to(self.device)
            state_dict = load_file(wavlm_path)
            self.ssl_model.load_state_dict(state_dict)
        else:
            print("[*] Downloading/Loading from torchaudio bundle...")
            self.ssl_model = bundle.get_model().to(self.device)
            
        self.ssl_model.eval()
        self.resampler = None # Logic for resampling below

    def _process_ssl_features(self, features: List[torch.Tensor], layers: List[int]) -> torch.Tensor:
        if len(layers) > 1:
            selected = [features[i - 1] for i in layers]
            return torch.stack(selected, dim=0).mean(dim=0)
        return features[layers[0] - 1]

    def _normalize_ssl_features(self, features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # KanadeModel: mean/std across time steps for each sample and feature dimension
        mean = torch.mean(features, dim=1, keepdim=True)  # (B, 1, C)
        std = torch.std(features, dim=1, keepdim=True)   # (B, 1, C)
        return (features - mean) / (std + eps)

    @torch.inference_mode()
    def encode(
        self, 
        waveform: torch.Tensor, 
        sr: int = 24000,
        enforce_token_count: bool = True,
        safety_dur: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio to tokens and global embedding.
        
        Args:
            waveform: Audio tensor (samples,) or (batch, samples)
            sr: Sample rate
            enforce_token_count: If True, truncate tokens to expected count based on duration
            safety_dur: Safety margin in seconds when enforcing token count (default: 0.3s)
            
        Returns:
            (tokens, global_embedding)
        """
        self.ensure_ssl_extractor()
        
        # Handle string path or numpy array
        if isinstance(waveform, str):
            waveform, sr = torchaudio.load(waveform)
        elif isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
            
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Calculate audio duration for token enforcement
        audio_duration = waveform.shape[-1] / sr
        
        # 1. Resample to 16kHz (SSL model requirement)
        target_sr = 16000
        if sr != target_sr:
            if not hasattr(self, "_resampler") or self._resampler_sr != sr:
                self._resampler = torchaudio.transforms.Resample(sr, target_sr).to(self.device)
                self._resampler_sr = sr
            waveform_ssl = self._resampler(waveform.to(self.device))
        else:
            waveform_ssl = waveform.to(self.device)
            
        # 2. Extract SSL Features
        # WavLM base plus: 16kHz input -> 320 hop -> 50Hz features
        all_ssl_features, _ = self.ssl_model.extract_features(waveform_ssl)
        
        # Original KanadeModel uses 1-based layering: [6, 9] -> indices [5, 8]
        # and [1, 2, 3, 4] -> indices [0, 1, 2, 3]
        local_ssl = self._process_ssl_features(all_ssl_features, [6, 9])
        local_ssl = self._normalize_ssl_features(local_ssl)
        
        global_ssl = self._process_ssl_features(all_ssl_features, [1, 2, 3, 4])
        
        # 3. Global Embedding
        global_emb = self.global_encoder(global_ssl).squeeze(0)
        
        # 4. Local Content tokens
        local_encoded = self.local_encoder(local_ssl)
        # Downsample: factor 2
        local_downsampled = self.conv_downsample(local_encoded.transpose(1, 2)).transpose(1, 2)
        _, indices = self.local_quantizer.encode(local_downsampled)
        
        indices = indices.squeeze(0)
        
        # 5. Enforce token count to prevent silence artifacts
        if enforce_token_count:
            expected_tokens = int((audio_duration + safety_dur) * 25)  # 25 tokens/sec
            if len(indices) > expected_tokens:
                indices = indices[:expected_tokens]
        
        return indices, global_emb

def create_standalone_encoder(
    weights_path: str = None,
    device: str = "cpu"
) -> StandaloneEncoder:
    """
    Factory function to create and initialize standalone encoder.
    
    Args:
        weights_path: Path to encoder weights (default: auto-resolve)
        device: Device to use
        
    Returns:
        Initialized encoder
    """
    weights_path = ensure_weights_path(weights_path)
        
    encoder = StandaloneEncoder(device=device)
    encoder.load_weights(weights_path)
    return encoder


if __name__ == "__main__":
    # Test stub
    encoder = create_standalone_encoder(device="cpu")
    # encoder.load_weights("model.safetensors")
    print("[+] Standalone Encoder initialized.")    
