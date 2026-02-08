import torch
from safetensors.torch import load_file, save_file
import os
import sys

def extract_encoder(input_path, output_path):
    print(f"[*] Loading full model from {input_path}...")
    weights = load_file(input_path)
    
    encoder_keys = [
        "local_encoder.",
        "local_quantizer.",
        "global_encoder.",
        "conv_downsample."
    ]
    
    extracted_weights = {}
    for k, v in weights.items():
        original_key = k
        if k.startswith("model."):
            k = k[6:]
        
        if any(k.startswith(prefix) for prefix in encoder_keys):
            # We keep the names as they are (with local_encoder. etc prefixes)
            # because StandaloneEncoder expects them that way if it strips 'model.'
            extracted_weights[k] = v
            
    print(f"[+] Extracted {len(extracted_weights)} keys.")
    save_file(extracted_weights, output_path)
    print(f"[#] Saved encoder weights to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_encoder.py <input.safetensors> <output.safetensors>")
        sys.exit(1)
    
    extract_encoder(sys.argv[1], sys.argv[2])
