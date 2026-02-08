from setuptools import setup, find_packages

setup(
    name="enc",
    version="0.2.3",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "safetensors",
        "numpy",
        "huggingface_hub",
        "soundfile",
    ],
    description="GPU-Optimized Audio Encoder for Bitwav",
    author="Humair Munir",
    url="https://github.com/humair-m/enc",
)
