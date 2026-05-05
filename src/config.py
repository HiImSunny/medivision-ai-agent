import os

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen-VL-Chat")

# Set to True to force mock mode, False to attempt real model loading.
# Auto-overridden to True if the model fails to load at runtime.
MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() == "true"

# Inference device — 'cuda' covers both NVIDIA and AMD ROCm.
DEVICE = os.environ.get("DEVICE", "cuda")

# Generation settings
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))

# HF auth token (optional, set in HF Space secrets)
HF_TOKEN = os.environ.get("HF_TOKEN", None)
