import os

# vLLM server on AMD Developer Cloud (OpenAI-compatible endpoint)
VLLM_API_URL = os.environ.get("VLLM_API_URL", "http://localhost:8000")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")

# Generation settings
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
