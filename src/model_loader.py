import os
from src.config import MODEL_NAME, DEVICE, MAX_NEW_TOKENS, TEMPERATURE, HF_TOKEN
import src.config as config

_model = None
_processor = None


def _try_load_real_model():
    """Attempt to load Qwen-VL-Chat via transformers + optimum[amd]."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image

    print(f"[ModelLoader] Loading {MODEL_NAME} on device '{DEVICE}'...")
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    # Load tokenizer (Qwen-VL uses AutoTokenizer)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"[ModelLoader] Model loaded on {device}.")
    return model, tokenizer


def get_model_and_processor():
    global _model, _processor

    if _model is not None:
        return _model, _processor

    if config.MOCK_MODE:
        print("[ModelLoader] MOCK_MODE=True — skipping real model load.")
        return None, None

    try:
        _model, _processor = _try_load_real_model()
    except Exception as exc:
        print(f"[ModelLoader] Real model load failed ({exc}). Enabling MOCK_MODE.")
        config.MOCK_MODE = True
        _model, _processor = None, None

    return _model, _processor


def generate_response(prompt: str, image_path: str = None) -> str:
    """
    Run inference with the loaded model, or return a sentinel for mock mode.
    Returns None when in mock mode so callers can use their own mock logic.
    """
    import torch
    from PIL import Image

    model, tokenizer = get_model_and_processor()
    if model is None:
        return None  # caller handles mock

    device = next(model.parameters()).device

    if image_path:
        query = tokenizer.from_list_format([
            {"image": image_path},
            {"text": prompt},
        ])
    else:
        query = prompt

    inputs = tokenizer(query, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0,
        )

    # Decode only the newly generated tokens
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)
