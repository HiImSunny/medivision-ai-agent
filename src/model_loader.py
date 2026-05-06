"""
Inference backend: calls the vLLM server on AMD Developer Cloud via the
OpenAI-compatible API.  No local model weights are loaded here.
"""
import base64
import mimetypes
import os

import src.config as config

# Lazy singleton — created on first call to generate_response()
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(
            base_url=f"{config.VLLM_API_URL}/v1",
            api_key=os.environ.get("VLLM_API_KEY", "not-required"),
        )
    return _client


def _encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_data, mime_type) for an image file."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime_type


def generate_response(prompt: str, image_path: str = None) -> str | None:
    """
    Send a request to the vLLM endpoint and return the model's text output.
    Returns None when MOCK_MODE is active so callers fall back to mock logic.
    """
    if config.MOCK_MODE:
        return None

    try:
        client = _get_client()

        if image_path:
            b64, mime = _encode_image(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=messages,
            max_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
        )
        return response.choices[0].message.content

    except Exception as exc:
        print(f"[ModelLoader] vLLM call failed ({exc}). Falling back to mock mode.")
        config.MOCK_MODE = True
        return None
