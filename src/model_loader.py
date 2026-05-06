"""
Inference backend: calls the vLLM server on AMD Developer Cloud via the
OpenAI-compatible API.  No local model weights are loaded here.
"""
import base64
import mimetypes
import os

import src.config as config

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
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime_type


def check_connection() -> tuple[bool, str]:
    """
    Ping the vLLM server's /v1/models endpoint.
    Returns (is_connected: bool, status_message: str).
    """
    if config.MOCK_MODE:
        print("[Connection] MOCK_MODE=true — skipping connection check.")
        return False, "Mock mode enabled"

    import requests as req

    url = f"{config.VLLM_API_URL}/v1/models"
    api_key = os.environ.get("VLLM_API_KEY", "not-required")
    print(f"[Connection] Checking AMD Cloud at {url} ...")

    try:
        r = req.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
        if r.status_code == 200:
            models = [m.get("id", "?") for m in r.json().get("data", [])]
            print(f"[Connection] OK — models available: {models}")
            return True, f"Connected · {config.VLLM_API_URL}"
        print(f"[Connection] FAILED — HTTP {r.status_code}: {r.text[:200]}")
        return False, f"HTTP {r.status_code}"
    except req.exceptions.ConnectionError as exc:
        print(f"[Connection] FAILED — ConnectionError: {exc}")
        return False, f"ConnectionError: {exc}"
    except req.exceptions.Timeout:
        print(f"[Connection] FAILED — Timeout after 5s (server may be down or IP wrong)")
        return False, "Timeout (5s)"
    except Exception as exc:
        print(f"[Connection] FAILED — Unexpected error: {type(exc).__name__}: {exc}")
        return False, f"{type(exc).__name__}: {exc}"


def generate_response(prompt: str, image_path: str = None) -> str | None:
    """
    Send a request to the vLLM endpoint and return the model's text output.
    Returns None on failure so callers fall back to mock logic.
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
