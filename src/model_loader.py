"""
Inference backend: calls the vLLM server on AMD Developer Cloud via the
OpenAI-compatible API.  No local model weights are loaded here.
"""
import base64
import mimetypes
import os
import time

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
        print(f"[Connection] FAILED — Timeout after 5s")
        return False, "Timeout (5s)"
    except Exception as exc:
        print(f"[Connection] FAILED — {type(exc).__name__}: {exc}")
        return False, f"{type(exc).__name__}: {exc}"


def generate_response(prompt: str, image_path: str = None) -> tuple[str, dict]:
    """
    Send a request to the vLLM endpoint and return (text_output, metrics).

    metrics keys:
        latency_ms  – wall-clock time for the API call in milliseconds
        total_tokens – total tokens used (prompt + completion), or 0 if unavailable
        tokens_per_sec – completion tokens / latency, or 0 if unavailable

    Raises RuntimeError if the backend is unreachable or returns an error.
    """
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
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=messages,
            max_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        tokens_per_sec = (completion_tokens / (latency_ms / 1000)) if latency_ms > 0 and completion_tokens > 0 else 0

        metrics = {
            "latency_ms": round(latency_ms),
            "total_tokens": total_tokens,
            "tokens_per_sec": round(tokens_per_sec, 1),
        }
        return response.choices[0].message.content, metrics

    except Exception as exc:
        raise RuntimeError(f"AMD Cloud backend unreachable: {exc}") from exc
