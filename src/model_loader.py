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


def generate_response(
    system_prompt: str,
    user_prompt: str,
    image_path: str = None,
    image_path_2: str = None,
    max_tokens: int = None,
    temperature: float = None,
    force_json: bool = False,
) -> tuple[str, dict]:
    """
    Send a chat completion to the vLLM endpoint with proper system/user separation.

    system_prompt → role: system
    user_prompt   → role: user (may include 0, 1, or 2 images)

    Returns (text_output, metrics).
    metrics keys: latency_ms, total_tokens, tokens_per_sec
    """
    try:
        client = _get_client()
        _max_tokens = max_tokens if max_tokens is not None else config.MAX_NEW_TOKENS
        _temperature = temperature if temperature is not None else config.TEMPERATURE

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if image_path or image_path_2:
            content = []
            if image_path:
                b64, mime = _encode_image(image_path)
                content.append({"type": "image_url",
                                 "image_url": {"url": f"data:{mime};base64,{b64}"}})
            if image_path_2:
                b64, mime = _encode_image(image_path_2)
                content.append({"type": "image_url",
                                 "image_url": {"url": f"data:{mime};base64,{b64}"}})
            content.append({"type": "text", "text": user_prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_prompt})

        kwargs = dict(
            model=config.MODEL_NAME,
            messages=messages,
            max_tokens=_max_tokens,
            temperature=_temperature,
        )
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        response = client.chat.completions.create(**kwargs)
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


def generate_text(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = None,
    temperature: float = None,
    force_json: bool = False,
) -> tuple[str, dict]:
    """Text-only call — no image encoding."""
    return generate_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        force_json=force_json,
    )
