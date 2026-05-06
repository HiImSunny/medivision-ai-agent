import json
import re
from src.model_loader import generate_response

# Supported language codes: en, vn, zh, es, fr, ja

_LANG_INSTRUCTIONS = {
    "en": "Respond in English.",
    "vn": "Respond in Vietnamese (Tiếng Việt).",
    "zh": "Respond in Simplified Chinese (简体中文).",
    "es": "Respond in Spanish (Español).",
    "fr": "Respond in French (Français).",
    "ja": "Respond in Japanese (日本語).",
}


def _build_prompt(image_path: str | None, text_description: str, lang: str) -> str:
    lang_instruction = _LANG_INSTRUCTIONS.get(lang, _LANG_INSTRUCTIONS["en"])
    has_image = bool(image_path)
    return (
        "You are MediVision, a professional dermatology and wound-care assistant.\n"
        f"{lang_instruction}\n"
        "The user has provided"
        + (" an image of a skin condition and" if has_image else "")
        + f" the following symptom description:\n\n{text_description}\n\n"
        "Analyze the above and respond with a single JSON object using these exact keys:\n"
        "  \"diagnosis\": a concise clinical condition name (e.g. 'Contact Dermatitis', "
        "'Superficial Laceration', 'Cellulitis') — NOT a restatement of the patient's "
        "symptoms. Use the standard medical term, translated into the response language.\n"
        "  \"severity\": one of Low | Medium | High | Urgent\n"
        "  \"recommended_actions\": list of 3-5 actionable strings\n"
        "  \"confidence_score\": integer 0-100\n"
        "Return only the JSON object, no extra text."
    )


def _parse_response(raw: str) -> dict:
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return {
                "diagnosis": data.get("diagnosis", "Unknown"),
                "severity": data.get("severity", "Low"),
                "recommended_actions": data.get("recommended_actions", []),
                "confidence_score": int(data.get("confidence_score", 70)),
            }
        except (json.JSONDecodeError, ValueError):
            pass
    raise ValueError(f"Could not parse model response as JSON: {raw[:200]}")


def analyze_image_and_text(
    image_path: str | None,
    text_description: str,
    language: str = "en",
) -> dict:
    """
    Run analysis via AMD Cloud backend.
    Raises RuntimeError if the backend is unreachable.
    Raises ValueError if the model response cannot be parsed.
    """
    lang = language.lower()
    prompt = _build_prompt(image_path, text_description, lang)
    raw = generate_response(prompt, image_path=image_path)
    return _parse_response(raw)
