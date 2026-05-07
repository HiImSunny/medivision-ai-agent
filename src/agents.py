import json
import re

from src.model_loader import generate_response, generate_text
from src.prompts import VISION_AGENT_SYSTEM, CLINICAL_AGENT_SYSTEM, PATIENT_AGENT_SYSTEM, SOAP_AGENT_SYSTEM, CHAT_AGENT_SYSTEM

_LANG_NAMES = {
    "en": "English",
    "vn": "Vietnamese",
    "zh": "Simplified Chinese",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
}


_NO_IMAGE_DESC = "(No image provided — assessment based on patient symptom text only.)"
_ZERO_METRICS = {"latency_ms": 0, "total_tokens": 0, "tokens_per_sec": 0}


def vision_agent(image_path_1, image_path_2, symptoms: str) -> tuple[str, dict]:
    """Step 1: strictly objective visual description. Returns (description_text, metrics)."""
    if not image_path_1 and not image_path_2:
        return _NO_IMAGE_DESC, _ZERO_METRICS.copy()
    two_images = bool(image_path_2)
    user_msg = VISION_AGENT_SYSTEM + "\n\n"
    if two_images:
        user_msg += "TWO images are provided: the first image is Day 1, the second image is Day X.\n\n"
    user_msg += f"Patient symptom text: {symptoms or '(none provided)'}"
    return generate_response(user_msg, image_path=image_path_1 or None,
                             image_path_2=image_path_2 or None)


def clinical_agent(visual_description: str, symptoms: str) -> tuple[dict, dict]:
    """Step 2: clinical reasoning → strict JSON. Returns (parsed_dict, metrics)."""
    prompt = (
        CLINICAL_AGENT_SYSTEM + "\n\n"
        f"VISUAL DESCRIPTION:\n{visual_description}\n\n"
        f"PATIENT SYMPTOMS:\n{symptoms or '(none provided)'}"
    )
    raw, metrics = generate_text(prompt)
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        raise ValueError(f"Clinical agent did not return JSON: {raw[:300]}")
    data = json.loads(match.group())
    return {
        "triage_level":        data.get("triage_level", "Low"),
        "possible_conditions": data.get("possible_conditions", []),
        "clinical_assessment": data.get("clinical_assessment", ""),
        "recommendation":      data.get("recommendation", ""),
    }, metrics


def chat_agent(question: str, context: dict, history: list, lang: str) -> tuple[str, dict]:
    """Follow-up Q&A. Returns (answer_text, metrics)."""
    lang_name = _LANG_NAMES.get(lang, "English")
    ctx_block = (
        f"ANALYSIS CONTEXT:\n"
        f"- Visual description: {context.get('visual_description', '(none)')}\n"
        f"- Possible conditions: {', '.join(context.get('possible_conditions', []))}\n"
        f"- Triage level: {context.get('triage_level', 'Low')}\n"
        f"- Patient message given: {context.get('patient_message', '(none)')}"
    )
    history_block = ""
    for user_msg, bot_msg in (history or []):
        history_block += f"\nPatient: {user_msg}\nAssistant: {bot_msg}"
    prompt = (
        CHAT_AGENT_SYSTEM + "\n\n"
        f"TARGET LANGUAGE: {lang_name}\n\n"
        f"{ctx_block}\n"
        f"{history_block}\n\n"
        f"Patient: {question}\nAssistant:"
    )
    answer, metrics = generate_text(prompt)
    return answer.strip(), metrics


def format_agent(clinical_json: dict, visual_description: str,
                 symptoms: str, lang: str) -> tuple[str, str, dict]:
    """Step 3a+3b: patient message and SOAP note as two separate LLM calls."""
    lang_name = _LANG_NAMES.get(lang, "English")
    context = (
        f"TARGET LANGUAGE: {lang_name}\n\n"
        f"PATIENT ORIGINAL COMPLAINT: {symptoms or '(none)'}\n\n"
        f"VISUAL DESCRIPTION (Objective):\n{visual_description}\n\n"
        f"CLINICAL JSON:\n{json.dumps(clinical_json, ensure_ascii=False, indent=2)}"
    )
    patient_msg, m3a = generate_text(PATIENT_AGENT_SYSTEM + "\n\n" + context)
    soap,        m3b = generate_text(SOAP_AGENT_SYSTEM    + "\n\n" + context)
    metrics = {
        "latency_ms":    m3a["latency_ms"] + m3b["latency_ms"],
        "total_tokens":  m3a["total_tokens"] + m3b["total_tokens"],
        "tokens_per_sec": round((m3a.get("tokens_per_sec", 0) + m3b.get("tokens_per_sec", 0)) / 2, 1),
    }
    return patient_msg.strip(), soap.strip(), metrics
