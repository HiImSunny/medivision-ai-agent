import json
import re


def _clean_icd10(code: str) -> str:
    """Strip any non-ASCII or non-alphanumeric prefix/suffix from ICD-10 codes.
    Models like Qwen sometimes prepend the Chinese translation before the code."""
    return re.sub(r"[^A-Za-z0-9.\-]", "", code)

from src.model_loader import generate_response, generate_text
from src.prompts import (
    VISION_AGENT_SYSTEM,
    CLINICAL_AGENT_SYSTEM,
    PATIENT_AGENT_SYSTEM,
    SOAP_AGENT_SYSTEM,
    CHAT_AGENT_SYSTEM,
)

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


def _extract_json(raw: str) -> dict:
    """Robustly extract first JSON object from LLM output, stripping markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Scan for first balanced {...} block
    depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(cleaned[start:i + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"No valid JSON object found in response: {raw[:300]}")


def vision_agent(image_path_1, image_path_2, symptoms: str) -> tuple[str, dict]:
    """
    Step 1: objective visual description.
    Returns (description_text, metrics).
    """
    if not image_path_1 and not image_path_2:
        return _NO_IMAGE_DESC, _ZERO_METRICS.copy()

    two_images = bool(image_path_2)
    user_prompt = ""
    if two_images:
        user_prompt += "TWO images provided: first image is Day 1 (baseline), second image is Day X (follow-up).\n\n"
    user_prompt += f"Patient symptom report: {symptoms or '(none provided)'}\n\nAnalyze the image(s) as instructed."

    return generate_response(
        system_prompt=VISION_AGENT_SYSTEM,
        user_prompt=user_prompt,
        image_path=image_path_1 or None,
        image_path_2=image_path_2 or None,
        max_tokens=600,
        temperature=0.0,
    )


def clinical_agent(visual_description: str, symptoms: str, lang: str = "en") -> tuple[dict, dict]:
    """
    Step 2: clinical reasoning → structured JSON with richer schema.
    Returns (parsed_dict, metrics).
    """
    lang_name = _LANG_NAMES.get(lang, "English")
    user_prompt = (
        f"TARGET LANGUAGE: {lang_name}\n\n"
        f"VISUAL DESCRIPTION:\n{visual_description}\n\n"
        f"PATIENT SYMPTOMS:\n{symptoms or '(none provided)'}"
    )

    raw, metrics = generate_text(
        system_prompt=CLINICAL_AGENT_SYSTEM,
        user_prompt=user_prompt,
        max_tokens=800,
        temperature=0.0,
        force_json=True,
    )

    data = _extract_json(raw)

    # Normalise possible_conditions — support new {name, probability, icd10} schema
    # and gracefully handle plain-string fallback from older model outputs
    raw_conditions = data.get("possible_conditions", [])
    conditions = []
    for item in raw_conditions:
        if isinstance(item, dict):
            conditions.append({
                "name":        str(item.get("name", item.get("condition", "Unknown"))),
                "probability": int(item.get("probability", item.get("match_probability", 50))),
                "icd10":       _clean_icd10(str(item.get("icd10", item.get("icd10_code", "")))),
            })
        elif isinstance(item, str):
            conditions.append({"name": item, "probability": 50, "icd10": ""})

    return {
        "triage_level":        data.get("triage_level", "Low"),
        "urgency_reason":      data.get("urgency_reason", ""),
        "possible_conditions": conditions,
        "red_flags":           data.get("red_flags", []),
        "watch_symptoms":      data.get("watch_symptoms", []),
        "clinical_assessment": data.get("clinical_assessment", ""),
        "recommendation":      data.get("recommendation", ""),
    }, metrics


def chat_agent(question: str, context: dict, history: list, lang: str) -> tuple[str, dict]:
    """
    Follow-up Q&A. Returns (answer_text, metrics).
    """
    lang_name = _LANG_NAMES.get(lang, "English")

    conditions_text = ", ".join(
        c["name"] if isinstance(c, dict) else c
        for c in context.get("possible_conditions", [])
    )

    ctx_block = (
        f"ANALYSIS CONTEXT:\n"
        f"- Visual description: {context.get('visual_description', '(none)')}\n"
        f"- Possible conditions: {conditions_text}\n"
        f"- Triage level: {context.get('triage_level', 'Low')}\n"
        f"- Urgency reason: {context.get('urgency_reason', '')}\n"
        f"- Red flags: {'; '.join(context.get('red_flags', [])) or 'none'}\n"
        f"- Patient message: {context.get('patient_message', '(none)')}"
    )

    history_block = ""
    for user_msg, bot_msg in (history or []):
        history_block += f"\nPatient: {user_msg}\nAssistant: {bot_msg}"

    user_prompt = (
        f"TARGET LANGUAGE: {lang_name}\n\n"
        f"{ctx_block}\n"
        f"{history_block}\n\n"
        f"Patient: {question}\nAssistant:"
    )

    answer, metrics = generate_text(
        system_prompt=CHAT_AGENT_SYSTEM,
        user_prompt=user_prompt,
        max_tokens=300,
        temperature=0.3,
    )
    return answer.strip(), metrics


def format_agent(clinical_json: dict, visual_description: str,
                 symptoms: str, lang: str) -> tuple[str, str, dict]:
    """
    Step 3a + 3b: patient message and SOAP note as two separate LLM calls.
    Returns (patient_message, soap_note, combined_metrics).
    """
    lang_name = _LANG_NAMES.get(lang, "English")
    context = (
        f"TARGET LANGUAGE: {lang_name}\n\n"
        f"PATIENT ORIGINAL COMPLAINT: {symptoms or '(none)'}\n\n"
        f"VISUAL DESCRIPTION (Objective):\n{visual_description}\n\n"
        f"CLINICAL JSON:\n{json.dumps(clinical_json, ensure_ascii=False, indent=2)}"
    )

    patient_msg, m3a = generate_text(
        system_prompt=PATIENT_AGENT_SYSTEM,
        user_prompt=context,
        max_tokens=500,
        temperature=0.4,
    )
    soap, m3b = generate_text(
        system_prompt=SOAP_AGENT_SYSTEM,
        user_prompt=context,
        max_tokens=600,
        temperature=0.0,
    )

    metrics = {
        "latency_ms":     m3a["latency_ms"] + m3b["latency_ms"],
        "total_tokens":   m3a["total_tokens"] + m3b["total_tokens"],
        "tokens_per_sec": round(
            (m3a.get("tokens_per_sec", 0) + m3b.get("tokens_per_sec", 0)) / 2, 1
        ),
    }
    return patient_msg.strip(), soap.strip(), metrics
