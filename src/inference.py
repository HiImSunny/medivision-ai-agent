from src.agents import vision_agent, clinical_agent, format_agent


class MediVisionPipeline:
    def process(self, image_path_1, image_path_2, symptoms: str,
                lang: str = "en", region: str = "") -> dict:
        """
        Run the 3-step agentic pipeline:
          Step 1 — Vision Agent: objective visual description
          Step 2 — Clinical Agent: triage JSON
          Step 3 — Format Agent: patient message + SOAP note

        Returns dict with keys:
            triage_level, possible_conditions, patient_message,
            soap_note, visual_description, _metrics
        """
        symptoms_full = f"{'Region: ' + region + '. ' if region else ''}{symptoms}"

        visual_desc, m1 = vision_agent(image_path_1, image_path_2, symptoms_full)
        clinical,    m2 = clinical_agent(visual_desc, symptoms_full, lang=lang)
        patient_msg, soap, m3 = format_agent(clinical, visual_desc, symptoms_full, lang)

        metrics = {
            "latency_ms":    m1["latency_ms"] + m2["latency_ms"] + m3["latency_ms"],
            "total_tokens":  m1["total_tokens"] + m2["total_tokens"] + m3["total_tokens"],
            "tokens_per_sec": round(
                (m1.get("tokens_per_sec", 0) + m2.get("tokens_per_sec", 0) + m3.get("tokens_per_sec", 0)) / 3, 1
            ),
        }
        return {
            "triage_level":        clinical["triage_level"],
            "possible_conditions": clinical["possible_conditions"],
            "patient_message":     patient_msg,
            "soap_note":           soap,
            "visual_description":  visual_desc,
            "_metrics":            metrics,
            # kept for follow-up chat context
            "_clinical":           clinical,
        }
