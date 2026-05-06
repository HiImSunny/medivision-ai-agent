import gradio as gr
from src.inference import MediVisionPipeline
from src.model_loader import check_connection
import src.config as config

# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------
_pipeline: MediVisionPipeline | None = None


def get_pipeline() -> MediVisionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MediVisionPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Backend connection status
# ---------------------------------------------------------------------------

def get_backend_status_html() -> str:
    connected, _ = check_connection()
    if connected:
        dot, label, color = "#22c55e", "AMD Cloud · Live", "#86efac"
    else:
        dot, label, color = "#f97316", "Demo Mode", "#fdba74"
    return (
        f"<div style='text-align:center; margin-bottom:4px;'>"
        f"<span style='font-size:0.75rem; color:{color}; font-family:monospace;'>"
        f"<span style='color:{dot};'>●</span> {label}"
        f"</span></div>"
    )


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_SEVERITY_COLOR = {
    "Low":       ("#22c55e", "#dcfce7"),
    "Medium":    ("#eab308", "#fef9c3"),
    "High":      ("#f97316", "#ffedd5"),
    "Urgent":    ("#ef4444", "#fee2e2"),
    "Thấp":      ("#22c55e", "#dcfce7"),
    "Trung bình": ("#eab308", "#fef9c3"),
    "Cao":       ("#f97316", "#ffedd5"),
    "Khẩn cấp":  ("#ef4444", "#fee2e2"),
}


def _severity_badge(severity: str) -> str:
    color, bg = _SEVERITY_COLOR.get(severity, ("#6b7280", "#f3f4f6"))
    return (
        f"<span style='background:{bg}; color:{color}; font-weight:700; "
        f"padding:4px 14px; border-radius:999px; font-size:0.9rem; "
        f"border:2px solid {color};'>{severity}</span>"
    )


def _confidence_bar(score: int) -> str:
    if score == 0:
        return ""
    fill = "#ED1C24" if score >= 85 else "#f97316" if score >= 70 else "#eab308"
    return (
        f"<div style='margin:6px 0 2px;'>"
        f"  <div style='font-size:0.8rem; color:#9ca3af; margin-bottom:4px;'>"
        f"    Confidence Score / Độ tin cậy: <b style='color:#fff;'>{score}%</b></div>"
        f"  <div style='background:#374151; border-radius:9999px; height:10px; overflow:hidden;'>"
        f"    <div style='background:{fill}; width:{score}%; height:100%; "
        f"border-radius:9999px; transition:width 0.6s ease;'></div>"
        f"  </div>"
        f"</div>"
    )


def _build_result_html(result: dict, lang: str) -> str:
    diag    = result.get("diagnosis", "")
    sev     = result.get("severity", "Low")
    actions = result.get("recommended_actions", [])
    score   = result.get("confidence_score", 0)

    if lang == "vn":
        diag_label     = "Gợi ý chẩn đoán"
        severity_label = "Mức độ nghiêm trọng"
        actions_label  = "Khuyến nghị"
        disclaimer = (
            "Đây là trợ lý AI, không phải bác sĩ. "
            "Hãy tham khảo chuyên gia y tế cho các tình trạng nghiêm trọng."
        )
    else:
        diag_label     = "Diagnosis Suggestion"
        severity_label = "Severity"
        actions_label  = "Recommended Actions"
        disclaimer = (
            "This is an AI assistant, not a licensed physician. "
            "Always consult a healthcare professional for serious conditions."
        )

    actions_html = "".join(
        f"<li style='margin:5px 0; color:#d1d5db;'>{a}</li>"
        for a in actions
    ) if actions else "<li style='color:#6b7280;'>—</li>"

    if config.MOCK_MODE:
        backend_tag = (
            "<span style='font-size:0.7rem; background:#431407; color:#fdba74; "
            "padding:2px 8px; border-radius:4px; margin-left:8px; "
            "border:1px solid #c2410c;'>Demo Mode</span>"
        )
        backend_info = "Demo Mode · Representative Response"
    else:
        backend_tag = (
            "<span style='font-size:0.7rem; background:#052e16; color:#86efac; "
            "padding:2px 8px; border-radius:4px; margin-left:8px; "
            "border:1px solid #16a34a;'>AMD Cloud</span>"
        )
        backend_info = "AMD MI300X · ROCm · Qwen2.5-VL-7B"

    return f"""
<div style='background:#111827; border:1px solid #ED1C24; border-radius:12px;
            padding:20px; font-family:Arial,sans-serif; color:#f9fafb;'>

  <!-- Header -->
  <div style='display:flex; align-items:center; gap:10px; margin-bottom:16px;'>
    <div style='background:#ED1C24; width:4px; border-radius:2px; height:36px;'></div>
    <div>
      <div style='font-size:1.1rem; font-weight:700; color:#ED1C24;'>
        MediVision Analysis {backend_tag}
      </div>
      <div style='font-size:0.75rem; color:#6b7280;'>{backend_info}</div>
    </div>
  </div>

  <!-- Diagnosis -->
  <div style='background:#1f2937; border-radius:8px; padding:14px; margin-bottom:12px;'>
    <div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:.05em;
                color:#9ca3af; margin-bottom:6px;'>{diag_label}</div>
    <div style='font-size:1.05rem; font-weight:600; color:#f9fafb;'>{diag}</div>
  </div>

  <!-- Severity -->
  <div style='background:#1f2937; border-radius:8px; padding:14px; margin-bottom:12px;'>
    <div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:.05em;
                color:#9ca3af; margin-bottom:8px;'>{severity_label}</div>
    {_severity_badge(sev)}
  </div>

  <!-- Confidence bar -->
  <div style='background:#1f2937; border-radius:8px; padding:14px; margin-bottom:12px;'>
    {_confidence_bar(score)}
  </div>

  <!-- Recommended actions -->
  <div style='background:#1f2937; border-radius:8px; padding:14px; margin-bottom:12px;'>
    <div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:.05em;
                color:#9ca3af; margin-bottom:8px;'>{actions_label}</div>
    <ul style='margin:0; padding-left:20px; list-style-type:disc;'>
      {actions_html}
    </ul>
  </div>

  <!-- Disclaimer -->
  <div style='background:#1a1a2e; border-left:4px solid #ED1C24; border-radius:4px;
              padding:10px 14px; font-size:0.78rem; color:#9ca3af;'>
    ⚠️ {disclaimer}
  </div>

</div>
"""


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict(image, symptoms: str, lang_choice: str):
    lang = "vn" if lang_choice == "Tiếng Việt" else "en"

    if not image and not symptoms.strip():
        placeholder = (
            "Please upload an image or enter symptoms."
            if lang == "en"
            else "Vui lòng tải lên hình ảnh hoặc nhập triệu chứng."
        )
        return (
            f"<p style='color:#9ca3af; text-align:center;'>{placeholder}</p>",
            get_backend_status_html(),
        )

    result = get_pipeline().process(image, symptoms.strip(), lang=lang)
    return _build_result_html(result, lang), get_backend_status_html()


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
/* ── Reset & Base ─────────────────────────────────────── */
body, .gradio-container {
    background-color: #030712 !important;
    color: #f9fafb !important;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
}

/* ── AMD Red accent on focused inputs ─────────────────── */
input:focus, textarea:focus {
    border-color: #ED1C24 !important;
    box-shadow: 0 0 0 2px rgba(237,28,36,0.25) !important;
}

/* ── Primary button ────────────────────────────────────── */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, #ED1C24 0%, #b01318 100%) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s !important;
}
button.primary:hover { opacity: 0.88 !important; }

/* ── Card panels ────────────────────────────────────────── */
.gr-box, .gr-panel {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 10px !important;
}

/* ── Labels ─────────────────────────────────────────────── */
label span, .gr-form > label {
    color: #9ca3af !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Gradio footer hiding ───────────────────────────────── */
footer { display: none !important; }

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
"""

# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

HEADER_HTML = """
<div style='text-align:center; padding:24px 0 8px; user-select:none;'>
  <div style='font-size:2rem; font-weight:900; letter-spacing:-0.02em;'>
    <span style='color:#ED1C24;'>Medi</span><span style='color:#f9fafb;'>Vision</span>
  </div>
  <div style='color:#9ca3af; font-size:0.9rem; margin-top:4px;'>
    Multimodal Medical Imaging AI Agent
  </div>
  <div style='margin-top:10px; display:inline-flex; gap:8px; flex-wrap:wrap;
              justify-content:center;'>
    <span style='background:#1f2937; color:#ED1C24; font-size:0.72rem; font-weight:700;
                 padding:3px 12px; border-radius:999px; border:1px solid #ED1C24;'>
      AMD Instinct™ MI300X
    </span>
    <span style='background:#1f2937; color:#9ca3af; font-size:0.72rem; font-weight:600;
                 padding:3px 12px; border-radius:999px; border:1px solid #374151;'>
      ROCm · Qwen2.5-VL-7B
    </span>
    <span style='background:#1f2937; color:#9ca3af; font-size:0.72rem; font-weight:600;
                 padding:3px 12px; border-radius:999px; border:1px solid #374151;'>
      AMD Developer Hackathon 2026
    </span>
  </div>
</div>
"""

FOOTER_HTML = """
<div style='text-align:center; padding:16px 0 4px; border-top:1px solid #1f2937; margin-top:8px;'>
  <span style='color:#4b5563; font-size:0.75rem;'>
    Powered by
    <span style='color:#ED1C24; font-weight:700;'>AMD MI300X + ROCm</span>
    &nbsp;·&nbsp; Track 3: Vision &amp; Multimodal AI
    &nbsp;·&nbsp; MIT License
  </span>
</div>
"""

with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="MediVision — AMD MI300X") as demo:

    gr.HTML(HEADER_HTML)

    # Connection status bar — auto-populated on load, refreshed after each analysis
    status_bar = gr.HTML(value="<div style='height:36px;'></div>")

    with gr.Row(equal_height=False):
        # ── Left column: inputs ───────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            input_img = gr.Image(
                type="filepath",
                label="Upload Image / Tải lên hình ảnh",
                height=230,
            )
            symptoms_txt = gr.Textbox(
                label="Symptoms Description / Mô tả triệu chứng",
                placeholder=(
                    "Describe what you feel — e.g. itchy red patch for 3 days, "
                    "slight burning sensation...\n\n"
                    "Mô tả triệu chứng — ví dụ: vết đỏ ngứa 3 ngày, hơi rát..."
                ),
                lines=4,
            )
            lang_radio = gr.Radio(
                choices=["English", "Tiếng Việt"],
                value="English",
                label="Language / Ngôn ngữ",
            )
            submit_btn = gr.Button(
                "🔬  Analyze  /  Phân tích",
                variant="primary",
                size="lg",
            )

            gr.Examples(
                examples=[
                    [None, "I have a red, itchy rash on my forearm for 3 days. It burns slightly.", "English"],
                    [None, "Small wound on my hand after a cut, slightly swollen with some redness.", "English"],
                    [None, "Vết thương nhỏ ở bàn tay, hơi sưng và có dấu hiệu đỏ xung quanh.", "Tiếng Việt"],
                    [None, "Phát ban đỏ trên cánh tay, ngứa nhiều, đã 2 ngày.", "Tiếng Việt"],
                ],
                inputs=[input_img, symptoms_txt, lang_radio],
                label="Quick Examples / Ví dụ nhanh",
            )

        # ── Right column: output ──────────────────────────────────────────
        with gr.Column(scale=1, min_width=340):
            output_html = gr.HTML(
                value=(
                    "<div style='color:#4b5563; text-align:center; padding:60px 0; "
                    "font-size:0.9rem;'>"
                    "Upload an image and/or describe symptoms, then click Analyze.<br>"
                    "<span style='font-size:0.8rem;'>"
                    "Tải ảnh và/hoặc mô tả triệu chứng, rồi nhấn Phân tích.</span>"
                    "</div>"
                ),
                label="MediVision Analysis",
            )

    submit_btn.click(
        fn=predict,
        inputs=[input_img, symptoms_txt, lang_radio],
        outputs=[output_html, status_bar],
        api_name="analyze",
    )

    # Populate status bar immediately when the page loads
    demo.load(fn=get_backend_status_html, inputs=[], outputs=status_bar)

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
