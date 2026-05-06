import gradio as gr
from src.inference import MediVisionPipeline
from src.model_loader import check_connection

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
# Language config
# ---------------------------------------------------------------------------

_LANG_MAP = {
    "English":    "en",
    "Tiếng Việt": "vn",
    "中文":        "zh",
    "Español":    "es",
    "Français":   "fr",
    "日本語":      "ja",
}

_LANG_CHOICES = list(_LANG_MAP.keys())

_I18N = {
    "en": {
        "img_label":              "Upload Medical Image",
        "symptoms_label":         "Symptoms Description",
        "symptoms_placeholder":   "Describe what you feel — e.g. itchy red patch for 3 days, slight burning sensation...",
        "analyze_btn":            "🔬  Analyze",
        "output_label":           "Analysis Result",
        "empty_output":           "Upload an image and/or describe symptoms, then click Analyze.",
        "diag_label":             "Diagnosis Suggestion",
        "severity_label":         "Severity",
        "actions_label":          "Recommended Actions",
        "confidence_label":       "Confidence Score",
        "disclaimer":             "This is an AI assistant, not a licensed physician. Always consult a healthcare professional for serious conditions.",
        "placeholder":            "Please upload an image or enter symptoms.",
        "backend_live":           "AMD Cloud · Live",
        "backend_offline":        "AMD Cloud · Offline",
        "error_title":            "Backend Unavailable",
        "error_body":             "AMD Cloud backend is unreachable. Please try again later.",
        "examples_label":         "Quick Examples",
    },
    "vn": {
        "img_label":              "Tải lên hình ảnh y tế",
        "symptoms_label":         "Mô tả triệu chứng",
        "symptoms_placeholder":   "Mô tả những gì bạn cảm thấy — ví dụ: vết đỏ ngứa 3 ngày, hơi rát...",
        "analyze_btn":            "🔬  Phân tích",
        "output_label":           "Kết quả phân tích",
        "empty_output":           "Tải ảnh và/hoặc mô tả triệu chứng, rồi nhấn Phân tích.",
        "diag_label":             "Gợi ý chẩn đoán",
        "severity_label":         "Mức độ nghiêm trọng",
        "actions_label":          "Khuyến nghị",
        "confidence_label":       "Độ tin cậy",
        "disclaimer":             "Đây là trợ lý AI, không phải bác sĩ. Hãy tham khảo chuyên gia y tế cho các tình trạng nghiêm trọng.",
        "placeholder":            "Vui lòng tải lên hình ảnh hoặc nhập triệu chứng.",
        "backend_live":           "AMD Cloud · Trực tuyến",
        "backend_offline":        "AMD Cloud · Ngoại tuyến",
        "error_title":            "Hệ thống không khả dụng",
        "error_body":             "Không thể kết nối AMD Cloud. Vui lòng thử lại sau.",
        "examples_label":         "Ví dụ nhanh",
    },
    "zh": {
        "img_label":              "上传医学图像",
        "symptoms_label":         "症状描述",
        "symptoms_placeholder":   "描述您的感受 — 例如：前臂红色瘙痒皮疹持续3天，略有灼热感...",
        "analyze_btn":            "🔬  分析",
        "output_label":           "分析结果",
        "empty_output":           "请上传图片和/或描述症状，然后点击分析。",
        "diag_label":             "诊断建议",
        "severity_label":         "严重程度",
        "actions_label":          "推荐措施",
        "confidence_label":       "置信度",
        "disclaimer":             "本工具为AI助手，不能替代执业医师。如有严重病情，请务必咨询专业医疗人员。",
        "placeholder":            "请上传图片或输入症状描述。",
        "backend_live":           "AMD Cloud · 在线",
        "backend_offline":        "AMD Cloud · 离线",
        "error_title":            "后端不可用",
        "error_body":             "AMD Cloud 后端无法访问，请稍后重试。",
        "examples_label":         "快速示例",
    },
    "es": {
        "img_label":              "Subir imagen médica",
        "symptoms_label":         "Descripción de síntomas",
        "symptoms_placeholder":   "Describa lo que siente — ej. erupción roja con picazón en el antebrazo desde hace 3 días...",
        "analyze_btn":            "🔬  Analizar",
        "output_label":           "Resultado del análisis",
        "empty_output":           "Suba una imagen y/o describa sus síntomas, luego haga clic en Analizar.",
        "diag_label":             "Sugerencia de diagnóstico",
        "severity_label":         "Severidad",
        "actions_label":          "Acciones recomendadas",
        "confidence_label":       "Puntuación de confianza",
        "disclaimer":             "Este es un asistente de IA, no un médico autorizado. Consulte siempre a un profesional de la salud para condiciones graves.",
        "placeholder":            "Por favor, suba una imagen o describa sus síntomas.",
        "backend_live":           "AMD Cloud · En línea",
        "backend_offline":        "AMD Cloud · Sin conexión",
        "error_title":            "Backend no disponible",
        "error_body":             "El backend de AMD Cloud no está disponible. Por favor, inténtelo más tarde.",
        "examples_label":         "Ejemplos rápidos",
    },
    "fr": {
        "img_label":              "Télécharger une image médicale",
        "symptoms_label":         "Description des symptômes",
        "symptoms_placeholder":   "Décrivez ce que vous ressentez — ex. éruption rouge et prurigineuse sur l'avant-bras depuis 3 jours...",
        "analyze_btn":            "🔬  Analyser",
        "output_label":           "Résultat de l'analyse",
        "empty_output":           "Téléchargez une image et/ou décrivez vos symptômes, puis cliquez sur Analyser.",
        "diag_label":             "Suggestion de diagnostic",
        "severity_label":         "Sévérité",
        "actions_label":          "Actions recommandées",
        "confidence_label":       "Score de confiance",
        "disclaimer":             "Ceci est un assistant IA, pas un médecin agréé. Consultez toujours un professionnel de santé pour les situations graves.",
        "placeholder":            "Veuillez télécharger une image ou décrire vos symptômes.",
        "backend_live":           "AMD Cloud · En ligne",
        "backend_offline":        "AMD Cloud · Hors ligne",
        "error_title":            "Backend indisponible",
        "error_body":             "Le backend AMD Cloud est inaccessible. Veuillez réessayer plus tard.",
        "examples_label":         "Exemples rapides",
    },
    "ja": {
        "img_label":              "医療画像をアップロード",
        "symptoms_label":         "症状の説明",
        "symptoms_placeholder":   "感じていることを説明してください — 例：3日前から前腕に赤くかゆい発疹、少し灼熱感...",
        "analyze_btn":            "🔬  分析する",
        "output_label":           "分析結果",
        "empty_output":           "画像をアップロードし、症状を説明してから分析ボタンをクリックしてください。",
        "diag_label":             "診断提案",
        "severity_label":         "重症度",
        "actions_label":          "推奨アクション",
        "confidence_label":       "信頼スコア",
        "disclaimer":             "これはAIアシスタントであり、有資格の医師ではありません。深刻な症状については必ず医療専門家に相談してください。",
        "placeholder":            "画像をアップロードするか、症状を入力してください。",
        "backend_live":           "AMD Cloud · オンライン",
        "backend_offline":        "AMD Cloud · オフライン",
        "error_title":            "バックエンド利用不可",
        "error_body":             "AMD Cloudバックエンドに接続できません。後でもう一度お試しください。",
        "examples_label":         "クイック例",
    },
}


def _detect_lang_from_header(accept_language: str) -> str:
    """Map Accept-Language header to a display name from _LANG_CHOICES."""
    if not accept_language:
        return "English"
    first = accept_language.split(",")[0].split(";")[0].strip().lower()
    code = first.split("-")[0]
    mapping = {"vi": "Tiếng Việt", "zh": "中文", "es": "Español", "fr": "Français", "ja": "日本語"}
    return mapping.get(code, "English")


# ---------------------------------------------------------------------------
# Backend status
# ---------------------------------------------------------------------------

def get_backend_status_html(lang: str = "en") -> str:
    t = _I18N.get(lang, _I18N["en"])
    connected, _ = check_connection()
    if connected:
        dot, label, color = "#22c55e", t["backend_live"], "#86efac"
    else:
        dot, label, color = "#ef4444", t["backend_offline"], "#fca5a5"
    return (
        f"<div style='font-size:0.75rem; color:{color}; font-family:monospace; "
        f"white-space:nowrap; padding:6px 0;'>"
        f"<span style='color:{dot};'>●</span> {label}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Severity / confidence helpers
# ---------------------------------------------------------------------------

_SEVERITY_COLOR = {
    "Low": ("#22c55e", "#dcfce7"),    "Thấp": ("#22c55e", "#dcfce7"),
    "低": ("#22c55e", "#dcfce7"),     "Baja": ("#22c55e", "#dcfce7"),
    "Faible": ("#22c55e", "#dcfce7"), "軽度": ("#22c55e", "#dcfce7"),

    "Medium": ("#eab308", "#fef9c3"),   "Trung bình": ("#eab308", "#fef9c3"),
    "中": ("#eab308", "#fef9c3"),       "Media": ("#eab308", "#fef9c3"),
    "Modérée": ("#eab308", "#fef9c3"), "中等度": ("#eab308", "#fef9c3"),

    "High": ("#f97316", "#ffedd5"),   "Cao": ("#f97316", "#ffedd5"),
    "高": ("#f97316", "#ffedd5"),     "Alta": ("#f97316", "#ffedd5"),
    "Élevée": ("#f97316", "#ffedd5"), "重度": ("#f97316", "#ffedd5"),

    "Urgent": ("#ef4444", "#fee2e2"),    "Khẩn cấp": ("#ef4444", "#fee2e2"),
    "紧急": ("#ef4444", "#fee2e2"),     "Urgente": ("#ef4444", "#fee2e2"),
    "Urgente": ("#ef4444", "#fee2e2"),  "緊急": ("#ef4444", "#fee2e2"),
}


def _severity_badge(severity: str) -> str:
    color, bg = _SEVERITY_COLOR.get(severity, ("#6b7280", "#f3f4f6"))
    return (
        f"<span style='background:{bg}; color:{color}; font-weight:700; "
        f"padding:4px 14px; border-radius:999px; font-size:0.9rem; "
        f"border:2px solid {color};'>{severity}</span>"
    )


def _confidence_bar(score: int, label: str) -> str:
    if score == 0:
        return ""
    fill = "#ED1C24" if score >= 85 else "#f97316" if score >= 70 else "#eab308"
    return (
        f"<div style='margin:6px 0 2px;'>"
        f"  <div style='font-size:0.8rem; color:#9ca3af; margin-bottom:4px;'>"
        f"    {label}: <b style='color:#fff;'>{score}%</b></div>"
        f"  <div style='background:#374151; border-radius:9999px; height:10px; overflow:hidden;'>"
        f"    <div style='background:{fill}; width:{score}%; height:100%; "
        f"border-radius:9999px; transition:width 0.6s ease;'></div>"
        f"  </div>"
        f"</div>"
    )


def _empty_output_html(lang: str) -> str:
    t = _I18N.get(lang, _I18N["en"])
    return (
        f"<div style='color:#4b5563; text-align:center; padding:60px 0; font-size:0.9rem;'>"
        f"{t['empty_output']}"
        f"</div>"
    )


def _build_result_html(result: dict, lang: str) -> str:
    t       = _I18N.get(lang, _I18N["en"])
    diag    = result.get("diagnosis", "")
    sev     = result.get("severity", "Low")
    actions = result.get("recommended_actions", [])
    score   = result.get("confidence_score", 0)

    actions_html = "".join(
        f"<li style='margin:5px 0; color:#d1d5db;'>{a}</li>" for a in actions
    ) if actions else "<li style='color:#6b7280;'>—</li>"

    backend_tag = (
        "<span style='font-size:0.7rem; background:#052e16; color:#86efac; "
        "padding:2px 8px; border-radius:4px; margin-left:8px; "
        "border:1px solid #16a34a;'>AMD Cloud</span>"
    )

    return f"""
<div style='background:#111827; border:1px solid #ED1C24; border-radius:12px;
            padding:20px; font-family:Arial,sans-serif; color:#f9fafb;'>

  <div style='display:flex; align-items:center; gap:10px; margin-bottom:16px;'>
    <div style='background:#ED1C24; width:4px; border-radius:2px; height:36px;'></div>
    <div>
      <div style='font-size:1.1rem; font-weight:700; color:#ED1C24;'>
        MediVision {backend_tag}
      </div>
      <div style='font-size:0.75rem; color:#6b7280;'>AMD MI300X · ROCm · Qwen2.5-VL-7B</div>
    </div>
  </div>

  <div style='background:#1f2937; border-radius:8px; padding:14px; margin-bottom:12px;'>
    <div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:.05em;
                color:#9ca3af; margin-bottom:6px;'>{t['diag_label']}</div>
    <div style='font-size:1.05rem; font-weight:600; color:#f9fafb;'>{diag}</div>
  </div>

  <div style='background:#1f2937; border-radius:8px; padding:14px; margin-bottom:12px;'>
    <div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:.05em;
                color:#9ca3af; margin-bottom:8px;'>{t['severity_label']}</div>
    {_severity_badge(sev)}
  </div>

  <div style='background:#1f2937; border-radius:8px; padding:14px; margin-bottom:12px;'>
    {_confidence_bar(score, t['confidence_label'])}
  </div>

  <div style='background:#1f2937; border-radius:8px; padding:14px; margin-bottom:12px;'>
    <div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:.05em;
                color:#9ca3af; margin-bottom:8px;'>{t['actions_label']}</div>
    <ul style='margin:0; padding-left:20px; list-style-type:disc;'>
      {actions_html}
    </ul>
  </div>

  <div style='background:#1a1a2e; border-left:4px solid #ED1C24; border-radius:4px;
              padding:10px 14px; font-size:0.78rem; color:#9ca3af;'>
    ⚠️ {t['disclaimer']}
  </div>

</div>
"""


# ---------------------------------------------------------------------------
# UI update helpers
# ---------------------------------------------------------------------------

def _ui_updates(lang_choice: str):
    """Return gr.update() for all translatable UI components."""
    lang = _LANG_MAP.get(lang_choice, "en")
    t = _I18N[lang]
    return (
        gr.update(label=t["img_label"]),
        gr.update(label=t["symptoms_label"], placeholder=t["symptoms_placeholder"]),
        gr.update(value=t["analyze_btn"]),
        gr.update(value=_empty_output_html(lang)),
    )


def on_lang_change(lang_choice: str):
    return _ui_updates(lang_choice)


def on_load(request: gr.Request):
    lang_display = _detect_lang_from_header(
        request.headers.get("accept-language", "")
    )
    img_upd, sym_upd, btn_upd, out_upd = _ui_updates(lang_display)
    lang = _LANG_MAP.get(lang_display, "en")
    return lang_display, img_upd, sym_upd, btn_upd, out_upd, get_backend_status_html(lang)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict(image, symptoms: str, lang_choice: str):
    lang = _LANG_MAP.get(lang_choice, "en")
    t = _I18N[lang]

    if not image and not symptoms.strip():
        return _empty_output_html(lang), get_backend_status_html(lang)

    try:
        result = get_pipeline().process(image, symptoms.strip(), lang=lang)
        return _build_result_html(result, lang), get_backend_status_html(lang)
    except Exception as exc:
        error_html = (
            "<div style='background:#111827; border:1px solid #ef4444; border-radius:12px; "
            "padding:24px; font-family:Arial,sans-serif; text-align:center;'>"
            "<div style='font-size:1.5rem; margin-bottom:12px;'>⚠️</div>"
            f"<div style='font-size:1rem; font-weight:700; color:#ef4444; margin-bottom:8px;'>{t['error_title']}</div>"
            f"<div style='font-size:0.85rem; color:#9ca3af; margin-bottom:16px;'>{t['error_body']}</div>"
            f"<div style='font-size:0.75rem; color:#6b7280; font-family:monospace; "
            f"background:#1f2937; padding:8px 12px; border-radius:6px;'>{exc}</div>"
            "</div>"
        )
        return error_html, get_backend_status_html(lang)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
body, .gradio-container {
    background-color: #030712 !important;
    color: #f9fafb !important;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
}
input:focus, textarea:focus {
    border-color: #ED1C24 !important;
    box-shadow: 0 0 0 2px rgba(237,28,36,0.25) !important;
}
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, #ED1C24 0%, #b01318 100%) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s !important;
}
button.primary:hover { opacity: 0.88 !important; }
.gr-box, .gr-panel {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 10px !important;
}
label span, .gr-form > label {
    color: #9ca3af !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
footer { display: none !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }

/* ── Topbar: status left, lang right ─────────────────────── */
#topbar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 4px 0 !important;
    border-bottom: 1px solid #1f2937;
    margin-bottom: 12px;
}
#topbar > .gr-row,
#topbar > div { width: 100% !important; }
#lang-col { min-width: 180px !important; max-width: 200px !important; }
#lang-col label span { text-transform: none !important; font-size: 0.78rem !important; }
"""

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

HEADER_HTML = """
<div style='text-align:center; padding:24px 0 8px; user-select:none;'>
  <div style='font-size:2rem; font-weight:900; letter-spacing:-0.02em;'>
    <span style='color:#ED1C24;'>Medi</span><span style='color:#f9fafb;'>Vision</span>
  </div>
  <div style='color:#9ca3af; font-size:0.9rem; margin-top:4px;'>
    Multilingual Multimodal Medical Imaging AI Agent
  </div>
  <div style='margin-top:10px; display:inline-flex; gap:8px; flex-wrap:wrap; justify-content:center;'>
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
      EN · VI · ZH · ES · FR · JA
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
    Powered by <span style='color:#ED1C24; font-weight:700;'>AMD MI300X + ROCm</span>
    &nbsp;·&nbsp; Track 3: Vision &amp; Multimodal AI &nbsp;·&nbsp; MIT License
  </span>
</div>
"""

with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="MediVision — AMD MI300X") as demo:

    gr.HTML(HEADER_HTML)

    # ── Topbar: status (left) + language picker (right) ──────────────────────
    with gr.Row(elem_id="topbar"):
        with gr.Column(scale=5):
            status_bar = gr.HTML(value="<div style='height:24px;'></div>")
        with gr.Column(scale=0, elem_id="lang-col"):
            lang_radio = gr.Dropdown(
                choices=_LANG_CHOICES,
                value="English",
                label="🌐 Language",
                container=False,
                show_label=False,
            )

    # ── Main content ──────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=300):
            input_img = gr.Image(
                type="filepath",
                label="Upload Medical Image",
                height=230,
            )
            symptoms_txt = gr.Textbox(
                label="Symptoms Description",
                placeholder="Describe what you feel — e.g. itchy red patch for 3 days...",
                lines=4,
            )
            submit_btn = gr.Button("🔬  Analyze", variant="primary", size="lg")

            gr.Examples(
                examples=[
                    [None, "I have a red, itchy rash on my forearm for 3 days. It burns slightly.", "English"],
                    [None, "Small wound on my hand after a cut, slightly swollen with some redness.", "English"],
                    [None, "Vết thương nhỏ ở bàn tay, hơi sưng và có dấu hiệu đỏ xung quanh.", "Tiếng Việt"],
                    [None, "手臂上出现红色瘙痒皮疹，已持续3天，略有灼热感。", "中文"],
                    [None, "Tengo una erupción roja y con picazón en el antebrazo desde hace 3 días.", "Español"],
                    [None, "J'ai une éruption rouge et prurigineuse sur l'avant-bras depuis 3 jours.", "Français"],
                    [None, "3日前から前腕に赤くてかゆい発疹があり、少し灼熱感があります。", "日本語"],
                ],
                inputs=[input_img, symptoms_txt, lang_radio],
                label="Quick Examples",
            )

        with gr.Column(scale=1, min_width=340):
            output_html = gr.HTML(
                value=_empty_output_html("en"),
                label="Analysis Result",
            )

    # ── Events ───────────────────────────────────────────────────────────────

    lang_radio.change(
        fn=on_lang_change,
        inputs=[lang_radio],
        outputs=[input_img, symptoms_txt, submit_btn, output_html],
    )

    submit_btn.click(
        fn=predict,
        inputs=[input_img, symptoms_txt, lang_radio],
        outputs=[output_html, status_bar],
        api_name="analyze",
    )

    demo.load(
        fn=on_load,
        inputs=[],
        outputs=[lang_radio, input_img, symptoms_txt, submit_btn, output_html, status_bar],
    )

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
