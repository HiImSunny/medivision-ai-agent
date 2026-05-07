import json
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
        "metrics_latency":        "Latency",
        "metrics_throughput":     "Throughput",
        "metrics_tokens":         "tokens",
        "region_label":           "Affected Body Region",
        "region_none":            "Not specified",
        "region_optional_label":  "Affected Body Region (Optional)",
        "input_hint":             "Provide an image, describe symptoms, or both — at least one is required. Body region is optional.",
        "map_label":              "Anatomical Map",
        "map_select":             "click to select",
        "map_selected":           "{n} region(s) selected",
        "img_mode_label":         "Upload Mode",
        "img_mode_standard":      "Standard (1 image)",
        "img_mode_compare":       "Compare (2 images — before & after)",
        "img_label_day1":         "Medical Image (Day 1)",
        "img_label_dayx":         "Comparison Image (Day X)",
        "tab_patient":            "Patient View",
        "tab_doctor":             "Export for Doctor (SOAP)",
        "critical_warning":       "⚠️ CRITICAL: Severe symptoms detected. Please visit a medical facility within 24 hours.",
        "conditions_label":       "Possible Conditions",
        "soap_copy_btn":          "Copy SOAP Note",
        "soap_empty":             "Run an analysis to generate the SOAP note.",
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
        "metrics_latency":        "Độ trễ",
        "metrics_throughput":     "Thông lượng",
        "metrics_tokens":         "token",
        "region_label":           "Vùng cơ thể bị ảnh hưởng",
        "region_none":            "Không xác định",
        "region_optional_label":  "Vùng cơ thể bị ảnh hưởng (Không bắt buộc)",
        "input_hint":             "Cung cấp ảnh, mô tả triệu chứng hoặc cả hai — cần ít nhất một trong hai. Vùng cơ thể là tùy chọn.",
        "map_label":              "Bản đồ giải phẫu",
        "map_select":             "nhấn để chọn",
        "map_selected":           "{n} vùng đã chọn",
        "img_mode_label":         "Chế độ tải ảnh",
        "img_mode_standard":      "Tiêu chuẩn (1 ảnh)",
        "img_mode_compare":       "So sánh (2 ảnh — trước & sau)",
        "img_label_day1":         "Ảnh y tế (Ngày 1)",
        "img_label_dayx":         "Ảnh so sánh (Ngày X)",
        "tab_patient":            "Dành cho bệnh nhân",
        "tab_doctor":             "Xuất cho bác sĩ (SOAP)",
        "critical_warning":       "⚠️ CẢNH BÁO: Triệu chứng nghiêm trọng được phát hiện. Vui lòng đến cơ sở y tế trong vòng 24 giờ.",
        "conditions_label":       "Tình trạng có thể",
        "soap_copy_btn":          "Sao chép SOAP",
        "soap_empty":             "Thực hiện phân tích để tạo ghi chú SOAP.",
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
        "metrics_latency":        "延迟",
        "metrics_throughput":     "吞吐量",
        "metrics_tokens":         "tokens",
        "region_label":           "受影响的身体部位",
        "region_none":            "未指定",
        "region_optional_label":  "受影响的身体部位（可选）",
        "input_hint":             "请上传图片或描述症状（至少提供其中一项）。身体部位为可选项。",
        "map_label":              "解剖图",
        "map_select":             "点击选择",
        "map_selected":           "已选 {n} 个部位",
        "img_mode_label":         "上传模式",
        "img_mode_standard":      "标准模式（1张图片）",
        "img_mode_compare":       "对比模式（2张图片 — 前后对比）",
        "img_label_day1":         "医疗图像（第1天）",
        "img_label_dayx":         "对比图像（第X天）",
        "tab_patient":            "患者视图",
        "tab_doctor":             "导出给医生（SOAP）",
        "critical_warning":       "⚠️ 严重警告：检测到严重症状。请在24小时内前往医疗机构就诊。",
        "conditions_label":       "可能的病症",
        "soap_copy_btn":          "复制SOAP记录",
        "soap_empty":             "运行分析以生成SOAP记录。",
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
        "metrics_latency":        "Latencia",
        "metrics_throughput":     "Rendimiento",
        "metrics_tokens":         "tokens",
        "region_label":           "Región corporal afectada",
        "region_none":            "No especificado",
        "region_optional_label":  "Región corporal afectada (Opcional)",
        "input_hint":             "Suba una imagen, describa síntomas, o ambos — se requiere al menos uno. La región corporal es opcional.",
        "map_label":              "Mapa anatómico",
        "map_select":             "haga clic para seleccionar",
        "map_selected":           "{n} región(es) seleccionada(s)",
        "img_mode_label":         "Modo de carga",
        "img_mode_standard":      "Estándar (1 imagen)",
        "img_mode_compare":       "Comparar (2 imágenes — antes y después)",
        "img_label_day1":         "Imagen médica (Día 1)",
        "img_label_dayx":         "Imagen de comparación (Día X)",
        "tab_patient":            "Vista del paciente",
        "tab_doctor":             "Exportar para médico (SOAP)",
        "critical_warning":       "⚠️ CRÍTICO: Síntomas graves detectados. Por favor, acuda a un centro médico en las próximas 24 horas.",
        "conditions_label":       "Posibles condiciones",
        "soap_copy_btn":          "Copiar nota SOAP",
        "soap_empty":             "Ejecute un análisis para generar la nota SOAP.",
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
        "metrics_latency":        "Latence",
        "metrics_throughput":     "Débit",
        "metrics_tokens":         "tokens",
        "region_label":           "Région corporelle affectée",
        "region_none":            "Non spécifié",
        "region_optional_label":  "Région corporelle affectée (Facultatif)",
        "input_hint":             "Fournissez une image, décrivez vos symptômes, ou les deux — au moins un est requis. La région corporelle est facultative.",
        "map_label":              "Carte anatomique",
        "map_select":             "cliquer pour sélectionner",
        "map_selected":           "{n} région(s) sélectionnée(s)",
        "img_mode_label":         "Mode de téléchargement",
        "img_mode_standard":      "Standard (1 image)",
        "img_mode_compare":       "Comparer (2 images — avant et après)",
        "img_label_day1":         "Image médicale (Jour 1)",
        "img_label_dayx":         "Image de comparaison (Jour X)",
        "tab_patient":            "Vue patient",
        "tab_doctor":             "Exporter pour le médecin (SOAP)",
        "critical_warning":       "⚠️ CRITIQUE : Symptômes graves détectés. Veuillez vous rendre dans un établissement médical dans les 24 heures.",
        "conditions_label":       "Conditions possibles",
        "soap_copy_btn":          "Copier la note SOAP",
        "soap_empty":             "Lancez une analyse pour générer la note SOAP.",
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
        "metrics_latency":        "レイテンシ",
        "metrics_throughput":     "スループット",
        "metrics_tokens":         "トークン",
        "region_label":           "患部の体の部位",
        "region_none":            "指定なし",
        "region_optional_label":  "患部の体の部位（任意）",
        "input_hint":             "画像または症状（あるいは両方）を入力してください — どちらか一方が必要です。体の部位は任意です。",
        "map_label":              "解剖マップ",
        "map_select":             "クリックして選択",
        "map_selected":           "{n} 部位選択中",
        "img_mode_label":         "アップロードモード",
        "img_mode_standard":      "標準（画像1枚）",
        "img_mode_compare":       "比較（画像2枚 — 経過観察）",
        "img_label_day1":         "医療画像（第1日）",
        "img_label_dayx":         "比較画像（第X日）",
        "tab_patient":            "患者向け",
        "tab_doctor":             "医師向けエクスポート（SOAP）",
        "critical_warning":       "⚠️ 重大：重篤な症状が検出されました。24時間以内に医療機関を受診してください。",
        "conditions_label":       "考えられる疾患",
        "soap_copy_btn":          "SOAPノートをコピー",
        "soap_empty":             "分析を実行してSOAPノートを生成します。",
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
    "Low":    ("#22c55e", "#dcfce7"),
    "Medium": ("#eab308", "#fef9c3"),
    "High":   ("#f97316", "#ffedd5"),
    "Urgent": ("#ef4444", "#fee2e2"),
}

# Translate English severity keys to display language
_SEVERITY_TRANSLATE = {
    "en": {"Low": "Low",       "Medium": "Medium",     "High": "High",   "Urgent": "Urgent"},
    "vn": {"Low": "Thấp",      "Medium": "Trung bình", "High": "Cao",    "Urgent": "Khẩn cấp"},
    "zh": {"Low": "低",         "Medium": "中",          "High": "高",     "Urgent": "紧急"},
    "es": {"Low": "Leve",      "Medium": "Moderada",   "High": "Alta",   "Urgent": "Urgente"},
    "fr": {"Low": "Faible",    "Medium": "Modérée",    "High": "Élevée", "Urgent": "Urgente"},
    "ja": {"Low": "軽度",       "Medium": "中等度",      "High": "重度",   "Urgent": "緊急"},
}


# ---------------------------------------------------------------------------
# Anatomical Region Selector
# ---------------------------------------------------------------------------

# Regions shown as a compact grid; value passed to prompt in English
_BODY_REGIONS = [
    "Head / Face", "Neck", "Chest", "Abdomen",
    "Upper Back", "Lower Back", "Left Arm", "Right Arm",
    "Left Hand", "Right Hand", "Left Leg", "Right Leg",
    "Left Foot", "Right Foot", "Groin / Genital", "Buttocks",
]

# Translated display names — same order as _BODY_REGIONS
_REGION_TRANSLATIONS = {
    "en": ["Head / Face", "Neck", "Chest", "Abdomen",
           "Upper Back", "Lower Back", "Left Arm", "Right Arm",
           "Left Hand", "Right Hand", "Left Leg", "Right Leg",
           "Left Foot", "Right Foot", "Groin / Genital", "Buttocks"],
    "vn": ["Đầu / Mặt", "Cổ", "Ngực", "Bụng",
           "Lưng trên", "Lưng dưới", "Tay trái", "Tay phải",
           "Bàn tay trái", "Bàn tay phải", "Chân trái", "Chân phải",
           "Bàn chân trái", "Bàn chân phải", "Bẹn / Sinh dục", "Mông"],
    "zh": ["头部 / 面部", "颈部", "胸部", "腹部",
           "上背部", "下背部", "左臂", "右臂",
           "左手", "右手", "左腿", "右腿",
           "左脚", "右脚", "腹股沟 / 生殖器", "臀部"],
    "es": ["Cabeza / Cara", "Cuello", "Pecho", "Abdomen",
           "Espalda alta", "Espalda baja", "Brazo izquierdo", "Brazo derecho",
           "Mano izquierda", "Mano derecha", "Pierna izquierda", "Pierna derecha",
           "Pie izquierdo", "Pie derecho", "Ingle / Genitales", "Nalgas"],
    "fr": ["Tête / Visage", "Cou", "Poitrine", "Abdomen",
           "Haut du dos", "Bas du dos", "Bras gauche", "Bras droit",
           "Main gauche", "Main droite", "Jambe gauche", "Jambe droite",
           "Pied gauche", "Pied droit", "Aine / Organes génitaux", "Fesses"],
    "ja": ["頭部 / 顔", "首", "胸部", "腹部",
           "上背部", "下背部", "左腕", "右腕",
           "左手", "右手", "左脚", "右脚",
           "左足", "右足", "鼠径部 / 性器", "臀部"],
}


def _localized_regions(lang: str) -> list:
    return _REGION_TRANSLATIONS.get(lang, _REGION_TRANSLATIONS["en"])


def _display_to_en(display: str) -> str:
    """Map any localized display name (any lang) back to the English region key."""
    if display in _BODY_REGIONS:
        return display
    for translations in _REGION_TRANSLATIONS.values():
        if display in translations:
            return _BODY_REGIONS[translations.index(display)]
    return ""


# Each region maps to one or more SVG shape IDs that should highlight
_REGION_SHAPE_MAP = {
    "Head / Face":      ["svg-head"],
    "Neck":             ["svg-neck"],
    "Chest":            ["svg-chest"],
    "Abdomen":          ["svg-abdomen"],
    "Upper Back":       ["svg-upper-back"],
    "Lower Back":       ["svg-lower-back"],
    "Left Arm":         ["svg-left-arm"],
    "Right Arm":        ["svg-right-arm"],
    "Left Hand":        ["svg-left-hand"],
    "Right Hand":       ["svg-right-hand"],
    "Left Leg":         ["svg-left-leg"],
    "Right Leg":        ["svg-right-leg"],
    "Left Foot":        ["svg-left-foot"],
    "Right Foot":       ["svg-right-foot"],
    "Groin / Genital":  ["svg-groin"],
    "Buttocks":         ["svg-buttocks"],
}

_DIM   = "#374151"   # default fill
_HI    = "#ED1C24"   # highlighted fill


def _body_map_svg(selected: list, lang: str = "en") -> str:
    t = _I18N.get(lang, _I18N["en"])
    active = set()
    for r in (selected or []):
        for sid in _REGION_SHAPE_MAP.get(r, []):
            active.add(sid)

    def f(sid):
        return _HI if sid in active else _DIM

    def stroke(sid):
        return "#ED1C24" if sid in active else "#4b5563"

    def glow(sid):
        return "drop-shadow(0 0 3px #ED1C24)" if sid in active else "none"

    # Build svg_id → localized name for <title> tooltips
    locs = _REGION_TRANSLATIONS.get(lang, _REGION_TRANSLATIONS["en"])
    _svg_title: dict[str, str] = {}
    for en_name, svg_ids in _REGION_SHAPE_MAP.items():
        loc_name = locs[_BODY_REGIONS.index(en_name)] if en_name in _BODY_REGIONS else en_name
        for sid in svg_ids:
            _svg_title[sid] = loc_name

    def title(sid):
        return f"<title>{_svg_title.get(sid, sid)}</title>"

    counter_html = (
        f'<div style="font-size:0.58rem; color:#ED1C24; font-family:monospace; font-weight:600;">'
        f'{t["map_selected"].format(n=len(active))}</div>'
        if active else
        f'<div style="font-size:0.58rem; color:#4b5563; font-family:monospace;">{t["map_select"]}</div>'
    )

    return f"""
<div style='display:flex; flex-direction:column; align-items:center; gap:6px;
            padding:8px 4px; user-select:none;'>
  <div style='font-size:0.58rem; color:#64748b; font-family:monospace;
              letter-spacing:0.06em; text-transform:uppercase;'>{t["map_label"]}</div>
  <svg viewBox="0 0 80 180" width="76" height="170"
       xmlns="http://www.w3.org/2000/svg" style='overflow:visible;'>
    <style>
      .bpart {{ transition: fill 0.2s, filter 0.2s; cursor:pointer; }}
      .bpart:hover {{ fill: #f97316 !important; filter: drop-shadow(0 0 4px #f97316); }}
    </style>
    <!-- Head -->
    <ellipse class="bpart" id="svg-head" cx="40" cy="13" rx="11" ry="12"
             fill="{f('svg-head')}" stroke="{stroke('svg-head')}" stroke-width="0.8"
             style="filter:{glow('svg-head')}">{title('svg-head')}</ellipse>
    <!-- Neck -->
    <rect class="bpart" id="svg-neck" x="35" y="24" width="10" height="8" rx="2"
          fill="{f('svg-neck')}" stroke="{stroke('svg-neck')}" stroke-width="0.8"
          style="filter:{glow('svg-neck')}">{title('svg-neck')}</rect>
    <!-- Chest -->
    <rect class="bpart" id="svg-chest" x="22" y="32" width="36" height="22" rx="4"
          fill="{f('svg-chest')}" stroke="{stroke('svg-chest')}" stroke-width="0.8"
          style="filter:{glow('svg-chest')}">{title('svg-chest')}</rect>
    <!-- Abdomen -->
    <rect class="bpart" id="svg-abdomen" x="22" y="55" width="36" height="20" rx="4"
          fill="{f('svg-abdomen')}" stroke="{stroke('svg-abdomen')}" stroke-width="0.8"
          style="filter:{glow('svg-abdomen')}">{title('svg-abdomen')}</rect>
    <!-- Upper Back (overlay stripe) -->
    <rect class="bpart" id="svg-upper-back" x="22" y="32" width="36" height="11" rx="4"
          fill="{'#ED1C24' if 'svg-upper-back' in active else 'none'}" opacity="0.45"
          stroke="{'#ED1C24' if 'svg-upper-back' in active else 'none'}" stroke-width="0.6">{title('svg-upper-back')}</rect>
    <!-- Lower Back (overlay stripe) -->
    <rect class="bpart" id="svg-lower-back" x="22" y="55" width="36" height="10" rx="4"
          fill="{'#ED1C24' if 'svg-lower-back' in active else 'none'}" opacity="0.45"
          stroke="{'#ED1C24' if 'svg-lower-back' in active else 'none'}" stroke-width="0.6">{title('svg-lower-back')}</rect>
    <!-- Arms -->
    <rect class="bpart" id="svg-left-arm" x="7" y="32" width="13" height="38" rx="5"
          fill="{f('svg-left-arm')}" stroke="{stroke('svg-left-arm')}" stroke-width="0.8"
          style="filter:{glow('svg-left-arm')}">{title('svg-left-arm')}</rect>
    <rect class="bpart" id="svg-right-arm" x="60" y="32" width="13" height="38" rx="5"
          fill="{f('svg-right-arm')}" stroke="{stroke('svg-right-arm')}" stroke-width="0.8"
          style="filter:{glow('svg-right-arm')}">{title('svg-right-arm')}</rect>
    <!-- Hands -->
    <ellipse class="bpart" id="svg-left-hand" cx="13" cy="76" rx="7" ry="5"
             fill="{f('svg-left-hand')}" stroke="{stroke('svg-left-hand')}" stroke-width="0.8"
             style="filter:{glow('svg-left-hand')}">{title('svg-left-hand')}</ellipse>
    <ellipse class="bpart" id="svg-right-hand" cx="67" cy="76" rx="7" ry="5"
             fill="{f('svg-right-hand')}" stroke="{stroke('svg-right-hand')}" stroke-width="0.8"
             style="filter:{glow('svg-right-hand')}">{title('svg-right-hand')}</ellipse>
    <!-- Groin -->
    <rect class="bpart" id="svg-groin" x="27" y="76" width="26" height="8" rx="3"
          fill="{f('svg-groin')}" stroke="{stroke('svg-groin')}" stroke-width="0.8"
          style="filter:{glow('svg-groin')}">{title('svg-groin')}</rect>
    <!-- Buttocks overlay -->
    <rect class="bpart" id="svg-buttocks" x="27" y="76" width="26" height="8" rx="3"
          fill="{'#ED1C24' if 'svg-buttocks' in active else 'none'}" opacity="0.55"
          stroke="{'#ED1C24' if 'svg-buttocks' in active else 'none'}" stroke-width="0.6">{title('svg-buttocks')}</rect>
    <!-- Legs -->
    <rect class="bpart" id="svg-left-leg" x="22" y="85" width="15" height="52" rx="5"
          fill="{f('svg-left-leg')}" stroke="{stroke('svg-left-leg')}" stroke-width="0.8"
          style="filter:{glow('svg-left-leg')}">{title('svg-left-leg')}</rect>
    <rect class="bpart" id="svg-right-leg" x="43" y="85" width="15" height="52" rx="5"
          fill="{f('svg-right-leg')}" stroke="{stroke('svg-right-leg')}" stroke-width="0.8"
          style="filter:{glow('svg-right-leg')}">{title('svg-right-leg')}</rect>
    <!-- Feet -->
    <ellipse class="bpart" id="svg-left-foot" cx="29" cy="142" rx="10" ry="5"
             fill="{f('svg-left-foot')}" stroke="{stroke('svg-left-foot')}" stroke-width="0.8"
             style="filter:{glow('svg-left-foot')}">{title('svg-left-foot')}</ellipse>
    <ellipse class="bpart" id="svg-right-foot" cx="51" cy="142" rx="10" ry="5"
             fill="{f('svg-right-foot')}" stroke="{stroke('svg-right-foot')}" stroke-width="0.8"
             style="filter:{glow('svg-right-foot')}">{title('svg-right-foot')}</ellipse>
  </svg>
  {counter_html}
</div>
"""


def _severity_badge(severity: str) -> str:
    color, bg = _SEVERITY_COLOR.get(severity, ("#6b7280", "#f3f4f6"))
    return (
        f"<span style='background:{bg}; color:{color}; font-weight:700; "
        f"padding:4px 14px; border-radius:999px; font-size:0.9rem; "
        f"border:2px solid {color};'>{severity}</span>"
    )


def _metrics_bar(metrics: dict, t: dict) -> str:
    latency_ms   = metrics.get("latency_ms", 0)
    tok_per_sec  = metrics.get("tokens_per_sec", 0)
    total_tokens = metrics.get("total_tokens", 0)

    latency_val    = f"{latency_ms:,} ms" if latency_ms else "—"
    throughput_val = f"{tok_per_sec} {t['metrics_tokens']}/s" if tok_per_sec else "—"
    tokens_val     = f"{total_tokens:,} {t['metrics_tokens']}" if total_tokens else "—"

    def chip(icon, label, val):
        return (
            f"<div style='display:flex; align-items:center; gap:5px; "
            f"background:#0f172a; border:1px solid #1e3a5f; border-radius:6px; "
            f"padding:4px 10px;'>"
            f"<span style='color:#ED1C24; font-size:0.8rem;'>{icon}</span>"
            f"<span style='color:#94a3b8; font-size:0.68rem;'>{label}</span>"
            f"<span style='color:#e2e8f0; font-size:0.72rem; font-weight:700;'>{val}</span>"
            f"</div>"
        )

    return (
        f"<div style='display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px; align-items:center;'>"
        f"<span style='font-size:0.68rem; color:#ED1C24; font-family:monospace; "
        f"font-weight:700; letter-spacing:0.05em; margin-right:2px;'>⚡ AMD MI300X</span>"
        f"{chip('⏱', t['metrics_latency'], latency_val)}"
        f"{chip('🚀', t['metrics_throughput'], throughput_val)}"
        f"{chip('◈', t['metrics_tokens'], tokens_val)}"
        f"</div>"
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


def _empty_soap_html(lang: str) -> str:
    return _build_soap_html("", lang)


def _build_result_html(result: dict, lang: str) -> str:
    t             = _I18N.get(lang, _I18N["en"])
    triage        = result.get("triage_level", "Low")
    patient_msg   = result.get("patient_message", "")
    conditions    = result.get("possible_conditions", [])
    metrics       = result.get("_metrics", {})

    backend_tag = (
        "<span style='font-size:0.7rem; background:#052e16; color:#86efac; "
        "padding:2px 8px; border-radius:4px; margin-left:8px; "
        "border:1px solid #16a34a;'>AMD Cloud</span>"
    )

    # Triage color
    triage_colors = {
        "High":   ("#ef4444", "#7f1d1d"),
        "Medium": ("#f97316", "#431407"),
        "Low":    ("#22c55e", "#052e16"),
    }
    t_color, t_bg = triage_colors.get(triage, ("#22c55e", "#052e16"))

    # Red-flag flashing banner
    critical_banner = ""
    if triage == "High":
        critical_banner = f"""
  <div style='animation:redflash 1s ease-in-out infinite;
              background:#7f1d1d; border:2px solid #ef4444; border-radius:8px;
              padding:14px 18px; margin-bottom:16px; text-align:center;'>
    <span style='color:#fca5a5; font-weight:900; font-size:0.95rem; line-height:1.5;'>
      {t['critical_warning']}
    </span>
  </div>"""

    # Possible conditions chips
    cond_chips = "".join(
        f"<span style='background:#1e3a5f; color:#93c5fd; font-size:0.72rem; "
        f"padding:3px 10px; border-radius:999px; border:1px solid #2563eb;'>{c}</span>"
        for c in conditions
    ) if conditions else "<span style='color:#6b7280;'>—</span>"

    # Patient message paragraphs
    msg_html = "".join(
        f"<p style='margin:0 0 8px; color:#d1d5db; line-height:1.6;'>{line}</p>"
        for line in patient_msg.split("\n") if line.strip()
    ) if patient_msg else "<p style='color:#6b7280;'>—</p>"

    return f"""
<div style='background:#111827; border:1px solid #ED1C24; border-radius:12px;
            padding:20px; font-family:Arial,sans-serif; color:#f9fafb;'>

  <div style='display:flex; align-items:center; gap:10px; margin-bottom:12px;'>
    <div style='background:#ED1C24; width:4px; border-radius:2px; height:36px;'></div>
    <div>
      <div style='font-size:1.1rem; font-weight:700; color:#ED1C24;'>
        MediVision {backend_tag}
      </div>
      <div style='font-size:0.75rem; color:#6b7280;'>AMD MI300X · ROCm · Qwen2.5-VL-7B · 3-Step Pipeline</div>
    </div>
  </div>

  {_metrics_bar(metrics, t)}
  {critical_banner}

  <div style='background:#1f2937; border-radius:8px; padding:14px; margin-bottom:12px;'>
    <div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:.05em;
                color:#9ca3af; margin-bottom:6px;'>{t['severity_label']}</div>
    <span style='background:{t_bg}; color:{t_color}; font-weight:700;
                 padding:4px 16px; border-radius:999px; font-size:0.9rem;
                 border:2px solid {t_color};'>{triage}</span>
  </div>

  <div style='background:#1f2937; border-radius:8px; padding:14px; margin-bottom:12px;'>
    <div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:.05em;
                color:#9ca3af; margin-bottom:8px;'>{t['conditions_label']}</div>
    <div style='display:flex; flex-wrap:wrap; gap:6px;'>{cond_chips}</div>
  </div>

  <div style='background:#1f2937; border-radius:8px; padding:14px; margin-bottom:12px;'>
    <div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:.05em;
                color:#9ca3af; margin-bottom:8px;'>{t['actions_label']}</div>
    {msg_html}
  </div>

  <div style='background:#1a1a2e; border-left:4px solid #ED1C24; border-radius:4px;
              padding:10px 14px; font-size:0.78rem; color:#9ca3af;'>
    ⚠️ {t['disclaimer']}
  </div>

</div>
"""


def _build_soap_html(soap_text: str, lang: str = "en") -> str:
    t = _I18N.get(lang, _I18N["en"])
    if not soap_text:
        return (
            f"<div style='color:#4b5563; text-align:center; padding:40px 0; font-size:0.9rem;'>"
            f"{t['soap_empty']}</div>"
        )
    lines_html = "".join(
        f"<div style='padding:3px 0; color:{'#ED1C24' if line.startswith('S ') or line.startswith('O ') or line.startswith('A ') or line.startswith('P ') else '#d1d5db'}; "
        f"font-weight:{'700' if line[:2] in ('S ', 'O ', 'A ', 'P ') else '400'};'>{line}</div>"
        for line in soap_text.split("\n") if line.strip()
    )
    return f"""
<div style='background:#0f172a; border:1px solid #1e3a5f; border-radius:12px;
            padding:20px; font-family:monospace; font-size:0.82rem; line-height:1.7;'>
  <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;'>
    <span style='color:#ED1C24; font-weight:700; font-size:0.9rem; font-family:sans-serif;'>
      SOAP Clinical Note
    </span>
    <button onclick="navigator.clipboard.writeText(this.dataset.text).then(()=>this.textContent='{t['soap_copy_btn']} ✓').catch(()=>null)"
            data-text="{soap_text.replace(chr(34), '&quot;')}"
            style='background:#1e3a5f; color:#93c5fd; border:1px solid #2563eb; border-radius:6px;
                   padding:4px 12px; cursor:pointer; font-size:0.72rem;'>
      {t['soap_copy_btn']}
    </button>
  </div>
  {lines_html}
</div>
"""


# ---------------------------------------------------------------------------
# UI update helpers
# ---------------------------------------------------------------------------

def _error_html(t: dict, exc: Exception) -> str:
    return (
        "<div style='background:#111827; border:1px solid #ef4444; border-radius:12px; "
        "padding:24px; font-family:Arial,sans-serif; text-align:center;'>"
        "<div style='font-size:1.5rem; margin-bottom:12px;'>⚠️</div>"
        f"<div style='font-size:1rem; font-weight:700; color:#ef4444; margin-bottom:8px;'>{t['error_title']}</div>"
        f"<div style='font-size:0.85rem; color:#9ca3af; margin-bottom:16px;'>{t['error_body']}</div>"
        f"<div style='font-size:0.75rem; color:#6b7280; font-family:monospace; "
        f"background:#1f2937; padding:8px 12px; border-radius:6px;'>{exc}</div>"
        "</div>"
    )


def _ui_updates(lang_choice: str, current_regions=None):
    """Return gr.update() for the 5 translatable input-area components (no output_html)."""
    lang = _LANG_MAP.get(lang_choice, "en")
    t = _I18N[lang]
    new_choices = _localized_regions(lang)
    # Translate current selection into new language
    translated = []
    for r in (current_regions or []):
        en = _display_to_en(r)
        if en in _BODY_REGIONS:
            translated.append(new_choices[_BODY_REGIONS.index(en)])
    hint_html = (
        f"<p style='font-size:0.75rem; color:#6b7280; margin:4px 0 10px;'>{t['input_hint']}</p>"
    )
    return (
        gr.update(label=t["img_label"], choices=[t["img_mode_standard"], t["img_mode_compare"]]),
        gr.update(label=t["img_label_day1"]),
        gr.update(label=t["img_label_dayx"]),
        gr.update(label=t["symptoms_label"], placeholder=t["symptoms_placeholder"]),
        gr.update(value=t["analyze_btn"]),
        gr.update(label=t["region_optional_label"], choices=new_choices, value=translated),
        gr.update(value=hint_html),
    )


def _regions_to_prompt(selected) -> str:
    """Convert list of display names (any lang) to English prompt string."""
    if not selected:
        return ""
    if isinstance(selected, str):
        selected = [selected]
    en_regions = [_display_to_en(r) for r in selected]
    return ", ".join(r for r in en_regions if r)


def on_region_change(selected, lang_choice: str = "English"):
    """Re-render the body map SVG when selection changes (map display→EN first)."""
    if isinstance(selected, str):
        selected = [selected]
    en_keys = [_display_to_en(r) for r in (selected or [])]
    lang = _LANG_MAP.get(lang_choice, "en")
    return _body_map_svg(en_keys, lang)


def on_svg_click(svg_id: str, current_regions: list, lang_choice: str) -> tuple:
    """Toggle a body region from an SVG click. Returns (new_dropdown_value, new_svg_html)."""
    # Map svg-id → English region name
    svg_to_region = {v[0]: k for k, v in _REGION_SHAPE_MAP.items() if v}
    clicked_en = svg_to_region.get(svg_id, "")
    if not clicked_en:
        return current_regions, _body_map_svg([_display_to_en(r) for r in (current_regions or [])])

    lang = _LANG_MAP.get(lang_choice, "en")
    choices = _localized_regions(lang)
    en_list = _BODY_REGIONS
    clicked_display = choices[en_list.index(clicked_en)] if clicked_en in en_list else clicked_en

    current = list(current_regions or [])
    # Toggle: if already selected remove, else add
    en_current = [_display_to_en(r) for r in current]
    if clicked_en in en_current:
        current = [r for r in current if _display_to_en(r) != clicked_en]
    else:
        current.append(clicked_display)

    new_en = [_display_to_en(r) for r in current]
    return current, _body_map_svg(new_en, lang)


def on_lang_change(lang_choice: str, image, symptoms: str, selected_regions):
    lang = _LANG_MAP.get(lang_choice, "en")
    t = _I18N[lang]
    mode_upd, day1_upd, dayx_upd, sym_upd, btn_upd, region_upd, hint_upd = _ui_updates(
        lang_choice, current_regions=selected_regions
    )

    region = _regions_to_prompt(selected_regions)

    has_content = bool(image) or bool(symptoms and symptoms.strip())
    if has_content:
        try:
            result = get_pipeline().process(image, None, (symptoms or "").strip(), lang=lang, region=region)
            out_upd  = _build_result_html(result, lang)
            soap_upd = _build_soap_html(result.get("soap_note", ""), lang)
        except Exception as exc:
            out_upd  = _error_html(t, exc)
            soap_upd = _empty_soap_html(lang)
    else:
        out_upd  = _empty_output_html(lang)
        soap_upd = _empty_soap_html(lang)

    return mode_upd, day1_upd, dayx_upd, sym_upd, btn_upd, region_upd, hint_upd, out_upd, soap_upd, get_backend_status_html(lang)


def on_load(request: gr.Request):
    lang_display = _detect_lang_from_header(
        request.headers.get("accept-language", "")
    )
    mode_upd, day1_upd, dayx_upd, sym_upd, btn_upd, region_upd, hint_upd = _ui_updates(
        lang_display, current_regions=[]
    )
    lang = _LANG_MAP.get(lang_display, "en")
    return (
        lang_display,
        mode_upd, day1_upd, dayx_upd,
        sym_upd, btn_upd, region_upd, hint_upd,
        _body_map_svg([], lang),
        _empty_output_html(lang),
        _empty_soap_html(lang),
        get_backend_status_html(lang),
    )


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict(image_1, image_2, symptoms: str, lang_choice: str, selected_regions):
    lang = _LANG_MAP.get(lang_choice, "en")
    t = _I18N[lang]

    if not image_1 and not image_2 and not (symptoms or "").strip():
        return _empty_output_html(lang), _empty_soap_html(lang), get_backend_status_html(lang)

    region = _regions_to_prompt(selected_regions)

    try:
        result = get_pipeline().process(
            image_1, image_2, (symptoms or "").strip(), lang=lang, region=region
        )
        return (
            _build_result_html(result, lang),
            _build_soap_html(result.get("soap_note", ""), lang),
            get_backend_status_html(lang),
        )
    except Exception as exc:
        return _error_html(t, exc), _empty_soap_html(lang), get_backend_status_html(lang)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');

body, .gradio-container {
    background-color: #030712 !important;
    color: #f9fafb !important;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
}
input:focus, textarea:focus {
    border-color: #ED1C24 !important;
    box-shadow: 0 0 0 2px rgba(237,28,36,0.25) !important;
}

/* ── CTA button — full-width AMD red with glow ── */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, #ED1C24 0%, #b01318 100%) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    padding: 14px 0 !important;
    border-radius: 8px !important;
    box-shadow: 0 0 16px rgba(237,28,36,0.45), 0 4px 12px rgba(0,0,0,0.4) !important;
    transition: box-shadow 0.25s ease, opacity 0.2s !important;
}
button.primary:hover {
    opacity: 0.92 !important;
    box-shadow: 0 0 28px rgba(237,28,36,0.65), 0 6px 16px rgba(0,0,0,0.5) !important;
}
button.primary:active { opacity: 0.8 !important; transform: scale(0.99) !important; }

.gr-box, .gr-panel {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 10px !important;
}
label span, .gr-form > label {
    color: #cbd5e1 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
footer { display: none !important; }
@keyframes redflash {
  0%, 100% { opacity: 1; box-shadow: 0 0 12px rgba(239,68,68,0.6); }
  50%       { opacity: 0.7; box-shadow: 0 0 24px rgba(239,68,68,0.9); }
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }

/* ── Topbar ── */
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

/* ── Mobile: hide drag-and-drop text ── */
@media (pointer: coarse), (max-width: 768px) {
    .upload-container [data-testid="drop-zone"] .upload-text:first-child,
    .svelte-upload .file-preview-title,
    span.drag-text, .drag-drop-label { display: none !important; }
    .upload-container { min-height: 120px !important; }
    /* Prioritise camera/file button visual */
    .upload-container button { font-size: 0.9rem !important; padding: 10px 20px !important; }
}

/* ── Quick Examples — horizontal scroll on mobile ── */
@media (max-width: 768px) {
    .gr-samples table, table.gr-samples-table {
        display: flex !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
        gap: 8px !important;
        padding-bottom: 6px !important;
    }
    .gr-samples tr, .gr-samples-table tr {
        display: inline-flex !important;
        flex-direction: column !important;
        min-width: 220px !important;
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 8px !important;
        padding: 8px 10px !important;
        white-space: normal !important;
    }
    .gr-samples thead, .gr-samples-table thead { display: none !important; }
}


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
    Multilingual Dermatology &amp; Wound Care AI Assistant
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

BLOCKS_JS = """
() => {
  /* ── SVG body-map click → hidden bridge input ── */
  document.addEventListener('click', function(e) {
    var el = e.target.closest('.bpart');
    if (!el) return;
    var svgId = el.id;
    if (!svgId) return;
    var bridge = document.getElementById('svg-click-bridge');
    if (!bridge) return;
    var input = bridge.querySelector('input, textarea');
    if (!input) return;
    var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set
              || Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(input, svgId);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });

  /* ── Tab label i18n via MutationObserver on lang dropdown ── */
  var TAB_LABELS = {
    'English':    ['Patient View', 'Export for Doctor (SOAP)'],
    'Ti\\u1ebfng Vi\\u1ec7t': ['D\\u00e0nh cho b\\u1ec7nh nh\\u00e2n', 'Xu\\u1ea5t cho b\\u00e1c s\\u0129 (SOAP)'],
    '\\u4e2d\\u6587': ['\\u60a3\\u8005\\u89c6\\u56fe', '\\u5bfc\\u51fa\\u7ed9\\u533b\\u751f\\uff08SOAP\\uff09'],
    'Espa\\u00f1ol': ['Vista del paciente', 'Exportar para m\\u00e9dico (SOAP)'],
    'Fran\\u00e7ais': ['Vue patient', 'Exporter pour le m\\u00e9decin (SOAP)'],
    '\\u65e5\\u672c\\u8a9e': ['\\u60a3\\u8005\\u5411\\u3051', '\\u533b\\u5e2b\\u5411\\u3051\\u30a8\\u30af\\u30b9\\u30dd\\u30fc\\u30c8\\uff08SOAP\\uff09'],
  };

  function updateTabLabels(langDisplay) {
    var labels = TAB_LABELS[langDisplay] || TAB_LABELS['English'];
    var tabContainer = document.getElementById('output-tabs');
    if (!tabContainer) return;
    var buttons = tabContainer.querySelectorAll('button');
    var tabButtons = Array.from(buttons).filter(function(b) {
      return b.closest('#output-tabs') === tabContainer || b.parentElement.closest('#output-tabs') === tabContainer;
    }).slice(0, 2);
    if (tabButtons.length < 2) tabButtons = Array.from(tabContainer.querySelectorAll('button')).slice(0, 2);
    tabButtons.forEach(function(btn, i) {
      if (labels[i] === undefined) return;
      btn.childNodes.forEach(function(n) {
        if (n.nodeType === 3 && n.nodeValue.trim()) n.nodeValue = labels[i];
      });
      /* fallback: if no direct text node, set innerText carefully */
      if (!btn.childNodes.length || !Array.from(btn.childNodes).some(function(n){ return n.nodeType === 3 && n.nodeValue.trim(); })) {
        btn.textContent = labels[i];
      }
    });
  }

  function watchLangDropdown() {
    var langCol = document.getElementById('lang-col');
    if (!langCol) { setTimeout(watchLangDropdown, 500); return; }
    var observer = new MutationObserver(function() {
      var inp = langCol.querySelector('input[type="text"], .wrap-inner span');
      var val = inp ? (inp.value || inp.textContent || '').trim() : '';
      if (val) updateTabLabels(val);
    });
    observer.observe(langCol, { childList: true, subtree: true, characterData: true });
  }
  watchLangDropdown();
}
"""

with gr.Blocks(css=CSS, js=BLOCKS_JS, theme=gr.themes.Base(), title="MediVision — Dermatology & Wound Care AI") as demo:

    gr.HTML(HEADER_HTML)

    # ── Topbar ───────────────────────────────────────────────────────────────
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

    # Hidden bridge: SVG clicks write svg-element-id here → triggers on_svg_click
    svg_click_bridge = gr.Textbox(
        value="",
        visible=False,
        elem_id="svg-click-bridge",
    )

    # ── Main content ──────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=300):
            img_mode = gr.Radio(
                choices=[_I18N["en"]["img_mode_standard"], _I18N["en"]["img_mode_compare"]],
                value=_I18N["en"]["img_mode_standard"],
                label=_I18N["en"]["img_mode_label"],
                elem_id="img-mode-radio",
            )
            with gr.Row(equal_height=True):
                input_img = gr.Image(
                    type="filepath",
                    label=_I18N["en"]["img_label_day1"],
                    height=200,
                )
                input_img_2 = gr.Image(
                    type="filepath",
                    label=_I18N["en"]["img_label_dayx"],
                    height=200,
                    visible=False,
                )
            symptoms_txt = gr.Textbox(
                label="Symptoms Description",
                placeholder="Describe what you feel — e.g. itchy red patch for 3 days...",
                lines=4,
            )

            input_hint_html = gr.HTML(
                value="<p style='font-size:0.75rem; color:#6b7280; margin:4px 0 10px;'>"
                      f"{_I18N['en']['input_hint']}</p>",
                elem_id="input-hint",
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=0, min_width=80):
                    body_map_html = gr.HTML(value=_body_map_svg([]))
                with gr.Column(scale=1):
                    region_selector = gr.Dropdown(
                        choices=_BODY_REGIONS,
                        value=[],
                        multiselect=True,
                        label="Affected Body Region",
                        container=True,
                    )

            submit_btn = gr.Button("🔬  Analyze", variant="primary", size="lg", elem_id="analyze-btn")

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
            with gr.Tabs(elem_id="output-tabs"):
                with gr.TabItem(_I18N["en"]["tab_patient"], elem_id="tab-patient"):
                    output_html = gr.HTML(value=_empty_output_html("en"))
                with gr.TabItem(_I18N["en"]["tab_doctor"], elem_id="tab-doctor"):
                    soap_html = gr.HTML(value=_empty_soap_html("en"))

    # ── Events ───────────────────────────────────────────────────────────────

    # Image mode toggle: show/hide second image upload
    img_mode.change(
        fn=lambda m: gr.update(visible=_I18N["en"]["img_mode_compare"] in m),
        inputs=[img_mode],
        outputs=[input_img_2],
    )

    # SVG click → toggle region in dropdown + re-render SVG
    svg_click_bridge.input(
        fn=on_svg_click,
        inputs=[svg_click_bridge, region_selector, lang_radio],
        outputs=[region_selector, body_map_html],
    )

    # Dropdown change → re-render SVG (keeps sync when user edits dropdown directly)
    region_selector.change(
        fn=on_region_change,
        inputs=[region_selector, lang_radio],
        outputs=[body_map_html],
    )

    lang_radio.change(
        fn=on_lang_change,
        inputs=[lang_radio, input_img, symptoms_txt, region_selector],
        outputs=[img_mode, input_img, input_img_2, symptoms_txt, submit_btn,
                 region_selector, input_hint_html, output_html, soap_html, status_bar],
    )

    submit_btn.click(
        fn=predict,
        inputs=[input_img, input_img_2, symptoms_txt, lang_radio, region_selector],
        outputs=[output_html, soap_html, status_bar],
        api_name="analyze",
    )

    demo.load(
        fn=on_load,
        inputs=[],
        outputs=[
            lang_radio,
            img_mode, input_img, input_img_2,
            symptoms_txt, submit_btn, region_selector, input_hint_html,
            body_map_html, output_html, soap_html, status_bar,
        ],
    )

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
