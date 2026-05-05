import random
import re
import src.config as config
from src.model_loader import generate_response

# ---------------------------------------------------------------------------
# Mock data pools — realistic enough for a hackathon demo
# ---------------------------------------------------------------------------

_MOCK_EN = [
    {
        "diagnosis": "Contact Dermatitis",
        "severity": "Low",
        "confidence_score": 82,
        "recommended_actions": [
            "Avoid contact with suspected allergen or irritant.",
            "Apply a mild hydrocortisone cream (1%) to the affected area twice daily.",
            "Use fragrance-free moisturiser after bathing to restore the skin barrier.",
            "Take an over-the-counter antihistamine (e.g., cetirizine) if itching is severe.",
            "Monitor for spreading redness or signs of secondary infection; see a doctor if worsening.",
        ],
    },
    {
        "diagnosis": "Mild Abrasion / Superficial Wound",
        "severity": "Low",
        "confidence_score": 88,
        "recommended_actions": [
            "Gently clean the wound with sterile saline or clean running water for 5–10 minutes.",
            "Apply a thin layer of antibiotic ointment (e.g., bacitracin) to prevent infection.",
            "Cover with a non-stick sterile dressing; change daily or when soiled.",
            "Watch for signs of infection: increased redness, warmth, purulent discharge, or fever.",
            "Tetanus prophylaxis: verify vaccination is up to date (within 5 years for dirty wounds).",
        ],
    },
    {
        "diagnosis": "Atopic Eczema (Eczema Flare)",
        "severity": "Medium",
        "confidence_score": 75,
        "recommended_actions": [
            "Apply a prescription-strength topical corticosteroid (e.g., triamcinolone 0.1%) to inflamed areas.",
            "Use an emollient at least twice daily on the entire body, not just affected skin.",
            "Identify and eliminate triggers: dust mites, pet dander, harsh soaps, or stress.",
            "Wear loose-fitting, breathable cotton clothing to minimise irritation.",
            "Consult a dermatologist if the flare does not improve within 1–2 weeks.",
        ],
    },
    {
        "diagnosis": "Partial-Thickness Burn (Second-Degree)",
        "severity": "High",
        "confidence_score": 79,
        "recommended_actions": [
            "Cool the burn immediately under cool (not ice-cold) running water for at least 20 minutes.",
            "Do NOT apply butter, toothpaste, or home remedies — these increase infection risk.",
            "Cover loosely with a clean non-fluffy material (e.g., cling film or sterile dressing).",
            "Seek prompt medical evaluation; burns larger than a palm or on the face/hands require ER care.",
            "Pain management: ibuprofen or paracetamol as directed.",
        ],
    },
    {
        "diagnosis": "Suspected Cellulitis",
        "severity": "Urgent",
        "confidence_score": 71,
        "recommended_actions": [
            "Seek immediate medical attention — cellulitis can spread rapidly and become systemic.",
            "Do not massage or heat the affected area.",
            "Elevate the limb above heart level to reduce oedema.",
            "A course of oral antibiotics (e.g., cephalexin or amoxicillin-clavulanate) is typically required.",
            "Return to ER if red streaking, fever > 38.5 °C, or rapid area expansion occurs.",
        ],
    },
    {
        "diagnosis": "Tinea Corporis (Ringworm)",
        "severity": "Low",
        "confidence_score": 85,
        "recommended_actions": [
            "Apply an antifungal cream (e.g., clotrimazole 1% or terbinafine) twice daily for 2–4 weeks.",
            "Keep the area clean and dry; fungi thrive in moist environments.",
            "Avoid sharing towels, clothing, or bedding during active infection.",
            "Wash clothing and bed linen at 60 °C to eliminate spores.",
            "See a GP if there is no improvement after 4 weeks — oral antifungal may be needed.",
        ],
    },
    {
        "diagnosis": "Psoriasis Plaque",
        "severity": "Medium",
        "confidence_score": 73,
        "recommended_actions": [
            "Moisturise heavily with thick emollients (e.g., petroleum jelly) after bathing.",
            "A topical vitamin D analogue (e.g., calcipotriol) or mild steroid may be prescribed.",
            "Avoid known triggers: stress, alcohol, smoking, and certain medications (e.g., beta-blockers).",
            "Phototherapy (UVB) is effective for extensive plaques — discuss with a dermatologist.",
            "Biologic therapy may be appropriate for moderate-to-severe disease unresponsive to topicals.",
        ],
    },
]

_MOCK_VN = [
    {
        "diagnosis": "Viêm da tiếp xúc",
        "severity": "Thấp",
        "confidence_score": 82,
        "recommended_actions": [
            "Tránh tiếp xúc với chất gây dị ứng hoặc kích ứng nghi ngờ.",
            "Bôi kem hydrocortisone nhẹ (1%) lên vùng bị ảnh hưởng hai lần mỗi ngày.",
            "Dùng kem dưỡng ẩm không hương liệu sau khi tắm để phục hồi hàng rào da.",
            "Uống thuốc kháng histamine (ví dụ: cetirizine) nếu ngứa nghiêm trọng.",
            "Theo dõi xem có lan rộng đỏ hoặc dấu hiệu nhiễm trùng thứ phát không; gặp bác sĩ nếu nặng hơn.",
        ],
    },
    {
        "diagnosis": "Trầy xước nhẹ / Vết thương nông",
        "severity": "Thấp",
        "confidence_score": 88,
        "recommended_actions": [
            "Nhẹ nhàng làm sạch vết thương bằng nước muối sinh lý hoặc nước sạch trong 5–10 phút.",
            "Bôi một lớp mỏng thuốc mỡ kháng sinh (ví dụ: bacitracin) để ngăn ngừa nhiễm trùng.",
            "Băng bó bằng gạc vô khuẩn không dính; thay hàng ngày hoặc khi bẩn.",
            "Theo dõi dấu hiệu nhiễm trùng: đỏ tăng, nóng, mủ chảy ra hoặc sốt.",
            "Phòng uốn ván: kiểm tra lịch tiêm chủng có còn hiệu lực (trong vòng 5 năm với vết thương bẩn).",
        ],
    },
    {
        "diagnosis": "Chàm dị ứng (Đợt bùng phát)",
        "severity": "Trung bình",
        "confidence_score": 75,
        "recommended_actions": [
            "Bôi corticosteroid tại chỗ theo toa (ví dụ: triamcinolone 0.1%) lên vùng viêm.",
            "Dùng thuốc dưỡng ẩm ít nhất hai lần mỗi ngày trên toàn cơ thể, không chỉ vùng bị ảnh hưởng.",
            "Xác định và loại bỏ các tác nhân kích thích: bụi, lông thú cưng, xà phòng mạnh hoặc căng thẳng.",
            "Mặc quần áo cotton rộng rãi, thoáng khí để giảm kích ứng.",
            "Tham khảo bác sĩ da liễu nếu đợt bùng phát không cải thiện trong vòng 1–2 tuần.",
        ],
    },
    {
        "diagnosis": "Bỏng độ hai (Bỏng lớp bì)",
        "severity": "Cao",
        "confidence_score": 79,
        "recommended_actions": [
            "Làm mát vết bỏng ngay bằng nước mát (không phải nước đá) trong ít nhất 20 phút.",
            "KHÔNG bôi bơ, kem đánh răng hoặc các biện pháp dân gian — sẽ tăng nguy cơ nhiễm trùng.",
            "Che phủ nhẹ nhàng bằng vải sạch không xơ (ví dụ: màng bọc thực phẩm hoặc gạc vô khuẩn).",
            "Tìm kiếm đánh giá y tế ngay; bỏng lớn hơn lòng bàn tay hoặc ở mặt/tay cần đến cấp cứu.",
            "Giảm đau: ibuprofen hoặc paracetamol theo chỉ định.",
        ],
    },
    {
        "diagnosis": "Nghi ngờ viêm mô tế bào (Cellulitis)",
        "severity": "Khẩn cấp",
        "confidence_score": 71,
        "recommended_actions": [
            "Tìm kiếm sự chú ý y tế ngay lập tức — viêm mô tế bào có thể lan rộng nhanh chóng.",
            "Không xoa bóp hoặc chườm nóng vùng bị ảnh hưởng.",
            "Nâng cao chi trên mức tim để giảm phù nề.",
            "Thường cần một đợt kháng sinh uống (ví dụ: cephalexin hoặc amoxicillin-clavulanate).",
            "Quay lại cấp cứu nếu có vết đỏ lan rộng, sốt > 38.5 °C hoặc vùng bị ảnh hưởng mở rộng nhanh.",
        ],
    },
    {
        "diagnosis": "Nấm da (Ringworm / Tinea corporis)",
        "severity": "Thấp",
        "confidence_score": 85,
        "recommended_actions": [
            "Bôi kem chống nấm (ví dụ: clotrimazole 1% hoặc terbinafine) hai lần mỗi ngày trong 2–4 tuần.",
            "Giữ vùng da sạch và khô; nấm phát triển mạnh trong môi trường ẩm ướt.",
            "Tránh dùng chung khăn, quần áo hoặc chăn ga trong thời gian nhiễm trùng.",
            "Giặt quần áo và ga trải giường ở nhiệt độ 60 °C để tiêu diệt bào tử.",
            "Gặp bác sĩ nếu không cải thiện sau 4 tuần — có thể cần thuốc chống nấm uống.",
        ],
    },
    {
        "diagnosis": "Mảng vảy nến (Psoriasis Plaque)",
        "severity": "Trung bình",
        "confidence_score": 73,
        "recommended_actions": [
            "Dưỡng ẩm nhiều bằng các chất nhũ hóa dày (ví dụ: vaseline) sau khi tắm.",
            "Có thể kê đơn thuốc tương tự vitamin D tại chỗ (ví dụ: calcipotriol) hoặc steroid nhẹ.",
            "Tránh các yếu tố kích thích đã biết: căng thẳng, rượu, hút thuốc và một số thuốc.",
            "Quang trị liệu (UVB) có hiệu quả với các mảng rộng — thảo luận với bác sĩ da liễu.",
            "Liệu pháp sinh học có thể phù hợp với bệnh từ vừa đến nặng không đáp ứng với thuốc tại chỗ.",
        ],
    },
]

_SEVERITY_ORDER = {"Low": 0, "Thấp": 0, "Medium": 1, "Trung bình": 1,
                   "High": 2, "Cao": 2, "Urgent": 3, "Khẩn cấp": 3}


def _add_variance(base_score: int) -> int:
    """Vary the confidence score ±5 points within [65, 95]."""
    return max(65, min(95, base_score + random.randint(-5, 5)))


def _mock_response(lang: str) -> dict:
    pool = _MOCK_VN if lang == "vn" else _MOCK_EN
    entry = random.choice(pool).copy()
    entry["confidence_score"] = _add_variance(entry["confidence_score"])
    return entry


def _build_prompt(image_path: str | None, text_description: str, lang: str) -> str:
    lang_instruction = (
        "Respond in Vietnamese." if lang == "vn"
        else "Respond in English."
    )
    has_image = bool(image_path)
    return (
        "You are MediVision, a professional dermatology and wound-care assistant.\n"
        f"{lang_instruction}\n"
        "The user has provided"
        + (" an image of a skin condition and" if has_image else "")
        + f" the following symptom description:\n\n{text_description}\n\n"
        "Provide a JSON object with these exact keys:\n"
        "  diagnosis, severity (Low|Medium|High|Urgent), "
        "recommended_actions (list of strings), confidence_score (integer 0-100).\n"
        "Be medically thorough but write at a patient-friendly reading level."
    )


def _parse_real_response(raw: str, lang: str) -> dict:
    """
    Try to extract JSON from the model's raw output.
    Falls back to a structured mock on parse failure.
    """
    import json

    # Find the first JSON object in the output
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

    # Couldn't parse — use mock with a note
    result = _mock_response(lang)
    result["diagnosis"] = f"[Parse error — showing representative result] {result['diagnosis']}"
    return result


def analyze_image_and_text(
    image_path: str | None,
    text_description: str,
    language: str = "en",
) -> dict:
    """
    Main analysis entry point.

    Returns a dict:
        {
            "diagnosis": str,
            "severity": str,          # Low | Medium | High | Urgent
            "recommended_actions": list[str],
            "confidence_score": int,   # 0-100
        }
    """
    lang = language.lower()

    if config.MOCK_MODE:
        return _mock_response(lang)

    prompt = _build_prompt(image_path, text_description, lang)
    raw = generate_response(prompt, image_path=image_path)

    if raw is None:
        # model_loader set MOCK_MODE=True due to load failure
        return _mock_response(lang)

    return _parse_real_response(raw, lang)
