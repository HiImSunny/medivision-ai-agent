from src.agent import analyze_image_and_text


class MediVisionPipeline:
    def process(self, image_path, symptoms: str, lang: str = "en") -> dict:
        """
        Run the full analysis pipeline.

        Returns:
            dict with keys: diagnosis, severity, recommended_actions, confidence_score
        """
        if not image_path and not symptoms.strip():
            placeholder = (
                "Please upload an image or describe your symptoms."
                if lang == "en"
                else "Vui lòng tải lên hình ảnh hoặc mô tả triệu chứng của bạn."
            )
            return {
                "diagnosis": placeholder,
                "severity": "Low",
                "recommended_actions": [],
                "confidence_score": 0,
            }

        return analyze_image_and_text(
            image_path=image_path,
            text_description=symptoms,
            language=lang,
        )
