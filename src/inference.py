from src.agent import analyze_image_and_text


class MediVisionPipeline:
    def process(self, image_path, symptoms: str, lang: str = "en", region: str = "") -> dict:
        """
        Run the full analysis pipeline.
        Raises RuntimeError if the AMD Cloud backend is unreachable.

        Returns:
            dict with keys: diagnosis, severity, recommended_actions, confidence_score, _metrics
        """
        return analyze_image_and_text(
            image_path=image_path,
            text_description=symptoms,
            language=lang,
            region=region,
        )
