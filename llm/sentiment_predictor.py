"""
LLM-powered market sentiment prediction.

Uses LLMs to extract sentiment signals from news headlines or
market commentary and convert them to numerical scores.

Usage::

    from llm.sentiment_predictor import SentimentPredictor
    sp = SentimentPredictor()
    score = sp.predict("Fed raises rates by 25bps, signals more hikes")
    print(score)  # e.g. -0.6
"""

__all__ = ["SentimentPredictor"]

import json

from loguru import logger

from llm.base import BaseLLM


class SentimentPredictor:
    """Extracts sentiment scores from text via LLM.

    Attributes:
        llm: Underlying LLM client.
    """

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self.llm = llm or BaseLLM()
        logger.info("SentimentPredictor initialised")

    def predict(self, text: str) -> float:
        """Predict sentiment score for a piece of text.

        Args:
            text: News headline, market commentary, or earnings summary.

        Returns:
            Sentiment score in [-1.0, 1.0] where -1 is most bearish
            and +1 is most bullish.
        """
        prompt = (
            f"Rate the market sentiment of the following text on a scale "
            f"from -1.0 (extremely bearish) to +1.0 (extremely bullish).\n\n"
            f'Text: "{text}"\n\n'
            f"Respond with ONLY a JSON object: {{\"score\": <float>, \"reasoning\": \"<brief>\"}}"
        )

        system = (
            "You are a financial sentiment analysis model. Respond only "
            "with valid JSON containing a 'score' field (float between -1 and 1) "
            "and a 'reasoning' field (brief explanation)."
        )

        response = self.llm.generate(prompt, system=system)

        try:
            parsed = json.loads(response)
            score = float(parsed["score"])
            score = max(-1.0, min(1.0, score))
            logger.info(
                "Sentiment: {:.2f} — {}",
                score,
                parsed.get("reasoning", "N/A"),
            )
            return score
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse sentiment response: {} — {}", exc, response[:100])
            return 0.0

    def predict_batch(self, texts: list[str]) -> list[float]:
        """Predict sentiment for multiple texts.

        Args:
            texts: List of text strings.

        Returns:
            List of sentiment scores.
        """
        scores = [self.predict(t) for t in texts]
        logger.info("Batch sentiment: {} texts, mean={:.3f}", len(scores), sum(scores) / max(len(scores), 1))
        return scores
