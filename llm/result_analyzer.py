"""
LLM-powered backtest result analysis.

Produces natural-language interpretation of backtest metrics,
identifies strengths/weaknesses, and suggests improvements.

Usage::

    from llm.result_analyzer import ResultAnalyzer
    analyzer = ResultAnalyzer()
    analysis = analyzer.analyze(result.summary())
"""

__all__ = ["ResultAnalyzer"]

from loguru import logger

from llm.base import BaseLLM


class ResultAnalyzer:
    """Analyses backtest results using LLM.

    Attributes:
        llm: Underlying LLM client.
    """

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self.llm = llm or BaseLLM()
        logger.info("ResultAnalyzer initialised")

    def analyze(self, metrics: dict[str, float], context: str = "") -> str:
        """Analyse backtest metrics and provide insights.

        Args:
            metrics: Dict of metric name to value (from BacktestResult.summary()).
            context: Optional additional context about the strategy.

        Returns:
            Natural-language analysis.
        """
        metrics_text = "\n".join(f"  - {k}: {v:.4f}" for k, v in metrics.items())
        prompt = (
            f"Analyse the following backtest results and provide insights:\n\n"
            f"Metrics:\n{metrics_text}\n\n"
        )
        if context:
            prompt += f"Strategy Context:\n{context}\n\n"

        prompt += (
            "Provide:\n"
            "1. Overall assessment (1-2 sentences)\n"
            "2. Key strengths\n"
            "3. Key weaknesses\n"
            "4. Specific improvement suggestions\n"
            "5. Risk warnings\n"
        )

        system = (
            "You are a quantitative analyst reviewing backtest results. "
            "Be specific, data-driven, and actionable. Reference the actual "
            "numbers in your analysis."
        )

        response = self.llm.generate(prompt, system=system)
        logger.info("Generated result analysis ({} chars)", len(response))
        return response
