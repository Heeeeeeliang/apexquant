"""
LLM-powered strategy generation.

Uses Claude/OpenAI to generate trading strategy ideas based on
market conditions, predictor outputs, and backtest results.

Usage::

    from llm.strategy_generator import StrategyGenerator
    gen = StrategyGenerator()
    idea = gen.generate_strategy(
        market_context="High volatility, downtrend",
        predictor_summary={"volatility": 0.85, "turning_point": 0.3},
    )
"""

__all__ = ["StrategyGenerator"]

from loguru import logger

from llm.base import BaseLLM


class StrategyGenerator:
    """Generates trading strategy proposals via LLM.

    Attributes:
        llm: Underlying LLM client.
    """

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self.llm = llm or BaseLLM()
        logger.info("StrategyGenerator initialised")

    def generate_strategy(
        self,
        market_context: str,
        predictor_summary: dict[str, float],
        constraints: str = "",
    ) -> str:
        """Generate a strategy proposal based on current conditions.

        Args:
            market_context: Description of current market regime.
            predictor_summary: Dict of predictor name to latest signal value.
            constraints: Optional risk or style constraints.

        Returns:
            Strategy proposal as formatted text.
        """
        signals_text = "\n".join(
            f"  - {k}: {v:.3f}" for k, v in predictor_summary.items()
        )
        prompt = (
            f"Given the following market context and predictor signals, "
            f"propose a trading strategy.\n\n"
            f"Market Context:\n{market_context}\n\n"
            f"Predictor Signals:\n{signals_text}\n\n"
        )
        if constraints:
            prompt += f"Constraints:\n{constraints}\n\n"

        prompt += (
            "Respond with:\n"
            "1. Strategy name\n"
            "2. Entry conditions\n"
            "3. Exit conditions\n"
            "4. Position sizing rule\n"
            "5. Risk management\n"
        )

        system = (
            "You are a senior quantitative strategist. Propose concrete, "
            "implementable strategies based on the three-layer signal "
            "framework (volatility, turning point, meta-label)."
        )

        response = self.llm.generate(prompt, system=system)
        logger.info("Generated strategy proposal ({} chars)", len(response))
        return response
