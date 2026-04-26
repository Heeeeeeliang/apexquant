"""
Base LLM client with provider abstraction.

Supports Anthropic (Claude) and OpenAI backends via a unified
``generate`` interface.

Usage::

    from llm.base import BaseLLM
    llm = BaseLLM(provider="anthropic", model="claude-sonnet-4-20250514")
    response = llm.generate("Explain momentum trading.")
"""

__all__ = ["BaseLLM"]

from typing import Any

from loguru import logger

from config import get


class BaseLLM:
    """Unified LLM client supporting Anthropic and OpenAI.

    Attributes:
        provider: ``"anthropic"`` or ``"openai"``.
        model: Model identifier string.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.provider = provider or get("llm.provider", "anthropic")
        self.model = model or get("llm.model", "claude-sonnet-4-20250514")
        self.temperature = temperature or get("llm.temperature", 0.3)
        self.max_tokens = max_tokens or get("llm.max_tokens", 4096)
        self._client: Any = None
        logger.info("BaseLLM: provider={}, model={}", self.provider, self.model)

    def _get_client(self) -> Any:
        """Lazily initialise the API client.

        Returns:
            The provider-specific client instance.

        Raises:
            ValueError: If provider is not supported.
        """
        if self._client is not None:
            return self._client

        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic()
        elif self.provider == "openai":
            import openai

            self._client = openai.OpenAI()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        logger.info("Initialised {} client", self.provider)
        return self._client

    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a text response from the LLM.

        Args:
            prompt: User prompt.
            system: Optional system prompt.

        Returns:
            The model's text response.
        """
        client = self._get_client()

        if self.provider == "anthropic":
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system if system else "You are a quantitative trading expert.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        elif self.provider == "openai":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=messages,
            )
            text = response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        logger.info("LLM response: {} chars", len(text))
        return text
