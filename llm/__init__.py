"""
ApexQuant LLM Module
=====================

Provides AI-powered strategy generation, result analysis, and
sentiment prediction using Claude or OpenAI models.

Usage::

    from llm import StrategyGenerator, ResultAnalyzer, SentimentPredictor
"""

__all__ = ["BaseLLM", "StrategyGenerator", "ResultAnalyzer", "SentimentPredictor"]

from llm.base import BaseLLM
from llm.strategy_generator import StrategyGenerator
from llm.result_analyzer import ResultAnalyzer
from llm.sentiment_predictor import SentimentPredictor
