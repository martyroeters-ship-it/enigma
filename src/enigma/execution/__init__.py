"""Execution plane. Paper-only when introduced; no live trading (ADR-010)."""

from enigma.execution.degradation import DegradationDetector
from enigma.execution.paper import PaperBroker, PaperQuote

__all__ = ["DegradationDetector", "PaperBroker", "PaperQuote"]
