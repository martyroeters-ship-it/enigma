"""Research plane: hypotheses, features, experiments, trial ledger."""

from enigma.research.checks import check_proposal, fingerprint
from enigma.research.context import build_research_context
from enigma.research.deterministic import DeterministicHypothesisSource
from enigma.research.generator import HypothesisGenerator
from enigma.research.observations import compute_observations
from enigma.research.registry import HypothesisRegistry

__all__ = [
    "DeterministicHypothesisSource",
    "HypothesisGenerator",
    "HypothesisRegistry",
    "build_research_context",
    "check_proposal",
    "compute_observations",
    "fingerprint",
]
