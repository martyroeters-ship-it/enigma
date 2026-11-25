"""Strategy packages: the v1 bar-strategy interface, package building, and the sandbox runner.

LLM codegen (`enigma.strategy.codegen`) is intentionally not re-exported here so importing
sandbox/leakage modules (used by validation) never loads the model gateway.
"""

from enigma.strategy.data import symbol_arrays
from enigma.strategy.interface import INTERFACE_VERSION, NAT_INT64, Strategy, StrategyMeta
from enigma.strategy.leakage import (
    TruncationCheckResult,
    TruncationMismatch,
    evaluate_truncations,
    pit_timestamp_violations,
    sample_checkpoints,
    truncation_invariance_check,
)
from enigma.strategy.package import build_package, package_content_hash, verify_package
from enigma.strategy.sandbox import (
    ALLOW_WEAK_ENV,
    SandboxExecutionError,
    SandboxJob,
    SandboxLimits,
    SandboxResult,
    SandboxRunner,
    SandboxUnavailableError,
)
from enigma.strategy.spec import check_consistency, derive_spec
from enigma.strategy.static_check import (
    ALLOWED_IMPORTS,
    StaticViolation,
    StrategyRejectedError,
    check_source,
)

__all__ = [
    "ALLOWED_IMPORTS",
    "ALLOW_WEAK_ENV",
    "INTERFACE_VERSION",
    "NAT_INT64",
    "SandboxExecutionError",
    "SandboxJob",
    "SandboxLimits",
    "SandboxResult",
    "SandboxRunner",
    "SandboxUnavailableError",
    "StaticViolation",
    "Strategy",
    "StrategyMeta",
    "StrategyRejectedError",
    "TruncationCheckResult",
    "TruncationMismatch",
    "build_package",
    "check_consistency",
    "check_source",
    "derive_spec",
    "evaluate_truncations",
    "package_content_hash",
    "pit_timestamp_violations",
    "sample_checkpoints",
    "symbol_arrays",
    "truncation_invariance_check",
    "verify_package",
]
