"""Versioned corrections to validation comparisons.

Historical validation reports are not rewritten. A correction changes how a new
validation is judged and can record which stored summaries are candidates for a
later validation-plane replay. Research agents do not receive holdout rows.
"""

from __future__ import annotations

import json

from enigma.schemas.validation import RedactedValidationSummary
from enigma.store import Stores

G02_EQUALITY_CORRECTION = "g02-embargo-equality-strict"
_FLAG = "correction.g02-embargo-equality"


def research_dataset_overlaps_embargo(time_end_ns: int, embargo_start_ns: int) -> bool:
    """True when a research dataset extends past the embargo start.

    Equality is not an overlap. The catalog keeps rows with available_at <= end, and
    the embargo start is itself the last research instant. A dataset that ends exactly
    there does not contain a holdout bar. An experiment window that ends after the
    embargo start is still an overlap; that comparison is unchanged.
    """
    return time_end_ns > embargo_start_ns


def note_g02_equality_candidates(stores: Stores) -> dict[str, object]:
    """List stored summaries that failed G02, without opening a holdout or replacing a report.

    The list is an audit of candidates. It is not a new validation outcome. Recomputing
    those outcomes requires the validation plane and the original packages, and it must
    be stored as a new report id.
    """
    candidates: list[str] = []
    for summary in stores.documents.list(RedactedValidationSummary, limit=10_000):
        failed = set(summary.failed_gate_names)
        if "G02" in failed:
            candidates.append(summary.id)
    payload: dict[str, object] = {
        "correction": G02_EQUALITY_CORRECTION,
        "candidate_summary_ids": candidates,
        "rewritten": False,
        "holdout_rows_read": False,
    }
    stores.flags.set(_FLAG, json.dumps(payload, sort_keys=True), "system")
    stores.audit.record(
        "validation_service",
        "validation.correction_noted",
        G02_EQUALITY_CORRECTION,
        {"candidates": len(candidates), "rewritten": False},
    )
    return payload
