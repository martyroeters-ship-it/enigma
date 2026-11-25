"""Canonical forms of external identifiers (DOI, arXiv, OpenAlex, ...) for deduplication."""

from __future__ import annotations

import re
from collections.abc import Mapping

_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)
_ARXIV_NEW_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
_ARXIV_OLD_RE = re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.IGNORECASE)
_OPENALEX_RE = re.compile(r"\b([WAISCV]\d{4,})\b")

STRONG_ID_KEYS = ("doi", "arxiv", "openalex", "github", "wikipedia", "s2")


def normalize_doi(value: str) -> str | None:
    match = _DOI_RE.search(value.strip())
    if match is None:
        return None
    return match.group(1).rstrip(".,;)").lower()


def normalize_arxiv(value: str) -> str | None:
    """`http://arxiv.org/abs/2506.08573v1` -> `2506.08573` (version dropped)."""
    text = value.strip()
    text = re.sub(r"^(arxiv:)", "", text, flags=re.IGNORECASE)
    match = _ARXIV_NEW_RE.search(text)
    if match is not None:
        return match.group(1)
    match = _ARXIV_OLD_RE.search(text)
    return match.group(1).lower() if match is not None else None


def normalize_openalex(value: str) -> str | None:
    match = _OPENALEX_RE.search(value.strip().rsplit("/", 1)[-1])
    return match.group(1).upper() if match is not None else None


def normalize_external_ids(ids: Mapping[str, str]) -> dict[str, str]:
    """Canonical forms for known id schemes; unknown keys pass through stripped."""
    out: dict[str, str] = {}
    for key, raw in ids.items():
        if not raw:
            continue
        k = key.lower()
        value: str | None
        if k == "doi":
            value = normalize_doi(raw)
        elif k == "arxiv":
            value = normalize_arxiv(raw)
        elif k == "openalex":
            value = normalize_openalex(raw)
        elif k == "github":
            value = raw.strip().lower()
        else:
            value = raw.strip()
        if value:
            out[k] = value
    doi = out.get("doi")
    if doi is not None and "arxiv" not in out and doi.startswith("10.48550/arxiv."):
        arxiv = normalize_arxiv(doi.removeprefix("10.48550/arxiv."))
        if arxiv:
            out["arxiv"] = arxiv
    return out


def id_keys(ids: Mapping[str, str]) -> set[str]:
    """`{"doi:10.1/x", "arxiv:2506.08573"}` for strong identifier matching."""
    norm = normalize_external_ids(ids)
    return {f"{k}:{norm[k]}" for k in STRONG_ID_KEYS if k in norm}
