"""Intelligence engine: external research retrieval, extraction, and provenance."""

from enigma.intelligence.engine import ResearchEngine
from enigma.intelligence.evidence import detect_contradictions, extract_evidence
from enigma.intelligence.expansion import QueryExpander, TfidfQueryExpander
from enigma.intelligence.extract import (
    DocumentFetcher,
    ExtractedDocument,
    html_to_text,
    pdf_to_text,
)
from enigma.intelligence.http import (
    HttpBudget,
    HttpClient,
    HttpResponse,
    HttpStatusError,
    RawStore,
    ResponseTooLargeError,
    RobotsDisallowedError,
)
from enigma.intelligence.models import (
    AuthorRef,
    ProviderStats,
    ResearchQuery,
    ResearchResult,
    SearchHit,
)
from enigma.intelligence.persist import PersistReport, persist_result
from enigma.intelligence.providers import SearchProvider, default_providers
from enigma.intelligence.untrusted import clean_text, strip_control_chars, wrap_untrusted

__all__ = [
    "AuthorRef",
    "DocumentFetcher",
    "ExtractedDocument",
    "HttpBudget",
    "HttpClient",
    "HttpResponse",
    "HttpStatusError",
    "PersistReport",
    "ProviderStats",
    "QueryExpander",
    "RawStore",
    "ResearchEngine",
    "ResearchQuery",
    "ResearchResult",
    "ResponseTooLargeError",
    "RobotsDisallowedError",
    "SearchHit",
    "SearchProvider",
    "TfidfQueryExpander",
    "clean_text",
    "default_providers",
    "detect_contradictions",
    "extract_evidence",
    "html_to_text",
    "pdf_to_text",
    "persist_result",
    "strip_control_chars",
    "wrap_untrusted",
]
