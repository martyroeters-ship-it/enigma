"""Document fetch and text extraction (HTML via selectolax, PDF via pypdf), with hard limits.

Extracted text is untrusted data: control characters are stripped and nothing is executed.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from pypdf import PdfReader
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from enigma.core.errors import EnigmaError
from enigma.intelligence.http import HttpClient
from enigma.intelligence.untrusted import clean_text

DEFAULT_MAX_CHARS = 400_000
DEFAULT_MAX_PDF_PAGES = 40
DEFAULT_MAX_PDF_BYTES = 15 * 1024 * 1024

_STRIP_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "iframe",
    "svg",
    "canvas",
    "template",
    "button",
    "select",
    "object",
    "embed",
)
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "tr",
        "br",
        "section",
        "article",
        "blockquote",
        "pre",
        "dd",
        "dt",
        "figcaption",
        "td",
        "th",
        "table",
        "ul",
        "ol",
    }
)


class ExtractionError(EnigmaError):
    """The document could not be parsed into text."""


@dataclass(frozen=True)
class ExtractedDocument:
    url: str
    content_type: str
    title: str | None
    text: str
    raw_uri: str
    content_hash: str
    truncated: bool
    pages: int | None = None


def html_to_text(html: str, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[str | None, str]:
    """Return `(title, clean_text)`; scripts, styles, navigation and forms are removed."""
    tree = HTMLParser(html)
    title_node = tree.css_first("title")
    title = clean_text(title_node.text()) if title_node is not None else None
    tree.strip_tags(list(_STRIP_TAGS))
    for selector in ('[role="navigation"]', '[aria-hidden="true"]', ".navbox", ".mw-editsection"):
        for node in tree.css(selector):
            node.decompose()
    root = tree.css_first("main") or tree.css_first("article") or tree.body or tree.root
    if root is None:
        return title, ""
    parts: list[str] = []
    for node in root.traverse(include_text=True):
        if node.tag == "-text":
            parts.append(node.text_content or "")
        elif node.tag in _BLOCK_TAGS:
            parts.append("\n")
    return title or None, clean_text("".join(parts))[:max_chars]


def pdf_to_text(
    data: bytes, max_pages: int = DEFAULT_MAX_PDF_PAGES, max_chars: int = DEFAULT_MAX_CHARS
) -> tuple[str, int, bool]:
    """Return `(text, pages_read, truncated)`. Encrypted or malformed PDFs raise ExtractionError."""
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ExtractionError("encrypted PDFs are not processed")
        total = len(reader.pages)
        chunks: list[str] = []
        size = 0
        pages_read = 0
        for page in reader.pages[:max_pages]:
            chunk = page.extract_text() or ""
            chunks.append(chunk)
            pages_read += 1
            size += len(chunk)
            if size >= max_chars:
                break
    except ExtractionError:
        raise
    except Exception as exc:  # pypdf raises a wide range of parser errors on hostile input
        raise ExtractionError(f"PDF parse failed: {type(exc).__name__}") from exc
    text = clean_text("\n".join(chunks))
    truncated = pages_read < total or len(text) > max_chars
    return text[:max_chars], pages_read, truncated


class DocumentFetcher:
    """Fetch a public page (robots.txt respected) and extract its text."""

    def __init__(
        self,
        client: HttpClient,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
        max_pdf_bytes: int = DEFAULT_MAX_PDF_BYTES,
    ) -> None:
        self.client = client
        self.max_chars = max_chars
        self.max_pdf_pages = max_pdf_pages
        self.max_pdf_bytes = max_pdf_bytes

    def fetch(self, url: str) -> ExtractedDocument:
        response = self.client.get(url, respect_robots=True, max_bytes=self.max_pdf_bytes)
        ctype = response.content_type
        if ctype == "application/pdf" or response.content[:5] == b"%PDF-":
            text, pages, truncated = pdf_to_text(
                response.content, self.max_pdf_pages, self.max_chars
            )
            return ExtractedDocument(
                response.url,
                "application/pdf",
                None,
                text,
                response.raw_uri,
                response.content_hash,
                truncated,
                pages,
            )
        if ctype in ("text/html", "application/xhtml+xml", ""):
            title, text = html_to_text(response.text, self.max_chars)
            return ExtractedDocument(
                response.url,
                "text/html",
                title,
                text,
                response.raw_uri,
                response.content_hash,
                len(text) >= self.max_chars,
            )
        if ctype.startswith("text/"):
            text = clean_text(response.text)
            return ExtractedDocument(
                response.url,
                ctype,
                None,
                text[: self.max_chars],
                response.raw_uri,
                response.content_hash,
                len(text) > self.max_chars,
            )
        raise ExtractionError(f"unsupported content type {ctype!r} at {response.url}")
