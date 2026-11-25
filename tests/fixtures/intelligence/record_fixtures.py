"""Re-record the real API responses used by the offline intelligence tests.

Run manually (network required): `uv run python tests/fixtures/intelligence/record_fixtures.py`.
Responses are stored verbatim; tests replay them with respx. They are recorded data, not live
operation, and the tests say so.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).parent
UA = "Enigma-Research/0.1 (+https://github.com/martyroeters-ship-it/enigma)"
MAILTO = "enigma-research@users.noreply.github.com"
TOPIC = "perpetual futures funding rate"
OA_SELECT = (
    "id,doi,display_name,publication_date,language,authorships,abstract_inverted_index,"
    "cited_by_count,referenced_works,ids,primary_location,type"
)


def get(client: httpx.Client, name: str, url: str, params: dict[str, Any]) -> httpx.Response:
    time.sleep(3.0)
    response = client.get(url, params=params)
    response.raise_for_status()
    suffix = ".xml" if "xml" in response.headers.get("content-type", "") else ".json"
    (HERE / f"{name}{suffix}").write_bytes(response.content)
    print(name, response.status_code, len(response.content), file=sys.stderr)  # noqa: T201
    return response


def main() -> None:
    with httpx.Client(headers={"User-Agent": UA}, timeout=30, follow_redirects=True) as c:
        get(
            c,
            "arxiv_search",
            "https://export.arxiv.org/api/query",
            {
                "search_query": "all:perpetual AND all:futures AND all:funding AND all:rate",
                "start": 0,
                "max_results": 4,
                "sortBy": "relevance",
            },
        )
        oa = get(
            c,
            "openalex_search",
            "https://api.openalex.org/works",
            {
                "search": TOPIC,
                "per_page": 4,
                "select": OA_SELECT,
                "mailto": MAILTO,
            },
        ).json()
        top = next(w for w in oa["results"] if w.get("referenced_works"))
        refs = [r.rsplit("/", 1)[-1] for r in top["referenced_works"][:3]]
        get(
            c,
            "openalex_refs",
            "https://api.openalex.org/works",
            {
                "filter": "openalex_id:" + "|".join(refs),
                "per_page": 3,
                "select": OA_SELECT,
                "mailto": MAILTO,
            },
        )
        cited = next((w for w in oa["results"] if w.get("cited_by_count")), oa["results"][0])
        get(
            c,
            "openalex_cites",
            "https://api.openalex.org/works",
            {
                "filter": "cites:" + cited["id"].rsplit("/", 1)[-1],
                "per_page": 3,
                "select": OA_SELECT,
                "mailto": MAILTO,
                "sort": "cited_by_count:desc",
            },
        )
        get(
            c,
            "crossref_search",
            "https://api.crossref.org/works",
            {
                "query": TOPIC,
                "rows": 4,
                "select": "DOI,title,abstract,author,issued,URL,type,is-referenced-by-count",
                "mailto": MAILTO,
            },
        )
        get(
            c,
            "github_search",
            "https://api.github.com/search/repositories",
            {
                "q": f"{TOPIC} in:name,description,readme",
                "per_page": 3,
            },
        )
        base = {"action": "query", "format": "json", "formatversion": 2}
        en = get(
            c,
            "wikipedia_en_search",
            "https://en.wikipedia.org/w/api.php",
            {
                **base,
                "list": "search",
                "srsearch": TOPIC,
                "srlimit": 3,
            },
        ).json()
        titles = [r["title"] for r in en["query"]["search"]]
        props = {
            "prop": "extracts|info|pageprops",
            "exintro": 1,
            "explaintext": 1,
            "exlimit": "max",
            "inprop": "url",
            "ppprop": "wikibase_item",
            "redirects": 1,
        }
        get(
            c,
            "wikipedia_en_extracts",
            "https://en.wikipedia.org/w/api.php",
            {
                **base,
                **props,
                "titles": "|".join(titles),
            },
        )
        links = get(
            c,
            "wikipedia_en_langlinks_de",
            "https://en.wikipedia.org/w/api.php",
            {
                **base,
                "prop": "langlinks",
                "lllang": "de",
                "titles": "|".join(titles),
                "lllimit": "max",
            },
        ).json()
        de_titles = [
            p["langlinks"][0]["title"] for p in links["query"]["pages"] if p.get("langlinks")
        ]
        get(
            c,
            "wikipedia_de_search",
            "https://de.wikipedia.org/w/api.php",
            {
                **base,
                "list": "search",
                "srsearch": TOPIC,
                "srlimit": 2,
            },
        )
        get(
            c,
            "wikipedia_de_extracts",
            "https://de.wikipedia.org/w/api.php",
            {
                **base,
                **props,
                "titles": "|".join(de_titles or ["Terminkontrakt"]),
            },
        )
    total = sum(p.stat().st_size for p in HERE.glob("*.json")) + sum(
        p.stat().st_size for p in HERE.glob("*.xml")
    )
    print("total bytes", total, json.dumps(sorted(p.name for p in HERE.iterdir())), file=sys.stderr)  # noqa: T201


if __name__ == "__main__":
    main()
