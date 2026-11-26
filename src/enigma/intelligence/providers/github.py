"""GitHub repository search (unauthenticated by default; `GITHUB_TOKEN` raises the rate limit)."""

from __future__ import annotations

import os

from enigma.intelligence.models import AuthorRef, SearchHit
from enigma.intelligence.providers.base import ProviderBase, parse_date, strip_markup
from enigma.schemas.knowledge import SourceKind

API_URL = "https://api.github.com/search/repositories"


class GitHubProvider(ProviderBase):
    name = "github"

    def search(self, query: str, limit: int, language: str | None) -> list[SearchHit]:
        if language not in (None, "en"):
            return []
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = self.client.get(
            API_URL,
            params={"q": f"{query} in:name,description,readme", "per_page": limit},
            headers=headers,
        )
        hits: list[SearchHit] = []
        for rank, repo in enumerate(response.json().get("items") or []):
            full_name = repo.get("full_name")
            url = repo.get("html_url")
            if not full_name or not url:
                continue
            owner = (repo.get("owner") or {}).get("login")
            rec_uri, rec_hash = self._record_json(repo)
            hits.append(
                SearchHit(
                    kind=SourceKind.CODE_REPOSITORY,
                    title=str(full_name),
                    url=str(url),
                    abstract=strip_markup(repo.get("description")),
                    authors=(AuthorRef(name=str(owner), external_ids={"github": str(owner)}),)
                    if owner
                    else (),
                    published_at=parse_date(repo.get("created_at")),
                    external_ids={"github": str(full_name)},
                    popularity=int(repo.get("stargazers_count") or 0),
                    **self._provenance(response, rec_uri, rec_hash, query, rank),
                )
            )
        return hits
