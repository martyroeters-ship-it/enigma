"""Artifact locations relative to an institutional root.

A stored URI is ``inst:<anchor>/<relative>``. The anchors are the directories
the institution owns: ``home``, ``research``, ``validation`` and ``execution``.
``catalog`` is the data directory of the caller that stored the file, used only
when that caller has no institutional roots. Absolute paths are legacy values
from a home that was copied to another machine. They are resolved by finding
the same relative suffix under the current roots, and only when that match is
unique.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from enigma.core.config import Settings

PREFIX = "inst:"
_SPECIFIC = frozenset({"research", "validation", "execution"})


@dataclass(frozen=True, slots=True)
class StorageRoots:
    """Directories that move together when an institutional home is relocated."""

    home: Path
    research: Path
    validation: Path
    execution: Path

    @classmethod
    def from_settings(cls, settings: Settings) -> StorageRoots:
        return cls(
            settings.home,
            settings.research_dir,
            settings.validation_dir,
            settings.execution_dir,
        )

    def anchors(self) -> dict[str, Path]:
        return {
            "research": self.research,
            "validation": self.validation,
            "execution": self.execution,
            "home": self.home,
        }


def dataset_parquet(catalog_root: Path, name: str, content_hash: str) -> Path:
    """Content-addressed Parquet path under a plane's data catalog."""
    digest = content_hash.removeprefix("sha256:")[:16]
    return catalog_root / "datasets" / name / digest / "data.parquet"


def within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def bind_uri(
    path: Path,
    roots: StorageRoots | None = None,
    *,
    catalog_root: Path | None = None,
) -> str:
    """Portable URI for `path`. An absolute path is returned only when it is outside every root."""
    resolved = path.resolve()
    named: list[tuple[str, Path]] = []
    if roots is not None:
        named.extend(roots.anchors().items())
    if catalog_root is not None:
        named.append(("catalog", catalog_root))
    best: tuple[tuple[int, int], str, Path] | None = None
    for name, root in named:
        try:
            relative = resolved.relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        rank = (1 if name in _SPECIFIC else 0, len(root.resolve().parts))
        if best is None or rank > best[0]:
            best = (rank, name, relative)
    if best is None:
        return resolved.as_posix()
    relative_posix = best[2].as_posix()
    if relative_posix == ".":
        return f"{PREFIX}{best[1]}/"
    return f"{PREFIX}{best[1]}/{relative_posix}"


def is_legacy_absolute(value: str) -> bool:
    """True for a filesystem path stored before URIs were portable."""
    if value.startswith((PREFIX, "http://", "https://", "sha256:", "mailto:")):
        return False
    if value.startswith("/"):
        return True
    return len(value) >= 3 and value[1] == ":" and value[2] in {"\\", "/"}


def join_uri(
    uri: str,
    roots: StorageRoots | None = None,
    *,
    catalog_root: Path | None = None,
) -> Path | None:
    """Path this URI names. The file does not have to exist."""
    if uri.startswith(PREFIX):
        anchor, separator, relative = uri.removeprefix(PREFIX).partition("/")
        if not separator or not anchor:
            return None
        base = _anchor(anchor, roots, catalog_root)
        if base is None:
            return None
        return base / relative
    path = Path(uri)
    if not path.is_absolute():
        if catalog_root is None:
            return None
        return catalog_root / path
    if path.is_file():
        return path
    recovered = recover_absolute(uri, _search_roots(roots, catalog_root))
    return recovered if recovered is not None else path


def resolve_uri(
    uri: str,
    roots: StorageRoots | None = None,
    *,
    catalog_root: Path | None = None,
) -> Path | None:
    """Existing file for `uri`, or None."""
    candidate = join_uri(uri, roots, catalog_root=catalog_root)
    if candidate is not None and candidate.is_file():
        return candidate
    return None


def recover_absolute(uri: str, roots: Iterable[Path]) -> Path | None:
    """Find `uri` under `roots` by its longest unique relative suffix."""
    parts = _posix_parts(uri)
    if parts is None or len(parts) < 2:
        return None
    search = []
    for root in roots:
        try:
            if root.is_dir():
                search.append(root.resolve())
        except OSError:
            continue
    found: dict[int, set[Path]] = {}
    start = 1 if parts[0] == "/" else 0
    for index in range(start, len(parts)):
        suffix = Path(*parts[index:])
        hits = {
            (root / suffix).resolve()
            for root in search
            if (root / suffix).is_file()
        }
        if hits:
            found[len(parts) - index] = hits
    if not found:
        return None
    hits = found[max(found)]
    if len(hits) == 1:
        return next(iter(hits))
    return None


def _anchor(name: str, roots: StorageRoots | None, catalog_root: Path | None) -> Path | None:
    if name == "catalog":
        return catalog_root
    if roots is None:
        return None
    return roots.anchors().get(name)


def _search_roots(roots: StorageRoots | None, catalog_root: Path | None) -> list[Path]:
    found: list[Path] = []
    if roots is not None:
        found.extend(roots.anchors().values())
        found.append(roots.research / "data")
        found.append(roots.validation / "data")
        found.append(roots.research / "raw" / "intelligence")
    if catalog_root is not None:
        found.append(catalog_root)
    return found


def _posix_parts(uri: str) -> tuple[str, ...] | None:
    if not uri.startswith("/"):
        return None
    return PurePosixPath(uri).parts


def find_digest(roots: Sequence[Path], digest: str) -> Path | None:
    """Content-addressed file whose name starts with `digest`, when exactly one exists."""
    hits: list[Path] = []
    for root in roots:
        folder = root / digest[:2]
        if not folder.is_dir():
            continue
        hits.extend(path for path in folder.glob(digest + "*") if path.is_file())
    if len(hits) == 1:
        return hits[0]
    return None
