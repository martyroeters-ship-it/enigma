"""Rewrite stored artifact paths so a copied institutional home still opens.

Document history is append-only. A migration adds a new version whose content
hashes are the hashes already recorded; it does not edit the previous version
or the audit chain. A file whose bytes do not match the recorded hash blocks
the migration and is not rewritten.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from enigma.core.config import Settings
from enigma.core.hashing import sha256_file
from enigma.core.paths import (
    StorageRoots,
    bind_uri,
    dataset_parquet,
    find_digest,
    is_legacy_absolute,
    resolve_uri,
    within,
)
from enigma.data.hashing import frame_content_hash
from enigma.schemas.research import StrategyPackage
from enigma.schemas.validation import HoldoutManifest
from enigma.store.db import Database
from enigma.store.documents import DocumentStore, model_for_kind
from enigma.validation.holdout import VALIDATION_DB_NAME
from enigma.validation.holdout import frame_content_hash as holdout_hash

_PATH_FIELDS = frozenset({"uri", "artifact_uri", "raw_uri", "raw_response_uri"})
_EMBEDDED_PATH = re.compile(r"(?:/tmp/|/home/|/Users/|/var/folders/)")
_ACTOR = "home_migrate"


@dataclass(frozen=True, slots=True)
class PathIssue:
    status: str
    subject: str
    field: str
    uri: str
    detail: str
    dataset_id: str = ""
    symbols: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HomeReport:
    ok: int
    rewritten: int
    legacy: int
    issues: tuple[PathIssue, ...]
    notes: tuple[str, ...]
    refused: bool

    @property
    def unresolved(self) -> tuple[PathIssue, ...]:
        return tuple(issue for issue in self.issues if issue.status == "unresolved")

    @property
    def mismatches(self) -> tuple[PathIssue, ...]:
        return tuple(issue for issue in self.issues if issue.status == "mismatch")

    def text(self) -> str:
        lines = [
            f"portable references: {self.ok}",
            f"rewritten: {self.rewritten}",
            f"legacy absolute paths that still resolve: {self.legacy}",
            f"unresolved: {len(self.unresolved)}",
            f"integrity failures: {len(self.mismatches)}",
        ]
        for issue in (*self.mismatches, *self.unresolved):
            lines.append(
                f"{issue.status}: {issue.subject} {issue.field} {issue.uri} ({issue.detail})"
            )
        lines.extend(self.notes)
        if self.refused:
            lines.append("migration refused: stored hashes do not match the files")
        elif self.unresolved:
            lines.append(
                "unresolved artifacts are left in place; continuous research quarantines "
                "the objectives that need them and does not validate those objectives"
            )
        return "\n".join(lines)


@dataclass
class _Write:
    label: str
    kind: str
    doc_id: str
    body: dict[str, Any]
    path: Path | None = None
    workflow: str = ""
    step: str = ""
    status: str = ""


@dataclass
class _Scan:
    roots: StorageRoots
    ok: int = 0
    legacy: int = 0
    issues: list[PathIssue] = field(default_factory=list)
    writes: list[_Write] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, issue: PathIssue) -> None:
        if issue.status == "ok":
            self.ok += 1
            return
        if issue.status == "legacy":
            self.legacy += 1
        self.issues.append(issue)


def inspect_home(settings: Settings, stores: Any) -> HomeReport:
    """Read every stored artifact reference. Does not rewrite anything."""
    return _finish(settings, stores, apply=False)


def migrate_home(settings: Settings, stores: Any) -> HomeReport:
    """Rewrite resolved absolute paths to portable URIs. Idempotent."""
    return _finish(settings, stores, apply=True)


def _finish(settings: Settings, stores: Any, *, apply: bool) -> HomeReport:
    scan = _Scan(StorageRoots.from_settings(settings))
    _scan_manifests(scan, settings)
    _scan_documents(scan, stores.documents, "control", settings.research_dir / "data")
    _scan_checkpoints(scan, stores)
    _scan_packages(scan, stores.documents)
    validation_db = settings.validation_dir / VALIDATION_DB_NAME
    if validation_db.is_file():
        database = Database(validation_db)
        try:
            view = DocumentStore(database)
            _scan_documents(scan, view, "validation", settings.validation_dir / "data")
            _verify_holdouts(scan, view, settings.validation_dir / "data")
        finally:
            database.close()
    refused = any(issue.status == "mismatch" for issue in scan.issues)
    rewritten = 0
    if apply and not refused:
        rewritten = _apply(scan, stores)
        stores.audit.record(
            "system",
            "home.migrated",
            None,
            {
                "rewritten": rewritten,
                "unresolved": sum(1 for issue in scan.issues if issue.status == "unresolved"),
                "legacy": scan.legacy,
                "subjects": [item.doc_id for item in scan.writes][:40],
            },
        )
    elif apply and refused:
        stores.audit.record(
            "system",
            "home.migrate_refused",
            None,
            {
                "mismatches": [
                    {"subject": issue.subject, "detail": issue.detail}
                    for issue in scan.issues
                    if issue.status == "mismatch"
                ][:40]
            },
        )
    return HomeReport(
        ok=scan.ok,
        rewritten=rewritten if apply else 0,
        legacy=scan.legacy,
        issues=tuple(scan.issues),
        notes=tuple(scan.notes),
        refused=refused,
    )


def _apply(scan: _Scan, stores: Any) -> int:
    validation_db = scan.roots.validation / VALIDATION_DB_NAME
    validation: DocumentStore | None = None
    database: Database | None = None
    if validation_db.is_file() and any(item.label == "validation" for item in scan.writes):
        database = Database(validation_db)
        validation = DocumentStore(database)
    written = 0
    try:
        for item in scan.writes:
            if item.label == "manifest" and item.path is not None:
                _write_manifest(item.path, item.body)
            elif item.label == "checkpoint":
                stores.checkpoints.save(
                    item.doc_id, item.workflow, item.step, item.body, item.status
                )
            elif item.label == "validation" and validation is not None:
                model = model_for_kind(item.kind)
                validation.put(model.model_validate(item.body), recorded_by=_ACTOR)
            elif item.label == "control":
                model = model_for_kind(item.kind)
                stores.documents.put(model.model_validate(item.body), recorded_by=_ACTOR)
            else:
                continue
            written += 1
    finally:
        if database is not None:
            database.close()
    return written


def _write_manifest(path: Path, body: dict[str, Any]) -> None:
    from enigma.schemas.research import DatasetVersion  # noqa: PLC0415

    payload = DatasetVersion.model_validate(body).model_dump_json(indent=2)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _scan_manifests(scan: _Scan, settings: Settings) -> None:
    for catalog in (settings.research_dir / "data", settings.validation_dir / "data"):
        root = catalog / "datasets"
        if not root.is_dir():
            continue
        for manifest in sorted(root.glob("*/*/dataset.json")):
            body = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(body, dict):
                continue
            updated, changed = _rewrite(scan, body, f"manifest:{manifest.parent.name}", catalog)
            if changed:
                scan.writes.append(
                    _Write(
                        "manifest",
                        "dataset_version",
                        str(body.get("id", "")),
                        updated,
                        manifest,
                    )
                )


def _scan_documents(scan: _Scan, documents: DocumentStore, label: str, catalog: Path) -> None:
    rows = documents.db.query("SELECT kind, id, body FROM documents_latest ORDER BY kind, id")
    for row in rows:
        body = json.loads(row["body"])
        if not isinstance(body, dict):
            continue
        subject = f"{label}:{row['kind']}:{row['id']}"
        updated, changed = _rewrite(scan, body, subject, catalog)
        if not changed:
            continue
        try:
            model_for_kind(str(row["kind"])).model_validate(updated)
        except (KeyError, ValueError) as exc:
            scan.add(
                PathIssue(
                    "mismatch",
                    subject,
                    "body",
                    "",
                    f"rewritten document does not validate: {exc}",
                    dataset_id=str(body.get("id", "")),
                )
            )
            continue
        scan.writes.append(_Write(label, str(row["kind"]), str(row["id"]), updated))


def _scan_checkpoints(scan: _Scan, stores: Any) -> None:
    for row in stores.checkpoints.list():
        state = row["state"]
        if not isinstance(state, dict):
            continue
        updated, changed = _rewrite(scan, state, f"checkpoint:{row['workflow_id']}", None)
        if changed:
            scan.writes.append(
                _Write(
                    "checkpoint",
                    "checkpoint",
                    str(row["workflow_id"]),
                    updated,
                    workflow=str(row["workflow"]),
                    step=str(row["step"]),
                    status=str(row["status"]),
                )
            )


def _scan_packages(scan: _Scan, documents: DocumentStore) -> None:
    packages = documents.list(StrategyPackage, limit=10_000)
    embedded = 0
    for package in packages:
        if _EMBEDDED_PATH.search(package.source_code):
            embedded += 1
            scan.notes.append(
                f"strategy package {package.id} embeds an absolute path in its source; "
                "left unchanged so its content hash stays valid"
            )
    scan.notes.append(
        f"strategy packages checked: {len(packages)}; source is stored in the document, "
        f"not as a filesystem path; embedded absolute paths left unchanged: {embedded}"
    )


def _verify_holdouts(scan: _Scan, documents: DocumentStore, catalog: Path) -> None:
    manifests = documents.list(HoldoutManifest, limit=10_000)
    checked = 0
    for manifest in manifests:
        checked += 1
        frame = _holdout_frame(scan, manifest, catalog)
        if frame is None:
            continue
        rows = frame.filter(pl.col("ts") >= manifest.holdout_start).filter(
            pl.col("ts") <= manifest.data_end
        )
        actual = holdout_hash(rows)
        if actual != manifest.holdout_content_hash:
            scan.add(
                PathIssue(
                    "mismatch",
                    manifest.id,
                    "holdout_content_hash",
                    "",
                    f"sealed holdout hash {manifest.holdout_content_hash} != file hash {actual}",
                    dataset_id=manifest.dataset_ids[0] if manifest.dataset_ids else "",
                    symbols=tuple(manifest.symbols),
                )
            )
    scan.notes.append(
        f"sealed holdout manifests checked: {checked}; boundaries and recorded hashes "
        "were not rewritten"
    )


def _holdout_frame(scan: _Scan, manifest: HoldoutManifest, catalog: Path) -> pl.DataFrame | None:
    frames: list[pl.DataFrame] = []
    for dataset_id in manifest.dataset_ids:
        path = _manifest_for_id(catalog, dataset_id)
        if path is None:
            scan.add(
                PathIssue(
                    "unresolved",
                    manifest.id,
                    "dataset_ids",
                    dataset_id,
                    "sealed holdout dataset is not in this validation catalog",
                    dataset_id=dataset_id,
                    symbols=tuple(manifest.symbols),
                )
            )
            return None
        body = json.loads(path.read_text(encoding="utf-8"))
        located, _detail = _locate(scan.roots, body, "uri", str(body.get("uri", "")), catalog)
        if located is None:
            scan.add(
                PathIssue(
                    "unresolved",
                    manifest.id,
                    "uri",
                    str(body.get("uri", "")),
                    "sealed holdout parquet is not under the validation catalog",
                    dataset_id=dataset_id,
                    symbols=tuple(manifest.symbols),
                )
            )
            return None
        frames.append(pl.read_parquet(located))
    if not frames:
        return None
    return pl.concat(frames, how="vertical")


def _manifest_for_id(catalog: Path, dataset_id: str) -> Path | None:
    root = catalog / "datasets"
    if not root.is_dir():
        return None
    for manifest in root.glob("*/*/dataset.json"):
        body = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(body, dict) and body.get("id") == dataset_id:
            return manifest
    return None


def _rewrite(
    scan: _Scan, body: dict[str, Any], subject: str, catalog: Path | None
) -> tuple[dict[str, Any], bool]:
    updated: dict[str, Any] = {}
    changed = False
    for key, value in body.items():
        if key in _PATH_FIELDS and isinstance(value, str) and value:
            new_value, field_changed = _one_path(scan, body, key, value, subject, catalog)
            updated[key] = new_value
            changed = changed or field_changed
        elif isinstance(value, dict):
            child, child_changed = _rewrite(scan, value, subject, catalog)
            updated[key] = child
            changed = changed or child_changed
        elif isinstance(value, list):
            items, list_changed = _rewrite_list(scan, value, subject, catalog)
            updated[key] = items
            changed = changed or list_changed
        else:
            updated[key] = value
    return updated, changed


def _rewrite_list(
    scan: _Scan, values: list[Any], subject: str, catalog: Path | None
) -> tuple[list[Any], bool]:
    items: list[Any] = []
    changed = False
    for value in values:
        if isinstance(value, dict):
            child, child_changed = _rewrite(scan, value, subject, catalog)
            items.append(child)
            changed = changed or child_changed
        else:
            items.append(value)
    return items, changed


def _one_path(
    scan: _Scan,
    parent: dict[str, Any],
    key: str,
    uri: str,
    subject: str,
    catalog: Path | None,
) -> tuple[str, bool]:
    located, detail = _locate(scan.roots, parent, key, uri, catalog)
    dataset_id = str(parent.get("id", "")) if _is_dataset(parent) else ""
    symbols = _symbols(parent)
    if located is None:
        scan.add(PathIssue("unresolved", subject, key, uri, detail, dataset_id, symbols))
        return uri, False
    problem = _verify(parent, key, located)
    if problem:
        scan.add(PathIssue("mismatch", subject, key, uri, problem, dataset_id, symbols))
        return uri, False
    portable = bind_uri(located, scan.roots)
    if portable == uri:
        scan.add(PathIssue("ok", subject, key, uri, "portable", dataset_id, symbols))
        return uri, False
    if uri.startswith("inst:"):
        scan.add(PathIssue("ok", subject, key, uri, "portable", dataset_id, symbols))
        return uri, False
    scan.add(PathIssue("legacy", subject, key, uri, portable, dataset_id, symbols))
    return portable, True


def _locate(
    roots: StorageRoots,
    parent: dict[str, Any],
    key: str,
    uri: str,
    catalog: Path | None,
) -> tuple[Path | None, str]:
    if _is_dataset(parent) and key == "uri":
        plane_catalog = _catalog_for(roots, str(parent.get("plane", "")), catalog)
        addressed = dataset_parquet(plane_catalog, str(parent["name"]), str(parent["content_hash"]))
        if addressed.is_file():
            return addressed, ""
        found = resolve_uri(uri, roots, catalog_root=plane_catalog)
        if found is not None and within(found, plane_catalog):
            return found, ""
        return None, f"dataset parquet is not under {plane_catalog}"
    found = resolve_uri(uri, roots, catalog_root=catalog)
    if found is not None:
        return found, ""
    digest = _digest_hint(parent, key)
    if digest:
        found = find_digest(_raw_dirs(roots), digest)
        if found is not None:
            return found, ""
    if is_legacy_absolute(uri) or uri.startswith("inst:"):
        return None, "artifact is not under the current institutional home"
    return None, "artifact path is not a portable institutional URI"


def _verify(parent: dict[str, Any], key: str, path: Path) -> str:
    if _is_dataset(parent) and key == "uri":
        expected = str(parent["content_hash"])
        try:
            actual = frame_content_hash(pl.read_parquet(path))
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            return f"unreadable parquet: {exc}"
        if actual != expected:
            return f"content hash {actual} != recorded {expected}"
        return ""
    if key == "raw_uri" and parent.get("content_hash"):
        return _file_hash(path, str(parent["content_hash"]))
    if key == "raw_response_uri" and parent.get("raw_response_hash"):
        return _file_hash(path, str(parent["raw_response_hash"]))
    if key == "artifact_uri" and parent.get("result_hash"):
        prefix = str(parent["result_hash"]).removeprefix("sha256:")[:32]
        if prefix not in path.name:
            return f"artifact name does not contain result hash {prefix}"
    return ""


def _file_hash(path: Path, expected: str) -> str:
    try:
        actual = sha256_file(path)
    except OSError as exc:
        return f"unreadable file: {exc}"
    if actual != expected:
        return f"file hash {actual} != recorded {expected}"
    return ""


def _is_dataset(body: dict[str, Any]) -> bool:
    return "content_hash" in body and "row_count" in body and "columns" in body and "name" in body


def _catalog_for(roots: StorageRoots, plane: str, fallback: Path | None) -> Path:
    if plane == "validation":
        return roots.validation / "data"
    if plane == "research":
        return roots.research / "data"
    if fallback is not None:
        return fallback
    return roots.research / "data"


def _symbols(body: dict[str, Any]) -> tuple[str, ...]:
    found: list[str] = []
    for item in body.get("instruments") or ():
        if isinstance(item, dict) and item.get("symbol"):
            found.append(str(item["symbol"]))
    raw = body.get("symbols")
    if isinstance(raw, list):
        found.extend(str(item) for item in raw)
    return tuple(sorted(set(found)))


def _digest_hint(parent: dict[str, Any], key: str) -> str:
    field = "content_hash" if key == "raw_uri" else "raw_response_hash"
    value = parent.get(field)
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return ""
    return value.removeprefix("sha256:")


def _raw_dirs(roots: StorageRoots) -> tuple[Path, ...]:
    return (
        roots.research / "raw" / "intelligence",
        roots.research / "raw",
        roots.validation / "raw",
        roots.home / "raw",
    )
