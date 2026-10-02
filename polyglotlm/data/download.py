"""Source manifest, licence audit and fetching of raw text.

A manifest lists every corpus that feeds a run, with its licence. Nothing is fetched until the
whole manifest passes the licence audit. Supported URIs: local paths and http(s). `${DATA_BUCKET}`
in a URI expands to the DATA_BUCKET environment variable (a local or mounted directory).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from polyglotlm.logs import get_logger

log = get_logger(__name__)

# SPDX-style identifiers of licences that allow redistribution of derived model weights.
ALLOWED_LICENCES: frozenset[str] = frozenset(
    {
        "CC0-1.0",
        "CC-BY-3.0",
        "CC-BY-4.0",
        "CC-BY-SA-3.0",
        "CC-BY-SA-4.0",
        "MIT",
        "Apache-2.0",
        "ODC-BY-1.0",
        "PDDL-1.0",
        "GODL-India",
        "public-domain",
    }
)


class LicenceError(ValueError):
    """Raised when a source's licence is not on the allowlist."""


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    lang: str
    uri: str
    licence: str
    sha256: str | None = None  # pin the content for reproducible snapshots


class SourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[Source]

    def for_lang(self, lang: str) -> list[Source]:
        return [source for source in self.sources if source.lang == lang]


def load_manifest(path: str | Path) -> SourceManifest:
    with Path(path).open(encoding="utf-8") as fh:
        return SourceManifest.model_validate(yaml.safe_load(fh))


def audit_licences(manifest: SourceManifest, extra_allowed: frozenset[str] = frozenset()) -> None:
    """Raise LicenceError naming every source whose licence is not allowed."""
    allowed = ALLOWED_LICENCES | extra_allowed
    offenders = [f"{s.name} ({s.licence})" for s in manifest.sources if s.licence not in allowed]
    if offenders:
        raise LicenceError(
            "sources with licences outside the allowlist: "
            + ", ".join(offenders)
            + f". Allowed: {sorted(allowed)}"
        )
    log.info("licence_audit_passed", sources=len(manifest.sources))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_uri(uri: str) -> str:
    if "${DATA_BUCKET}" not in uri:
        return uri
    bucket = os.environ.get("DATA_BUCKET")
    if not bucket:
        raise RuntimeError(f"source URI {uri!r} needs the DATA_BUCKET environment variable")
    return uri.replace("${DATA_BUCKET}", bucket.rstrip("/\\"))


def fetch(source: Source, dest_dir: Path) -> Path:
    """Copy or download one source into `dest_dir`, verify its hash, return the local path."""
    uri = resolve_uri(source.uri)
    suffix = ".jsonl" if uri.endswith(".jsonl") else ".txt"
    dest = dest_dir / f"{source.name}{suffix}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if uri.startswith(("http://", "https://")):
        log.info("download_start", source=source.name, uri=uri)
        with urllib.request.urlopen(uri, timeout=60) as response, dest.open("wb") as out:
            shutil.copyfileobj(response, out)
    elif "://" in uri:
        raise ValueError(f"unsupported URI scheme in {uri!r}; use a local path or http(s)")
    else:
        if not Path(uri).exists():
            raise FileNotFoundError(f"source {source.name!r}: {uri} does not exist")
        shutil.copyfile(uri, dest)
    digest = sha256_file(dest)
    if source.sha256 and digest != source.sha256:
        dest.unlink()
        raise ValueError(
            f"source {source.name!r}: sha256 mismatch (expected {source.sha256}, got {digest})"
        )
    log.info("source_fetched", source=source.name, lang=source.lang, sha256=digest)
    return dest
