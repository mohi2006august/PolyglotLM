"""Corpus build: fetch -> normalise -> redact PII -> language ID -> quality -> dedupe -> split.

Output per language is `train.txt` and `heldout.txt` (one NFC document per line) plus a
snapshot manifest with content hashes, so a run can be reproduced and audited.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polyglotlm.config import ENGLISH, TrainingRecipe
from polyglotlm.data.clean import QualityFilter, ToxicityFilter, normalise, redact_pii
from polyglotlm.data.dedupe import MinHashDeduper
from polyglotlm.data.download import (
    SourceManifest,
    audit_licences,
    fetch,
    load_manifest,
    sha256_file,
)
from polyglotlm.data.langid import LangIdentifier, build_identifier
from polyglotlm.logs import get_logger
from polyglotlm.text import split_words

log = get_logger(__name__)

MIN_LANGID_CONFIDENCE = 0.5


@dataclass
class LangStats:
    lang: str
    read: int = 0
    kept: int = 0
    heldout: int = 0
    words: int = 0
    pii_redactions: int = 0
    dropped: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "read": self.read,
            "kept": self.kept,
            "heldout": self.heldout,
            "words": self.words,
            "pii_redactions": self.pii_redactions,
            "dropped": dict(sorted(self.dropped.items())),
        }


def read_documents(path: Path) -> Iterator[str]:
    """Yield raw documents: one per non-empty line (`.txt`) or the `text` field (`.jsonl`)."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)["text"] if path.suffix == ".jsonl" else line


def is_heldout(text: str, fraction: float, seed: int) -> bool:
    """Deterministic, order-independent split: the same document always lands on the same side."""
    digest = hashlib.sha1(f"{seed}:{text}".encode(), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 < fraction


class DocumentCleaner:
    """The per-document chain for one language. `clean` returns None for dropped documents."""

    def __init__(
        self,
        lang: str,
        quality: QualityFilter,
        identifier: LangIdentifier,
        toxicity: ToxicityFilter,
        deduper: MinHashDeduper,
    ) -> None:
        self.lang = lang
        self.quality = quality
        self.identifier = identifier
        self.toxicity = toxicity
        self.deduper = deduper

    def _reject_reason(self, text: str) -> str | None:
        if reason := self.quality.reject_reason(text):
            return reason
        detected, confidence = self.identifier.identify(text)
        if detected != self.lang or confidence < MIN_LANGID_CONFIDENCE:
            return "wrong_language"
        if self.toxicity.is_toxic(text):
            return "toxic"
        return "duplicate" if self.deduper.is_duplicate(text) else None

    def clean(self, raw: str, stats: LangStats) -> str | None:
        stats.read += 1
        text, redactions = redact_pii(normalise(raw, self.lang))
        if reason := self._reject_reason(text):
            stats.dropped[reason] += 1
            return None
        stats.pii_redactions += redactions
        return text


def build_language(
    recipe: TrainingRecipe, lang: str, raw_files: list[Path], identifier: LangIdentifier
) -> LangStats:
    """Clean every raw file of one language and write its train / held-out split."""
    cleaner = DocumentCleaner(
        lang=lang,
        quality=QualityFilter(script=recipe.script_of(lang)),
        identifier=identifier,
        toxicity=ToxicityFilter.from_dir(recipe.toxicity_lexicon_dir, lang),
        deduper=MinHashDeduper(seed=recipe.seed),
    )
    stats = LangStats(lang=lang)
    train_path, heldout_path = recipe.paths.corpus(lang), recipe.heldout_path(lang)
    for path in (train_path, heldout_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with (
        train_path.open("w", encoding="utf-8", newline="\n") as train,
        heldout_path.open("w", encoding="utf-8", newline="\n") as heldout,
    ):
        for raw_file in raw_files:
            for raw in read_documents(raw_file):
                if (text := cleaner.clean(raw, stats)) is None:
                    continue
                to_heldout = is_heldout(text, recipe.heldout_fraction, recipe.seed)
                (heldout if to_heldout else train).write(text + "\n")
                stats.heldout += to_heldout
                stats.kept += not to_heldout
                stats.words += len(split_words(text))
    if not stats.kept or not stats.heldout:
        log.warning("corpus_split_empty", lang=lang, train=stats.kept, heldout=stats.heldout)
    log.info("corpus_built", lang=lang, **stats.as_dict())
    return stats


def _languages_to_build(requested: list[str], manifest: SourceManifest) -> list[str]:
    """Requested languages plus English (replay + regression) whenever a source provides it."""
    langs = list(dict.fromkeys(requested))
    if ENGLISH not in langs and manifest.for_lang(ENGLISH):
        langs.append(ENGLISH)
    missing = [lang for lang in langs if not manifest.for_lang(lang)]
    if missing:
        raise ValueError(f"no sources in the manifest for languages: {missing}")
    return langs


def build_corpora(recipe: TrainingRecipe, langs: list[str]) -> dict[str, Any]:
    """Run the whole data stage and write the snapshot manifest. Returns the snapshot."""
    if not recipe.sources:
        raise ValueError("recipe has no `sources` manifest")
    manifest = load_manifest(recipe.sources)
    audit_licences(manifest)
    identifier = build_identifier(recipe)
    snapshot: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": recipe.seed,
        "heldout_fraction": recipe.heldout_fraction,
        "sources": [],
        "languages": {},
    }
    for lang in _languages_to_build(langs, manifest):
        raw_files = [fetch(source, recipe.paths.raw(lang)) for source in manifest.for_lang(lang)]
        for source, raw_file in zip(manifest.for_lang(lang), raw_files, strict=True):
            snapshot["sources"].append(
                {**source.model_dump(exclude={"sha256"}), "sha256": sha256_file(raw_file)}
            )
        stats = build_language(recipe, lang, raw_files, identifier)
        snapshot["languages"][lang] = {
            **stats.as_dict(),
            "train_sha256": sha256_file(recipe.paths.corpus(lang)),
            "heldout_sha256": sha256_file(recipe.heldout_path(lang)),
        }
    recipe.paths.snapshot.parent.mkdir(parents=True, exist_ok=True)
    recipe.paths.snapshot.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log.info("snapshot_written", path=str(recipe.paths.snapshot))
    return snapshot
