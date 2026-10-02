"""Recipe data models. A recipe is the single source of truth for a reproducible run."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from polyglotlm.text import SCRIPT_RANGES

ENGLISH = "en"
ENGLISH_SCRIPT = "Latn"

# Display names used in teacher prompts and reports; unknown codes fall back to the code itself.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "ur": "Urdu",
    "bn": "Bengali",
    "mr": "Marathi",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


class LanguageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str  # "ta", "hi", "ur", "bn"
    script: str  # "Taml", "Deva", "Arab", "Beng"
    new_tokens: int = Field(default=8000, ge=0)
    # Share of CPT tokens. None means "derive from corpus size" (temperature sampling).
    cpt_mix_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    # Held-out native text for perplexity. None means the pipeline default under the workdir.
    heldout_ppl_path: str | None = None

    @model_validator(mode="after")
    def _known_script(self) -> LanguageConfig:
        if self.script not in SCRIPT_RANGES:
            raise ValueError(f"unknown script {self.script!r}; known: {sorted(SCRIPT_RANGES)}")
        if self.code == ENGLISH:
            raise ValueError("English is the replay language; do not list it as a target")
        return self


class StageConfig(BaseModel):
    """Optimisation settings for one training stage (CPT or SFT)."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=10_000, ge=1)
    batch_size: int = Field(default=8, ge=1)
    block_size: int = Field(default=1024, ge=8)
    lr: float = Field(default=1e-4, gt=0.0)  # LoRA matrices
    embed_lr: float = Field(default=5e-4, gt=0.0)  # embeddings + LM head
    lr_warmup_steps: int = Field(default=200, ge=0)  # LoRA ramp after the embedding warm-up
    weight_decay: float = Field(default=0.0, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    eval_every: int = Field(default=500, ge=1)
    eval_blocks: int = Field(default=64, ge=1)  # held-out blocks scored per language
    patience: int = Field(default=3, ge=1)  # evals without improvement before stopping


DEFAULT_TOPICS: tuple[str, ...] = (
    "farming and the monsoon",
    "health and visiting a clinic",
    "government schemes and documents",
    "school and exams",
    "banking and digital payments",
    "local markets and prices",
    "public transport",
    "cooking and food at home",
    "festivals and family events",
    "weather and the seasons",
)


class DistillConfig(BaseModel):
    """FR-05 teacher distillation settings."""

    model_config = ConfigDict(extra="forbid")

    teacher: str = "claude-opus-5-5"  # a Claude model id, or `demo` for the offline demo teacher
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    seed_path: str | None = None  # English {"instruction", "response"} JSONL
    seed_licence: str = "UNSET"  # SPDX id; audited like corpus sources
    n: int = Field(default=1000, ge=1)  # generated examples across all target languages
    native_fraction: float = Field(default=0.5, ge=0.0, le=1.0)  # generated directly in-language
    pairs_per_call: int = Field(default=5, ge=1, le=20)
    min_chrf: float = Field(default=45.0, ge=0.0, le=100.0)  # back-translation agreement
    min_script_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    review_fraction: float = Field(default=0.05, ge=0.0, le=1.0)  # sent to native reviewers
    workers: int = Field(default=8, ge=1)
    topics: list[str] = Field(default_factory=lambda: list(DEFAULT_TOPICS))


class EmbedConfig(BaseModel):
    """FR-08 multilingual sentence embeddings distilled from an English teacher."""

    model_config = ConfigDict(extra="forbid")

    # `base`: mean-pooled states of the base LLM on English; otherwise a sentence-transformers id.
    teacher: str = "base"
    parallel_dir: str | None = None  # extra <dir>/<lang>.tsv files with `english<TAB>target`
    max_steps: int = Field(default=2000, ge=1)
    batch_size: int = Field(default=32, ge=1)
    lr: float = Field(default=2e-4, gt=0.0)
    max_length: int = Field(default=128, ge=8)
    heldout_fraction: float = Field(default=0.1, gt=0.0, lt=0.5)
    eval_every: int = Field(default=200, ge=1)


class TrainingRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model: str  # e.g. "meta-llama/Llama-3.2-3B" or a local directory
    languages: list[LanguageConfig] = Field(min_length=1)
    english_replay: float = Field(default=0.15, ge=0.0, lt=1.0)
    lora_rank: int = Field(default=64, ge=1)
    train_embeddings: bool = True
    embed_warmup_steps: int = Field(default=1000, ge=0)

    workdir: str = "artifacts"
    sources: str | None = None  # data source manifest (see polyglotlm.data.download)
    langid_model: str | None = None  # fastText lid.176 model; None uses script-based language ID
    toxicity_lexicon_dir: str | None = None  # <dir>/<lang>.txt, one blocked term per line
    seed: int = 0
    mix_temperature: float = Field(default=0.3, gt=0.0, le=1.0)
    lora_alpha: int | None = None  # defaults to 2 * lora_rank
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    embedding_init: Literal["fvt", "dictionary"] = "fvt"
    dictionary_dir: str | None = None  # <dir>/<lang>.tsv with `target<TAB>english` lines
    eval_tasks_dir: str | None = None  # <dir>/<lang>.jsonl benchmark items (see polyglotlm.eval)
    heldout_fraction: float = Field(default=0.02, gt=0.0, lt=0.5)
    max_english_regression: float = Field(default=3.0, ge=0.0)  # points
    cpt: StageConfig = Field(default_factory=StageConfig)
    sft: StageConfig = Field(default_factory=lambda: StageConfig(max_steps=2_000, lr=5e-5))
    distill: DistillConfig = Field(default_factory=DistillConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)

    @model_validator(mode="after")
    def _consistent(self) -> TrainingRecipe:
        codes = [lang.code for lang in self.languages]
        if len(set(codes)) != len(codes):
            raise ValueError(f"duplicate language codes: {codes}")
        ratios = [lang.cpt_mix_ratio for lang in self.languages]
        explicit = [r for r in ratios if r is not None]
        if explicit and len(explicit) != len(ratios):
            raise ValueError("set cpt_mix_ratio on every language or on none")
        if explicit and abs(sum(explicit) + self.english_replay - 1.0) > 1e-6:
            raise ValueError("cpt_mix_ratio values plus english_replay must sum to 1.0")
        return self

    @property
    def paths(self) -> Paths:
        return Paths(Path(self.workdir))

    @property
    def codes(self) -> list[str]:
        return [lang.code for lang in self.languages]

    @property
    def all_codes(self) -> list[str]:
        """Target languages plus English. Every checkpoint is evaluated on all of them."""
        return [*self.codes, ENGLISH]

    def language(self, code: str) -> LanguageConfig:
        for lang in self.languages:
            if lang.code == code:
                return lang
        raise KeyError(f"language {code!r} is not in the recipe (has: {self.codes})")

    def script_of(self, code: str) -> str:
        return ENGLISH_SCRIPT if code == ENGLISH else self.language(code).script

    def heldout_path(self, code: str) -> Path:
        if code != ENGLISH and (explicit := self.language(code).heldout_ppl_path):
            return Path(explicit)
        return self.paths.heldout(code)


class Paths:
    """Conventional artifact layout under a recipe's workdir."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    def raw(self, lang: str) -> Path:
        """Directory holding one fetched file per source of `lang`."""
        return self.workdir / "raw" / lang

    def corpus(self, lang: str) -> Path:
        return self.workdir / "corpora" / lang / "train.txt"

    def heldout(self, lang: str) -> Path:
        return self.workdir / "corpora" / lang / "heldout.txt"

    @property
    def snapshot(self) -> Path:
        return self.workdir / "corpora" / "snapshot.json"

    def candidates(self, lang: str) -> Path:
        return self.workdir / "tokenizer" / "candidates" / lang

    @property
    def extended_tokenizer(self) -> Path:
        return self.workdir / "tokenizer" / "extended"

    def fertility(self, name: str) -> Path:
        return self.workdir / "fertility" / f"{name}.json"

    def model(self, stage: str) -> Path:
        return self.workdir / "models" / stage

    def metrics(self, stage: str) -> Path:
        return self.workdir / "metrics" / f"{stage}.jsonl"

    @property
    def sft_data(self) -> Path:
        return self.workdir / "sft" / "train.jsonl"

    @property
    def sft_rejected(self) -> Path:
        return self.workdir / "sft" / "rejected.jsonl"

    def review_queue(self, lang: str) -> Path:
        return self.workdir / "sft" / "review" / f"{lang}.jsonl"

    def eval(self, name: str) -> Path:
        return self.workdir / "eval" / f"{name}.json"

    @property
    def embedder(self) -> Path:
        return self.workdir / "embed"

    @property
    def report(self) -> Path:
        return self.workdir / "report"

    @property
    def gguf(self) -> Path:
        return self.workdir / "gguf"


def load_recipe(path: str | Path) -> TrainingRecipe:
    with Path(path).open(encoding="utf-8") as fh:
        return TrainingRecipe.model_validate(yaml.safe_load(fh))


def parse_langs(langs: str | None, recipe: TrainingRecipe) -> list[str]:
    """Parse a `hi,ta` CLI value; default to every target language in the recipe."""
    if not langs:
        return recipe.codes
    codes = [code.strip() for code in langs.split(",") if code.strip()]
    for code in codes:
        if code != ENGLISH:
            recipe.language(code)  # raises KeyError for unknown codes
    return codes
