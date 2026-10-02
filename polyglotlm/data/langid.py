"""Language identification.

Two backends behind one protocol:

* `ScriptLangID`: offline, dependency-free. Maps the dominant Unicode script to the recipe
  language written in it. It cannot separate languages that share a script (Hindi/Marathi,
  Urdu/Persian, Bengali/Assamese), so a recipe with two such languages must use fastText.
* `FastTextLangID`: wraps a fastText `lid.176` model when the optional `fasttext` package and a
  model file are available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from polyglotlm.config import ENGLISH, ENGLISH_SCRIPT, TrainingRecipe
from polyglotlm.logs import get_logger
from polyglotlm.text import script_counts

log = get_logger(__name__)


class LangIdentifier(Protocol):
    def identify(self, text: str) -> tuple[str | None, float]:
        """Return (language code or None, confidence in [0, 1])."""
        ...


class ScriptLangID:
    def __init__(self, script_to_lang: dict[str, str]) -> None:
        self.script_to_lang = script_to_lang

    @classmethod
    def from_recipe(cls, recipe: TrainingRecipe) -> ScriptLangID:
        mapping = {ENGLISH_SCRIPT: ENGLISH}
        for lang in recipe.languages:
            if lang.script in mapping:
                raise ValueError(
                    f"{lang.code!r} and {mapping[lang.script]!r} share script {lang.script}; "
                    "script-based language ID cannot separate them. Set `langid_model` in the "
                    "recipe to a fastText lid.176 model."
                )
            mapping[lang.script] = lang.code
        return cls(mapping)

    def identify(self, text: str) -> tuple[str | None, float]:
        counts = script_counts(text)
        total = sum(counts.values())
        if not total:
            return None, 0.0
        script, n = counts.most_common(1)[0]
        return self.script_to_lang.get(script), n / total


class FastTextLangID:
    def __init__(self, model_path: str | Path) -> None:
        try:
            import fasttext  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "fastText language ID requested but the `fasttext` package is not installed"
            ) from exc
        self._model: Any = fasttext.load_model(str(model_path))

    def identify(self, text: str) -> tuple[str | None, float]:
        labels, probs = self._model.predict(text.replace("\n", " "), k=1)
        if not labels:
            return None, 0.0
        return labels[0].removeprefix("__label__"), float(probs[0])


def build_identifier(recipe: TrainingRecipe) -> LangIdentifier:
    if recipe.langid_model:
        log.info("langid_backend", backend="fasttext", model=recipe.langid_model)
        return FastTextLangID(recipe.langid_model)
    log.info("langid_backend", backend="script")
    return ScriptLangID.from_recipe(recipe)
