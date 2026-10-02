"""FR-02 model surgery: resize the vocabulary and initialise the new rows."""

from __future__ import annotations

import json
from pathlib import Path

from polyglotlm.config import TrainingRecipe
from polyglotlm.logs import get_logger
from polyglotlm.modeling import load_model, save_model
from polyglotlm.surgery.init_embeddings import init_new_embeddings, load_dictionary
from polyglotlm.surgery.resize import resize_embeddings
from polyglotlm.tokenizer import load_tokenizer

log = get_logger(__name__)

SURGERY = "surgery"


def _recipe_dictionary(recipe: TrainingRecipe) -> dict[str, str] | None:
    if recipe.embedding_init != "dictionary":
        return None
    if not recipe.dictionary_dir:
        raise ValueError("embedding_init is `dictionary` but the recipe has no `dictionary_dir`")
    merged: dict[str, str] = {}
    for code in recipe.codes:
        path = Path(recipe.dictionary_dir) / f"{code}.tsv"
        if path.exists():
            merged.update(load_dictionary(path))
        else:
            log.warning("dictionary_missing", lang=code, path=str(path))
    return merged


def run_surgery(recipe: TrainingRecipe) -> Path:
    """Base model + extended tokenizer -> resized, initialised model saved with its tokenizer."""
    base_tokenizer = load_tokenizer(recipe.base_model)
    extended_tokenizer = load_tokenizer(recipe.paths.extended_tokenizer)
    model = load_model(recipe.base_model)
    old_size, new_size = resize_embeddings(model, len(extended_tokenizer))
    if old_size != len(base_tokenizer):
        raise ValueError(
            f"base model has {old_size} embedding rows but its tokenizer has {len(base_tokenizer)} "
            "tokens; new-token IDs would not line up with the new rows"
        )
    report = init_new_embeddings(
        model, base_tokenizer, extended_tokenizer, _recipe_dictionary(recipe)
    )
    out_dir = recipe.paths.model(SURGERY)
    save_model(model, extended_tokenizer, out_dir)
    summary = {
        "base_model": recipe.base_model,
        "old_vocab": old_size,
        "new_vocab": new_size,
        "method": recipe.embedding_init,
        **report.__dict__,
    }
    (out_dir / "surgery.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("surgery_done", path=str(out_dir), **summary)
    return out_dir
