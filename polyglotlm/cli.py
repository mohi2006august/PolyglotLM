"""`poly` command line. Heavy imports stay inside commands so `poly --help` is instant."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from polyglotlm import __version__
from polyglotlm.config import TrainingRecipe, load_recipe, parse_langs
from polyglotlm.logs import configure_logging

app = typer.Typer(no_args_is_help=True, add_completion=False, help="PolyglotLM toolkit.")
data_app = typer.Typer(no_args_is_help=True, help="Corpus building (FR-04).")
tokenizer_app = typer.Typer(no_args_is_help=True, help="Tokenizer extension (FR-01).")
train_app = typer.Typer(no_args_is_help=True, help="Continued pre-training and SFT (FR-03).")
app.add_typer(data_app, name="data")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(train_app, name="train")

RecipeOpt = Annotated[
    Path,
    typer.Option("--recipe", envvar="POLY_RECIPE", help="Training recipe YAML."),
]
LangsOpt = Annotated[
    str | None,
    typer.Option("--langs", help="Comma-separated language codes. Default: all in the recipe."),
]
DEFAULT_RECIPE = Path("recipes/indic-3b.yaml")


def _recipe(path: Path) -> TrainingRecipe:
    if not path.exists():
        raise typer.BadParameter(f"recipe not found: {path}", param_hint="--recipe")
    return load_recipe(path)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"polyglotlm {__version__}")
        raise typer.Exit


@app.callback()
def main(
    log_level: Annotated[str, typer.Option(help="DEBUG, INFO, WARNING or ERROR.")] = "INFO",
    version: Annotated[
        bool, typer.Option("--version", callback=_print_version, is_eager=True)
    ] = False,
) -> None:
    configure_logging(log_level.upper())


def _langs(value: str | None, recipe: TrainingRecipe) -> list[str]:
    try:
        return parse_langs(value, recipe)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0]), param_hint="--langs") from exc


@data_app.command("build")
def data_build(recipe: RecipeOpt = DEFAULT_RECIPE, langs: LangsOpt = None) -> None:
    """Fetch, clean, language-ID, dedupe and split corpora; write the snapshot manifest."""
    from polyglotlm.data.pipeline import build_corpora

    loaded = _recipe(recipe)
    snapshot = build_corpora(loaded, _langs(langs, loaded))
    for lang, stats in snapshot["languages"].items():
        typer.echo(
            f"{lang}: read {stats['read']}, train {stats['kept']}, heldout {stats['heldout']}, "
            f"dropped {sum(stats['dropped'].values())} {stats['dropped']}"
        )
    typer.echo(f"snapshot: {loaded.paths.snapshot}")


@app.command()
def fertility(
    recipe: RecipeOpt = DEFAULT_RECIPE,
    tokenizer: Annotated[
        str, typer.Option(help="`base`, `extended`, a local path or a Hub id.")
    ] = "base",
    langs: LangsOpt = None,
    max_docs: Annotated[
        int | None, typer.Option(help="Cap held-out documents per language.")
    ] = None,
) -> None:
    """Token tax report: tokens per word on held-out text, before and after extension."""
    from polyglotlm.tokenizer import resolve_tokenizer
    from polyglotlm.tokenizer.fertility import (
        fertility_report,
        load_report,
        render_table,
        save_report,
    )

    loaded = _recipe(recipe)
    codes = _langs(langs, loaded) if langs else loaded.all_codes
    report = fertility_report(resolve_tokenizer(tokenizer, loaded), loaded, codes, max_docs)
    name = tokenizer if tokenizer in ("base", "extended") else "custom"
    save_report(report, loaded.paths.fertility(name))
    base_path = loaded.paths.fertility("base")
    if name == "extended" and base_path.exists():
        typer.echo(render_table(load_report(base_path), report))
    else:
        typer.echo(render_table(report))


@tokenizer_app.command("extend")
def tokenizer_extend(
    recipe: RecipeOpt = DEFAULT_RECIPE,
    lang: Annotated[
        str | None, typer.Option(help="Comma-separated languages. Default: all in the recipe.")
    ] = None,
    new_tokens: Annotated[
        int | None, typer.Option(help="Override `new_tokens` for the selected languages.")
    ] = None,
) -> None:
    """Train candidate subwords, keep the top-N by tokens saved, append them to the base vocab."""
    from polyglotlm.tokenizer import load_tokenizer
    from polyglotlm.tokenizer.merge_vocab import extend_for_recipe

    loaded = _recipe(recipe)
    base = load_tokenizer(loaded.base_model)
    extended, selected = extend_for_recipe(loaded, base, _langs(lang, loaded), new_tokens)
    typer.echo(
        f"base vocab {len(base)} + {len(extended) - len(base)} new tokens "
        f"({len(selected)} selected) -> {loaded.paths.extended_tokenizer}"
    )


@app.command()
def surgery(recipe: RecipeOpt = DEFAULT_RECIPE) -> None:
    """Resize embeddings + LM head to the extended vocabulary and initialise new rows (FR-02)."""
    from polyglotlm.surgery import run_surgery

    typer.echo(f"surgery model -> {run_surgery(_recipe(recipe))}")


@train_app.command("cpt")
def train_cpt(recipe: RecipeOpt = DEFAULT_RECIPE) -> None:
    """Continued pre-training: LoRA + trainable embeddings, language mix + English replay."""
    from polyglotlm.train.cpt import run_cpt

    typer.echo(f"cpt model -> {run_cpt(_recipe(recipe))}")
