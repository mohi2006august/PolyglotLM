"""`poly` command line. Heavy imports stay inside commands so `poly --help` is instant."""

from __future__ import annotations

import json
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
distill_app = typer.Typer(no_args_is_help=True, help="Teacher distillation (FR-05).")
app.add_typer(data_app, name="data")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(train_app, name="train")
app.add_typer(distill_app, name="distill")

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


@distill_app.command("sft")
def distill_sft_cmd(
    recipe: RecipeOpt = DEFAULT_RECIPE,
    teacher: Annotated[
        str | None, typer.Option(help="Claude model id, or `demo`. Default: from the recipe.")
    ] = None,
    n: Annotated[int | None, typer.Option(help="Examples to generate across languages.")] = None,
    langs: LangsOpt = None,
) -> None:
    """Teacher-translated and natively generated instruction data with back-translation QA."""
    from polyglotlm.distill.teacher import build_teacher
    from polyglotlm.distill.teacher_generate import distill_sft, summary

    loaded = _recipe(recipe)
    chosen = build_teacher(teacher or loaded.distill.teacher, loaded.distill.effort)
    report = distill_sft(loaded, chosen, n, _langs(langs, loaded) if langs else None)
    typer.echo(json.dumps(summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"sft data -> {loaded.paths.sft_data}")


@distill_app.command("review")
def distill_review(recipe: RecipeOpt = DEFAULT_RECIPE) -> None:
    """Fold native-speaker review decisions (sft/review/*.jsonl) back into the SFT set."""
    from polyglotlm.distill.review import apply_reviews

    typer.echo(json.dumps(dict(apply_reviews(_recipe(recipe)))))


@train_app.command("sft")
def train_sft(recipe: RecipeOpt = DEFAULT_RECIPE) -> None:
    """Instruction tuning on the distilled set, early-stopped on held-out native text."""
    from polyglotlm.train.sft import run_sft

    typer.echo(f"sft model -> {run_sft(_recipe(recipe))}")


@app.command("eval")
def eval_cmd(
    recipe: RecipeOpt = DEFAULT_RECIPE,
    all_checkpoints: Annotated[
        bool, typer.Option("--all", help="Evaluate the base model and every trained stage.")
    ] = False,
    model: Annotated[str | None, typer.Option(help="A single checkpoint to evaluate.")] = None,
    name: Annotated[str, typer.Option(help="Result name for --model.")] = "custom",
    max_docs: Annotated[int, typer.Option(help="Held-out documents per language.")] = 200,
    max_items: Annotated[int | None, typer.Option(help="Benchmark items per language.")] = None,
) -> None:
    """Score checkpoints on every language: bits/char, token perplexity, benchmark tasks."""
    from polyglotlm.eval.harness import checkpoints, evaluate_checkpoint

    loaded = _recipe(recipe)
    if all_checkpoints == (model is not None):
        raise typer.BadParameter("pass exactly one of --all or --model")
    targets = checkpoints(loaded) if all_checkpoints else {name: str(model)}
    for target, path in targets.items():
        result = evaluate_checkpoint(loaded, target, path, max_docs, max_items)
        cells = [
            f"{lang}={scores['score']:.1f}"
            if scores.get("score") is not None
            else f"{lang}=bpc {scores['bits_per_char']:.3f}"
            for lang, scores in result["languages"].items()
        ]
        typer.echo(f"{target:<6} " + "  ".join(cells))


@app.command()
def report(
    recipe: RecipeOpt = DEFAULT_RECIPE,
    fail_on_regression: Annotated[
        bool, typer.Option(help="Exit 1 when English regresses beyond the recipe budget.")
    ] = False,
) -> None:
    """Render the report card (HTML) and the model card (Markdown)."""
    from polyglotlm.eval.report import english_regressed, write_report

    loaded = _recipe(recipe)
    typer.echo(f"report -> {write_report(loaded)}")
    if fail_on_regression and english_regressed(loaded):
        typer.echo("English regression beyond budget", err=True)
        raise typer.Exit(1)
