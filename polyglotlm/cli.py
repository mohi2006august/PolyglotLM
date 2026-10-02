"""`poly` command line. Heavy imports stay inside commands so `poly --help` is instant."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from polyglotlm import __version__
from polyglotlm.config import TrainingRecipe, load_recipe, parse_langs
from polyglotlm.logs import configure_logging

app = typer.Typer(no_args_is_help=True, add_completion=False, help="PolyglotLM toolkit.")

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
