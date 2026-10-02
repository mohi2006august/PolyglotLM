import json

import yaml

from polyglotlm.config import load_recipe
from polyglotlm.demo import run as demo_run
from tests.conftest import make_recipe


def test_shipped_demo_recipe_points_at_demo_inputs():
    recipe = load_recipe("recipes/tiny-demo.yaml")
    assert recipe.distill.teacher == "demo"
    assert recipe.distill.seed_path and recipe.eval_tasks_dir and recipe.embed.parallel_dir


def test_init_writes_every_input_the_recipe_needs(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(demo_run, "create_base", lambda out, **kw: built.append(out))
    recipe = make_recipe(
        tmp_path,
        eval_tasks_dir=str(tmp_path / "evals"),
        distill={"teacher": "demo", "seed_path": str(tmp_path / "seeds.jsonl")},
        embed={"parallel_dir": str(tmp_path / "parallel")},
    )
    demo_run.init_demo(recipe, docs_per_lang=30)

    manifest = yaml.safe_load(open(recipe.sources, encoding="utf-8"))
    assert {s["lang"] for s in manifest["sources"]} == {"en", "hi", "ta", "ur", "bn"}
    seeds = (tmp_path / "seeds.jsonl").read_text("utf-8").splitlines()
    assert len(seeds) == 80 and set(json.loads(seeds[0])) == {"instruction", "response"}
    assert len((tmp_path / "evals" / "ta.jsonl").read_text("utf-8").splitlines()) == 40
    assert "\t" in (tmp_path / "parallel" / "bn.tsv").read_text("utf-8").splitlines()[0]
    assert built and str(built[0]) == recipe.base_model


def test_stages_cover_the_whole_pipeline_in_order(tmp_path):
    names = [name for name, _ in demo_run.stages(make_recipe(tmp_path))]
    assert names == [
        "data",
        "fertility (base)",
        "tokenizer extend",
        "fertility (extended)",
        "surgery",
        "cpt",
        "distill",
        "sft",
        "eval",
        "report",
        "embed",
    ]
