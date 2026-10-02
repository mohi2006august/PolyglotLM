from pathlib import Path

import pytest
from pydantic import ValidationError

from polyglotlm.config import LanguageConfig, TrainingRecipe, load_recipe, parse_langs

RECIPES = Path(__file__).parent.parent / "recipes"


def _recipe(**overrides) -> TrainingRecipe:
    base = {
        "base_model": "base",
        "languages": [{"code": "hi", "script": "Deva"}, {"code": "ta", "script": "Taml"}],
    }
    return TrainingRecipe.model_validate({**base, **overrides})


@pytest.mark.parametrize("name", ["indic-3b.yaml", "tiny-demo.yaml"])
def test_shipped_recipes_load(name):
    recipe = load_recipe(RECIPES / name)
    assert recipe.codes == ["hi", "ta", "ur", "bn"]
    assert recipe.all_codes[-1] == "en"


def test_defaults_match_the_spec():
    recipe = _recipe()
    assert recipe.english_replay == 0.15
    assert recipe.lora_rank == 64
    assert recipe.train_embeddings is True
    assert recipe.embed_warmup_steps == 1000


def test_unknown_script_is_rejected():
    with pytest.raises(ValidationError, match="unknown script"):
        LanguageConfig(code="xx", script="Xxxx")


def test_english_cannot_be_a_target():
    with pytest.raises(ValidationError, match="replay language"):
        LanguageConfig(code="en", script="Latn")


def test_duplicate_codes_are_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        _recipe(languages=[{"code": "hi", "script": "Deva"}, {"code": "hi", "script": "Deva"}])


def test_explicit_mix_ratios_must_sum_with_replay():
    ok = _recipe(
        languages=[
            {"code": "hi", "script": "Deva", "cpt_mix_ratio": 0.5},
            {"code": "ta", "script": "Taml", "cpt_mix_ratio": 0.35},
        ]
    )
    assert ok.language("ta").cpt_mix_ratio == 0.35
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        _recipe(
            languages=[
                {"code": "hi", "script": "Deva", "cpt_mix_ratio": 0.5},
                {"code": "ta", "script": "Taml", "cpt_mix_ratio": 0.5},
            ]
        )
    with pytest.raises(ValidationError, match="every language or on none"):
        _recipe(
            languages=[
                {"code": "hi", "script": "Deva", "cpt_mix_ratio": 0.5},
                {"code": "ta", "script": "Taml"},
            ]
        )


def test_heldout_path_prefers_explicit_value():
    recipe = _recipe(
        workdir="w",
        languages=[
            {"code": "hi", "script": "Deva", "heldout_ppl_path": "custom/hi.txt"},
            {"code": "ta", "script": "Taml"},
        ],
    )
    assert recipe.heldout_path("hi") == Path("custom/hi.txt")
    assert recipe.heldout_path("ta") == Path("w/corpora/ta/heldout.txt")
    assert recipe.heldout_path("en") == Path("w/corpora/en/heldout.txt")


def test_parse_langs():
    recipe = _recipe()
    assert parse_langs(None, recipe) == ["hi", "ta"]
    assert parse_langs("ta, en", recipe) == ["ta", "en"]
    with pytest.raises(KeyError):
        parse_langs("xx", recipe)
