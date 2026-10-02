import json
import unicodedata
from collections import Counter

import pytest
from typer.testing import CliRunner

from polyglotlm.cli import app
from polyglotlm.demo.corpus import all_sentences
from polyglotlm.tokenizer import load_tokenizer
from polyglotlm.tokenizer.fertility import (
    fertility_report,
    measure_fertility,
    render_table,
    saving,
)
from polyglotlm.tokenizer.merge_vocab import (
    extend_tokenizer,
    piece_to_text,
    select_new_tokens,
    verify_append_only,
)
from tests.conftest import TARGETS


def test_piece_to_text_maps_word_start_marker_to_a_space():
    assert piece_to_text("\u2581नमस्ते") == " नमस्ते"
    assert piece_to_text("स्ते") == "स्ते"
    assert piece_to_text("\u2581") is None
    assert piece_to_text("a\u2581b") is None


def test_select_new_tokens_ranks_in_script_pieces_by_tokens_saved(built):
    base = load_tokenizer(built.base_model)
    counts = Counter({"\u2581बाज़ार": 50, "\u2581में": 400, "hello": 999, "\u2581கடை": 999, "क": 5})
    chosen = select_new_tokens(counts, base, "hi", "Deva", top_n=10)
    texts = [c.text for c in chosen]
    assert "hello" not in texts and " கடை" not in texts  # wrong script never qualifies
    assert all(c.gain > 0 for c in chosen)
    assert [c.gain for c in chosen] == sorted((c.gain for c in chosen), reverse=True)
    assert chosen[0].gain == chosen[0].freq * (chosen[0].base_tokens - 1)
    assert len(select_new_tokens(counts, base, "hi", "Deva", top_n=1)) == 1


def test_extension_is_append_only_and_leaves_english_untouched(extended):
    base = load_tokenizer(extended.base_model)
    ext = load_tokenizer(extended.paths.extended_tokenizer)
    new = json.loads((extended.paths.extended_tokenizer / "new_tokens.json").read_text("utf-8"))
    assert len(ext) == len(base) + len(new)
    assert 0 < len(new) <= sum(lang.new_tokens for lang in extended.languages)
    verify_append_only(base, ext, expected_new=len(new))
    for sentence in all_sentences("en")[:50]:
        assert ext.encode(sentence) == base.encode(sentence)


def test_verify_append_only_detects_a_shifted_vocabulary(extended):
    base = load_tokenizer(extended.base_model)
    ext = load_tokenizer(extended.paths.extended_tokenizer)
    with pytest.raises(AssertionError):
        verify_append_only(ext, base, expected_new=0)


def test_extend_tokenizer_does_not_mutate_the_base(built):
    base = load_tokenizer(built.base_model)
    size = len(base)
    extended = extend_tokenizer(base, [])
    assert len(base) == len(extended) == size


@pytest.mark.parametrize("lang", [*TARGETS, "en"])
@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_round_trip_is_lossless_on_every_script(extended, lang, form):
    ext = load_tokenizer(extended.paths.extended_tokenizer)
    for sentence in all_sentences(lang)[::7]:
        text = unicodedata.normalize(form, sentence)
        assert ext.decode(ext.encode(text, add_special_tokens=False)) == text


def test_fertility_drops_by_half_on_heldout_text_and_english_is_unchanged(extended):
    base = fertility_report(load_tokenizer(extended.base_model), extended, extended.all_codes)
    ext = fertility_report(
        load_tokenizer(extended.paths.extended_tokenizer), extended, extended.all_codes
    )
    for lang in TARGETS:
        assert saving(base[lang], ext[lang]) <= -0.5, (lang, base[lang], ext[lang])
    assert ext["en"].tokens == base["en"].tokens
    table = render_table(base, ext)
    assert "saving" in table and "hi" in table


def test_measure_fertility_counts_whitespace_words():
    class OneTokenPerChar:
        def encode(self, text, add_special_tokens=False):
            return list(text)

    row = measure_fertility(OneTokenPerChar(), "x", ["ab cd", "", "  efg  "])
    assert (row.docs, row.words, row.tokens) == (2, 3, 8)
    assert row.tokens_per_word == pytest.approx(8 / 3)


def test_cli_reports_token_tax_before_and_after(extended, tmp_path):
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(extended.model_dump_json(), "utf-8")
    runner = CliRunner()
    before = runner.invoke(app, ["fertility", "--recipe", str(recipe_path)])
    assert before.exit_code == 0, before.output
    after = runner.invoke(
        app, ["fertility", "--recipe", str(recipe_path), "--tokenizer", "extended"]
    )
    assert after.exit_code == 0, after.output
    assert "saving" in after.output
    assert extended.paths.fertility("base").exists()
    assert extended.paths.fertility("extended").exists()
