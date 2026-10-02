import json
import unicodedata

import pytest
import yaml

from polyglotlm.data.clean import QualityFilter, ToxicityFilter, normalise, redact_pii
from polyglotlm.data.dedupe import MinHashDeduper, content_hash
from polyglotlm.data.download import (
    LicenceError,
    Source,
    SourceManifest,
    audit_licences,
    fetch,
    resolve_uri,
)
from polyglotlm.data.langid import ScriptLangID
from polyglotlm.data.pipeline import build_corpora, is_heldout
from polyglotlm.demo.corpus import DEMO_LANGS, FRAMES, all_sentences, sample_documents
from tests.conftest import make_recipe

HINDI = "आज बाज़ार में बहुत भीड़ है। किसान कल शहर जाएँगे।"


def test_demo_frames_are_aligned_across_languages():
    for frame in FRAMES:
        shapes = {lang: tuple(len(values) for values in frame.slots[lang]) for lang in DEMO_LANGS}
        assert len(set(shapes.values())) == 1, shapes
    assert len({len(all_sentences(lang)) for lang in DEMO_LANGS}) == 1


def test_normalise_outputs_nfc_and_collapses_whitespace():
    messy = unicodedata.normalize("NFD", "  आज\tबाज़ार \n में  ")
    out = normalise(messy, "hi")
    assert out == unicodedata.normalize("NFC", "आज बाज़ार में")
    assert unicodedata.is_normalized("NFC", out)


def test_normalise_strips_controls_but_keeps_inner_joiners():
    assert normalise("क्‍ष‎ ‍abc﻿", "hi") == "क्‍ष abc"
    assert normalise("می‌‌خواہم", "ur") == "می‌خواہم"


def test_normalise_maps_arabic_letters_to_urdu_forms_only_for_urdu():
    assert normalise("كتاب عليـ", "ur") == "کتاب علی"
    assert normalise("كتاب", "hi") == "كتاب"


def test_redact_pii_replaces_emails_and_numbers():
    text, n = redact_pii("लिखें a.b+c@mail.example.org या +91 98765-43210 पर, कमरा 12 में।")
    assert n == 2
    assert "<email>" in text and "<phone>" in text
    assert "12" in text and "@" not in text and "98765" not in text


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (HINDI, None),
        ("आज", "too_short"),
        (f"{HINDI} http://a.example http://b.example", "urls"),
        ("The farmers will go to the market tomorrow morning.", "wrong_script"),
        ("आज 1234567890 1234567890 1234567890 है तो", "digits"),
        ("आज !!!! ???? ---- #### है तो", "symbols"),
        ("भीड़ " * 12, "repetitive"),
        ("आज " + "क" * 80 + " है तो सही", "long_word"),
    ],
)
def test_quality_filter_reasons(text, reason):
    assert QualityFilter(script="Deva").reject_reason(text.strip()) == reason


def test_quality_filter_does_not_count_pii_placeholders_as_foreign_script():
    text, _ = redact_pii("कृपया मुझे a@b.example पर लिखें या <phone> पर बात करें।")
    assert QualityFilter(script="Deva").reject_reason(text) is None


def test_toxicity_filter_uses_lexicon_and_is_a_noop_without_one(tmp_path):
    assert not ToxicityFilter.from_dir(None, "hi").is_toxic(HINDI)
    (tmp_path / "hi.txt").write_text("भीड़\n", "utf-8")
    assert ToxicityFilter.from_dir(str(tmp_path), "hi").is_toxic(HINDI)
    assert not ToxicityFilter.from_dir(str(tmp_path), "ta").is_toxic(HINDI)


def test_dedupe_drops_exact_and_near_duplicates_but_keeps_distinct_docs():
    docs = sample_documents("ta", 30, seed=1)
    deduper = MinHashDeduper(seed=0)
    assert not deduper.is_duplicate(docs[0])
    assert deduper.is_duplicate(docs[0])
    assert deduper.is_duplicate("  " + docs[0].replace(" ", "   "))
    assert deduper.is_duplicate(docs[0] + " " + docs[0].split()[0])
    assert not deduper.is_duplicate(docs[1])
    assert content_hash("A  b") == content_hash("a B")


def test_dedupe_signatures_are_reproducible():
    doc = sample_documents("bn", 1, seed=3)[0]
    first = MinHashDeduper(seed=7).signature(doc)
    assert (first == MinHashDeduper(seed=7).signature(doc)).all()
    assert (first != MinHashDeduper(seed=8).signature(doc)).any()


def test_script_langid_identifies_each_recipe_language(recipe):
    identifier = ScriptLangID.from_recipe(recipe)
    for lang in recipe.all_codes:
        detected, confidence = identifier.identify(all_sentences(lang)[0])
        assert detected == lang and confidence == 1.0
    assert identifier.identify("12345 !!") == (None, 0.0)


def test_script_langid_refuses_two_languages_in_one_script(tmp_path):
    shared = make_recipe(
        tmp_path,
        languages=[{"code": "hi", "script": "Deva"}, {"code": "mr", "script": "Deva"}],
    )
    with pytest.raises(ValueError, match="share script"):
        ScriptLangID.from_recipe(shared)


def test_licence_audit_names_every_offender():
    manifest = SourceManifest(
        sources=[
            Source(name="ok", lang="hi", uri="a", licence="CC0-1.0"),
            Source(name="nc", lang="hi", uri="b", licence="CC-BY-NC-4.0"),
            Source(name="unset", lang="ta", uri="c", licence="UNSET"),
        ]
    )
    with pytest.raises(LicenceError, match=r"nc \(CC-BY-NC-4.0\), unset \(UNSET\)"):
        audit_licences(manifest)
    audit_licences(manifest, extra_allowed=frozenset({"CC-BY-NC-4.0", "UNSET"}))


def test_fetch_verifies_sha256(tmp_path):
    raw = tmp_path / "raw.txt"
    raw.write_text("hello\n", "utf-8")
    good = fetch(Source(name="s", lang="en", uri=str(raw), licence="CC0-1.0"), tmp_path / "out")
    assert good.read_text("utf-8") == "hello\n"
    bad = Source(name="s", lang="en", uri=str(raw), licence="CC0-1.0", sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fetch(bad, tmp_path / "out")


def test_resolve_uri_expands_data_bucket(monkeypatch):
    monkeypatch.setenv("DATA_BUCKET", "/mnt/corpora/")
    assert resolve_uri("${DATA_BUCKET}/hi.txt") == "/mnt/corpora/hi.txt"
    monkeypatch.delenv("DATA_BUCKET")
    with pytest.raises(RuntimeError, match="DATA_BUCKET"):
        resolve_uri("${DATA_BUCKET}/hi.txt")


def test_heldout_split_is_deterministic():
    docs = sample_documents("hi", 400, seed=0)
    flags = [is_heldout(doc, 0.1, seed=0) for doc in docs]
    assert flags == [is_heldout(doc, 0.1, seed=0) for doc in docs]
    assert 0.04 < sum(flags) / len(flags) < 0.2


def test_build_corpora_cleans_splits_and_snapshots(recipe):
    snapshot = build_corpora(recipe, ["hi", "ta"])
    assert set(snapshot["languages"]) == {"hi", "ta", "en"}  # English always rides along
    assert {s["licence"] for s in snapshot["sources"]} == {"CC0-1.0"}

    for lang in ("hi", "ta", "en"):
        stats = snapshot["languages"][lang]
        train = recipe.paths.corpus(lang).read_text("utf-8").splitlines()
        heldout = recipe.heldout_path(lang).read_text("utf-8").splitlines()
        assert (len(train), len(heldout)) == (stats["kept"], stats["heldout"])
        assert train and heldout
        assert not set(train) & set(heldout)
        assert len(set(train)) == len(train)
        assert all(unicodedata.is_normalized("NFC", doc) for doc in train + heldout)
        assert stats["read"] == stats["kept"] + stats["heldout"] + sum(stats["dropped"].values())
        for reason in ("duplicate", "wrong_script", "too_short", "urls", "repetitive"):
            assert stats["dropped"].get(reason, 0) >= 1, (lang, reason, stats["dropped"])
        assert stats["pii_redactions"] == 2
        assert "@" not in "".join(train + heldout)

    on_disk = json.loads(recipe.paths.snapshot.read_text("utf-8"))
    assert on_disk["languages"] == snapshot["languages"]


def test_build_corpora_is_reproducible(recipe):
    first = build_corpora(recipe, recipe.codes)
    second = build_corpora(recipe, recipe.codes)
    assert first["languages"] == second["languages"]
    assert first["sources"] == second["sources"]


def test_build_corpora_refuses_unlicensed_sources(recipe):
    manifest = yaml.safe_load(open(recipe.sources, encoding="utf-8"))
    manifest["sources"][0]["licence"] = "proprietary"
    with open(recipe.sources, "w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh)
    with pytest.raises(LicenceError):
        build_corpora(recipe, recipe.codes)
    assert not recipe.paths.snapshot.exists()
