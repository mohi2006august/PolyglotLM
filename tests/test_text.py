import unicodedata

from polyglotlm.text import (
    char_script,
    dominant_script,
    in_script,
    nfc,
    script_ratio,
    split_words,
)


def test_nfc_composes_decomposed_input():
    decomposed = unicodedata.normalize("NFD", "क़िला é")
    assert nfc(decomposed) == unicodedata.normalize("NFC", "क़िला é")


def test_char_script_covers_target_scripts():
    assert char_script("क") == "Deva"
    assert char_script("த") == "Taml"
    assert char_script("ب") == "Arab"
    assert char_script("ব") == "Beng"
    assert char_script("a") == "Latn"
    assert char_script("1") is None


def test_script_ratio_ignores_digits_and_punctuation():
    assert script_ratio("नमस्ते, 123!", "Deva") == 1.0
    assert script_ratio("नमस्ते hello", "Deva") == 6 / 11
    assert script_ratio("123 !!", "Deva") == 0.0


def test_dominant_script():
    assert dominant_script("வணக்கம் world") == "Taml"
    assert dominant_script("...") is None


def test_in_script_allows_joiners_but_not_other_scripts():
    assert in_script("क्‍ष", "Deva")
    assert not in_script("कa", "Deva")
    assert not in_script("।", "Deva")  # danda is punctuation, not a letter
    assert not in_script("‍", "Deva")


def test_split_words_is_whitespace_based():
    assert split_words("  আমি  ভাত খাই ") == ["আমি", "ভাত", "খাই"]
