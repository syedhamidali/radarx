#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Text and BibTeX formatting of the reference registry entries."""

import xarray as xr

from radarx import citation
from radarx._provenance import docstring_dois, read_provenance, set_provenance

ARTICLE = {
    "authors": ["Smith, T. M.", "Elmore, K. L."],
    "year": 2004,
    "title": "A title",
    "type": "article",
    "journal": "A Journal",
}


def use_registry(monkeypatch, entries):
    monkeypatch.setattr(citation, "load_registry", lambda: entries)


def test_author_without_given_name():
    assert citation._text_authors(["Madonna"]) == "Madonna"


def test_book_edition_is_listed():
    book = {**ARTICLE, "type": "book", "edition": "2nd", "publisher": "A Press"}
    text = citation._text(book)
    assert "2nd ed." in text and "A Press" in text


def test_url_is_used_when_there_is_no_doi():
    entry = {**ARTICLE, "url": "https://example.org/paper"}
    assert citation._text(entry).endswith("https://example.org/paper")


def test_bibtex_keys_stay_unique():
    used = set()
    keys = [citation._bibtex_key(ARTICLE, used) for _ in range(3)]
    assert keys == ["smith2004", "smith2004a", "smith2004b"]


def test_bibtex_of_a_proceedings_paper_has_a_booktitle():
    entry = {**ARTICLE, "type": "inproceedings", "journal": "11th Conference"}
    text = citation._bibtex(entry, set())
    assert text.startswith("@inproceedings") and "booktitle" in text
    assert "volume" not in text  # fields without a value are left out


def test_short_citation_of_an_unknown_key_is_the_key(monkeypatch):
    use_registry(monkeypatch, {})
    assert citation._short("no-such-key") == "no-such-key"


def test_short_citation_without_authors_is_the_title(monkeypatch):
    use_registry(monkeypatch, {"k": {"title": "A data set", "type": "dataset"}})
    assert citation._short("k") == "A data set"


def test_equal_short_citations_get_letters(monkeypatch):
    use_registry(monkeypatch, {"a": ARTICLE, "b": {**ARTICLE, "title": "Other"}})
    assert citation._short_list(["a", "b"]) == [
        "Smith and Elmore (2004a)",
        "Smith and Elmore (2004b)",
    ]


def test_methods_lists_a_repeated_method_once():
    ds = xr.Dataset({"a": ("x", [1.0]), "b": ("x", [2.0])})
    set_provenance(ds, "A method", [])
    ds["a"].attrs.update(ds.attrs)  # the same provenance on the data set and a variable
    assert len(read_provenance(ds)) == 2
    assert citation.methods(ds).count("Method: A method") == 1


def test_docstring_without_text_has_no_dois():
    assert docstring_dois(None) == []
    assert docstring_dois("") == []


def test_bibtex_leaves_out_an_empty_journal():
    entry = {key: value for key, value in ARTICLE.items() if key != "journal"}
    assert "journal" not in citation._bibtex(entry, set())
