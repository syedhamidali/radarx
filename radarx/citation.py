#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Citations
=========

Results of the :mod:`radarx.retrieve` and :mod:`radarx.grid` functions carry
the method that produced them and the DOIs of its papers (see
:mod:`radarx._provenance`). :func:`cite` turns these attributes into
citations, :func:`methods` into a short description.
"""

import re
import unicodedata

from ._provenance import METHOD, _version, load_registry, read_provenance

__all__ = ["cite", "methods"]

_STYLES = ("text", "bibtex")


def _entry(key):
    registry = load_registry()
    if key in registry:
        return registry[key]
    lower = {k.lower(): v for k, v in registry.items()}
    return lower.get(key.lower())


def _ascii(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _split_name(author):
    family, _, given = author.partition(",")
    return family.strip(), given.strip()


def _text_authors(authors):
    names = []
    for i, author in enumerate(authors):
        family, given = _split_name(author)
        if not given:
            names.append(family)
        elif i == 0:
            names.append(f"{family}, {given}")
        else:
            names.append(f"{given} {family}")
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def _sentence(text):
    return text if text[-1:] in ".?!" else text + "."


def _text(entry, version=None):
    kind = entry.get("type", "article")
    head = _text_authors(entry.get("authors", []))
    if entry.get("year"):
        head = f"{head}, {entry['year']}"
    title = entry["title"]
    if kind == "software" and version:
        title = f"{title} (version {version})"
    parts = [f"{head}: {_sentence(title)}"]
    if kind == "book":
        if entry.get("edition"):
            parts.append(f"{entry['edition']} ed.")
        if entry.get("publisher"):
            parts.append(_sentence(entry["publisher"]))
        tail = []
    elif kind in ("dataset", "software"):
        if entry.get("publisher"):
            parts.append(_sentence(entry["publisher"]))
        tail = []
    else:
        where = entry.get("journal", "")
        number = entry.get("volume", "")
        if number and entry.get("issue"):
            number = f"{number} ({entry['issue']})"
        tail = [x for x in (where, number, entry.get("pages", "")) if x]
        if tail:
            parts.append(", ".join(tail) + ("," if entry.get("doi") else "."))
    text = " ".join(parts)
    if entry.get("doi"):
        text += f" https://doi.org/{entry['doi']}"
    elif entry.get("url"):
        text += f" {entry['url']}"
    return text


_BIBTEX_ESCAPE = str.maketrans({"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_"})


def _bibtex_key(entry, used):
    family, _ = _split_name((entry.get("authors") or ["anon"])[0])
    base = re.sub(r"[^a-z]", "", _ascii(family).lower()) or "anon"
    base += str(entry.get("year", ""))
    key = base
    letter = 0
    while key in used:
        key = base + chr(ord("a") + letter)
        letter += 1
    used.add(key)
    return key


def _bibtex(entry, used, version=None):
    kind = entry.get("type", "article")
    entry_type = {
        "article": "article",
        "book": "book",
        "inproceedings": "inproceedings",
        "software": "misc",
        "dataset": "misc",
    }.get(kind, "misc")
    title = entry["title"]
    if kind == "software" and version:
        title = f"{title} (version {version})"
    fields = [
        ("author", " and ".join(entry.get("authors", []))),
        ("title", "{" + title.translate(_BIBTEX_ESCAPE) + "}"),
    ]
    journal = entry.get("journal", "")
    if entry_type == "article":
        fields.append(("journal", journal))
    elif entry_type == "inproceedings":
        fields.append(("booktitle", journal))
    if kind == "software" and version:
        fields.append(("version", str(version)))
    for name in ("volume", "issue", "pages", "edition", "publisher", "year"):
        if entry.get(name):
            fields.append(("number" if name == "issue" else name, str(entry[name])))
    if entry.get("doi"):
        fields.append(("doi", entry["doi"]))
    if entry.get("url"):
        fields.append(("url", entry["url"]))
    lines = []
    for name, value in fields:
        if not value:
            continue
        if name == "title":
            lines.append(f"  title = {{{value}}},")
        else:
            lines.append(f"  {name} = {{{value.translate(_BIBTEX_ESCAPE)}}},")
    key = _bibtex_key(entry, used)
    return f"@{entry_type}{{{key},\n" + "\n".join(lines) + "\n}"


def _version_for_citation():
    version = str(_version())
    return None if version == "999" else version


def _objects(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def _references_of(obj):
    """References and method names found on an object or a decorated function."""
    refs = []
    method = getattr(obj, "__radarx_provenance__", None)
    if method is not None:
        return list(method["references"])
    for item in read_provenance(obj):
        for ref in item["references"]:
            if ref not in refs:
                refs.append(ref)
    return refs


def cite(obj, style="text"):
    """
    Citations for the methods that produced a result.

    Reads the provenance attributes that the ``radarx.retrieve`` and
    ``radarx.grid`` functions write on their results (``radarx_references``
    on a Dataset, a DataArray and every node of a DataTree), removes
    duplicates and adds radarx itself.

    Parameters
    ----------
    obj : xarray.Dataset, xarray.DataArray, xarray.DataTree, function or list
        A result of radarx, a list of results (a processing chain), or a
        radarx function to cite before running it.
    style : {"text", "bibtex"}, optional
        Plain text in the style of the AMS journals, or one BibTeX entry per
        reference. Default ``"text"``.

    Returns
    -------
    list of str
        The citation of radarx first, then one string per reference in the
        order of first appearance. A reference that is not in the registry is
        given as its DOI link.

    Notes
    -----
    The citation of radarx uses the Zenodo concept DOI, which stands for all
    versions, and the installed radarx version. The references are listed in
    ``radarx/data/references.json``.

    Examples
    --------
    >>> mask = sweep.radarx.echo_mask()  # doctest: +SKIP
    >>> radarx.cite(mask)  # doctest: +SKIP
    """
    if style not in _STYLES:
        raise ValueError(f"style must be one of {_STYLES}, not {style!r}")
    keys = []
    for item in _objects(obj):
        for key in _references_of(item):
            if key not in keys:
                keys.append(key)
    version = _version_for_citation()
    used = set()
    out = []
    software = _entry("radarx")
    out.append(
        _bibtex(software, used, version)
        if style == "bibtex"
        else _text(software, version)
    )
    for key in keys:
        entry = _entry(key)
        if entry is None:
            out.append(
                f"@misc{{{key},\n  doi = {{{key}}}\n}}"
                if style == "bibtex"
                else f"https://doi.org/{key}"
            )
        elif style == "bibtex":
            out.append(_bibtex(entry, used))
        else:
            out.append(_text(entry))
    return out


def _short(key):
    entry = _entry(key)
    if entry is None:
        return key
    families = [_split_name(a)[0] for a in entry.get("authors", [])]
    if not families:
        return entry["title"]
    if len(families) == 1:
        who = families[0]
    elif len(families) == 2:
        who = f"{families[0]} and {families[1]}"
    else:
        who = f"{families[0]} et al."
    return f"{who} ({entry['year']})" if entry.get("year") else who


def methods(obj):
    """
    Short description of the method behind a function or a result.

    Parameters
    ----------
    obj : function or xarray.Dataset, xarray.DataArray, xarray.DataTree
        A radarx function, or a result that carries provenance attributes.

    Returns
    -------
    str
        The method name and the papers it is based on. For a function, the
        first line of its docstring comes first.
    """
    lines = []
    info = getattr(obj, "__radarx_provenance__", None)
    if info is not None:
        doc = (obj.__doc__ or "").strip().splitlines()
        lines.append(f"{obj.__name__}: {doc[0].strip()}" if doc else obj.__name__)
        lines.append(f"Method: {info['method']}")
        refs = info["references"]
        lines.append(
            "Based on: " + "; ".join(_short(r) for r in refs)
            if refs
            else "Based on: no published method"
        )
        return "\n".join(lines)
    seen = []
    for item in read_provenance(obj):
        key = (tuple(item["method"]), tuple(item["references"]))
        if key in seen:
            continue
        seen.append(key)
        lines.append("Method: " + " | ".join(item["method"]))
        refs = item["references"]
        lines.append(
            "Based on: " + "; ".join(_short(r) for r in refs)
            if refs
            else "Based on: no published method"
        )
        lines.append(f"Written by radarx {item['version']}")
    if not lines:
        lines.append(f"No radarx provenance found (no {METHOD} attribute).")
    return "\n".join(lines)
