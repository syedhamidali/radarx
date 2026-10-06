#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Move the changelog fragments into a new version section of history.md.

Usage: ``python ci/release_changelog.py 0.5.0 [YYYY-MM-DD]``
"""

import datetime as dt
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
INCLUDE = "## Unreleased\n\n```{include} changes/unreleased.md\n```\n"


def main(version, date=None):
    date = date or dt.date.today().isoformat()
    fragments = sorted(
        (DOCS / "changes").glob("[0-9]*.md"), key=lambda p: int(p.name.split(".")[0])
    )
    if not fragments:
        sys.exit("no changelog fragments in docs/changes")
    lines = [
        line.rstrip()
        for path in fragments
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    history = DOCS / "history.md"
    text = history.read_text()
    if INCLUDE not in text:
        sys.exit("history.md has no Unreleased include block")
    section = f"## {version} ({date})\n\n" + "\n".join(lines) + "\n"
    history.write_text(text.replace(INCLUDE, INCLUDE + "\n" + section, 1))
    for path in fragments:
        path.unlink()
    print(f"moved {len(fragments)} fragments into history.md as {version}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
