#!/usr/bin/env python3
"""Keep the Japanese edition honest about how current each chapter is.

File parity alone lets the two editions drift: an English chapter can be rewritten while
its Japanese file still holds the previous version and still says it is a translation.
Every Japanese chapter therefore carries one status line directly under its title, and
this module decides which of two wordings it must be:

* current: the chapter tracks the English text section for section;
* behind: the chapter is based on an earlier English version, and says so with a link.

The decision is mechanical. A faithful translation has a predictable amount of Japanese
text per English word, a similar number of sections, and the same code blocks, so a
chapter outside those bounds is reported as behind. ``scripts/validate_docs.py`` fails when
a chapter's status line disagrees with its measurements, which means a rewrite of an
English chapter cannot merge until its translation is redone or marked as behind.

Run ``python3 scripts/translation_status.py --write`` after translating or after an
English rewrite; ``--report`` lists the chapters that are behind and why.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTICLE_GLOB = "[0-9][0-9]-*/*.md"

STATUS_MARK = "**翻訳の状態:**"
BEHIND_PHRASE = "以前の版に基づいて"
# Japanese characters per English prose word. Technical prose translated in full lands
# between roughly 1.6 and 2.8 once code and identifiers are excluded; the bounds leave room
# for terse chapters and for chapters dense with untranslated product names.
CHARS_PER_WORD = (1.3, 3.2)
SECTION_RATIO = (0.7, 1.4)
# Code is copied, not translated, so a large difference means the two files hold different
# versions of the chapter.
CODE_LINE_SLACK = 40
CODE_LINE_FACTOR = 2

FENCED_BLOCK_RE = re.compile(r"^(```|~~~)[^\n]*\n(.*?)^\1[ \t]*$", re.DOTALL | re.MULTILINE)
JAPANESE_CHAR_RE = re.compile(r"[぀-ヿ一-鿿]")
H2_RE = re.compile(r"^## ", re.MULTILINE)
LEGACY_NOTE_RE = re.compile(r"翻訳|translated from English|英語版から|英語版 `|英語原文")


def measure(text: str) -> tuple[str, int, int]:
    code_lines = sum(block.count("\n") for _, block in FENCED_BLOCK_RE.findall(text))
    prose = FENCED_BLOCK_RE.sub("", text)
    return prose, len(H2_RE.findall(prose)), code_lines


def drift_reasons(english: str, japanese: str) -> list[str]:
    """Explain why a Japanese chapter does not track its English source; empty if it does."""

    english_prose, english_sections, english_code = measure(english)
    japanese_prose, japanese_sections, japanese_code = measure(japanese)
    reasons: list[str] = []

    words = max(1, len(english_prose.split()))
    density = len(JAPANESE_CHAR_RE.findall(japanese_prose)) / words
    if not CHARS_PER_WORD[0] <= density <= CHARS_PER_WORD[1]:
        reasons.append(
            f"{density:.2f} Japanese characters per English word "
            f"(expected {CHARS_PER_WORD[0]}-{CHARS_PER_WORD[1]})"
        )

    ratio = japanese_sections / max(1, english_sections)
    if not SECTION_RATIO[0] <= ratio <= SECTION_RATIO[1]:
        reasons.append(f"{japanese_sections} sections against {english_sections} in English")

    low, high = sorted((english_code, japanese_code))
    if high - low > CODE_LINE_SLACK and high > CODE_LINE_FACTOR * max(1, low):
        reasons.append(f"{japanese_code} code lines against {english_code} in English")

    return reasons


def status_line(route: str, behind: bool) -> str:
    link = f"../../{route}.md"
    if behind:
        return (
            f"> {STATUS_MARK} この日本語版は英語版の{BEHIND_PHRASE}おり、現行の[英語版]({link})とは"
            "構成や内容が異なります。最新の内容は英語版を参照してください。"
        )
    return f"> {STATUS_MARK} 現行の[英語版]({link})からの翻訳です。"


def chapter_routes() -> list[str]:
    return sorted(str(path.relative_to(ROOT))[: -len(".md")] for path in ROOT.glob(ARTICLE_GLOB))


def read_pair(route: str) -> tuple[str, str] | None:
    english = ROOT / f"{route}.md"
    japanese = ROOT / "ja" / f"{route}.md"
    if not english.exists() or not japanese.exists():
        return None
    return english.read_text(encoding="utf-8"), japanese.read_text(encoding="utf-8")


def current_status(japanese: str) -> str | None:
    """Return 'behind', 'current', or None when the chapter has no status line."""

    for line in japanese.splitlines()[:8]:
        if STATUS_MARK in line:
            return "behind" if BEHIND_PHRASE in line else "current"
    return None


def with_status(japanese: str, route: str, behind: bool) -> str:
    lines = japanese.split("\n")
    title_index = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
    if title_index is None:
        raise ValueError(f"ja/{route}.md: missing '# ' title")

    head = lines[: title_index + 1]
    body = lines[title_index + 1 :]
    # Drop earlier translation notes, which came in a dozen wordings and all claimed
    # the chapter was a current translation.
    kept: list[str] = []
    scanning = True
    for index, line in enumerate(body):
        if scanning and index < 7:
            if not line.strip():
                continue
            if line.startswith(">") and (STATUS_MARK in line or LEGACY_NOTE_RE.search(line)):
                continue
        scanning = False
        kept.append(line)
    return "\n".join([*head, "", status_line(route, behind), "", *kept])


def validate(errors: list[str]) -> None:
    for route in chapter_routes():
        pair = read_pair(route)
        if pair is None:
            continue
        english, japanese = pair
        reasons = drift_reasons(english, japanese)
        status = current_status(japanese)
        path = f"ja/{route}.md"
        if status is None:
            errors.append(
                f"{path}: missing translation status line under the title "
                "(run scripts/translation_status.py --write)"
            )
        elif reasons and status != "behind":
            errors.append(
                f"{path}: no longer tracks the English chapter ({'; '.join(reasons)}). "
                "Retranslate it, or run scripts/translation_status.py --write to mark it as behind"
            )
        elif not reasons and status == "behind":
            errors.append(
                f"{path}: tracks the English chapter again but is still marked as behind "
                "(run scripts/translation_status.py --write)"
            )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="rewrite every chapter's status line")
    group.add_argument("--report", action="store_true", help="list chapters that are behind, with reasons")
    group.add_argument("--check", action="store_true", help="fail when a status line is wrong")
    args = parser.parse_args(argv)

    if args.check:
        errors: list[str] = []
        validate(errors)
        for error in errors:
            print(error, file=sys.stderr)
        return 1 if errors else 0

    behind = 0
    routes = chapter_routes()
    for route in routes:
        pair = read_pair(route)
        if pair is None:
            continue
        english, japanese = pair
        reasons = drift_reasons(english, japanese)
        behind += bool(reasons)
        if args.report and reasons:
            print(f"ja/{route}.md: {'; '.join(reasons)}")
        if args.write:
            updated = with_status(japanese, route, bool(reasons))
            if updated != japanese:
                (ROOT / "ja" / f"{route}.md").write_text(updated, encoding="utf-8")
    print(f"{behind} of {len(routes)} Japanese chapters are behind their English source.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
