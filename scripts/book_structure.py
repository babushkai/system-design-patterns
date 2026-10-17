#!/usr/bin/env python3
"""Single source of truth for the book's structure.

`book.manifest.json` owns part/chapter order plus the localized sidebar titles.
Everything that repeats that order is derived from it here:

* VitePress sidebars and the section menu (``.vitepress/config.mts`` imports the manifest)
* the PDF/EPUB chapter list (``build-book.yml`` runs ``--chapters``)
* the README table of contents (``--write-readme``, verified by ``--check``)

Run ``python3 scripts/book_structure.py --check`` to confirm the repository agrees
with the manifest.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "book.manifest.json"
README_PATH = ROOT / "README.md"
CONFIG_PATH = ROOT / ".vitepress" / "config.mts"
BOOK_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "build-book.yml"

LOCALES = ("en", "ja")
TOC_START = "<!-- BOOK_TOC:START -->"
TOC_END = "<!-- BOOK_TOC:END -->"
MANIFEST_IMPORT = "book.manifest.json"
CHAPTERS_FLAG = "--chapters"
ARTICLE_GLOB = "[0-9][0-9]-*/*.md"
CHAPTER_HEADING_RE = re.compile(r"^# (.+?)\s*$", re.MULTILINE)
FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Chapter numbers are reading order, so they stay contiguous inside a part: when two
# chapters merge, the survivors move down and no hole is left where a chapter used to
# be. The parts below still carry holes from earlier merges; closing one means
# renumbering the part (rename files, update routes, regenerate the README table of
# contents) and deleting its entry here.
LEGACY_NUMBER_GAPS: dict[str, tuple[int, ...]] = {
    "01-foundations": (7,),
    "07-real-time": (2, 3, 4),
    "13-data-pipelines": (3,),
    "14-search-systems": (5,),
}


class ManifestError(Exception):
    """Raised when the manifest cannot be read or is shaped incorrectly."""


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_parts(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"missing manifest: {rel(path)}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{rel(path)}: invalid JSON: {exc}") from exc

    parts = document.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ManifestError(f"{rel(path)}: 'parts' must be a non-empty list")
    return parts


def title_of(entry: dict[str, Any], locale: str) -> str:
    titles = entry.get("title")
    if not isinstance(titles, dict):
        raise ManifestError(f"entry {entry.get('dir') or entry.get('route')}: 'title' must be an object")
    value = titles.get(locale)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"entry {entry.get('dir') or entry.get('route')}: missing title for '{locale}'")
    return value


def part_heading(part: dict[str, Any], locale: str) -> str:
    return f"{part['number']}. {title_of(part, locale)}"


def readme_label(part: dict[str, Any]) -> str:
    return part.get("readmeTitle") or title_of(part, "en")


def route_prefix(locale: str) -> str:
    if locale not in LOCALES:
        raise ManifestError(f"unknown locale: {locale}")
    return "/ja/" if locale == "ja" else "/"


def route_link(route: str, locale: str) -> str:
    return f"{route_prefix(locale)}{route}"


def markdown_path(route: str, locale: str = "en") -> Path:
    relative = f"{route}.md"
    return ROOT / (f"ja/{relative}" if locale == "ja" else relative)


def chapter_entries(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [chapter for part in parts for chapter in part["chapters"]]


def chapter_markdown_paths(parts: list[dict[str, Any]], locale: str = "en") -> list[str]:
    paths = []
    for chapter in chapter_entries(parts):
        relative = f"{chapter['route']}.md"
        paths.append(f"ja/{relative}" if locale == "ja" else relative)
    return paths


def sidebar(locale: str, parts: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Build a VitePress sidebar: one collapsible group per part, in print order."""

    groups = []
    for index, part in enumerate(parts or load_parts()):
        groups.append(
            {
                "text": part_heading(part, locale),
                # The first part stays open so the book opens on its foundations.
                "collapsed": index > 0,
                "items": [
                    {"text": title_of(chapter, locale), "link": route_link(chapter["route"], locale)}
                    for chapter in part["chapters"]
                ],
            }
        )
    return groups


def chapter_heading(path: Path) -> str:
    """Return a chapter's H1, which is the title the README table of contents uses."""

    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = CHAPTER_HEADING_RE.match(line)
        if match:
            return match.group(1).strip()
    raise ManifestError(f"{rel(path)}: missing '# ' chapter title for the README contents")


def render_toc(parts: list[dict[str, Any]] | None = None) -> str:
    blocks = []
    for part in parts or load_parts():
        lines = [f"### Part {part['number']}: {readme_label(part)}"]
        lines += [
            f"- [{chapter_heading(markdown_path(chapter['route']))}]({chapter['route']}.md)"
            for chapter in part["chapters"]
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_readme(readme_text: str, parts: list[dict[str, Any]] | None = None) -> str:
    start = readme_text.index(TOC_START) + len(TOC_START)
    end = readme_text.index(TOC_END)
    before = readme_text[:start]
    after = readme_text[end + len(TOC_END) :].lstrip("\n")
    return f"{before}\n\n{render_toc(parts)}\n\n{TOC_END}\n\n{after}"


def render_manifest(parts: list[dict[str, Any]], doc: str | None = None) -> str:
    """Serialize a manifest with one chapter per line so structure diffs stay readable."""

    def inline(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))

    lines = ["{"]
    if doc:
        lines.append(f"  {inline('_doc')}: {inline(doc)},")
    lines.append('  "parts": [')
    for part_index, part in enumerate(parts):
        body = [f"      {inline('number')}: {part['number']},"]
        body.append(f"      {inline('dir')}: {inline(part['dir'])},")
        body.append(f"      {inline('title')}: {inline(part['title'])},")
        for key in ("readmeTitle",):
            if part.get(key):
                body.append(f"      {inline(key)}: {inline(part[key])},")
        body.append(f"      {inline('chapters')}: [")
        chapters = list(part["chapters"])
        for chapter_index, chapter in enumerate(chapters):
            comma = "," if chapter_index < len(chapters) - 1 else ""
            entry = {"route": chapter["route"], "title": chapter["title"]}
            body.append(f"        {inline(entry)}{comma}")
        body.append("      ]")
        part_comma = "," if part_index < len(parts) - 1 else ""
        lines.append("    {")
        lines += body
        lines.append(f"    }}{part_comma}")
    lines += ["  ]", "}"]
    return "\n".join(lines) + "\n"


def corpus_paths() -> set[str]:
    paths = set()
    for path in ROOT.glob(ARTICLE_GLOB):
        paths.add(str(path.relative_to(ROOT)))
    return paths


def validate(errors: list[str], parts: list[dict[str, Any]] | None = None) -> None:
    """Assert the repository structure agrees with the manifest."""

    try:
        parts = parts if parts is not None else load_parts()
    except ManifestError as exc:
        errors.append(str(exc))
        return

    expected_corpus = corpus_paths()
    listed = chapter_markdown_paths(parts)
    seen: set[str] = set()
    numbers: list[int] = []

    for index, part in enumerate(parts):
        label = f"{rel(MANIFEST_PATH)}: part {part.get('number')}"
        numbers.append(part.get("number"))
        if part.get("number") != index + 1:
            errors.append(f"{label}: part numbers must be sequential from 1 (position {index + 1})")

        directory = part.get("dir")
        if not directory:
            errors.append(f"{label}: 'dir' is required")
            continue
        if not (ROOT / directory).is_dir():
            errors.append(f"{label}: section directory does not exist: {directory}")
        if index > 0 and directory <= parts[index - 1].get("dir", ""):
            errors.append(f"{label}: sections must be listed in ascending directory order")

        for locale in LOCALES:
            try:
                title_of(part, locale)
            except ManifestError as exc:
                errors.append(f"{rel(MANIFEST_PATH)}: {exc}")

        if not part.get("chapters"):
            errors.append(f"{label}: parts must list at least one chapter")

        routes = [chapter["route"] for chapter in part.get("chapters", [])]
        if routes != sorted(routes):
            errors.append(f"{label}: chapters must follow filename order so print and README order agree")
        if len(set(routes)) != len(routes):
            errors.append(f"{label}: duplicate chapter route")

        for route in routes:
            source = f"{route}.md"
            if Path(route).parent.as_posix() != directory:
                errors.append(f"{label}: chapter {source} must live in {directory}/")
            if source in seen:
                errors.append(f"{label}: chapter listed more than once: {source}")
            seen.add(source)
            for locale in LOCALES:
                if not markdown_path(route, locale).exists():
                    errors.append(f"{label}: missing article for route: {rel(markdown_path(route, locale))}")

    for path in sorted(expected_corpus - seen):
        errors.append(f"{rel(MANIFEST_PATH)}: chapter missing from the book: {path}")
    for path in sorted(seen - expected_corpus):
        errors.append(f"{rel(MANIFEST_PATH)}: stale chapter entry: {path}")

    validate_readme_toc(errors, parts)
    validate_chapter_numbering(errors, parts)
    validate_config(errors)
    validate_book_workflow(errors)


def validate_chapter_numbering(errors: list[str], parts: list[dict[str, Any]]) -> None:
    """Assert chapter numbers are contiguous inside every part."""

    for part in parts:
        directory = part.get("dir")
        if not directory:
            continue

        numbers: list[int] = []
        for chapter in part.get("chapters", []):
            route = chapter.get("route", "")
            match = re.match(r"^(\d{2})-", Path(route).name)
            if not match:
                errors.append(
                    f"{rel(MANIFEST_PATH)}: {route} must start with a two-digit chapter number"
                )
                continue
            numbers.append(int(match.group(1)))

        duplicated = sorted({number for number in numbers if numbers.count(number) > 1})
        if duplicated:
            errors.append(
                f"{rel(MANIFEST_PATH)}: part {part.get('number')} ({directory}) reuses chapter "
                f"numbers {', '.join(f'{n:02d}' for n in duplicated)}"
            )

        gaps = tuple(number for number in range(1, max(numbers) + 1) if numbers.count(number) != 1)
        legacy = LEGACY_NUMBER_GAPS.get(directory, ())
        if gaps == legacy:
            continue

        label = f"{rel(MANIFEST_PATH)}: part {part.get('number')} ({directory})"
        unexpected = [number for number in gaps if number not in legacy]
        closed = [number for number in legacy if number not in gaps]
        if unexpected:
            errors.append(
                f"{label}: chapter numbers skip {', '.join(f'{n:02d}' for n in unexpected)} "
                "- renumber the part so the sequence is contiguous, then regenerate the README"
            )
        if closed:
            errors.append(
                f"{label}: chapter numbers no longer skip {', '.join(f'{n:02d}' for n in closed)} "
                f"- remove that entry from LEGACY_NUMBER_GAPS in {rel(Path(__file__))}"
            )


def validate_readme_toc(errors: list[str], parts: list[dict[str, Any]]) -> None:
    if not README_PATH.exists():
        errors.append(f"{rel(README_PATH)}: missing")
        return

    text = README_PATH.read_text(encoding="utf-8")
    if TOC_START not in text or TOC_END not in text:
        errors.append(
            f"{rel(README_PATH)}: table of contents must be wrapped in "
            f"{TOC_START} and {TOC_END} (run scripts/book_structure.py --write-readme)"
        )
        return

    if render_readme(text, parts) != text:
        errors.append(
            f"{rel(README_PATH)}: table of contents is out of date "
            "(run scripts/book_structure.py --write-readme)"
        )


def validate_config(errors: list[str]) -> None:
    if not CONFIG_PATH.exists():
        errors.append(f"{rel(CONFIG_PATH)}: missing")
        return

    text = CONFIG_PATH.read_text(encoding="utf-8")
    if MANIFEST_IMPORT not in text:
        errors.append(f"{rel(CONFIG_PATH)}: must import {MANIFEST_IMPORT}")

    hardcoded = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"""["']/(?:ja/)?\d{2}-[0-9a-z-]+/[0-9a-z-]+["']""", line)
    ]
    if hardcoded:
        errors.append(
            f"{rel(CONFIG_PATH)}: chapter routes must come from {MANIFEST_IMPORT}, not be "
            f"hardcoded ({len(hardcoded)} line(s), first: {hardcoded[0]})"
        )


def validate_book_workflow(errors: list[str]) -> None:
    if not BOOK_WORKFLOW_PATH.exists():
        errors.append(f"{rel(BOOK_WORKFLOW_PATH)}: missing")
        return

    text = BOOK_WORKFLOW_PATH.read_text(encoding="utf-8")
    if CHAPTERS_FLAG not in text:
        errors.append(
            f"{rel(BOOK_WORKFLOW_PATH)}: must collect chapters with "
            f"scripts/book_structure.py {CHAPTERS_FLAG}"
        )

    hardcoded = [
        match.group(0)
        for line in text.splitlines()
        for match in re.finditer(r"""["'](?:ja/)?\d{2}-[0-9a-z-]+/[0-9a-z-]+\.md["']""", line)
    ]
    if hardcoded:
        errors.append(
            f"{rel(BOOK_WORKFLOW_PATH)}: chapter paths must come from {MANIFEST_IMPORT}, "
            f"not be hardcoded ({len(hardcoded)} entry(ies), first: {hardcoded[0]})"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chapters", action="store_true", help="print English chapter paths in print order")
    parser.add_argument("--toc", action="store_true", help="print the generated README table of contents")
    parser.add_argument("--sidebar", choices=LOCALES, help="print a VitePress sidebar as JSON")
    parser.add_argument("--write-readme", action="store_true", help="rewrite the README table of contents")
    parser.add_argument("--write-manifest", action="store_true", help="rewrite the manifest in canonical format")
    parser.add_argument("--check", action="store_true", help="verify the repository matches the manifest")
    args = parser.parse_args(argv)

    try:
        parts = load_parts()
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.chapters:
        sys.stdout.write("\n".join(chapter_markdown_paths(parts)) + "\n")
    elif args.toc:
        sys.stdout.write(render_toc(parts) + "\n")
    elif args.sidebar:
        print(json.dumps(sidebar(args.sidebar, parts), ensure_ascii=False, indent=2))
    elif args.write_readme:
        README_PATH.write_text(render_readme(README_PATH.read_text(encoding="utf-8"), parts), encoding="utf-8")
        print(f"updated {rel(README_PATH)}")
    elif args.write_manifest:
        document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        MANIFEST_PATH.write_text(render_manifest(parts, document.get("_doc")), encoding="utf-8")
        print(f"updated {rel(MANIFEST_PATH)}")
    elif args.check:
        errors: list[str] = []
        validate(errors, parts)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(
            f"Book structure matches the manifest: {len(chapter_entries(parts))} chapters "
            f"in {len(parts)} parts."
        )
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
