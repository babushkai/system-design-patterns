#!/usr/bin/env python3
"""Validate properties that are visible only after VitePress renders the book."""

from __future__ import annotations

import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / ".vitepress" / "dist"
ARTICLE_RE = re.compile(r"^(?:ja/)?[0-9][0-9]-[^/]+/[^/]+\.md$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE_RE = re.compile(r"`+[^`\n]*`+")
DISPLAY_MATH_RE = re.compile(r"(?<!\\)\$\$")
INLINE_MATH_RE = re.compile(r"(?<![$\\])\$[^$\n]+?(?<!\\)\$(?!\$)")


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []
        self.ids: list[str] = []
        self.math_errors = 0
        self.math_containers = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        element_id = values.get("id")
        if element_id:
            self.ids.append(element_id)

        if tag == "a" and values.get("href"):
            self.hrefs.append(values["href"] or "")

        classes = set((values.get("class") or "").split())
        if tag == "mjx-container" and "MathJax" in classes:
            self.math_containers += 1
        if tag == "mjx-merror" or "mjx-merror" in classes:
            self.math_errors += 1


def strip_fenced_and_inline_code(text: str) -> str:
    lines: list[str] = []
    fence = ""
    for line in text.splitlines():
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if not fence:
                fence = marker
            elif marker == fence:
                fence = ""
            lines.append("")
        else:
            lines.append("" if fence else INLINE_CODE_RE.sub("", line))
    return "\n".join(lines)


def source_has_math(path: Path) -> bool:
    prose = strip_fenced_and_inline_code(path.read_text(encoding="utf-8"))
    return bool(DISPLAY_MATH_RE.search(prose) or INLINE_MATH_RE.search(prose))


def output_for_source(path: Path) -> Path:
    relative = path.relative_to(ROOT).with_suffix(".html")
    return DIST / relative


def resolve_html_target(source: Path, raw_href: str) -> tuple[Path, str] | None:
    parsed = urlsplit(raw_href)
    if parsed.scheme or parsed.netloc:
        return None

    path = unquote(parsed.path)
    fragment = unquote(parsed.fragment)
    if not path:
        return source, fragment

    # Only page links have heading fragments worth validating. Downloads, feeds,
    # and static assets are owned by their publishing steps.
    suffix = PurePosixPath(path).suffix
    if suffix and suffix not in {".html"}:
        return None

    if path.startswith("/"):
        relative = PurePosixPath(path.lstrip("/"))
    else:
        source_dir = PurePosixPath(source.relative_to(DIST).parent.as_posix())
        relative = source_dir / PurePosixPath(path)

    parts: list[str] = []
    for part in relative.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)

    normalized = PurePosixPath(*parts)
    if not parts:
        normalized = PurePosixPath("index.html")
    elif path.endswith("/"):
        normalized /= "index.html"
    elif not normalized.suffix:
        normalized = normalized.with_suffix(".html")
    return DIST / Path(normalized.as_posix()), fragment


def main() -> int:
    errors: list[str] = []
    if not DIST.exists():
        print(f"built site does not exist: {DIST}", file=sys.stderr)
        return 1

    pages: dict[Path, PageParser] = {}
    for path in sorted(DIST.rglob("*.html")):
        parser = PageParser()
        parser.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()] = parser

        duplicates = sorted(
            element_id for element_id, count in Counter(parser.ids).items() if count > 1
        )
        if duplicates:
            shown = ", ".join(duplicates[:5])
            errors.append(
                f"{path.relative_to(DIST)}: duplicate rendered HTML id(s): {shown}"
            )
        if parser.math_errors:
            errors.append(
                f"{path.relative_to(DIST)}: rendered {parser.math_errors} MathJax error node(s)"
            )

    for source in sorted(ROOT.rglob("*.md")):
        relative = source.relative_to(ROOT).as_posix()
        if not ARTICLE_RE.match(relative) or not source_has_math(source):
            continue
        output = output_for_source(source).resolve()
        parser = pages.get(output)
        if not parser:
            errors.append(f"{relative}: expected rendered page is missing")
        elif not parser.math_containers:
            errors.append(f"{relative}: source math produced no MathJax container")

    checked_links: set[tuple[Path, str, Path, str]] = set()
    for source, parser in pages.items():
        for href in parser.hrefs:
            resolved = resolve_html_target(source, href)
            if not resolved:
                continue
            target, fragment = resolved
            target = target.resolve()
            key = (source, href, target, fragment)
            if key in checked_links:
                continue
            checked_links.add(key)

            target_parser = pages.get(target)
            if not target_parser:
                errors.append(
                    f"{source.relative_to(DIST)}: rendered link target is missing: {href}"
                )
                continue
            if fragment and fragment not in set(target_parser.ids):
                errors.append(
                    f"{source.relative_to(DIST)}: rendered anchor is missing: {href}"
                )

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    math_pages = sum(1 for parser in pages.values() if parser.math_containers)
    print(
        f"Built-site validation passed: {len(pages)} HTML pages, "
        + f"{math_pages} pages with rendered math."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
