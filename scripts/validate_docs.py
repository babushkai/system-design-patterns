#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {
    ".git",
    ".next",
    ".vitepress",
    "build",
    "dist",
    "node_modules",
    "public",
    "site",
    "website",
}

MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)")
SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
ARTICLE_COUNT_RE = re.compile(r"(\d+)\s+articles|(\d+)記事")
STAT_COUNT_RE = re.compile(
    r'<span class="(?:home-)?stat-(?:num|number)">(\d+)</span>\s*'
    + r'<span class="(?:home-)?stat-label">(Articles|記事)</span>',
    re.DOTALL,
)
BOOK_CHAPTER_RE = re.compile(r"""["']((?:ja/)?\d{2}-[^"']+\.md)["']""")
VITEPRESS_LINK_RE = re.compile(r"""link:\s*["']([^"']+)["']""")
VITEPRESS_HREF_RE = re.compile(r"""href:\s*["']([^"']+)["']""")
VITEPRESS_ASSET_RE = re.compile(r"""["'](/(?:icons/[^"'\s]+|logo)\.svg)["']""")
VITEPRESS_BASE_RE = re.compile(r"""base:\s*["']([^"']+)["']""")
HTML_ROOT_HREF_RE = re.compile(r"""\shref=["'](/[^"'#?]*)["']""")
LANDING_STAT_RE = re.compile(
    r"""\[\s*["'](\d+)["']\s*,\s*["'](Articles|記事)["']\s*\]"""
)
LEGACY_MATH_DELIMITER_RE = re.compile(r"\\[()\[\]]")
INLINE_CODE_RE = re.compile(r"`+[^`\n]*`+")
ARTICLE_ROUTE_RE = re.compile(r"^(?:ja/)?[0-9][0-9]-[^/]+/[^/]+$")
DOUBLE_DOLLAR_RE = re.compile(r"(?<!\\)\$\$")
SINGLE_DOLLAR_RE = re.compile(r"(?<!\$)\$(?!\$)")
INLINE_DOLLAR_MATH_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
VITEPRESS_SOURCE_FILES = [
    Path(".vitepress/config.mts"),
    Path(".vitepress/theme/components/LandingPage.vue"),
]
HOMEPAGE_SOURCES = [
    Path("index.md"),
    Path("ja/index.md"),
    Path(".vitepress/theme/components/LandingPage.vue"),
]

DEEP_SYSTEM_SECTIONS = [
    Path("16-ml-systems"),
    Path("17-llm-systems"),
]
DEEP_SYSTEM_REQUIRED_HEADINGS = [
    re.compile(r"^## TL;DR$", re.MULTILINE),
    re.compile(r"^## Failure Modes(?:\b|:)", re.MULTILINE),
    re.compile(r"^## Decision Framework(?:\b|:)", re.MULTILINE),
    re.compile(r"^## Key Takeaways$", re.MULTILINE),
    re.compile(r"^## References$", re.MULTILINE),
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*.md"):
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        files.append(path)
    return sorted(files)


def strip_fenced_code(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    fence = ""

    for line in text.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if not in_fence:
                in_fence = True
                fence = marker
            elif marker == fence:
                in_fence = False
                fence = ""
            lines.append("")
            continue
        lines.append("" if in_fence else line)

    return "\n".join(lines)


def target_from_link(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split()[0] if target.split() else ""
    return target.strip()


def is_external_or_anchor(target: str) -> bool:
    return (
        not target
        or target.startswith("#")
        or target.startswith("mailto:")
        or bool(SCHEME_RE.match(target))
    )


def local_candidates(source: Path, target: str) -> list[Path]:
    target = unquote(target).split("#", 1)[0].split("?", 1)[0]
    if not target:
        return []

    base = (
        ROOT / target.lstrip("/") if target.startswith("/") else source.parent / target
    )
    candidates = [base]

    if base.suffix == "":
        candidates.append(base.with_suffix(".md"))
        candidates.append(base / "index.md")

    return candidates


def validate_markdown_links(errors: list[str]) -> None:
    for path in iter_markdown_files():
        text = strip_fenced_code(path.read_text(encoding="utf-8"))
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in MARKDOWN_LINK_RE.finditer(line):
                target = target_from_link(match.group(1))
                if is_external_or_anchor(target):
                    continue

                candidates = local_candidates(path, target)
                if candidates and not any(
                    candidate.exists() for candidate in candidates
                ):
                    errors.append(
                        f"{rel(path)}:{lineno}: missing local link target: {target}"
                    )


def article_paths(prefix: str = "") -> set[Path]:
    base = ROOT / prefix
    return {
        path.relative_to(base)
        for path in base.glob("[0-9][0-9]-*/*.md")
        if path.is_file()
    }


def validate_article_parity(errors: list[str]) -> int:
    english = article_paths()
    japanese = article_paths("ja")

    if english != japanese:
        for path in sorted(english - japanese):
            errors.append(f"ja/{path}: missing Japanese article")
        for path in sorted(japanese - english):
            errors.append(f"{path}: missing English article")

    if len(english) != len(japanese):
        errors.append(f"article count mismatch: en={len(english)} ja={len(japanese)}")

    return len(english)


def validate_advertised_counts(errors: list[str], expected: int) -> None:
    files = [ROOT / path for path in HOMEPAGE_SOURCES]

    for path in files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for match in ARTICLE_COUNT_RE.finditer(text):
            value = int(match.group(1) or match.group(2))
            if value != expected:
                errors.append(
                    f"{rel(path)}: advertised article count is {value}, expected {expected}"
                )
        for match in STAT_COUNT_RE.finditer(text):
            value = int(match.group(1))
            if value != expected:
                errors.append(
                    f"{rel(path)}: stats article count is {value}, expected {expected}"
                )
        for match in LANDING_STAT_RE.finditer(text):
            value = int(match.group(1))
            if value != expected:
                errors.append(
                    f"{rel(path)}: stats article count is {value}, expected {expected}"
                )


def validate_frontmatter(errors: list[str]) -> None:
    for path in iter_markdown_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0].strip() != "---":
            continue
        if not any(line.strip() == "---" for line in lines[1:]):
            errors.append(f"{rel(path)}: frontmatter is missing a closing delimiter")


def validate_deep_system_chapters(errors: list[str]) -> None:
    """Keep the ML/LLM fieldbook chapters structurally consistent.

    This intentionally checks only the durable reader contract. Topic-specific middle
    sections remain free-form because a feature-store chapter and a GPU-inference
    chapter should not be forced into an identical internal outline.
    """

    for section in DEEP_SYSTEM_SECTIONS:
        for path in sorted((ROOT / section).glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            prose = strip_fenced_code(raw)
            positions: list[int] = []
            for heading in DEEP_SYSTEM_REQUIRED_HEADINGS:
                match = heading.search(prose)
                if not match:
                    errors.append(
                        f"{rel(path)}: missing required heading matching {heading.pattern}"
                    )
                    break
                positions.append(match.start())

            if positions and positions != sorted(positions):
                errors.append(
                    f"{rel(path)}: required deep-system headings are out of order"
                )


def normalize_heading(value: str) -> str:
    """Normalize cosmetic Markdown differences before comparing headings."""

    value = re.sub(r"[`*_]", "", value)
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return value.rstrip(":")


def validate_unique_h2_headings(errors: list[str]) -> None:
    """Reject appended rewrites that introduce the same chapter section twice.

    Repeated H2 headings were the clearest mechanical signal of chapters that had a
    shallow tutorial followed by a second "deep dive" over the same material. H3
    headings remain unrestricted because recurring substructures such as failure
    traces can legitimately appear under different mechanisms.
    """

    for path in iter_markdown_files():
        prose = strip_fenced_code(path.read_text(encoding="utf-8"))
        first_seen: dict[str, int] = {}
        for lineno, line in enumerate(prose.splitlines(), start=1):
            match = re.match(r"^##\s+(.+?)\s*$", line)
            if not match:
                continue
            heading = normalize_heading(match.group(1))
            if heading in first_seen:
                errors.append(
                    f"{rel(path)}:{lineno}: duplicate H2 heading "
                    + f"(first at line {first_seen[heading]}): {match.group(1)}"
                )
            else:
                first_seen[heading] = lineno


def validate_unique_chapter_titles(errors: list[str]) -> None:
    """Require one distinct H1 for every chapter in each language."""

    for prefix in ("", "ja"):
        seen: dict[str, Path] = {}
        base = ROOT / prefix
        for relative in sorted(article_paths(prefix)):
            path = base / relative
            prose = strip_fenced_code(path.read_text(encoding="utf-8"))
            titles = re.findall(r"^#\s+(.+?)\s*$", prose, flags=re.MULTILINE)
            if len(titles) != 1:
                errors.append(
                    f"{rel(path)}: expected exactly one chapter H1, found {len(titles)}"
                )
                continue
            title = normalize_heading(titles[0])
            previous = seen.get(title)
            if previous:
                errors.append(
                    f"{rel(path)}: duplicate chapter title also used by {rel(previous)}: "
                    + titles[0]
                )
            else:
                seen[title] = path


def normalize_substantial_prose(value: str) -> str:
    """Normalize prose while preserving enough wording to avoid false matches."""

    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[`*_~]", "", value)
    return re.sub(r"[\W_]+", " ", value.casefold(), flags=re.UNICODE).strip()


def validate_unique_substantial_prose(errors: list[str]) -> None:
    """Reject copied explanatory paragraphs across the English book.

    Repeated short phrases, tables, quotations, and code are intentionally ignored.
    Lists remain in scope because copied checklists were a source of chapter filler.
    A 240-character normalized threshold catches boilerplate-sized prose without
    forcing independent chapters to avoid ordinary technical vocabulary.
    """

    first_seen: dict[str, tuple[Path, int]] = {}
    paths = [ROOT / path for path in sorted(article_paths())]
    paths += [ROOT / "ja" / path for path in sorted(article_paths("ja"))]
    for path in paths:
        prose = strip_fenced_code(path.read_text(encoding="utf-8"))
        paragraphs: list[tuple[int, str]] = []
        paragraph_lines: list[str] = []
        paragraph_start = 1
        for lineno, line in enumerate([*prose.splitlines(), ""], start=1):
            if line.strip():
                if not paragraph_lines:
                    paragraph_start = lineno
                paragraph_lines.append(line)
                continue
            if paragraph_lines:
                paragraphs.append((paragraph_start, "\n".join(paragraph_lines)))
                paragraph_lines = []

        for lineno, paragraph in paragraphs:
            stripped = paragraph.lstrip()
            if not stripped or re.match(r"^(?:#|>|\||:::)", stripped):
                continue

            normalized = normalize_substantial_prose(paragraph)
            if len(normalized) < 240:
                continue

            previous = first_seen.get(normalized)
            if previous:
                previous_path, previous_line = previous
                errors.append(
                    f"{rel(path)}:{lineno}: substantial prose duplicates "
                    + f"{rel(previous_path)}:{previous_line}"
                )
            else:
                first_seen[normalized] = (path, lineno)


def validate_portable_math_delimiters(errors: list[str]) -> None:
    """Keep equations portable across VitePress, GitHub, and Pandoc."""

    for path in iter_markdown_files():
        prose = strip_fenced_code(path.read_text(encoding="utf-8"))
        for lineno, line in enumerate(prose.splitlines(), start=1):
            line_without_code = INLINE_CODE_RE.sub("", line)
            match = LEGACY_MATH_DELIMITER_RE.search(line_without_code)
            if match:
                errors.append(
                    f"{rel(path)}:{lineno}: use $...$ or $$...$$ math delimiters "
                    + f"instead of {match.group(0)}"
                )


def validate_dollar_math_delimiters(errors: list[str]) -> None:
    """Catch prose currency that MathJax/Pandoc would parse as an equation."""

    paths = [ROOT / path for path in sorted(article_paths())]
    paths += [ROOT / "ja" / path for path in sorted(article_paths("ja"))]

    for path in paths:
        prose = strip_fenced_code(path.read_text(encoding="utf-8"))
        in_display_math = False
        display_start = 0

        for lineno, line in enumerate(prose.splitlines(), start=1):
            clean = INLINE_CODE_RE.sub("", line)
            display_count = len(DOUBLE_DOLLAR_RE.findall(clean))

            if in_display_math:
                if display_count % 2:
                    in_display_math = False
                continue

            if display_count % 2:
                in_display_math = True
                display_start = lineno
                clean = clean.split("$$", 1)[0]

            clean = re.sub(r"(?<!\\)\$\$.*?(?<!\\)\$\$", "", clean)
            clean = clean.replace(r"\$", "")
            delimiters = list(SINGLE_DOLLAR_RE.finditer(clean))
            if len(delimiters) % 2:
                errors.append(
                    f"{rel(path)}:{lineno}: unbalanced $ inline-math delimiter; "
                    + "write currency as an ISO code such as USD 100"
                )
                continue

            for match in INLINE_DOLLAR_MATH_RE.finditer(clean):
                if re.match(r"\s*\d[\d,.]*\s+[A-Za-z]", match.group(1)):
                    errors.append(
                        f"{rel(path)}:{lineno}: probable currency parsed as math: "
                        + match.group(0)
                    )

        if in_display_math:
            errors.append(
                f"{rel(path)}:{display_start}: unclosed $$ display-math delimiter"
            )


def validate_book_workflow_paths(errors: list[str]) -> None:
    path = ROOT / ".github/workflows/build-book.yml"
    if not path.exists():
        return

    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        for match in BOOK_CHAPTER_RE.finditer(line):
            target = ROOT / match.group(1)
            if not target.exists():
                errors.append(
                    f"{rel(path)}:{lineno}: missing chapter path: {match.group(1)}"
                )


def validate_chapter_manifests(errors: list[str]) -> None:
    """Keep the README, web sidebar, and book build in sync with the corpus."""

    expected = article_paths()

    workflow = ROOT / ".github/workflows/build-book.yml"
    if workflow.exists():
        listed = [
            Path(match.group(1))
            for match in BOOK_CHAPTER_RE.finditer(workflow.read_text(encoding="utf-8"))
            if not match.group(1).startswith("ja/")
        ]
        counts: dict[Path, int] = {}
        for path in listed:
            counts[path] = counts.get(path, 0) + 1
        for path, count in sorted(counts.items()):
            if count > 1:
                errors.append(f"{rel(workflow)}: chapter listed {count} times: {path}")
        for path in sorted(expected - set(listed)):
            errors.append(f"{rel(workflow)}: English chapter missing from book: {path}")
        for path in sorted(set(listed) - expected):
            errors.append(f"{rel(workflow)}: stale book chapter entry: {path}")
        if set(listed) == expected and listed != sorted(expected):
            errors.append(
                f"{rel(workflow)}: chapter order must follow section and filename order"
            )

    config = ROOT / ".vitepress/config.mts"
    if config.exists():
        routes = {
            match.group(1).split("#", 1)[0].rstrip("/")
            for match in VITEPRESS_LINK_RE.finditer(config.read_text(encoding="utf-8"))
            if ARTICLE_ROUTE_RE.match(match.group(1).strip("/").split("#", 1)[0])
        }
        expected_routes = {
            "/" + str(path.with_suffix("")) for path in expected
        } | {
            "/ja/" + str(path.with_suffix("")) for path in expected
        }
        for route in sorted(expected_routes - routes):
            errors.append(f"{rel(config)}: chapter missing from sidebar: {route}")
        for route in sorted(routes - expected_routes):
            errors.append(f"{rel(config)}: stale sidebar chapter route: {route}")

    readme = ROOT / "README.md"
    if readme.exists():
        targets = {
            Path(target_from_link(match.group(1)).split("#", 1)[0])
            for match in MARKDOWN_LINK_RE.finditer(
                strip_fenced_code(readme.read_text(encoding="utf-8"))
            )
            if target_from_link(match.group(1)).endswith(".md")
        }
        for path in sorted(expected - targets):
            errors.append(f"{rel(readme)}: English chapter missing from contents: {path}")


def validate_vitepress_workflow_links(errors: list[str]) -> None:
    for relative in VITEPRESS_SOURCE_FILES:
        path = ROOT / relative
        if not path.exists():
            continue

        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for regex in (VITEPRESS_LINK_RE, VITEPRESS_HREF_RE):
                for match in regex.finditer(line):
                    link = match.group(1)
                    if link.startswith(("http://", "https://")):
                        continue
                    if not link.startswith("/"):
                        continue
                    if link == "/" or link.endswith(".svg"):
                        continue

                    target = link.strip("/")
                    candidates = [ROOT / f"{target}.md", ROOT / target / "index.md"]
                    if not any(candidate.exists() for candidate in candidates):
                        errors.append(
                            f"{rel(path)}:{lineno}: missing VitePress link target: {link}"
                        )


def validate_generated_assets(errors: list[str]) -> None:
    files = [
        ROOT / ".vitepress/config.mts",
        ROOT / ".vitepress/theme/custom.css",
        ROOT / ".vitepress/theme/components/LandingPage.vue",
        ROOT / "404.md",
        ROOT / "index.md",
        ROOT / "ja/index.md",
    ]

    for path in files:
        if not path.exists():
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in VITEPRESS_ASSET_RE.finditer(line):
                target = match.group(1).lstrip("/")
                if not (ROOT / "public" / target).exists():
                    errors.append(f"{rel(path)}:{lineno}: missing asset: /{target}")


def validate_homepage_html_links(errors: list[str]) -> None:
    config = ROOT / ".vitepress/config.mts"
    if not config.exists():
        return

    base_match = VITEPRESS_BASE_RE.search(config.read_text(encoding="utf-8"))
    if not base_match:
        errors.append(f"{rel(config)}: missing VitePress base setting")
        return
    site_base = base_match.group(1)

    for relative in HOMEPAGE_SOURCES:
        path = ROOT / relative
        if not path.exists():
            continue

        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in HTML_ROOT_HREF_RE.finditer(line):
                link = match.group(1)
                if not link.startswith(site_base):
                    errors.append(
                        f"{rel(path)}:{lineno}: raw HTML link must include site base "
                        + f"{site_base}: {link}"
                    )
                    continue

                target = "/" + link[len(site_base) :]
                if target == "/":
                    continue
                candidates = local_candidates(path, target)
                if candidates and not any(
                    candidate.exists() for candidate in candidates
                ):
                    errors.append(
                        f"{rel(path)}:{lineno}: missing homepage link target: {link}"
                    )


def main() -> int:
    errors: list[str] = []

    article_count = validate_article_parity(errors)
    validate_markdown_links(errors)
    validate_advertised_counts(errors, article_count)
    validate_frontmatter(errors)
    validate_deep_system_chapters(errors)
    validate_unique_chapter_titles(errors)
    validate_unique_h2_headings(errors)
    validate_unique_substantial_prose(errors)
    validate_portable_math_delimiters(errors)
    validate_dollar_math_delimiters(errors)
    validate_book_workflow_paths(errors)
    validate_chapter_manifests(errors)
    validate_vitepress_workflow_links(errors)
    validate_generated_assets(errors)
    validate_homepage_html_links(errors)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"Documentation validation passed: {article_count} articles per language.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
