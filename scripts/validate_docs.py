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
    validate_book_workflow_paths(errors)
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
