#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

# Part/chapter structure lives in book.manifest.json; scripts/book_structure.py owns
# the checks that keep the sidebars, book build, and README contents in step with it.
from book_structure import validate as validate_book_structure

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

# The runbook chapters carry an incident-readiness pair: the settings that decide whether
# the chapter's contract holds on a live deployment, and a symptom-first route through the
# failures those settings produce. Naming the pair here keeps the two from drifting apart
# one chapter at a time, which is how a reference book turns into a description of
# mechanisms that nobody can operate. Entries are either a part directory, which enrolls
# every chapter in it, or an individual chapter, so a part can adopt the pair chapter by
# chapter instead of in one sweeping edit.
RUNBOOK_SECTIONS = [
    Path("05-messaging"),
    Path("02-distributed-databases/01-single-leader-replication.md"),
    Path("02-distributed-databases/02-multi-leader-replication.md"),
    Path("02-distributed-databases/03-leaderless-replication.md"),
]
RUNBOOK_REQUIRED_HEADINGS = [
    re.compile(r"^## Product Knob Map$", re.MULTILINE),
    re.compile(r"^## Incident Playbook$", re.MULTILINE),
]

# Every chapter opens with the same scannable summary so readers (and the site's
# excerpt/OG pipeline) can rely on the first section being the verdict, not a preamble.
LEAD_SUMMARY_HEADING = "TL;DR"
CANONICAL_REFERENCES_HEADINGS = {"references": "## References", "参考文献": "## 参考文献"}
# Renamed variants kept the book reader-facing for a while; each of these once appeared
# in published chapters, so they are rejected by name to stop the old spellings returning.
STALE_REFERENCES_HEADINGS = {
    "primary reference": "## References",
    "primary references": "## References",
    "primary source": "## References",
    "primary sources": "## References",
    "further reading": "## References",
    "参考文献": "## 参考文献",
    "参考資料": "## 参考文献",
    "さらなる読み物": "## 参考文献",
}
EM_DASH = "\u2014"
RELATED_CHAPTERS_HEADING = "## Related Chapters"
# A chapter read in isolation is a dead end: the book is used as a reference, so every
# page has to be both reachable from the rest of the book and able to send the reader
# somewhere. Three outbound links is the smallest number that offers a real choice, and
# one inbound edge is the smallest proof that a chapter was placed into the book rather
# than appended to it.
MIN_CHAPTER_OUTLINKS = 3
# The closing section is a chapter's own contribution to that graph, so it is checked as
# a contract instead of left to taste. Three is the smallest list that offers a real
# choice; eight is where "read this next" turns into an index nobody finishes. Every
# bullet carries the reason the edge exists: a bare link leaves the reader to guess which
# of four neighboring chapters answers their question, which is how reading paths die.
MAX_RELATED_LINKS = 8
RELATED_BULLET_RE = re.compile(r"^- \[([^\]]+)\]\(([^)\s]+\.md)(?:#[^)]*)?\): (\S.+)$")
RELATED_RATIONALE_CHARS = (20, 150)
# A chapter is read for its mechanisms and remembered for its judgment, so the section a
# reader reaches last has to convert the mechanisms into something they can carry into
# their own design review. The role is checked rather than one exact name: a chapter about
# migrations closes on choosing a strategy and a paper analysis closes on lessons that
# transfer, and flattening both to "Key Takeaways" would lose the difference.
SYNTHESIS_SECTION_ROLES = ("takeaway", "decision", "review", "lesson", "transfer", "choos")
# A heading with no blank line after it renders as a heading in CommonMark but not in
# every tool that consumes the book build, and it is what a careless heading rewrite
# leaves behind.
HEADING_RE = re.compile(r"^#{1,6} ")
# Every chapter is read out of order, so the page outline is the reader's only map of a
# chapter. Section names are therefore title case with minor words lowercase, and one name is
# spelled one way across the book: two spellings of "Capacity and cost model" read as two
# different sections in an outline and in site search. The rule only ever adds capitalisation,
# so a deliberate lowercase word (a variable such as top-k, an acronym such as mTLS, a call
# name such as fsync) still passes. English only: the Japanese mirror keeps its own headings.
SECTION_CASE_MINOR_WORDS = frozenset(
    "a an and as at but by for from in into nor of on or the to under up via versus vs plus per".split()
)
SECTION_CASE_ALWAYS_LOWER = frozenset({"fsync", "fdatasync", "etcd"})
SECTION_CASE_CHUNK_RE = re.compile(r"[A-Za-z0-9'\u2019]+|[^\sA-Za-z0-9'\u2019]+|\s+")
SECTION_CASE_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\u2019]*")
SECTION_CASE_BOUNDARY_RE = re.compile(r"[:\u2014\u2013]")
SECTION_CASE_NUMBERING_RE = re.compile(r"^\d+\.\s*")


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


def validate_runbook_sections(errors: list[str]) -> None:
    """Keep the incident-readiness pair in the chapters that carry it.

    A Product Knob Map names the product settings that decide whether the chapter's
    contract holds once it is deployed, and an Incident Playbook turns the chapter's
    failure traces into symptom-first actions. Both belong in the middle of a chapter,
    so this pins presence and relative order only: which knobs and which symptoms earn a
    place is editorial judgment that no heading pattern can encode.
    """

    for section in RUNBOOK_SECTIONS:
        entry = ROOT / section
        chapters = sorted(entry.glob("*.md")) if entry.is_dir() else [entry]
        for path in chapters:
            if not path.is_file():
                errors.append(f"{rel(path)}: listed as a runbook chapter but missing")
                continue
            prose = strip_fenced_code(path.read_text(encoding="utf-8"))
            positions: list[int] = []
            for heading in RUNBOOK_REQUIRED_HEADINGS:
                match = heading.search(prose)
                if not match:
                    errors.append(
                        f"{rel(path)}: missing required heading matching {heading.pattern}"
                    )
                    break
                positions.append(match.start())

            if len(positions) == len(RUNBOOK_REQUIRED_HEADINGS) and positions != sorted(
                positions
            ):
                errors.append(
                    f"{rel(path)}: required runbook headings are out of order"
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


def h2_headings(text: str) -> list[tuple[int, str]]:
    """Return (lineno, heading text) for every H2 outside fenced code."""

    headings: list[tuple[int, str]] = []
    for lineno, line in enumerate(strip_fenced_code(text).splitlines(), start=1):
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            headings.append((lineno, match.group(1)))
    return headings


def validate_chapter_contract(errors: list[str]) -> None:
    """Enforce the shared reader contract for every chapter in both languages.

    Chapters are read out of order: most readers arrive from a search engine or an
    internal link rather than from the previous page. Three properties make that
    survivable, and each one used to be held together by convention alone. The first
    section is the verdict so a reader can stop reading in ten seconds; references sit
    under one predictable heading so evidence is always in the same place; and English
    prose keeps one dash discipline so the book reads as a single author.

    Japanese chapters are held to the lead-summary and heading-spelling rules but not
    yet to having a references section, because parts of the mirror still predate the
    English reference lists. Tighten that once the affected articles are re-translated.
    """

    for prefix in ("", "ja"):
        canonical_key = "参考文献" if prefix else "references"
        canonical = CANONICAL_REFERENCES_HEADINGS[canonical_key]

        for relative in sorted(article_paths(prefix)):
            path = ROOT / prefix / relative
            headings = h2_headings(path.read_text(encoding="utf-8"))
            if not headings:
                errors.append(f"{rel(path)}: chapter has no H2 sections")
                continue

            first_lineno, first = headings[0]
            if normalize_heading(first) != LEAD_SUMMARY_HEADING.casefold():
                errors.append(
                    f"{rel(path)}:{first_lineno}: the first H2 must be "
                    + f"'## {LEAD_SUMMARY_HEADING}', found '## {first}'"
                )

            references: list[int] = []
            for lineno, heading in headings:
                key = normalize_heading(heading)
                if key == canonical_key:
                    references.append(lineno)
                elif key in CANONICAL_REFERENCES_HEADINGS or (
                    key in STALE_REFERENCES_HEADINGS
                ):
                    errors.append(
                        f"{rel(path)}:{lineno}: references heading must be "
                        + f"'{canonical}', found '## {heading}'"
                    )

            if len(references) > 1:
                errors.append(
                    f"{rel(path)}:{references[-1]}: duplicate '{canonical}' section "
                    + f"(first at line {references[0]})"
                )
            elif not references and not prefix:
                errors.append(f"{rel(path)}: missing '{canonical}' section")

            # End matter has a fixed shape: evidence first, then the pointer to the next
            # chapter. Reversed order makes readers scroll past navigation to reach the
            # sources they came to check, and it breaks the "last section is a link out"
            # affordance that the reading tracks in the README rely on.
            related = [
                lineno for lineno, heading in headings
                if normalize_heading(heading) == "related chapters"
            ]
            if len(related) > 1:
                errors.append(
                    f"{rel(path)}:{related[-1]}: duplicate '{RELATED_CHAPTERS_HEADING}' "
                    + f"section (first at line {related[0]})"
                )
            if references and related and min(related) < max(references):
                errors.append(
                    f"{rel(path)}:{min(related)}: '{RELATED_CHAPTERS_HEADING}' must come "
                    + f"after '{canonical}'"
                )


def chapter_link_title(path: Path) -> str:
    """Return the title that a link to this chapter should display.

    Chapter H1s carry qualifiers after a colon, pipe, or parenthesis ("Spanner (OSDI
    2012): Evidence-First Paper Analysis"), and those qualifiers belong on the page rather
    than in every link pointing at it. Link text repeats the part before the qualifier so
    it still matches the heading the reader lands on.
    """

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return re.split(r"\s*[:|(]", line[2:].strip(), maxsplit=1)[0].strip()
    return ""


def validate_related_chapters(errors: list[str]) -> None:
    """Require every English chapter to end by handing the reader on with a reason.

    Readers reach a chapter from a search result or one internal link and then choose what
    to read next with only that page in front of them. The closing section is where the
    choice gets made, so it has to say why a neighboring chapter is worth the next twenty
    minutes: three edges with reasons beat an index of everything adjacent, and a rationale
    is the only thing that separates a designed reading path from a keyword match. Keeping
    the list short, deduplicated, and pointed at chapters rather than anchors is what makes
    it readable instead of merely dense. At least one edge also has to leave the part: a
    chapter whose every pointer lands next door teaches the reader that the book stops at
    the directory boundary, which is exactly how a 160-chapter reference turns into 19
    small ones.

    Japanese chapters are exempt because the mirror has no Related Chapters sections yet.
    That gap is a translation backlog, not a standard to hold new writing to, so remove
    this exemption once the mirror carries the section.
    """

    chapters = {str(relative) for relative in article_paths()}
    for relative in sorted(article_paths()):
        path = ROOT / relative
        text = strip_fenced_code(path.read_text(encoding="utf-8"))
        lines = text.splitlines()
        headings = h2_headings(text)
        related = [
            lineno
            for lineno, title in headings
            if normalize_heading(title) == "related chapters"
        ]
        if not related:
            errors.append(f"{rel(path)}: missing '{RELATED_CHAPTERS_HEADING}' section")
            continue

        heading_lineno = related[0]
        if headings[-1][0] != heading_lineno:
            errors.append(
                f"{rel(path)}:{heading_lineno}: '{RELATED_CHAPTERS_HEADING}' must be the "
                + "last section, so the next chapter is what a reader reaches last"
            )
        separator = next(
            (i for i in range(heading_lineno - 2, -1, -1) if lines[i].strip()), None
        )
        if separator is None or lines[separator].strip() != "---":
            errors.append(
                f"{rel(path)}:{heading_lineno}: '{RELATED_CHAPTERS_HEADING}' must be "
                + "separated from the references by '---'"
            )

        end = next(
            (i for i in range(heading_lineno, len(lines)) if HEADING_RE.match(lines[i])),
            len(lines),
        )
        bullets = [line for line in lines[heading_lineno:end] if line.startswith("- ")]
        if not MIN_CHAPTER_OUTLINKS <= len(bullets) <= MAX_RELATED_LINKS:
            errors.append(
                f"{rel(path)}:{heading_lineno}: '{RELATED_CHAPTERS_HEADING}' needs "
                + f"{MIN_CHAPTER_OUTLINKS} to {MAX_RELATED_LINKS} links, found "
                + f"{len(bullets)}"
            )

        seen: set[str] = set()
        for offset, bullet in enumerate(bullets, start=1):
            match = RELATED_BULLET_RE.match(bullet)
            if not match:
                errors.append(
                    f"{rel(path)}:{heading_lineno}: related link must read "
                    + "'- [Chapter title](path.md): why to read it', found "
                    + f"'{bullet[:60]}'"
                )
                continue

            title, target, rationale = match.groups()
            resolved = os.path.relpath(
                os.path.normpath(os.path.join(str(path.parent), unquote(target.split("#", 1)[0]))),
                ROOT,
            )
            if resolved == str(relative):
                errors.append(f"{rel(path)}: related link points at the chapter itself")
            elif resolved in seen:
                errors.append(f"{rel(path)}: duplicate related link to {resolved}")
            elif resolved not in chapters:
                errors.append(
                    f"{rel(path)}: related link must target another chapter, found {target}"
                )
            else:
                seen.add(resolved)
                expected = chapter_link_title(ROOT / resolved)
                if title != expected:
                    errors.append(
                        f"{rel(path)}:{heading_lineno}: related link text must match the "
                        + f"target title, found '{title}' for {resolved}"
                    )

            low, high = RELATED_RATIONALE_CHARS
            if not low <= len(rationale) <= high:
                errors.append(
                    f"{rel(path)}:{heading_lineno}: related link rationale must be "
                    + f"{low} to {high} characters, found {len(rationale)} for {target}"
                )
            elif len(rationale.split()) < 3 or rationale.endswith("."):
                errors.append(
                    f"{rel(path)}:{heading_lineno}: related link rationale must state a "
                    + f"relationship in a sentence fragment, found '{rationale[:60]}'"
                )

        own_part = str(relative).split(os.sep)[0]
        if seen and all(resolved.split(os.sep)[0] == own_part for resolved in seen):
            errors.append(
                f"{rel(path)}:{heading_lineno}: every related link stays inside part "
                + f"{own_part}; at least one edge should cross into another part, because a "
                + "reader finished with this chapter is not finished with the book"
            )


def validate_chapter_synthesis_section(errors: list[str]) -> None:
    """Require every English chapter to close its content on a synthesis section.

    A chapter that ends on a mechanism leaves the reader holding parts with no instruction
    on when to use them, and that is the difference between reference material and a book
    someone prepares with. The closing section is where trade-offs get resolved into a
    position: which option wins under which constraint, which question to ask about a system
    the reader did not build, which lesson survives contact with another domain. Making it
    the last thing before the sources means the judgment is what stays in memory, and it
    gives an editor an obvious place to check that the chapter actually concluded something.

    Japanese chapters are exempt for now because the mirror still carries sections whose
    names predate this convention.
    """

    for relative in sorted(article_paths("")):
        path = ROOT / relative
        headings = h2_headings(strip_fenced_code(path.read_text(encoding="utf-8")))
        references = [
            index
            for index, (_, heading) in enumerate(headings)
            if normalize_heading(heading) == "references"
        ]
        if not references or len(headings[: references[0]]) < 2:
            # Missing or misordered end matter is reported by validate_chapter_contract.
            continue

        lineno, heading = headings[references[0] - 1]
        role = SECTION_CASE_NUMBERING_RE.sub("", normalize_heading(heading))
        if not any(marker in role for marker in SYNTHESIS_SECTION_ROLES):
            errors.append(
                f"{rel(path)}:{lineno}: the section before '## References' must be a closing "
                + "synthesis (takeaways, a decision framework, review questions, or lessons), "
                + f"not the mechanism section '{heading}'"
            )


def validate_english_dash_style(errors: list[str]) -> None:
    """Keep English prose on the en-dash-only style the book was written in.

    The book has never used an em dash in English copy; ranges such as 1998-2004 use a
    hyphen or an en dash instead. Enforcing it keeps newly written chapters consistent
    rather than leaving the rule buried in contributor expectations.
    """

    for path in iter_markdown_files():
        relative = path.relative_to(ROOT)
        if relative.parts[0] == "ja":
            continue
        prose = strip_fenced_code(path.read_text(encoding="utf-8"))
        for lineno, line in enumerate(prose.splitlines(), start=1):
            if EM_DASH in line:
                errors.append(
                    f"{rel(path)}:{lineno}: English Markdown must not use an em dash; "
                    + "rewrite with a comma, colon, or parentheses"
                )


def canonical_section_heading(value: str) -> str:
    """Return the book casing for one English section heading.

    Words are only ever raised to title case, never lowered, and a word that already carries
    an internal capital is left alone. That keeps acronyms (BM25, mTLS), camel-case names
    (FlashAttention), and single-letter variables (top-k) intact without a vocabulary.
    """

    parts: list[str] = []
    force_capital = True
    for chunk in SECTION_CASE_CHUNK_RE.finditer(value):
        token = chunk.group(0)
        if SECTION_CASE_WORD_RE.fullmatch(token):
            lowered = token.lower()
            keeps_case = (
                not token[0].islower()
                or any(character.isupper() for character in token[1:])
                or len(token) == 1
                or lowered in SECTION_CASE_ALWAYS_LOWER
                or (not force_capital and lowered in SECTION_CASE_MINOR_WORDS)
            )
            parts.append(token if keeps_case else token[0].upper() + token[1:])
            force_capital = False
        else:
            parts.append(token)
            if SECTION_CASE_BOUNDARY_RE.search(token):
                force_capital = True
    return "".join(parts).strip()


def validate_english_section_casing(errors: list[str]) -> None:
    """Keep English section names on the one casing convention the book publishes.

    Section headings are the outline readers scan when a chapter is longer than a screen,
    which is most of them. When the same section name appears twice in the book under two
    spellings, an outline that promises "Verification strategy" in one chapter and
    "Verification Strategy" in another stops being a reliable index of what the book covers,
    and neither spelling can be recommended to whoever writes the next chapter.
    """

    for path in iter_markdown_files():
        if path.relative_to(ROOT).parts[0] == "ja":
            continue
        for lineno, heading in h2_headings(strip_fenced_code(path.read_text(encoding="utf-8"))):
            body = SECTION_CASE_NUMBERING_RE.sub("", heading)
            numbering = heading[: len(heading) - len(body)]
            canonical = canonical_section_heading(body)
            if canonical != body:
                errors.append(
                    f'{rel(path)}:{lineno}: section heading must read "{numbering}{canonical}", '
                    + f'not "{heading}"'
                )


def validate_markdown_layout(errors: list[str]) -> None:
    """Keep the Markdown layout and section names the book build depends on.

    A heading needs a blank line under it: CommonMark tolerates its absence, Pandoc and
    some editors do not, and a heading rewrite that deletes the line is otherwise
    invisible in a diff. The related-chapters section is named in one place so readers
    learn where to look for the next chapter instead of learning three synonyms. Only
    chapters are in scope: the README contents block is generated, and its compact
    part headings are deliberate.
    """

    for prefix in ("", "ja"):
        for relative in sorted(article_paths(prefix)):
            path = ROOT / prefix / relative
            prose = strip_fenced_code(path.read_text(encoding="utf-8"))
            lines = prose.splitlines()
            for lineno, line in enumerate(lines, start=1):
                if not HEADING_RE.match(line):
                    continue
                following = lines[lineno] if lineno < len(lines) else ""
                if following.strip():
                    errors.append(
                        f"{rel(path)}:{lineno}: heading must be followed by a blank line"
                    )
                if line.startswith("## Related") and line.strip() != RELATED_CHAPTERS_HEADING:
                    errors.append(
                        f"{rel(path)}:{lineno}: related-reading section must be named "
                        + f"'{RELATED_CHAPTERS_HEADING}', found '{line.strip()}'"
                    )


def chapter_link_graph() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return (outbound, inbound) chapter-to-chapter links keyed by part/file.md.

    Only English chapters count, and only links that resolve to another chapter: the
    graph is about the reading path through the book, so a citation to a paper or a link
    into the README is not an edge. Image links are excluded for the same reason.
    """

    chapters = {}
    for relative in sorted(article_paths()):
        chapters[str(relative)] = ROOT / relative

    outbound = {key: set() for key in chapters}
    inbound = {key: set() for key in chapters}
    for key, path in chapters.items():
        for target in MARKDOWN_LINK_RE.findall(path.read_text(encoding="utf-8")):
            target = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not target.endswith(".md") or SCHEME_RE.match(target) or target.startswith("/"):
                continue
            resolved = os.path.relpath(
                os.path.normpath(os.path.join(str(path.parent), target)), ROOT
            )
            if resolved in chapters and resolved != key:
                outbound[key].add(resolved)
                inbound[resolved].add(key)
    return outbound, inbound


def validate_reading_graph(errors: list[str]) -> None:
    """Keep every chapter reachable, and keep every chapter able to continue the reading.

    A chapter with no inbound edge is invisible in a book that is mostly read by
    following links from a nearby topic, and one with almost no outbound edges strands a
    reader who has finished it. Both states arrive the same way: a chapter is written as
    a strong standalone piece and nobody wires it into its neighbours afterwards.
    """

    outbound, inbound = chapter_link_graph()
    for key in sorted(outbound):
        if len(outbound[key]) < MIN_CHAPTER_OUTLINKS:
            errors.append(
                f"{key}: links to {len(outbound[key])} other chapters; a chapter must "
                + f"offer at least {MIN_CHAPTER_OUTLINKS}, normally in a "
                + f"'{RELATED_CHAPTERS_HEADING}' section"
            )
        if not inbound[key]:
            errors.append(
                f"{key}: no other chapter links to it; add it to the nearest chapters' "
                + f"'{RELATED_CHAPTERS_HEADING}' sections"
            )


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
    validate_runbook_sections(errors)
    validate_unique_chapter_titles(errors)
    validate_unique_h2_headings(errors)
    validate_english_section_casing(errors)
    validate_chapter_contract(errors)
    validate_related_chapters(errors)
    validate_chapter_synthesis_section(errors)
    validate_markdown_layout(errors)
    validate_reading_graph(errors)
    validate_english_dash_style(errors)
    validate_unique_substantial_prose(errors)
    validate_portable_math_delimiters(errors)
    validate_dollar_math_delimiters(errors)
    validate_book_structure(errors)
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
