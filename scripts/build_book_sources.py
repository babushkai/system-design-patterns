#!/usr/bin/env python3
"""Prepare chapter sources for the PDF/EPUB build.

The site renders chapters with VitePress, which draws Mermaid diagrams in the browser and
resolves ``../part/chapter.md`` links as routes. Pandoc does neither, so a book built
straight from the repository ships every diagram as source code and every cross-reference
as a link to a file that is not in the book. This script writes a transformed copy of each
chapter that pandoc can consume:

* Mermaid blocks are rendered to PNG with mermaid-cli (same theme as the site) and
  replaced by an image sized to fit the page.
* Links to other chapters become in-book anchors; each chapter title carries the matching
  identifier. Links to Markdown files that are not chapters point at the published site.
* Each part gets a divider page listing its chapters, so the book has the same part
  structure as the sidebar.

Usage (see .github/workflows/build-book.yml):

    python3 scripts/book_structure.py --chapters > build/chapters.txt
    python3 scripts/build_book_sources.py --chapter-list build/chapters.txt \
        --out build/src --mmdc "npx --no-install mmdc" --require-diagrams > build/sources.txt

It prints the generated files in print order, relative to the parent of ``--out``, which is
the directory pandoc must run from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import shlex
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from book_structure import ROOT, chapter_heading, load_parts, markdown_path, readme_label

SITE_URL = "https://design.babushkai.com"
MERMAID_THEME_PATH = ROOT / ".vitepress" / "mermaid-theme.json"
# Diagrams are rasterized at this multiple of their CSS size: sharp enough for print
# without making the EPUB several times larger than its text.
DIAGRAM_SCALE = 2
# CSS-pixel dimensions of the text block on an A4 page with one-inch margins, at 96 dpi.
# A diagram wider or taller than this is scaled down; a smaller one keeps its natural size
# instead of being stretched across the page.
PAGE_WIDTH_PX = 602
PAGE_HEIGHT_PX = 800

FENCE_RE = re.compile(r"^(```|~~~)")
MERMAID_BLOCK_RE = re.compile(r"^(```|~~~)mermaid[^\n]*\n(.*?)\n\1[ \t]*$", re.DOTALL | re.MULTILINE)
MARKDOWN_LINK_RE = re.compile(r"\]\(([^)\s]+?\.md)(#[^)\s]*)?\)")
H1_RE = re.compile(r"^# (.+?)\s*$")


def chapter_id(route: str) -> str:
    return "ch-" + route.replace("/", "-")


def split_fences(text: str) -> list[tuple[bool, str]]:
    """Split Markdown into (is_code, chunk) runs so prose rewrites never touch code."""

    chunks: list[tuple[bool, str]] = []
    buffer: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        match = FENCE_RE.match(line)
        if fence is None and match:
            if buffer:
                chunks.append((False, "".join(buffer)))
            buffer = [line]
            fence = match.group(1)
        elif fence is not None and line.startswith(fence) and line.strip() == fence:
            buffer.append(line)
            chunks.append((True, "".join(buffer)))
            buffer = []
            fence = None
        else:
            buffer.append(line)
    if buffer:
        chunks.append((fence is not None, "".join(buffer)))
    return chunks


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG file")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def image_width_percent(path: Path) -> int:
    width, height = png_size(path)
    css_width, css_height = width / DIAGRAM_SCALE, height / DIAGRAM_SCALE
    scale = min(1.0, PAGE_WIDTH_PX / css_width, PAGE_HEIGHT_PX / css_height)
    return max(10, min(100, round(css_width * scale / PAGE_WIDTH_PX * 100)))


class DiagramRenderer:
    def __init__(self, out_dir: Path, mmdc: list[str] | None, jobs: int) -> None:
        self.out_dir = out_dir
        self.mmdc = mmdc
        self.jobs = jobs
        self.sources: dict[str, str] = {}
        self.origins: dict[str, str] = {}
        self.failed: dict[str, str] = {}

    def register(self, source: str, origin: str) -> str:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        self.sources[digest] = source
        self.origins.setdefault(digest, origin)
        return digest

    def image_path(self, digest: str) -> Path:
        return self.out_dir / f"{digest}.png"

    def _render(self, digest: str) -> None:
        target = self.image_path(digest)
        if target.exists():
            return
        source = self.out_dir / f"{digest}.mmd"
        source.write_text(self.sources[digest] + "\n", encoding="utf-8")
        command = [
            *self.mmdc,
            "--input", str(source),
            "--output", str(target),
            "--scale", str(DIAGRAM_SCALE),
            "--backgroundColor", "white",
            "--configFile", str(MERMAID_THEME_PATH),
            "--puppeteerConfigFile", str(self.out_dir / "puppeteer.json"),
            "--quiet",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not target.exists():
            # Keep mermaid's parse error and drop the Node stack trace that follows it.
            lines = (result.stderr or result.stdout).splitlines()
            message = [line for line in lines if line.strip() and not line.lstrip().startswith("at ")]
            self.failed[digest] = "\n".join(message[:6]) or "no error output"

    def render_all(self) -> None:
        if not self.mmdc or not self.sources:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # CI runners refuse to start Chromium's sandbox for an unprivileged user.
        (self.out_dir / "puppeteer.json").write_text(
            json.dumps({"args": ["--no-sandbox", "--disable-setuid-sandbox"]}), encoding="utf-8"
        )
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            list(pool.map(self._render, sorted(self.sources)))


def rewrite_links(chunk: str, source_route: str, routes: set[str]) -> str:
    source_dir = posixpath.dirname(source_route)

    def replace(match: re.Match[str]) -> str:
        target = match.group(1)
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
            return match.group(0)
        resolved = posixpath.normpath(posixpath.join(source_dir, target))
        route = resolved[: -len(".md")]
        if route in routes:
            return f"](#{chapter_id(route)})"
        return f"]({SITE_URL}/{route}{match.group(2) or ''})"

    return MARKDOWN_LINK_RE.sub(replace, chunk)


def transform_chapter(text: str, route: str, routes: set[str], renderer: DiagramRenderer) -> str:
    def replace_diagram(match: re.Match[str]) -> str:
        return f"\n@@DIAGRAM:{renderer.register(match.group(2).strip(), f'{route}.md')}@@\n"

    text = MERMAID_BLOCK_RE.sub(replace_diagram, text)
    out: list[str] = []
    titled = False
    for is_code, chunk in split_fences(text):
        if is_code:
            out.append(chunk)
            continue
        chunk = rewrite_links(chunk, route, routes)
        if not titled:
            lines = chunk.splitlines(keepends=True)
            for index, line in enumerate(lines):
                match = H1_RE.match(line)
                if match:
                    lines[index] = f"# {match.group(1)} {{#{chapter_id(route)}}}\n"
                    titled = True
                    break
            chunk = "".join(lines)
        out.append(chunk)
    if not titled:
        raise SystemExit(f"{route}.md: no '# ' chapter title to anchor cross-references to")
    return "".join(out).rstrip("\n") + "\n"


def place_diagrams(text: str, renderer: DiagramRenderer, base: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        digest = match.group(1)
        image = renderer.image_path(digest)
        if image.exists():
            relative = image.relative_to(base).as_posix()
            return f"![]({relative}){{width={image_width_percent(image)}%}}"
        # No renderer available: keep the source readable instead of dropping the figure.
        return "```text\n" + renderer.sources[digest] + "\n```"

    return re.sub(r"@@DIAGRAM:([0-9a-f]{16})@@", replace, text)


def part_page(part: dict, routes_in_part: list[str]) -> str:
    lines = [f"# Part {part['number']}: {readme_label(part)} {{.unnumbered #part-{part['number']}}}", ""]
    for route in routes_in_part:
        lines.append(f"- [{chapter_heading(markdown_path(route))}](#{chapter_id(route)})")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chapter-list", required=True, help="file from book_structure.py --chapters")
    parser.add_argument("--out", required=True, help="directory for transformed sources")
    parser.add_argument("--mmdc", default="mmdc", help="mermaid-cli command, e.g. 'npx --no-install mmdc'")
    parser.add_argument("--jobs", type=int, default=4, help="diagrams rendered in parallel")
    parser.add_argument(
        "--check-locale",
        action="append",
        default=[],
        metavar="LOCALE",
        help="also render the diagrams of this translation (for example 'ja') to catch syntax "
        "errors there; its chapters are not added to the book",
    )
    parser.add_argument(
        "--require-diagrams",
        action="store_true",
        help="fail when a diagram cannot be rendered instead of leaving its source in the book",
    )
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    base = out.parent
    out.mkdir(parents=True, exist_ok=True)

    listed = [line.strip() for line in Path(args.chapter_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    routes = [path[: -len(".md")] for path in listed]
    route_set = set(routes)

    mmdc = shlex.split(args.mmdc)
    if shutil.which(mmdc[0]) is None:
        if args.require_diagrams:
            print(f"mermaid-cli command not found: {mmdc[0]}", file=sys.stderr)
            return 1
        print(f"warning: {mmdc[0]} not found; diagrams stay as source text", file=sys.stderr)
        mmdc = None
    renderer = DiagramRenderer(out / "diagrams", mmdc, args.jobs)

    transformed: dict[str, str] = {}
    for route in routes:
        text = (ROOT / f"{route}.md").read_text(encoding="utf-8")
        transformed[route] = transform_chapter(text, route, route_set, renderer)

    for locale in args.check_locale:
        for route in routes:
            translated = ROOT / locale / f"{route}.md"
            if not translated.exists():
                continue
            for match in MERMAID_BLOCK_RE.finditer(translated.read_text(encoding="utf-8")):
                renderer.register(match.group(2).strip(), f"{locale}/{route}.md")

    renderer.render_all()
    if renderer.failed:
        for digest, message in renderer.failed.items():
            print(f"{renderer.origins[digest]}: diagram failed to render\n{message}\n", file=sys.stderr)
        if args.require_diagrams:
            return 1

    outputs: list[Path] = []
    for part in load_parts():
        in_part = [chapter["route"] for chapter in part["chapters"] if chapter["route"] in route_set]
        if not in_part:
            continue
        divider = out / f"part-{part['number']:02d}.md"
        divider.write_text(part_page(part, in_part), encoding="utf-8")
        outputs.append(divider)
        for route in in_part:
            target = out / f"{route}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(place_diagrams(transformed[route], renderer, base), encoding="utf-8")
            outputs.append(target)

    missing = route_set - {chapter["route"] for part in load_parts() for chapter in part["chapters"]}
    if missing:
        print(f"chapters not in the manifest: {', '.join(sorted(missing))}", file=sys.stderr)
        return 1

    rendered = sum(1 for digest in renderer.sources if renderer.image_path(digest).exists())
    checked = f" (including {', '.join(args.check_locale)})" if args.check_locale else ""
    print(
        f"prepared {len(routes)} chapters, {rendered}/{len(renderer.sources)} diagrams rendered{checked}",
        file=sys.stderr,
    )
    sys.stdout.write("\n".join(path.relative_to(base).as_posix() for path in outputs) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
