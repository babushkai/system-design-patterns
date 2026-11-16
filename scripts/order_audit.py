#!/usr/bin/env python3
"""Audit the book's reading order: can a reader get anywhere, and in what order?

Two properties decide whether a few hundred chapters behave like a book rather than a
folder of notes, and neither one is visible by eye at that size:

* **Reachability.** A reader normally arrives at one chapter from search or a single
  link and then decides what to read next while standing on that page. A chapter that
  nothing outside its own part points at is reachable only from the sidebar, so the
  reading path stops at the directory boundary even though the prose claims otherwise.
* **Order.** Print order is a promise to the straight-through reader that the machinery
  has been built before it gets used. Most forward links are deliberate hand-offs ("X
  owns this; read it when the question is yours"), which cost nothing. Forward links
  whose surrounding sentence uses prerequisite language ("defined in", "requires",
  "assumes") are the ones that make a linear reader stop and go back, so they are
  counted separately and reported rather than merely waved through.

Reachability is enforced against a baseline: the count may fall and never rise, which is
what keeps a new part from silently becoming an island. Order is reported, not enforced,
because the fix for a bad order is usually a renumbering, and renumbering changes
published URLs (see `.github/CONTRIBUTING.md`).

Usage:

    python scripts/order_audit.py                       # human report
    python scripts/order_audit.py --json                # metrics plus the offending edges
    python scripts/order_audit.py --top 15              # longest read-aheads
    python scripts/order_audit.py --check               # what validate_docs.py runs
    python scripts/order_audit.py --write-baseline      # record an improvement
    python scripts/order_audit.py --write-baseline --force   # record a regression too
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from book_structure import chapter_entries, load_parts

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).with_name("order_audit_baseline.json")

LINK_RE = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\s#]+)(?:#[^)]*)?\)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
H2_RE = re.compile(r"^## (?!#)(.*\S)\s*$")

RELATED_SECTION = "Related Chapters"

# Sections whose links are deliberately forward-looking. The TL;DR names the chapters that
# own the rest of the topic, and Related Chapters is a curated map rather than a dependency
# list, so neither says anything about whether the body can be read in print order.
NON_DEPENDENCY_SECTIONS = {"TL;DR", RELATED_SECTION}

# A forward link is only a reading-order problem when the sentence around it claims the
# target has already been established. This is a heuristic used for ranking and reporting,
# never for enforcement: an editorial "as we will see" should not fail a build.
PREREQ_RE = re.compile(
    r"\b(introduce[sd]?|define[sd]?|establish[esd]?|cover[sd]?|explain[sd]?|"
    r"set out|sets out|describe[sd]?|detail[s]?|specif[ye][sd]?|builds? on|rests on|"
    r"require[sd]?|need[s]?|assume[sd]?|depends? on|follows from|derived from|given in)\b",
    re.IGNORECASE,
)


class OrderAuditError(Exception):
    """Raised when the corpus cannot be read at all."""


def _chapter_links(path: Path) -> list[tuple[str, str, str]]:
    """Return (target route, line, enclosing section) for every internal chapter link."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - corrupt checkout
        raise OrderAuditError(f"cannot read {path}: {exc}") from exc

    edges: list[tuple[str, str, str]] = []
    section = ""
    in_fence = False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = H2_RE.match(line)
        if heading:
            section = heading.group(1).strip()
            continue
        for match in LINK_RE.finditer(line):
            target = match.group(1)
            if target.startswith(("http:", "https:", "mailto:", "/", "#")):
                continue
            resolved = (path.parent / target).resolve()
            try:
                relative = resolved.relative_to(ROOT)
            except ValueError:
                continue
            route = relative.as_posix()
            if not route.endswith(".md"):
                continue
            edges.append((route[:-3], line.strip(), section))
    return edges


def audit(parts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Measure reachability and print-order friction for the current manifest order."""

    parts = parts if parts is not None else load_parts()
    chapters = chapter_entries(parts)
    order = [chapter["route"] for chapter in chapters]
    rank = {route: index for index, route in enumerate(order)}
    part_of = {chapter["route"]: part["number"] for part in parts for chapter in part["chapters"]}

    inbound_cross_part: dict[str, int] = defaultdict(int)
    related_edges: set[tuple[str, str]] = set()
    forward: list[dict[str, Any]] = []
    forward_weight_by_part: dict[int, int] = defaultdict(int)

    for route in order:
        path = ROOT / f"{route}.md"
        if not path.exists():
            continue
        for target, context, section in _chapter_links(path):
            if target not in rank or target == route:
                continue
            if part_of[target] != part_of[route]:
                inbound_cross_part[target] += 1
            if section == RELATED_SECTION:
                related_edges.add((route, target))
            # A roadmap link sends the reader somewhere; it does not claim the reader has
            # already been there. Only body prose can break print order.
            if section in NON_DEPENDENCY_SECTIONS or rank[target] <= rank[route]:
                continue
            prereq = bool(PREREQ_RE.search(context))
            forward_weight_by_part[part_of[route]] += 3 if prereq else 1
            forward.append(
                {
                    "source": route,
                    "target": target,
                    "span": rank[target] - rank[route],
                    "prerequisite": prereq,
                    "context": context[:200],
                }
            )

    sidebar_only = [route for route in order if inbound_cross_part[route] == 0]
    one_way = sorted(
        f"{source} -> {target}"
        for source, target in related_edges
        if part_of[source] != part_of[target] and (target, source) not in related_edges
    )

    return {
        "chapters": len(order),
        "parts": len(parts),
        "sidebar_only_chapters": sidebar_only,
        "one_way_cross_part_handoffs": len(one_way),
        "one_way_cross_part_handoff_edges": one_way,
        "cross_part_related_edges": sum(
            1 for source, target in related_edges if part_of[source] != part_of[target]
        ),
        "forward_prerequisite_weight": sum(3 for edge in forward if edge["prerequisite"]),
        "read_aheads": sorted(
            (edge for edge in forward if edge["prerequisite"]),
            key=lambda edge: -edge["span"],
        ),
        "forward_weight_by_part": dict(sorted(forward_weight_by_part.items())),
    }


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any] | None:
    """Read the recorded floor, or return None when the audit has no baseline yet."""

    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OrderAuditError(f"{path.relative_to(ROOT)}: invalid JSON: {exc}") from exc


def write_baseline(
    metrics: dict[str, Any], path: Path = BASELINE_PATH, *, force: bool = False
) -> None:
    """Record the current metrics as the new floor.

    Recording is how an improvement becomes binding, so it refuses to launder a regression:
    moving the floor backwards takes ``--force``, which keeps "I had no time to wire the new
    chapter in" from turning into the book's new standard.
    """

    existing = load_baseline(path)
    if existing is not None and not force:
        worse = regressions(metrics, existing)
        if worse:
            raise OrderAuditError(
                "refusing to widen the floor; fix these or pass --force:\n  "
                + "\n  ".join(worse)
            )

    payload = {
        "_doc": (
            "Reachability floor for scripts/order_audit.py. sidebar_only_chapters lists the "
            "chapters that nothing outside their own part links to; the audit fails when a new "
            "name appears. one_way_cross_part_handoffs is a ceiling that may only fall."
        ),
        "sidebar_only_chapters": sorted(metrics["sidebar_only_chapters"]),
        "one_way_cross_part_handoffs": metrics["one_way_cross_part_handoffs"],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def regressions(metrics: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Name every way ``metrics`` is worse than the recorded floor."""

    found: list[str] = []
    allowed = set(baseline.get("sidebar_only_chapters", []))
    for route in metrics["sidebar_only_chapters"]:
        if route in allowed:
            continue
        found.append(
            f"{route}: nothing outside its own part links to it, so the only way a reader "
            "arrives is the sidebar; add a Related Chapters entry in the chapter that owns "
            "the neighboring question"
        )

    ceiling = baseline.get("one_way_cross_part_handoffs")
    count = metrics["one_way_cross_part_handoffs"]
    if isinstance(ceiling, int) and count > ceiling:
        found.append(
            f"{count} cross-part hand-offs are stated in one direction only, up from the "
            f"baseline of {ceiling}; when a chapter sends the reader somewhere, answer the "
            "return question in the target's Related Chapters list"
        )
    return found


def validate(errors: list[str], parts: list[dict[str, Any]] | None = None) -> None:
    """Fail when a chapter becomes unreachable by link-following, or a hand-off loses its answer."""

    try:
        metrics = audit(parts)
        baseline = load_baseline()
    except OrderAuditError as exc:
        errors.append(f"{Path(__file__).name}: {exc}")
        return

    if baseline is None:
        errors.append(
            f"{Path(__file__).name}: no baseline at "
            f"{BASELINE_PATH.relative_to(ROOT)}; run scripts/order_audit.py --write-baseline"
        )
        return

    errors.extend(regressions(metrics, baseline))


def report(metrics: dict[str, Any], top: int) -> str:
    lines = [
        f"{metrics['chapters']} chapters in {metrics['parts']} parts",
        "",
        f"Reachability (enforced against {BASELINE_PATH.relative_to(ROOT)})",
        f"  sidebar-only chapters .......... {len(metrics['sidebar_only_chapters'])}",
        f"  cross-part related edges ....... {metrics['cross_part_related_edges']}",
        f"  one-way hand-offs .............. {metrics['one_way_cross_part_handoffs']}",
        "",
        "Print order (reported, not enforced; fixing it means renumbering published URLs)",
        f"  forward prerequisite weight .... {metrics['forward_prerequisite_weight']}",
    ]
    if metrics["sidebar_only_chapters"]:
        lines.append("")
        lines.append("  chapters reachable only from the sidebar:")
        lines += [f"    - {route}" for route in metrics["sidebar_only_chapters"]]
    busiest = sorted(metrics["forward_weight_by_part"].items(), key=lambda item: -item[1])[:5]
    lines.append("")
    lines.append("  parts whose chapters lean forward on later material the most:")
    for number, weight in busiest:
        lines.append(f"    Part {number:>2}: {weight:>4}")
    lines.append("")
    lines.append("  longest read-aheads (a body sentence assumes a chapter printed later):")
    for edge in metrics["read_aheads"][:top]:
        lines.append(f"    +{edge['span']:>3}  {edge['source']}  ->  {edge['target']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print metrics as JSON")
    parser.add_argument("--top", type=int, default=12, help="read-aheads to list (default 12)")
    parser.add_argument("--check", action="store_true", help="fail if reachability regressed")
    parser.add_argument("--write-baseline", action="store_true", help="record current metrics")
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --write-baseline, allow the floor to widen",
    )
    args = parser.parse_args(argv)

    try:
        metrics = audit()
        if args.write_baseline:
            write_baseline(metrics, force=args.force)
            print(
                f"wrote {BASELINE_PATH.relative_to(ROOT)}: "
                f"{len(metrics['sidebar_only_chapters'])} sidebar-only chapters, "
                f"{metrics['one_way_cross_part_handoffs']} one-way hand-offs"
            )
            return 0
    except OrderAuditError as exc:
        print(f"order audit failed: {exc}", file=sys.stderr)
        return 1

    errors: list[str] = []
    if args.check:
        validate(errors)

    if args.json:
        print(json.dumps({**metrics, "regressions": errors}, indent=2))
        return 1 if errors else 0

    print(report(metrics, args.top))
    if args.check:
        if errors:
            print("\nReading-order audit found regressions:", file=sys.stderr)
            for error in errors:
                print(f"  {error}", file=sys.stderr)
            return 1
        print("\nReading-order audit: no regression against the baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
