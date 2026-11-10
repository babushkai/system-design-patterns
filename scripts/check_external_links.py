#!/usr/bin/env python3
"""Check every external link in the English chapters.

Local validation proves a reference has a link; this script proves the link still leads
somewhere. It runs on a schedule rather than on pull requests, because a publisher's
outage should not block an unrelated change.

Results fall into three groups:

* dead: the server answered 404 or 410, or the hostname no longer resolves. These fail the
  run and need a new target.
* blocked: 401, 403, 429 and similar. Publishers such as the ACM Digital Library refuse
  automated clients, so these say nothing about the page and are only reported.
* unreachable: timeouts and 5xx responses, reported so that a link failing week after week
  gets noticed.

Usage:

    python3 scripts/check_external_links.py            # all chapters
    python3 scripts/check_external_links.py 20-production-readiness
"""

from __future__ import annotations

import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
ARTICLE_GLOB = "[0-9][0-9]-*/*.md"
# The URL may contain one level of balanced parentheses, as Microsoft Learn and DOI
# links do.
LINK_RE = re.compile(r"\]\((https?://(?:[^()\s]|\([^()\s]*\))+)\)")
DEAD_STATUSES = {404, 410}
USER_AGENT = "Mozilla/5.0 (compatible; fieldbook-link-check)"
TIMEOUT_SECONDS = 25
WORKERS = 16


def collect(scope: list[str]) -> dict[str, set[str]]:
    urls: dict[str, set[str]] = defaultdict(set)
    for path in sorted(ROOT.glob(ARTICLE_GLOB)):
        relative = str(path.relative_to(ROOT))
        if scope and not any(relative.startswith(prefix) for prefix in scope):
            continue
        for url in LINK_RE.findall(path.read_text(encoding="utf-8")):
            urls[url.split("#")[0]].add(relative)
    return urls


def fetch(url: str, attempt: int = 0) -> str:
    """Return 'ok', 'dead', 'blocked', or 'unreachable' with a detail suffix."""

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-512"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return "ok" if response.status < 400 else f"unreachable {response.status}"
    except urllib.error.HTTPError as error:
        if error.code in DEAD_STATUSES:
            return f"dead {error.code}"
        if error.code == 416:
            return "ok"
        if error.code < 500:
            return f"blocked {error.code}"
        return f"unreachable {error.code}"
    except urllib.error.URLError as error:
        if isinstance(error.reason, socket.gaierror):
            # One failed lookup is usually the resolver, not the site: ask again before
            # reporting a hostname as gone.
            if attempt == 0:
                time.sleep(3)
                return fetch(url, attempt=1)
            return "dead hostname does not resolve"
        return f"unreachable {error.reason}"
    except (TimeoutError, OSError, ValueError) as error:
        return f"unreachable {error}"


def main(argv: list[str]) -> int:
    urls = collect(argv)
    if not urls:
        print("no external links found for that scope", file=sys.stderr)
        return 1

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = dict(zip(urls, pool.map(fetch, urls)))

    groups: dict[str, list[str]] = defaultdict(list)
    for url, result in results.items():
        groups[result.split()[0]].append(url)

    for group in ("blocked", "unreachable", "dead"):
        for url in sorted(groups[group]):
            host = urlsplit(url).netloc
            print(f"{group:12} {results[url]:28} {url}  <- {', '.join(sorted(urls[url]))}  [{host}]")

    print(
        f"{len(urls)} links: {len(groups['ok'])} ok, {len(groups['blocked'])} blocked by the "
        f"publisher, {len(groups['unreachable'])} unreachable, {len(groups['dead'])} dead"
    )
    return 1 if groups["dead"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
