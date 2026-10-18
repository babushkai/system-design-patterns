# Contributing to the fieldbook

Every chapter is a reference page, not a tutorial step: readers arrive from a search box
or one internal link, decide in ten seconds whether the page is theirs, and leave with a
decision they can defend. The rules below exist to keep that true for every chapter
without any one author holding the conventions in their head. Each rule is enforced by
`scripts/validate_docs.py` unless it says otherwise.

## What a Chapter Owns

One concept, stated as a contract: what the mechanism promises, what it costs, and how it
fails. Neighboring concepts are linked rather than summarized, so a reader never finds two
half-explanations that disagree. If a chapter starts explaining its neighbor, move the
paragraph to the neighbor and link it from both sides.

## Chapter Anatomy

```markdown
# Title that matches the manifest

## TL;DR                     First H2, Always: The Verdict and the Contract
...mechanism sections...     behavior under load, contention, partial failure, migration
## Key Takeaways             Last Content Section: Judgment, Not Another Mechanism
## References                Primary Sources, Predictable Heading, No Bibliography
---

## Related Chapters          Last Section, Always
```

- The first H2 is `## TL;DR` so a reader can stop early with an answer.
- Evidence lives under `## References` (Japanese: `## 参考文献`). Synonyms such as
  "Further reading" are rejected: readers learn where sources are once.
- The section before `## References` resolves the mechanisms into a position: takeaways, a
  decision framework, review questions, or lessons that transfer. See "Closing synthesis".
- `## Related Chapters` closes the page, separated from the references by `---`, so the
  next chapter is always the last thing a reader reaches.
- A heading is followed by a blank line. CommonMark forgives its absence, Pandoc and many
  editors do not.

### Section names

Title case, minor words lowercase, one spelling per name across the book:

```markdown
## Workload and Contract                    ## Top-k Execution and Dynamic Pruning
## State, Authority, and Invariants         ## Capacity and Cost Model
```

- The outline is the only map of a chapter longer than a screen, and readers meet it in
  every chapter. Two spellings of one name read as two sections in an outline and in site
  search, and neither spelling can be recommended to whoever writes the next chapter.
- Reordering a name counts as a second spelling: `## Workload Contract` beside the book's
  `## Workload and Contract` splits one section into two, so the check compares the words
  of every English section name regardless of order.
- The check adds capitalisation and never removes it, so deliberate lowercase stays:
  `## Does fsync Actually Sync?`, `## Top-k Execution and Dynamic Pruning`, and
  `## Composition with Overload Budgets` all pass as written.
- Numbered outlines are allowed where a chapter refers to its sections by number; the rule
  ignores the `4. ` prefix. The Japanese mirror keeps its own heading conventions, so the
  check reads English files only.

### Closing synthesis section

Every English chapter ends its content, immediately before `## References`, with a section
whose heading names the judgment role it plays: takeaways, a decision framework, design
review questions, or lessons that transfer.

```markdown
## Key Takeaways                     Numbered conclusions a reader can quote
## Decision Framework                Situation to choice, usually a table
## Design Review Questions           What to ask about a system you did not build
## Lessons That Generalize           What survives into the next design
```

- The last section is what a reader remembers, so it has to answer "which option wins under
  which constraint" rather than add another mechanism. A chapter that ends on `Rollout and
  Recovery` or `Verification Matrix` has described a system and concluded nothing.
- The check matches the role, not one name, because the roles differ: 61 chapters close on
  takeaways, 58 on a decision framework, and the rest on review questions or transferable
  lessons. Renaming all of them to `## Key Takeaways` would flatten that difference.
- Write it from this chapter's own numbers and trade-offs. Five to seven items in the
  chapter's voice beats a generic summary, and a takeaway that would still read correctly
  at the end of any other chapter is not a takeaway.

### Product Knob Map and Incident Playbook

Every chapter in `05-messaging` plus the replication chapters of `02-distributed-databases`
carry an operational pair, and the check names both headings so a new chapter cannot describe
a mechanism without the parts that make it usable on a shift:

```markdown
## Capacity and Cost Model     ...capacity section
## Product Knob Map            Real settings that decide the contract, then read-only checks
...
## Operations and Observability Metrics and the runbook list
## Incident Playbook           Symptom-first routes: look, decide, do, verify
```

- `## Product Knob Map` sits immediately before the capacity section, where a reader decides
  what to configure. One table maps product and scope to the knob, the invariant it enforces,
  and how the default misleads; a fenced block after it prints the live values.
- Mutating commands stay out of that block. Someone under pressure copies what is on the
  screen, so the test for every line is that it could run on production at three in the
  morning and change nothing.
- `## Incident Playbook` follows the observability or security section, after every mechanism
  it operates on has been introduced. Each subsection is one symptom written in look, decide,
  do, verify order, and names the reflex action it replaces: the costliest move in these
  subsystems is usually the plausible-looking one.
- Adopt the pair chapter by chapter. `RUNBOOK_SECTIONS` accepts a part directory or an
  individual chapter, so a part under construction enrolls the chapters it has finished
  instead of leaving a heading promise over the ones that are not ready.
- Hedge version-dependent defaults and tell the reader to read values off the running system.
  The section is a map of where the levers are, not an oracle for someone else's cluster.

### Related Chapters format

Three to eight bullets, each one link plus the reason the edge exists:

```markdown
- [Consistency Models](04-consistency-models.md): what consistency means once you leave the linearizable end of the spectrum
```

- Link text is the target's H1 up to the first `:` or `|` or `(`, so it matches the heading
  the reader lands on. Qualifiers stay on the target page.
- The rationale after `: ` is 20 to 150 characters, states the relationship between the two
  chapters, and is not a summary of the target ("see also", "more on this"). A bare link is
  a keyword match; a reasoned link is a reading path.
- Targets are other English chapters, once each, with no self-links and no links to
  anchors, images, or the README.
- At least one edge leaves the part. A chapter whose every pointer lands next door teaches
  the reader that the book stops at the directory boundary, and the leaf chapters are where
  it happens: a timer service is a distributed-time chapter with a cron expression attached,
  and a label queue is a stream-processing chapter whose workers are people.
- Keep the list curated: eight is the ceiling because a pointer list nobody finishes is
  worse than four that are read.

## Evidence Discipline

Claims carry one of three labels, and the label is the point rather than the decoration:

- **Documented** - a cited primary source states it, for its named version or scope.
- **Inference** - it follows from stated mechanisms; it is not attributed to a deployment.
- **Reference design** - this book's proposal; numbers stay illustrative until measured.

Reported figures need a source, illustrative arithmetic needs its assumptions stated, and
a paper chapter says what the paper established rather than what the vendor later claimed.

## Prose

English copy uses hyphens and en dashes, never an em dash: ranges read as `1998-2004`, and
parentheses or commas carry the aside. Sentences stay declarative and name the mechanism
that does the work, so "the broker re-reads the log after failure" beats "things recover".

## Japanese Mirror

`ja/` mirrors the English routes one to one and is translated as time allows. It is exempt
from the References and Related Chapters rules while that catch-up happens, which means
new English structure does not have to be mirrored in the same pull request. Do not edit
Japanese punctuation or headings to match English style: the mirror keeps its own
typography, and `scripts/validate_docs.py` checks route parity rather than prose.

## Adding, Renaming, or Moving a Chapter

Structure lives in `book.manifest.json`, and the sidebars, print build, and README contents
are generated from it:

1. Add the file under its part directory (`NN-part-slug/NN-chapter-slug.md`).
2. Add the route and both titles to `book.manifest.json`.
3. Run `python scripts/book_structure.py --write-readme` to regenerate the contents block.
4. Run `python scripts/book_structure.py --check`, which fails if a file, sidebar entry, or
   README line has drifted from the manifest.

Chapter numbers are reading order and stay contiguous inside a part, so a list never shows a
hole where a chapter used to be. Add new chapters at the end of a part; when a merge removes
one, renumber the survivors with `git mv`, update their routes in `book.manifest.json`, and
regenerate the README. `scripts/book_structure.py --check` enforces both rules and names the
four parts whose numbers still carry holes from earlier merges. Renumbering changes published
URLs, so treat it as a maintainer decision rather than routine cleanup.

## Checks Before Opening a Pull Request

```bash
python scripts/validate_docs.py       # structure, links, graph, evidence headings, dash style
npm ci && npm run docs:build          # the site actually builds
python scripts/validate_built_site.py # rendered links, anchors, and assets resolve
```

`validate_docs.py` is the CI gate on every pull request. When it rejects something a
contributor keeps getting wrong, the fix is usually a clearer error message or a rule here,
not an exception in the checker.
