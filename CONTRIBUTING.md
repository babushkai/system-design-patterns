# Contributing

This repository is an architecture fieldbook, not a catalog of products. A chapter should teach the invariant mechanics that survive framework churn, show where the abstractions leak in production, and give the reader enough quantitative machinery to make and defend a design decision.

## ML and LLM chapter contract

Parts 16 and 17 use a common production-oriented spine. Topic-specific sections belong between the opening summary and the terminal reasoning sections, but every chapter must contain these headings in this order:

1. `## TL;DR`
2. `## Failure Modes`
3. `## Decision Framework` (an optional descriptive suffix is allowed)
4. `## Key Takeaways`
5. `## References`

The shared ending is deliberate. A reader should be able to open any chapter and quickly answer the same questions: how does this fail, when should I choose it, what survives as the essential mental model, and which primary sources support the claims?

Each ML or LLM chapter must also:

- separate model quality from system correctness and from business outcomes;
- define the data plane, control plane, durable state, and ownership boundaries when they exist;
- derive important latency, throughput, memory, capacity, or statistical relationships instead of presenting unexplained rules of thumb;
- cover observability, security/privacy, cost, and rollout consequences where they materially affect the design;
- state assumptions next to numerical examples and avoid presenting benchmark numbers as universal constants;
- prefer primary papers, official specifications, and official project documentation in references;
- use products as concrete examples of a mechanism, not as the organizing structure of the chapter;
- link to the canonical chapter rather than duplicating a shallow explanation of a concept.

Depth is not a word count. A chapter is deep when it derives the mechanism, makes state and ownership explicit, quantifies the important constraint, follows failures across component boundaries, and gives the reader enough evidence to choose between designs. Do not pad chapters with generic production checklists, repeated best-practice bullets, decorative diagrams, framework inventories, or arbitrary minimum length. Use a diagram only when it materially clarifies a relationship, and put an operational gate next to the mechanism that makes the gate necessary.

### Writing style

- Lead with the mechanism and the production consequence.
- Prefer connected explanatory prose for hard ideas; use tables for exact comparisons and short lists only when the items are genuinely parallel.
- A diagram must teach a relationship that is harder to understand linearly. Do not add decorative diagrams.
- Code should demonstrate a correctness boundary, operational contract, or failure mode. Avoid long framework tutorials that will age faster than the principle they illustrate.
- Name the tempting wrong design and explain the concrete incident it creates.
- Distinguish **event time** from processing time, **assignment** from exposure, **model version** from serving policy, and **request success** from task success wherever those distinctions apply.
- Avoid unsupported claims such as “exactly once,” “real time,” “production ready,” or “safe.” Define the scope and measurement behind them.

Run the structural and link checks with:

```bash
python3 scripts/validate_docs.py
```

Then render the book to catch Mermaid, Markdown, and navigation regressions:

```bash
npm run docs:build
```
