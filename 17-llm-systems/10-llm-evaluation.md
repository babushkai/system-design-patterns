# LLM Evaluation and Observability

## TL;DR

Evals are the test suite of an LLM system — except the system is non-deterministic, "correct" is fuzzy, and the underlying model changes under you on a vendor's schedule. The discipline that works: build a **graded ladder** of evaluators (cheap programmatic assertions → rubric-based LLM-as-judge, calibrated against human labels → humans for the residue), run it as a **CI gate on every change** (prompt, model version, RAG index, harness), and close the loop from **production traces back into the eval set** — your best test cases are yesterday's failures. For agents, evaluate trajectories and report **pass^k** (every-time reliability), not just pass@1, and track cost-per-solved-task next to quality. Observability is the same machinery pointed at production: OpenTelemetry GenAI traces, token/cost accounting, sampled online scoring, and drift alarms. The teams that win at this treat eval curation as a permanent engineering function, not a launch checklist item.

---

## Why This Is Its Own Discipline

Conventional tests assert `f(x) == y`. LLM systems break every assumption behind that:

- **Non-determinism** — the same input yields different outputs across runs (and across provider-side model updates you didn't opt into).
- **No single right answer** — ten phrasings of a correct summary; grading is a *judgment*, which must itself be engineered.
- **Multi-component pipelines** — retrieval, prompts, tools, and the model each degrade independently; end-to-end scores alone can't localize a regression ([RAG evaluation](./04-rag-patterns.md) splits retrieval metrics from generation metrics for exactly this reason).
- **Silent regressions** — a prompt tweak that fixes one case breaks five others; nothing crashes; quality drifts from 91% to 84% and nobody notices until users do. This is the same failure shape as [harness regressions](./09-harness-engineering.md) — invisible without a suite.

So the goal is statistical: a versioned dataset, scored by versioned evaluators, producing numbers you can compare across versions of everything else.

## The Evaluator Ladder

```mermaid
graph TD
    L1["LEVEL 1 — programmatic (run on everything)<br/>schema validates · regex/contains · exact match ·<br/>code compiles, tests pass · latency/cost bounds"]
    L2["LEVEL 2 — model-graded (run on the suite + samples)<br/>rubric-per-criterion judges · pairwise comparison ·<br/>groundedness vs retrieved context"]
    L3["LEVEL 3 — human (run on disagreements + high stakes)<br/>domain review · preference labels →<br/>which become judge calibration data"]
    L1 --> L2 --> L3
    L3 -.->|"labels calibrate"| L2
```

**Level 1: assertions.** Cheap, deterministic, run on every output forever. Structured-output validity, required/forbidden content, tool-call well-formedness, and — for code — *execution*: the test suite passing is the gold grader, which is why coding evals are the most trustworthy category ([verifier asymmetry](./01-agent-fundamentals.md) again). Squeeze everything you can into this level; every check moved here from level 2 gets faster, cheaper, and unarguable.

**Level 2: LLM-as-judge — useful, biased, calibratable.** A judge model scores outputs against an explicit rubric. The known biases are documented and must be engineered around: **position bias** in pairwise comparisons (swap order, average), **verbosity bias** (longer ≈ scored higher; control for length), **self-preference** (a model favors its own outputs; judge with a different family where it matters), and **score compression** (judges avoid extremes; prefer binary/ternary criteria over 1–10 scales). The non-negotiable: **calibrate the judge against human labels** — a few hundred labeled examples, measure agreement (Cohen's κ); a judge you haven't calibrated is a random number generator with confidence. Rubrics beat vibes:

```python
JUDGE_RUBRIC = """Score the response on each criterion. Answer PASS or FAIL each.

1. GROUNDED: every factual claim is supported by the provided context
2. COMPLETE: addresses all parts of the user's question
3. SAFE_REFUSAL: if the request was out of policy, did it refuse correctly?
4. NO_FABRICATED_CITATIONS: every cited source exists in the context

Question: {question}
Context: {context}
Response: {response}

Return JSON: {"grounded": "PASS|FAIL", "complete": ..., "reasons": {...}}"""
# Binary criteria → higher judge agreement than scalar scores; reasons → debuggability
```

**Level 3: humans** grade what machines can't and, more importantly, **produce the labels that keep level 2 honest**. Route judge-uncertain and judge-disagreement cases here — that's the active-learning loop that grows calibration data where it's most informative.

## The Dataset Is the Asset

Evaluators are replaceable; the curated dataset is the compounding asset. Practices that separate working eval programs from decorative ones:

- **Seed from reality, grow from failure.** Start with 50–200 real cases across intents; every production incident, bad-feedback trace, and support escalation becomes a case (the "promote this trace to the suite" button from [harness engineering](./09-harness-engineering.md)). Synthetic generation fills coverage gaps — edge inputs, injection attempts, every refusal category — but synthetic-only suites overfit to the generator's imagination.
- **Slice, don't average.** One aggregate score hides everything. Tag cases by intent, language, difficulty, and customer tier; report per-slice. "92% overall" with "61% on Japanese billing questions" is the actionable finding ([the same lesson as journey-level SLOs](../11-observability/05-slos-error-budgets.md)).
- **Version it like code** — cases, rubrics, and judge prompts in git; changing a grader re-baselines history, so record (dataset@v, grader@v, system@v) with every run.
- **Watch for leakage and rot:** few-shot examples drifting into the eval set, and stale cases ("respond as of 2024") that punish correct current behavior.

## Agent Evals: Trajectories, Reliability, Cost

Single-response grading misses what makes agents hard. Add three dimensions:

1. **Outcome vs. trajectory.** Grade the end state programmatically (tests pass, ticket updated, file exists — checkable world-state beats output text). Separately grade the *path*: tool-error rate, redundant calls, loop detection, unsafe-action attempts caught by the gate. A success that took 40 flailing turns is a different product than one that took 6.
2. **pass@1 vs. pass^k.** pass@1 ("succeeds on one try") flatters; **pass^k** ("succeeds k out of k") measures what reliability users feel — an agent at 80% pass@1 is at ~33% pass^5. Report both; sell neither alone. Run k ≥ 3 trials per case because single-run agent results are noise.
3. **Cost and latency as first-class metrics.** Cost-per-*solved*-task, turns, wall clock — a "5% smarter" change that doubles tokens is usually a regression ([the same unit economics](./09-harness-engineering.md)).

Public benchmarks (SWE-bench Verified, τ-bench, OSWorld, GAIA) calibrate model+harness choices; they do not measure *your* product — your suite does.

## CI Integration and Statistics

Wire the suite as a gate on every change to prompts, model pins, retrieval indexes, tools, or harness logic:

- **Tiered execution:** level-1 assertions on every commit (fast, free); the judged suite on merge and before any model swap; full k-trial agent runs nightly.
- **Respect the noise.** With 200 cases, an 89% → 91% "improvement" is likely nothing. Use paired comparisons on identical cases (McNemar/bootstrap CIs), set regression thresholds above the noise floor, and treat "no significant change" as a real verdict.
- **Model-swap protocol:** new model versions (including provider-side silent updates — pin versions where offered) run the full suite plus a diff review of *changed* cases before rollout, behind a [flag](../15-deployment/02-feature-flags.md) with online comparison.

### Measurement design

An eval result is an estimate, not a property of a model in isolation. Store the tuple `(system_revision, dataset_revision, evaluator_revision, run_seed, environment)` and the per-case observations. A headline mean without the case-level outcomes cannot be audited, sliced, or re-scored when a grader changes.

For a binary metric with observed pass rate \(\hat p\) over \(n\) independent cases, uncertainty is roughly \(\sqrt{\hat p(1-\hat p)/n}\), though clustered cases and repeated agent trials violate independence. Grouped bootstrap or hierarchical models are safer when cases share users, repositories, or documents. For paired system comparisons, use the same cases and seeds where possible; a discordant-pair analysis is more sensitive than comparing two unrelated means.

Do not spend all evaluation budget on model selection. Repeatedly choosing the best checkpoint on a visible set overfits that set even if each run is statistically “significant.” Keep a locked release set, rotate production-derived cases into development only after the selection decision, and report the number of selection attempts. When metrics are close, the correct decision may be “no evidence of improvement.”

### Judge calibration and disagreement

Calibrate a judge by task slice, not one global agreement number. A judge can agree with experts on easy English cases while failing on code, multilingual input, refusals, or subtle grounding. Keep human labels blinded to model identity, randomize pair order, measure false-accept and false-reject rates for each rubric dimension, and route judge-human disagreements to review.

Use multiple judges only when their errors are sufficiently independent or their disagreement is itself a triage signal. Averaging three correlated judges creates a precise estimate of shared bias. Objective assertions—schema, test result, citation existence, authorization, latency, spend—should remain outside the judge.

### End-to-end versus component attribution

An end-to-end regression can originate in data availability, retrieval candidate recall, prompt rendering, context assembly, model behavior, tool execution, policy, or the grader. Maintain component probes with fixed upstream artifacts: evaluate generation on a frozen evidence packet; evaluate retrieval against known source spans; evaluate the harness with a deterministic mock model; evaluate policy with adversarial tool proposals. This causal ladder is faster than inspecting a final answer and guessing which layer changed.

## Production: Observability and Online Evaluation

Offline evals predict; production confirms. The same scoring machinery runs on live traffic:

- **Trace everything** with OpenTelemetry GenAI semantic conventions — one trace per request/session, spans per model call/tool/retrieval, attributes for model, tokens (in/out/cached), cost, latency, user/tenant tier ([Distributed Tracing](../11-observability/01-distributed-tracing.md)). This is the substrate for everything below.
- **Online scoring on a sample:** run cheap judges (groundedness, refusal-correctness, toxicity) on 1–5% of production responses asynchronously; alert on rate shifts. This is your drift detector — for the model, the retrieval index, *and* the user mix.
- **Capture feedback signals** deliberately: explicit (thumbs, ratings — sparse and biased) and implicit (regeneration, copy events, abandoned sessions, human-agent takeovers — denser and more honest). Wire them to traces so feedback becomes a labeled case in one click.
- **Guardrail metrics as SLIs:** refusal rate, injection-block rate, PII-redaction hits — sudden moves in either direction are incidents ([SLOs](../11-observability/05-slos-error-budgets.md): quality SLOs with burn-rate alerting work here too).
- **Close the loop:** production failure → traced → triaged → promoted to dataset → regression-tested forever. That pipeline, running weekly, is the whole game — eval coverage grows exactly where reality demonstrated you were weak.

---

## Failure Modes

**Benchmark theater.** A public benchmark rises while real task success does not because the harness, data distribution, or verifier differs. Treat public suites as diagnostics; make the production-shaped, versioned suite the release authority.

**Judge drift.** A judge model or rubric update changes scores, creating a fake regression or improvement. Version judges, retain raw decisions, re-score a calibration panel, and do not compare numbers across unqualified judge revisions.

**Leakage and contamination.** Training examples, few-shot demonstrations, or prior model outputs appear in evaluation inputs. Split by entity and time, scan near duplicates, and maintain provenance.

**Averaging away harm.** An overall gain hides a severe regression for a protected language, tenant, refusal class, or high-impact action. Gate non-negotiable slices separately and publish the full distribution.

**Pass-rate optimism.** One successful stochastic run is reported as capability. Use repeated trials and pass^k for reliability; report cost and tail latency alongside success.

**Proxy optimization.** Teams optimize judge score, length, or user engagement and receive more verbose, agreeable, or evasive answers. Pair proxy metrics with expert labels and objective outcome checks.

**Feedback selection bias.** Only users who care leave ratings, and accepted answers are not necessarily correct. Sample exposures, include takeovers and regenerations, and label a representative slice rather than only complaints.

**Trace without promotion.** Failures are observable but never become regression cases. Assign ownership for triage, preserve the environment and source artifacts, and promote a reviewed failure into the development or locked set.

## Decision Framework

Use the cheapest evaluator that can make the decision trustworthy:

| Need | Appropriate evidence |
|---|---|
| Protocol or safety invariant | Deterministic assertion and policy test |
| Artifact correctness | Executable verifier, test suite, database state, or diff checker |
| Groundedness or nuanced quality | Calibrated rubric judge plus human disagreement review |
| High-impact or irreversible decision | Human authority with evidence package |
| Agent reliability | Repeated trials, pass^k, trajectory and side-effect metrics |
| Model/prompt/index migration | Paired slice evaluation, shadow traffic, canary, rollback |
| Production drift | Trace-linked sampled scoring and outcome feedback |

Choose the dataset before the grader, and the acceptance decision before tuning the system. If no evaluator can distinguish an improvement from a regression, the product requirement is underspecified; adding a more confident judge does not solve it. Evals become an engineering function when they own dataset lineage, grader calibration, release thresholds, production sampling, and the path from incident to permanent regression case.

## Key Takeaways

- An eval is a versioned measurement experiment over a dataset, evaluator, system revision, and environment—not a single score.
- Put executable and policy checks at the bottom of the evaluator ladder, use calibrated judges for residual semantics, and reserve humans for disagreement and consequence.
- Evaluate retrieval, generation, harness, policy, and end-to-end outcomes separately so a regression has an actionable owner.
- Report slices, uncertainty, pass^k reliability, cost per solved task, and tail latency; means hide the failures users experience.
- Treat contamination, judge drift, selection bias, and proxy optimization as first-class system failures.
- Production traces become valuable only when reviewed failures are promoted into a maintained, versioned regression corpus.

---

## References

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — the judge-bias catalog (position, verbosity, self-preference)
- [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) — Hamel Husain; the practitioner's playbook this article compresses
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the tracing standard
- [SWE-bench Verified](https://www.swebench.com/), [τ-bench](https://arxiv.org/abs/2406.12045), [OSWorld](https://os-world.github.io/) — agent benchmarks and their grading designs
- [Anthropic: define your success criteria & develop tests](https://docs.anthropic.com/en/docs/build-with-claude/define-success) — eval-first development guidance
- [Harness Engineering](./09-harness-engineering.md) and [RAG Patterns](./04-rag-patterns.md) — where these evals plug into the systems they protect
