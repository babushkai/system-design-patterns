# Ranking and Evaluation Systems

Retrieval selects eligible documents; ranking orders them. A production ranking system is a versioned, deadline-bound decision pipeline over candidates, features, policy constraints, and experiment assignment. Its evidence must distinguish product improvement from logging bias.

Ranking covers multiple stages, feature consistency, labels, learning-to-rank objectives, offline judgments, online experiments, personalization, diversification, rollout, and relevance operations. [Lexical Query Execution](02-full-text-search.md) and [Vector Retrieval Systems](03-vector-search.md) cover candidate generation; [Search Index Architecture and Internals](01-inverted-indexes.md) covers the physical index lifecycle.

## Decision contract and objective

Write the ranking contract before choosing a model:

```text
RankRequest {
  query_context
  candidates[] { document_id, retrieval_source, retrieval_score, source_version }
  subject_context
  policy_version
  experiment_assignments
  deadline
}

RankResponse {
  ordered_results[] { document_id, decision_id, score_components? }
  candidate_generation
  feature_snapshot
  model_version
  policy_version
  degraded_mode
}
```

The business objective is normally multi-dimensional. Search can optimize successful task completion while constraining latency, zero-result rate, unsafe-content exposure, seller or source concentration, cancellation, and long-term retention. A single proxy such as click-through rate can favor clickbait, duplicate results, accidental clicks, or answers that cause users to return because the first answer failed.

Define primary metrics, guardrails, and invariants separately. A relevance model may optimize expected graded relevance; policy may require eligibility and regional restrictions; diversification may constrain near-duplicates; the serving system must meet a latency/error budget. Mixing these into one opaque score makes changes difficult to audit and rollback.

## State and invariants

Ranking depends on versioned state across online and offline systems:

| State | Examples | Required provenance |
|---|---|---|
| candidate set | lexical, vector, curated, sponsored | retriever and generation |
| feature definitions | freshness, popularity, lexical match, embeddings | schema, transformation code, owner |
| feature values | online point reads and offline training rows | event/source time, entity version |
| labels | judgments, clicks, conversions, reformulations | collection policy and window |
| model | trees, linear weights, neural reranker | artifact digest, training data/checkpoint |
| decision policy | eligibility, blending, diversity, fallbacks | reviewed immutable version |
| experiment state | unit, allocation, start/end, exclusions | assignment namespace and analysis plan |

Enforce these invariants:

**Candidate provenance is preserved.** A model cannot learn or operate correctly if it cannot tell whether a document came from lexical, vector, or another source and which candidates were never retrieved.

**Training features are point-in-time correct.** A training example at event time `T` uses only feature values knowable at `T`. Joining today’s popularity or inventory onto last month’s click logs leaks the future.

**Online and offline feature semantics match.** A feature has one definition with tested batch and serving implementations, or one shared transformation. Same field name is not evidence of parity.

**Every served decision is attributable.** Given a decision ID, operators can recover candidate generation, feature versions or source checkpoints, model digest, policy version, experiment assignment, and degraded path without logging sensitive raw values unnecessarily.

**Hard policy is outside learned preference.** Authorization, legal removal, inventory eligibility, and safety restrictions cannot be traded away because a model score is high.

**Fallbacks preserve safety.** A feature or model outage may reduce relevance; it must not bypass eligibility or tenant boundaries.

## Ranking as a staged architecture

Search fleets rarely apply the most expensive scorer to every document. A common pipeline is:

1. lexical, vector, and other retrievers produce candidate lists;
2. a fusion/deduplication stage builds a bounded union with provenance;
3. a cheap first-stage scorer reduces thousands of candidates to hundreds;
4. online features are fetched in batches;
5. a richer model reranks tens or hundreds;
6. policy, diversity, and business constraints produce the final list;
7. result rendering logs an impression only when the item was actually observable under the measurement policy.

Each boundary has a recall and cost contract. Candidate union size, per-source quota, first-stage cutoff, feature timeout, and reranker cutoff are model inputs even when they live in configuration. Evaluate them together. Increasing reranker quality is irrelevant if an aggressive first stage drops the useful document.

The coordinator passes an absolute deadline and reserves time for downstream rendering. If a feature batch misses its budget, the policy chooses a named degraded mode: default the noncritical feature, use a smaller model, fall back to a lexical score, or fail the request. Arbitrary partial feature vectors make scores incomparable and are difficult to train for.

## Candidate fusion and score boundaries

Lexical scores, vector distances, popularity, and model probabilities are not naturally comparable. Raw linear interpolation is valid only after a calibrated or empirically justified transformation. Rank-based fusion such as reciprocal rank fusion avoids assuming score scale, but introduces its own rank constant and per-source depth policy.

Candidate identity and deduplication require product semantics. Two documents may represent the same item, translated editions, variants, or canonical/duplicate URLs. Deduplicate before expensive ranking where possible, but retain group provenance so diversity and attribution remain correct.

Track candidate recall using a judged or behavioral target: the fraction of known relevant outcomes present in the candidate union and after every cutoff. Slice it by query class, language, filter selectivity, freshness, and retriever. End-to-end NDCG alone cannot reveal that vector retrieval improved coverage while the first-stage scorer discarded those additions.

## Feature platform and point-in-time correctness

Features fall into different operational classes:

- **request features**: locale, device class, query length, intent;
- **query-document features**: lexical score, phrase match, embedding similarity;
- **document features**: quality, freshness, inventory, authority;
- **subject features**: prior interactions or preferences, when permitted;
- **cross features**: subject-category affinity or geographic distance;
- **contextual policy signals**: experiment, surface, regulatory region.

For every feature define entity key, type, valid range, event-time semantics, freshness, default behavior, access policy, and lineage. Online feature lookup should be a batched request keyed by candidate IDs and a feature-view version, not hundreds of serial RPCs. The response identifies source timestamps so the ranker can enforce freshness bounds.

Offline training joins are temporal. Suppose a click occurs at 10:05 and product price changes at 10:07. The row for that click must use the last price at or before 10:05, subject to the product’s event-time correction policy. A standard key join to the latest table leaks 10:07 into the past. Late events require replayable, versioned correction rather than silently rewriting labels without changing the dataset version.

Parity tests replay captured requests through offline and online transforms, comparing values with declared tolerance. Monitor feature missingness, staleness, range, distribution, and correlation, not just RPC health. A perfectly available feature service can serve semantically wrong data.

## Learning-to-rank approaches

**Pointwise models** predict a label or outcome for each candidate independently. They are simple and work with standard regression/classification infrastructure, but the loss does not directly represent ordering and can be dominated by the many easy negatives.

**Pairwise models** learn preferences between document pairs for the same query. RankNet uses a probabilistic pairwise loss; tree-based ranking methods can weight pairs according to potential metric change. Pair construction and sampling matter: all quadratic pairs are rarely necessary, and biased pairs reproduce the collection policy.

**Listwise methods** optimize a surrogate over a ranked list or weight gradients by ranking-metric impact. LambdaMART combines boosted trees with lambda gradients related to ranking swaps. It remains strong for heterogeneous tabular features and predictable CPU serving. Neural cross-encoders can model rich query-document interaction but consume substantially more compute, so they usually operate on a small reranking set.

The training objective should correspond to the product task and label quality. A more complex loss does not repair biased clicks, missing candidates, or future leakage. Evaluate model families under the same candidate set, features, hardware, and deadline.

Calibration is separate from ordering. A ranker can order well while its scores are poor probabilities. Calibrate only when downstream policy interprets score magnitude, and validate by slice and time. Recalibration can change thresholds without changing order, which deserves its own rollout.

## Labels, judgments, and behavioral bias

Human judgments should use a written rubric tied to user intent, with graded labels where degrees of usefulness matter. Sample queries from production strata plus known failure classes; uniform random query sampling overweights the head only if requests themselves are the desired unit, while uniform unique-query sampling can overrepresent rare noise. Record sampling weights.

Measure assessor agreement and adjudicate ambiguous categories. Disagreement can reveal an underspecified product objective rather than poor assessors. Keep sensitive or safety domains with specialized reviewers and access controls.

Clicks are cheap but biased by what the old ranker displayed. Position, presentation, snippet, device, trust, and prior exposure influence examination. Non-click is not a clean negative when the result was below the viewport. Conversion labels have delay and attribution ambiguity; dwell time can reward confusing content.

Randomized interventions can estimate examination propensities, but they impose user cost and require ethical review. Inverse-propensity weighting can correct a known logging policy under overlap assumptions, while increasing variance when propensities are small. Clip or regularize weights only with an explicit bias/variance analysis. Counterfactual estimators cannot evaluate actions the logging policy never took with positive probability.

Record the serving propensity or experiment policy needed for later analysis. Trying to reconstruct it from application version months later is unreliable.

## Offline evaluation

Use multiple metrics because each encodes different user behavior:

- **Precision@k** measures relevant fraction in the displayed prefix;
- **Recall@k** measures coverage of known relevant items;
- **MRR** emphasizes the rank of the first relevant answer;
- **NDCG@k** discounts lower positions and supports graded relevance;
- **ERR** models a user’s probability of stopping after satisfying results;
- **candidate recall** measures whether the pipeline preserved useful documents before final ranking;
- **coverage and exposure metrics** detect empty slices or concentration.

Always report the judgment set, cutoff, gain and discount definitions, unjudged-document treatment, confidence interval, and slices. An aggregate improvement can hide regression in one language or intent. Paired analysis at the query level is usually more powerful than treating every query-document pair as independent because systems rank the same queries.

The evaluation corpus needs a lifecycle. Add production failures, remove invalid judgments only with provenance, refresh time-sensitive queries, and keep stable holdouts to detect overfitting. If every tuning iteration is chosen on one “test” set, it has become training data.

A useful failure taxonomy includes no candidates, wrong intent, analyzer/rewrite error, stale or forbidden document, retrieval omission, feature error, misordering, duplicate cluster, insufficient diversity, and rendering/measurement error. Assign each class to the owning subsystem. Otherwise relevance teams compensate for indexing bugs with ranker weights.

## Online experimentation

An experiment declares hypothesis, unit of randomization, eligibility, control/treatment policy versions, primary metric, guardrails, minimum detectable effect, power, duration, stopping rule, and analysis method before exposure.

Assignment must be stable and namespace-separated. One suitable construction is:

```text
bucket = first_64_bits(SHA-256(
    "search-ranking:experiment-2026-07" || 0x00 || subject_id
)) mod 10_000
```

The namespace prevents accidental correlation with another experiment that hashes the same subject ID. Do not use a process-language `hash()` whose seed or implementation can change between runs. Assignment, eligibility, and exposure are distinct: analyze a subject as exposed only under the prespecified policy, while retaining intent-to-treat data where required.

Power depends on baseline rate, minimum effect, variance, allocation, clustering, repeated observations, and test design. For illustration, a two-sided independent two-proportion normal approximation with baseline CTR 5%, a 5% relative lift (0.25 percentage point absolute), alpha 0.05, and power 0.80 gives 122,124 observations **per arm** before attrition, clustering, peeking correction, or multiple metrics. The number is not a default; use historical variance and the actual randomization unit.

Users issue repeated queries, so query impressions are not necessarily independent. Analyze at the assignment unit or use cluster-robust/hierarchical methods. Account for novelty, weekday cycles, delayed conversions, bots, carryover, and interference between marketplace participants. Do not stop when a p-value first crosses a threshold; use a fixed horizon or a prespecified sequential design.

Guardrails should include latency/errors, abandonment, reformulation, safety/authorization outcomes, zero results, source concentration, and downstream task success where relevant. Segment analysis is diagnostic, but uncontrolled slicing and metric fishing inflate false discoveries. Confirm important slice effects in a follow-up or with a multiplicity-aware plan.

Interleaving can compare two rankers in one result list and often detects preference with less traffic. Team-draft and probabilistic methods must attribute items correctly, randomize presentation fairly, handle duplicates, and respect policy. Interleaving estimates preference under the mixed presentation; it does not replace a longer experiment for absolute business effects, latency, or ecosystem feedback.

## Personalization and diversification

Personalization requires an explicit benefit, consent/legal basis, retention policy, and non-personalized fallback. Keep subject features isolated by tenant and purpose. Do not expose sensitive attributes to a general feature store merely because they correlate with engagement.

Cold-start policy can use request context and aggregate priors. Missing history is a normal state, not an error. Cap the contribution of unstable or sparse personal features so one anomalous event cannot dominate ranking. Provide deletion and “reset personalization” workflows that remove both online state and future training influence according to policy.

Diversification trades marginal relevance for list utility. Maximal marginal relevance, intent coverage, source caps, and category constraints are different policies. Express constraints separately from the base score and evaluate both utility and exposure. A source cap can improve variety or unfairly suppress the only relevant source; validate by query intent.

Fairness questions require a declared subject: users, content providers, sellers, regions, or classes protected by law or policy. Measure exposure conditional on relevance and opportunity where appropriate. There is no universal fairness metric; governance chooses the constraint, while the platform makes its effect measurable and auditable.

## Serving, rollout, and rollback

Package model artifact, feature-view version, candidate policy, thresholds, and post-ranking policy into one immutable ranking release. A model digest alone cannot reproduce behavior. Validate schema and feature compatibility before loading; reject partial activation.

Roll out through offline replay, shadow scoring, small stable canary, controlled experiment, and staged traffic expansion. Shadow traffic validates latency, feature availability, score distributions, and disagreements without affecting users, but it cannot measure behavioral outcomes. Canary assignment should be deterministic and independent of experiment assignment unless the plan intentionally combines them.

Rollback switches the complete release pointer. Keep the prior model and compatible online feature views warm for the rollback window. If a new release causes writes (such as updating subject state), define whether those effects are forward-compatible. A read-path rollback that leaves incompatible learned state behind is incomplete.

Fallback order is explicit and tested: full model; reduced feature/model release; stable non-personalized model; lexical baseline. Hard policy runs in every mode. Track degraded-mode rate as a product-quality metric, not only an availability detail.

## Capacity and cost model

Consider an illustrative peak workload:

- 8,000 searches/s;
- 1,200 fused candidates per request;
- cheap first-stage scorer measured at 0.8 microseconds per candidate on target hardware;
- 150 candidates receive an online feature batch averaging 4 KiB total response per request;
- 80 candidates receive a reranker measured at 35 microseconds per candidate;
- target CPU utilization 55%, excluding feature-service CPU.

First-stage demand is `8,000 * 1,200 * 0.8 µs = 7.68` CPU-seconds/s. Reranking is `8,000 * 80 * 35 µs = 22.4` CPU-seconds/s. Together they require about 55 logical cores at 55% target utilization, before serialization, policy, logging, and failure reserve. The expensive stage dominates despite seeing far fewer candidates.

Feature traffic is `8,000 * 4 KiB`, about 31 MiB/s of response payload before protocol overhead and replication. More important is `8,000 * 150 = 1.2 million` candidate-feature lookups/s logically; batching and locality determine whether the feature tier sees 8,000 RPC/s or a fan-out storm. Model p50 alone is insufficient: measure queueing and p99 by candidate count and degraded path.

Training and evaluation cost includes point-in-time joins, judgment collection, artifact retention, and replay. A larger candidate log can dominate storage: at 8,000 requests/s, logging 1,200 candidate rows of even 100 bytes would produce about 894 MiB/s. Log compact provenance, sample detailed traces under a governed policy, and retain enough to reproduce decisions without indiscriminately copying content.

## Concrete failure trace: offline/online feature skew

A new `document_age_hours` feature is computed from event time in training but from a delayed warehouse ingestion timestamp online. Offline NDCG improves, shadow scores look plausible, and the canary begins. Recently updated documents receive large, inconsistent age values online, so they disappear from results. Aggregate feature missingness remains zero.

Containment switches the complete ranking release to the prior version. Diagnosis compares decision-linked online feature values with point-in-time offline recomputation and finds a distribution shift only in fresh documents. Repair moves both paths to the same versioned definition, adds event/source timestamp to the feature record, and blocks activation unless parity tests pass on boundary cases.

This incident demonstrates why feature availability is not correctness. Alerts need freshness, range, distribution, and parity slices; the release manifest needs the feature definition, not merely a field name.

## Security, privacy, and abuse

Treat ranking inputs as untrusted. Validate numeric finiteness, categorical domains, feature vector length, candidate count, and model schema. Crafted documents can manipulate token repetition, embeddings, freshness, or popularity. Detect coordinated interaction fraud, but keep fraud models and enforcement reviewable; opaque suppression creates its own governance risk.

Model artifacts and feature definitions are executable decision inputs. Sign or digest them, restrict publication, audit activation, and isolate loading. Training data and judgment tools can expose sensitive queries and documents; apply least privilege, redaction, retention, and purpose limitation.

Explanation endpoints must not reveal private features, hidden policy, model internals exploitable for abuse, or the existence of unauthorized documents. Operational explanations can expose version and coarse score components to authorized staff while storing detailed traces in a protected system.

## Observability and operations

Join technical and quality telemetry by decision and release version:

- candidates per source and recall through each cutoff;
- feature latency, missingness, staleness, range, and drift;
- model queue/service time, batch size, score distribution, and saturation;
- fallback/degraded-mode rate and reason;
- duplicates, diversity, policy removals, and insufficient-result rate;
- offline metric suite by stable slice;
- online primary, guardrail, and long-term metrics with uncertainty;
- assignment/exposure balance and sample-ratio mismatch;
- source-to-label delay and training-serving version skew.

Metric labels use bounded release and slice IDs, not raw queries or subjects. A protected sampled trace should reconstruct a decision from candidate provenance through final policy. Runbooks cover feature-tier failure, corrupt model artifact, quality regression, assignment bug, delayed label pipeline, and emergency policy removal.

## Verification strategy

- **Feature contract tests** compare offline/online transforms and point-in-time joins, including late events.
- **Model tests** validate schema, finite outputs, deterministic behavior within declared tolerance, and monotonic/domain constraints where required.
- **Pipeline replay** runs logged candidates through old and new complete releases and attributes every ordering change.
- **Candidate tests** measure recall after each retriever and cutoff against judged outcomes.
- **Experiment tests** verify stable domain-separated assignment, allocation, mutual exclusions, exposure logging, and sample-ratio alerts.
- **Policy tests** prove hard restrictions and tenant boundaries in full and every fallback mode.
- **Load tests** vary candidate counts, feature latency, batch size, and model saturation under deadlines.
- **Fault tests** inject stale/missing features, model-load failure, malformed values, partial candidate sources, and telemetry outage.

Before launch, require a claims ledger: which quality, latency, capacity, safety, and business claims are supported by which artifact. “The model passed” is not a deployment decision.

## Decision framework

Adopt additional ranking complexity only after answering:

1. Is the limiting error candidate recall, feature quality, ordering, policy, or presentation?
2. What user outcome and guardrails define success?
3. Are labels and logging policy adequate for the proposed objective?
4. Can offline features be reconstructed point-in-time and served consistently online?
5. What candidate budget and latency reserve does the model require at peak?
6. How are personalization, diversity, safety, and business rules separated and governed?
7. Can the complete release be shadowed, canaried, attributed, and rolled back?
8. Which exact offline and online evidence will justify promotion?

Often the best next change is better judgments, candidate coverage, or feature correctness, not a larger model.

## References

- [Christopher J. C. Burges et al.: Learning to Rank using Gradient Descent (RankNet)](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
- [Christopher J. C. Burges: From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)
- [Kalervo Järvelin and Jaana Kekäläinen: Cumulated Gain-Based Evaluation of IR Techniques](https://doi.org/10.1145/582415.582418)
- [Olivier Chapelle et al.: Expected Reciprocal Rank for Graded Relevance](https://dl.acm.org/doi/10.1145/1645953.1646033)
- [Thorsten Joachims et al.: Unbiased Learning-to-Rank with Biased Feedback](https://www.cs.cornell.edu/people/tj/publications/joachims_etal_17a.pdf)
- [Lihong Li et al.: A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146)
- [Filip Radlinski and Nick Craswell: Optimized Interleaving for Online Retrieval Evaluation](https://www.microsoft.com/en-us/research/publication/optimized-interleaving-for-online-retrieval-evaluation/)
- [NIST/SEMATECH e-Handbook of Statistical Methods: Comparing Proportions](https://www.itl.nist.gov/div898/handbook/prc/section3/prc33.htm)
- [Google: Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
