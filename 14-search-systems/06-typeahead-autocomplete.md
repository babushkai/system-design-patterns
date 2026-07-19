# Typeahead and Autocomplete

Typeahead predicts useful completions while a person is still typing. It looks like a small search box feature, but its workload is unusually demanding: requests arrive on nearly every keystroke, prefixes are extremely skewed, useful rankings change quickly, and the output can amplify private, abusive, or manipulated queries. The safe architecture is usually a purpose-built suggestion service, not a full search query with a prefix operator.

This chapter owns suggestion generation, prefix data structures, popularity and freshness pipelines, abuse controls, low-latency serving, personalization boundaries, and client interaction. General lexical retrieval belongs to [Lexical Query Execution](02-full-text-search.md); model training and experiment methodology belong to [Ranking and Evaluation](04-ranking-algorithms.md).

## Workload and product contract

A request is more than a prefix:

```text
SuggestRequest {
  raw_prefix
  locale
  surface
  tenant
  subject_context?       // only when policy permits personalization
  result_limit
  request_sequence
  deadline
}

SuggestResponse {
  suggestions[] { display_text, canonical_target?, action, provenance }
  generation
  policy_version
  personalized
  incomplete
}
```

Define what a suggestion represents. It may be a prior query, catalog entity, navigation target, command, or generated completion. Mixing these sources without typed actions creates ambiguity and security problems. A displayed string that navigates to an entity is not the same as a query that should be executed literally.

The contract should specify:

- minimum prefix length and supported normalization by locale;
- maximum results and deterministic tie-breaking;
- freshness bound for trends, removals, and safety policy;
- whether history or coarse context can personalize results;
- response behavior when one source or policy service is unavailable;
- accessibility and client cancellation behavior;
- deletion semantics for source content and user history.

The primary performance objective is usually end-to-end time from a stable keystroke to rendered suggestions, not server p50 alone. Client debounce, network scheduling, stale-response suppression, and rendering all contribute.

## State and invariants

Separate authoritative source state from derived serving state:

| State | Examples | Owner |
|---|---|---|
| eligible candidates | catalog names, approved queries, navigation actions | product/source systems |
| aggregate signals | frequency, unique users, recency, conversions | governed event pipeline |
| safety decisions | deny lists, policy labels, manual removals | trust/safety control plane |
| suggestion generation | prefix map/FST, weights, source versions | build pipeline |
| online overlays | short-lived trends or inventory status | streaming projection |
| subject history | recent eligible actions | isolated personalization store |
| client state | latest request sequence and selected item | client application |

Enforce these invariants:

**Only eligible candidates become visible.** Frequency never overrides authorization, safety, legal removal, inventory, or tenant policy.

**Aggregate publication satisfies privacy thresholds.** A unique or rare private query cannot become a global suggestion merely because it was observed once. Eligibility requires a defined aggregation window, distinct-subject threshold or privacy mechanism, and abuse screening.

**Generation publication is atomic.** A response comes from a complete base generation plus compatible overlays, not half of a rebuilt prefix map.

**Normalization is identical for build and lookup.** Unicode form, case folding, whitespace, punctuation, transliteration, and locale rules are versioned. Display text remains separate from the normalized lookup key.

**A stale client response cannot replace a newer one.** The client renders only the response matching its latest request sequence and current normalized prefix.

**Deletion dominates stale popularity.** Once a candidate is removed at source version `v`, delayed counts or replayed events below `v` cannot reintroduce it.

## Data plane and control plane

The **offline data plane** collects events, validates candidate eligibility, aggregates privacy-safe signals, joins canonical entity state, computes scores, builds compact prefix structures, verifies them, and publishes immutable generations.

The **online data plane** normalizes requests, routes lookups, reads precomputed candidates, applies fresh eligibility overlays and limited contextual ranking, returns typed actions, and emits privacy-minimized telemetry. It should do bounded work independent of the total candidate corpus.

The **control plane** owns normalization versions, source priorities, scoring policy, privacy and safety rules, generation manifests, shard maps, canary state, emergency removals, and rollback. Serving nodes consume signed or authenticated immutable snapshots. If the control plane is temporarily unavailable, they may serve a pinned generation only within its allowed policy age; safety-sensitive removals need a highly available fast overlay.

An end-to-end flow is:

```text
eligible source records + governed interaction events
        -> normalize and aggregate by time window
        -> remove ineligible, rare, and manipulated candidates
        -> score by locale/surface/source
        -> materialize bounded top candidates per prefix
        -> build and verify immutable generation
        -> canary and atomically publish
        -> serve with short-lived trend and removal overlays
```

Every output candidate retains provenance: source type, canonical ID, source version, score-policy version, and safety decision. Raw user text is not automatically a candidate source.

## Prefix structures

### Trie and radix tree

A trie follows one symbol per edge. Lookup is proportional to prefix length, but naïve nodes waste memory on pointers and maps. Path compression combines single-child chains into radix edges. Storing top suggestions at every node makes lookup fast but duplicates candidate references across prefixes and makes online updates expensive.

Tries are useful for mutable overlays and small dictionaries. They are less attractive for a large mostly immutable base unless represented compactly.

### Minimal finite-state structures

A deterministic acyclic finite-state automaton/transducer shares equivalent suffix states among sorted keys and can attach outputs such as candidate-list offsets or weights. This often produces a compact memory-mappable representation with predictable traversal. The trade-off is build complexity and immutability: large changes are usually handled by constructing a new generation, not mutating nodes in place.

Keep the transducer’s job narrow. It maps a normalized prefix to an offset in a candidate table or encoded weighted completions. Candidate metadata and display strings can live in separate immutable blocks. Separating structure from payload lets a serving node page only the needed records and validate block checksums.

### Precomputed prefix lists

Materializing top `M` candidates for each indexed prefix gives nearly constant serving work. It is effective when the corpus changes in batches and `M` is modest. Costs are build amplification and lost long-tail recall: a context-specific candidate not in the precomputed `M` cannot be recovered by reranking.

Do not estimate storage as `queries * average_prefix_length * full_string_size` without measuring prefix sharing and encoding. Build a representative sample and record unique normalized prefixes, candidates per prefix, bytes per edge/state, bytes per candidate reference, display-payload bytes, and compression. The model differs radically for natural-language queries, SKUs, and multilingual entities.

A hybrid design commonly uses an immutable compact base plus small mutable overlays for trends, emergency removals, and newly created entities. Lookup merges bounded lists with stable deduplication and policy filters. Overlay size must be capped; otherwise it becomes an unplanned second primary index.

## Normalization and locale semantics

Normalize lookup keys deterministically under a named version. A pipeline might apply validated UTF-8 decoding, Unicode normalization, locale-aware case handling, whitespace folding, and selected punctuation rules. Do not apply universal accent stripping or transliteration without product evidence: it can merge distinct words and names.

Use Unicode code points or grapheme-aware client behavior consistently. Truncating UTF-8 by bytes can create invalid keys; slicing code points can still split a visible grapheme. The client may display the raw input while the service uses a normalized routing and lookup key.

Locale is part of the key because token boundaries, case, script variants, and useful suggestions differ. A fallback chain such as `language-region -> language -> global` is explicit and provenance-preserving. Global fallback must still respect regional policy and catalog eligibility.

Normalization migrations require dual generations. Build `n+1` from canonical source strings, shadow lookups using captured raw prefixes, compare coverage/collisions, then atomically route clients. Re-normalizing old normalized keys loses information and compounds prior mistakes.

## Ranking suggestions

A suggestion score can combine governed signals:

- distinct-subject frequency in multiple windows;
- recency or trend acceleration;
- successful downstream action, corrected for exposure;
- candidate/source quality and availability;
- prefix match quality and edit cost;
- locale and surface affinity;
- bounded subject history or context when permitted;
- abuse, safety, and concentration penalties.

Raw counts create feedback loops: being suggested increases exposure, which increases count, which increases rank. Log whether an action originated from a suggestion, measure unique subjects rather than raw repeated events, cap contribution per subject/device, and keep an exploration or editorial path for new candidates. Trend scoring should compare a recent window with a longer baseline and minimum support; a ratio with a tiny denominator promotes noise.

Precompute expensive global scoring offline. Online ranking should merge a small number of lists, check fresh eligibility, and apply cheap context. Keep a stable deterministic tie-break such as canonical candidate ID so pagination, caches, and experiments do not churn.

The general experiment and learned-ranking lifecycle is covered in the ranking chapter. Typeahead-specific evaluation includes prefix coverage, accepted-suggestion rate, time to successful action, keystrokes saved, inappropriate suggestion rate, diversity, zero-suggestion rate, stale/removed exposure, and end-to-end latency. Acceptance alone can reward obvious but unhelpful completions, so include downstream success and reformulation.

## Freshness and trending overlays

Batch generations give compact serving and reproducibility; trend signals need shorter latency. Use an overlay keyed by normalized prefix and candidate ID, populated from a durable stream with event-time windows and versioned checkpoints. Its record includes score contribution, support, expiry, source version, and policy version.

The serving merge applies the base list, compatible trend overlay, removal overlay, and optional subject history in a declared order. An overlay newer than the base can reference a candidate absent from base only if it carries enough canonical metadata and eligibility proof. Otherwise defer it until the next build.

Late events update aggregate windows idempotently. Exactly-once transport is unnecessary if aggregation keys include event identity and the state store/checkpoint commit is atomic; blindly incrementing on replay inflates trends. Event time and processing time must be distinct so a backlog recovery does not make yesterday’s query appear suddenly popular.

Emergency removal is a separate high-priority negative overlay with independent availability and audit. It should suppress a canonical candidate across base, trends, personalization, caches, and replicas within a tested bound. The next full generation incorporates the removal, after which the overlay can expire safely.

## Typos and fuzzy completions

Fuzzy lookup expands work rapidly, especially for short prefixes. One edit on a two-character prefix can touch much of the dictionary and produce surprising suggestions. Gate it by minimum grapheme length, locale, candidate support, and a strict visited-state/candidate budget.

Options include traversing the term automaton jointly with a Levenshtein automaton, consulting a deletion/spelling index, or correcting only after an exact-prefix miss. Keyboard adjacency and phonetic rules are locale/input-method-specific features, not universally valid edit costs.

Expose when a suggestion corrects rather than completes the prefix. The client should not silently replace user input. Exact matches generally retain a protected path so fuzzy popularity does not displace what the user actually typed.

## Personalization boundary

Personal suggestions can use a subject’s explicitly eligible recent queries, entities, or actions. Store them separately from the global aggregate index, under per-subject authorization, retention, and deletion. Merge a bounded personal list at request time; never let one person’s raw history enter global suggestions without aggregate privacy controls.

Incognito, logged-out, child, enterprise, and regulated contexts may require different behavior or no personalization. The response declares whether personalization was applied. A feature-store failure falls back to global eligible suggestions, not to another user or an unscoped cache entry.

Cache keys include tenant, normalized prefix, locale, surface, generation, and policy-relevant context. Personalized results are either not shared or keyed to a protected subject/cohort identity with appropriate isolation. Caching only by prefix is a common privacy leak.

## Sharding, caching, and serving

Prefixes are highly skewed: empty and one-character prefixes dominate traffic, while long prefixes have enormous key cardinality. Serve or reject empty prefixes deliberately. Cache hot short prefixes at the edge or process, but bound policy staleness and make removals override cache entries.

For a sharded prefix map, build and lookup must use the same stable normalization and routing function. One safe specification is:

```text
route_key = NFC(casefold_under_locale(raw_prefix))
shard = first_64_bits(SHA-256(
    "typeahead-prefix-shard:v4" || 0x00 || UTF8(route_key)
)) mod shard_count
```

The domain separator prevents accidental coupling with other hashes; SHA-256 avoids process-randomized language hashes; explicit Unicode/UTF-8 rules keep platforms consistent. Modulo routing makes shard-count changes disruptive, so production manifests should map many fixed virtual buckets to physical shards. The hash selects a virtual bucket, and the generation pins the bucket map.

An alternative is range partitioning by prefix, which supports local traversal but creates hot alphabet/script ranges. Replicate hot ranges or split them adaptively under a versioned map. Consistent hashing helps movement but does not solve a single hot key: cache and replicate those entries.

Serving nodes memory-map immutable generations, verify manifest and block checksums before readiness, warm the highest-traffic prefixes, and atomically swap the active generation. Keep the prior generation open for rollback and for in-flight requests. Never overwrite files under active readers.

## Client protocol

Debounce reduces requests but adds visible latency. Choose it from measured typing intervals and network latency, and allow immediate requests after actions such as paste or navigation. Cancel obsolete requests when possible, but also attach a monotonically increasing `request_sequence`; networks and servers can deliver cancelled work late.

The client renders a response only if its sequence and normalized prefix match current state. Keyboard navigation, selection, focus, and screen-reader announcements follow the appropriate combobox/listbox accessibility pattern. Suggestions must not steal focus or execute on mere highlight. Typed actions distinguish “submit query,” “navigate,” and “run command.”

Client telemetry records impressions only for suggestions actually rendered, with position and generation. Do not log every raw prefix indiscriminately; prefixes can contain names, secrets, medical terms, or pasted credentials. Apply collection minimization at the client boundary.

## Capacity and cost model

Consider an illustrative product workload:

- 40,000 peak active typing sessions;
- measured 2.4 suggest requests/s per active session after debounce and cancellation;
- 80% process/edge cache hit ratio on the observed prefix distribution;
- cache misses route to one base shard and one overlay service;
- measured uncached service CPU is 0.35 ms and response payload averages 1.2 KiB;
- target service CPU utilization 45% for burst and generation-swap headroom.

Ingress is `40,000 * 2.4 = 96,000` requests/s. A measured 80% cache hit ratio leaves 19,200 uncached requests/s, not including invalidations or cold starts. CPU demand is `19,200 * 0.00035 = 6.72` CPU-seconds/s; at 45% target utilization, this component needs about 15 logical cores before TLS, overlay fan-out, logging, and failure reserve.

Response bandwidth at the edge is `96,000 * 1.2 KiB`, about 110 MiB/s before protocol overhead. Origin payload is lower with caching, but a policy-version bump can invalidate the hot set at once. Load-test a cold generation and stagger rollout. If clients fail to cancel or debounce because of a regression, ingress can multiply with typing rate; enforce server-side per-session/tenant budgets.

For build capacity, measure input candidates, distinct normalized prefixes, emitted candidate references, sort/shuffle bytes, final generation bytes, and peak temporary disk. If 30 million candidates produce a measured 420 million prefix-candidate pairs at 24 encoded bytes before compaction, the intermediate is about 9.4 GiB. This is an illustrative arithmetic input, not a universal amplification factor; real strings, locales, and prefix caps determine it.

## Concrete failure trace: query-log poisoning

An attacker sends the same offensive phrase from thousands of automated requests. The daily pipeline counts raw events, builds a new generation, and the phrase becomes the first completion for a popular two-character prefix. Caches propagate it globally before manual detection.

Containment publishes an emergency negative overlay and purges/bypasses affected cache entries. The generation is rolled back, but rollback alone is insufficient if the old generation also contains the phrase. Repair removes poisoned events from the governed aggregate, rebuilds from a known checkpoint, and records the affected policy/generation lineage.

Prevention combines distinct-subject support, per-actor contribution caps, bot/fraud signals, candidate allow/deny policy, minimum support for trends, manual review for high-exposure prefixes, staged canary publication, and automated diffs of newly promoted suggestions. Telemetry alerts on abrupt score/share changes and on candidates entering high-traffic prefixes for the first time.

## Security, privacy, and abuse resistance

Autocomplete is an output publication system. Apply stronger review to suggestions than to ordinary search results because the product proactively displays them. Enforce tenant and region boundaries before all caches. Escape display text; typed navigation targets must be validated server-side and limited to approved schemes/routes.

Protect build inputs and manifests with authenticated writers, provenance, checksums, and audited activation. A compromised popularity pipeline can control prominent text without changing application code. Limit candidate length, token count, Unicode control characters, and payload size to prevent rendering abuse and resource exhaustion.

Query-event collection needs purpose limitation, access control, short retention where feasible, deletion workflows, and aggregation privacy. Redact or drop patterns likely to be credentials or sensitive identifiers before durable logging. Differential privacy may be appropriate for published aggregates, but it does not replace eligibility, minimum support, or abuse controls.

## Operations and observability

Track by locale, surface, tenant, prefix-length bucket, generation, and policy version:

- end-to-end and server queue/service latency;
- request rate, cancellation, stale-response discard, and client debounce behavior;
- cache hit rate plus saved origin work and policy staleness;
- empty response, result count, exact/fuzzy/fallback path, and overlay contribution;
- generation age, load/verification failures, active-reader count, and rollback status;
- source-to-suggestion lag and deletion/removal propagation time;
- candidate support, new high-exposure candidates, concentration, and abuse removals;
- accepted suggestions, successful downstream actions, reformulation, and keystrokes saved;
- privacy threshold drops and raw-event access/audit anomalies.

Never place raw prefixes or subject IDs in metric labels. Keep tightly sampled, access-controlled diagnostics with redaction and explicit retention.

Runbooks cover offensive/unsafe suggestion, legal or privacy removal, corrupt generation, stale trend overlay, hot-prefix overload, bad normalization rollout, cross-tenant cache leak, and event-pipeline replay. Exercise emergency suppression end-to-end, including client and CDN caches.

## Verification strategy

- **Normalization golden tests** cover scripts, combining marks, locale-specific case, whitespace, emoji, malformed input, and client/server parity.
- **Structure tests** compare trie/FST/precomputed lookups with a simple sorted-list oracle over generated candidate sets.
- **Routing tests** verify domain-separated SHA-256 virtual-bucket assignment identically across languages and architectures.
- **Generation tests** kill builders at file boundaries, corrupt blocks, and prove only complete manifests activate.
- **Ranking tests** replay fixed aggregate windows and prove deterministic ties, support thresholds, and removal dominance.
- **Privacy/security tests** attempt rare-query promotion, bot amplification, tenant cache crossover, unsafe display text, and history leakage.
- **Client tests** reorder and delay responses, exercise keyboard/screen-reader behavior, and verify impressions only after rendering.
- **Load tests** include hot one-character prefixes, cache flush, cold generation, overlay outage, and a client request-amplification bug.
- **Migration tests** shadow raw prefixes across normalization/generation versions and reconcile canonical candidate coverage.

The golden corpus should include adversarial and sensitive cases maintained by the appropriate reviewers, not only popular benign prefixes.

## Decision framework

Choose the simplest serving structure that satisfies the measured corpus and freshness needs:

- use a database/index prefix query for a small, low-QPS, non-sensitive catalog where its worst-case work is bounded;
- use a mutable trie/radix structure for small dynamic dictionaries;
- use an immutable FST or precomputed prefix map for a large read-heavy base;
- add a bounded streaming overlay only when trend/new-entity freshness creates measurable value;
- add fuzzy matching or personalization only with explicit latency, privacy, safety, and evaluation contracts.

Before launch, answer:

1. Which sources are allowed to publish suggestions, and under what privacy threshold?
2. What normalization and locale contract is shared by build, routing, lookup, and client?
3. How are popularity feedback and manipulation controlled?
4. What is the cold-cache/hot-prefix capacity plan?
5. How quickly can any candidate be suppressed across generations and caches?
6. Can a complete generation be reproduced, canaried, rolled back, and reconciled?
7. Which end-to-end quality and safety evidence justifies each ranking change?

Autocomplete quality is inseparable from publication governance. A fast prefix lookup that leaks or amplifies harmful text is a failed design.

## References

- [Jan Daciuk et al.: Incremental Construction of Minimal Acyclic Finite-State Automata](https://aclanthology.org/J00-1002/)
- [Apache Lucene: FST Package](https://lucene.apache.org/core/10_1_0/core/org/apache/lucene/util/fst/package-summary.html)
- [Surajit Chaudhuri and Raghav Kaushik: Extending Autocompletion to Tolerate Errors](https://doi.org/10.1145/1376616.1376705)
- [Holger Bast and Ingmar Weber: Type Less, Find More: Fast Autocompletion Search with a Succinct Index](https://doi.org/10.1145/1148170.1148248)
- [NIST: Privacy Framework](https://www.nist.gov/privacy-framework)
- [W3C: WAI-ARIA Authoring Practices, Combobox Pattern](https://www.w3.org/WAI/ARIA/apg/patterns/combobox/)
- [Unicode Standard Annex #15: Unicode Normalization Forms](https://unicode.org/reports/tr15/)
- [Unicode Standard Annex #29: Unicode Text Segmentation](https://unicode.org/reports/tr29/)
