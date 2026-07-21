# Lexical Query Execution

Lexical search turns a request into bounded work over dictionaries, postings, stored values, and aggregation structures. Correctness requires consistent text semantics across indexing and querying; efficiency requires selective postings traversal, deadline-aware top-k execution, and admission control for adversarial queries.

Analysis and shard-local execution cover query parsing, tokenization, Boolean and positional retrieval, BM25, filters, top-k pruning, facets, distributed scoring, pagination, and query admission. [Search Index Architecture and Internals](01-inverted-indexes.md) covers segment durability, merge lifecycle, sharding, and recovery; [Ranking and Evaluation](04-ranking-algorithms.md) covers learned reranking and relevance experiments; [Typeahead and Autocomplete](06-typeahead-autocomplete.md) covers autocomplete serving.

## Workload and contract

A query API should expose semantics instead of pretending every search is one string:

```text
SearchRequest {
  query_text
  locale
  lexical_fields[]
  required_filters[]
  optional_clauses[]
  prohibited_clauses[]
  phrase_constraints[]
  sort
  page_cursor
  result_limit
  facet_requests[]
  deadline
  generation_policy
}
```

The response needs enough provenance to interpret it: index generation, query/analyzer version, total-hit relation (`exact`, `lower_bound`, or `unknown`), partial-result flag, timed-out shard IDs, next-page cursor, and ranking-policy version. Returning `total_hits: 10,000` without saying it is a lower bound is an API bug.

Define these service properties:

- **semantic stability**: the same policy version interprets text the same way at index and query time;
- **authorization completeness**: forbidden documents cannot influence returned hits, counts, facets, highlights, or timing beyond the declared threat model;
- **bounded work**: every request has limits on clauses, expansions, visited postings, aggregation memory, shard fan-out, and wall time;
- **snapshot pagination**: a continuation cursor identifies a stable generation and last sort key, or the API explicitly allows duplicates and omissions;
- **partial-result honesty**: missing shards or early termination are visible in the response contract.

## State, versions, and invariants

Lexical execution depends on state that is often misclassified as “configuration”:

- field mappings and which fields are searchable, filterable, sortable, or stored;
- analyzer graphs: Unicode normalization, character filters, tokenizer, token filters, stemming, stop words, synonyms;
- similarity parameters and corpus statistics;
- query-rewrite rules and expansion dictionaries;
- field aliases and schema compatibility;
- expensive-query budgets and tenant policy.

Version that state as one immutable **query policy**. A document records the analyzer/schema version used to produce its tokens. A request records the query-policy version used to interpret it. Silent in-place synonym or stemming changes create a split semantic index: old documents contain tokens from one language and new documents from another.

Important invariants are:

**Index/query analyzer compatibility.** Exact compatibility is not always required—edge n-grams are often produced only at index time—but each field declares a tested pair of index and query analyzers.

**Filters apply before information leaves the authorization boundary.** Security predicates are not best-effort post-processing.

**Score comparability is scoped.** Scores produced by different similarities, model versions, or incompatible shard statistics are not blindly merged as though they were calibrated probabilities.

**Continuation state is tamper-evident.** A cursor binds tenant, normalized query, filters, sort, generation, expiry, and last key; changing any field invalidates it.

## Analysis: preserving meaning before retrieval

Analysis maps text to a token graph, not merely a bag of lowercase words. Consider “C++ developer in Île-de-France.” Unicode normalization, language detection, punctuation handling, token boundaries, synonym expansion, and stemming can each alter meaning. Lowercasing may be correct for one field and destructive for a case-sensitive identifier. Removing “in” may be harmless in prose and wrong in a product code.

The same surface query can require different field treatment:

| Field | Index representation | Query treatment |
|---|---|---|
| title | positions, frequencies, normalized tokens | phrase-aware lexical analysis |
| product SKU | exact normalized keyword | no stemming or fuzzy expansion |
| category | exact ordinal/keyword | structured filter |
| body | positions, frequencies, optional shingles | language-specific analysis |
| timestamp | numeric point plus column value | range filter and sort |

Synonyms are directional product rules. Expanding “laptop” to “notebook” may help recall; equating “apple” with the company in every context damages precision. Multi-token synonyms form a graph: “nyc” can map to “new york city” while phrase positions remain meaningful. Flattening that graph incorrectly can create matches that were never present.

Query understanding can classify navigational, exact-identifier, broad informational, and structured-filter intent, but it should output an explicit plan fragment rather than mutate a query invisibly. Rewrite provenance—original term, rule ID, version, and expansion type—is essential for relevance debugging.

## Retrieval primitives

For a term query, seek the field’s term dictionary and iterate its sorted postings. Boolean conjunction advances iterators to the greatest current document ordinal; skip data avoids decoding every gap. Disjunction maintains a heap or specialized union over candidate ordinals. Negation should normally subtract from a selective positive set; a pure `NOT` over the corpus is effectively a scan.

Phrase queries intersect document candidates first, then compare positions. If terms occur at positions `p`, `p+1`, and `p+2`, an exact three-token phrase matches. Slop widens the positional relation and therefore the work. Store offsets only where highlighting requires them; positions and offsets have material index cost.

Prefix, wildcard, regex, and fuzzy queries enumerate matching dictionary terms and then combine their postings. Their danger is expansion, not syntax. A leading wildcard or broad edit-distance search can enumerate a large fraction of the dictionary. The planner should estimate expansion cardinality, cap it, or use a purpose-built field such as n-grams. “Allow expensive queries” is a capacity policy, not a parser switch.

Filters use exact terms, numeric/geospatial structures, or cached bitsets. They do not contribute to lexical relevance, but they change candidate density. A highly selective filter can lead the plan; a dense filter may be cheaper as a membership check during scoring. The planner needs segment-level statistics because selectivity differs across segments and tenants.

## BM25 as a retrieval score

BM25 rewards terms that are rare in the corpus and frequent in a document, with saturation and document-length normalization. For query term `t` and document `d`, a common form is:

```text
score(t,d) = IDF(t) *
             tf(t,d) * (k1 + 1) /
             (tf(t,d) + k1 * (1 - b + b * length(d) / average_length))
```

`k1` controls term-frequency saturation; `b` controls length normalization. These are corpus- and field-dependent parameters, not magic constants. A title and a long body should generally be separate fields with separate statistics and boosts. Query-time field boosts express product policy; learned ranking can later combine lexical evidence with other features.

IDF and average length are properties of the scoring corpus. With shard-local statistics, the same document can receive different scores depending on placement. Global statistics improve consistency but require distribution, caching, or an extra query phase and become stale. The choice is workload-specific: local statistics may be acceptable when shards are large and randomly distributed; small tenant or time shards can skew severely.

BM25 scores are relative evidence, not calibrated relevance probabilities. A score of 12 in one query is not necessarily “twice as relevant” as 6, nor directly comparable across queries.

## Query planning

Represent a request as a logical tree, then lower it into segment-local iterators and collectors. Planning steps include:

1. validate syntax and resource budgets;
2. normalize and analyze under a pinned query-policy version;
3. expand aliases, synonyms, fuzzy terms, and prefixes within explicit limits;
4. estimate term and filter selectivity from segment statistics;
5. order conjunctions and filters to reduce candidates early;
6. select scoring, sorting, aggregation, and highlighting collectors;
7. attach cancellation checks and per-stage work counters;
8. execute, returning provenance about truncation or partial work.

Plan caches are safe only when the key includes every semantic dependency: tenant, schema/query-policy version, normalized query shape, security-filter shape, requested fields, sort, and aggregation definitions. Cache compiled structure, not user-specific filter values or authorization results, unless those values are part of an isolated key.

Cost estimation should be adaptive. Dictionary document frequency is useful, but correlated clauses violate independence assumptions. Record actual visited postings, matched documents, heap operations, and aggregation cardinality, then compare them with estimates. A planner that never measures its error will repeatedly choose the same pathological plans.

## Top-k execution and dynamic pruning

Sorting every matching document costs too much when the caller needs ten results. A collector maintains a min-heap of the current top `k`; its minimum score is the competitive threshold. Any candidate or postings block whose provable maximum score cannot exceed that threshold may be skipped.

WAND-style algorithms combine per-term upper bounds to choose candidate pivots. Block-Max WAND stores tighter upper bounds for postings blocks, allowing more skips. Correctness depends on **safe upper bounds**: underestimating a block maximum can discard a document that belongs in the true top-k. Quantization, field boosts, phrase contributions, and downstream scoring features must be included conservatively or applied in a later phase.

Dynamic pruning works best after competitive scores establish a high threshold. A broad disjunction with uniformly low scores may still visit many postings. Index sorting, selective filters, or a first-stage candidate budget can help, but early termination changes semantics. The response must distinguish exact top-k from budget-truncated candidates if the product permits approximation.

Two-phase ranking is common:

```text
lexical retrieval over many documents
    -> bounded candidate set
    -> feature extraction
    -> learned or expensive reranker
    -> policy filters/diversification
    -> top results
```

The retrieval candidate size controls both recall and downstream cost. Ranking cannot rescue a relevant document that retrieval never produced. Candidate-recall measurement belongs in the ranking chapter.

## Facets and aggregations

Facets answer questions over the matched population, such as counts by brand or price bucket. They are not free metadata attached to hits. Execution may scan matching ordinals and increment counters through column-oriented values, intersect precomputed bitsets, or use specialized global ordinals.

Clarify the scope of every count:

- all matches before navigation filters;
- matches after all active filters;
- matches excluding the facet’s own filter for drill-sideways navigation;
- an exact count;
- a sampled, shard-truncated, or lower-bound estimate.

High-cardinality group-by can consume memory proportional to distinct values. Bound bucket count, per-shard candidates, and total aggregation memory. Distributed top buckets are subtle: if each shard returns only its local top `m`, a globally frequent value that is never locally top `m` can be missed. Overfetching reduces error but does not automatically prove exactness. Exact aggregation may require a second phase or complete enumeration, which should be a separate workload tier.

## Distributed coordination and deadlines

The coordinator routes to one replica of each relevant shard, propagates an absolute deadline, and merges responses. Absolute deadlines avoid giving every hop a fresh timeout. Each stage reserves time for network return and final reduction; a shard should stop expensive work when its result can no longer arrive usefully.

For global top-k, shards generally return more than `k` when later reranking or score normalization can change order. The coordinator merges with a deterministic tie-break such as `(score, stable_document_id)`. Floating-point differences across architectures and non-deterministic ties otherwise cause pagination churn.

Decide failure semantics by request class. An interactive catalog query may return clearly marked partial results after one shard misses its deadline. A legal discovery or administrative export may require all shards and fail closed. Retrying another replica consumes the remaining deadline and capacity; it is useful for isolated slowness and harmful during shared overload.

Cancellation must travel to dictionary expansion, postings iteration, reranking, highlighting, and aggregation. Dropping the client connection without cancelling shard work leaves a fleet processing results nobody can use.

## Pagination and snapshot semantics

Offset pagination forces each shard to find and retain `offset + k` candidates, so work and coordination memory grow with page depth. Prefer keyset/search-after pagination using the full deterministic sort tuple. A cursor should include:

- index generation or point-in-time handle;
- normalized query and filter digest;
- query-policy and ranking version;
- last sort values and stable document ID;
- tenant, expiry, and signature.

Without a pinned generation, inserts and deletes between pages can cause duplicates or omissions even with a stable sort. Pinning readers consumes resources, so points in time need leases, limits, and expiry. Bulk export is usually better served by a separate scan/snapshot API than by pretending interactive top-k search is an unbounded iterator.

## Capacity and cost model

Use a query corpus, not one benchmark string. For an illustrative peak:

- 6,000 requests/s;
- 10 shards per request;
- p50 of 18 postings blocks decoded per shard, p99 of 2,000;
- measured mean shard CPU service time 2.2 ms on the production query mix;
- 4% of queries request facets, adding a measured 7 ms per participating shard;
- target query CPU utilization 50% to retain tail and recovery headroom.

Base shard traffic is `6,000 * 10 = 60,000` shard requests/s. Base CPU demand is `60,000 * 0.0022 = 132` CPU-seconds/s. Facets add `60,000 * 0.04 * 0.007 = 16.8` CPU-seconds/s. At 50% target utilization, those measured components require about 298 logical cores before coordination, highlighting, cache misses, and failure reserve.

The mean conceals the p99 expansion. Admission needs predicted and observed work units: expanded terms, postings blocks, candidate documents, aggregation buckets, and reranker candidates. Charge expensive requests more tokens than a selective term lookup. Per-tenant concurrency alone is insufficient when one query consumes a thousand times the CPU of another.

Network cost is roughly `shard_requests * candidate_count * bytes_per_candidate`. If 60,000 shard requests/s each return 100 candidates at 40 bytes of metadata, that is about 229 MiB/s before protocol overhead. Returning 1,000 candidates “just in case” multiplies both network and coordinator heap work by ten. Candidate overfetch must be justified by measured recall.

## Concrete failure trace: synonym rollout creates invisible documents

Version 7 indexes “notebook computer” as tokens `notebook` and `computer`. A control-plane edit changes the query synonym rule so “laptop” maps only to token `portable_computer`, but no reindex occurs. Query nodes receive the rule at different times. Requests for “laptop” now miss old documents on updated nodes and behave differently across replicas.

The root cause is treating semantic state as mutable configuration. Recovery is not clearing caches. Roll back to the last compatible query policy, then build a new physical index if the desired index-time representation changes. Canary analysis should replay a fixed judgment set against both versions and compare token traces, zero-result rate, candidate coverage, and top-k changes before broad activation.

The prevention is an analyzer compatibility declaration plus atomic policy activation. Serving telemetry attaches the query-policy and index-schema versions, making mixed-version behavior observable.

## Security and abuse resistance

Parse structured query DSLs from typed input; never concatenate user text into a privileged query. Restrict callable fields and operators by role. Regex, scripts, nested joins, broad fuzzy expansion, large facets, and arbitrary highlighting require stronger budgets or an offline tier.

Authorization filters must be injected by trusted server code and included in every retrieval and aggregation path. They should be visible in audited plan provenance but redacted from user-facing explanations when sensitive. Query, click, and zero-result logs can contain personal, confidential, or adversarial content; minimize, access-control, and expire them.

Highlighting source text can produce injection if snippets are rendered as HTML. Escape source content and add markup only through a trusted renderer. Synonym and rewrite dictionaries are control-plane code in data form: review changes, constrain expansion, and audit who activated them.

## Operations and observability

Break latency into queueing, analysis/rewrite, dictionary seek, postings traversal, scoring, aggregation, fetch/highlight, shard network, and coordinator reduction. Track:

- query rate and latency by normalized query shape, tenant, shard count, and policy version;
- expanded terms, visited postings blocks, scored candidates, and early-termination reason;
- planner estimate versus actual work;
- top-k heap size, candidates returned per shard, and coordinator bytes;
- aggregation buckets, memory, truncation, and accuracy relation;
- timeouts, cancellations, partial results, retries, and work completed after cancellation;
- zero-result and low-result rates by locale and policy version;
- cache hit ratio only alongside saved work and eviction pressure.

Do not put raw high-cardinality query text in metric labels. Use bounded query-shape IDs and keep sampled text in an access-controlled debugging store.

## Verification strategy

- **Analyzer golden tests** pin token graphs for multilingual, punctuation, identifier, emoji, and synonym cases.
- **Differential tests** compare optimized Boolean, phrase, and WAND execution with a simple exhaustive scorer on generated small corpora.
- **Upper-bound tests** prove every pruning bound is at least the actual maximum contribution under all enabled boosts.
- **Facet tests** compare distributed results with an exact single-node oracle, including adversarial shard distributions.
- **Pagination tests** mutate the live index while verifying point-in-time cursors have neither duplicates nor omissions.
- **Authorization tests** attempt leakage through hits, counts, facets, highlights, explanations, caches, and timing-sensitive endpoints.
- **Load tests** replay the production distribution, including expensive-tail queries, while enforcing deadlines and cancellation.
- **Fault tests** delay and omit shards and verify the declared partial-result behavior.

Relevance tests answer whether results are useful; execution tests answer whether the engine produced the intended candidate set within its resource contract. Both are required and should not be conflated.

## Decision framework

Before adding a query feature, answer:

1. What exact semantics does it promise, including empty, partial, and paginated results?
2. Which index structures and analyzer versions does it depend on?
3. What is its worst plausible expansion, postings, memory, and fan-out cost?
4. Can the planner bound or reject that work before saturation?
5. Are top-k and facet results exact, lower-bound, sampled, or approximate?
6. Can authorization be enforced inside every execution path?
7. How will a policy change be shadowed, canaried, rolled back, and explained?

Add fuzzy matching, facets, highlighting, or global statistics only when their measured product value justifies their indexing, latency, and operational cost. Search quality comes from explicit semantics and evidence, not from enabling every operator.

## References

- [Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze: Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
- [Stephen E. Robertson et al.: Okapi at TREC-3](https://trec.nist.gov/pubs/trec3/papers/city.ps.gz)
- [Apache Lucene: BM25Similarity](https://lucene.apache.org/core/10_1_0/core/org/apache/lucene/search/similarities/BM25Similarity.html)
- [Apache Lucene: Query API](https://lucene.apache.org/core/10_1_0/core/org/apache/lucene/search/Query.html)
- [Apache Lucene: IndexSearcher](https://lucene.apache.org/core/10_1_0/core/org/apache/lucene/search/IndexSearcher.html)
- [Andrei Z. Broder et al.: Efficient Query Evaluation using a Two-Level Retrieval Process](https://research.ibm.com/publications/efficient-query-evaluation-using-a-two-level-retrieval-process)
- [Shuai Ding and Torsten Suel: Faster Top-k Document Retrieval Using Block-Max Indexes](https://dl.acm.org/doi/10.1145/2009916.2009924)
- [Elasticsearch: Paginate Search Results](https://www.elastic.co/guide/en/elasticsearch/reference/current/paginate-search-results.html)
- [Elasticsearch: Search Shard Routing](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-shard-routing.html)
