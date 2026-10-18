# Data Retention and Compliance Engineering

## TL;DR

Deletion is the hardest write in a modern system, because it has to reach every copy of a fact including the ones that were designed to be immutable. A create statement fans out to a handful of places through code you own. A delete has to reach the primary table, the read replicas, the change-data-capture topic where old records are log entries with their own retention clock, the search index, the cache, the analytics lake holding snapshot copies, the feature store, the training dataset that was frozen last quarter, the embeddings derived from it, the vendor that emails your customers, and ninety days of backups. **Documented**: this is why erasure obligations are so routinely satisfied by policy documents rather than by mechanisms. Regulators did not ask for a distributed protocol; they asked for a right, and most organizations answered with a ticket queue and an intern.

Two properties make retention an architectural concern instead of a legal one. First, retention is a *write path feature*: the decision that determines whether deletion is possible at all is made when the data model, the event schema, and the storage layout are chosen, and it cannot be retrofitted cheaply afterwards. Second, retention has two boundaries and both fail expensively: a ceiling that is too high creates regulatory exposure, discovery surface, and storage cost, while a floor that is too low destroys your ability to debug, defend a chargeback, reproduce a customer conversation, or answer an auditor. **Inference**: the most common real-world failure is not over-retention of everything; it is inconsistent retention per copy, where each store keeps what its own engine happens to make easy.

**Reference design**: every field carries a classification and a retention class declared next to the schema; time-partitioned or key-shredded storage makes bulk expiry cheap; deletion is an idempotent, observable fan-out with a propagation service level and a receipt ledger; backups expire on their own clock and restores re-apply an erasure log; residency is expressed as a routing constraint chosen at account creation; and controls map to automated evidence rather than screenshots. The chapter ends with the failure modes that actually show up in audits and incidents, and the metrics that make the whole thing visibly working instead of merely promised.

## Deletion Is the Hardest Write

A create is local ambition; a delete is global bookkeeping. The asymmetry comes from four sources, each of which needs its own answer:

| Source of difficulty | Why it resists deletion | What it forces in the design |
|---|---|---|
| Physical storage layout | Row-level deletes leave tombstones or dead tuples; space and sometimes visibility lag behind intent | Time-based partitioning, compaction awareness, expiry-friendly layout |
| Immutable and append-only copies | Event logs, lake snapshots, audit trails, and backups are built so that history cannot change | Separate erasure mechanism per copy class, usually re-processing or key destruction |
| Derived artifacts | Aggregates, embeddings, indexes, features, and model weights are functions of the original | Either recompute, or prove the derivation is non-identifying |
| Third-party copies | Vendors hold data under contracts you cannot compile | Deletion APIs, contractual SLAs, and periodic attestation instead of tests |

**Inference**: because every one of those copies has an independent owner and clock, "delete user" cannot be a transaction. It is a workflow: durable intent, idempotent per-copy application, receipts, retries with quarantine, and a reconciliation sweep that finds the copies nobody registered. Systems that model it as a single database call produce exactly two outcomes: silent non-deletion, or an outage caused by a synchronous fan-out to a vendor that was slow today.

There is also a correctness trap on the other side. Deleting aggressively breaks referential integrity that operational code depends on: invoices referencing deleted users, audit trails needing the actor who performed an action five years ago, support tickets quoting messages a customer asked to erase. The standard resolution is pseudonymization plus tombstones: replace identifying attributes with an irreversible local identifier, keep the structural facts and their timestamps, and record which fields were erased and when. This satisfies most erasure obligations while preserving the join keys that keep billing, accounting, and audit trails intact, and it is nearly always cheaper than physical removal.

## Classifying Data Before It Is Written

Retention cannot be decided per query, so it has to be declared per field at schema-authoring time. A workable scheme is small; four classes cover most systems:

| Class | Examples | Default handling |
|---|---|---|
| Ephemeral | Request bodies, debug traces, transient tokens, short-lived sessions | Never persisted, or persisted with a redaction filter and hours-scale TTL |
| Operational | Orders, messages, sessions, device records | Retention tied to the business object's useful life plus a dispute window |
| Regulated | Identity attributes, payment artifacts, health or communications data, IP addresses where in scope | Declared ceiling, pseudonymize early, deletion contract with every recipient |
| Durable-by-law | Invoices, tax records, safety evidence, agreements | Retention floor with legal hold semantics, immutable audit copy separated from the mutable primary |

Two implementation choices matter more than the taxonomy itself. **Declare it in code**, next to the column: an annotation or schema directive that a linter can read means a new field without a class fails a check, which is the only enforcement that survives a deadline. A retention policy that lives in a wiki is a statement of intent; one that generates the expiry job and the audit query is a mechanism. **Classify at ingest, not at query time**, because anything computed later requires scanning everything, and the scan is where projects die.

The second-order benefit is discoverability: a machine-readable classification makes it possible to answer "which stores hold regulated field X" in minutes rather than in a quarter of interviews. That question is the same one threat modeling asks (chapter on abuse cases), and the same inventory answers both.

## TTL Mechanics Across Storage Engines

Every engine deletes differently, and the differences have operational consequences that show up as incidents rather than as documentation errors:

| Store | Deletion mechanism | Operational consequence |
|---|---|---|
| LSM-tree key-value store | Tombstone written; space reclaimed during compaction | A mass expiry creates a tombstone avalanche: reads slow down until compaction catches up, so schedule bulk deletes and watch read amplification |
| B-tree relational store | Row delete leaves a dead tuple until vacuum or cleanup | Long-running transactions block cleanup and turn a retention job into table bloat; batch the deletes |
| Time-partitioned tables or indexes | Drop a partition | The cheapest correct design for append-only event data: expiry is O(1) metadata, no vacuum, no compaction storm |
| Object storage | Lifecycle rules transition and expire objects | Rules must also cover noncurrent versions and incomplete multipart uploads, which are billed and retained invisibly otherwise |
| Search index | Delete by document id, or rebuild | A queryable index is a copy: it needs its own delete path and its own consistency check against the source of truth |
| Cache | Expiry only, no authoritative delete | Deletion correctness depends on invalidation; negative caching can resurrect "deleted" for the TTL window |
| Log-structured event topic | Time or size based segment expiry | Historic events containing personal data expire on the topic clock, which may be longer than the ceiling for the field inside them |
| Encrypted at rest with per-subject keys | Destroy the key (crypto-shredding) | The only O(1) deletion for immutable copies, and it converts a retention problem into a key-management inventory problem |

**Inference**: pick the storage layout that makes the required retention the cheap operation. For high-volume event data, time-partitioned storage plus lifecycle expiry beats any amount of clever delete tooling. Where data is genuinely immutable (audit trails, lake snapshots), per-subject or per-account encryption keys are the only mechanism whose cost does not grow with history, with two honest caveats: key destruction is irreversible by construction, and it only counts if you can prove that plaintext existed nowhere else, which requires the inventory from the previous section.

## The Propagation Problem

Once a subject requests erasure, or a retention ceiling expires, the system needs a fan-out that is durable, idempotent, observable, and reconcilable:

```text
erasure intent (durable, per subject + scope, append-only ledger)
        |
        v
  deletion orchestrator -----> receipt ledger (copy -> status -> timestamp)
        |
        +--> primary store      : pseudonymize + tombstone          [minutes]
        +--> CDC topic          : publish delete event              [seconds]
        +--> search index       : delete by id, verify count        [minutes]
        +--> cache tier         : invalidate + short TTL            [seconds]
        +--> analytics lake     : rewrite/expire rows or partitions [hours]
        +--> feature store      : recompute or drop keyed features  [hours]
        +--> ML datasets        : mark dataset, schedule retrain    [weeks]
        +--> vendor APIs        : contractual deletion + attestation[days]
        +--> backups            : expire on backup clock; re-apply
                                   the erasure log after any restore [n/a]

reconciliation sweep: sample subjects per copy class -> assert absence
```

Three design rules carry most of the weight. **Make application idempotent**, because retries, redeliveries, and replays are guaranteed and a second delete must be a no-op rather than an error or a cascade. **Keep receipts**, so that "did this actually happen in the lake" is a query rather than an argument; a receipt ledger with copy, status, and timestamp turns an audit into a report. **Set and measure a propagation service level**, for example: ninety-nine percent of copies converged within twenty-four hours, worst copy within thirty days for vendors, and any copy that cannot meet the contract gets removed from the architecture or documented as accepted risk in a decision record.

Backups deserve their own rule because they are where correct systems quietly violate retention: never attempt to edit a backup (that destroys its integrity guarantees and is usually impossible), instead expire backups on a clock shorter than the maximum ceiling, and treat every restore as an event that must re-apply the erasure log accumulated since the backup point. **Inference**: if the erasure ledger were not durable and complete, restores would silently resurrect deleted data, which means the retention mechanism and the disaster recovery design (chapter on reconstruction) are one system, not two.

## Backups, Legal Hold, and Restore Correctness

Legal hold is the inverse control: it pauses deletion for a named scope while an investigation, dispute, or obligation is open. It needs real machinery, because a hold that is a Jira ticket will be forgotten and a hold that blocks all expiry turns the whole estate into indefinite retention.

The minimum viable design: a hold ledger naming scope (subject, tenant, date range), reason class, requester role, opening date, and mandatory review interval; an expiry engine that consults the ledger before deleting anything in scope; immutable snapshots taken while a hold is active, so that the held state cannot be altered by ordinary writes; and a release path that triggers catch-up deletion with the same idempotent fan-out. Precedence has to be written down somewhere reviewable: hold beats ceiling, ceiling beats convenience, and any manual override emits an audit event with an owner.

Two failure classes are worth designing against explicitly. A hold applied *after* data has already expired is unenforceable, which argues for a short grace period between "expiry eligible" and "physically deleted" in high-risk domains. And holds that never release create a growing population of data that no one can lawfully delete; publishing the backlog age as a metric makes this visible before it becomes a discovery problem.

## Residency, Sovereignty, and Routing

Data residency turns retention into placement: some datasets may only be stored or processed in a named jurisdiction, and metadata about them is often in scope too. **Inference**: treat region as an immutable property of the tenant chosen at account creation, because retrofitting residency onto an existing global dataset requires moving rows while serving traffic, and that is a migration project with regulatory deadlines attached.

The mechanics that hold up:

- **Partition by jurisdiction from the start.** Region becomes part of the routing key (chapter on multi-region), so control-plane lookups stay global only for data that may be global: tenant id, plan, region tag.
- **Enumerate what counts as an export.** Replication, cross-region read replicas, log shipping, telemetry with user identifiers, support tooling access, and vendor subprocessors are all exports under most readings of the rule, and each needs either a regional instance or a documented lawful transfer mechanism.
- **Watch secondary indexes and caches.** A global search index built from regional data is a residency violation that no schema review will catch; index and cache tiers need their own placement constraints and their own tests.
- **Expect capacity consequences.** Per-region pools mean per-region headroom, dedicated instance families in some regions, and failover plans that cannot assume global spare capacity (chapter on compute platform scheduling). Cell-based isolation (chapter on cells) gives you the same property with a smaller blast radius.
- **Encode placement as policy-as-code.** A deploy or a schema migration that would write regulated data into an out-of-region bucket should fail a check, not fail an audit eighteen months later.

## Compliance as a Testable Property

Auditable controls are engineering artifacts: a control is a statement about system behavior, and statements about behavior can be checked automatically. The mapping exercise is short and its payoff is permanent:

| Control intent | Manual version | Automated version |
|---|---|---|
| Access minimization | Quarterly spreadsheet review of who has what | Identity provider group membership diffed against resource policy; findings block deploys |
| Encryption in transit and at rest | Screenshots of console settings | Terraform or controller-level assertion plus a drift scan that fails on deviation |
| Audit logging completeness | Trust | Test that asserts every mutating handler emits an audit event with actor, action, and object |
| Log immutability | Policy document | Write-once storage configuration plus periodic verification of the hash chain |
| Retention ceilings | Documented schedule | Expiry jobs generated from the schema annotations, plus a sweep asserting no records exceed their class ceiling |
| Deletion effectiveness | Ticket closed | Reconciliation sweep sampling subjects per copy and asserting absence |
| Subprocessor inventory | Slide in the security review | Registry of outbound integrations generated from egress configuration and tested against an allow-list |

Two standards dominate these conversations, and both are more tractable once expressed this way: SOC 2 type II is evidence over a period, so continuous generation beats quarterly scrambling; ISO/IEC 27001 Annex A controls map almost one-to-one onto the automated checks above. The legal obligations underneath have their own engineering shapes: erasure and access requests become an API with a service level and tests (a data subject access request is a query across every copy class, which is exactly the inventory problem), storage limitation becomes a ceiling per class, processor obligations become contract terms plus verified deletion APIs, and transfer restrictions become routing constraints.

**Inference**: teams that treat compliance as a test suite end up with better engineering than teams that treat it as paperwork, because the controls are real. The failure mode of the opposite approach is not failing the audit; it is passing the audit while the system does something else.

## Retention for Derived Data and Models

Derived artifacts are where retention programs go to quietly fail, because their lifecycle belongs to nobody. Three rules keep the surface manageable:

- **Aggregates**: retain when they cannot be traced back to an individual at current cardinality, which usually means a minimum cell size plus suppression of rare categories; document the threshold rather than assuming anonymity survives future joins.
- **Feature stores and indexes**: derive from source with a keyed delete path so that erasure propagates (chapter on feature stores), and record the freshness contract so that a stale feature is not mistaken for a live one.
- **Datasets, embeddings, and weights**: freeze datasets with an explicit snapshot date and classification (chapter on dataset management), then treat deletion as a retraining input rather than pretending a frozen snapshot can be edited. Exact unlearning inside trained weights remains expensive and imprecise; the honest engineering answer is bounded training windows, consent and deletion signals carried through to the next training run, and documented risk acceptance for the interim.

Log and telemetry retention deserves its own note because it is the largest volume in most estates: redact at emit rather than filtering later, cap cardinality of identifiers, and separate the short high-resolution window from the long low-resolution one. Retention decisions here are also cost decisions (chapter on FinOps), which is the rare case where the compliance ceiling and the budget ceiling point the same way.

## Failure Modes of Retention Design

| Failure mode | Symptom | Fix |
|---|---|---|
| Soft delete restored from backup | Deleted accounts reappear after a restore or a replica promotion | Durable erasure log replayed on any restore or promotion |
| Per-copy TTL disagreement | Lake keeps rows the primary dropped years ago | Retention class declared once, expiry jobs generated per copy from it |
| Regulated fields in logs and traces | Debug payload with identifiers reaches a thirty-day telemetry store | Redaction at emit plus a CI check on log schema and trace attributes |
| Deletion amplification storm | Bulk expiry degrades reads cluster-wide | Batch deletes, schedule off-peak, watch tombstone and vacuum metrics |
| Unbounded derived retention | A 2021 training snapshot still contains erased subjects | Snapshot classification plus retrain-on-threshold policy |
| Ceiling below the floor | Chargeback or safety investigation with no evidence available | Durable-by-law class separated from operational data, with hold semantics |
| Crypto-shredding without key inventory | Key destroyed, plaintext still readable elsewhere, or unrecoverable data lost | Key hierarchy documented per class; drill the shred before relying on it |
| Vendor without a deletion API | Erasure request closes when the ticket closes | Deletion SLA in the contract, or remove the vendor from that data path (chapter on dependencies) |
| Timezone and boundary bugs | Records expire a day early or late depending on ingest versus event time | One clock convention per class, tested at the boundary explicitly |

## Key Takeaways

- Retention is a write path feature: schema, storage layout, and event schema decide whether deletion is cheap, and none of them can be changed retroactively without a migration.
- Choose the layout that makes expiry the easy operation (time partitioning, lifecycle rules) and reserve crypto-shredding for genuinely immutable copies where you can prove key coverage.
- Deletion is an idempotent distributed workflow with receipts, a propagation service level, and a reconciliation sweep; anything less produces silent non-deletion.
- Never edit backups; expire them on their own clock and replay the erasure log after every restore or replica promotion.
- Pseudonymize plus tombstone rather than physically removing rows wherever referential integrity, accounting, or audit trails need the join key.
- Legal hold needs a ledger, precedence rules, immutable snapshots, mandatory review intervals, and a release path; otherwise it becomes indefinite retention by accident.
- Residency is a routing constraint decided at account creation, and secondary indexes, caches, telemetry, and support tooling are all exports that need their own tests.
- Map every control to an automated check: continuous evidence beats screenshots, and the same checks make the system genuinely safer rather than merely auditable.
- Derived data (aggregates, features, snapshots, embeddings, weights) needs explicit rules, because exact unlearning is impractical and honest design means bounded training windows plus documented risk acceptance.
- Both boundaries matter: a ceiling that is too high creates exposure and cost, and a floor that is too low destroys your ability to debug, defend, and comply with durable-by-law obligations.

## References

- Regulation (EU) 2016/679 (GDPR), Article 5(1)(e) storage limitation, Article 17 right to erasure, Article 28 processor obligations, Article 30 records of processing, Articles 44 onward on transfers. The legal text behind most retention and residency engineering work.
- California Consumer Privacy Act as amended by the CPRA, including the right to delete and notice-at-collection requirements. A second jurisdiction with different mechanics over the same engineering surface.
- PCI Security Standards Council. *PCI DSS v4.x*, requirement 3 on protecting and minimizing stored account data. The clearest example of a standard that obliges deletion when data is no longer needed.
- ISO/IEC 27001 Annex A controls and SOC 2 Trust Services Criteria category CC6 on logical and physical access. The two frameworks whose control ids most audit requests are actually phrased in.
- NIST SP 800-88 Rev. 1, *Guidelines for Media Sanitization*, NIST, 2014. The clearest published taxonomy of clear, purge, and destroy, and why the difference is a design choice.
- van Oorschot, P. C. and Wiener, M. K., "On Streaming Encryption and Key Management," Selected Areas in Cryptography, 1998; Rosenzweig, P., "Skipjack, Key Escrow, and Split-Key Cryptography," IEEE Annals of the History of Computing, 1997. The academic basis for crypto-shredding: destroy the key when you cannot find the data.
- HIPAA Security Rule, 45 CFR 164.316(b)(2)(i), which requires documentation retained for six years. Worth reading closely: the durable-by-law obligation attaches to documents about your controls, not to the patient data itself.

---

## Related Chapters

- [Verification Strategy for Distributed Systems](./04-verification-strategy-for-distributed-systems.md): absence of regulated data per copy class is an invariant you can and should test continuously
- [LSM Trees](../03-storage-engines/02-lsm-trees.md): tombstones, compaction, and the read amplification that turns a retention job into an incident
- [Change Data Capture](../13-data-pipelines/04-change-data-capture.md): deletes are events too, and lagging consumers quietly resurrect what you removed
- [Lakehouse Table Formats](../13-data-pipelines/05-lakehouse-table-formats.md): snapshot isolation and row-level deletes are how a data lake honors a ceiling at all
- [Disaster Recovery and Data Reconstruction](../15-deployment/05-disaster-recovery.md): restores re-introduce deleted data unless the erasure log is replayed
- [Multi-Tenant Isolation and Tenant Lifecycle Architecture](../06-scaling/12-multi-tenancy.md): offboarding a tenant is retention, holds, and key destruction applied at scale
- [Production Logging Architecture](../11-observability/03-logging.md): the highest-volume retention decision in most estates is what gets emitted in the first place
