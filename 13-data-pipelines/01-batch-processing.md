# Batch Execution: DAGs, Shuffle, and Safe Reprocessing

## TL;DR

A batch pipeline transforms a **bounded, versioned input** into a **new output generation**. Correctness requires a pinned input boundary, repeatable task attempts, attempt-isolated files, and one commit protocol for output publication. Performance is governed by wide dependencies—shuffle volume, hot partitions, spill I/O, and the slowest task—not the average worker.

Use batch execution when the input can be bounded and recomputation fits the freshness objective. Use continuous execution when waiting for the bound is too slow. Some platforms use both, but that should mean one declared result contract with two operational paths, not two implementations whose business logic quietly diverges.

---

## 1. Begin with the Workload Contract

“Run this SQL every day” is a schedule, not a contract. Before choosing an engine, define:

| Contract field | Question that must have one answer |
|---|---|
| **Input boundary** | Which immutable snapshot, partitions, transaction IDs, or source offsets belong to this run? |
| **Completeness rule** | Is a day complete at source close, after a lateness allowance, or only after upstream manifests arrive? |
| **Result key** | Is output replaced by partition, table snapshot, model version, or another generation identifier? |
| **Repeatability** | Which nondeterminism is forbidden or pinned: current time, random seed, mutable lookup, unordered tie? |
| **Correction policy** | Does late or corrected input create a replacement generation, a compensating record, or an audit revision? |
| **Failure contract** | Can a retry repeat tasks? Can a reader ever observe a partial run? |
| **Freshness objective** | Deadline from input close to committed output, including retries and validation. |
| **Retention** | How long are the exact source snapshot, code, config, and dependency versions replayable? |

A useful run identity is a tuple such as:

> (pipeline version, logical interval, input snapshot IDs, parameter digest, attempt)

The first four fields identify the logical computation. The attempt identifies physical retries. Output publication must be idempotent with respect to the logical identity, not merely unique per attempt.

### Invariants

Required invariants:

1. Every committed result was computed from one recorded input boundary.
2. A failed or speculative attempt cannot become visible independently.
3. A retry either publishes the same logical result or fails validation; it does not append a second copy.
4. Readers resolve one committed generation and never infer completeness by listing a directory.
5. Reprocessing does not destroy the currently served generation.
6. Data quality failure prevents publication, while cleanup failure does not revoke an already committed result.

These invariants are more precise than “the engine is exactly once.”

---

## 2. Batch, Replay, and the Lambda/Kappa Choice

Batch and stream execution answer different questions about **when input is considered available**. Neither is inherently more accurate.

- A bounded job can finalize only after its declared input boundary is closed.
- An event-time stream can emit complete, updateable results if its lateness and correction contract permits it.
- A replayable stream can also be processed as bounded offset ranges.

The classic Lambda architecture maintains a batch-built base view plus a low-latency speed view. It reduces freshness while retaining a rebuild path, but creates a difficult merge invariant:

> served(key) = merge(base at boundary B, delta strictly after B)

Two code paths, two state models, and an ambiguous boundary often produce semantic drift. Use this pattern only when the low-latency path cannot itself be replayed or when the correction algorithm genuinely needs a bounded dataset.

The Kappa approach keeps one retained event log and rebuilds with another consumer group. It avoids dual transformation logic, but it is safe only when:

- source retention covers detection, recovery, and catch-up;
- the new processor can consume faster than new data arrives;
- state can be rebuilt before the cutover deadline;
- external lookups and schemas are versioned or reproducible;
- a new output generation can be validated before routing changes.

“One API for batch and streaming” does not erase these operational differences. A bounded backfill and a continuously checkpointed job still have different failure, capacity, and publication paths.

---

## 3. Execution Model: Control Plane and Data Plane

~~~mermaid
flowchart TB
    subgraph CP["Control plane"]
        O["Orchestrator<br/>dependencies and retry policy"]
        C["Coordinator / driver<br/>DAG, stages, task attempts"]
        M["Run metadata<br/>input boundary, lineage, status"]
        Q["Quality gate"]
        P["Commit coordinator<br/>publish generation"]
        O --> C
        C --> M
        Q --> P
    end

    subgraph DP["Data plane"]
        I[("Versioned inputs")]
        A["Stage A tasks<br/>scan and map"]
        S[("Shuffle blocks<br/>local disk / service")]
        B["Stage B tasks<br/>join and aggregate"]
        T[("Attempt-isolated files")]
        R[("Committed table or manifest")]
        I --> A
        A --> S
        S --> B
        B --> T
        T --> R
    end

    C -.assigns.-> A
    C -.assigns.-> B
    M -.pins.-> I
    T -.metrics and samples.-> Q
    P -.atomic publication.-> R
~~~

The **control plane** decides what should run, records state, retries attempts, validates results, and publishes them. The **data plane** scans, exchanges, transforms, and writes bytes. Keeping the distinction explicit prevents a common mistake: adding worker capacity to a job blocked on catalog commits, scheduler limits, or a missing upstream manifest.

### Stage DAG

A logical query becomes a directed acyclic graph of operators. The engine pipelines **narrow dependencies** when each output partition depends on a small, known set of input partitions. A **wide dependency** requires redistribution—join, group, distinct, sort, or repartition—and usually creates a stage boundary.

For each stage, record:

- input and output partitioning;
- expected rows and bytes, not just row count;
- whether aggregation can combine locally before exchange;
- memory-intensive operators and spill policy;
- retryable side effects;
- the materialization or shuffle recovery boundary.

The critical path is the longest chain of stage completion barriers. Adding parallelism to a fast scan stage does not shorten a job dominated by one skewed reduce partition.

---

## 4. Shuffle Is a Distributed External Sort

A hash aggregation or equi-join usually follows this physical sequence:

1. Each upstream task assigns records to destination partitions using a partition function.
2. Records accumulate in serialized buffers; combinable aggregates may reduce them locally.
3. When memory pressure crosses an engine-specific limit, buffers sort and spill to local storage.
4. Spill runs are merged into shuffle blocks indexed by destination partition.
5. Downstream tasks fetch their blocks from every upstream task or a remote shuffle service.
6. The downstream operator merges, hashes, sorts, joins, or aggregates the fetched records.

The exchange includes serialization, memory copies, disk writes and reads, network transfer, checksums, and retry amplification. “The input is only 2 TB” says little if a many-to-many join emits 12 TB of intermediate rows.

### Capacity model

Define measured quantities for one representative run:

- <code>D</code>: compressed bytes read from the source.
- <code>A</code>: shuffle amplification, shuffle bytes divided by <code>D</code>.
- <code>W</code>: concurrent workers.
- <code>b_scan</code>, <code>b_net</code>, <code>b_disk</code>: sustainable per-worker scan, network, and local spill throughput after contention.
- <code>C</code>: total CPU-seconds of parsing, decompression, expression, hashing, and serialization.
- <code>c_cpu</code>: usable CPU cores.
- <code>k_skew</code>: slowest-partition work divided by mean-partition work.

Ignoring overlap and startup, lower bounds are:

> scan time ≥ D / (W × b_scan)
>
> shuffle transfer time ≥ (A × D) / (W × b_net)
>
> spill time ≥ spill bytes / (W × b_disk)
>
> CPU time ≥ C / c_cpu

The stage duration is at least the largest resource bound, then inflated by skew, retries, queueing, and barriers. Do not add all bounds blindly: engines overlap some I/O and CPU. Measure the execution timeline to learn which resources overlap.

Cost must include more than worker-hours:

> run cost = compute time + storage requests + shuffle service + temporary bytes + output bytes + retry work + maintenance work

Object-store request count and temporary retention can dominate small-file jobs even when total bytes are modest.

---

## 5. Skew: The Slowest Partition Sets the Barrier

Uniform hashing balances keys, not work. One customer may own half the events; one JSON value may be ten times larger; a join key such as null may collect unrelated rows into one partition.

### Detect skew before changing the plan

Inspect distributions at several levels:

- bytes and records per input partition;
- output bytes per map task and destination partition;
- task duration, CPU time, spill bytes, fetch wait, and garbage collection;
- estimated and observed heavy hitters;
- join fan-out per key, including null and default values.

Ratios are diagnostic, not universal alarms. Compare the slowest partition to the median and explain the bytes or CPU behind the difference.

### Mitigation algorithms

Choose a remedy that preserves semantics:

- **Map-side combine:** partially aggregate associative and commutative operations before shuffle.
- **Heavy-key salting:** split a known hot key into subkeys, aggregate or join in parallel, then perform a second merge. The second phase is mandatory.
- **Skew-aware join:** replicate only the hot-key slice of the smaller relation or split the larger side adaptively.
- **Range-boundary sampling:** estimate balanced ranges for ordered data, then monitor distribution drift.
- **Pre-aggregation:** reduce event grain before an expensive join when the query permits it.
- **Broadcast join:** avoid a shuffle only when the serialized relation, per-worker memory, network fan-out, and concurrent broadcasts have been measured.
- **Isolate exceptional keys:** process them in a separate path with an explicit merge contract.

Increasing the partition count cannot split a single indivisible key in a hash partitioner. It may create more scheduler overhead and files without reducing the hot task.

---

## 6. Memory, Spill, and Local-Disk Failure

Memory pressure occurs in execution buffers, hash tables, sort runs, deserialized objects, caches, and language runtimes. A job can have free aggregate cluster memory while one task exceeds its container.

An external algorithm is healthy when it spills predictably and local storage has headroom. It is unhealthy when repeated spills, merge passes, garbage collection, and re-fetches form a feedback loop:

~~~text
large partition
  → hash table exceeds memory
  → spill runs multiply
  → local disk and GC slow the task
  → shuffle blocks live longer
  → node disk fills
  → task and peers fail
  → retries recreate more shuffle
~~~

Operate local storage as a finite cache:

- reserve capacity for concurrent tasks, spill runs, shuffle blocks, and retry overlap;
- monitor bytes spilled, spill count, merge time, disk utilization, inode/file-descriptor pressure, and shuffle fetch failures;
- separate executor loss from application logic failure;
- use a remote shuffle service only after understanding its own durability, capacity, and garbage-collection boundary.

Caching an intermediate is useful when reuse saves more CPU and I/O than materialization costs. It is not a correctness mechanism; lost cached partitions may be recomputed.

---

## 7. Stragglers and Speculative Execution

A straggler is an unusually slow attempt caused by skew, a degraded host, noisy neighbors, remote reads, or runtime pauses. Speculative execution launches another attempt and accepts a winner.

Speculation is safe only when attempts are isolated:

- output paths contain run, stage, task, and attempt identity;
- the committer chooses one winner and discards losers;
- task code has no non-idempotent external side effect;
- metrics and accumulators are not mistaken for transactional output.

### Concrete failure trace: the losing attempt publishes

1. Task attempt A writes <code>part-17</code> slowly to the final directory.
2. The scheduler launches speculative attempt B.
3. B finishes first and writes another object named <code>part-17</code>.
4. The job succeeds and readers start scanning.
5. A resumes after the success signal and completes its upload.
6. Depending on naming and storage semantics, A overwrites B, creates a duplicate, or leaves a mixed multipart object.

The scheduler did exactly what speculation promises; the output protocol was wrong. Fix it by writing immutable attempt-specific objects, electing a winner, and publishing only a manifest that references winner files. Direct per-record calls to billing, email, or an HTTP API do not become safe because the task is “retryable”; move effects behind an idempotency key or a committed outbox.

---

## 8. Commit Protocols: Visibility Is a Metadata Decision

A safe file-producing job separates **write** from **publish**:

1. Pin the input boundary and allocate a logical run ID.
2. Each attempt writes immutable files under an attempt-isolated namespace.
3. The coordinator records the winning attempt for every task.
4. Quality checks validate file counts, schema, row constraints, and reconciliation totals.
5. One atomic protocol action publishes the winning file set: a table snapshot, manifest pointer, transactional log entry, or warehouse transaction.
6. Losing and orphan files are cleaned later with a safety window.

Filesystem rename-based committers and object-store committers are not interchangeable. Object stores commonly implement rename as copy plus delete; a directory listing is not a transaction. Prefer a table or manifest protocol whose atomicity boundary is documented for the actual catalog and storage system.

### Coordinator crash after data writes

If the coordinator crashes after all files exist but before metadata publication, readers must continue seeing the previous generation. On retry, the coordinator may reuse validated files or write new ones, then attempt publication. Orphan cleanup must distinguish abandoned objects from an in-flight or retryable commit.

“Exactly once” here means **one logical output generation becomes visible**, although tasks, uploads, and commit attempts may execute more than once.

---

## 9. Backfill and Reprocessing Protocol

Never repair history by deleting or overwriting the live output first. Treat a backfill as a migration:

1. **Freeze the specification.** Record code artifact, configuration, schema mappings, lookup versions, input snapshots, and nondeterministic seeds.
2. **Choose the range.** State whether downstream cumulative tables, model features, and aggregates outside the range are affected.
3. **Estimate blast radius.** Source read load, shuffle and object-store bandwidth, table commit conflicts, downstream cache invalidation, and retention.
4. **Write a shadow generation.** Use a new table branch, snapshot, versioned prefix, or partition generation.
5. **Throttle and checkpoint progress.** Make range chunks independently retryable without exposing them.
6. **Validate.** Compare counts, sums, distinct keys, null distributions, sampled records, and business invariants against source and prior output.
7. **Publish atomically.** Switch a metadata pointer or merge through a transactional table protocol.
8. **Observe and retain rollback.** Keep the old generation through the rollback and reader-retention window.

For Kappa-style replay, use a new consumer group and new sink generation. Let it catch up, compare it with the current view at the same source coordinate, then switch routing. Clearing the live sink before replay converts a repair into an outage.

### Replay feasibility

If backlog is <code>L</code> events, arrival rate is <code>r_in</code>, and sustainable processing rate is <code>r_proc</code>, catch-up time is:

> catch-up time = L / (r_proc − r_in), requiring r_proc > r_in

Retention must exceed detection delay + provisioning + replay time + validation + cutover margin. This same bound matters for streaming recovery and is developed in the next chapter.

---

## 10. Operations, Security, and Governance

### Orchestration and migrations

- Model dependencies as data availability and committed generations, not only clock times.
- Distinguish retryable infrastructure errors, deterministic data errors, and quality-gate failures.
- Cap concurrent backfills separately from recurring production jobs.
- Roll out query or schema changes with a shadow run and representative historical intervals, including worst-case skew.
- Version stateful UDFs, runtime libraries, lookup tables, and serialization formats.
- Prefer expand/contract changes when old and new readers overlap.

### Security

- Give workers read access only to declared inputs and write access only to their attempt namespace.
- Reserve publication rights for the commit coordinator.
- Encrypt shuffle, temporary, checkpoint, and output data; local spill is still data at rest.
- Redact sensitive values from task logs, dead-letter samples, and query plans.
- Carry classification and deletion obligations into derived tables and retained generations.
- Audit who launched a backfill, which code and input it used, and which generation it replaced.

### Observability

Track the pipeline at three levels:

**Run:** input boundary, queue delay, stage critical path, retries, committed generation, freshness, validation status.

**Stage:** input/output/shuffle bytes, partition distribution, spill, fetch wait, CPU, memory pressure, lost executors, speculative attempts.

**Data:** schema compatibility, row and key counts, null rates, duplicate keys, distribution drift, source-to-output reconciliation, late partitions.

Do not alert only on “job failed.” A successful job that published incomplete or duplicate data is the more dangerous incident.

---

## 11. Verification and Failure Injection

Before trusting a pipeline:

- run the same logical job twice and compare canonicalized output;
- kill workers during scan, shuffle, spill, and write;
- kill the coordinator immediately before and after metadata publication;
- force two attempts for one task and prove only one output is referenced;
- inject a dominant key, an oversized record, empty input, and maximum join fan-out;
- exhaust local spill space in a staging environment;
- delay one upstream partition beyond the completeness boundary;
- replay an already committed run and prove publication is idempotent;
- run readers pinned before, during, and after a commit and verify snapshot consistency;
- start orphan cleanup while a deliberately slow writer is active and prove live files survive.

Verification should assert invariants at the protocol boundary, not only compare a few example rows.

---

## 12. Decision Framework

| Question | Favors bounded batch | Favors continuous stream | May justify both |
|---|---|---|---|
| Can the input close before the freshness deadline? | Yes | No | Different serving and correction deadlines |
| Is the algorithm naturally global or iterative? | Often | Only with large/complex state | Stream for provisional view, batch for declared revision |
| Can the source be replayed for the full recovery window? | Snapshot retained | Log retained | Batch snapshot supplements short log retention |
| Can one implementation run in both modes with identical semantics? | Useful | Useful | Prefer this before dual codebases |
| Can replay catch up faster than arrival? | Not required during a bounded run | Required after outage | Batch rebuild may be safer |
| Must readers see one atomic generation? | Natural fit | Requires transactional/upsert sink | Common serving contract |

Prefer the simplest execution mode that meets freshness, replay, and correction requirements. Add a second path only after naming the invariant that one path cannot satisfy.

---

## Primary References

- Jeffrey Dean and Sanjay Ghemawat, [MapReduce: Simplified Data Processing on Large Clusters](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/).
- Matei Zaharia et al., [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf).
- Apache Spark, [RDD Programming Guide: Shuffle Operations](https://spark.apache.org/docs/latest/rdd-programming-guide.html#shuffle-operations).
- Apache Hadoop, [OutputCommitter API](https://hadoop.apache.org/docs/current/api/org/apache/hadoop/mapreduce/OutputCommitter.html).
- Apache Hadoop, [S3A Committers: Architecture and Implementation](https://hadoop.apache.org/docs/current/hadoop-aws/tools/hadoop-aws/committer_architecture.html).
- Tyler Akidau et al., [The Dataflow Model](https://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf).
- Apache Iceberg, [Table Specification](https://iceberg.apache.org/spec/), for snapshot and optimistic-commit mechanics used by file tables.

---

**Next:** [Stream Execution](02-stream-processing.md) applies the same contract-first reasoning to unbounded input, event time, state, checkpoints, replay, and backpressure.
