# MapReduce (OSDI 2004): Evidence-First Paper Analysis

MapReduce places the distributed-systems boundary around two programmer-supplied transformations; the runtime owns task placement, shuffle, retry, straggler mitigation, and output publication. Its lasting contribution is the contract that makes a large class of batch computations safely replayable.

## Publication identity and scope

- **Paper:** *MapReduce: Simplified Data Processing on Large Clusters*
- **Authors:** Jeffrey Dean and Sanjay Ghemawat, Google
- **Venue and version:** 6th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2004, pages 137–150
- **System described:** Google's C++ MapReduce implementation running over the Google File System (GFS) on early-2000s commodity clusters

Every number below belongs to that implementation and the paper's test cluster. It is not a benchmark for Hadoop, Spark, a cloud data warehouse, or a current Google service. The paper also describes a **bounded batch** computation. Event time, unbounded streams, iterative in-memory execution, SQL optimization, and exactly-once interaction with arbitrary external systems are outside its design.

Scope here: what the 2004 paper establishes; [Batch Processing](../13-data-pipelines/01-batch-processing.md) covers the general design space.

## Workload and problem boundary

Google repeatedly needed to transform very large collections: construct inverted indexes, aggregate URL access counts, reverse link graphs, sort records, and run graph computations. The per-record logic was usually simple; the hard, repeated work was distributing it across unreliable machines.

The paper narrows that problem deliberately:

1. Input is a finite collection of key/value records stored in a distributed file system.
2. A `Map` function emits zero or more intermediate key/value pairs.
3. The runtime groups all intermediate values for a key.
4. A `Reduce` function consumes one key and its grouped values, then emits output records.

This form gives the runtime a useful independence property: different map invocations can run in any order, and different reduce partitions can run independently after their inputs are available. It does **not** mean every algorithm naturally fits one MapReduce. Multi-stage jobs are possible, but their orchestration and repeated materialization remain visible costs.

## State model and correctness contract

The important state is spread across three durability classes.

| State | Location in the paper | Consequence of loss |
|---|---|---|
| Input and final reduce output | GFS | Replicated by the underlying file system |
| Intermediate map output | Worker-local disk | The completed map task must run again if that worker disappears |
| Scheduling metadata | One master process | Reconstructed from task state while the master lives; a master failure aborts the computation in the implementation described |

Section 3.3 gives the semantic boundary. For deterministic `Map` and `Reduce`, atomic commit of each task's output makes a successful distributed execution equivalent to a non-faulting sequential execution. A map worker writes one temporary file per reduce partition; the master accepts the location from one successful attempt. A reduce attempt writes a temporary output and atomically renames it to its final file. Competing attempts may execute, but only a committed result becomes visible.

With nondeterministic user functions, the guarantee is weaker. Each reduce partition may reflect one completed execution of each map task, but different reduce partitions can observe intermediate data from different successful attempts. That is a precise warning against treating retry as magic exactly-once execution.

The paper's contract also stops at managed files. A mapper that charges a card, mutates a remote database, or sends an email can repeat that side effect after a retry. Such work needs an idempotency protocol; see [Idempotency](../01-foundations/08-idempotency.md).

## Execution protocol

Section 3 and Figure 1 describe six steps.

1. The library splits input into `M` pieces and starts one master plus worker processes.
2. The master assigns idle workers map or reduce tasks.
3. A mapper reads its split, invokes user code, buffers intermediate records, partitions them into `R` regions, and periodically writes those regions to local disk.
4. The master forwards map-output locations to reducers.
5. A reducer fetches its partition from every mapper, externally sorts the records so equal keys are adjacent, and invokes the user's reducer once per key.
6. Successful reducers publish `R` final output files. The caller receives their names, not one implicitly concatenated file.

Two implementation choices carry much of the system's performance.

### Data locality

GFS knows which machines hold each input block. The scheduler tries to run a map on a machine containing the corresponding replica, or at least in the same network switch. This converts many network reads into local-disk reads. Section 3.4 is therefore not a minor optimization: it is why the sort experiment's input rate can exceed its network shuffle and replicated-output rates.

### Task granularity

The paper normally chooses `M` and `R` much larger than the worker count. Fine-grained tasks improve load balancing, speed recovery by redistributing only unfinished work, and expose enough alternatives for locality. The master pays `O(M + R)` scheduling state plus `O(MR)` map-to-reduce location metadata, so granularity is not free. The paper reports input splits typically 16–64 MB for that environment; this is an implementation observation, not a universal modern threshold.

### Partition, ordering, and combiners

The default partitioner hashes the intermediate key, ensuring all values for a key reach one reducer. Applications can supply a different function—for example, range partitioning for ordered output—but then skew and partition-boundary quality become application concerns. Within a reduce partition, the runtime sorts by key; a custom grouping comparator can expose secondary-order patterns.

A combiner performs local partial aggregation before shuffle. It is valid only when that partial operation preserves the reducer's meaning. The paper's word-count sum works; an arbitrary reducer is not automatically a legal combiner.

## Failure and recovery reasoning

The master pings workers. When a worker stops responding:

- in-progress map and reduce tasks become idle and are rescheduled;
- completed map tasks are also rescheduled because their output lived on the failed worker's disk;
- completed reduce tasks remain complete because final output is in GFS.

Reducers already reading from a failed mapper are told about the replacement location. This is lineage-based recovery: regenerate cheap derived state from durable input rather than synchronously replicating every intermediate byte.

The published implementation does **not** replicate the master. Section 3.3 says periodic checkpoints would make recovery possible, but because there is one master, the implementation aborts and lets the client retry the entire job when it fails. That limitation matters when comparing MapReduce with later durable workflow engines.

Section 3.6 addresses slow rather than dead workers. Near completion, the master launches backup copies of remaining tasks; the first successful attempt wins. The paper notes that a faulty machine can make a task deterministically fail, so the system also records failures and can skip a bad input record when the user enables that escape hatch. Skipping is an availability policy that changes the result, not a correctness-preserving retry.

## Quantitative evidence, with methodology

The evaluation in Section 5 used a cluster of roughly 1,800 machines. Each had two 2 GHz Intel Xeon processors, 4 GB of memory, two 160 GB IDE disks, and gigabit Ethernet. The measured network bisection bandwidth was approximately 100–200 Gbit/s. Programs ran on a weekend afternoon when CPUs, disks, and network were mostly idle. Those details are part of every result.

- **Grep (Figure 2):** the job scanned `10^10` records of 100 bytes—about 1 TB—split into 15,000 pieces, looking for a three-character pattern found in 92,337 records. With 1,764 workers, scan rate peaked above 30 GB/s. Maps finished around 80 seconds; end-to-end time was about 150 seconds, including roughly one minute of startup and GFS metadata overhead.
- **Sort (Figure 3a):** the job sorted the same record count and size, with 15,000 map tasks and 4,000 reduce tasks. Input peaked near 13 GB/s, shuffle finished around 600 seconds, and two-way-replicated output finished around 850 seconds. Including startup, elapsed time was 891 seconds. Because the output was replicated twice, the system wrote about 2 TB for a 1 TB logical result.
- **Backup-task ablation (Figure 3b):** disabling backup tasks stretched sort from 891 to 1,283 seconds—a 44% increase. Five straggling reducers accounted for the long tail. This isolates the mechanism's effect in that run; it does not predict a fixed 44% improvement elsewhere.
- **Failure injection (Figure 3c):** the authors killed 200 of 1,746 worker processes several minutes into sort. The cluster scheduler restarted them, lost local map output was recomputed, and the job completed in 933 seconds—42 seconds slower than the normal run.

The evaluation demonstrates feasibility, failure recovery, and the value of locality and speculative execution. It does not compare cost, energy, multi-tenant interference, small-job latency, or alternative batch engines under matched durability semantics.

## Assumptions and limits

- The abstraction favors associative grouping over arbitrary communication. Iterative algorithms repeatedly materialize and reread state.
- A single hot key is owned by one reduce partition unless the application changes the keying strategy.
- The runtime assumes input and final output use a reliable distributed file system; MapReduce itself is not the durable storage layer.
- Deterministic functions make retry semantics clean. External side effects, nondeterminism, or order-sensitive reductions weaken that model.
- The master is a scalability and availability boundary in the published implementation.
- Batch completion, not low per-record latency, is the objective.

These are design choices, not historical defects. They make a narrow, common workload unusually operable.

## What later systems retained and changed

Google's 2010 FlumeJava paper retained data-parallel operators but replaced hand-wired chains of MapReduce jobs with immutable parallel collections, deferred execution, and whole-pipeline optimization. The 2015 Dataflow model retained partitioned data-parallel execution while adding explicit event-time windows, triggers, and correctness/latency/cost choices for unbounded data. Neither result means MapReduce secretly had those semantics; they identify the next boundaries the original model did not cover.

Modern engines also often retain intermediates in memory, build DAGs rather than a fixed map/shuffle/reduce shape, and use distributed schedulers with replicated control state. The durable lesson remains: define a replayable unit, make publication atomic, keep enough lineage to reconstruct lost derived state, and design straggler handling as part of the completion protocol.

## Design review questions

Use the paper as a reasoning framework, not a product prescription:

1. Which inputs are durable, and which intermediate states can be regenerated?
2. Is a task deterministic under retry? If not, what exact weaker result is acceptable?
3. What operation atomically publishes an attempt, and how are losing attempts cleaned up?
4. Does partitioning preserve correctness while avoiding hot reducers?
5. Is locality worth scheduler delay in the actual network/storage topology?
6. How is slow progress distinguished from failure, and what is the resource cost of duplicate attempts?
7. Does the workload require bounded batch semantics, or event-time and continuously updated state?

## Primary sources

- [Dean and Ghemawat, *MapReduce: Simplified Data Processing on Large Clusters* (OSDI 2004), official Google Research PDF](https://storage.googleapis.com/gweb-research2023-media/pubtools/4449.pdf)
- [Google Research publication record for MapReduce](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/)
- [Chambers et al., *FlumeJava: Easy, Efficient Data-Parallel Pipelines* (PLDI 2010)](https://research.google/pubs/flumejava-easy-efficient-data-parallel-pipelines/)
- [Akidau et al., *The Dataflow Model* (VLDB 2015)](https://research.google/pubs/the-dataflow-model-a-practical-approach-to-balancing-correctness-latency-and-cost-in-massive-scale-unbounded-out-of-order-data-processing/)
