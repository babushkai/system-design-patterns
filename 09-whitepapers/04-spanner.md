# Spanner (OSDI 2012): Evidence-First Paper Analysis

Spanner's distinguishing contribution is not merely “a database synchronized by atomic clocks.” It composes **per-shard consensus, multi-version state, pessimistic transactions, and a bounded time API** so commit timestamps respect real-time order. TrueTime does not eliminate coordination; it makes the remaining uncertainty explicit and sometimes waits it out.

## Publication identity and version discipline

- **Paper:** *Spanner: Google's Globally-Distributed Database*
- **Authors:** James C. Corbett, Jeffrey Dean, Michael Epstein, Andrew Fikes, Christopher Frost, J. J. Furman, Sanjay Ghemawat, Andrey Gubarev, Christopher Heiser, Peter Hochschild, Wilson Hsieh, Sebastian Kanthak, Eugene Kogan, Hongyi Li, Alexander Lloyd, Sergey Melnik, David Mwaura, David Nagle, Sean Quinlan, Rajesh Rao, Lindsay Rolig, Yasushi Saito, Michal Szymaniak, Christopher Taylor, Ruth Wang, and Dale Woodford
- **Venue and base version:** 10th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2012, pages 251–264
- **Expanded version:** ACM Transactions on Computer Systems 31(3), 2013, reorganized with minor corrections and additional material

The analysis below targets the OSDI 2012 design. Where the expanded paper provides clearer organization, it is identified as the 2013 version. Neither paper is current Cloud Spanner product documentation.

[Distributed Transactions](../02-distributed-databases/07-distributed-transactions.md), [Consensus](../02-distributed-databases/08-consensus-algorithms.md), and [Distributed Time](../01-foundations/05-distributed-time.md) cover the component concepts. This chapter focuses on their joint proof obligation.

## Problem and service contract

Google wanted a database that could automatically shard data across datacenters, synchronously replicate it, move it as load or policy changed, and still offer transactions over arbitrary rows. Earlier manually sharded MySQL deployments made resharding an application and organizational event; weaker NoSQL stores could not protect cross-row invariants.

Spanner's published contract includes:

- read-write transactions with external consistency;
- lock-free read-only transactions and reads at a timestamp;
- synchronous Paxos replication within each data partition;
- automatic movement of data through named placement units;
- a semi-relational schema with primary keys and interleaved tables;
- globally meaningful commit timestamps derived from bounded time uncertainty.

“Global” is a placement capability, not a promise that every transaction has one latency. A transaction spanning Paxos groups and continents pays the required consensus and two-phase-commit paths.

## Physical and logical state

A deployment is a **universe**, divided into zones. A zone contains a zonemaster and many spanservers; location proxies help clients find the spanserver serving data. A spanserver holds 100–1,000 tablet-like data structures in the paper.

Each spanserver tablet stores mappings from `(key, timestamp)` to bytes, giving multi-version concurrency control. A tablet is associated with a Paxos state machine; replicas maintain the same ordered mutations, and one replica holds a timed leader lease. Reads and writes use the leader when the operation requires current serialization, while sufficiently up-to-date replicas can serve historical reads.

The application data model is more structured than Bigtable's. Tables have schemas and primary keys. Child rows can be **interleaved** under parent rows so related records share key prefixes and placement. A **directory** (a set of rows with a common prefix) is the unit of placement. A directory can split into fragments as it grows, but the placement abstraction is what administrators move and replicate.

## Invariants that make the design work

### Paxos-group ownership

Each data partition's committed mutations are ordered by its Paxos group. Leader lease intervals for one group must not overlap. Consensus tolerates a minority of unavailable replicas; it cannot commit new mutations without a quorum.

### External consistency

If transaction `T1` commits before transaction `T2` begins in real time, Spanner promises `timestamp(T1) < timestamp(T2)`. This is stronger than serializability alone because the serialization order respects observable real-time order.

### Safe historical reads

A replica may serve timestamp `t` only after its safe time is at least `t`. Section 4 defines safe time from Paxos progress and, where prepared transactions exist, the minimum prepare timestamp. This prevents a snapshot from omitting a transaction that could later commit at or before the requested time.

### Atomic multi-group commit

Every participant either exposes the transaction's writes at the chosen commit timestamp or none does. Paxos makes each participant's prepare/commit state durable; two-phase commit coordinates the outcome across groups. Paxos and 2PC solve different problems and neither replaces the other.

## TrueTime and timestamp assignment

`TT.now()` returns an interval `[earliest, latest]` guaranteed to contain absolute time. Its half-width, usually called epsilon, bounds time uncertainty; TrueTime does not assume zero clock skew. Time masters use GPS and atomic-clock references with different failure modes; per-machine daemons poll several masters, reject inconsistent sources, and increase uncertainty conservatively between polls.

For a read-write transaction, strict two-phase locking protects reads and writes. Participants first prepare through their Paxos groups. The coordinator leader chooses a commit timestamp no earlier than all prepare timestamps and no earlier than `TT.now().latest`. Before exposing success, it performs **commit wait** until `TT.after(commit_timestamp)` is true.

The reasoning is the important part:

1. Choosing at least `TT.now().latest` places the timestamp after any absolute time that could be “now” at assignment.
2. Waiting until absolute time is definitely past that timestamp ensures another transaction beginning after the response receives a later time bound.
3. Therefore real-time commit order is reflected in timestamp order.

Commit wait adds at least the remaining uncertainty to latency. Better clock bounds improve performance; they are not optional for correctness under this protocol.

## Transaction and read protocols

### Read-write path

The client sends reads to leaders, which acquire read locks. Writes are buffered at the client until commit. For a transaction confined to one Paxos group, the leader can coordinate locally. Across groups, the client chooses one participant as 2PC coordinator; participants acquire write locks, log prepare through Paxos, and return prepare timestamps. The coordinator logs the outcome, chooses the commit timestamp, waits out TrueTime uncertainty, then tells participants to commit and release locks.

This path can block on lock conflicts and may leave prepared participants waiting for coordinator recovery. Replicating transaction state makes the wait recoverable, not nonexistent.

### Read-only path

A read-only transaction is declared read-only in advance. Spanner assigns one timestamp and reads every key at that snapshot without acquiring data locks. A replica can serve the request when its applied state and transaction safe time cover the timestamp. Client-specified stale reads can often use nearby replicas and avoid the leader.

The predeclaration matters. A transaction cannot take lock-free snapshot reads and later decide to write without changing the protocol and its guarantees.

## Failure and recovery behavior

- **Replica failure:** the Paxos group continues if a quorum remains. A replacement replays replicated state.
- **Leader failure:** another replica can lead after lease safety permits it. The timed lease avoids simultaneous leaders but can delay recovery after an abrupt failure.
- **Zone failure:** groups whose remaining replicas form quorums continue; placement policy decides which correlated failures are survivable.
- **Transaction coordinator failure:** durable prepare and coordinator records allow recovery to determine the outcome. Locks may remain until that resolution.
- **Time-reference failure:** TrueTime widens its interval using conservative drift bounds. Correctness is maintained by longer waits; availability or latency can degrade if uncertainty becomes too large.

The failure model assumes bounded clock drift is enforced and Byzantine data corruption is outside the consensus protocol. Operators must keep replicas across genuinely independent failure domains; “five replicas” on one correlated substrate do not provide the intended fault tolerance.

## Evaluation evidence with its boundaries

The OSDI evaluation mixes microbenchmarks, failure injection, TrueTime measurements, and F1 production observations. It is evidence for mechanisms, not an independent market comparison.

- The paper reports production epsilon as a sawtooth from about 1 to 7 ms over a 30-second poll interval, around 4 ms most of the time. The configured drift contribution was 200 microseconds/s, with roughly 1 ms attributed to master communication. Failures and congestion could create larger spikes.
- Figure 6 sampled several thousand spanservers across datacenters up to 2,200 km apart, plotting the 90th, 99th, and 99.9th percentiles immediately after time-master polls. A roughly one-hour increase followed maintenance that shut down two time masters in one datacenter. The graph demonstrates tails and operational sensitivity, not a universal 7 ms cap.
- In the zone-failure experiment (Figure 5), groups served an aggregate 50,000 reads/s before a kill. Gracefully handing leadership away before the leader zone stopped caused an ungraphed 3–4% throughput drop. Killing it without warning drove completions nearly to zero; groups recovered leaders over approximately the 10-second lease interval, then throughput returned.
- F1's 24-hour production measurements reported mean client-observed latencies of 8.7 ms over 21.5 billion reads, 72.3 ms over 31.2 million single-site commits, and 103.0 ms over 32.1 million multi-site commits. The read standard deviation was 376.4 ms and write tails included lock contention, emphasizing that means hide broad distributions.
- The F1 dataset was tens of terabytes uncompressed. The previous manual MySQL resharding had taken more than two years of work across dozens of teams. That is organizational evidence for automatic placement, not a direct throughput benchmark.

## Limits and assumptions

- Synchronous quorum and 2PC latency follow replica placement; geography cannot be abstracted away.
- Strict 2PL can block and deadlock; contention appears in tail latency.
- TrueTime requires specialized, continuously operated time infrastructure and enforced drift bounds.
- The 2012 system did not yet provide automatic secondary indexes; F1 built consistent indexes using transactions.
- Availability requires a quorum for every touched group and recoverable transaction coordinators.
- The original SQL/query surface was limited; the paper is primarily a storage and transaction design.
- Reported measurements come from Google-controlled hardware, workloads, and authors.

## Later SQL evolution

The 2017 *Spanner: Becoming a SQL System* paper documents the move from a storage API used through F1 toward a more complete SQL database: query distribution, schema changes, optimizer work, and the operational consequences of offering SQL directly. It retained the foundational combination of Paxos, MVCC, TrueTime, and externally consistent transactions while changing the interface and execution layer substantially.

Later Spanner product features should be evaluated from their own documentation and papers. They must not be back-projected into the 2012 protocol.

## Design review questions

1. Which transactions actually span ranges or regions, and what latency does their quorum/2PC path imply?
2. Is real-time-respecting serialization required, or would a weaker snapshot model suffice?
3. What explicit bound contains clock uncertainty, and what happens when it widens?
4. Can stale reads run at nearby replicas, and how is replica safe time proven?
5. Which data should share a placement directory, and can that choice create a hot range?
6. How are prepared transactions resolved after coordinator loss?
7. Do benchmarks report topology, replica count, contention, percentiles, and transaction locality?

## Primary sources

- [Corbett et al., *Spanner: Google's Globally-Distributed Database* (OSDI 2012), official USENIX PDF](https://www.usenix.org/system/files/conference/osdi12/osdi12-final-16.pdf)
- [Google Research publication record for the OSDI paper](https://research.google/pubs/spanner-googles-globally-distributed-database-2/)
- [Corbett et al., expanded *Spanner: Google's Globally Distributed Database* (ACM TOCS 2013)](https://storage.googleapis.com/gweb-research2023-media/pubtools/1974.pdf)
- [Bacon et al., *Spanner: Becoming a SQL System* (SIGMOD 2017)](https://research.google/pubs/spanner-becoming-a-sql-system/)
