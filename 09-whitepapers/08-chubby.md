# Chubby (OSDI 2006): Evidence-First Paper Analysis

Chubby is a paper about packaging consensus into an operable coordination service. Google chose a file-and-lock interface, coarse-grained advisory locks, sessions, caching, and sequencers so application teams did not each need to implement Paxos. Its surprise was that reliable naming and configuration became more popular than locks.

## Publication identity and scope

- **Paper:** *The Chubby Lock Service for Loosely-Coupled Distributed Systems*
- **Author:** Mike Burrows, Google
- **Venue and version:** 7th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2006, pages 335–350
- **System described:** production Chubby cells used inside Google, including operating experience through the paper's publication

The paper does not fully publish Google's Paxos implementation, and Chubby is not a high-throughput database or a data-plane mutex around every request. It is a low-volume coordination substrate for leader election, small configuration records, naming, and coarse ownership.

See [Distributed Locks](../01-foundations/09-distributed-locks.md) and [Leader Election](../02-distributed-databases/09-leader-election.md) for general patterns. Here the concern is the paper's concrete lease, cache, and fencing behavior.

## Why a lock service instead of a consensus library

Google systems repeatedly needed to elect a primary, advertise its location, and store small authoritative metadata. A correct consensus library would still force every application to design replicated state, membership, recovery, and an API. Chubby centralizes that expertise.

The service chooses familiar file-system-like names because they are easy to inspect and integrate. Locks are **advisory**: Chubby does not intercept access to the protected resource. Every participant must honor the protocol, and a stale lock holder must be prevented from corrupting the real resource through fencing.

The intended lock duration is minutes or hours, not milliseconds. Clients cache file data and handles; high-frequency data operations stay in the application system. This workload boundary is what lets one master serve many thousands of processes.

## Cell architecture and replicated state

A typical Chubby **cell** contains five replicas. They use Paxos to elect one master and replicate a database log. The master serves reads and writes; other replicas can become master after failure. A majority must remain available for the cell to process operations safely.

The namespace resembles directories and files. Nodes may be permanent or ephemeral, hold small byte strings and metadata, and be opened through handles. Clients can acquire shared or exclusive locks on handles and subscribe to events such as content changes, child changes, or master failover.

Key replicated state includes:

- namespace nodes and contents;
- access-control lists;
- open-handle and lock ownership state tied to sessions;
- monotonically increasing lock acquisition numbers;
- the database log and snapshots needed to reconstruct it.

The paper's interface deliberately omits many file-system semantics. There are no hard links, and operations are shaped around coordination rather than bulk file manipulation.

## Sessions, leases, and cache coherence

A client establishes a session with the master and maintains it using KeepAlive RPCs. The master promises the session remains valid for a lease interval; the client extends its local view using replies that arrive before its own conservative deadline. If communication is lost long enough, both sides eventually treat the session as expired, releasing ephemeral nodes and locks.

This lease is not proof that the process holding a lock is still capable of safely touching an external resource. Pauses, partitions, and delayed messages can make the former owner continue running after the service reassigns the lock.

Chubby therefore exposes a **sequencer** containing the lock name, acquisition mode, and monotonically increasing generation. A protected server can reject operations carrying an older generation. Where the resource cannot validate sequencers, Chubby supports a lock-delay period after session loss before another client may acquire the lock. Sequencers are stronger because they fence the old owner at the resource; delay only reduces risk under a timing assumption.

Clients cache file contents, metadata, open handles, and negative lookups. Cache validity is tied to the session. Before modifying cached data, the master sends invalidations and waits for acknowledgements or for the relevant cache leases to expire. Only then does it perform the update. This keeps reads cheap without allowing a disconnected client to treat stale cached content as indefinitely current.

Caching explains both scale and semantics: a stable configuration file can be read locally thousands of times while one KeepAlive stream maintains validity. The master load depends more on sessions and changes than on application read frequency.

## Lock and naming workflows

A primary-election pattern uses one well-known Chubby name:

1. contenders open the node and attempt its exclusive lock;
2. the winner publishes its address in the file;
3. clients read and cache the address;
4. on session loss, Chubby releases ownership and invalidates watchers;
5. the next winner receives a higher sequencer generation.

The lock serializes primary identity; it does not replicate the application's data. The primary still needs a recovery protocol for its own state.

The paper found naming to be Chubby's most popular use. DNS time-to-live creates a tension between quick failover and query volume. Chubby's explicit invalidations let unchanged names remain cached without polling each name. A protocol-conversion service could expose Chubby-backed names through simpler interfaces.

## Failure and recovery reasoning

- **Master failure:** remaining replicas use Paxos to elect a new master. Clients discover the change during KeepAlive and rebuild master-specific session state.
- **Minority replica failure:** a quorum continues; failed replicas are repaired or replaced.
- **Loss of quorum:** the cell cannot safely serve updates or renew authoritative coordination state. Client sessions eventually expire.
- **Client or network partition:** the master stops renewing the session, releases locks after expiry, and invalidates cached state. A former holder may still execute, which is why external fencing matters.
- **Overloaded master:** delayed KeepAlives can drop many sessions, converting a performance incident into ownership churn. The master lengthens lease intervals under load.
- **Corrupt or buggy replica:** Paxos crash-fault replication alone does not mask arbitrary corruption. The paper reports real software and operator-caused data-loss events.

Replica replacement is deliberately conservative. Chubby's configuration changes through a special mechanism because replacing several replicas too quickly can destroy the majority intersection on which safety depends.

## Production evidence with units and caveats

Section 4 mixes a typical-cell snapshot with outage history. It is unusually useful because it reports uncomfortable results as well as scale.

### Typical cell snapshot

One representative cell had run 18 days since failover; its previous failover took 14 seconds. It served 22,000 direct and 32,000 proxied clients, 12,000 open files, 1,000 exclusive locks, and no shared locks in the snapshot. RPC volume was 1,000–2,000/s, of which 93% was KeepAlive. `SetContents` occurred at 680 parts per million of RPCs and lock acquisition at 31 ppm. These ratios validate the low-write, coarse-lock workload.

The default session lease was 12 seconds and could rise to about 60 seconds under load. Google had observed 90,000 clients talking directly to one master. That is an observed scale point, not a supported-capacity formula.

### Outages and durability

Across a sample totaling 700 cell-days, the authors recorded 61 non-datacenter-shutdown outages. Most lasted 15 seconds or less; 52 were under 30 seconds. The remaining nine were attributed to network maintenance (four), suspected connectivity problems (two), software errors (two), and overload (one).

Over a few dozen cell-years, Chubby lost data six times: four database-software errors and two operator errors; none was attributed to hardware. The paper also reports twice correcting corruption introduced by software on non-master replicas. Consensus reduced a class of machine failures but did not eliminate software or operational risk.

### Latency

Server mean request latency stayed well below 1 ms until overload. Client-observed local-cell reads were under 1 ms, while reads between antipodes reached about 250 ms. Writes, including lock operations, added roughly 5–10 ms for the database log update and could wait tens of seconds when invalidating data cached by a failed client. These figures belong to Google's 2006 network and workload; the causal lesson is that cache-lease invalidation can dominate write tails.

### Scaling lesson

The authors conclude that reducing master communication mattered more than micro-optimizing request handling. Proxies can aggregate KeepAlives by the number of clients behind them; cache sharing reduces reads less dramatically. Partitioning the namespace was designed but not then used in production. Planned mechanisms are not evaluation evidence.

## Design errors and operational lessons

The paper calls out several surprises:

- Chubby became a name service, making read caching and invalidation more central than expected.
- A complex 7,000-line C++ client library was difficult to reproduce safely in Java; protocol-conversion servers avoided divergent implementations.
- Clients sometimes retried or polled abusively, so the service needed quotas, backoff, and detection rather than trusting perfect library use.
- Developers wanted notifications, but event delivery and session transitions create subtle duplicate/missed-event handling requirements.
- Many outages came from maintenance, networking, overload, and software rather than simultaneous disk crashes.

This is the paper's deeper lesson: a coordination protocol is only as reliable as its client library, overload behavior, operations, and fencing integration.

## Assumptions and limitations

- A majority of replicas is required; Chubby favors consistency over partition-side availability.
- One master per cell bounds write and session-processing capacity.
- Locks are advisory and do not automatically fence an old owner.
- The namespace stores small coordination data, not bulk application state.
- Session expiry trades failover speed against false expiration during pauses and network delay.
- Paxos internals and complete proofs are not published in this paper.
- Crash-fault replication does not protect against all software, operator, or correlated errors.

## Later systems retained and changed

ZooKeeper later retained a small hierarchical namespace, sessions, ephemeral nodes, and notifications while exposing a different ordering and API model through the Zab protocol. Other coordination services use key/value revisions, watch streams, and lease-attached keys rather than Chubby's file handles and locks. The family resemblance is useful, but their fencing, read consistency, membership, and watch guarantees differ.

The part worth carrying forward is not the pathname syntax. It is the separation between a small, strongly coordinated control plane and a large application data plane, plus explicit session and fencing semantics.

## Design review questions

1. Is the protected operation coarse enough that a coordination-service round trip is appropriate?
2. What resource validates a monotonically increasing fencing token?
3. When a client pauses beyond its lease, can it still corrupt external state?
4. Which reads may be cached, and what blocks a write until invalidation is safe?
5. Can KeepAlive overload cause correlated session loss?
6. How are membership changes sequenced without losing quorum intersection?
7. Are restore, corruption detection, client-library behavior, and operator error tested—not only replica crashes?

## Primary sources

- [Burrows, *The Chubby Lock Service for Loosely-Coupled Distributed Systems* (OSDI 2006), official Google PDF](https://research.google.com/archive/chubby-osdi06.pdf)
- [Google Research publication record and HTML paper](https://research.google/pubs/the-chubby-lock-service-for-loosely-coupled-distributed-systems/)
- [Hunt et al., *ZooKeeper: Wait-free coordination for Internet-scale systems* (USENIX ATC 2010)](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Hunt.pdf)
