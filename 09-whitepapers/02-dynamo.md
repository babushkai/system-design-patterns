# Dynamo (SOSP 2007): Evidence-First Paper Analysis

Dynamo is a study in making **write acceptance during failure** the primary invariant. It is not a generic claim that eventual consistency is faster, nor is it the architecture of the later Amazon DynamoDB service. The paper describes an internal Amazon key/value system whose applications were prepared to receive concurrent versions and reconcile them.

## Publication identity and reading boundary

- **Paper:** *Dynamo: Amazon's Highly Available Key-value Store*
- **Authors:** Giuseppe DeCandia, Deniz Hastorun, Madan Jampani, Gunavardhan Kakulapati, Avinash Lakshman, Alex Pilchin, Swaminathan Sivasubramanian, Peter Vosshall, and Werner Vogels
- **Venue and version:** 21st ACM Symposium on Operating Systems Principles (SOSP), 2007
- **System described:** multiple internal Dynamo instances operated by Amazon services, using the design and production experience available at publication

The 2007 system is **not Amazon DynamoDB**. The names share lineage, but the 2022 DynamoDB paper describes leader-based Multi-Paxos replication, a managed control plane, and no application-visible vector-clock reconciliation. See [Amazon DynamoDB (2022)](./14-dynamodb-2022.md) for that later design.

This chapter examines how the paper composes [Leaderless Replication](../02-distributed-databases/03-leaderless-replication.md), [Partitioning](../02-distributed-databases/05-partitioning-strategies.md), and [Conflict Resolution](../02-distributed-databases/04-conflict-resolution.md) for Dynamo's workload and evidence.

## The service problem Dynamo chose

Amazon's shopping and service-oriented workloads composed a page from calls to many downstream services. Section 2 observes that a typical page request contacted more than 150 services, so one slow or unavailable dependency could damage the whole customer request. Dynamo therefore optimized for a narrow contract:

- simple `get(key)` and `put(key, value, context)` operations;
- values generally smaller than 1 MB;
- no cross-key transaction or isolation guarantee;
- operations routed across commodity servers and multiple data centers;
- writes should remain acceptable during server and network failures;
- services should tune durability, consistency, latency, and availability through `N`, `R`, and `W`.

The paper explicitly targets applications willing to weaken consistency and resolve divergence. A bank transfer, globally unique username allocation, or invariant spanning multiple keys is outside that contract unless another coordination layer supplies the missing guarantee.

## State, versions, and invariants

For each key, Dynamo may retain several causally concurrent object versions. A successful `get` returns the object plus opaque context; the client must return that context on `put`. Internally, the context contains a vector clock.

A vector clock is a set of `(node, counter)` pairs. If every component of clock `A` is less than or equal to the corresponding component of `B`, then `A` is an ancestor of `B` and may be discarded. If neither dominates, the versions are concurrent siblings. Dynamo does not pretend it can infer business meaning from them: the application reconciles siblings and writes back a new version that descends from both.

This makes three boundaries explicit:

1. **Causal history is metadata, not wall-clock order.** Last-write-wins by timestamp would erase a concurrent update.
2. **Convergence is not immediate.** Read repair, hinted handoff, and anti-entropy progressively align replicas.
3. **Availability shifts complexity upward.** The storage layer can accept both writes, but only the application knows whether two shopping-cart edits should be unioned, rejected, or otherwise merged.

Vector clocks can grow as many nodes coordinate writes. Section 4.4 describes truncating old `(node, counter)` pairs using physical timestamps when a threshold is reached. The authors say they had not seen this cause problems in production at the time, but the truncation can lose causal ancestry. It is a pragmatic bounded-metadata compromise, not a proof-preserving optimization.

## Partition and replica protocol

Table 1 is the paper's compact design map: consistent hashing for partitioning, vector clocks for versioning, sloppy quorums and hinted handoff for temporary failure, Merkle trees for permanent divergence, and gossip for membership and failure detection.

### Consistent-hash ring and preference lists

Keys hash onto a ring. Each storage node owns multiple virtual nodes, so physical capacity and failure domains can be represented by multiple positions. A key's **preference list** walks clockwise to choose `N` distinct physical nodes. The distinct-node rule matters: replicas placed on several virtual positions of one machine would not survive that machine's loss.

The paper evaluates three partitioning strategies and reports that its third strategy (fixed, equal-sized partitions assigned to nodes) improved bootstrapping and archival because partition boundaries no longer changed with membership. This is a useful correction to the folklore that “virtual nodes” alone solve all operational movement problems.

### Coordinator and tunable quorum

A request can enter through a load balancer or a partition-aware client. One node coordinates the operation. It sends a write to the first `N` healthy nodes in the key's preference list and waits for `W` replies; a read contacts replicas and waits for `R` replies.

`R + W > N` creates overlap only under the assumptions of the replica set and version protocol. It does not make Dynamo linearizable: concurrent coordinators, sloppy substitutes, delayed replicas, and application reconciliation remain part of the model. The common production configuration reported in Section 6 was `(N,R,W) = (3,2,2)`, but that is evidence about several Amazon instances, not a universal setting.

### Sloppy quorum and hinted handoff

During failure, Dynamo walks beyond the normal `N` owners and stores a replica on another healthy node. The substitute records a **hint** naming the intended owner. When the owner recovers, the substitute transfers the object and can delete its temporary copy.

This is why the scheme is called sloppy: it preserves the requested count of successful storage operations among reachable nodes, not necessarily a quorum over the canonical replica set. It favors write availability, but expands the places a read or repair process may need to consider.

### Read repair and Merkle-tree anti-entropy

The coordinator compares returned versions during reads and updates lagging replicas, repairing hot data as a side effect of foreground traffic. For cold data, replicas exchange Merkle trees. Equal subtree hashes prove that the corresponding key ranges match; unequal branches are recursively narrowed, limiting transfer to divergent ranges.

Merkle trees are tied to ownership ranges. Membership changes can invalidate trees and trigger recomputation; this is one reason the paper's later fixed-partition strategy made operations simpler.

## Membership and failure handling

Failure detection is local: a node treats another as failed when it stops responding to messages, but other nodes may still reach it. This avoids a global agreement step on the write path. Permanent membership changes are explicit administrative actions recorded locally and spread through gossip, reducing the chance that transient reachability problems rewrite the ring.

The **seed** mechanism prevents logically separate rings from forming: selected nodes are discoverable through an external mechanism and present in every node's membership view. Gossip then spreads membership and token ownership.

Recovery is layered by duration:

- short outage: sloppy quorum plus hinted handoff;
- missed or lost handoff: read repair when the key is accessed;
- long-lived replica divergence: Merkle-tree anti-entropy;
- physical node addition/removal: ownership transfer according to the partition map.

The system cannot make progress for an operation if fewer than the configured response count are reachable. Nor does it protect correctness when application reconciliation is non-associative, loses information, or is not retried safely.

## Evidence from the production study

Section 6 states that measurements came from a live `(3,2,2)` deployment of a couple hundred homogeneous nodes spanning multiple data centers. Results should be read with that scope.

- Amazon services typically specified that 99.9% of reads and writes finish within 300 ms. Figure 4's 30-day trace put observed 99.9th-percentile latencies around 200 ms, roughly an order of magnitude above averages, with a diurnal pattern. The paper does not provide a transferable latency curve for arbitrary hardware or object sizes.
- A memory write buffer reduced peak-traffic 99.9th-percentile latency by a factor of five with a buffer of 1,000 objects (Figure 5). The paper is explicit that a server crash could lose buffered writes; one replica therefore performed a durable write. This experiment demonstrates a durability/latency trade, not a free cache win.
- Table 2 compares client-driven with server-driven coordination over 24 hours. Removing the load balancer and an extra hop improved 99.9th-percentile latency by at least 30 ms and average latency by 3–4 ms.
- Across two years of use by many internal services, applications received non-timeout responses for 99.9995% of requests, and the authors reported no data-loss event “to date.” This is an observational production statement, not a controlled proof of annual availability or durability.
- Section 6.6 says the full-membership model worked for a couple hundred nodes and would be challenging at tens of thousands. That is an acknowledged scalability boundary of gossiping the whole ring, not evidence that gossip itself is unscalable at every size.

The paper also reports an important operational failure mode: anti-entropy and handoff competed with foreground traffic. Dynamo introduced admission control that monitored foreground disk latency, lock contention, transaction timeouts, and queue waits, then limited background work. Repair capacity is part of serving capacity.

## Limits and hidden costs

- Only single-key atomicity is provided; no isolation spans keys.
- Application owners must design, test, and monitor conflict reconciliation.
- Hot keys cannot be divided by ring partitioning alone.
- Sloppy replicas and asynchronous repair complicate reasoning about where the newest version resides.
- Vector-clock metadata is bounded with a heuristic that can discard causal information.
- Full membership is held at every node; the paper does not establish behavior at enormous cluster sizes.
- Evaluation is author-operated and workload-specific. It does not compare Dynamo with a strong-consistency design under matched failure, cost, and durability targets.

These costs explain why Dynamo is a choice for a particular invariant hierarchy, not a default database architecture.

## Published design versus later evolution

The later DynamoDB service retained partitioned key/value access, explicit capacity management, and attention to high-percentile predictability. The 2022 paper says it changed the core replication model to Multi-Paxos leaders, offered strong reads, centralized much control-plane work, and removed application-visible sibling reconciliation. In other words, the product retained Dynamo's workload decomposition while rejecting several mechanisms that imposed operational or application complexity.

Other Dynamo-inspired systems retained different subsets: consistent hashing, tunable quorums, repair, or vector-like version metadata. “Dynamo-style” therefore does not identify one consistency model. A design review must name which replica set, conflict, read, repair, and membership rules are actually present.

## Decision framework

Choose a Dynamo-like design only after answering:

1. Which writes must remain accepted during a partition, and what conflicting outcomes are legal?
2. Can the merge function preserve all business invariants and converge regardless of delivery order?
3. What do `N`, `R`, and `W` mean when canonical owners are unavailable?
4. How are cold keys repaired if read repair never touches them?
5. What foreground latency budget is reserved while handoff and anti-entropy run?
6. Can one hot key exceed a node's capacity?
7. Is exposing reconciliation to every application worth the availability gained over a leader-based design?

## Primary sources

- [DeCandia et al., *Dynamo: Amazon's Highly Available Key-value Store* (SOSP 2007), official author-hosted PDF](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf)
- [Werner Vogels, official publication page and HTML paper](https://www.allthingsdistributed.com/2007/10/amazons_dynamo.html)
- [Elhemali et al., *Amazon DynamoDB: A Scalable, Predictably Performant, and Fully Managed NoSQL Database Service* (USENIX ATC 2022)](https://www.usenix.org/conference/atc22/presentation/elhemali)
