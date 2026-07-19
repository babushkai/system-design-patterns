# Kafka (NetDB 2011): Evidence-First Paper Analysis

The original Kafka paper is valuable precisely because it describes a much smaller system than “Kafka” means today. Its thesis is that log aggregation becomes cheap when the broker stores **partitioned append-only files**, consumers pull by byte offset, and retention is independent of acknowledgements. Replication, durable producer acknowledgements, idempotent production, transactions, and KRaft were not in the published design.

## Publication identity and historical boundary

- **Paper:** *Kafka: a Distributed Messaging System for Log Processing*
- **Authors:** Jay Kreps, Neha Narkhede, and Jun Rao, LinkedIn
- **Venue and version:** NetDB workshop, 2011
- **System described:** LinkedIn's first production Kafka architecture, benchmarked against ActiveMQ 5.4 and RabbitMQ 2.4

This chapter never uses a modern Kafka guarantee as evidence for the 2011 paper. Apache's later replication design is discussed separately below.

For present-day pattern language, see [Message Queues](../05-messaging/01-message-queues.md), [Message Ordering](../05-messaging/03-message-ordering.md), and [Delivery Guarantees](../05-messaging/04-delivery-guarantees.md).

## Problem and workload

LinkedIn needed to collect large volumes of operational and activity events from online services, feed both near-real-time consumers and Hadoop/data-warehouse loading, retain data for replay, and add consumers without multiplying producer integrations.

Traditional enterprise messaging systems carried features and state that this workload did not always need: per-message delivery tracking, rich headers, broker-managed subscriber cursors, and deletion tied to consumption. Kafka narrowed the problem:

- messages belong to topics;
- a topic has independent ordered partitions;
- producers append batches;
- consumers pull sequential ranges using offsets;
- a consumer group divides partitions among its members;
- time/size retention, not successful delivery, controls deletion.

That is closer to a distributed commit log than a work queue. It enables replay and multiple independent consumers, but ordering stops at a partition and consumers own progress semantics.

## Durable state and core invariants

In the paper, each topic partition is an ever-growing logical log represented by segment files. A message ID is its **byte offset** in the partition, not a broker-assigned opaque UUID. The offset lets a consumer request “messages beginning here” without the broker maintaining per-consumer delivery records.

Three invariants follow:

1. A broker appends to the end of a partition and never reorders existing bytes.
2. Within one consumer group, one consumer owns a partition at a time, so that partition's records are delivered in log order to one processing stream.
3. Consumer progress is separate from log retention. Rewinding the offset replays data; a slow consumer does not force immediate retention of every message unless configured retention still covers it.

There is no total order across partitions. If business events require a single order, their partitioning key must route them to one partition, concentrating that ordered stream's throughput and availability on one broker in the original design.

## Storage and transfer path

The broker writes messages to ordinary append-only files and relies on the operating system page cache rather than maintaining a second application cache. Old segments can be removed as coarse files according to retention, avoiding per-message deletion and free-list management.

The paper emphasizes end-to-end batching:

- producers send a message set in one request;
- brokers append that set contiguously;
- consumers fetch a large byte range;
- Linux `sendfile` transfers page-cache data to the socket without copying it through a user-space buffer.

Sequential I/O, page-cache reuse, fewer system calls, and a compact record format reinforce one another. “Disk-backed” therefore does not mean every consumer read waits for a physical disk: recent log tails are often already in the page cache.

Producers choose a partition, either randomly for balance or by a semantic key so related records share order. The 2011 broker does not transparently split one partition; the chosen partition count is both the parallelism ceiling for a consumer group and an operational placement decision.

## Consumer coordination and delivery semantics

Brokers and consumers register in ZooKeeper. A consumer-group member observes membership or broker changes, sorts consumers and available partitions, and deterministically claims its share. Algorithm 1 writes ownership and reads each partition's stored offset from ZooKeeper.

The decentralized rebalance can race: two consumers may attempt ownership concurrently and retry after conflict. It also pauses or moves partitions when group membership changes. The paper favors this client-side coordination to keep brokers stateless with respect to consumers.

Delivery semantics depend on when the consumer stores its offset relative to processing:

- save before processing: a crash can skip work;
- process before saving: a crash can replay records, producing duplicates;
- atomically store output and source offset in the same destination transaction: the destination can make replay exactly-once relative to itself.

The LinkedIn Hadoop loader used the third pattern by committing data and offsets together in HDFS only after a successful MapReduce task. The broker itself did not provide transactions. The paper explicitly says applications that care about duplicates need offset-based or business-key deduplication.

## Failure behavior in the published system

Each record carries a CRC. On broker recovery, Kafka scans and removes records with inconsistent CRCs, protecting against partial writes and detected corruption.

The larger limitation is stark: **the 2011 implementation has no built-in replication**. If a broker fails, its unconsumed partitions are unavailable. If its storage is permanently damaged, those messages are lost. Section 6 lists replication as future work.

Producer durability is also weak in the benchmarked path. The producer did not wait for a broker acknowledgement; the paper says this raises throughput but provides no guarantee that every published message reached the broker. Asynchronous broker flush means even receipt and process failure do not necessarily imply durable media persistence.

ZooKeeper preserves group metadata, but it does not replicate log bytes. A healthy coordination service cannot recover a destroyed partition.

These facts are why it is incorrect to attach modern `acks=all`, ISR, idempotent-producer, or transaction guarantees to the NetDB results.

## Experimental evidence and fair interpretation

Section 5 used two Linux machines, each with eight 2 GHz cores, 16 GB RAM, six disks in RAID 10, and a 1 Gbit link. One machine ran the broker; the other ran a single producer or consumer. Brokers asynchronously flushed their persistence stores. This is a single-broker throughput experiment, not a distributed-failure benchmark.

### Producer experiment

Each system received 10 million messages of 200 bytes. Kafka used batch sizes 1 and 50; the authors could not configure comparable producer batching in ActiveMQ or RabbitMQ and assumed batch size 1 for them.

- Kafka averaged 50,000 messages/s at batch 1.
- Kafka averaged 400,000 messages/s at batch 50, nearly saturating the 1 Gbit link.
- The paper reports at least twice RabbitMQ's throughput and orders of magnitude above ActiveMQ in that setup.

The comparison is not feature-equivalent. Kafka's producer did not wait for acknowledgements, while the competing products offered more messaging features. The paper itself says the purpose is to demonstrate the potential of specialization, not prove the other systems inferior.

The record-format result is similarly bounded: the measured overhead was 9 bytes per Kafka message versus 144 bytes for ActiveMQ, making ActiveMQ use 70% more space for this 200-byte, 10-million-message dataset. Header requirements and indexing explain much of the difference.

### Consumer experiment

A single consumer fetched 10 million messages. Fetches requested up to 1,000 messages or about 200 KB; all records fit in memory/page cache. Kafka averaged 22,000 messages/s, more than four times ActiveMQ and RabbitMQ in the reported run. The authors attribute the result to compact records, no broker-side per-message delivery updates, and `sendfile`.

Because everything was cache-resident, the experiment does not establish cold-storage throughput, multi-consumer scaling, replication cost, tail latency, or recovery time.

### Production snapshot

At publication, LinkedIn reported hundreds of gigabytes and close to one billion messages/day, with roughly 10-second average end-to-end latency into the offline-analysis pipeline without much tuning. This describes that deployment and pipeline, not a Kafka latency guarantee.

## Limits and assumptions

- One partition is the ordering and consumer-parallelism unit; skewed partition keys create hot brokers.
- The paper's producer path may lose messages before broker receipt or durable flush.
- No broker replication means machine loss can become data loss.
- Consumer-side offsets make replay powerful but move duplicate/skip correctness to the application.
- ZooKeeper-based group coordination can rebalance and temporarily stop partition processing.
- Time-based retention assumes consumers recover before required data expires.
- The benchmark favors Kafka's narrow feature set and does not match durability semantics.

## What later Kafka changed

Apache Kafka's later replication design added a leader and replicas for each partition, an in-sync replica set (ISR), a high watermark for committed records, and producer acknowledgement choices. That design changes both failure recovery and the meaning of “published.” Subsequent releases added broker-managed consumer offsets, idempotent producers, transactions, stream processing, and eventually a Raft-based metadata quorum. These are significant new protocols, not implementation details hidden in the 2011 paper.

What survived is the architectural core: partitioned append logs, immutable offsets, batched sequential I/O, pull-based consumers, replay, and retention decoupled from individual subscriptions.

## Design review questions

1. What entity requires ordering, and can all its events safely share one partition?
2. When may a producer acknowledge: process receipt, page cache, local fsync, or replicated commit?
3. Where is consumer progress stored relative to the side effect it represents?
4. How long can a consumer be down before retention destroys required replay data?
5. What happens to availability and durability when a partition leader or disk fails?
6. Are benchmark comparisons matched for acknowledgements, flush, replication, batch size, and cache state?
7. Is partition count sufficient for growth without making coordination and recovery excessive?

## Primary sources

- [Kreps, Narkhede, and Rao, *Kafka: a Distributed Messaging System for Log Processing* (NetDB 2011), archived paper PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/09/Kafka.pdf)
- [Apache Kafka project archive of the 2011 NetDB material](https://cwiki.apache.org/confluence/download/attachments/27822226/Kafka-netdb-06-2011.pdf)
- [Apache Kafka, official high-level replication design](https://cwiki.apache.org/confluence/spaces/KAFKA/pages/27840158/Kafka%2BReplication)
- [Apache Kafka, official detailed replication design](https://cwiki.apache.org/confluence/spaces/KAFKA/pages/27844516/kafka%2BDetailed%2BReplication%2BDesign%2BV3)
