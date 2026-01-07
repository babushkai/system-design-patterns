# Kafka: A Distributed Messaging System for Log Processing

## Paper Overview

- **Title**: Kafka: a Distributed Messaging System for Log Processing
- **Authors**: Jay Kreps, Neha Narkhede, Jun Rao (LinkedIn)
- **Published**: NetDB Workshop 2011
- **Context**: LinkedIn needed high-throughput, low-latency log processing

## TL;DR

Kafka is a distributed commit log that provides:
- **High throughput** through sequential disk I/O and batching
- **Scalability** via partitioned topics
- **Durability** through replication
- **Simple consumer model** with offset-based tracking

## Problem Statement

### Log Processing Challenges

```
┌─────────────────────────────────────────────────────────────────┐
│                   LinkedIn's Requirements                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Activity Data:                                                 │
│  ┌─────────────────────────────────────────────┐                │
│  │  - Page views: billions per day             │                │
│  │  - User actions: clicks, searches, etc.     │                │
│  │  - System metrics: CPU, memory, latency     │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  Use Cases:                                                     │
│  ┌─────────────────────────────────────────────┐                │
│  │  - Real-time analytics dashboards           │                │
│  │  - Offline batch processing (Hadoop)        │                │
│  │  - Search indexing                          │                │
│  │  - Recommendation systems                   │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  Existing Solutions Fall Short:                                 │
│  ┌─────────────────────────────────────────────┐                │
│  │  - Traditional MQ: Too slow, not scalable   │                │
│  │  - Log files: No real-time, hard to manage  │                │
│  │  - Custom solutions: Complex, fragile       │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Kafka Architecture

### Core Concepts

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kafka Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TOPIC: Named feed of messages                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                        Topic "clicks"                     │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │ Partition 0:  [M0][M1][M2][M3][M4][M5]...          │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │ Partition 1:  [M0][M1][M2][M3][M4]...              │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │ Partition 2:  [M0][M1][M2][M3][M4][M5][M6]...      │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  PARTITION: Ordered, immutable sequence of messages             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                           │   │
│  │   Offset:  0    1    2    3    4    5    6    7          │   │
│  │          ┌────┬────┬────┬────┬────┬────┬────┬────┐       │   │
│  │          │ M0 │ M1 │ M2 │ M3 │ M4 │ M5 │ M6 │ M7 │       │   │
│  │          └────┴────┴────┴────┴────┴────┴────┴────┘       │   │
│  │                                              ▲            │   │
│  │                                          append-only      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  BROKER: Server that stores partitions                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Broker 1        Broker 2        Broker 3                │   │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐              │   │
│  │  │ P0, P3   │   │ P1, P4   │   │ P2, P5   │              │   │
│  │  └──────────┘   └──────────┘   └──────────┘              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Message Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Message Flow                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Producers                  Kafka Cluster               Consumers│
│  ┌───────┐                  ┌─────────────┐            ┌───────┐│
│  │ App 1 │──────────────────│             │────────────│ App A ││
│  └───────┘                  │             │            └───────┘│
│  ┌───────┐     publish      │   Broker    │   consume  ┌───────┐│
│  │ App 2 │──────────────────│   Cluster   │────────────│ App B ││
│  └───────┘                  │             │            └───────┘│
│  ┌───────┐                  │             │            ┌───────┐│
│  │ App 3 │──────────────────│             │────────────│ App C ││
│  └───────┘                  └─────────────┘            └───────┘│
│                                    │                            │
│                                    │                            │
│                              ┌─────┴─────┐                      │
│                              │ ZooKeeper │                      │
│                              │ (metadata)│                      │
│                              └───────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Log-Based Storage

### Append-Only Log

```python
class KafkaLog:
    """Kafka's log-based storage implementation."""
    
    def __init__(self, log_dir: str, segment_size: int = 1024 * 1024 * 1024):
        self.log_dir = log_dir
        self.segment_size = segment_size  # 1GB default
        self.segments = []  # List of log segments
        self.active_segment = None
        self.next_offset = 0
    
    def append(self, messages: list) -> list:
        """
        Append messages to log.
        
        Returns list of offsets for appended messages.
        Key insight: Sequential writes are FAST on disk.
        """
        offsets = []
        
        for message in messages:
            # Check if we need a new segment
            if self._need_new_segment():
                self._roll_segment()
            
            # Write message to active segment
            offset = self.next_offset
            self.active_segment.append(offset, message)
            offsets.append(offset)
            self.next_offset += 1
        
        return offsets
    
    def read(self, start_offset: int, max_bytes: int) -> list:
        """
        Read messages starting from offset.
        
        Returns messages up to max_bytes.
        """
        messages = []
        bytes_read = 0
        
        # Find segment containing start_offset
        segment = self._find_segment(start_offset)
        
        while segment and bytes_read < max_bytes:
            # Read from segment
            segment_messages = segment.read(
                start_offset, 
                max_bytes - bytes_read
            )
            
            for msg in segment_messages:
                messages.append(msg)
                bytes_read += len(msg.value)
                start_offset = msg.offset + 1
            
            # Move to next segment
            segment = self._next_segment(segment)
        
        return messages
    
    def _need_new_segment(self) -> bool:
        """Check if active segment is full."""
        if not self.active_segment:
            return True
        return self.active_segment.size >= self.segment_size
    
    def _roll_segment(self):
        """Create new segment, close old one."""
        if self.active_segment:
            self.active_segment.close()
            self.segments.append(self.active_segment)
        
        self.active_segment = LogSegment(
            base_offset=self.next_offset,
            path=f"{self.log_dir}/{self.next_offset}.log"
        )


class LogSegment:
    """Individual log segment file."""
    
    def __init__(self, base_offset: int, path: str):
        self.base_offset = base_offset
        self.path = path
        self.file = open(path, 'ab+')
        self.index = SparseIndex()  # Offset -> file position
        self.size = 0
    
    def append(self, offset: int, message: bytes):
        """Append message to segment file."""
        # Write message with header
        position = self.file.tell()
        record = self._encode_record(offset, message)
        self.file.write(record)
        self.size += len(record)
        
        # Update sparse index (every N messages)
        if offset % 4096 == 0:
            self.index.add(offset, position)
    
    def read(self, offset: int, max_bytes: int) -> list:
        """Read messages from offset."""
        # Use index to find approximate position
        position = self.index.lookup(offset)
        self.file.seek(position)
        
        messages = []
        bytes_read = 0
        
        while bytes_read < max_bytes:
            record = self._read_record()
            if record is None:
                break
            
            if record.offset >= offset:
                messages.append(record)
                bytes_read += len(record.value)
        
        return messages
```

### Efficient I/O

```python
class EfficientIO:
    """Kafka's I/O optimizations."""
    
    def __init__(self):
        self.page_cache = {}  # OS page cache simulation
    
    def zero_copy_send(self, socket, file_path: str, 
                       offset: int, length: int):
        """
        Zero-copy transfer from file to socket.
        
        Traditional:
        1. Read file -> kernel buffer
        2. Copy kernel buffer -> user buffer
        3. Copy user buffer -> socket buffer
        4. Send socket buffer -> NIC
        
        Zero-copy (sendfile):
        1. Read file -> kernel buffer
        2. Send kernel buffer -> NIC
        
        Eliminates 2 copies and context switches!
        """
        import os
        # os.sendfile(socket.fileno(), file.fileno(), offset, length)
        # This is a system call that does zero-copy
        pass
    
    def batched_compression(self, messages: list) -> bytes:
        """
        Compress multiple messages together.
        
        Better compression ratio than individual messages.
        """
        import gzip
        
        # Batch messages into single payload
        batch = b''.join(
            self._encode_message(m) for m in messages
        )
        
        # Compress entire batch
        compressed = gzip.compress(batch)
        
        return compressed
    
    def page_cache_friendly_writes(self):
        """
        Kafka leverages OS page cache.
        
        1. Write to memory-mapped file
        2. OS flushes to disk asynchronously
        3. Reads served from page cache
        
        Result: Near-memory speed for recent data
        """
        pass
```

## Producer

### Publishing Messages

```python
class KafkaProducer:
    """Kafka producer implementation."""
    
    def __init__(self, bootstrap_servers: list):
        self.brokers = bootstrap_servers
        self.metadata = self._fetch_metadata()
        self.batch_size = 16384  # 16KB
        self.linger_ms = 5  # Wait up to 5ms for more messages
        self.batches = defaultdict(list)  # Topic-partition -> messages
    
    def send(self, topic: str, key: bytes, value: bytes, 
             partition: int = None) -> Future:
        """
        Send message to topic.
        
        Returns Future that resolves when message is acknowledged.
        """
        # Determine partition
        if partition is None:
            partition = self._partition(topic, key)
        
        # Create produce record
        record = ProducerRecord(
            topic=topic,
            partition=partition,
            key=key,
            value=value,
            timestamp=time.time_ns()
        )
        
        # Add to batch
        tp = TopicPartition(topic, partition)
        future = Future()
        self.batches[tp].append((record, future))
        
        # Check if batch is full
        if self._batch_full(tp):
            self._send_batch(tp)
        
        return future
    
    def _partition(self, topic: str, key: bytes) -> int:
        """
        Determine partition for message.
        
        If key is provided: hash(key) % num_partitions
        If no key: round-robin
        """
        num_partitions = self.metadata.partitions_for_topic(topic)
        
        if key:
            # Murmur2 hash for consistent partitioning
            hash_value = murmur2(key)
            return hash_value % num_partitions
        else:
            # Round-robin for key-less messages
            return self._next_partition(topic)
    
    def _send_batch(self, tp: TopicPartition):
        """Send accumulated batch to broker."""
        batch = self.batches[tp]
        if not batch:
            return
        
        # Find leader for partition
        leader = self.metadata.leader_for(tp)
        
        # Create produce request
        records = [record for record, _ in batch]
        request = ProduceRequest(
            topic=tp.topic,
            partition=tp.partition,
            records=records,
            acks='all',  # Wait for all replicas
            timeout_ms=30000
        )
        
        try:
            # Send to leader
            response = self._send_request(leader, request)
            
            # Complete futures
            for i, (_, future) in enumerate(batch):
                offset = response.base_offset + i
                future.set_result(RecordMetadata(
                    topic=tp.topic,
                    partition=tp.partition,
                    offset=offset
                ))
        except Exception as e:
            for _, future in batch:
                future.set_exception(e)
        finally:
            self.batches[tp] = []


class Partitioner:
    """Custom partitioning strategies."""
    
    def partition_by_key(self, key: bytes, num_partitions: int) -> int:
        """Hash-based partitioning for ordering by key."""
        return murmur2(key) % num_partitions
    
    def partition_round_robin(self, num_partitions: int) -> int:
        """Round-robin for load balancing."""
        self.counter = getattr(self, 'counter', 0)
        partition = self.counter % num_partitions
        self.counter += 1
        return partition
    
    def partition_by_custom_logic(self, record, num_partitions: int) -> int:
        """Custom logic based on message content."""
        # Example: partition by user region
        region = record.headers.get('region')
        region_to_partition = {'us': 0, 'eu': 1, 'asia': 2}
        return region_to_partition.get(region, 0)
```

## Consumer

### Consumer Groups

```
┌─────────────────────────────────────────────────────────────────┐
│                    Consumer Groups                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Topic "orders" with 4 partitions                               │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  P0          P1          P2          P3                │     │
│  └──┬───────────┬───────────┬───────────┬─────────────────┘     │
│     │           │           │           │                       │
│     │           │           │           │                       │
│  Consumer Group A                                               │
│  (3 consumers)                                                  │
│  ┌─────────────────────────────────────────────┐                │
│  │  Consumer A1    Consumer A2    Consumer A3  │                │
│  │    (P0, P1)       (P2)          (P3)        │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  Consumer Group B                                               │
│  (2 consumers)                                                  │
│  ┌────────────────────────────────────┐                         │
│  │  Consumer B1        Consumer B2    │                         │
│  │    (P0, P1)          (P2, P3)      │                         │
│  └────────────────────────────────────┘                         │
│                                                                  │
│  Key Points:                                                    │
│  - Each partition assigned to exactly one consumer in group     │
│  - Consumer can handle multiple partitions                      │
│  - Different groups receive all messages independently          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Consumer Implementation

```python
class KafkaConsumer:
    """Kafka consumer implementation."""
    
    def __init__(self, group_id: str, bootstrap_servers: list):
        self.group_id = group_id
        self.brokers = bootstrap_servers
        self.subscriptions = set()
        self.assignments = {}  # TopicPartition -> offset
        self.coordinator = None
    
    def subscribe(self, topics: list):
        """Subscribe to topics."""
        self.subscriptions.update(topics)
        self._join_group()
    
    def poll(self, timeout_ms: int = 1000) -> list:
        """
        Poll for new messages.
        
        Returns list of ConsumerRecords.
        """
        records = []
        
        for tp, offset in self.assignments.items():
            # Fetch from broker
            leader = self._leader_for(tp)
            fetch_response = self._fetch(
                leader, tp, offset, max_bytes=1048576
            )
            
            for record in fetch_response.records:
                records.append(ConsumerRecord(
                    topic=tp.topic,
                    partition=tp.partition,
                    offset=record.offset,
                    key=record.key,
                    value=record.value,
                    timestamp=record.timestamp
                ))
                # Update local offset
                self.assignments[tp] = record.offset + 1
        
        return records
    
    def commit(self, offsets: dict = None):
        """
        Commit offsets to Kafka.
        
        Offsets are stored in internal __consumer_offsets topic.
        """
        if offsets is None:
            offsets = self.assignments.copy()
        
        # Send commit request to coordinator
        request = OffsetCommitRequest(
            group_id=self.group_id,
            offsets=offsets
        )
        self.coordinator.commit_offsets(request)
    
    def _join_group(self):
        """Join consumer group and get partition assignments."""
        # Find group coordinator
        self.coordinator = self._find_coordinator()
        
        # Join group
        join_response = self.coordinator.join_group(
            group_id=self.group_id,
            member_id='',
            protocol_type='consumer',
            protocols=[('range', self.subscriptions)]
        )
        
        if join_response.is_leader:
            # Leader assigns partitions
            assignments = self._assign_partitions(
                join_response.members,
                self.subscriptions
            )
            self.coordinator.sync_group(assignments)
        else:
            # Follower receives assignments
            sync_response = self.coordinator.sync_group({})
            self.assignments = sync_response.assignments


class PartitionAssignor:
    """Partition assignment strategies."""
    
    def range_assign(self, consumers: list, 
                     partitions: list) -> dict:
        """
        Range assignment: consecutive partitions to each consumer.
        
        Good for: Joining data across topics with same partitioning
        """
        assignments = defaultdict(list)
        partitions = sorted(partitions)
        
        num_consumers = len(consumers)
        partitions_per_consumer = len(partitions) // num_consumers
        extra = len(partitions) % num_consumers
        
        partition_idx = 0
        for i, consumer in enumerate(sorted(consumers)):
            count = partitions_per_consumer + (1 if i < extra else 0)
            for _ in range(count):
                assignments[consumer].append(partitions[partition_idx])
                partition_idx += 1
        
        return assignments
    
    def round_robin_assign(self, consumers: list, 
                           partitions: list) -> dict:
        """
        Round-robin: distribute partitions evenly.
        
        Good for: Load balancing when topics have different sizes
        """
        assignments = defaultdict(list)
        consumers = sorted(consumers)
        
        for i, partition in enumerate(sorted(partitions)):
            consumer = consumers[i % len(consumers)]
            assignments[consumer].append(partition)
        
        return assignments
```

### Offset Management

```python
class OffsetManager:
    """Manage consumer offsets."""
    
    def __init__(self):
        self.committed_offsets = {}  # Stored in __consumer_offsets
        self.pending_offsets = {}
    
    def auto_commit(self, assignments: dict, interval_ms: int = 5000):
        """
        Automatic offset commit.
        
        Risk: May lose messages on crash between process and commit
        """
        while True:
            time.sleep(interval_ms / 1000)
            self.commit(assignments)
    
    def manual_commit_sync(self, offsets: dict):
        """
        Synchronous manual commit.
        
        Use case: At-least-once processing
        1. Poll messages
        2. Process messages
        3. Commit offsets
        
        If crash between 2-3, messages reprocessed (at-least-once)
        """
        self._commit_to_kafka(offsets)
    
    def manual_commit_async(self, offsets: dict, callback):
        """
        Asynchronous commit with callback.
        
        Higher throughput but harder error handling.
        """
        future = self._commit_to_kafka_async(offsets)
        future.add_callback(callback)
    
    def reset_offset(self, tp: TopicPartition, strategy: str):
        """
        Reset offset when no committed offset exists.
        
        Strategies:
        - 'earliest': Start from beginning
        - 'latest': Start from end (new messages only)
        """
        if strategy == 'earliest':
            return self._get_earliest_offset(tp)
        elif strategy == 'latest':
            return self._get_latest_offset(tp)
```

## Replication

### Leader-Follower Replication

```
┌─────────────────────────────────────────────────────────────────┐
│                   Partition Replication                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Partition 0 (Replication Factor = 3)                           │
│                                                                  │
│  Broker 1                Broker 2                Broker 3       │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐ │
│  │   LEADER    │        │  FOLLOWER   │        │  FOLLOWER   │ │
│  │             │        │             │        │             │ │
│  │  [0][1][2]  │───────>│  [0][1][2]  │        │  [0][1][2]  │ │
│  │  [3][4][5]  │        │  [3][4][5]  │<───────│  [3][4]     │ │
│  │  [6][7]     │        │  [6]        │        │             │ │
│  │      ▲      │        │             │        │             │ │
│  └──────┼──────┘        └─────────────┘        └─────────────┘ │
│         │                                                       │
│    Producers write                                              │
│    to leader only                                               │
│                                                                  │
│  ISR (In-Sync Replicas): {Broker 1, Broker 2}                  │
│  - Broker 3 is behind, not in ISR                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Replication Protocol

```python
class ReplicationManager:
    """Kafka replication management."""
    
    def __init__(self, broker_id: int):
        self.broker_id = broker_id
        self.replicas = {}  # partition -> ReplicaState
        self.isr = {}  # partition -> set of in-sync replicas
    
    def handle_produce(self, partition: int, records: list, 
                       acks: str) -> ProduceResponse:
        """
        Handle produce request on leader.
        
        acks options:
        - '0': Fire and forget (fastest, may lose data)
        - '1': Wait for leader ack (balanced)
        - 'all'/-1: Wait for all ISR acks (safest)
        """
        # Append to local log
        local_log = self.replicas[partition].log
        base_offset = local_log.append(records)
        
        if acks == '0':
            # No wait
            return ProduceResponse(base_offset=base_offset)
        
        if acks == '1':
            # Just leader
            return ProduceResponse(base_offset=base_offset)
        
        if acks == 'all' or acks == '-1':
            # Wait for ISR
            high_watermark = self._wait_for_isr(
                partition, base_offset + len(records)
            )
            return ProduceResponse(
                base_offset=base_offset,
                high_watermark=high_watermark
            )
    
    def _wait_for_isr(self, partition: int, target_offset: int) -> int:
        """Wait until all ISR replicas have caught up."""
        while True:
            all_caught_up = True
            for replica_id in self.isr[partition]:
                if replica_id == self.broker_id:
                    continue
                replica_offset = self._get_replica_offset(
                    partition, replica_id
                )
                if replica_offset < target_offset:
                    all_caught_up = False
                    break
            
            if all_caught_up:
                return target_offset
            
            time.sleep(0.001)  # 1ms
    
    def fetch_from_leader(self, partition: int, 
                          fetch_offset: int) -> list:
        """Follower fetches from leader."""
        leader = self._find_leader(partition)
        
        response = leader.fetch(
            partition=partition,
            offset=fetch_offset,
            max_bytes=1048576
        )
        
        # Append to local log
        self.replicas[partition].log.append(response.records)
        
        return response.records


class HighWatermark:
    """
    High watermark: offset up to which all ISR replicas have received.
    
    Consumers can only read up to high watermark.
    This ensures consumers don't read uncommitted data.
    """
    
    def __init__(self):
        self.hwm = 0  # High watermark offset
        self.leo = 0  # Log end offset
    
    def update_hwm(self, isr_offsets: dict):
        """
        Update high watermark based on ISR offsets.
        
        HWM = min(offsets of all ISR replicas)
        """
        if isr_offsets:
            self.hwm = min(isr_offsets.values())
    
    def can_consumer_read(self, offset: int) -> bool:
        """Consumer can only read up to HWM."""
        return offset < self.hwm
```

## ZooKeeper Integration

### Metadata Management

```python
class ZooKeeperMetadata:
    """Kafka's use of ZooKeeper (pre-KRaft)."""
    
    def __init__(self, zk_client):
        self.zk = zk_client
    
    def get_broker_list(self) -> list:
        """Get list of live brokers."""
        # /brokers/ids/[0,1,2,...]
        broker_ids = self.zk.get_children('/brokers/ids')
        brokers = []
        for bid in broker_ids:
            data = self.zk.get(f'/brokers/ids/{bid}')
            brokers.append(json.loads(data))
        return brokers
    
    def get_topic_partitions(self, topic: str) -> dict:
        """Get partition info for topic."""
        # /brokers/topics/{topic}
        data = self.zk.get(f'/brokers/topics/{topic}')
        return json.loads(data)
    
    def get_partition_leader(self, topic: str, 
                             partition: int) -> int:
        """Get current leader for partition."""
        # /brokers/topics/{topic}/partitions/{partition}/state
        path = f'/brokers/topics/{topic}/partitions/{partition}/state'
        data = self.zk.get(path)
        state = json.loads(data)
        return state['leader']
    
    def register_broker(self, broker_id: int, host: str, port: int):
        """Register broker on startup."""
        # Ephemeral node - deleted when broker disconnects
        path = f'/brokers/ids/{broker_id}'
        data = json.dumps({
            'host': host,
            'port': port,
            'timestamp': time.time()
        })
        self.zk.create(path, data, ephemeral=True)
    
    def elect_controller(self, broker_id: int) -> bool:
        """
        Controller election via ZooKeeper.
        
        First broker to create /controller wins.
        """
        try:
            self.zk.create('/controller', 
                          str(broker_id), 
                          ephemeral=True)
            return True
        except NodeExistsError:
            return False
```

## Performance Optimizations

### Batching and Compression

```python
class PerformanceOptimizations:
    """Kafka performance techniques."""
    
    def producer_batching(self):
        """
        Batch multiple messages into single request.
        
        Benefits:
        - Fewer network round trips
        - Better compression ratio
        - More efficient disk writes
        """
        batch = []
        batch_size = 0
        max_batch_size = 16384  # 16KB
        
        while batch_size < max_batch_size:
            msg = self.queue.get(timeout=0.005)  # linger.ms
            if msg:
                batch.append(msg)
                batch_size += len(msg)
        
        # Send entire batch as one request
        self.send_batch(batch)
    
    def compression_comparison(self, messages: list):
        """
        Compression codec comparison.
        
        GZIP: Best ratio, highest CPU
        Snappy: Fast, moderate ratio
        LZ4: Fastest, good ratio
        ZSTD: Best balance of ratio and speed
        """
        import gzip
        import snappy
        import lz4.frame
        import zstd
        
        data = b''.join(messages)
        
        results = {
            'gzip': gzip.compress(data),
            'snappy': snappy.compress(data),
            'lz4': lz4.frame.compress(data),
            'zstd': zstd.compress(data)
        }
        
        for codec, compressed in results.items():
            ratio = len(compressed) / len(data)
            print(f"{codec}: {ratio:.2%} of original")
    
    def sequential_io(self):
        """
        Why Kafka is fast: Sequential I/O.
        
        Random I/O: ~100 ops/sec (seek time)
        Sequential I/O: ~100 MB/sec (no seek)
        
        Kafka only appends, never modifies.
        This enables sustained high throughput.
        """
        pass
```

## Exactly-Once Semantics

### Idempotent Producer

```python
class IdempotentProducer:
    """
    Exactly-once producer (Kafka 0.11+).
    
    Guarantees each message written exactly once,
    even with retries.
    """
    
    def __init__(self):
        self.producer_id = None
        self.sequence_numbers = {}  # partition -> sequence
    
    def initialize(self):
        """Get producer ID from broker."""
        response = self.broker.init_producer_id()
        self.producer_id = response.producer_id
    
    def send_idempotent(self, topic: str, partition: int, 
                        record: bytes) -> int:
        """
        Send with idempotency.
        
        Broker tracks (producer_id, partition, sequence).
        Duplicates are detected and deduplicated.
        """
        tp = TopicPartition(topic, partition)
        
        # Get next sequence number
        if tp not in self.sequence_numbers:
            self.sequence_numbers[tp] = 0
        sequence = self.sequence_numbers[tp]
        
        # Send with PID and sequence
        request = ProduceRequest(
            producer_id=self.producer_id,
            sequence=sequence,
            records=[record]
        )
        
        response = self.broker.produce(request)
        
        if response.error == 'DUPLICATE_SEQUENCE':
            # Already written, safe to ignore
            pass
        else:
            self.sequence_numbers[tp] = sequence + 1
        
        return response.offset


class TransactionalProducer:
    """
    Transactional producer for atomic writes.
    
    Enables exactly-once across multiple partitions.
    """
    
    def __init__(self, transactional_id: str):
        self.transactional_id = transactional_id
        self.producer_id = None
        self.epoch = 0
    
    def begin_transaction(self):
        """Begin new transaction."""
        self.broker.begin_transaction(
            transactional_id=self.transactional_id,
            producer_id=self.producer_id,
            epoch=self.epoch
        )
    
    def send(self, topic: str, partition: int, record: bytes):
        """Send within transaction."""
        self.broker.produce_transactional(
            transactional_id=self.transactional_id,
            topic=topic,
            partition=partition,
            record=record
        )
    
    def commit_transaction(self):
        """Commit transaction atomically."""
        self.broker.commit_transaction(
            transactional_id=self.transactional_id,
            producer_id=self.producer_id,
            epoch=self.epoch
        )
    
    def abort_transaction(self):
        """Abort transaction, discard all writes."""
        self.broker.abort_transaction(
            transactional_id=self.transactional_id,
            producer_id=self.producer_id,
            epoch=self.epoch
        )
```

## Key Results

### Production Performance

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kafka Performance                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Throughput (per broker):                                       │
│  ┌─────────────────────────────────────────────┐                │
│  │  Producer: 200,000+ messages/sec            │                │
│  │  Consumer: 400,000+ messages/sec            │                │
│  │  Aggregate: 2 million+ msg/sec (cluster)    │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  Latency:                                                       │
│  ┌─────────────────────────────────────────────┐                │
│  │  Produce (acks=1): 2-5ms                    │                │
│  │  Produce (acks=all): 5-15ms                 │                │
│  │  Consume: 1-2ms                             │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  Storage Efficiency:                                            │
│  ┌─────────────────────────────────────────────┐                │
│  │  With compression: 5-10x reduction          │                │
│  │  Sequential writes: ~600 MB/sec per disk    │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  At LinkedIn (2011):                                            │
│  ┌─────────────────────────────────────────────┐                │
│  │  10+ billion messages per day               │                │
│  │  1+ TB of data per day                      │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Influence and Legacy

### Impact on Industry

1. **Log-centric architecture**: Made append-only logs mainstream
2. **Stream processing**: Enabled Kafka Streams, ksqlDB
3. **Event sourcing**: Foundation for event-driven systems
4. **Microservices**: Standard for inter-service communication

### Evolution

```
┌──────────────────────────────────────────────────────────────┐
│                    Kafka Evolution                           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  2011: Original Paper                                        │
│  - Basic pub/sub                                             │
│  - Simple consumer model                                     │
│                                                               │
│  2015: Kafka 0.9                                             │
│  - New consumer API                                          │
│  - Security (SSL, SASL)                                      │
│                                                               │
│  2017: Kafka 0.11                                            │
│  - Exactly-once semantics                                    │
│  - Idempotent producer                                       │
│  - Transactions                                              │
│                                                               │
│  2022: Kafka 3.3 (KRaft)                                     │
│  - Remove ZooKeeper dependency                               │
│  - Self-managed metadata                                     │
│  - Simplified operations                                     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Key Takeaways

1. **Sequential I/O is fast**: Append-only enables high throughput
2. **Batch everything**: Messages, compression, network I/O
3. **Simple consumer model**: Offset-based is elegant and efficient
4. **Partitioning for scale**: Horizontal scaling via partitions
5. **Replication for durability**: ISR ensures no data loss
6. **Consumer groups for parallelism**: Easy to scale consumption
7. **Log as truth**: All data in the log, everything else derived
