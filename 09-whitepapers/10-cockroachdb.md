# CockroachDB: The Resilient Geo-Distributed SQL Database

## Paper Overview

- **Title**: CockroachDB: The Resilient Geo-Distributed SQL Database
- **Authors**: Rebecca Taft et al. (Cockroach Labs)
- **Published**: SIGMOD 2020
- **Context**: Building a geo-distributed SQL database inspired by Spanner

## TL;DR

CockroachDB is an open-source, distributed SQL database that provides:
- **Serializable ACID transactions** across global deployments
- **Geo-partitioning** for data locality compliance
- **Automatic sharding and rebalancing** with no manual intervention
- **Multi-active availability** without a single leader

## Problem Statement

### Cloud Database Requirements

```
┌─────────────────────────────────────────────────────────────────┐
│               Modern Application Requirements                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SQL + ACID Transactions                                     │
│     ┌─────────────────────────────────────────────┐             │
│     │  Developers know SQL                        │             │
│     │  Business logic needs transactions          │             │
│     │  Strong consistency for correctness         │             │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
│  2. Horizontal Scalability                                      │
│     ┌─────────────────────────────────────────────┐             │
│     │  Add nodes to increase capacity             │             │
│     │  No single point of failure                 │             │
│     │  Automatic data distribution                │             │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
│  3. Geographic Distribution                                     │
│     ┌─────────────────────────────────────────────┐             │
│     │  Low latency for global users               │             │
│     │  Data residency compliance (GDPR)           │             │
│     │  Survive region failures                    │             │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
│  4. Cloud Native                                                │
│     ┌─────────────────────────────────────────────┐             │
│     │  Run on any cloud or on-prem                │             │
│     │  Kubernetes-native deployment               │             │
│     │  Pay-as-you-grow                            │             │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why Not Existing Solutions?

```
┌─────────────────────────────────────────────────────────────────┐
│               Existing Solutions' Limitations                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Traditional RDBMS (MySQL, PostgreSQL):                         │
│  ┌─────────────────────────────────────────────────┐            │
│  │  ✗ Single-node or primary-replica only          │            │
│  │  ✗ Manual sharding is complex                   │            │
│  │  ✗ Cross-shard transactions difficult           │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
│  NoSQL (Cassandra, MongoDB):                                    │
│  ┌─────────────────────────────────────────────────┐            │
│  │  ✗ Limited transaction support                  │            │
│  │  ✗ Eventual consistency issues                  │            │
│  │  ✗ No SQL (or limited SQL)                      │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
│  Google Spanner:                                                │
│  ┌─────────────────────────────────────────────────┐            │
│  │  ✓ Excellent, but...                            │            │
│  │  ✗ Proprietary (GCP only)                       │            │
│  │  ✗ Requires specialized hardware (TrueTime)     │            │
│  │  ✗ Expensive                                    │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
│  CockroachDB Goal:                                              │
│  ┌─────────────────────────────────────────────────┐            │
│  │  Open-source Spanner-like capabilities          │            │
│  │  Works on commodity hardware                    │            │
│  │  Runs anywhere (any cloud, on-prem)             │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 CockroachDB Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                        SQL Layer                          │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │   │
│  │  │   Parser   │  │  Planner   │  │ Optimizer  │          │   │
│  │  └────────────┘  └────────────┘  └────────────┘          │   │
│  │  ┌────────────────────────────────────────────┐          │   │
│  │  │              Distributed Executor           │          │   │
│  │  └────────────────────────────────────────────┘          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Transaction Layer                       │   │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐  │   │
│  │  │  Transaction     │  │  Parallel Commit Protocol    │  │   │
│  │  │  Coordinator     │  │  (2PC without blocking)      │  │   │
│  │  └──────────────────┘  └──────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Distribution Layer                      │   │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐  │   │
│  │  │  Range           │  │  Raft Consensus              │  │   │
│  │  │  Addressing      │  │  (per range)                 │  │   │
│  │  └──────────────────┘  └──────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     Storage Layer                         │   │
│  │  ┌──────────────────────────────────────────────────────┐│   │
│  │  │              Pebble (LSM-tree storage)               ││   │
│  │  │              (formerly RocksDB)                      ││   │
│  │  └──────────────────────────────────────────────────────┘│   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Distribution

```
┌─────────────────────────────────────────────────────────────────┐
│                     Range-Based Sharding                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Table: users (sorted by primary key)                           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Key Space                                                │   │
│  │                                                           │   │
│  │  [A───────F]  [G───────L]  [M───────R]  [S───────Z]      │   │
│  │     Range 1      Range 2      Range 3      Range 4       │   │
│  │                                                           │   │
│  │  Each range: ~512MB (configurable)                       │   │
│  │  Each range: 3 replicas via Raft                         │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Range Replica Placement:                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                           │   │
│  │  Node 1          Node 2          Node 3          Node 4  │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌────────┐│   │
│  │  │ R1(L)   │    │ R1(F)   │    │ R1(F)   │    │        ││   │
│  │  │ R2(F)   │    │ R2(L)   │    │         │    │ R2(F)  ││   │
│  │  │ R3(F)   │    │         │    │ R3(L)   │    │ R3(F)  ││   │
│  │  │         │    │ R4(F)   │    │ R4(F)   │    │ R4(L)  ││   │
│  │  └─────────┘    └─────────┘    └─────────┘    └────────┘│   │
│  │                                                           │   │
│  │  L = Leaseholder (serves reads)                          │   │
│  │  F = Follower (Raft replica)                             │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Transaction Protocol

### MVCC and Timestamps

```python
class CockroachMVCC:
    """
    CockroachDB's Multi-Version Concurrency Control.
    
    Every value is versioned with a hybrid logical clock (HLC)
    timestamp, enabling lock-free reads and serializable isolation.
    """
    
    def __init__(self):
        self.hlc = HybridLogicalClock()
    
    def write(self, key: bytes, value: bytes, txn: Transaction):
        """
        Write a value with MVCC versioning.
        
        Format: key@timestamp -> value
        """
        # Allocate write timestamp
        timestamp = self.hlc.now()
        
        # Check for write-write conflicts
        latest = self.get_latest_version(key)
        if latest and latest.timestamp > txn.read_timestamp:
            # Another transaction wrote after we started
            # Must retry with higher timestamp
            raise WriteTooOldError(latest.timestamp)
        
        # Write intent (uncommitted version)
        intent = WriteIntent(
            key=key,
            value=value,
            timestamp=timestamp,
            txn_id=txn.id,
            txn_record=txn.record_key
        )
        
        self.storage.put(intent)
        return timestamp
    
    def read(self, key: bytes, timestamp: Timestamp) -> bytes:
        """
        Read value at or before given timestamp.
        
        MVCC enables reads without blocking writes.
        """
        # Find latest version <= timestamp
        version = self.get_version_at(key, timestamp)
        
        if version is None:
            return None
        
        # Check if it's an intent (uncommitted)
        if version.is_intent:
            return self._handle_intent_on_read(version, timestamp)
        
        return version.value
    
    def _handle_intent_on_read(self, intent, read_timestamp):
        """
        Handle encountering an uncommitted write intent.
        
        Options:
        1. Wait for transaction to complete
        2. Push the transaction's timestamp
        3. Abort the blocking transaction
        """
        blocking_txn = self.get_transaction_record(intent.txn_id)
        
        if blocking_txn.status == 'COMMITTED':
            # Transaction committed, intent is valid
            return intent.value
        elif blocking_txn.status == 'ABORTED':
            # Transaction aborted, ignore intent
            return self.read(intent.key, read_timestamp)
        else:
            # Transaction pending - try to push
            success = self.push_transaction(
                blocking_txn, 
                push_to=read_timestamp
            )
            if success:
                return intent.value
            else:
                raise IntentError("Cannot resolve intent")


class HybridLogicalClock:
    """
    Hybrid Logical Clock for CockroachDB.
    
    Combines physical time with logical counter
    to provide causality without TrueTime.
    """
    
    def __init__(self):
        self.physical_time = 0
        self.logical_counter = 0
        self.max_offset = 500  # 500ms max clock skew
    
    def now(self) -> Timestamp:
        """
        Get current HLC timestamp.
        
        HLC = (physical_time, logical_counter)
        """
        wall_time = self._get_wall_time()
        
        if wall_time > self.physical_time:
            self.physical_time = wall_time
            self.logical_counter = 0
        else:
            self.logical_counter += 1
        
        return Timestamp(self.physical_time, self.logical_counter)
    
    def update(self, received: Timestamp):
        """
        Update clock based on received timestamp.
        
        Ensures causality: if A -> B, then ts(A) < ts(B)
        """
        wall_time = self._get_wall_time()
        
        if wall_time > self.physical_time and wall_time > received.physical:
            self.physical_time = wall_time
            self.logical_counter = 0
        elif received.physical > self.physical_time:
            self.physical_time = received.physical
            self.logical_counter = received.logical + 1
        elif received.physical == self.physical_time:
            self.logical_counter = max(
                self.logical_counter, 
                received.logical
            ) + 1
        else:
            self.logical_counter += 1
        
        return Timestamp(self.physical_time, self.logical_counter)
```

### Parallel Commit Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                    Parallel Commit Protocol                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Traditional 2PC:                                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Coordinator        Participant A      Participant B    │    │
│  │      │                   │                   │          │    │
│  │      │───Prepare────────>│                   │          │    │
│  │      │───Prepare─────────┼──────────────────>│          │    │
│  │      │<──Vote Yes────────│                   │          │    │
│  │      │<──Vote Yes────────┼───────────────────│          │    │
│  │      │                   │                   │          │    │
│  │      │   (Commit Point)  │                   │          │    │
│  │      │                   │                   │          │    │
│  │      │───Commit─────────>│                   │          │    │
│  │      │───Commit──────────┼──────────────────>│          │    │
│  │      │<──Ack─────────────│                   │          │    │
│  │      │<──Ack─────────────┼───────────────────│          │    │
│  │      │                   │                   │          │    │
│  │  Client waits for all 4 round trips!                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  CockroachDB Parallel Commit:                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Gateway           Range A             Range B          │    │
│  │      │                   │                   │          │    │
│  │      │───Write Intent───>│                   │          │    │
│  │      │───Write Intent────┼──────────────────>│          │    │
│  │      │───Staging Txn────>│  (first range)    │          │    │
│  │      │                   │                   │          │    │
│  │      │<──Ack─────────────│                   │          │    │
│  │      │<──Ack─────────────┼───────────────────│          │    │
│  │      │<──Ack─────────────│                   │          │    │
│  │      │                   │                   │          │    │
│  │  Client gets response after 1 round trip!               │    │
│  │                                                          │    │
│  │  Async commit happens in background                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Transaction Implementation

```python
class CockroachTransaction:
    """CockroachDB transaction implementation."""
    
    def __init__(self, gateway):
        self.gateway = gateway
        self.id = uuid.uuid4()
        self.status = 'PENDING'
        self.read_timestamp = gateway.hlc.now()
        self.write_timestamp = None
        self.intents = []  # Written keys
        self.in_flight_writes = []
    
    def begin(self):
        """Begin transaction."""
        # Create transaction record
        self.record = TransactionRecord(
            id=self.id,
            status='PENDING',
            timestamp=self.read_timestamp,
            in_flight_writes=[]
        )
        # Record stored on first write's range
    
    def read(self, key: bytes) -> bytes:
        """
        Read key within transaction.
        
        Uses read timestamp for consistent snapshot.
        """
        range_desc = self.gateway.lookup_range(key)
        leaseholder = range_desc.leaseholder
        
        # Read at transaction's read timestamp
        result = leaseholder.mvcc_read(key, self.read_timestamp)
        
        # Handle write-read conflict
        if result.uncertain:
            # Value written in our uncertainty window
            # Must refresh read timestamp
            self._refresh_timestamp(result.write_timestamp)
            return self.read(key)
        
        return result.value
    
    def write(self, key: bytes, value: bytes):
        """
        Write key within transaction.
        
        Creates write intent, not committed value.
        """
        range_desc = self.gateway.lookup_range(key)
        leaseholder = range_desc.leaseholder
        
        # Write intent through Raft
        intent = WriteIntent(
            key=key,
            value=value,
            txn_id=self.id,
            timestamp=self.read_timestamp
        )
        
        leaseholder.raft_propose(intent)
        self.intents.append(intent)
    
    def commit(self) -> bool:
        """
        Commit transaction using parallel commit.
        
        1. Stage transaction (mark as STAGING)
        2. Write intents in parallel
        3. Return to client after all acks
        4. Async: Mark COMMITTED, resolve intents
        """
        # Stage transaction record
        self.record.status = 'STAGING'
        self.record.in_flight_writes = [i.key for i in self.intents]
        
        first_range = self.gateway.lookup_range(self.intents[0].key)
        first_range.leaseholder.update_txn_record(self.record)
        
        # All intents already written, just need acks
        # Client can return after staging is durable
        
        # Async: Finalize commit
        self._async_finalize_commit()
        
        return True
    
    def _async_finalize_commit(self):
        """
        Finalize commit in background.
        
        1. Mark transaction COMMITTED
        2. Resolve all intents to committed values
        """
        # Mark committed
        self.record.status = 'COMMITTED'
        self._update_txn_record()
        
        # Resolve intents
        for intent in self.intents:
            range_desc = self.gateway.lookup_range(intent.key)
            range_desc.leaseholder.resolve_intent(
                intent.key, 
                self.id, 
                'COMMITTED'
            )
    
    def _refresh_timestamp(self, new_timestamp):
        """
        Refresh read timestamp to handle write-read conflicts.
        
        Only possible if no writes yet, or reads can be re-validated.
        """
        if self.intents:
            raise TransactionRetryError(
                "Cannot refresh with outstanding writes"
            )
        
        self.read_timestamp = new_timestamp + 1
```

## Raft Consensus

### Per-Range Raft Groups

```
┌─────────────────────────────────────────────────────────────────┐
│                   Raft Per Range                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Range 42: Keys [user:100, user:200)                            │
│                                                                  │
│  Node A              Node B              Node C                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Range 42  │    │   Range 42  │    │   Range 42  │         │
│  │   (Leader)  │    │  (Follower) │    │  (Follower) │         │
│  │             │    │             │    │             │         │
│  │  Raft Log:  │    │  Raft Log:  │    │  Raft Log:  │         │
│  │  [1,2,3,4,5]│───>│  [1,2,3,4,5]│    │  [1,2,3,4,5]│         │
│  │             │    │             │    │             │         │
│  │ Leaseholder │    │             │    │             │         │
│  │ (serves     │    │             │    │             │         │
│  │  reads)     │    │             │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│        │                   │                  │                 │
│        │              Replication             │                 │
│        │◄──────────────────┴──────────────────┘                 │
│                                                                  │
│  Each range is an independent Raft group                       │
│  Thousands of ranges = thousands of Raft groups                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Leaseholder Optimization

```python
class RangeLeaseholder:
    """
    Leaseholder optimization for reads.
    
    One replica per range holds the lease and
    serves reads without Raft consensus.
    """
    
    def __init__(self, range_id: int):
        self.range_id = range_id
        self.lease = None
        self.raft_log = []
        self.state_machine = {}
    
    def acquire_lease(self) -> bool:
        """
        Acquire range lease via Raft.
        
        Lease is a time-bounded lock on serving reads.
        """
        lease = Lease(
            holder=self.node_id,
            start=self.hlc.now(),
            expiration=self.hlc.now() + LEASE_DURATION
        )
        
        # Propose lease acquisition through Raft
        success = self.raft_propose(LeaseRequest(lease))
        
        if success:
            self.lease = lease
            return True
        return False
    
    def serve_read(self, key: bytes, timestamp: Timestamp) -> bytes:
        """
        Serve read request as leaseholder.
        
        No Raft required - just check local state!
        """
        # Verify we still hold lease
        if not self._is_lease_valid():
            raise NotLeaseholderError()
        
        # Ensure timestamp is within lease period
        if timestamp > self.lease.expiration:
            # Need to extend lease
            self._extend_lease()
        
        # Read from local state machine
        return self.state_machine.get(key, timestamp)
    
    def serve_write(self, key: bytes, value: bytes, 
                    txn: Transaction) -> bool:
        """
        Serve write request.
        
        Writes always go through Raft.
        """
        # Propose write intent through Raft
        proposal = WriteIntentProposal(
            key=key,
            value=value,
            txn_id=txn.id,
            timestamp=txn.write_timestamp
        )
        
        return self.raft_propose(proposal)
    
    def _is_lease_valid(self) -> bool:
        """Check if our lease is still valid."""
        return (
            self.lease is not None and
            self.hlc.now() < self.lease.expiration
        )


class LeaseTransfer:
    """
    Transfer lease to optimize for locality.
    
    Move lease closer to where queries originate.
    """
    
    def transfer_lease(self, from_node, to_node, range_id):
        """
        Transfer lease from one node to another.
        
        Used for:
        - Follow-the-workload (move lease to query source)
        - Geo-partitioning (keep lease in correct region)
        - Load balancing
        """
        # Current leaseholder proposes transfer
        proposal = LeaseTransferProposal(
            range_id=range_id,
            from_node=from_node,
            to_node=to_node,
            new_lease=Lease(
                holder=to_node,
                start=self.hlc.now(),
                expiration=self.hlc.now() + LEASE_DURATION
            )
        )
        
        return from_node.raft_propose(proposal)
```

## SQL Layer

### Distributed SQL Execution

```
┌─────────────────────────────────────────────────────────────────┐
│               Distributed Query Execution                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query: SELECT * FROM orders                                    │
│         WHERE region = 'US' AND amount > 100                    │
│         ORDER BY created_at DESC LIMIT 10                       │
│                                                                  │
│  Gateway Node (receives query)                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. Parse SQL                                             │   │
│  │  2. Plan query                                            │   │
│  │  3. Optimize (cost-based)                                 │   │
│  │  4. Distribute to relevant ranges                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│              ┌──────────────┼──────────────┐                    │
│              ▼              ▼              ▼                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Node 2     │  │   Node 3     │  │   Node 4     │          │
│  │              │  │              │  │              │          │
│  │  TableReader │  │  TableReader │  │  TableReader │          │
│  │  Filter      │  │  Filter      │  │  Filter      │          │
│  │  (local scan)│  │  (local scan)│  │  (local scan)│          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                    │
│         └─────────────────┼─────────────────┘                    │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Gateway Node                                             │   │
│  │  - Merge sorted results                                   │   │
│  │  - Apply LIMIT                                            │   │
│  │  - Return to client                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Query Optimizer

```python
class CockroachOptimizer:
    """CockroachDB's cost-based query optimizer."""
    
    def __init__(self):
        self.stats = TableStatistics()
    
    def optimize(self, query) -> PhysicalPlan:
        """
        Optimize query using cost-based optimization.
        
        1. Generate logical plan
        2. Explore equivalent plans
        3. Estimate costs
        4. Choose lowest cost plan
        """
        # Parse to AST
        ast = self.parse(query)
        
        # Build logical plan
        logical_plan = self.build_logical_plan(ast)
        
        # Explore equivalent plans (Cascades-style)
        memo = self.explore(logical_plan)
        
        # Extract best physical plan
        best_plan = self.extract_best(memo)
        
        return best_plan
    
    def estimate_cost(self, plan) -> Cost:
        """
        Estimate plan cost.
        
        Considers:
        - Row counts from statistics
        - Data distribution (which ranges)
        - Network hops
        - I/O costs
        """
        cost = Cost()
        
        for operator in plan.operators:
            if isinstance(operator, TableScan):
                rows = self.stats.estimate_rows(
                    operator.table,
                    operator.filter
                )
                cost.rows += rows
                cost.io += rows * self.COST_PER_ROW
                
            elif isinstance(operator, IndexScan):
                rows = self.stats.estimate_index_rows(
                    operator.index,
                    operator.bounds
                )
                cost.rows += rows
                cost.io += rows * self.COST_PER_INDEX_ROW
                
            elif isinstance(operator, DistributedJoin):
                # Add network cost for cross-node joins
                cost.network += self.estimate_join_shuffle(operator)
        
        return cost
    
    def locality_aware_planning(self, query) -> PhysicalPlan:
        """
        Optimize for data locality.
        
        - Push filters to range leaseholders
        - Minimize cross-region data movement
        - Use local indexes when possible
        """
        plan = self.optimize(query)
        
        # For each scan, determine which ranges
        for scan in plan.get_scans():
            ranges = self.lookup_ranges(scan.table, scan.span)
            
            # Group by locality (region/zone)
            by_locality = self.group_by_locality(ranges)
            
            # Assign scan processors close to data
            for locality, local_ranges in by_locality.items():
                processor = self.allocate_processor(locality)
                scan.assign_ranges(processor, local_ranges)
        
        return plan
```

## Geo-Partitioning

### Regional Data Placement

```
┌─────────────────────────────────────────────────────────────────┐
│                   Geo-Partitioning                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CREATE TABLE users (                                           │
│      id UUID PRIMARY KEY,                                       │
│      region STRING NOT NULL,                                    │
│      name STRING,                                               │
│      email STRING                                               │
│  ) PARTITION BY LIST (region) (                                 │
│      PARTITION us VALUES IN ('us-east', 'us-west'),             │
│      PARTITION eu VALUES IN ('eu-west', 'eu-central'),          │
│      PARTITION asia VALUES IN ('asia-east', 'asia-south')       │
│  );                                                             │
│                                                                  │
│  ALTER PARTITION us OF TABLE users                              │
│      CONFIGURE ZONE USING constraints='[+region=us]';           │
│  ALTER PARTITION eu OF TABLE users                              │
│      CONFIGURE ZONE USING constraints='[+region=eu]';           │
│  ALTER PARTITION asia OF TABLE users                            │
│      CONFIGURE ZONE USING constraints='[+region=asia]';         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                                                          │    │
│  │   US Region          EU Region          Asia Region     │    │
│  │   ┌─────────┐       ┌─────────┐       ┌─────────┐       │    │
│  │   │  users  │       │  users  │       │  users  │       │    │
│  │   │ (us-*)  │       │ (eu-*)  │       │(asia-*) │       │    │
│  │   │         │       │         │       │         │       │    │
│  │   │ 3 reps  │       │ 3 reps  │       │ 3 reps  │       │    │
│  │   │ in US   │       │ in EU   │       │ in Asia │       │    │
│  │   └─────────┘       └─────────┘       └─────────┘       │    │
│  │                                                          │    │
│  │   GDPR compliant: EU data stays in EU                   │    │
│  │   Low latency: Data close to users                      │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Locality-Optimized Reads

```python
class GeoPartitioning:
    """Geo-partitioning for data locality and compliance."""
    
    def configure_partition(self, table: str, partition: str, 
                           constraints: list):
        """
        Configure partition placement constraints.
        
        Constraints like [+region=us] ensure all
        replicas are in US region.
        """
        zone_config = ZoneConfig(
            constraints=constraints,
            num_replicas=3,
            lease_preferences=[['+region=us']]  # Prefer US leaseholder
        )
        
        self.apply_zone_config(table, partition, zone_config)
    
    def route_query(self, query, user_region: str) -> str:
        """
        Route query to appropriate region.
        
        For partitioned tables, queries on partition
        key go directly to correct region.
        """
        # Extract partition key from query
        partition_key = self.extract_partition_key(query)
        
        if partition_key:
            # Route to partition's region
            target_region = self.get_partition_region(partition_key)
            return self.get_gateway(target_region)
        else:
            # Global query - use closest gateway
            return self.get_closest_gateway(user_region)
    
    def follow_the_workload(self, range_id: int):
        """
        Automatically move leaseholder to where queries originate.
        
        Reduces cross-region latency for reads.
        """
        # Track query origins for this range
        query_origins = self.get_query_origins(range_id)
        
        # Find dominant region
        dominant_region = max(
            query_origins.items(), 
            key=lambda x: x[1]
        )[0]
        
        # Transfer lease if not already there
        current_leaseholder = self.get_leaseholder(range_id)
        if current_leaseholder.region != dominant_region:
            target = self.find_replica_in_region(
                range_id, 
                dominant_region
            )
            self.transfer_lease(range_id, target)
```

## Clock Skew Handling

### Uncertainty Interval

```python
class ClockSkewHandling:
    """
    CockroachDB's approach to clock skew without TrueTime.
    
    Key insight: Use uncertainty intervals instead of
    guaranteed clock bounds.
    """
    
    def __init__(self):
        self.max_offset = timedelta(milliseconds=500)
    
    def read_with_uncertainty(self, key: bytes, 
                               read_timestamp: Timestamp) -> Result:
        """
        Read handling clock uncertainty.
        
        If a value was written in our uncertainty window,
        we can't be sure if it happened before or after us.
        """
        # Uncertainty window: [read_ts, read_ts + max_offset]
        uncertainty_limit = read_timestamp + self.max_offset
        
        # Get value at read timestamp
        value = self.mvcc_get(key, read_timestamp)
        
        # Check for uncertain values
        uncertain_value = self.mvcc_get_in_range(
            key, 
            read_timestamp, 
            uncertainty_limit
        )
        
        if uncertain_value:
            # Value written during uncertainty window
            # Must restart transaction with higher timestamp
            return Result(
                uncertain=True,
                write_timestamp=uncertain_value.timestamp
            )
        
        return Result(value=value, uncertain=False)
    
    def restart_on_uncertainty(self, txn: Transaction, 
                               uncertain_timestamp: Timestamp):
        """
        Restart transaction after uncertainty error.
        
        Move read timestamp forward to clear uncertainty.
        """
        # New read timestamp is after uncertain write
        txn.read_timestamp = uncertain_timestamp + 1
        
        # Clear read set (must re-read everything)
        txn.read_set.clear()
        
        # Re-execute transaction
        return self.execute_transaction(txn)


class ClockSynchronization:
    """
    Clock synchronization strategies.
    
    CockroachDB relies on well-synchronized clocks.
    """
    
    def configure_clock_sync(self):
        """
        Configure clock synchronization.
        
        Options:
        1. NTP with tight tolerances
        2. PTP (Precision Time Protocol)
        3. Cloud provider time services
        """
        # Cloud providers offer synchronized clocks:
        # - AWS: Amazon Time Sync Service
        # - GCP: Compute Engine NTP
        # - Azure: Virtual Machine time sync
        
        # Configure max offset based on sync quality
        self.max_offset = self.measure_clock_offset()
    
    def adjust_for_clock_skew(self, cluster_max_offset: float):
        """
        Trade-off: lower offset = less uncertainty,
        but requires better synchronization.
        
        Default: 500ms (safe for most deployments)
        With good sync: 100-250ms
        """
        self.max_offset = cluster_max_offset
```

## Automatic Operations

### Range Splitting and Merging

```python
class RangeManagement:
    """Automatic range splitting and merging."""
    
    def __init__(self):
        self.max_range_size = 512 * 1024 * 1024  # 512MB
        self.min_range_size = 128 * 1024 * 1024  # 128MB
    
    def check_and_split(self, range_id: int):
        """
        Split range if too large.
        
        Automatic - no manual intervention needed.
        """
        range_size = self.get_range_size(range_id)
        
        if range_size > self.max_range_size:
            # Find split point (middle key)
            split_key = self.find_split_key(range_id)
            
            # Propose split through Raft
            self.raft_propose_split(range_id, split_key)
    
    def check_and_merge(self, range_id: int):
        """
        Merge adjacent ranges if too small.
        
        Reduces overhead of many small ranges.
        """
        range_size = self.get_range_size(range_id)
        
        if range_size < self.min_range_size:
            # Find adjacent range
            adjacent = self.get_adjacent_range(range_id)
            
            if adjacent and self.can_merge(range_id, adjacent):
                self.raft_propose_merge(range_id, adjacent)


class RebalancingScheduler:
    """Automatic replica rebalancing."""
    
    def rebalance(self):
        """
        Continuously rebalance replicas across nodes.
        
        Goals:
        - Even distribution of ranges per node
        - Even distribution of leaseholders
        - Respect zone constraints
        - Minimize rebalancing traffic
        """
        while True:
            # Find imbalanced nodes
            node_loads = self.get_node_loads()
            
            overloaded = [n for n in node_loads if n.load > self.avg_load * 1.1]
            underloaded = [n for n in node_loads if n.load < self.avg_load * 0.9]
            
            for src in overloaded:
                if not underloaded:
                    break
                
                dst = underloaded.pop()
                
                # Find range to move
                range_to_move = self.select_range_to_move(src)
                
                # Move replica
                self.move_replica(range_to_move, src, dst)
            
            time.sleep(60)  # Check every minute
    
    def respect_constraints(self, range_id: int, target: Node) -> bool:
        """
        Check if target respects zone constraints.
        
        Cannot move replica to node that violates
        locality constraints.
        """
        zone_config = self.get_zone_config(range_id)
        
        for constraint in zone_config.constraints:
            if not target.matches(constraint):
                return False
        
        return True
```

## Key Results

### Performance

```
┌─────────────────────────────────────────────────────────────────┐
│                   CockroachDB Performance                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Throughput (YCSB benchmarks):                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  - Read-heavy (95% read):  ~120K ops/sec (3 nodes)      │    │
│  │  - Write-heavy (50% write): ~50K ops/sec (3 nodes)      │    │
│  │  - Scales linearly with nodes                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Latency:                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  - Single-region read:    1-2ms                         │    │
│  │  - Single-region write:   5-10ms                        │    │
│  │  - Cross-region write:    ~RTT + 10ms                   │    │
│  │  - Parallel commit:       ~50% latency reduction        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Availability:                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  - Survives single node failure: Yes                    │    │
│  │  - Survives AZ failure: Yes (with 3+ AZs)              │    │
│  │  - Survives region failure: Yes (with 3+ regions)      │    │
│  │  - Recovery time: Seconds (Raft leader election)        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Influence and Legacy

### Impact

```
┌──────────────────────────────────────────────────────────────┐
│                   CockroachDB's Impact                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Democratized Spanner:                                       │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  - Open source (BSL, transitioning to open)         │     │
│  │  - Runs on commodity hardware (no TrueTime)         │     │
│  │  - Cloud-agnostic (any cloud, on-prem)              │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
│  Key Innovations:                                            │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  - Parallel commit (reduces 2PC latency)            │     │
│  │  - HLC for causality without TrueTime               │     │
│  │  - Automatic range management                       │     │
│  │  - SQL compatibility (PostgreSQL wire protocol)     │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
│  Adoption:                                                   │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  - DoorDash, Netflix, Bose, others                  │     │
│  │  - Cloud offering (CockroachDB Serverless)          │     │
│  │  - Growing NewSQL category                          │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Key Takeaways

1. **Spanner for everyone**: Serializable geo-distributed SQL without TrueTime
2. **Range-based sharding**: Simple model, automatic splits/merges
3. **Raft per range**: Independent consensus groups scale horizontally
4. **Leaseholder optimization**: Serve reads without Raft round trips
5. **Parallel commit**: Reduce 2PC latency by pipelining
6. **HLC for causality**: Logical clocks work without specialized hardware
7. **Geo-partitioning**: Data residency compliance built in
