# Single-Leader Replication

## TL;DR

Single-leader (master-slave) replication routes all writes through one node (the leader) and replicates to followers. It provides simple consistency guarantees and is the most common replication model. Trade-offs: leader is a bottleneck and single point of failure; failover is complex. Use synchronous replication for durability, asynchronous for performance.

---

## How It Works

### Basic Architecture

```
                    Writes
                      │
                      ▼
              ┌───────────────┐
              │    Leader     │
              │   (Primary)   │
              └───────┬───────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Follower │ │ Follower │ │ Follower │
    │    1     │ │    2     │ │    3     │
    └──────────┘ └──────────┘ └──────────┘
          │           │           │
          └───────────┼───────────┘
                      │
                   Reads
```

### Write Path

```
1. Client sends write to leader
2. Leader writes to local storage
3. Leader sends replication log to followers
4. Followers apply changes
5. (Optional) Leader waits for acknowledgment
6. Leader responds to client
```

### Replication Log

The leader maintains a log of all changes:

```
Log entry:
  - Log Sequence Number (LSN): 12345
  - Operation: INSERT
  - Table: users
  - Data: {id: 1, name: "Alice"}
  - Timestamp: 2024-01-15T10:30:00Z

Followers:
  1. Fetch entries after their last known LSN
  2. Apply entries in order
  3. Update their LSN position
```

---

## Synchronous vs Asynchronous Replication

### Synchronous Replication

Leader waits for follower acknowledgment before confirming write.

```
┌────────┐    ┌────────┐    ┌──────────┐
│ Client │    │ Leader │    │ Follower │
└───┬────┘    └───┬────┘    └────┬─────┘
    │             │              │
    │──write(x)──►│              │
    │             │──replicate──►│
    │             │              │
    │             │◄────ack──────│
    │             │              │
    │◄───ok───────│              │
    │             │              │
```

**Guarantees:**
- Data exists on at least 2 nodes before ack
- Follower is always up-to-date

**Trade-offs:**
- Write latency includes replication time
- Follower failure blocks writes
- Usually only 1 sync follower (semi-sync)

### Asynchronous Replication

Leader confirms immediately, replicates in background.

```
┌────────┐    ┌────────┐    ┌──────────┐
│ Client │    │ Leader │    │ Follower │
└───┬────┘    └───┬────┘    └────┬─────┘
    │             │              │
    │──write(x)──►│              │
    │◄───ok───────│              │
    │             │              │
    │             │──replicate──►│
    │             │              │
    │             │◄────ack──────│
```

**Trade-offs:**
- Fast writes (no waiting)
- Data loss possible if leader fails
- Followers may lag behind

### Semi-Synchronous

One follower is synchronous, others asynchronous.

```
Leader ──sync──► Follower 1 (must ack)
       └─async─► Follower 2 (background)
       └─async─► Follower 3 (background)
```

**Used by:** MySQL semi-sync, PostgreSQL sync_commit

---

## Replication Lag

### What Is Lag?

Time or operations between leader state and follower state.

```
Timeline:
  Leader:   [op1][op2][op3][op4][op5]
  Follower: [op1][op2][op3]
                        │
                  3 ops behind (lag)
```

### Measuring Lag

```sql
-- PostgreSQL
SELECT 
  client_addr,
  pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) as lag_bytes,
  replay_lag
FROM pg_stat_replication;

-- MySQL
SHOW SLAVE STATUS\G
-- Look at: Seconds_Behind_Master
```

### Causes of Lag

| Cause | Impact | Mitigation |
|-------|--------|------------|
| Network latency | Delay in log delivery | Faster network |
| Follower CPU | Slow apply | Better hardware |
| Large transactions | Big log entries | Smaller batches |
| Long-running queries | Apply blocked | Query timeouts |
| Follower disk I/O | Write bottleneck | Faster storage |

### Consistency Problems from Lag

**Read-your-writes violation:**
```
Client writes to leader
Client reads from lagged follower
  → Doesn't see own write
```

**Monotonic reads violation:**
```
Client reads from follower A (up-to-date)
Client reads from follower B (lagged)
  → Time appears to go backward
```

**Solutions:**
- Read from leader for your own data
- Sticky sessions (same follower)
- Include version/timestamp, wait if behind

---

## Handling Node Failures

### Follower Failure

Follower crashes and restarts.

```
Recovery:
1. Check last applied LSN in local storage
2. Request log entries from leader starting at LSN
3. Apply entries sequentially
4. Resume normal replication
```

### Leader Failure (Failover)

Leader crashes; need to promote a follower.

```
Steps:
1. Detect leader failure (timeout)
2. Choose new leader (most up-to-date follower)
3. Reconfigure followers to replicate from new leader
4. Redirect clients to new leader
5. (If old leader recovers) Demote to follower
```

### Failover Challenges

**Detecting failure:**
```
Is leader dead or just slow?

Too aggressive: false positive, unnecessary failover
Too conservative: extended downtime

Typical: 10-30 second timeout
```

**Choosing new leader:**
```
Options:
1. Most up-to-date follower (least data loss)
2. Pre-designated standby
3. Consensus among followers (Raft-style)
```

**Lost writes:**
```
Leader had commits not yet replicated:
  - Lost when new leader takes over
  - May cause conflicts if old leader recovers
  
Prevention:
  - Sync replication (at least 1 copy)
  - Don't ack until replicated
```

**Split brain:**
```
Old leader comes back, doesn't know it's demoted:
  Two nodes accept writes!
  
Prevention:
  - Fencing tokens
  - STONITH (kill old leader)
  - Epoch numbers
```

---

## Read Scaling

### Reading from Followers

Distribute read load across followers.

```
            ┌────────────────────────────┐
            │       Load Balancer        │
            └─────────────┬──────────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    │         │           │           │         │
    ▼         ▼           ▼           ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Leader │ │Follower│ │Follower│ │Follower│ │Follower│
│(writes)│ │(reads) │ │(reads) │ │(reads) │ │(reads) │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

### Read Scaling Math

```
Before scaling:
  Leader: 10,000 reads/sec, 1,000 writes/sec
  Bottleneck: Leader saturated

After adding 4 followers:
  Leader: 1,000 writes/sec (writes only)
  Followers: 2,500 reads/sec each
  Total reads: 10,000 reads/sec
  
Reads scale linearly with followers
Writes still limited to single leader
```

### Geo-Distribution

Place followers in different regions.

```
┌─────────────────┐
│ US-East (Leader)│
└────────┬────────┘
         │
    ┌────┼────┬────────────┐
    │    │    │            │
    ▼    ▼    ▼            ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│US-West│ │Europe│ │ Asia │ │US-East│
│Follow│ │Follow│ │Follow│ │Follow │
└──────┘ └──────┘ └──────┘ └──────┘

Users read from closest follower
Writes go to leader (higher latency for distant users)
```

---

## Statement-Based vs Row-Based Replication

### Statement-Based

Replicate the SQL statement.

```
Leader executes: INSERT INTO users VALUES (1, 'Alice')
Sends to followers: "INSERT INTO users VALUES (1, 'Alice')"
Followers execute same statement
```

**Problems:**
- Non-deterministic functions: `NOW()`, `RAND()`, `UUID()`
- Triggers, stored procedures may behave differently
- Order-dependent statements

### Row-Based (Logical)

Replicate the actual row changes.

```
Leader executes: INSERT INTO users VALUES (1, 'Alice')
Sends to followers: {table: users, type: INSERT, row: {id:1, name:'Alice'}}
Followers apply row change
```

**Advantages:**
- Deterministic
- Works with any statement
- Enables CDC (Change Data Capture)

**Trade-off:**
- Larger log for bulk updates
- Less human-readable

### Mixed Mode

Use statement-based when safe, row-based otherwise.

```
Simple INSERT → Statement-based (compact)
Statement with NOW() → Row-based (deterministic)
```

---

## Implementation Examples

### PostgreSQL Streaming Replication

```sql
-- Primary postgresql.conf
wal_level = replica
max_wal_senders = 10
synchronous_commit = on  -- or 'remote_apply'
synchronous_standby_names = 'follower1'

-- Replica recovery.conf (or standby.signal in PG12+)
primary_conninfo = 'host=primary port=5432 user=replicator'
recovery_target_timeline = 'latest'
```

### MySQL Replication

```sql
-- Leader
server-id = 1
log_bin = mysql-bin
binlog_format = ROW

-- Follower
server-id = 2
relay_log = relay-bin
read_only = ON

CHANGE MASTER TO
  MASTER_HOST = 'leader',
  MASTER_USER = 'replicator',
  MASTER_AUTO_POSITION = 1;
START SLAVE;
```

---

## Monitoring

### Key Metrics

| Metric | What It Shows | Alert Threshold |
|--------|---------------|-----------------|
| Replication lag | Follower behind leader | > 30 seconds |
| Log position diff | Bytes behind | > 100 MB |
| Follower state | Connected/disconnected | Not streaming |
| Apply rate | Log entries/second | Dropping |
| Disk usage | Log accumulation | > 80% |

### Health Checks

```python
def check_replication_health():
  leader_lsn = query_leader("SELECT pg_current_wal_lsn()")
  
  for follower in followers:
    follower_lsn = query_follower("SELECT pg_last_wal_replay_lsn()")
    lag = leader_lsn - follower_lsn
    
    if lag > threshold:
      alert(f"Follower {follower} lagging: {lag} bytes")
    
    if not follower.is_streaming:
      alert(f"Follower {follower} not connected")
```

---

## When to Use Single-Leader

### Good Fit

- Most reads, few writes (read-heavy workloads)
- Strong consistency requirements
- Simple operational model preferred
- Geographic read distribution
- Traditional OLTP applications

### Poor Fit

- Write-heavy workloads (leader bottleneck)
- Multi-region writes (latency to leader)
- Zero-downtime requirements (failover window)
- Conflicting writes from multiple locations

---

## Key Takeaways

1. **All writes through leader** - Simple consistency, single point of failure
2. **Sync replication for safety** - At cost of latency and availability
3. **Async for performance** - Accept potential data loss
4. **Replication lag is inevitable** - Design reads to handle it
5. **Failover is complex** - Split-brain, data loss, client redirect
6. **Scale reads with followers** - Writes don't scale
7. **Row-based replication is safer** - Deterministic, enables CDC
8. **Monitor lag continuously** - Early warning of problems
