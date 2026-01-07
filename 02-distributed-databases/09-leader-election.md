# Leader Election

## TL;DR

Leader election designates one node to coordinate actions in a distributed system. Use consensus-based election (Raft/Paxos) for strong guarantees, or simpler approaches (bully algorithm, lease-based) for less critical systems. The key challenge is ensuring exactly one leader exists at any time—split-brain is the enemy. Fencing tokens prevent stale leaders from causing damage.

---

## Why Elect a Leader?

### Simplifies Coordination

```
Without leader:
  All nodes coordinate → O(n²) messages
  Conflicts possible → complex resolution

With leader:
  Leader coordinates → O(n) messages
  Single decision maker → no conflicts
```

### Use Cases

```
Database:     Leader accepts writes
Queue:        Leader assigns partitions
Cache:        Leader manages keys
Scheduler:    Leader distributes tasks
Lock service: Leader grants locks
```

---

## Election Algorithms

### Bully Algorithm

Highest-ranked node wins.

```
Nodes have ranks: A(1) < B(2) < C(3) < D(4) < E(5)

C detects leader (E) failed:
  1. C sends ELECTION to D, E (higher ranks)
  2. D responds "I'm alive"
  3. D sends ELECTION to E
  4. E doesn't respond (failed)
  5. D becomes leader, broadcasts COORDINATOR
```

```
C           D           E (failed)
│           │               │
│──ELECTION►│               │
│           │──ELECTION────►│
│           │               ✗
│◄──ALIVE───│               │
│           │               │
│◄──COORDINATOR─────────────│
│           │               │
    D is new leader
```

**Pros:**
- Simple to implement
- Deterministic winner

**Cons:**
- Assumes reliable failure detection
- Highest rank always wins (inflexible)
- Not partition-tolerant

### Ring Algorithm

Token-based election around a ring.

```
Nodes form logical ring: A → B → C → D → A

B detects leader failed:
  1. B sends ELECTION(B) to C
  2. C adds self, sends ELECTION(B,C) to D
  3. D adds self, sends ELECTION(B,C,D) to A
  4. A adds self, sends ELECTION(B,C,D,A) to B
  5. B sees complete ring, picks highest, broadcasts COORDINATOR
```

**Pros:**
- No single point of failure
- All nodes participate

**Cons:**
- Slow (O(n) messages)
- Ring must be maintained
- Sensitive to failures during election

### Consensus-Based Election

Use Raft/Paxos for leader election.

```
Election is just agreeing on a value:
  "Who is the leader for term T?"

Raft:
  1. Candidate increments term
  2. Requests votes
  3. Majority grants → becomes leader
  4. Leader sends heartbeats to maintain authority
```

**Pros:**
- Partition-tolerant
- Strong guarantees
- Well-understood

**Cons:**
- Requires majority (2f+1 for f failures)
- More complex to implement

---

## Lease-Based Leadership

### Concept

Leader holds a time-limited lease.

```
Leader A acquires lease: valid until T+10s
Other nodes know: "A is leader until T+10s"

Before lease expires:
  A renews lease → continues as leader
  
If A crashes:
  Lease expires (T+10s)
  Others can acquire new lease
```

### Implementation

```python
class LeaseBasedLeader:
    def __init__(self, node_id, store):
        self.node_id = node_id
        self.store = store  # Distributed store like etcd
        self.lease_ttl = 10  # seconds
        
    def try_become_leader(self):
        # Try to acquire lease (atomic compare-and-swap)
        success = self.store.put_if_absent(
            key="/leader",
            value=self.node_id,
            ttl=self.lease_ttl
        )
        return success
    
    def renew_lease(self):
        # Extend lease if still leader
        current = self.store.get("/leader")
        if current == self.node_id:
            self.store.refresh("/leader", ttl=self.lease_ttl)
            return True
        return False
    
    def run(self):
        while True:
            if self.is_leader():
                if not self.renew_lease():
                    # Lost leadership
                    self.step_down()
            else:
                if self.try_become_leader():
                    self.become_leader()
            sleep(self.lease_ttl / 3)  # Renew well before expiry
```

### Clock Considerations

```
Problem: Clock skew

Leader:   thinks lease expires at 10:00:10
Follower: thinks lease expires at 10:00:05 (clock behind)

Follower might try to become leader early!

Solutions:
  1. Use distributed clock (NTP with tight bounds)
  2. Conservative grace period
  3. Fencing tokens (see below)
```

---

## Fencing Tokens

### The Problem

Stale leader doesn't know it's no longer leader.

```
Timeline:
  T=0:   Leader A acquires lease
  T=5:   A enters GC pause
  T=10:  Lease expires, B becomes leader
  T=15:  A wakes up, thinks it's still leader
  T=16:  A writes data (stale leader!)
  
Split-brain: Both A and B think they're leader
```

### Solution: Fencing Tokens

Monotonically increasing token with each lease.

```
Lease 1: token=100, holder=A
Lease 2: token=101, holder=B

Storage checks token on write:
  A attempts write with token=100
  Storage: "Current token is 101, rejecting 100"
  
Stale leader's writes rejected
```

### Implementation

```python
class FencedStorage:
    def __init__(self):
        self.current_token = 0
        self.data = {}
    
    def write(self, key, value, fencing_token):
        if fencing_token < self.current_token:
            raise StaleLeaderError(
                f"Token {fencing_token} < current {self.current_token}"
            )
        self.current_token = fencing_token
        self.data[key] = value

class Leader:
    def __init__(self, lease_service, storage):
        self.lease = lease_service
        self.storage = storage
        
    def do_work(self):
        token = self.lease.get_token()
        # All operations include token
        self.storage.write("key", "value", token)
```

---

## Leader Election in Practice

### Using etcd

```go
// Create session with TTL
session, err := concurrency.NewSession(client, concurrency.WithTTL(10))

// Create election on path
election := concurrency.NewElection(session, "/my-election/")

// Campaign to become leader (blocks until elected)
err = election.Campaign(ctx, "node-1")

// Now leader - do leader work
doLeaderWork()

// Resign if needed
election.Resign(ctx)
```

### Using ZooKeeper

```java
// Create ephemeral sequential node
String path = zk.create(
    "/election/leader-",
    nodeId.getBytes(),
    ZooDefs.Ids.OPEN_ACL_UNSAFE,
    CreateMode.EPHEMERAL_SEQUENTIAL
);

// Check if lowest sequence number
List<String> children = zk.getChildren("/election", false);
Collections.sort(children);

if (children.get(0).equals(path.substring("/election/".length()))) {
    // I am the leader
    becomeLeader();
} else {
    // Watch the node before me
    String watchPath = children.get(children.indexOf(myNode) - 1);
    zk.exists("/election/" + watchPath, watchCallback);
}
```

### Using Redis (Redlock)

```python
# Acquire lock with TTL
lock_key = "leader-lock"
lock_value = str(uuid.uuid4())  # Unique value for this node

# SET if not exists, with TTL
acquired = redis.set(lock_key, lock_value, nx=True, ex=10)

if acquired:
    try:
        # I am leader
        do_leader_work()
    finally:
        # Release only if still own the lock
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        redis.eval(lua_script, 1, lock_key, lock_value)
```

---

## Handling Split-Brain

### Detection

```
Symptoms:
  - Multiple nodes claiming leadership
  - Conflicting writes
  - Inconsistent state

Detection approaches:
  1. Heartbeat monitoring
  2. Quorum checks
  3. Fencing token validation
  4. State reconciliation
```

### Prevention

```
1. Majority quorum (can't have two majorities)
   Election requires N/2 + 1 votes
   
2. Fencing tokens
   Storage rejects old leaders

3. STONITH (Shoot The Other Node In The Head)
   Forcibly terminate other leader
   
4. Lease expiration
   Old leader's lease must expire before new election
```

### Recovery

```
If split-brain detected:
  1. Stop all leaders
  2. Compare states
  3. Reconcile conflicts
  4. Re-elect single leader
  5. Resume operations
```

---

## Leader Health Monitoring

### Heartbeats

```
Leader sends periodic heartbeats:
  Every 1 second: "I'm alive, term=5"

Followers track:
  last_heartbeat = now()
  
  if now() - last_heartbeat > election_timeout:
      start_election()
      
Typical values:
  Heartbeat interval: 100-500ms
  Election timeout: 1-5 seconds
```

### Quorum-Based Liveness

```
Leader checks it can still reach quorum:

def leader_loop():
    while is_leader:
        acks = send_heartbeat_to_all()
        if count(acks) < quorum:
            # Can't reach quorum, step down
            step_down()
        sleep(heartbeat_interval)
```

### Application-Level Health

```
Sometimes leader is alive but unhealthy:
  - Out of memory
  - Disk full
  - Can't process requests

Application health check:
  def is_healthy():
      return (
          memory_available() and
          disk_available() and
          can_process_request()
      )
  
  if not is_healthy():
      step_down()
```

---

## Graceful Leadership Transfer

### Planned Handoff

```
For maintenance, upgrades, rebalancing:

1. Current leader L1 prepares successor L2
2. L1 ensures L2's log is up-to-date
3. L1 sends TimeoutNow to L2 (start election immediately)
4. L2 wins election (most up-to-date)
5. L1 steps down

No availability gap
```

### Raft Leadership Transfer

```
Leader L1 wants to transfer to L2:

1. L1 stops accepting new client requests
2. L1 replicates all entries to L2
3. L1 sends TimeoutNow to L2
4. L2 starts election with incremented term
5. L2 wins (has all data)
6. L1 becomes follower
```

---

## Anti-Patterns

### No Fencing

```
Bad:
  if am_i_leader():
      do_write()
      
Problem: Leader status might have changed mid-operation

Good:
  token = get_fencing_token()
  do_write(token)  # Storage validates token
```

### Clock-Dependent Logic

```
Bad:
  if lease_expiry > now():
      am_leader = True
      
Problem: Clocks can be wrong

Good:
  Use lease refresh mechanism
  Include fencing tokens
  Use distributed consensus
```

### Ignoring Network Partitions

```
Bad:
  if ping(other_nodes):
      am_leader = True
      
Problem: Partition can create multiple "leaders"

Good:
  Require quorum
  Use fencing tokens
  Accept that minority partition can't elect leader
```

---

## Comparison of Approaches

| Approach | Consistency | Availability | Complexity |
|----------|-------------|--------------|------------|
| Bully | Weak | Low | Low |
| Ring | Weak | Medium | Low |
| Consensus (Raft) | Strong | Medium | High |
| Lease (etcd) | Strong | Medium | Medium |
| Lease (Redis) | Medium | High | Medium |

---

## Key Takeaways

1. **Leader simplifies coordination** - Single decision maker
2. **Split-brain is the enemy** - Prevent with quorums + fencing
3. **Fencing tokens are essential** - Reject stale leaders
4. **Leases need renewal** - Grace period for failures
5. **Consensus is safest** - Raft/Paxos for strong guarantees
6. **Clocks are unreliable** - Don't trust timestamps alone
7. **Monitor leader health** - Step down if unhealthy
8. **Plan for handoff** - Graceful transfer for maintenance
