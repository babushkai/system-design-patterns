# Distributed Transactions

## TL;DR

Distributed transactions coordinate atomic operations across multiple nodes or services. Two-phase commit (2PC) is the classic approach but blocks on coordinator failure. Three-phase commit reduces blocking but adds latency. Sagas handle long-running transactions with compensating actions. Modern systems often avoid distributed transactions entirely, using eventual consistency with idempotency instead.

---

## Why Distributed Transactions Are Hard

### The Problem

```
Transfer $100 from Account A (Node 1) to Account B (Node 2)

Step 1: Debit A (-$100)   ← Node 1
Step 2: Credit B (+$100)  ← Node 2

What if:
  - Node 2 crashes after Step 1?
  - Network fails between steps?
  - Either node says "no"?
  
Money disappears or duplicates!
```

### ACID in Distributed Systems

```
Atomicity:  All nodes commit or all abort
Consistency: All nodes maintain invariants
Isolation:  Concurrent transactions don't interfere
Durability: Committed data survives failures

Challenge: Achieving these across network boundaries
```

---

## Two-Phase Commit (2PC)

### The Protocol

```
Phase 1: Prepare (Voting)
  Coordinator → All participants: "Can you commit?"
  Participants: Acquire locks, prepare to commit
  Participants → Coordinator: "Yes" or "No"

Phase 2: Commit (Decision)
  If all voted "Yes":
    Coordinator → All participants: "Commit"
    Participants: Make changes permanent
  Else:
    Coordinator → All participants: "Abort"
    Participants: Roll back
```

### Sequence Diagram

```
Coordinator          Participant A          Participant B
     │                    │                      │
     │───PREPARE─────────►│                      │
     │───PREPARE──────────┼─────────────────────►│
     │                    │                      │
     │◄──VOTE_YES─────────│                      │
     │◄──VOTE_YES─────────┼──────────────────────│
     │                    │                      │
     │───COMMIT──────────►│                      │
     │───COMMIT───────────┼─────────────────────►│
     │                    │                      │
     │◄──ACK──────────────│                      │
     │◄──ACK──────────────┼──────────────────────│
```

### State Machine

```
Participant states:
  INITIAL → PREPARED → COMMITTED
                    ↘ ABORTED

Coordinator states:
  INITIAL → WAITING → COMMITTED/ABORTED

Key rule: Once PREPARED, must wait for coordinator decision
```

### The Blocking Problem

```
Scenario: Coordinator crashes after sending PREPARE

Participant A: PREPARED, waiting for decision...
Participant B: PREPARED, waiting for decision...

Both are blocked:
  - Can't commit (don't know if others voted yes)
  - Can't abort (coordinator might have decided commit)
  - Resources locked indefinitely

Solution: Wait for coordinator recovery (or timeout and abort)
```

### 2PC Implementation

```python
class Coordinator:
    def execute_transaction(self, participants, operations):
        # Phase 1: Prepare
        votes = []
        for p in participants:
            try:
                vote = p.prepare(operations[p])
                votes.append(vote)
                self.log.write(f"VOTE:{p}:{vote}")
            except Timeout:
                self.log.write(f"VOTE:{p}:TIMEOUT")
                votes.append("NO")
        
        # Decision
        if all(v == "YES" for v in votes):
            decision = "COMMIT"
        else:
            decision = "ABORT"
        
        self.log.write(f"DECISION:{decision}")
        self.log.fsync()  # Durable decision!
        
        # Phase 2: Execute decision
        for p in participants:
            p.execute_decision(decision)

class Participant:
    def prepare(self, operation):
        self.acquire_locks(operation)
        self.log.write(f"PREPARED:{operation}")
        self.log.fsync()
        return "YES"
    
    def execute_decision(self, decision):
        if decision == "COMMIT":
            self.apply_changes()
            self.release_locks()
        else:
            self.rollback()
            self.release_locks()
```

---

## Three-Phase Commit (3PC)

### Motivation

Add a "pre-commit" phase to reduce blocking.

```
Phase 1: CanCommit (Voting)
  Coordinator → Participants: "Can you commit?"
  Participants → Coordinator: "Yes" or "No"

Phase 2: PreCommit
  If all Yes: Coordinator → Participants: "PreCommit"
  Participants acknowledge, prepare to commit

Phase 3: DoCommit
  Coordinator → Participants: "DoCommit"
  Participants commit
```

### Non-Blocking Property

```
Key insight: Participant in PreCommit state knows:
  - All participants voted Yes
  - Safe to commit after timeout (no need to wait for coordinator)

If coordinator crashes during PreCommit:
  Participants can elect new coordinator
  New coordinator can complete the commit
```

### Limitations

```
Problem: Network partition can still cause inconsistency

Partition:
  Coordinator + Participant A: Decide to abort
  Participant B: In PreCommit, times out, commits

Result: A aborted, B committed → Inconsistency

3PC helps with coordinator crashes, not partitions
```

---

## Saga Pattern

### Concept

Long-running transaction as a sequence of local transactions.
Each local transaction has a compensating transaction.

```
Saga: Book a trip
  T1: Reserve flight    →  C1: Cancel flight reservation
  T2: Reserve hotel     →  C2: Cancel hotel reservation
  T3: Reserve car       →  C3: Cancel car reservation
  T4: Charge credit card → C4: Refund credit card

If T3 fails:
  Run C2, C1 (reverse order)
  Trip booking failed, all reservations released
```

### Choreography (Event-Driven)

```
Each service listens for events and reacts:

Flight Service:
  On "TripRequested" → Reserve flight, emit "FlightReserved"
  On "HotelFailed" → Cancel flight, emit "FlightCancelled"

Hotel Service:
  On "FlightReserved" → Reserve hotel, emit "HotelReserved"
  On failure → emit "HotelFailed"

No central coordinator
Services react to events
```

```
┌─────────────┐     FlightReserved     ┌─────────────┐
│   Flight    │───────────────────────►│    Hotel    │
│   Service   │◄───────────────────────│   Service   │
└─────────────┘     HotelFailed        └─────────────┘
       │                                      │
       │ FlightCancelled                      │ HotelReserved
       ▼                                      ▼
    [Done]                              ┌─────────────┐
                                        │     Car     │
                                        │   Service   │
                                        └─────────────┘
```

### Orchestration (Central Coordinator)

```
Saga Coordinator:
  1. Call Flight Service → Reserve flight
  2. Call Hotel Service → Reserve hotel
  3. If fail → Call Flight Service → Cancel
  4. Call Car Service → Reserve car
  5. If fail → Call Hotel → Cancel, Call Flight → Cancel
  6. Success → Done

Central coordinator knows the saga state
Easier to understand and debug
Single point of failure
```

```
                  ┌──────────────────┐
                  │ Saga Coordinator │
                  └────────┬─────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │  Flight   │    │   Hotel   │    │    Car    │
   │  Service  │    │  Service  │    │  Service  │
   └───────────┘    └───────────┘    └───────────┘
```

### Compensating Transactions

```python
class BookTripSaga:
    def execute(self, trip_request):
        try:
            flight = self.flight_service.reserve(trip_request.flight)
            try:
                hotel = self.hotel_service.reserve(trip_request.hotel)
                try:
                    car = self.car_service.reserve(trip_request.car)
                    try:
                        self.payment_service.charge(trip_request.payment)
                    except PaymentError:
                        self.car_service.cancel(car)
                        raise
                except CarError:
                    self.hotel_service.cancel(hotel)
                    raise
            except HotelError:
                self.flight_service.cancel(flight)
                raise
        except FlightError:
            raise SagaFailed("Could not book trip")
```

### Saga Guarantees

```
NOT ACID:
  - No isolation (intermediate states visible)
  - No atomicity (compensation is best-effort)

ACD (Atomicity through saga, Consistency, Durability):
  - Eventually consistent
  - Compensation may fail (need retries, manual intervention)
```

---

## Transactional Outbox

### Problem

How to update database AND send message atomically?

```
Naive approach:
  1. Update database
  2. Send message to queue
  
Failure mode:
  Database updated, message send fails
  OR
  Message sent, database update fails
```

### Solution

Write message to outbox table in same transaction.

```sql
BEGIN TRANSACTION;
  -- Business update
  UPDATE accounts SET balance = balance - 100 WHERE id = 1;
  
  -- Outbox entry
  INSERT INTO outbox (id, payload, created_at)
  VALUES (uuid(), '{"event":"Debited","amount":100}', NOW());
COMMIT;
```

Separate process polls outbox, publishes messages:

```python
def publish_outbox():
    while True:
        events = db.query("SELECT * FROM outbox ORDER BY created_at LIMIT 100")
        for event in events:
            try:
                message_queue.publish(event.payload)
                db.execute("DELETE FROM outbox WHERE id = ?", event.id)
            except PublishError:
                pass  # Retry next iteration
        sleep(100ms)
```

### CDC (Change Data Capture) Alternative

```
Database transaction log → CDC → Message queue

Example: Debezium
  Reads MySQL binlog
  Publishes to Kafka

No polling, lower latency
Guaranteed ordering
```

---

## XA Transactions

### Standard Interface

```
XA: eXtended Architecture (X/Open standard)

Coordinator: Transaction Manager
Participants: Resource Managers (databases, queues)

Interface:
  xa_start()    - Begin transaction
  xa_end()      - End transaction branch
  xa_prepare()  - Prepare to commit
  xa_commit()   - Commit
  xa_rollback() - Rollback
```

### Java Example (JTA)

```java
// Get XA resources
UserTransaction tx = (UserTransaction) ctx.lookup("java:comp/UserTransaction");

try {
    tx.begin();
    
    // Operation on database 1
    Connection conn1 = dataSource1.getConnection();
    conn1.prepareStatement("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
         .executeUpdate();
    
    // Operation on database 2
    Connection conn2 = dataSource2.getConnection();
    conn2.prepareStatement("UPDATE accounts SET balance = balance + 100 WHERE id = 2")
         .executeUpdate();
    
    tx.commit();  // 2PC happens here
} catch (Exception e) {
    tx.rollback();
}
```

### Limitations

```
- Performance overhead (prepare phase, logging)
- Blocking (resources locked during 2PC)
- Homogeneous participants (all must support XA)
- Not widely supported in modern systems
```

---

## Avoiding Distributed Transactions

### Design to Avoid

```
1. Single database per service
   No cross-database transactions needed

2. Eventual consistency with idempotency
   Accept that operations complete asynchronously
   
3. Aggregate boundaries
   All related data in one partition
   
4. Command Query Responsibility Segregation (CQRS)
   Separate read and write models
```

### Idempotent Operations

```
Instead of distributed transaction:

1. Assign unique ID to operation
2. Store ID with each change
3. On retry, check if ID already processed

No coordination needed
Each service handles idempotency locally
```

### Reservation Pattern

```
Instead of: Debit A, Credit B (distributed)

Use:
  1. Create pending transfer record
  2. Reserve funds from A (mark as held)
  3. Credit B
  4. Complete transfer (release hold from A)

Each step is local
Failures leave system in recoverable state
```

---

## Comparison

| Approach | Consistency | Latency | Complexity | Use Case |
|----------|-------------|---------|------------|----------|
| 2PC | Strong | High | Medium | Databases |
| 3PC | Strong | Higher | High | Critical systems |
| Saga | Eventual | Low | High | Microservices |
| Outbox | Eventual | Low | Medium | Event-driven |
| Avoid | Eventual | Lowest | Low | Most systems |

---

## Key Takeaways

1. **2PC guarantees atomicity** - But blocks on failure
2. **3PC reduces blocking** - But doesn't handle partitions
3. **Sagas are eventual** - Compensation may fail
4. **Outbox pattern is reliable** - Database + events atomically
5. **XA is heavyweight** - Use only when necessary
6. **Best approach: avoid** - Design for eventual consistency
7. **Idempotency is key** - Makes retries safe
8. **Each step should be recoverable** - Know how to compensate or retry
