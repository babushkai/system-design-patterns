# Event Sourcing

## TL;DR

Event sourcing stores all changes to application state as a sequence of events. Instead of storing current state, you store the history of what happened. Current state is derived by replaying events. Benefits: complete audit trail, temporal queries, debugging. Costs: complexity, eventual consistency, storage growth. Often paired with CQRS.

---

## Traditional vs Event Sourcing

### Traditional (State-Based)

```
Database stores current state:

Users table:
  id: 123
  balance: 500
  updated_at: 2024-01-15

Problem: History is lost
  What was the balance yesterday?
  How did we get to 500?
  Unknown
```

### Event Sourcing

```
Database stores events:

Events table:
  AccountCreated(id=123, balance=1000)
  MoneyWithdrawn(id=123, amount=200)
  MoneyDeposited(id=123, amount=300)
  MoneyWithdrawn(id=123, amount=600)

Current state: Replay events
  1000 - 200 + 300 - 600 = 500 ✓

Complete history preserved
```

---

## Core Concepts

### Event

```python
@dataclass
class Event:
    event_id: str
    aggregate_id: str
    event_type: str
    timestamp: datetime
    data: dict
    version: int

# Example
AccountCreated(
    event_id="evt-001",
    aggregate_id="account-123",
    event_type="AccountCreated",
    timestamp="2024-01-15T10:00:00Z",
    data={"owner": "Alice", "initial_balance": 1000},
    version=1
)
```

### Event Store

```
Append-only log of events

┌──────────────────────────────────────────────┐
│ Event 1 │ Event 2 │ Event 3 │ ... │ Event N │
└──────────────────────────────────────────────┘
     ↑
  Append only (no updates, no deletes)
```

### Aggregate

```
Domain entity that groups related events
Events always belong to an aggregate

Account aggregate:
  Events: Created, Deposited, Withdrawn, Closed
  
Order aggregate:
  Events: Placed, Confirmed, Shipped, Delivered
```

### Command

```
Represents intent to change state
Validated, then generates events

Command: Withdraw(account_id=123, amount=100)
  
Validation:
  - Account exists? ✓
  - Sufficient balance? ✓
  
Result: MoneyWithdrawn event generated
```

---

## Event Store Implementation

### Schema

```sql
CREATE TABLE events (
    event_id UUID PRIMARY KEY,
    aggregate_id VARCHAR(255) NOT NULL,
    aggregate_type VARCHAR(255) NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    event_data JSONB NOT NULL,
    metadata JSONB,
    version INT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    UNIQUE (aggregate_id, version)  -- Optimistic concurrency
);

CREATE INDEX idx_events_aggregate ON events(aggregate_id, version);
CREATE INDEX idx_events_timestamp ON events(timestamp);
```

### Append Events

```python
class EventStore:
    def append(self, aggregate_id, events, expected_version):
        with transaction():
            # Check optimistic concurrency
            current = self.get_latest_version(aggregate_id)
            if current != expected_version:
                raise ConcurrencyError(
                    f"Expected version {expected_version}, got {current}"
                )
            
            # Append events
            for i, event in enumerate(events):
                event.version = expected_version + i + 1
                self.db.insert(event)
            
            # Publish events
            for event in events:
                self.publish(event)
```

### Load Aggregate

```python
def load_aggregate(aggregate_id):
    # Get all events for aggregate
    events = event_store.get_events(aggregate_id)
    
    # Replay to rebuild state
    aggregate = Account()
    for event in events:
        aggregate.apply(event)
    
    return aggregate

class Account:
    def apply(self, event):
        if event.type == "AccountCreated":
            self.id = event.data["id"]
            self.balance = event.data["initial_balance"]
        elif event.type == "MoneyDeposited":
            self.balance += event.data["amount"]
        elif event.type == "MoneyWithdrawn":
            self.balance -= event.data["amount"]
```

---

## Snapshots

### The Problem

```
Account with 10,000 events
Every load: replay 10,000 events
Very slow!
```

### Snapshot Solution

```
Every N events, save current state as snapshot

Events: 1-1000
Snapshot at event 1000: {balance: 5000, ...}
Events: 1001-2000

Load process:
  1. Load snapshot (if exists)
  2. Replay only events after snapshot
  
Replay 1000 events instead of 2000
```

### Implementation

```python
def load_aggregate_with_snapshot(aggregate_id):
    # Try to load snapshot
    snapshot = snapshot_store.get_latest(aggregate_id)
    
    if snapshot:
        aggregate = deserialize(snapshot.state)
        start_version = snapshot.version + 1
    else:
        aggregate = Account()
        start_version = 0
    
    # Replay events since snapshot
    events = event_store.get_events(
        aggregate_id, 
        from_version=start_version
    )
    
    for event in events:
        aggregate.apply(event)
    
    return aggregate

def save_snapshot(aggregate_id, aggregate, version):
    snapshot_store.save(
        aggregate_id=aggregate_id,
        state=serialize(aggregate),
        version=version
    )
```

---

## Projections

### Concept

```
Events (source of truth)
    ↓ Project
Read Models (optimized for queries)

Same events → multiple projections
Each optimized for specific use case
```

### Examples

```
Events:
  AccountCreated(id=1, owner="Alice")
  MoneyDeposited(id=1, amount=1000)
  AccountCreated(id=2, owner="Bob")
  MoneyWithdrawn(id=1, amount=500)

Projection: Account Balances
  {id: 1, balance: 500}
  {id: 2, balance: 0}

Projection: Activity Timeline
  [
    {time: T1, action: "Account 1 created"},
    {time: T2, action: "Deposit of 1000 to Account 1"},
    ...
  ]

Projection: Owner Directory
  {Alice: [1], Bob: [2]}
```

### Building Projections

```python
class BalanceProjection:
    def __init__(self):
        self.balances = {}
    
    def handle(self, event):
        if event.type == "AccountCreated":
            self.balances[event.data["id"]] = event.data.get("initial_balance", 0)
        elif event.type == "MoneyDeposited":
            self.balances[event.aggregate_id] += event.data["amount"]
        elif event.type == "MoneyWithdrawn":
            self.balances[event.aggregate_id] -= event.data["amount"]
    
    def rebuild_from_start(self):
        self.balances = {}
        for event in event_store.get_all_events():
            self.handle(event)
```

---

## Benefits

### Complete Audit Trail

```
Every change is recorded
Who did what, when

Question: "Why is balance 500?"
Answer: Replay events and see each change
```

### Temporal Queries

```python
def get_balance_at_time(account_id, timestamp):
    events = event_store.get_events(
        account_id,
        before=timestamp
    )
    
    balance = 0
    for event in events:
        if event.type == "MoneyDeposited":
            balance += event.data["amount"]
        elif event.type == "MoneyWithdrawn":
            balance -= event.data["amount"]
    
    return balance

# What was balance on Jan 1?
get_balance_at_time("account-123", "2024-01-01")
```

### Debugging

```
Bug in production:
  1. Capture events that led to bug
  2. Replay locally
  3. Debug with full history
  4. Fix and test with same events
```

### Schema Evolution

```
Events are facts about the past
Don't change events, add new types

v1: UserCreated(name)
v2: UserCreated(name, email)  # New field

Old events still valid
New code handles both versions
```

---

## Challenges

### Eventual Consistency

```
Event stored → Projection updated (async)

Gap where projection is stale
UI might show outdated data

Solutions:
  - Accept eventual consistency
  - Read from event store for critical reads
  - Optimistic UI updates
```

### Storage Growth

```
Events never deleted
Storage grows forever

Mitigations:
  - Snapshots (reduce replay time)
  - Archival (move old events to cold storage)
  - Event compaction (carefully, for specific patterns)
```

### Event Schema Changes

```
Challenge: Past events are immutable

Solutions:
  - Version events explicitly
  - Upcasting: Transform old events when reading
  - Weak schema: Store as JSON, handle missing fields
```

```python
def upcast_event(event):
    if event.type == "UserCreated" and event.version == 1:
        # Add default email for v1 events
        event.data["email"] = None
        event.version = 2
    return event
```

### Complex Queries

```
Event store optimized for:
  - Append
  - Read by aggregate

NOT optimized for:
  - Complex queries across aggregates
  - Aggregations

Solution: Projections for query needs
```

---

## Event Sourcing Patterns

### Command → Event

```python
def handle_withdraw(cmd: WithdrawCommand):
    # Load aggregate
    account = load_aggregate(cmd.account_id)
    
    # Validate
    if account.balance < cmd.amount:
        raise InsufficientFundsError()
    
    # Generate event
    event = MoneyWithdrawn(
        account_id=cmd.account_id,
        amount=cmd.amount,
        timestamp=now()
    )
    
    # Store event
    event_store.append(cmd.account_id, [event], account.version)
    
    return event
```

### Saga/Process Manager

```
Coordinate multiple aggregates

OrderSaga:
  On OrderPlaced:
    Send ReserveInventory command
  
  On InventoryReserved:
    Send ChargePayment command
  
  On PaymentCharged:
    Send ShipOrder command
  
  On PaymentFailed:
    Send ReleaseInventory command
```

### Event Replay for Migration

```python
def migrate_to_new_projection():
    # Create new projection store
    new_projection = NewProjection()
    
    # Replay all events
    for event in event_store.get_all_events():
        new_projection.handle(event)
    
    # Switch over
    swap_projection(old_projection, new_projection)
```

---

## When to Use Event Sourcing

### Good Fit

```
✓ Strong audit requirements (finance, healthcare)
✓ Complex domain with business rules
✓ Need for temporal queries
✓ Event-driven architecture
✓ CQRS implementation
```

### Poor Fit

```
✗ Simple CRUD applications
✗ No audit requirements
✗ Team unfamiliar with pattern
✗ Very high-throughput writes (rebuilding is slow)
✗ Need for ad-hoc queries across data
```

---

## Key Takeaways

1. **Store events, not state** - State is derived
2. **Events are immutable** - Never update or delete
3. **Snapshots prevent slow rebuilds** - Take periodically
4. **Projections for queries** - Multiple views from same events
5. **Eventual consistency is normal** - Design for it
6. **Great for audit trails** - Complete history
7. **Complexity is real** - Not for simple CRUD
8. **Pairs well with CQRS** - Separate read/write models
