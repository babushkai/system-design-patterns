# CQRS (Command Query Responsibility Segregation)

## TL;DR

CQRS separates read and write operations into different models. Commands modify state; Queries read state. Each model can be optimized for its purpose. Write model ensures invariants; Read model optimized for queries. Often paired with event sourcing. Benefits: independent scaling, optimized models. Costs: complexity, eventual consistency.

---

## The Problem

### Traditional Architecture

```
┌─────────────────────────────────────────┐
│              Single Model               │
│  ┌─────────────────────────────────┐   │
│  │       Domain Objects            │   │
│  │  (used for reads AND writes)    │   │
│  └─────────────────────────────────┘   │
│                  │                      │
│  ┌───────────────┴───────────────┐     │
│  │          Database             │     │
│  │   (same tables for both)      │     │
│  └───────────────────────────────┘     │
└─────────────────────────────────────────┘

Problems:
  - Read and write patterns differ
  - Single model compromises both
  - Scaling challenges
```

### Read vs Write Characteristics

```
Writes:
  - Low volume (relative)
  - Complex validation
  - Transactional
  - Strong consistency needed

Reads:
  - High volume (typically 10-100x writes)
  - No validation needed
  - Often denormalized
  - Eventual consistency often OK
```

---

## CQRS Architecture

### Basic Structure

```
┌─────────────────────────────────────────────────────────┐
│                      Application                        │
├────────────────────────┬────────────────────────────────┤
│     Command Side       │         Query Side             │
│                        │                                │
│   ┌──────────────┐     │     ┌──────────────┐          │
│   │   Commands   │     │     │   Queries    │          │
│   └──────┬───────┘     │     └──────┬───────┘          │
│          ▼             │            ▼                   │
│   ┌──────────────┐     │     ┌──────────────┐          │
│   │   Handlers   │     │     │   Handlers   │          │
│   └──────┬───────┘     │     └──────┬───────┘          │
│          ▼             │            ▼                   │
│   ┌──────────────┐     │     ┌──────────────┐          │
│   │ Write Model  │     │     │ Read Model   │          │
│   │  (Domain)    │────────►  │ (Projections)│          │
│   └──────┬───────┘     │     └──────┬───────┘          │
│          ▼             │            ▼                   │
│   ┌──────────────┐     │     ┌──────────────┐          │
│   │  Write DB    │     │     │   Read DB    │          │
│   └──────────────┘     │     └──────────────┘          │
└────────────────────────┴────────────────────────────────┘
```

### Command Side

```python
# Command: Intent to change state
@dataclass
class CreateOrderCommand:
    customer_id: str
    items: List[OrderItem]

# Handler: Validates and executes
class CreateOrderHandler:
    def handle(self, cmd: CreateOrderCommand):
        # Load domain object
        customer = self.customer_repo.get(cmd.customer_id)
        
        # Business logic and validation
        order = customer.create_order(cmd.items)
        
        # Persist
        self.order_repo.save(order)
        
        # Publish event for read side
        self.events.publish(OrderCreated(order))
```

### Query Side

```python
# Query: Request for data
@dataclass
class GetOrderSummaryQuery:
    order_id: str

# Handler: Retrieves from read model
class GetOrderSummaryHandler:
    def handle(self, query: GetOrderSummaryQuery):
        # Simple read from optimized store
        return self.read_db.get(
            f"order_summary:{query.order_id}"
        )
```

---

## Synchronization

### Event-Based Sync

```
Write Side           Events           Read Side
    │                   │                 │
    │  Save order       │                 │
    ▼                   │                 │
[Write DB]             │                 │
    │                   │                 │
    │  OrderCreated ───►│                 │
    │                   │                 │
    │                   ▼                 │
    │              [Event Bus]           │
    │                   │                 │
    │                   │  ───────────────►
    │                   │                 │
    │                   │                 ▼
    │                   │            [Projector]
    │                   │                 │
    │                   │                 ▼
    │                   │           [Read DB]
```

### Projector Implementation

```python
class OrderProjector:
    def handle(self, event):
        if isinstance(event, OrderCreated):
            summary = OrderSummary(
                id=event.order_id,
                customer_name=event.customer_name,
                total=event.total,
                status="created"
            )
            self.read_db.save(f"order_summary:{event.order_id}", summary)
        
        elif isinstance(event, OrderShipped):
            summary = self.read_db.get(f"order_summary:{event.order_id}")
            summary.status = "shipped"
            summary.shipped_at = event.timestamp
            self.read_db.save(f"order_summary:{event.order_id}", summary)
```

---

## Read Model Optimization

### Denormalization

```
Write model (normalized):
  Orders:     id, customer_id, total
  Customers:  id, name, email
  Items:      id, order_id, product_id, qty

Read model (denormalized):
  OrderSummary:
    order_id
    customer_name      ← Copied from Customers
    customer_email     ← Copied from Customers
    total
    item_count         ← Computed
    product_names[]    ← Copied from Products
```

### Multiple Read Models

```
Same events → Multiple optimized views

OrderCreated, ItemAdded, OrderShipped events
  ↓
┌─────────────────────────────────────────────┐
│ OrderSummaryProjection (for order page)     │
│ CustomerOrdersProjection (for customer page)│
│ ShippingDashboard (for logistics)           │
│ AnalyticsProjection (for reports)           │
└─────────────────────────────────────────────┘
```

### Read Store Options

```
Different stores for different needs:

Order summary:       Redis (fast key-value)
Full-text search:    Elasticsearch
Analytics:           ClickHouse (columnar)
Customer dashboard:  PostgreSQL (relational)

Each optimized for its use case
```

---

## Eventual Consistency

### The Trade-off

```
Command completes at T=0
Event processed at T=100ms
Read model updated at T=100ms

Query at T=50ms: Sees old data!

"Write succeeded but I don't see my change"
```

### Handling Strategies

**Optimistic UI:**
```javascript
async function submitOrder(order) {
  // Update UI immediately
  showOrder(order);
  
  // Submit command
  await api.createOrder(order);
  
  // UI already shows expected state
}
```

**Read from Write Model:**
```python
def get_order(order_id, ensure_consistent=False):
    if ensure_consistent:
        # Read from write model (slower, consistent)
        return write_db.get_order(order_id)
    else:
        # Read from read model (faster, eventually consistent)
        return read_db.get_order_summary(order_id)
```

**Version Tracking:**
```python
def get_order_if_version(order_id, min_version):
    summary = read_db.get(order_id)
    
    if summary.version >= min_version:
        return summary
    
    # Wait for read model to catch up
    wait_for_version(order_id, min_version, timeout=5s)
    return read_db.get(order_id)
```

---

## CQRS + Event Sourcing

### Natural Fit

```
Event Sourcing:
  Events are source of truth

CQRS:
  Write: Append events
  Read: Project events to read models

┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  Commands   │────►│ Event Store  │────►│ Projections│
│             │     │ (write side) │     │(read side) │
└─────────────┘     └──────────────┘     └────────────┘
```

### Implementation

```python
# Write side: Event sourcing
def handle_withdraw(cmd):
    account = event_store.load(cmd.account_id)
    
    # Validate using events
    if account.balance < cmd.amount:
        raise InsufficientFunds()
    
    # Append event
    event_store.append(
        cmd.account_id,
        MoneyWithdrawn(amount=cmd.amount)
    )

# Read side: Projection
class BalanceProjection:
    def project(self, event):
        if isinstance(event, MoneyWithdrawn):
            current = redis.get(f"balance:{event.account_id}")
            redis.set(f"balance:{event.account_id}", current - event.amount)
```

---

## Without Event Sourcing

### Simpler CQRS

```python
# Write side: Traditional ORM
def create_order(cmd):
    order = Order(
        customer_id=cmd.customer_id,
        items=cmd.items
    )
    db.session.add(order)
    db.session.commit()
    
    # Publish event for read side
    publish(OrderCreated(order.id, order.total))

# Read side: Separate database
@event_handler(OrderCreated)
def project_order(event):
    summary = {
        "id": event.order_id,
        "total": event.total,
        "status": "created"
    }
    read_db.orders.insert(summary)
```

### Shared Database

```
Simplest CQRS: Same database, different access patterns

Write:
  Use ORM, complex objects
  Transactional writes

Read:
  Raw SQL or simple queries
  Read replicas
  Cached results
```

---

## When to Use CQRS

### Good Fit

```
✓ High read-to-write ratio
✓ Complex domain with business rules
✓ Need for different read models
✓ Performance requirements differ for reads vs writes
✓ Team comfortable with complexity
```

### Poor Fit

```
✗ Simple CRUD applications
✗ Low traffic systems
✗ Need for immediate consistency
✗ Small team, tight deadline
✗ Reads and writes have same patterns
```

### Evolution Path

```
Start simple:
  1. Single model, single database
  
Add read replicas:
  2. Write to primary, read from replica
  
Introduce projections:
  3. Separate read models, event-driven sync
  
Full CQRS:
  4. Different databases, full separation
  
Add Event Sourcing:
  5. Event store as write model
```

---

## Common Patterns

### Task-Based UI

```
Traditional: CRUD form with all fields

CQRS: Specific commands

Instead of:
  UpdateUser(id, name, email, phone, address, ...)

Use:
  ChangeUserEmail(id, email)
  UpdateUserAddress(id, address)
  ChangePhoneNumber(id, phone)

Benefits:
  - Clear intent
  - Specific validation
  - Better audit trail
```

### Read Model per View

```
Each UI view has its own projection

Dashboard:     DashboardProjection
Order List:    OrderListProjection
Order Detail:  OrderDetailProjection

No joins at query time
Each projection denormalized for its view
```

### Synchronous Read-After-Write

```python
def create_and_return_order(cmd):
    # Create order (write side)
    order_id = command_handler.create_order(cmd)
    
    # Wait for read model to sync
    summary = poll_until_exists(
        f"order_summary:{order_id}",
        timeout=5s
    )
    
    return summary
```

---

## Testing CQRS

### Command Testing

```python
def test_withdraw_insufficient_funds():
    # Given account with balance 100
    account = Account(balance=100)
    
    # When withdrawing 200
    cmd = WithdrawCommand(account_id=account.id, amount=200)
    
    # Then should raise error
    with pytest.raises(InsufficientFundsError):
        handler.handle(cmd)
```

### Projection Testing

```python
def test_order_projection():
    # Given events
    events = [
        OrderCreated(order_id="1", total=100),
        ItemAdded(order_id="1", item="Widget"),
        OrderShipped(order_id="1")
    ]
    
    # When projected
    projection = OrderProjection()
    for event in events:
        projection.handle(event)
    
    # Then summary correct
    summary = projection.get("1")
    assert summary.status == "shipped"
    assert summary.total == 100
```

---

## Key Takeaways

1. **Separate reads and writes** - Different models for different needs
2. **Optimize each side** - Write for invariants, read for queries
3. **Sync via events** - Publish on write, project on read
4. **Accept eventual consistency** - Or pay for immediate
5. **Multiple read models OK** - Different views from same events
6. **Pairs well with Event Sourcing** - Natural combination
7. **Not always needed** - Adds complexity
8. **Start simple, evolve** - Don't over-engineer initially
