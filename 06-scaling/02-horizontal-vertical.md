# Horizontal vs Vertical Scaling

## TL;DR

Vertical scaling (scaling up) adds more power to existing machines—more CPU, RAM, or storage. Horizontal scaling (scaling out) adds more machines to distribute the load. Most modern distributed systems use horizontal scaling for better fault tolerance and cost efficiency, though vertical scaling is simpler and works well for certain workloads.

---

## The Scaling Problem

```
Traffic Growth Over Time:

     Requests/sec
          │
     100k │                              ╱
          │                           ╱
      50k │                        ╱
          │                     ╱
      10k │              ╱─────╱
          │         ╱───╱
       1k │    ╱───╱
          │╱──╱
          └──────────────────────────────►
                                        Time

Question: How do we handle 100x traffic growth?
```

---

## Vertical Scaling (Scale Up)

```
Before:                        After:
┌──────────────────┐          ┌──────────────────┐
│     Server       │          │     Server       │
│                  │          │                  │
│  CPU: 4 cores    │    ───►  │  CPU: 64 cores   │
│  RAM: 16 GB      │          │  RAM: 512 GB     │
│  SSD: 500 GB     │          │  SSD: 10 TB NVMe │
│                  │          │                  │
└──────────────────┘          └──────────────────┘
```

### When Vertical Scaling Works Well

```python
# Single-threaded workloads benefit from faster CPUs
class SingleThreadedProcessor:
    """
    Vertical scaling helps: Faster CPU = faster processing
    Examples:
    - Complex calculations
    - Sequential data processing
    - Legacy applications
    """
    def process(self, data):
        result = complex_computation(data)  # CPU-bound
        return result

# Memory-intensive workloads
class InMemoryDatabase:
    """
    Vertical scaling helps: More RAM = more data in memory
    Examples:
    - Redis with large datasets
    - In-memory analytics
    - Caching layers
    """
    def __init__(self):
        self.data = {}  # All data in memory
    
    def get(self, key):
        return self.data.get(key)  # O(1) access
```

### Vertical Scaling Limits

```
AWS EC2 Instance Sizes (example):

Instance Type      vCPUs    Memory    Cost/hr
─────────────────────────────────────────────
t3.micro              2      1 GB     $0.01
t3.large              2      8 GB     $0.08
m5.xlarge             4     16 GB     $0.19
m5.4xlarge           16     64 GB     $0.77
m5.24xlarge          96    384 GB     $4.61
u-24tb1.metal       448   24,576 GB   $218.40  ← Maximum!

                    ┌─────────────────────────────────┐
                    │  CEILING: Physical limits        │
                    │  - Max CPU cores per socket      │
                    │  - Max RAM per motherboard       │
                    │  - Max I/O bandwidth             │
                    └─────────────────────────────────┘
```

---

## Horizontal Scaling (Scale Out)

```
Before:                        After:
┌──────────────────┐          ┌─────────────────────────────┐
│     Server       │          │       Load Balancer         │
│                  │          └──────────────┬──────────────┘
│  Handles 1000    │                         │
│  requests/sec    │    ───►    ┌────────────┼────────────┐
│                  │            │            │            │
└──────────────────┘       ┌────┴───┐   ┌────┴───┐   ┌────┴───┐
                           │Server 1│   │Server 2│   │Server 3│
                           │ 1000/s │   │ 1000/s │   │ 1000/s │
                           └────────┘   └────────┘   └────────┘
                           
                           Total: 3000 requests/sec
```

### Stateless Services Scale Horizontally

```python
from flask import Flask, request
import os

app = Flask(__name__)

# Stateless service - easy to scale horizontally
@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.json
    
    # No local state - any instance can handle this
    result = perform_calculation(data)
    
    return {"result": result, "server": os.getenv("HOSTNAME")}

# Each instance is identical and interchangeable
# ┌─────────┐  ┌─────────┐  ┌─────────┐
# │Instance1│  │Instance2│  │Instance3│
# │  Code   │  │  Code   │  │  Code   │
# │  (same) │  │  (same) │  │  (same) │
# └─────────┘  └─────────┘  └─────────┘
```

### Stateful Services Are Harder

```python
# BAD: Stateful service - hard to scale
class SessionStore:
    def __init__(self):
        self.sessions = {}  # Local state!
    
    def set_session(self, session_id, data):
        self.sessions[session_id] = data
    
    def get_session(self, session_id):
        return self.sessions.get(session_id)

# Problem: User might hit different server each request
# Request 1 → Server1 (session created here)
# Request 2 → Server2 (session not found!)

# SOLUTION: Externalize state
import redis

class ExternalSessionStore:
    def __init__(self):
        self.redis = redis.Redis(host='redis-cluster')
    
    def set_session(self, session_id, data):
        self.redis.setex(session_id, 3600, json.dumps(data))
    
    def get_session(self, session_id):
        data = self.redis.get(session_id)
        return json.loads(data) if data else None
```

```
Externalized State Architecture:

┌─────────────────────────────────────────────────┐
│                Load Balancer                     │
└─────────────────────┬───────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────┴───┐    ┌────┴───┐    ┌────┴───┐
   │Server 1│    │Server 2│    │Server 3│
   │Stateless│   │Stateless│   │Stateless│
   └────┬───┘    └────┬───┘    └────┬───┘
        │             │             │
        └─────────────┼─────────────┘
                      │
        ┌─────────────┴─────────────┐
        │     Redis Cluster         │
        │   (Shared State Store)    │
        └───────────────────────────┘
```

---

## Database Scaling Strategies

### Vertical Scaling (Simpler)

```sql
-- Single powerful database server
-- Handles all reads and writes

CREATE TABLE orders (
    id BIGINT PRIMARY KEY,
    user_id INT,
    amount DECIMAL(10,2),
    created_at TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_user_orders ON orders(user_id);
CREATE INDEX idx_created ON orders(created_at);

-- Upgrade server hardware when needed:
-- More RAM → larger buffer pool
-- Faster SSD → faster I/O
-- More cores → more parallel queries
```

### Horizontal Scaling (Read Replicas)

```
Write Scaling: Still limited to primary

                    ┌─────────────────┐
     Writes ───────►│    Primary      │
                    │   (Writable)    │
                    └────────┬────────┘
                             │
              Replication    │
              Stream         │
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Replica 1  │     │   Replica 2  │     │   Replica 3  │
│  (Read-only) │     │  (Read-only) │     │  (Read-only) │
└──────────────┘     └──────────────┘     └──────────────┘
        ▲                    ▲                    ▲
        │                    │                    │
        └────────────────────┴────────────────────┘
                             │
                          Reads
```

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import random

class DatabaseRouter:
    def __init__(self):
        self.primary = create_engine('postgresql://primary:5432/db')
        self.replicas = [
            create_engine('postgresql://replica1:5432/db'),
            create_engine('postgresql://replica2:5432/db'),
            create_engine('postgresql://replica3:5432/db'),
        ]
    
    def get_write_session(self):
        """All writes go to primary"""
        Session = sessionmaker(bind=self.primary)
        return Session()
    
    def get_read_session(self):
        """Reads distributed across replicas"""
        replica = random.choice(self.replicas)
        Session = sessionmaker(bind=replica)
        return Session()

# Usage
router = DatabaseRouter()

# Write operation
with router.get_write_session() as session:
    order = Order(user_id=123, amount=99.99)
    session.add(order)
    session.commit()

# Read operation (can hit any replica)
with router.get_read_session() as session:
    orders = session.query(Order).filter_by(user_id=123).all()
```

### Horizontal Scaling (Sharding)

```
Full Horizontal Write Scaling:

                    ┌─────────────────┐
                    │  Shard Router   │
                    └────────┬────────┘
                             │
          user_id % 4 = ?    │
                             │
     ┌───────────────┬───────┴───────┬───────────────┐
     │               │               │               │
     ▼               ▼               ▼               ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Shard 0 │    │ Shard 1 │    │ Shard 2 │    │ Shard 3 │
│users    │    │users    │    │users    │    │users    │
│0,4,8... │    │1,5,9... │    │2,6,10...│    │3,7,11...│
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

---

## Comparison Table

| Aspect | Vertical Scaling | Horizontal Scaling |
|--------|-----------------|-------------------|
| **Complexity** | Simple | Complex |
| **Downtime** | Required for upgrade | Zero-downtime possible |
| **Cost curve** | Exponential | Linear |
| **Failure impact** | Single point of failure | Fault tolerant |
| **Maximum capacity** | Hardware limits | Theoretically unlimited |
| **Data consistency** | Easy (single node) | Complex (distributed) |
| **Application changes** | Minimal | May require redesign |
| **Operational overhead** | Low | High |

---

## Cost Analysis

```
Vertical Scaling Cost Curve:

Cost ($)
    │
    │                                    ╱
    │                                 ╱──
    │                              ╱──
    │                           ╱──
    │                        ╱──
    │                    ╱───
    │               ╱────
    │         ╱─────
    │    ╱────
    │╱───
    └────────────────────────────────────►
                                   Capacity

Cost grows exponentially!
2x performance ≠ 2x cost (often 3-4x cost)


Horizontal Scaling Cost Curve:

Cost ($)
    │
    │                              ╱
    │                          ╱
    │                      ╱
    │                  ╱
    │              ╱
    │          ╱
    │      ╱
    │  ╱
    │╱
    └────────────────────────────────────►
                                   Capacity

Cost grows linearly (with some overhead)
2x servers ≈ 2x cost (+ coordination overhead)
```

```python
def calculate_scaling_cost():
    """Compare costs for 10x capacity increase"""
    
    # Vertical: Upgrade from m5.xlarge to m5.24xlarge
    vertical_before = 0.192 * 24 * 30  # $138/month
    vertical_after = 4.608 * 24 * 30   # $3,318/month
    vertical_ratio = vertical_after / vertical_before  # 24x cost!
    
    # Horizontal: Add more m5.xlarge instances
    horizontal_before = 0.192 * 24 * 30        # $138/month (1 instance)
    horizontal_after = 0.192 * 24 * 30 * 10    # $1,382/month (10 instances)
    horizontal_overhead = 0.10  # 10% for load balancer, coordination
    horizontal_total = horizontal_after * (1 + horizontal_overhead)
    horizontal_ratio = horizontal_total / horizontal_before  # 11x cost
    
    return {
        "vertical_cost_ratio": vertical_ratio,    # 24x
        "horizontal_cost_ratio": horizontal_ratio  # 11x
    }
```

---

## Hybrid Approach

Most production systems use both strategies:

```
Optimal Architecture:

┌─────────────────────────────────────────────────────────────┐
│                     Internet                                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │        Global Load Balancer         │
         │     (Horizontal - DNS-based)        │
         └─────────────────┬──────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
         ▼                                   ▼
┌─────────────────┐                 ┌─────────────────┐
│   Region: US    │                 │   Region: EU    │
│                 │                 │                 │
│  ┌───────────┐  │                 │  ┌───────────┐  │
│  │    LB     │  │                 │  │    LB     │  │
│  └─────┬─────┘  │                 │  └─────┬─────┘  │
│        │        │                 │        │        │
│   ┌────┴────┐   │                 │   ┌────┴────┐   │
│   │ │ │ │ │ │   │                 │   │ │ │ │ │ │   │
│   └─┬─┬─┬─┬─┘   │                 │   └─┬─┬─┬─┬─┘   │
│     │ │ │ │     │                 │     │ │ │ │     │
│   Web Servers   │                 │   Web Servers   │
│   (Horizontal)  │                 │   (Horizontal)  │
│                 │                 │                 │
│  ┌───────────┐  │                 │  ┌───────────┐  │
│  │  Cache    │  │                 │  │  Cache    │  │
│  │ (Vertical)│  │                 │  │ (Vertical)│  │
│  │ 256GB RAM │  │                 │  │ 256GB RAM │  │
│  └───────────┘  │                 │  └───────────┘  │
│                 │                 │                 │
│  ┌───────────┐  │                 │  ┌───────────┐  │
│  │ Database  │  │                 │  │ Database  │  │
│  │  Primary  │◄─┼─ Replication ──┼──│  Replica  │  │
│  │(Vertical) │  │                 │  │           │  │
│  └───────────┘  │                 │  └───────────┘  │
└─────────────────┘                 └─────────────────┘
```

```python
class HybridScalingStrategy:
    """
    Decision framework for when to scale vertically vs horizontally
    """
    
    def recommend_scaling(self, service_type: str, bottleneck: str) -> str:
        recommendations = {
            # Stateless services: Scale horizontally
            ("api", "cpu"): "horizontal",
            ("api", "memory"): "horizontal",
            ("worker", "cpu"): "horizontal",
            
            # Caches: Often vertical first, then horizontal
            ("cache", "memory"): "vertical_then_horizontal",
            ("cache", "cpu"): "horizontal",
            
            # Databases: Depends on workload
            ("database", "reads"): "horizontal_replicas",
            ("database", "writes"): "vertical_then_shard",
            ("database", "storage"): "horizontal_sharding",
            
            # Message queues: Horizontal by design
            ("queue", "throughput"): "horizontal_partitions",
        }
        
        return recommendations.get(
            (service_type, bottleneck),
            "evaluate_case_by_case"
        )
    
    def scale_decision(self, metrics: dict) -> dict:
        """Make scaling decision based on current metrics"""
        
        if metrics['cpu_usage'] > 80:
            if metrics['service_type'] == 'stateless':
                return {
                    "action": "scale_out",
                    "add_instances": self._calculate_instances_needed(
                        metrics['cpu_usage'],
                        target=60
                    )
                }
            else:
                return {
                    "action": "scale_up",
                    "new_instance_type": self._next_instance_type(
                        metrics['current_type']
                    )
                }
        
        if metrics['memory_usage'] > 85:
            return {
                "action": "scale_up",
                "reason": "Memory-bound workloads often benefit from vertical scaling"
            }
        
        return {"action": "no_scaling_needed"}
```

---

## Kubernetes Scaling Example

```yaml
# Horizontal Pod Autoscaler (Horizontal Scaling)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  minReplicas: 3
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
---
# Vertical Pod Autoscaler (Vertical Scaling)
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: cache-server-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cache-server
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: cache
      minAllowed:
        memory: "1Gi"
        cpu: "500m"
      maxAllowed:
        memory: "32Gi"
        cpu: "8"
      controlledResources: ["memory"]
```

---

## Decision Framework

```
                    Start
                      │
                      ▼
            ┌─────────────────┐
            │ Is the service  │
            │   stateless?    │
            └────────┬────────┘
                     │
          ┌──────────┴──────────┐
          │ Yes                 │ No
          ▼                     ▼
   ┌──────────────┐     ┌──────────────────┐
   │   Scale      │     │ Can state be     │
   │ Horizontally │     │ externalized?    │
   └──────────────┘     └────────┬─────────┘
                                 │
                      ┌──────────┴──────────┐
                      │ Yes                 │ No
                      ▼                     ▼
              ┌──────────────┐      ┌──────────────┐
              │ Externalize  │      │    Scale     │
              │ state, then  │      │  Vertically  │
              │   scale out  │      │    first     │
              └──────────────┘      └──────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │ Hit hardware │
                                    │   limits?    │
                                    └──────┬───────┘
                                           │
                                    ┌──────┴──────┐
                                    │ Yes         │ No
                                    ▼             ▼
                             ┌────────────┐  ┌──────────┐
                             │ Redesign   │  │ Continue │
                             │ for        │  │ vertical │
                             │ horizontal │  └──────────┘
                             └────────────┘
```

---

## Key Takeaways

1. **Start simple with vertical scaling**: For new systems, vertical scaling is simpler and often sufficient initially

2. **Plan for horizontal scaling**: Design stateless services from the start to enable easy horizontal scaling later

3. **Externalize state**: Move session data, caches, and shared state to external stores (Redis, databases) to enable horizontal scaling

4. **Use both strategies**: Production systems typically use vertical scaling for databases/caches and horizontal scaling for application servers

5. **Consider operational complexity**: Horizontal scaling adds complexity—load balancing, service discovery, distributed coordination

6. **Monitor cost curves**: Vertical scaling costs grow exponentially; switch to horizontal when the math stops working

7. **Fault tolerance**: Horizontal scaling provides natural redundancy; vertical scaling creates single points of failure
