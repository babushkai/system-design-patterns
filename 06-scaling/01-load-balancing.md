# Load Balancing

## TL;DR

Load balancing distributes incoming traffic across multiple servers to ensure no single server becomes overwhelmed, improving availability, reliability, and response times. Common algorithms include round-robin, least connections, weighted distribution, and consistent hashing.

---

## Why Load Balancing?

Without load balancing:

```
                    ┌─────────────────┐
                    │    Server 1     │
                    │   (overloaded)  │
All Traffic ───────►│   CPU: 100%     │
                    │   Memory: 95%   │
                    └─────────────────┘
                    
                    ┌─────────────────┐
                    │    Server 2     │
                    │     (idle)      │
                    │   CPU: 5%       │
                    └─────────────────┘
```

With load balancing:

```
                         ┌─────────────────┐
                    ┌───►│    Server 1     │
                    │    │   CPU: 50%      │
┌──────────────┐    │    └─────────────────┘
│    Load      │────┤
│   Balancer   │    │    ┌─────────────────┐
└──────────────┘    ├───►│    Server 2     │
        ▲           │    │   CPU: 50%      │
        │           │    └─────────────────┘
   All Traffic      │
                    │    ┌─────────────────┐
                    └───►│    Server 3     │
                         │   CPU: 50%      │
                         └─────────────────┘
```

---

## Load Balancer Types

### Layer 4 (Transport Layer)

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 4 Load Balancer                     │
│                                                              │
│  • Routes based on IP address and TCP/UDP port              │
│  • Cannot inspect packet contents                            │
│  • Very fast (no payload parsing)                            │
│  • Maintains TCP connection to backend                       │
└─────────────────────────────────────────────────────────────┘

Client ──TCP SYN──► LB ──TCP SYN──► Backend
       ◄──SYN-ACK──    ◄──SYN-ACK──
       ──ACK──────►    ──ACK──────►
       ──Data─────►    ──Data─────►
```

### Layer 7 (Application Layer)

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 7 Load Balancer                     │
│                                                              │
│  • Routes based on HTTP headers, URL, cookies               │
│  • Can modify requests/responses                             │
│  • SSL termination                                           │
│  • Content-based routing                                     │
└─────────────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │     L7 LB    │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  /api/*     │   │  /static/*  │   │  /images/*  │
│  API Server │   │  CDN Origin │   │  Image Svc  │
└─────────────┘   └─────────────┘   └─────────────┘
```

---

## Load Balancing Algorithms

### 1. Round Robin

```python
class RoundRobinBalancer:
    def __init__(self, servers: list[str]):
        self.servers = servers
        self.current = 0
    
    def get_server(self) -> str:
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server

# Usage
balancer = RoundRobinBalancer(["server1", "server2", "server3"])
# Request 1 → server1
# Request 2 → server2
# Request 3 → server3
# Request 4 → server1 (wraps around)
```

```
Request 1 ──► Server 1
Request 2 ──► Server 2
Request 3 ──► Server 3
Request 4 ──► Server 1  ← cycles back
Request 5 ──► Server 2
```

### 2. Weighted Round Robin

```python
class WeightedRoundRobinBalancer:
    def __init__(self, servers: dict[str, int]):
        # servers = {"server1": 3, "server2": 1, "server3": 2}
        self.servers = []
        for server, weight in servers.items():
            self.servers.extend([server] * weight)
        self.current = 0
    
    def get_server(self) -> str:
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server

# With weights {server1: 3, server2: 1, server3: 2}
# Expanded: [s1, s1, s1, s2, s3, s3]
# server1 gets 50% of traffic
# server2 gets ~17% of traffic
# server3 gets ~33% of traffic
```

### 3. Least Connections

```python
import threading
from dataclasses import dataclass

@dataclass
class Server:
    address: str
    active_connections: int = 0
    lock: threading.Lock = None
    
    def __post_init__(self):
        self.lock = threading.Lock()

class LeastConnectionsBalancer:
    def __init__(self, servers: list[str]):
        self.servers = [Server(addr) for addr in servers]
        self.lock = threading.Lock()
    
    def get_server(self) -> Server:
        with self.lock:
            # Find server with fewest connections
            server = min(self.servers, key=lambda s: s.active_connections)
            with server.lock:
                server.active_connections += 1
            return server
    
    def release_server(self, server: Server):
        with server.lock:
            server.active_connections -= 1

# Visualization
# Server 1: [████████░░] 8 connections
# Server 2: [██░░░░░░░░] 2 connections  ← next request goes here
# Server 3: [█████░░░░░] 5 connections
```

### 4. Weighted Least Connections

```python
class WeightedLeastConnectionsBalancer:
    def __init__(self, servers: dict[str, int]):
        # servers = {"server1": 3, "server2": 1}
        self.servers = {
            addr: {"weight": weight, "connections": 0}
            for addr, weight in servers.items()
        }
    
    def get_server(self) -> str:
        # Score = connections / weight (lower is better)
        best_server = min(
            self.servers.items(),
            key=lambda x: x[1]["connections"] / x[1]["weight"]
        )
        self.servers[best_server[0]]["connections"] += 1
        return best_server[0]

# Example:
# server1 (weight 3): 6 connections → score = 6/3 = 2.0
# server2 (weight 1): 1 connection  → score = 1/1 = 1.0 ← winner
```

### 5. IP Hash (Session Persistence)

```python
import hashlib

class IPHashBalancer:
    def __init__(self, servers: list[str]):
        self.servers = servers
    
    def get_server(self, client_ip: str) -> str:
        # Hash the IP to get consistent server selection
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        index = hash_value % len(self.servers)
        return self.servers[index]

# Same IP always routes to same server (sticky sessions)
# 192.168.1.100 → always server2
# 192.168.1.101 → always server1
```

### 6. Consistent Hashing

```python
import hashlib
from bisect import bisect_left

class ConsistentHashBalancer:
    def __init__(self, servers: list[str], replicas: int = 100):
        self.replicas = replicas
        self.ring = []
        self.server_map = {}
        
        for server in servers:
            self.add_server(server)
    
    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_server(self, server: str):
        for i in range(self.replicas):
            hash_key = self._hash(f"{server}:{i}")
            self.ring.append(hash_key)
            self.server_map[hash_key] = server
        self.ring.sort()
    
    def remove_server(self, server: str):
        for i in range(self.replicas):
            hash_key = self._hash(f"{server}:{i}")
            self.ring.remove(hash_key)
            del self.server_map[hash_key]
    
    def get_server(self, key: str) -> str:
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        idx = bisect_left(self.ring, hash_key)
        
        if idx == len(self.ring):
            idx = 0
        
        return self.server_map[self.ring[idx]]
```

```
Consistent Hash Ring:
                    0
                    │
           ┌────────┴────────┐
          S3                 S1
         /                    \
        /                      \
      270 ──────────────────── 90
        \                      /
         \                    /
          S2                 S1
           └────────┬────────┘
                    │
                   180

Key "user:123" hashes to position 45 → routes to S1
Key "user:456" hashes to position 200 → routes to S2

When S2 is removed:
- Only keys that were on S2 need to move
- Keys on S1 and S3 stay where they are
```

---

## Health Checks

```python
import asyncio
import aiohttp
from dataclasses import dataclass
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    endpoint: str
    interval_seconds: int = 10
    timeout_seconds: int = 5
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3

class HealthChecker:
    def __init__(self, servers: list[str], health_check: HealthCheck):
        self.servers = {s: HealthStatus.UNKNOWN for s in servers}
        self.check_counts = {s: {"healthy": 0, "unhealthy": 0} for s in servers}
        self.health_check = health_check
    
    async def check_server(self, server: str) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://{server}{self.health_check.endpoint}"
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(
                        total=self.health_check.timeout_seconds
                    )
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def update_health(self, server: str):
        is_healthy = await self.check_server(server)
        
        if is_healthy:
            self.check_counts[server]["healthy"] += 1
            self.check_counts[server]["unhealthy"] = 0
            
            if self.check_counts[server]["healthy"] >= \
               self.health_check.healthy_threshold:
                self.servers[server] = HealthStatus.HEALTHY
        else:
            self.check_counts[server]["unhealthy"] += 1
            self.check_counts[server]["healthy"] = 0
            
            if self.check_counts[server]["unhealthy"] >= \
               self.health_check.unhealthy_threshold:
                self.servers[server] = HealthStatus.UNHEALTHY
    
    def get_healthy_servers(self) -> list[str]:
        return [s for s, status in self.servers.items() 
                if status == HealthStatus.HEALTHY]
```

```
Health Check Flow:
                                    
  ┌──────────────┐     GET /health    ┌─────────────┐
  │     Load     │ ────────────────── │   Server    │
  │   Balancer   │     200 OK         │             │
  └──────────────┘ ◄────────────────  └─────────────┘
         │
         │ Every 10 seconds
         │
         ▼
  ┌──────────────────────────────────┐
  │  Health Status Table             │
  │  ┌──────────┬──────────────────┐ │
  │  │ Server   │ Status           │ │
  │  ├──────────┼──────────────────┤ │
  │  │ server1  │ ● HEALTHY        │ │
  │  │ server2  │ ● HEALTHY        │ │
  │  │ server3  │ ○ UNHEALTHY      │ │
  │  └──────────┴──────────────────┘ │
  └──────────────────────────────────┘
```

---

## Session Persistence (Sticky Sessions)

```python
import time
from dataclasses import dataclass

@dataclass
class Session:
    server: str
    created_at: float
    last_accessed: float

class StickySessionBalancer:
    def __init__(self, servers: list[str], session_ttl: int = 3600):
        self.servers = servers
        self.sessions = {}  # session_id -> Session
        self.session_ttl = session_ttl
        self.round_robin = RoundRobinBalancer(servers)
    
    def get_server(self, session_id: str = None) -> tuple[str, str]:
        now = time.time()
        
        # Check for existing session
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Check if session is still valid
            if now - session.last_accessed < self.session_ttl:
                session.last_accessed = now
                return session.server, session_id
            else:
                # Session expired
                del self.sessions[session_id]
        
        # Create new session
        server = self.round_robin.get_server()
        new_session_id = self._generate_session_id()
        self.sessions[new_session_id] = Session(
            server=server,
            created_at=now,
            last_accessed=now
        )
        
        return server, new_session_id
    
    def _generate_session_id(self) -> str:
        import uuid
        return str(uuid.uuid4())
```

```
Session Persistence Flow:

Request 1 (no cookie):
  Client ──► LB ──► Server2 (assigned)
         ◄── Set-Cookie: SERVERID=srv2

Request 2 (with cookie):
  Client ──► LB ──► Server2 (same server)
  Cookie: SERVERID=srv2
```

---

## Load Balancer Architectures

### Active-Passive (Failover)

```
                    ┌─────────────────┐
                    │   Active LB     │◄─── All Traffic
                    │   (Primary)     │
                    └────────┬────────┘
                             │
                    Heartbeat│
                             │
                    ┌────────▼────────┐
                    │  Passive LB     │
                    │  (Standby)      │
                    └─────────────────┘

On failure:
                    ┌─────────────────┐
                    │   Active LB     │ ╳ FAILED
                    │   (Primary)     │
                    └─────────────────┘
                             
                    ┌─────────────────┐
                    │  Passive LB     │◄─── All Traffic
                    │  (Now Active)   │     (VIP moves)
                    └─────────────────┘
```

### Active-Active

```
                         DNS Round Robin
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
     ┌─────────────────┐             ┌─────────────────┐
     │    LB 1         │             │    LB 2         │
     │  (Active)       │             │  (Active)       │
     └────────┬────────┘             └────────┬────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
         ┌────────┐      ┌────────┐      ┌────────┐
         │Server 1│      │Server 2│      │Server 3│
         └────────┘      └────────┘      └────────┘
```

---

## Global Server Load Balancing (GSLB)

```
                         User Request
                              │
                              ▼
                    ┌─────────────────┐
                    │   DNS Server    │
                    │   (GSLB)        │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  US-East     │ │  EU-West     │ │  Asia-Pac    │
    │  Data Center │ │  Data Center │ │  Data Center │
    │              │ │              │ │              │
    │  ┌────────┐  │ │  ┌────────┐  │ │  ┌────────┐  │
    │  │   LB   │  │ │  │   LB   │  │ │  │   LB   │  │
    │  └────────┘  │ │  └────────┘  │ │  └────────┘  │
    └──────────────┘ └──────────────┘ └──────────────┘

GSLB Routing Decisions:
- Geographic proximity
- Data center health
- Current load
- Network latency
```

---

## NGINX Load Balancer Configuration

```nginx
# Layer 7 Load Balancing
upstream backend {
    # Least connections algorithm
    least_conn;
    
    # Server definitions with weights
    server backend1.example.com:8080 weight=3;
    server backend2.example.com:8080 weight=2;
    server backend3.example.com:8080 weight=1;
    
    # Backup server (only used when others are down)
    server backup.example.com:8080 backup;
    
    # Health check parameters
    server backend4.example.com:8080 max_fails=3 fail_timeout=30s;
    
    # Keep connections alive to backends
    keepalive 32;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
    }
}

# IP Hash for session persistence
upstream sticky_backend {
    ip_hash;
    server backend1.example.com:8080;
    server backend2.example.com:8080;
    server backend3.example.com:8080;
}

# Content-based routing
server {
    listen 80;
    
    location /api/ {
        proxy_pass http://api_servers;
    }
    
    location /static/ {
        proxy_pass http://static_servers;
    }
    
    location /websocket {
        proxy_pass http://ws_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## HAProxy Configuration

```haproxy
global
    maxconn 50000
    log stdout format raw local0

defaults
    mode http
    timeout connect 5s
    timeout client 50s
    timeout server 50s
    option httplog
    option dontlognull

frontend http_front
    bind *:80
    
    # ACLs for content-based routing
    acl is_api path_beg /api
    acl is_static path_beg /static
    
    # Route based on ACLs
    use_backend api_servers if is_api
    use_backend static_servers if is_static
    default_backend web_servers

backend web_servers
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    
    server web1 192.168.1.10:8080 check weight 3
    server web2 192.168.1.11:8080 check weight 2
    server web3 192.168.1.12:8080 check weight 1

backend api_servers
    balance leastconn
    option httpchk GET /api/health
    
    # Sticky sessions using cookie
    cookie SERVERID insert indirect nocache
    
    server api1 192.168.1.20:8080 check cookie api1
    server api2 192.168.1.21:8080 check cookie api2

backend static_servers
    balance uri
    hash-type consistent
    
    server static1 192.168.1.30:8080 check
    server static2 192.168.1.31:8080 check

# Statistics page
listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 10s
```

---

## AWS Application Load Balancer (ALB)

```python
import boto3

def create_alb():
    elbv2 = boto3.client('elbv2')
    
    # Create load balancer
    alb = elbv2.create_load_balancer(
        Name='my-application-lb',
        Subnets=['subnet-12345', 'subnet-67890'],
        SecurityGroups=['sg-12345'],
        Scheme='internet-facing',
        Type='application',
        IpAddressType='ipv4'
    )
    
    alb_arn = alb['LoadBalancers'][0]['LoadBalancerArn']
    
    # Create target group
    target_group = elbv2.create_target_group(
        Name='my-targets',
        Protocol='HTTP',
        Port=80,
        VpcId='vpc-12345',
        HealthCheckProtocol='HTTP',
        HealthCheckPath='/health',
        HealthCheckIntervalSeconds=30,
        HealthyThresholdCount=2,
        UnhealthyThresholdCount=3,
        TargetType='instance'
    )
    
    tg_arn = target_group['TargetGroups'][0]['TargetGroupArn']
    
    # Register targets
    elbv2.register_targets(
        TargetGroupArn=tg_arn,
        Targets=[
            {'Id': 'i-1234567890abcdef0', 'Port': 80},
            {'Id': 'i-0987654321fedcba0', 'Port': 80}
        ]
    )
    
    # Create listener with rules
    elbv2.create_listener(
        LoadBalancerArn=alb_arn,
        Protocol='HTTPS',
        Port=443,
        Certificates=[
            {'CertificateArn': 'arn:aws:acm:...'}
        ],
        DefaultActions=[
            {'Type': 'forward', 'TargetGroupArn': tg_arn}
        ]
    )
    
    return alb_arn
```

---

## Algorithm Comparison

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| Round Robin | Uniform servers | Simple, fair distribution | Ignores server capacity |
| Weighted RR | Mixed capacity | Accounts for server power | Static weights |
| Least Connections | Varying request duration | Adapts to load | More overhead |
| IP Hash | Session persistence | No external session store | Uneven distribution |
| Consistent Hash | Cache servers | Minimal redistribution | Complex implementation |
| Random | Simple scenarios | No state needed | Potentially uneven |

---

## Key Takeaways

1. **Layer 4 vs Layer 7**: Layer 4 is faster but less flexible; Layer 7 enables content-based routing and SSL termination

2. **Algorithm choice matters**: Round-robin for uniform workloads, least-connections for variable request times, consistent hashing for caches

3. **Health checks are critical**: Implement robust health checks with appropriate thresholds to avoid flapping

4. **Session persistence trade-offs**: Sticky sessions simplify stateful apps but can cause uneven load distribution

5. **High availability**: Use active-passive or active-active configurations to eliminate single points of failure

6. **Monitor everything**: Track connection counts, response times, error rates, and server health metrics
