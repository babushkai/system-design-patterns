# ロードバランシング

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../06-scaling/01-load-balancing.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

ロードバランシングは、受信トラフィックを複数のサーバーに分散し、単一のサーバーが過負荷にならないようにすることで、可用性、信頼性、応答時間を向上させます。一般的なアルゴリズムには、ラウンドロビン、最小コネクション、重み付き分散、コンシステントハッシュがあります。

---

## なぜロードバランシングが必要なのか？

ロードバランシングなしの場合:

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

ロードバランシングありの場合:

```mermaid
graph LR
    Traffic[All Traffic] --> LB[Load Balancer]
    LB --> S1["Server 1<br/>CPU: 50%"]
    LB --> S2["Server 2<br/>CPU: 50%"]
    LB --> S3["Server 3<br/>CPU: 50%"]
```

---

## ロードバランサーの種類

### レイヤー4（トランスポート層）

```mermaid
sequenceDiagram
    participant Client
    participant LB as L4 Load Balancer
    participant Backend

    Note over LB: Routes by IP + TCP/UDP port<br/>No payload parsing (fast)<br/>Maintains TCP connection

    Client->>LB: TCP SYN
    LB->>Backend: TCP SYN
    Backend-->>LB: SYN-ACK
    LB-->>Client: SYN-ACK
    Client->>LB: ACK
    LB->>Backend: ACK
    Client->>LB: Data
    LB->>Backend: Data
```

L4バランシングは*パケット*の仕事です — 単位あたりの処理はナノ秒で測られます。だからこそ本番のL4データプレーン(GoogleのMaglev、MetaのKatran)は通常のソケットパスではなくカーネルバイパスやXDPの上に構築されます。パケット対リクエストの区別とカーネルパスの機構は[ネットワークトランスポートの内部構造](./14-network-transport-internals.md)にあります。

### レイヤー7（アプリケーション層）

```mermaid
graph TD
    LB["L7 Load Balancer<br/>Routes by HTTP headers, URL, cookies<br/>SSL termination · Content-based routing"]
    LB -->|"/api/*"| API["API Server"]
    LB -->|"/static/*"| CDN["CDN Origin"]
    LB -->|"/images/*"| IMG["Image Svc"]
```

---

## ロードバランシングアルゴリズム

### 1. ラウンドロビン

```nginx
# Round Robin — nginx default when no algorithm directive is specified.
# Requests cycle through the list in order: server1 → server2 → server3 → server1 …
upstream round_robin_backend {
    server server1.example.com:8080;   # receives request 1, 4, 7 …
    server server2.example.com:8080;   # receives request 2, 5, 8 …
    server server3.example.com:8080;   # receives request 3, 6, 9 …
}

server {
    listen 80;

    location / {
        proxy_pass http://round_robin_backend;
    }
}
```

```
Request 1 ──► Server 1
Request 2 ──► Server 2
Request 3 ──► Server 3
Request 4 ──► Server 1  ← cycles back
Request 5 ──► Server 2
```

### 2. 重み付きラウンドロビン

```nginx
# Weighted Round Robin — higher-weight servers receive proportionally more requests.
# With weights 3:1:2, server1 gets 50%, server2 ~17%, server3 ~33% of traffic.
upstream weighted_backend {
    server server1.example.com:8080 weight=3;   # 50% of traffic
    server server2.example.com:8080 weight=1;   # ~17% of traffic
    server server3.example.com:8080 weight=2;   # ~33% of traffic
}

server {
    listen 80;

    location / {
        proxy_pass http://weighted_backend;
    }
}
```

### 3. 最小コネクション

```nginx
# Least Connections — each new request goes to the server with the fewest
# active connections, adapting naturally to varying request durations.
#
# Visualization:
#   Server 1: [████████░░] 8 connections
#   Server 2: [██░░░░░░░░] 2 connections  ← next request goes here
#   Server 3: [█████░░░░░] 5 connections
upstream least_conn_backend {
    least_conn;

    server server1.example.com:8080;
    server server2.example.com:8080;
    server server3.example.com:8080;
}

server {
    listen 80;

    location / {
        proxy_pass http://least_conn_backend;
    }
}
```

### 4. 重み付き最小コネクション

```nginx
# Weighted Least Connections — combines least_conn with weight so that
# higher-capacity servers absorb proportionally more connections.
#
# Effective score = active_connections / weight  (lower wins)
# Example:
#   server1 (weight 3): 6 conns → score = 6/3 = 2.0
#   server2 (weight 1): 1 conn  → score = 1/1 = 1.0  ← next request goes here
upstream weighted_least_conn_backend {
    least_conn;

    server server1.example.com:8080 weight=3;
    server server2.example.com:8080 weight=1;
}

server {
    listen 80;

    location / {
        proxy_pass http://weighted_least_conn_backend;
    }
}
```

### 5. IPハッシュ（セッション永続化）

```nginx
# IP Hash — the client's IP address determines which server receives the
# request, giving sticky-session behaviour without cookies.
# Same IP always routes to the same server:
#   192.168.1.100 → always server2
#   192.168.1.101 → always server1
upstream ip_hash_backend {
    ip_hash;

    server server1.example.com:8080;
    server server2.example.com:8080;
    server server3.example.com:8080;
}

server {
    listen 80;

    location / {
        proxy_pass http://ip_hash_backend;
    }
}
```

### 6. コンシステントハッシュ

```
Consistent Hashing — Pseudocode

INIT(servers, replicas_per_server = 100):
    ring   ← empty sorted list        # positions on the hash ring (0 … 2^128)
    map    ← empty hash map           # ring position → server

    for each server in servers:
        ADD_SERVER(server)

ADD_SERVER(server):
    for i in 0 .. replicas_per_server:
        pos ← HASH(server + ":" + i)  # e.g. MD5, SHA-256
        INSERT pos into ring (keep sorted)
        map[pos] ← server

REMOVE_SERVER(server):
    for i in 0 .. replicas_per_server:
        pos ← HASH(server + ":" + i)
        DELETE pos from ring
        DELETE map[pos]

GET_SERVER(request_key):
    if ring is empty: return NULL

    pos ← HASH(request_key)
    idx ← first index in ring where ring[idx] >= pos   # binary search
    if idx == length(ring):
        idx ← 0                                        # wrap around
    return map[ring[idx]]

---
Why this matters:
  • Adding/removing a server remaps only ~1/N of keys (N = server count).
  • Virtual replicas (typically 100-200 per server) smooth out distribution.
  • Cannot be expressed as a single nginx directive — real implementations
    live in application code or specialised proxies (e.g. Envoy, Maglev).
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

## ヘルスチェック

```nginx
# Nginx — passive health checks (open-source) + active checks (nginx Plus)
upstream healthcheck_backend {
    server server1.example.com:8080 max_fails=3 fail_timeout=30s;
    server server2.example.com:8080 max_fails=3 fail_timeout=30s;
    server server3.example.com:8080 max_fails=3 fail_timeout=30s;
    # max_fails  — consecutive failures before marking the server as down
    # fail_timeout — how long the server stays marked down, and the window
    #                in which max_fails failures must occur
}

server {
    listen 80;

    location / {
        proxy_pass              http://healthcheck_backend;
        proxy_connect_timeout   5s;
        proxy_read_timeout      10s;
        proxy_next_upstream     error timeout http_502 http_503;
        #                       ↑ on failure, retry the next server automatically
    }

    # Self health endpoint (for upstream LBs or orchestrators to probe)
    location = /health {
        access_log off;
        return 200 "healthy\n";
    }
}
```

```haproxy
# HAProxy — active health checks with thresholds
backend app_servers
    balance roundrobin

    option httpchk GET /health          # active probe endpoint
    http-check expect status 200

    default-server inter 10s            # check every 10 s
                    fall  3             # 3 failures → mark DOWN
                    rise  2             # 2 successes → mark UP
                    timeout check 5s    # per-check timeout

    server srv1 192.168.1.10:8080 check
    server srv2 192.168.1.11:8080 check
    server srv3 192.168.1.12:8080 check
```

```mermaid
sequenceDiagram
    participant LB as Load Balancer
    participant S as Server

    loop Every 10 seconds
        LB->>S: GET /health
        S-->>LB: 200 OK
    end

    Note over LB: Health Status Table<br/>server1: HEALTHY<br/>server2: HEALTHY<br/>server3: UNHEALTHY
```

---

## セッション永続化（スティッキーセッション）

```nginx
# Sticky Sessions — nginx uses a cookie to pin a client to the same backend.
# Option A: ip_hash (no cookie, based on client IP)
upstream sticky_ip {
    ip_hash;

    server server1.example.com:8080;
    server server2.example.com:8080;
    server server3.example.com:8080;
}

# Option B: sticky cookie (nginx Plus — explicit cookie-based affinity)
# upstream sticky_cookie {
#     sticky cookie SERVERID expires=1h path=/;
#
#     server server1.example.com:8080;
#     server server2.example.com:8080;
#     server server3.example.com:8080;
# }

server {
    listen 80;

    location / {
        proxy_pass http://sticky_ip;
    }
}
```

```haproxy
# HAProxy — cookie-based sticky sessions (open-source)
backend sticky_servers
    balance roundrobin

    # Insert a SERVERID cookie; the client sends it back on subsequent requests
    # so HAProxy routes to the same backend.
    cookie SERVERID insert indirect nocache maxlife 1h

    server srv1 192.168.1.10:8080 check cookie srv1
    server srv2 192.168.1.11:8080 check cookie srv2
    server srv3 192.168.1.12:8080 check cookie srv3
```

```mermaid
sequenceDiagram
    participant C as Client
    participant LB as Load Balancer
    participant S2 as Server2

    Note over C,S2: Request 1 (no cookie)
    C->>LB: Request
    LB->>S2: Forward (assigned)
    S2-->>LB: Response
    LB-->>C: Set-Cookie: SERVERID=srv2

    Note over C,S2: Request 2 (with cookie)
    C->>LB: Cookie: SERVERID=srv2
    LB->>S2: Forward (same server)
    S2-->>LB: Response
    LB-->>C: Response
```

---

## ロードバランサーアーキテクチャ

### アクティブ-パッシブ（フェイルオーバー）

```mermaid
graph TD
    Traffic[All Traffic] --> Active["Active LB<br/>(Primary)"]
    Active ---|Heartbeat| Passive["Passive LB<br/>(Standby)"]

    style Passive stroke-dasharray: 5 5
```

障害発生時:

```mermaid
graph TD
    Active["Active LB<br/>(Primary) FAILED"] ~~~ Passive
    Traffic[All Traffic] --> Passive["Passive LB<br/>(Now Active)<br/>VIP moves"]

    style Active stroke:red,stroke-dasharray: 5 5
```

### アクティブ-アクティブ

```mermaid
graph TD
    DNS[DNS Round Robin] --> LB1["LB 1<br/>(Active)"]
    DNS --> LB2["LB 2<br/>(Active)"]
    LB1 --> S1[Server 1]
    LB1 --> S2[Server 2]
    LB1 --> S3[Server 3]
    LB2 --> S1
    LB2 --> S2
    LB2 --> S3
```

---

## グローバルサーバーロードバランシング（GSLB）

```mermaid
graph TD
    User[User Request] --> DNS["DNS Server<br/>(GSLB)"]
    DNS --> US["US-East Data Center<br/>LB"]
    DNS --> EU["EU-West Data Center<br/>LB"]
    DNS --> AP["Asia-Pac Data Center<br/>LB"]

    Note["Routing: geographic proximity,<br/>data center health, current load,<br/>network latency"]

    style Note fill:none,stroke-dasharray: 5 5
```

---

## nginx ロードバランサー設定

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

## HAProxy 設定

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

## アルゴリズム比較

| アルゴリズム | 適した用途 | メリット | デメリット |
|-----------|----------|------|------|
| ラウンドロビン | 均一なサーバー | シンプル、公平な分散 | サーバー容量を無視 |
| 重み付きラウンドロビン | 混合容量 | サーバー性能を考慮 | 静的な重み |
| 最小コネクション | リクエスト時間が変動 | 負荷に適応 | オーバーヘッドが大きい |
| IPハッシュ | セッション永続化 | 外部セッションストア不要 | 偏った分散 |
| コンシステントハッシュ | キャッシュサーバー | 最小限の再分散 | 実装が複雑 |
| ランダム | シンプルなシナリオ | 状態管理不要 | 偏る可能性 |

---

## 重要なポイント

1. **レイヤー4 vs レイヤー7**: レイヤー4はより高速ですが柔軟性に劣ります。レイヤー7はコンテンツベースルーティングとSSLターミネーションを可能にします

2. **アルゴリズムの選択が重要**: 均一なワークロードにはラウンドロビン、可変リクエスト時間には最小コネクション、キャッシュにはコンシステントハッシュを使用します

3. **ヘルスチェックは不可欠**: フラッピングを避けるため、適切な閾値でロバストなヘルスチェックを実装してください

4. **セッション永続化のトレードオフ**: スティッキーセッションはステートフルアプリを簡素化しますが、負荷の偏りを引き起こす可能性があります

5. **高可用性**: 単一障害点を排除するため、アクティブ-パッシブまたはアクティブ-アクティブ構成を使用してください

6. **すべてを監視**: コネクション数、応答時間、エラー率、サーバーヘルスメトリクスを追跡してください
