# リーダー選出

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../02-distributed-databases/09-leader-election.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

リーダー選出は、分散システムでアクションを調整するために1つのノードを指定します。強い保証にはコンセンサスベースの選出（Raft/Paxos）を使用し、それほど重要でないシステムにはシンプルなアプローチ（バリーアルゴリズム、リースベース）を使用します。主要な課題は、任意の時点で正確に1つのリーダーが存在することを保証することです。スプリットブレインが敵です。フェンシングトークンは古いリーダーによるダメージを防ぎます。

---

## なぜリーダーを選出するのか？

### 調整のシンプル化

```
Without leader:
  All nodes coordinate → O(n²) messages
  Conflicts possible → complex resolution

With leader:
  Leader coordinates → O(n) messages
  Single decision maker → no conflicts
```

### ユースケース

```
Database:     Leader accepts writes
Queue:        Leader assigns partitions
Cache:        Leader manages keys
Scheduler:    Leader distributes tasks
Lock service: Leader grants locks
```

---

## 選出アルゴリズム

### バリーアルゴリズム

最高ランクのノードが勝ちます。

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

**メリット：**
- 実装がシンプル
- 決定論的な勝者

**デメリット：**
- 信頼性のある障害検出を前提としている
- 常に最高ランクが勝つ（柔軟性がない）
- パーティション耐性がない

### リングアルゴリズム

リングに沿ったトークンベースの選出です。

```
Nodes form logical ring: A → B → C → D → A

B detects leader failed:
  1. B sends ELECTION(B) to C
  2. C adds self, sends ELECTION(B,C) to D
  3. D adds self, sends ELECTION(B,C,D) to A
  4. A adds self, sends ELECTION(B,C,D,A) to B
  5. B sees complete ring, picks highest, broadcasts COORDINATOR
```

**メリット：**
- 単一障害点がない
- すべてのノードが参加

**デメリット：**
- 遅い（O(n)メッセージ）
- リングを維持する必要がある
- 選出中の障害に敏感

### コンセンサスベースの選出

リーダー選出にRaft/Paxosを使用します。

```
Election is just agreeing on a value:
  "Who is the leader for term T?"

Raft:
  1. Candidate increments term
  2. Requests votes
  3. Majority grants → becomes leader
  4. Leader sends heartbeats to maintain authority
```

**メリット：**
- パーティション耐性あり
- 強い保証
- よく理解されている

**デメリット：**
- 過半数が必要（f個の障害に対して2f+1）
- 実装がより複雑

---

## リースベースのリーダーシップ

### コンセプト

リーダーが時間制限付きのリースを保持します。

```
Leader A acquires lease: valid until T+10s
Other nodes know: "A is leader until T+10s"

Before lease expires:
  A renews lease → continues as leader

If A crashes:
  Lease expires (T+10s)
  Others can acquire new lease
```

### 実装

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

### クロックの考慮事項

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

## フェンシングトークン

### 問題

古いリーダーがもはやリーダーではないことを知りません。

```
Timeline:
  T=0:   Leader A acquires lease
  T=5:   A enters GC pause
  T=10:  Lease expires, B becomes leader
  T=15:  A wakes up, thinks it's still leader
  T=16:  A writes data (stale leader!)

Split-brain: Both A and B think they're leader
```

### 解決策：フェンシングトークン

各リースで単調増加するトークンです。

```
Lease 1: token=100, holder=A
Lease 2: token=101, holder=B

Storage checks token on write:
  A attempts write with token=100
  Storage: "Current token is 101, rejecting 100"

Stale leader's writes rejected
```

### 実装

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

## 実践でのリーダー選出

### etcdの使用

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

### ZooKeeperの使用

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

### Redis（Redlock）の使用

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

## スプリットブレインの対処

### 検出

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

### 防止

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

### リカバリ

```
If split-brain detected:
  1. Stop all leaders
  2. Compare states
  3. Reconcile conflicts
  4. Re-elect single leader
  5. Resume operations
```

---

## リーダーヘルスモニタリング

### ハートビート

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

### クォーラムベースの生存確認

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

### アプリケーションレベルのヘルスチェック

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

## グレースフルリーダーシップ移管

### 計画的ハンドオフ

```
For maintenance, upgrades, rebalancing:

1. Current leader L1 prepares successor L2
2. L1 ensures L2's log is up-to-date
3. L1 sends TimeoutNow to L2 (start election immediately)
4. L2 wins election (most up-to-date)
5. L1 steps down

No availability gap
```

### Raftのリーダーシップ移管

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

## アンチパターン

### フェンシングなし

```
Bad:
  if am_i_leader():
      do_write()

Problem: Leader status might have changed mid-operation

Good:
  token = get_fencing_token()
  do_write(token)  # Storage validates token
```

### クロック依存ロジック

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

### ネットワークパーティションの無視

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

## アプローチの比較

| アプローチ | 一貫性 | 可用性 | 複雑さ |
|-----------|--------|--------|--------|
| バリー | 弱い | 低い | 低い |
| リング | 弱い | 中 | 低い |
| コンセンサス（Raft） | 強い | 中 | 高い |
| リース（etcd） | 強い | 中 | 中 |
| リース（Redis） | 中 | 高い | 中 |

---

## リースベースのリーダーシップ：詳細

### リースのエンドツーエンドの動作

リースは、ロックサービスがノードに発行する時間制限付きの許可です。保持者はTTLが期限切れになるまでリースを所有します。明示的な解放は不要です。保持者がクラッシュした場合、リースは単純に期限切れになり、別のノードが取得します。

```
Lifecycle of a lease:

  T=0   Node A: Grant(TTL=10s) → lease_id=abc123, revision=42
  T=3   Node A: KeepAlive(abc123) → TTL reset to 10s from now
  T=6   Node A: KeepAlive(abc123) → TTL reset to 10s from now
  ...
  T=22  Node A crashes. No more KeepAlive.
  T=32  Lease expires. Key "/leader" deleted automatically.
  T=32  Node B (watching "/leader") detects deletion → acquires new lease.
```

### 制限された非可用ウィンドウ

```
Worst case unavailability = lease_ttl + election_time
  e.g., 10s + 200ms ≈ 10.2s

Compare with heartbeat-based detection:
  unavailability = missed_heartbeats × interval + election_time
  e.g., 3 × 2s + 5s = 11s (unbounded with unlucky timing)
```

### etcd Lease API

```go
// Grant: create a new lease with a TTL
resp, _ := client.Grant(ctx, 10) // 10 second TTL

// Attach the lease to a key (leader registration)
_, _ = client.Put(ctx, "/service/leader", "node-1", clientv3.WithLease(resp.ID))

// KeepAlive: auto-renew the lease in the background
// Returns a channel; lease is renewed every TTL/3 automatically
keepAliveCh, _ := client.KeepAlive(ctx, resp.ID)

// Revoke: explicitly release the lease (graceful shutdown)
_, _ = client.Revoke(ctx, resp.ID)
```

フォロワーはキーの削除（`/service/leader`上の`mvccpb.DELETE`イベント）を監視し、リーダーのキーが消えたときに新しいリースの取得を試みます。

### クロックスキューのリスク

```
Lock service clock:  10:00:00  ──────────────────►  10:00:10 (lease expires)
Leader clock (fast): 10:00:02  ──────────────────►  10:00:12

Leader thinks it has 2 more seconds of valid lease.
Lock service already expired at leader's 10:00:12 = service's 10:00:10.
Window of danger: leader acts on an expired lease.
```

**緩和策：保守的な更新頻度**

TTLの1/3で更新します。これにより、クロックドリフト、ネットワーク遅延、GCポーズに対してTTLの2/3のバッファが得られます。

```
TTL = 10s → renew every ~3.3s
TTL = 30s → renew every ~10s

If renewal fails:
  - First miss: 6.7s remaining → retry immediately
  - Second miss: 3.3s remaining → start stepping down
  - Third miss: 0s → must stop all leader activity
```

---

## スプリットブレインとフェンシング：詳細

### GCポーズのシナリオ

```
T=0    Leader A acquires lease (token=100), starts processing writes
T=5    Leader A enters full GC pause (stop-the-world, 30 seconds)
T=10   Lease expires on lock service
T=11   Leader B acquires lease (token=101), starts accepting writes
T=35   Leader A's GC finishes. Local state says "I'm leader."
       A writes to storage → DATA CORRUPTION if unfenced
```

これは理論的なものではありません。大きなヒープを持つ本番環境で10秒以上のJVM GCポーズが記録されています。

### 安全策としてのフェンシングトークン

各リース付与は単調増加するトークンを返します。ストレージは古いトークンの書き込みを拒否しなければなりません。

```
  Lock Service          Leader A           Leader B           Storage
       │                    │                  │                  │
       │──lease(tok=100)──►│                  │                  │
       │                    │──write(tok=100)─────────────────►  │ ✓ accepted
       │                    │    (GC pause)    │                  │
       │──lease(tok=101)──────────────────────►│                  │
       │                    │                  │──write(tok=101)─►│ ✓ accepted
       │                    │  (GC resumes)    │                  │
       │                    │──write(tok=100)─────────────────►  │ ✗ REJECTED
       │                    │                  │                  │   (100 < 101)
```

### ZooKeeperとetcdのフェンシング値

```
ZooKeeper: use zxid (transaction ID) as fencing token.
  Globally incremented on every write. Use zxid from session creation.

etcd: use revision (global monotonic counter) as fencing token.
  resp, _ := client.Grant(ctx, 10)
  fencingToken := resp.Header.Revision  // monotonic across all keys
```

### STONITH：フェンシングトークンが使用できない場合

ストレージがフェンシングトークンをサポートしていない場合（レガシーデータベース、サードパーティAPI）、新しいリーダーが行動する前に古いリーダーが停止していることを確認します。

```
STONITH (Shoot The Other Node In The Head):
  - Power-cycle via IPMI/BMC, terminate VM via hypervisor API
  - Revoke network access via SDN rules

Used by: Pacemaker/Corosync, VMware HA, AWS (terminate EC2 via API)

Tradeoff:
  + Guarantees old leader cannot act
  - Requires infrastructure-level access and adds operational complexity
```

---

## 実践でのリーダー選出：エコシステムパターン

### Kubernetesのリーダー選出

Kubernetesは、組み込みのリーダー選出に`coordination.k8s.io/v1` Leaseリソースを使用します。コントローラーとオペレーターはこれを使用して、1つのインスタンスのみがアクティブであることを保証します。

```yaml
apiVersion: coordination.k8s.io/v1
kind: Lease
metadata:
  name: my-controller-leader
  namespace: kube-system
spec:
  holderIdentity: "controller-pod-abc"
  leaseDurationSeconds: 15
  acquireTime: "2025-01-15T10:00:00Z"
  renewTime: "2025-01-15T10:00:10Z"
  leaseTransitions: 3
```

```
Sidecar pattern:
  Pod = leader election sidecar + main application container
  Sidecar renews lease → exposes /healthz as "leader"
  Main container checks /healthz before doing leader work
  Lease lost → sidecar unhealthy → main stops leader tasks
```

`client-go`ライブラリは、キャンペーン、更新、ステップダウンのための設定可能なコールバックを持つ`leaderelection.LeaderElector`を提供します。

### Redis RedLock：論争

RedlockはN個の独立したRedisインスタンス（通常5個）にまたがる分散ロックを試みます。過半数（N/2+1）が時間制限内に許可した場合にロックが取得されます。

```
Redlock algorithm:
  1. Get current time T1
  2. SET key with NX + TTL on all N Redis instances
  3. Lock acquired if majority (≥3/5) granted AND (T2-T1) < TTL
  4. Effective TTL = original TTL - acquisition time

Kleppmann critique: GC pauses break safety, no fencing tokens, assumes bounded drift
Antirez response: clock drift bounded in practice with NTP

Verdict:
  ✓ Fine for distributed locks (mutual exclusion for efficiency)
  ✗ Questionable for leader election requiring strong correctness
```

### クラウドネイティブアプローチ

```
AWS:
  DynamoDB conditional writes:
    PutItem with ConditionExpression:
      "attribute_not_exists(leader_id) OR lease_expiry < :now"
    Atomic, strongly consistent, no separate lock service needed.

  ElastiCache (Redis) + Lua script:
    Use SET NX EX for single-instance leader lock.
    Avoid Redlock unless you need cross-AZ redundancy.

GCP:
  Cloud Spanner: TrueTime-based leases (bounded clock uncertainty).
  Chubby (internal): lease-based lock service that inspired ZooKeeper.
```

---

## 選出プロトコルの比較

| プロパティ | Raft | ZooKeeper ZNodes | etcd Lease | バリー |
|-----------|------|-----------------|------------|--------|
| **検出時間** | 1-5s | 5-30s（セッション） | 5-30s（TTL） | 2x RTT |
| **フェンシング** | ターム番号 | zxid | Revision | なし |
| **スプリットブレイン安全性** | 強い（クォーラム） | 強い（クォーラム） | 強い（Raftバック） | 弱い |
| **運用の複雑さ** | 高い | 中 | 中 | 低い |
| **パーティション時の振る舞い** | マイノリティはリーダーを失う | マイノリティはセッションを失う | マイノリティはリースを失う | 両方が選出 |
| **一貫性** | 線形化可能 | 線形化可能（書き込み） | 線形化可能 | なし |

### いつどれを使うか

```
Raft:       Already running Raft-based system, need strongest guarantees
ZooKeeper:  Existing ZK infra (Kafka/HBase), need ordered succession
etcd lease: Kubernetes-native, lightweight coordination, gRPC ecosystem
Bully:      Single datacenter, no partition tolerance needed, prototyping
```

> **スコープノート：** これらのプロトコルの基盤となるコンセンサスメカニズムの詳細な説明については、`08-consensus-algorithms.md`を参照してください。リーダー選出の上に構築されるトランザクション保証については、`07-distributed-transactions.md`を参照してください。

---

## 重要なポイント

1. **リーダーは調整をシンプルにする** - 単一の意思決定者です
2. **スプリットブレインは敵** - クォーラム + フェンシングで防止しましょう
3. **フェンシングトークンは不可欠** - 古いリーダーを拒否します
4. **リースには更新が必要** - 障害に対する猶予期間を設けましょう
5. **コンセンサスが最も安全** - 強い保証にはRaft/Paxosを使用しましょう
6. **クロックは信頼できない** - タイムスタンプだけに頼らないでください
7. **リーダーのヘルスを監視する** - 不健全な場合はステップダウンしましょう
8. **ハンドオフを計画する** - メンテナンスのためのグレースフルな移管を準備しましょう
