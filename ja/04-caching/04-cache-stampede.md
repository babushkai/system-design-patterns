# キャッシュスタンピード

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../04-caching/04-cache-stampede.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

キャッシュスタンピード（サンダリングハード）は、多数のリクエストが期限切れのキャッシュエントリを同時に再投入しようとし、データベースを圧倒する現象です。解決策には、ロック（1つのリクエストのみが再生成）、確率的早期期限切れ、リクエストコアレシングがあります。人気があり計算コストの高いキャッシュエントリに対する予防が重要です。

---

## 問題

### 通常動作

```
Time 0: 1000 requests/sec, cache hit
  Cache: [user:123] → data (TTL: 5min remaining)
  Database: idle

All requests served from cache ✓
```

### スタンピードシナリオ

```
Time T: Cache entry expires
  Cache: [user:123] → MISS

  Request 1: Cache miss → Query DB
  Request 2: Cache miss → Query DB
  Request 3: Cache miss → Query DB
  ...
  Request 1000: Cache miss → Query DB

  Database: 1000 simultaneous queries!
  Response time: 10x slower or timeout
  Possible cascade failure
```

### 問題の可視化

```
Requests over time:

        │ Cache expires
        │     ↓
────────┼─────╱╲───────────
        │    ╱  ╲
        │   ╱    ╲
        │  ╱      ╲─────────
        │ ╱
────────┴─────────────────────
             │
        Stampede window
```

---

## 解決策1：ロック

### 外部ロック

```python
def get_with_lock(key):
    value = cache.get(key)
    if value:
        return value

    # Try to acquire lock
    lock_key = f"lock:{key}"
    if cache.set(lock_key, "1", nx=True, ex=30):
        try:
            # Won the lock - fetch from DB
            value = database.get(key)
            cache.set(key, value, ex=3600)
            return value
        finally:
            cache.delete(lock_key)
    else:
        # Another process is refreshing
        # Wait and retry
        sleep(0.1)
        return get_with_lock(key)  # Retry
```

### 単純なロックの問題点

```
1. Retry storms
   1000 requests waiting, all retry simultaneously

2. Lock expiration
   Lock expires before DB query completes

3. Deadlock potential
   Lock holder crashes without releasing
```

### より良いロック：待機して古いデータを返す

```python
def get_with_stale_fallback(key):
    value, ttl = cache.get_with_ttl(key)

    if value and ttl > 0:
        return value  # Fresh data

    lock_key = f"lock:{key}"
    if cache.set(lock_key, "1", nx=True, ex=30):
        try:
            # Refresh in background
            value = database.get(key)
            cache.set(key, value, ex=3600)
        finally:
            cache.delete(lock_key)

    if value:
        return value  # Return stale data while refreshing

    # No stale data - must wait
    sleep(0.1)
    return cache.get(key)
```

---

## 解決策2：確率的早期期限切れ

### コンセプト

期限切れ前にランダムにリフレッシュし、負荷を分散します。

```python
import random
import math

def should_recompute(key, ttl_remaining, beta=1):
    """
    XFetch algorithm:
    Probability of recompute increases as TTL decreases
    """
    if ttl_remaining <= 0:
        return True

    # Probability increases exponentially as TTL approaches 0
    expiry_gap = beta * math.log(random.random())
    return -expiry_gap >= ttl_remaining
```

### 実装

```python
def get_with_probabilistic_refresh(key, compute_func, ttl=3600, beta=60):
    value, remaining_ttl = cache.get_with_ttl(key)

    if value is None or should_recompute(key, remaining_ttl, beta):
        # Recompute (either expired or probabilistically chosen)
        value = compute_func()
        cache.set(key, value, ex=ttl)

    return value
```

### 可視化

```
TTL timeline:
  |────────────────────────────────|
  Full TTL                         Expiry

Refresh probability:
  |░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓██|
  Low                    Medium   High

Some requests refresh early, spreading the load
```

---

## 解決策3：リクエストコアレシング

### コンセプト

重複リクエストを1つのデータベースクエリにまとめます。

```
Before coalescing:
  Request 1 ─────► DB Query
  Request 2 ─────► DB Query
  Request 3 ─────► DB Query

After coalescing:
  Request 1 ─────┐
  Request 2 ─────┼─► DB Query ─► All requests
  Request 3 ─────┘
```

### 実装

```python
import threading
from concurrent.futures import Future

class RequestCoalescer:
    def __init__(self):
        self.pending = {}  # key → Future
        self.lock = threading.Lock()

    def get(self, key, fetch_func):
        with self.lock:
            if key in self.pending:
                # Another request is fetching - wait for it
                return self.pending[key].result()

            # First request - create future
            future = Future()
            self.pending[key] = future

        try:
            # Fetch the data
            value = fetch_func(key)
            future.set_result(value)
            return value
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            with self.lock:
                del self.pending[key]

# Usage
coalescer = RequestCoalescer()

def get_user(user_id):
    cached = cache.get(f"user:{user_id}")
    if cached:
        return cached

    # Coalesce concurrent requests
    user = coalescer.get(
        f"user:{user_id}",
        lambda k: database.get_user(user_id)
    )

    cache.set(f"user:{user_id}", user)
    return user
```

### Go：singleflight

```go
import "golang.org/x/sync/singleflight"

var group singleflight.Group

func GetUser(userID string) (*User, error) {
    key := fmt.Sprintf("user:%s", userID)

    // Check cache
    if cached, ok := cache.Get(key); ok {
        return cached.(*User), nil
    }

    // Coalesce concurrent requests
    result, err, _ := group.Do(key, func() (interface{}, error) {
        user, err := database.GetUser(userID)
        if err != nil {
            return nil, err
        }
        cache.Set(key, user, time.Hour)
        return user, nil
    })

    if err != nil {
        return nil, err
    }
    return result.(*User), nil
}
```

---

## 解決策4：バックグラウンドリフレッシュ

### コンセプト

キャッシュを期限切れにさせない。期限切れ前にバックグラウンドでリフレッシュします。

```python
import threading

def get_with_background_refresh(key, ttl=3600, refresh_threshold=300):
    value, remaining_ttl = cache.get_with_ttl(key)

    if value and remaining_ttl > refresh_threshold:
        return value  # Fresh enough

    if value and remaining_ttl > 0:
        # Getting stale - trigger background refresh
        trigger_background_refresh(key)
        return value  # Return current value

    # Expired or missing - must fetch synchronously
    value = database.get(key)
    cache.set(key, value, ex=ttl)
    return value

def trigger_background_refresh(key):
    # Non-blocking refresh
    def refresh():
        value = database.get(key)
        cache.set(key, value, ex=3600)

    thread = threading.Thread(target=refresh)
    thread.start()
```

### リフレッシュトークンを使用

```python
def set_with_refresh(key, value, ttl=3600, refresh_at=300):
    """Store value with metadata for refresh"""
    data = {
        "value": value,
        "refresh_at": time.time() + ttl - refresh_at
    }
    cache.set(key, json.dumps(data), ex=ttl)

def get_with_refresh(key, fetch_func):
    raw = cache.get(key)
    if not raw:
        return fetch_sync(key, fetch_func)

    data = json.loads(raw)

    if time.time() > data["refresh_at"]:
        # Time to refresh in background
        trigger_async_refresh(key, fetch_func)

    return data["value"]
```

---

## 解決策5：リースベースのアプローチ

### コンセプト

最初のリクエスターがリフレッシュの「リース」を取得します。他のリクエストは待機するか古いデータを取得します。

```python
def get_with_lease(key, stale_ttl=60):
    # Try to get current value
    value = cache.get(key)
    if value:
        return value

    # Try to acquire lease
    lease_key = f"lease:{key}"
    lease_acquired = cache.set(lease_key, "1", nx=True, ex=30)

    if lease_acquired:
        # We have the lease - refresh
        value = database.get(key)
        cache.set(key, value, ex=3600)
        cache.delete(lease_key)
        return value
    else:
        # Someone else is refreshing
        # Check for stale value
        stale_value = cache.get(f"stale:{key}")
        if stale_value:
            return stale_value

        # No stale value - wait briefly
        sleep(0.1)
        return cache.get(key)

# Keep stale copy for fallback
def set_with_stale(key, value, ttl=3600, stale_ttl=86400):
    cache.set(key, value, ex=ttl)
    cache.set(f"stale:{key}", value, ex=stale_ttl)
```

---

## 予防戦略

### プレウォーミング

```python
def warm_cache_on_startup():
    """Pre-populate cache before taking traffic"""
    popular_keys = get_popular_keys()
    for key in popular_keys:
        value = database.get(key)
        cache.set(key, value, ex=3600)

    log.info(f"Warmed {len(popular_keys)} cache entries")
```

### TTLのずらし

```python
import random

def set_with_jitter(key, value, base_ttl=3600, jitter_percent=10):
    """Add random jitter to prevent synchronized expiration"""
    jitter = base_ttl * jitter_percent / 100
    actual_ttl = base_ttl + random.uniform(-jitter, jitter)
    cache.set(key, value, ex=int(actual_ttl))

# Instead of all keys expiring at exactly 1 hour:
# Keys expire between 54 and 66 minutes
```

### 非期限切れ + 非同期リフレッシュ

```python
def setup_cache_refresh_worker():
    """Background worker that refreshes cache entries"""
    while True:
        # Get keys approaching expiration
        keys = get_keys_expiring_soon(threshold=300)

        for key in keys:
            try:
                value = database.get(key)
                cache.set(key, value, ex=3600)
            except Exception as e:
                log.error(f"Failed to refresh {key}: {e}")

        sleep(60)
```

---

## 比較表

| 解決策 | 複雑さ | レイテンシ | DB負荷 | 古いデータ |
|----------|------------|---------|---------|------------|
| ロック | 中 | 高い | 最低 | 可能性あり |
| 確率的 | 低 | 通常 | 低 | なし |
| コアレシング | 中 | 通常 | 低 | なし |
| バックグラウンドリフレッシュ | 高 | 最低 | 中 | あり |
| リース + 古いデータ | 高 | 低 | 低 | あり |

---

## 確率的早期期限切れ

### XFetchアルゴリズムの詳細

XFetchアルゴリズムはVattaniらによって導入されました。TTLが期限切れになる前にキャッシュエントリを確率的に再計算することでスタンピードを防ぎます。核心的な洞察は、キャッシュされた値が期限切れに近づくにつれて、個々のリクエストが独立して再計算をトリガーするかどうかを決定し、期限切れの正確な瞬間に集中するのではなく、時間の経過とともにリビルド負荷を分散させることです。

### 計算式

```
should_recompute = random() < (time_since_compute / ttl) ^ beta
```

- `time_since_compute`: 値が最後に計算されてからの経過時間
- `ttl`: キャッシュエントリの合計生存時間
- `beta`: 積極性パラメータ（高い値 = より早い再計算）

`beta = 1` の場合、確率は線形に増加します。`beta > 1` の場合、再計算がより早く積極的にトリガーされます。`beta < 1` の場合、再計算は実際の期限切れ時間に近い場所に集中します。

### betaのチューニング

```
beta = 0.5  → Conservative: recomputes very close to expiry
              Good for cheap queries, high-throughput keys

beta = 1.0  → Linear: balanced spread across the TTL window
              Default choice for most workloads

beta = 2.0  → Aggressive: starts recomputing well before expiry
              Use for expensive queries (>500ms) or extremely hot keys

beta = 3.0+ → Very aggressive: almost always recomputes in the last third of TTL
              Rarely needed; consider background refresh instead
```

### なぜ機能するか

1秒あたり1000リクエストで1時間のTTLの場合、確率分布により、期限切れ前の最後の数秒間に約1つのリクエストが再計算をトリガーします。残りの999のリクエストはキャッシュされた値を提供し続けます。調整なし、ロックなし、分散状態なし -- 各ノードが独立したローカル判断を行います。

### 注意点

- 正確に1回の再計算を保証するものではありません。非常に高い同時実行性の下では、2つまたは3つの同時再計算が発生する可能性がありますが、1000回よりは桁違いに良好です
- クライアントが値が最初に計算された時刻を追跡する必要があります（キャッシュされた値と一緒に `compute_time` を保存）
- キャッシュエントリがほとんどアクセスされない場合は効果がありません。確率的トリガーは適切なタイミングで発火するためにリクエスト量に依存します

---

## ロックベースのリビルドパターン

### Mutexパターン

Mutexパターンは、1つのリクエストだけがキャッシュをリビルドし、他のすべてのリクエストは待機するか古いデータを受け取ることを保証します。キャッシュミスを最初に検出したリクエストが分散ロックを取得し、リビルドして、ロックを解放します。

### Redis分散Mutex

```python
def rebuild_with_mutex(key, compute_func, ttl=3600):
    value = cache.get(key)
    if value is not None:
        return value

    lock_key = f"lock:{key}"
    # SET NX = set if not exists, EX = expiry in seconds
    # Lock TTL must exceed worst-case compute time
    acquired = redis.execute_command("SET", lock_key, "1", "NX", "EX", 5)

    if acquired:
        try:
            value = compute_func()
            cache.set(key, value, ex=ttl)
            return value
        finally:
            redis.delete(lock_key)
    else:
        # Lock held by another request — spin-wait with backoff
        for attempt in range(10):
            time.sleep(0.05 * (2 ** attempt))  # exponential backoff
            value = cache.get(key)
            if value is not None:
                return value
        raise TimeoutError(f"Cache rebuild timeout for {key}")
```

`NX` フラグが重要な部分です。同じRedisクラスタを共有するすべてのアプリケーションインスタンスにわたって、アトミックなテストアンドセットセマンティクスを提供します。

### Stale-While-Revalidate

待機者をブロックする代わりに、正確に1つのリクエストがバックグラウンドでリビルドしている間、前の（古い）値を提供します。

```python
def stale_while_revalidate(key, compute_func, ttl=3600, stale_ttl=86400):
    value = cache.get(key)
    stale_value = cache.get(f"stale:{key}")

    if value is not None:
        return value

    lock_key = f"lock:{key}"
    if redis.execute_command("SET", lock_key, "1", "NX", "EX", 5):
        # Rebuild asynchronously if we have stale data to serve
        if stale_value is not None:
            threading.Thread(target=_rebuild, args=(key, compute_func, ttl)).start()
            return stale_value
        # No stale data — must rebuild synchronously
        return _rebuild(key, compute_func, ttl)
    else:
        return stale_value  # Serve stale while another request rebuilds
```

### パターン比較

| 項目 | Mutex | Stale-While-Revalidate | 確率的 |
|--------|-------|------------------------|---------------|
| 調整 | 分散ロック | 分散ロック + 古いデータストア | なし |
| 待機者のレイテンシ | リビルドまでブロック | 即時（古いデータ） | 即時（キャッシュ済み） |
| データの鮮度 | 常に最新 | 一時的に古い | 通常は最新 |
| インフラ | ロックサービスが必要 | ロック + デュアルキーストレージ | 追加インフラ不要 |
| 障害モード | ロック期限切れ → 短時間のスタンピード | 古いデータがより長く提供される | 時々二重計算 |
| 適用場面 | 低トラフィック、鮮度重視 | 高トラフィック、古さ許容 | 高トラフィック、ステートレス |

データの鮮度が譲れない場合（例：口座残高）はMutexを選択してください。低レイテンシが鮮度より重要な場合（例：商品カタログ、フィードタイムライン）はStale-While-Revalidateを選択してください。ゼロ調整のシンプルさが必要な場合は確率的を選択してください。

---

## 実際のスタンピード事例

### コールドスタートスタンピード

ローカルキャッシュが空の状態で新しいサービスインスタンスをデプロイ（またはクラッシュ後に再起動）すると、すべての着信リクエストが同時にキャッシュミスになります。

```
Deploy new instance → 0 cached entries → 100% miss rate → DB overwhelmed

Timeline:
  t=0s   Instance joins load balancer
  t=0.1s 500 requests arrive, all miss cache
  t=0.2s Database connection pool exhausted
  t=1s   Cascading timeouts, health check fails
  t=5s   Instance removed from load balancer
```

**対策：トラフィックルーティング前にプレウォームします。** 兄弟インスタンスまたはキーレジストリからトップNのホットキーを取得します。キャッシュヒット率がしきい値（例：80%）を超えた後にのみ、インスタンスを正常とマークします。Kubernetesでは、Podがサービスエンドポイントに入る前にキャッシュの準備状態を検証するstartup probeを使用してください。

### セレブリティツイートスタンピード

5000万人のフォロワーを持つセレブリティがツイートします。フォロワーのタイムラインキャッシュはすべてこのツイートを参照しています。ツイートオブジェクトのキャッシュエントリが期限切れになると（または編集による無効化が行われると）、数百万のタイムラインレンダリングが同時にデータベースから同じツイートを要求します。

```
Celebrity tweets → 50M follower timelines reference tweet:12345
Cache TTL expires on tweet:12345
  → 200K concurrent cache misses across 40 app servers
  → Single DB row receives 200K reads in <1 second
  → Row-level lock contention, replication lag spikes
```

**対策:** アクセス頻度がホットスポットしきい値を超えるキャッシュキーに対して、リースベースのリビルド + Stale-While-Revalidateを使用します。Twitterのアプローチ：共有キャッシュ層をバイパスして、人気オブジェクトをすべてのアプリサーバーのローカルキャッシュに複製する専用のホットキーキャッシュ層。

### CDNオリジンスタンピード

CDNキャッシュパージ（例：更新された静的アセットのデプロイ後）がすべてのエッジノードに同時に伝播します。すべてのエッジノードがキャッシュされたコピーを無効と見なし、同じ瞬間にオリジンサーバーに新しいバージョンを要求します。

```
Cache purge issued → 200 edge PoPs invalidate simultaneously
  → 200 concurrent requests to origin
  → Origin bandwidth saturated, 503 errors propagate to users
```

**対策:** CDN層でリクエストコアレシングを使用します（例：Cloudflareの「request collapsing」、Varnishの `grace` モード）。ソフトパージを使用してパージ伝播をずらします。コンテンツを削除するのではなく古いとマークし、各エッジが独立して再検証できるようにします。中間キャッシュでオリジンをシールドし、コアレスされたリクエストストームを吸収します。

---

## 監視と検出

### 主要メトリクス

スタンピードがカスケードする前に検出するために、以下のシグナルを追跡してください。

- **キャッシュミス率のスパイク**: ミス率の急上昇（例：数秒以内に2%から40%）が主要な指標です
- **データベースQPSの相関**: DBクエリ率をキャッシュTTL期限切れイベントと重ねて表示し、同期したスパイクがスタンピード動作を確認します
- **ロック競合率**: Mutexベースのパターンを使用している場合、ロック取得失敗を同時リビルド試行のプロキシとして監視します
- **p99レイテンシスパイク**: メディアンレイテンシが安定していても、スタンピードはテールレイテンシの爆発として現れます

### Redis診断

```bash
# Real-time hit/miss ratio
redis-cli INFO stats | grep keyspace
# Output: keyspace_hits:1234567  keyspace_misses:8901

# Compute hit rate
# hit_rate = keyspace_hits / (keyspace_hits + keyspace_misses)
# Healthy: > 0.95 for hot keys
# Stampede signal: sudden drop below 0.80
```

### アラートルール

```yaml
# Prometheus alert example
- alert: CacheStampedeDetected
  expr: |
    rate(cache_misses_total[1m]) > 2 * avg_over_time(rate(cache_misses_total[1m])[1h:1m])
  for: 30s
  labels:
    severity: warning
  annotations:
    summary: "Cache miss rate >2x baseline for 30s — potential stampede"
```

`for: 30s` 句は、短時間の良性のミススパイク（例：単一キーの期限切れ）からの誤報を回避します。アラートが発火した場合、DB負荷と照合し、`RANDOMKEY` のサンプリングまたはキーレベルのミス追跡を有効にして、どのキーがミスしているかを特定してください。

### ダッシュボードレイアウト

以下の4つのパネルを単一の時間軸ダッシュボードに重ねて表示します。

1. **キャッシュヒット率**（目標：>95%）-- 主要な健全性指標
2. **DB QPS** -- フラットであるべき。スパイクはスタンピードと相関
3. **キャッシュTTL期限切れヒストグラム** -- キーの期限切れスケジュールを表示。クラスタ化された期限切れはジッター設定ミスを示します
4. **ロック取得率**（Mutexパターン使用時）-- スパイクは同時リビルド競合を示します

スタンピードが発生すると、パネル1と2は互いに逆の動きをします。ヒット率が低下するとDB QPSがスパイクします。パネル3は根本原因が同期されたTTL期限切れかどうかを示し、パネル4はロック層が負荷を吸収しているか圧倒されているかを確認します。

---

## 重要なポイント

1. **スタンピードはデータベースを殺す** - 1つの人気キーが障害を引き起こす可能性
2. **ロックは冗長なクエリを防ぐ** - しかしレイテンシが増加
3. **確率的リフレッシュは負荷を分散** - シンプルで効果的
4. **コアレシングはリクエストをまとめる** - 同時リクエストに最適
5. **バックグラウンドリフレッシュはキャッシュを温かく保つ** - 真に期限切れにしない
6. **古いデータは多くの場合許容可能** - リフレッシュ中に古いデータを提供
7. **TTLをずらす** - 同期された期限切れを防止
8. **重要なキーをプレウォーム** - コールドスタートしない
