# キャッシュ無効化

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../04-caching/02-cache-invalidation.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

キャッシュ無効化は、古くなったキャッシュエントリを削除または更新するプロセスです。有名な格言が当てはまります。「コンピュータサイエンスで本当に難しいことは2つだけ：キャッシュ無効化と命名」。戦略にはTTL、明示的無効化、イベント駆動無効化、バージョニングがあります。古さの許容度、複雑さの予算、一貫性要件に基づいて選択してください。

---

## 問題

### データの古さ

```
Time 0: User reads profile (cached)
  Cache: {name: "Alice", age: 30}

Time 1: User updates profile
  Database: {name: "Alice", age: 31}
  Cache: {name: "Alice", age: 30}  ← stale!

Time 2: Another user reads profile
  Returns: {age: 30}  ← wrong!
```

### なぜ難しいのか

```
1. Distributed nature
   Cache and database are separate systems
   No atomic update across both

2. Multiple caches
   Browser cache, CDN, application cache, database cache
   All need invalidation

3. Cache dependencies
   Invalidating A might require invalidating B, C, D
   Complex graphs of dependencies
```

---

## TTLベースの期限切れ

### 動作の仕組み

```python
cache.set("user:123", user_data, ttl=300)  # 5 minutes

# After 300 seconds, entry expires
# Next read triggers cache miss
# Fresh data loaded from database
```

### TTLの選択

```
Short TTL (seconds):
  + Fresher data
  - More cache misses
  - Higher database load
  Use: Real-time data

Medium TTL (minutes):
  + Balance of freshness and efficiency
  - Still have staleness window
  Use: User profiles, product data

Long TTL (hours/days):
  + Very high cache hit rate
  - Potentially very stale
  Use: Static content, rarely changing data
```

### TTLのトレードオフ

```
TTL too short:
  Cache hit rate: 60%
  DB queries/sec: 400

TTL too long:
  Cache hit rate: 95%
  DB queries/sec: 50
  Staleness: up to 24 hours

Sweet spot: Based on change frequency
  If data changes every 10 min: TTL = 1 min
  If data changes every day: TTL = 1 hour
```

---

## 明示的無効化

### 書き込み時の削除

```python
def update_user(user_id, new_data):
    # Update database
    db.update("users", user_id, new_data)

    # Invalidate cache
    cache.delete(f"user:{user_id}")

    # Next read will cache fresh data
```

### 書き込み時の更新

```python
def update_user(user_id, new_data):
    # Update database
    db.update("users", user_id, new_data)

    # Update cache with new data
    cache.set(f"user:{user_id}", new_data)
```

### 削除と更新のトレードオフ

```
Delete:
  + Simpler (one operation)
  + Never caches wrong data
  - Next read is a miss

Update:
  + Cache always warm
  + Lower latency
  - Risk of caching wrong data
  - More complex error handling
```

---

## イベント駆動無効化

### メッセージキューの使用

```
┌─────────┐     ┌─────────┐     ┌─────────────┐
│ Writer  │────►│  Kafka  │────►│   Cache     │
│ Service │     │ / Redis │     │ Invalidator │
└─────────┘     └─────────┘     └─────────────┘

1. Writer updates database
2. Writer publishes invalidation event
3. Invalidator consumes event
4. Invalidator deletes cache entry
```

### 実装

```python
# Writer service
def update_user(user_id, data):
    db.update(user_id, data)
    publish_event("cache.invalidate", {
        "key": f"user:{user_id}",
        "type": "delete"
    })

# Invalidator service
def consume_invalidation(event):
    if event["type"] == "delete":
        cache.delete(event["key"])
    elif event["type"] == "update":
        new_value = db.read(event["key"])
        cache.set(event["key"], new_value)
```

### メリットとデメリット

```
Pros:
  + Decoupled invalidation
  + Reliable (message queue persistence)
  + Can batch invalidations
  + Works across services

Cons:
  + Additional latency (queue processing)
  + Message ordering challenges
  + Queue becomes critical path
  + More infrastructure
```

---

## Change Data Capture（CDC）

### 動作の仕組み

```
┌──────────┐     ┌───────────┐     ┌───────┐
│ Database │────►│ CDC Tool  │────►│ Cache │
│ (binlog) │     │ (Debezium)│     │       │
└──────────┘     └───────────┘     └───────┘

Database writes → Captured from transaction log
No application changes needed
```

### メリット

```
1. No code changes in writers
2. Captures all changes (including manual DB updates)
3. Exactly-once semantics possible
4. Works retroactively
```

### 例：Debezium

```json
// CDC event from Debezium
{
  "op": "u",  // update
  "before": {"id": 123, "name": "Alice", "age": 30},
  "after": {"id": 123, "name": "Alice", "age": 31},
  "source": {
    "table": "users",
    "ts_ms": 1634567890123
  }
}

// Consumer updates cache
cache.set(f"user:{event['after']['id']}", event['after'])
```

---

## バージョンベースの無効化

### バージョン付きキャッシュキー

```python
# Global version for user data
USER_VERSION = get_current_version()  # e.g., from config or DB

def cache_key(user_id):
    return f"user:v{USER_VERSION}:{user_id}"

def read_user(user_id):
    return cache.get(cache_key(user_id))

# Invalidate all users: increment version
def invalidate_all_users():
    increment_version()  # All old keys now orphaned
```

### エンティティごとのバージョン

```python
def cache_key(user_id, version):
    return f"user:{user_id}:v{version}"

def read_user(user_id):
    # Get current version from lightweight store
    version = version_store.get(f"user_version:{user_id}")
    return cache.get(cache_key(user_id, version))

def update_user(user_id, data):
    new_version = version_store.increment(f"user_version:{user_id}")
    db.update(user_id, data)
    cache.set(cache_key(user_id, new_version), data)
```

### メリット

```
+ Instant invalidation (version change)
+ No need to enumerate cached items
+ Supports bulk invalidation
+ Old versions expire naturally (TTL)

- Version lookup overhead
- Cache memory wasted on old versions
```

---

## タグベースの無効化

### コンセプト

```
Associate cache entries with tags
Invalidate by tag to remove related entries

cache.set("product:123", data, tags=["products", "category:electronics"])
cache.set("product:456", data, tags=["products", "category:electronics"])
cache.set("product:789", data, tags=["products", "category:clothing"])

# Invalidate all electronics
cache.invalidate_tag("category:electronics")
# Deletes: product:123, product:456
# Keeps: product:789
```

### 実装

```python
class TaggedCache:
    def set(self, key, value, tags=[]):
        self.cache.set(key, value)
        for tag in tags:
            self.tag_store.sadd(f"tag:{tag}", key)

    def invalidate_tag(self, tag):
        keys = self.tag_store.smembers(f"tag:{tag}")
        for key in keys:
            self.cache.delete(key)
        self.tag_store.delete(f"tag:{tag}")
```

### ユースケース

```
E-commerce:
  Tag products with category, brand, seller
  Seller updates info → invalidate all their products

CMS:
  Tag pages with author, topic, template
  Template change → invalidate all pages using it
```

---

## 無効化失敗の処理

### リトライパターン

```python
def invalidate_with_retry(key, max_retries=3):
    for attempt in range(max_retries):
        try:
            cache.delete(key)
            return True
        except CacheError:
            if attempt < max_retries - 1:
                sleep(exponential_backoff(attempt))

    # Log failure, add to dead letter queue
    log_failed_invalidation(key)
    return False
```

### グレースフルデグラデーション

```python
def update_user(user_id, data):
    # Database update is critical
    db.update(user_id, data)

    # Cache invalidation is best-effort
    try:
        cache.delete(f"user:{user_id}")
    except CacheError:
        # Log but don't fail the operation
        log.warn(f"Cache invalidation failed for user:{user_id}")
        # TTL will eventually expire the stale entry
```

### 一貫性レベル

```
Level 1: Best effort (fire and forget)
  - Log failure
  - Rely on TTL

Level 2: Retry with backoff
  - Retry N times
  - Log persistent failures

Level 3: Guaranteed
  - Use transactions or queues
  - Never acknowledge write until cache invalidated
  - Much more complex
```

---

## キャッシュの依存関係

### 問題

```
User profile caches:
  user:123 → {name, age, friends: [456, 789]}
  user:456 → {name, age, friends: [123]}
  user_friends:123 → [user:456, user:789]  # derived cache

When user:456 updates name:
  - user:456 invalidated ✓
  - user_friends:123 still has old data ✗
```

### 依存関係の追跡

```python
class DependencyAwareCache:
    def __init__(self):
        self.dependencies = {}  # key → set of dependent keys

    def set_with_deps(self, key, value, depends_on=[]):
        self.cache.set(key, value)
        for dep in depends_on:
            self.dependencies.setdefault(dep, set()).add(key)

    def invalidate(self, key):
        # Delete the key
        self.cache.delete(key)

        # Invalidate dependents recursively
        dependents = self.dependencies.pop(key, set())
        for dep_key in dependents:
            self.invalidate(dep_key)
```

### 深い依存関係の回避

```
Better approach: Denormalize and bound staleness

Instead of:
  user_friends:123 depends on user:456, user:789

Do:
  user_friends:123 = [{id: 456, name: "Bob"}, ...]
  TTL = 1 minute

Staleness bounded, no cascade invalidation
```

---

## 多層キャッシュの無効化

### 課題

```
Browser → CDN → App Cache → Database

All layers need invalidation!
```

### HTTPキャッシュヘッダー

```
Cache-Control: max-age=300  # Browser caches 5 min
ETag: "abc123"              # Version for revalidation

# On update:
# Return new ETag
# Client revalidates and gets new data
```

### CDN無効化

```python
def update_product(product_id, data):
    db.update(product_id, data)

    # Invalidate app cache
    app_cache.delete(f"product:{product_id}")

    # Invalidate CDN
    cdn.purge(f"/api/products/{product_id}")
    cdn.purge(f"/products/{product_id}.html")
```

### 無効化の伝播

```
Order of invalidation matters:
  1. CDN (users hit this first)
  2. App cache (backend requests)
  3. Database cache (if any)

Or use versioned URLs:
  /products/123?v=abc123
  New version = new URL = cache miss everywhere
```

---

## 大規模でのキャッシュ無効化

### 無効化時のサンダリングハード

人気のキャッシュキーが無効化されると、すべての同時リクエストがキャッシュミスを検出し、同時にデータベースにアクセスします。

```
Popular key "product:homepage-banner" invalidated
  → 10,000 requests/sec all miss cache
  → 10,000 concurrent DB queries → DB connection pool exhausted
```

対策：

```
1. Staggered TTLs: base_ttl + random(0, base_ttl * 0.1)
2. Probabilistic early expiration: P(refresh) increases as TTL nears end
3. Lock-based cache rebuild: first request acquires lock, others wait
```

```python
def get_with_lock(key):
    value = cache.get(key)
    if value is not None:
        return value
    lock_key = f"lock:{key}"
    if cache.set(lock_key, "1", nx=True, ex=5):  # acquire lock
        value = db.query(key)
        cache.set(key, value, ex=300)
        cache.delete(lock_key)
        return value
    else:
        sleep(0.05)
        return cache.get(key)  # retry once
```

### 一括無効化

```
Problem: invalidating all keys for user 123 requires enumerating them

Solution: version-based keys (v2:user:123:profile → v3:user:123:profile)
  Increment version → old keys are never read → expire via TTL
  O(1) invalidation, no enumeration, no multi-key delete
```

### データセンター間の無効化

DC-Aに書き込みが着地すると、DC-Bのキャッシュは明示的に無効化されるまで古いままです。

```
1. Pub/Sub invalidation bus (Kafka / Redis Pub/Sub)
   DC-A write → publish event → DC-B consumer deletes local cache
   Latency: 50-200ms cross-region. Message loss requires TTL fallback.

2. Bounded TTL with accepted staleness
   Set TTL low enough that staleness window is acceptable
   Simpler — no cross-DC messaging infrastructure needed

3. Lease-based invalidation
   Cache entry includes a lease token tied to the writer DC
   Remote DCs validate lease before serving cached data
```

### FacebookのMemcache無効化

Facebookの `mcsqueal` デーモンはMySQLのbinlogをテーリングし、リージョナルなMemcachedクラスタに無効化イベントを発行します。TAO論文（USENIX ATC 2013）に記載されており、アプリケーションレベルの無効化ロジックなしで1日あたり数十億の無効化を処理します。重要な洞察：コミットログが何が変更されたかの唯一の真実の源です。

---

## CDCベースの無効化

### パターン概要

```
┌──────────┐     ┌───────────┐     ┌─────────┐     ┌──────────────┐
│ Database │────►│ Debezium  │────►│  Kafka   │────►│    Cache     │
│ (binlog) │     │ Connector │     │  Topic   │     │  Invalidator │
└──────────┘     └───────────┘     └─────────┘     └──────────────┘

1. Any process writes to database (app, migration, admin script)
2. Debezium captures the change from the transaction log
3. Change event published to Kafka topic (table-level or row-level)
4. Cache invalidator consumes event and deletes/updates cache entry
```

### アプリケーションレベルの無効化に対する利点

```
Application-level: only catches writes through app code, misses direct
DB updates/migrations/admin scripts, every writer must remember to invalidate

CDC-based: captures ALL changes regardless of source, zero app code changes
for new writers, single invalidation path — no duplication, no forgetting
```

### 順序の保証

CDCイベントは単一パーティション（テーブル+主キー）内でコミット順に到着します。これにより、古い値が新しい値を上書きする競合状態を防ぎます。

```
Without ordering: T1 writes "Bob", T2 writes "Carol", T1's cache SET arrives last
  → cache holds "Bob" instead of "Carol"

With CDC ordering: events arrive in commit order
  → final cache state always reflects the latest write
  → use DELETE (not SET) for extra safety against reordering
```

### レイテンシ特性

一般的なエンドツーエンド：約50-200ms（p50）、約200-500ms（p99）。内訳：DB-to-binlog 約1ms、Debeziumキャプチャ 10-50ms、Kafkaのパブリッシュ+コンシューム 20-100ms、キャッシュ削除 1-5ms。

### CDCベース無効化の適用場面

```
Good fit: multi-service writes to same DB, critical cache consistency
(financial/inventory), audit trail needed, legacy direct-DB access

Poor fit: simple single-service CRUD, DB without binlog/WAL access,
invalidation latency budget < 50ms
```

---

## 一貫性ウィンドウ

### 定義

一貫性ウィンドウとは、データベース書き込みの完了からキャッシュにその書き込みが反映されるまでの時間（`T_cache_updated - T_db_commit`）です。このウィンドウ中の読み取りは古いデータを返す可能性があります。

### 不整合の原因

```
1. TTL expiration delay:         window = 0 to full TTL
2. Invalidation propagation:     window = 50-500ms (event-driven)
3. Read-through race condition:  read caches pre-write value after write
4. Cross-DC replication lag:     window = 50-200ms cross-region
```

### 一貫性の測定

```
1. Sample comparison: periodically read N random keys from cache and DB,
   calculate staleness rate and duration

2. Invalidation lag tracking: timestamp DB commit and cache invalidation,
   track p50/p95/p99 of the delta

3. Cache miss rate analysis: lower-than-expected misses indicate stale
   data being served; unexpected misses indicate invalidation is working
```

### SLAの定義

```
Example: "Cache consistent within 500ms of DB write for 99.9% of keys"

Monitoring:
  - Alert if p99 invalidation lag > 1s
  - Alert if staleness sample check fails > 0.1% of keys
```

### 古さの持続時間がビジネスに与える影響

```
Use Case          │  1s     │  5s      │  30s     │  5min
──────────────────┼─────────┼──────────┼──────────┼──────────
User profile      │ Fine    │ Fine     │ Fine     │ Acceptable
Inventory count   │ Risky   │ Oversell │ Oversell │ Dangerous
Product price     │ Fine    │ Risky    │ Wrong $  │ Wrong $
Session/auth      │ Fine    │ Fine     │ Security │ Security
Feature flags     │ Fine    │ Fine     │ Fine     │ Stale exp
Leaderboard       │ Fine    │ Fine     │ Fine     │ Fine
```

ユースケースごとの最悪の許容古さに基づいて、TTLと無効化戦略を選択してください。

---

## よくある無効化バグ

### 競合状態：書き込み後の古い読み取り

```
Thread A (reader):          Thread B (writer):
  cache miss
  read DB → old value
                              write DB → new value
                              delete cache
  set cache → old value ← BUG: stale value cached after invalidation
```

修正：常にバウンドされたTTLを使用してください。キャッシュミス時に短いTTL（例：30秒）、明示的なWrite-through時に長いTTL（例：300秒）を使用します。競合状態の古さは短いTTLに限定されます。

### DeleteとSet-Emptyの違い

```
cache.delete("avatar:123")      → cache miss → DB query → correct but costly
cache.set("avatar:123", null)   → cache hit  → return null → fast but risky
```

判断：「見つからない」が有効で一般的な状態の場合（例：ユーザー名の利用可能性チェック）、nullをキャッシュします。「見つからない」がまれまたはバグを示す場合（例：ユーザープロフィール）、キーを削除して次の読み取りでDBに対して検証させます。

### カスケード無効化：派生キャッシュの忘れ

```
user:123 name changed "Alice" → "Bob"
  user:123            → invalidated ✓
  user:123:feed       → still shows "Alice posted..." ✗
  team:456:members    → still lists "Alice" ✗
```

解決策：(1) タグベースの無効化 -- すべての派生キャッシュに "user:123" をタグ付けし、単一の `invalidate_tag` ですべてクリア; (2) エンティティから派生キーへの明示的な依存関係レジストリ; (3) カスケードロジックなしに古さを限定するための派生キャッシュへの短いTTL。

### デプロイ中の無効化

ローリングデプロイ中、古いコードと新しいコードが異なるキャッシュフォーマットを書き込む可能性があります。新しいコードが古いフォーマットのエントリを読むと、フィールドが欠落する可能性があります。

```
Solutions:
  1. Versioned cache keys — old "v1:user:123", new "v2:user:123"
     Both coexist safely, old expires via TTL

  2. Cache flush on deploy — simple but causes cold-cache stampede
     Mitigate with cache warming (pre-populate critical keys)

  3. Backward-compatible serialization — handle missing fields with defaults
     Avoids the problem entirely but requires discipline
```

---

## 重要なポイント

1. **TTLはセーフティネット** - 無効化が失敗しても古さを限定
2. **削除は更新よりシンプル** - 誤ったデータをキャッシュするリスクが少ない
3. **イベント駆動はスケールする** - ライターをキャッシュから分離
4. **CDCはすべてをキャプチャ** - 直接的なDB変更を含む
5. **一括無効化にはバージョンキー** - 即時、列挙不要
6. **関連データにはタグ** - グループをまとめて無効化
7. **障害をグレースフルに処理** - キャッシュはクリティカルパスではない
8. **依存関係に注意** - カスケード無効化は複雑
