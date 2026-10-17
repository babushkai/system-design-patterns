# 分散トレーシング

> **注記:** このドキュメントは英語版からの翻訳です。最新の内容や正確な情報については、[英語版オリジナル](../../11-observability/01-distributed-tracing.md)を参照してください。

## TL;DR

分散トレーシングは、リクエストが複数のサービスを流れる様子を追跡し、トランザクションの全体像を把握します。各トレースは、コンテキスト伝播によって接続された個々の操作を表すスパンで構成されます。レイテンシの問題をデバッグし、システムの振る舞いを理解するために不可欠です。

---

## トレーシングが解決する問題

マイクロサービスアーキテクチャでは、1つのユーザーリクエストが多くのサービスを経由します。

```
User Request
     │
     ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│   API   │───►│  Auth   │───►│  User   │───►│  Cache  │
│ Gateway │    │ Service │    │ Service │    │         │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
     │                             │
     │                             ▼
     │                        ┌─────────┐
     │                        │   DB    │
     │                        └─────────┘
     ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│  Order  │───►│ Payment │───►│ Notif.  │
│ Service │    │ Service │    │ Service │
└─────────┘    └─────────┘    └─────────┘

トレーシングがない場合:
- 「リクエストに2秒かかった」 - でもどこで？
- 「決済が失敗した」 - でもユーザーのコンテキストは？
- ログが8つの異なるサービスに散在している
```

---

## トレーシングの概念

### トレース

トレースは、システム全体を通じたリクエストの旅路を表します。

```
Trace ID: abc123

┌─────────────────────────────────────────────────────────────────┐
│                          Time →                                 │
│                                                                 │
│ ├────────────── API Gateway (500ms) ───────────────────────────┤│
│ │    ├────── Auth Service (50ms) ──────┤                       ││
│ │    │                                  │                       ││
│ │    │    ├── User Service (200ms) ────┤                       ││
│ │    │    │   ├─ DB Query (150ms) ─┤   │                       ││
│ │    │    │                            │                       ││
│ │    ├────────── Order Service (400ms) ───────────────────────┤││
│ │    │         │    ├── Payment (250ms) ──┤                   │││
│ │    │         │    │                      │                   │││
│ │    │         │    │    ├─ Notify (30ms) ─┤                  │││
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### スパン

スパンは、トレース内の1つの作業単位を表します。

```json
{
    "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
    "spanId": "00f067aa0ba902b7",
    "parentSpanId": "d5ebc7e161ade64a",
    "operationName": "HTTP GET /users/{id}",
    "serviceName": "user-service",
    "startTime": "2024-01-01T10:00:00.000Z",
    "duration": "200ms",
    "status": "OK",
    "attributes": {
        "http.method": "GET",
        "http.url": "/users/123",
        "http.status_code": 200,
        "user.id": "123"
    },
    "events": [
        {"timestamp": "...", "name": "cache.miss", "attributes": {"db.system": "redis"}},
        {"timestamp": "...", "name": "user.found"}
    ]
}
```

### コンテキスト伝播

トレースコンテキストはサービス間で受け渡す必要があります。

```
Service A                        Service B
    │                                │
    │  HTTP Request                  │
    │  Headers:                      │
    │    traceparent: 00-abc123-...  │
    │    tracestate: vendor=value    │
    │ ──────────────────────────────►│
    │                                │
    │                                │ トレースコンテキストを抽出
    │                                │ 同じ trace_id で
    │                                │ 子スパンを作成
```

### W3C Trace Context 標準

```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
             │  │                                │                  │
             │  │                                │                  └─ フラグ (サンプリング対象)
             │  │                                └─ 親スパン ID
             │  └─ トレース ID
             └─ バージョン

tracestate: congo=t61rcWkgMzE,rojo=00f067aa0ba902b7
            ベンダー固有のキーバリューペア
```

---

## 実装

### Go による計装の例

```go
package main

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
)

var tracer = otel.Tracer("order-service")

func processOrder(ctx context.Context, orderID string) (*Order, error) {
	ctx, span := tracer.Start(ctx, "process_order")
	defer span.End()
	span.SetAttributes(attribute.String("order.id", orderID))

	// Child span — validation
	order, err := validateOrder(ctx, orderID)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		return nil, err
	}

	// Child span — payment
	ctx, paySpan := tracer.Start(ctx, "process_payment")
	result, err := paymentService.Charge(ctx, order)
	if err != nil {
		paySpan.RecordError(err)
		paySpan.SetStatus(codes.Error, err.Error())
		paySpan.End()
		return nil, err
	}
	paySpan.SetAttributes(attribute.String("payment.method", result.Method))
	paySpan.End()

	span.SetStatus(codes.Ok, "")
	return order, nil
}
```

### HTTP における W3C Trace Context

Service A (order-service) から Service B (payment-service) への送信リクエスト:

```http
POST /api/v1/charge HTTP/1.1
Host: payment-service:8080
Content-Type: application/json
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
tracestate: order-svc=00f067aa0ba902b7

{"order_id": "ord_456", "amount": 99.00}
```

Service B からのレスポンス（同じトレースを継続）:

```http
HTTP/1.1 200 OK
Content-Type: application/json
traceresponse: 00-4bf92f3577b34da6a3ce929d0e0e4736-d5ebc7e161ade64a-01

{"status": "charged", "transaction_id": "txn_789"}
```

OTel SDK（または自動計装エージェント）は、インジェクションとエクストラクションを自動的に処理します。Kafka/gRPC では、同じ `traceparent` ヘッダーがメッセージヘッダーや gRPC メタデータを通じて伝播されます。

### メッセージキューにおけるコンテキスト伝播

```
Kafka Record Headers (order-service が生成):
  traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-a1b2c3d4e5f60718-01
  tracestate:  order-svc=a1b2c3d4e5f60718

Consumer (notification-service) がこれらのヘッダーを抽出 → 同じ trace_id で
子スパンを作成し、非同期フローをリンクします。
```

---

## サンプリング戦略

すべてをトレースするとコストがかかります。サンプリングによりオーバーヘッドを削減します。

### ヘッドベースサンプリング

トレース開始時に判定し、すべてのスパンに伝播します。

```
traceparent のフラグバイトがサンプリング判定を制御:

  サンプリング対象:     00-4bf92f35...-00f067aa...-01   ← flags=01 (sampled)
  サンプリング対象外:   00-4bf92f35...-00f067aa...-00   ← flags=00 (not sampled)

すべての下流サービスが親の判定を尊重します。
```

環境変数または OTel Collector の設定で構成します:

```yaml
# Application-side (env vars)
OTEL_TRACES_SAMPLER: parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG: "0.10"   # Sample 10% of root traces
```

### テールベースサンプリング

トレース完了後に、完全なトレースデータに基づいて判定します。

```
Collector がすべてのスパンを受信
         │
         ▼
┌─────────────────────┐
│  Tail-Based Sampler │
│                     │
│  ルール:            │
│  - 全エラーを保持   │
│  - 2秒超を保持      │
│  - 残りの1%を保持   │
└─────────────────────┘
         │
         ▼
    保存/破棄

メリット:
- 結果（エラー、レイテンシ）に基づいてサンプリング可能
- よりインテリジェントな判定

デメリット:
- 完全なトレースをバッファリングする必要がある
- リソース使用量が多い
- より複雑
```

### アダプティブサンプリング

```
アダプティブサンプリングは、目標のトレース/秒を維持するために
トラフィック量に基づいてレートを調整します。通常、アプリケーションコード
ではなく、OTel Collector のテールベースサンプリングで実装されます
（後述の Collector 構成を参照）。Jaeger のリモートサンプラーも
これをネイティブにサポートしています。

戦略:
  低トラフィック  → 100% サンプリング（すべてをキャプチャ）
  高トラフィック  → 目標 TPS に合わせて削減（例: 50 traces/sec）
  常に保持        → エラー、低速トレース（2秒超）
```

---

## トレーシングシステムのアーキテクチャ

### OpenTelemetry Collector

```
┌─────────────────────────────────────────────────────────────────┐
│                    Applications                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │Service A│  │Service B│  │Service C│  │Service D│            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                   │
└───────┼────────────┼────────────┼────────────┼───────────────────┘
        │            │            │            │
        └────────────┼────────────┼────────────┘
                     │            │
                     ▼            ▼
              ┌─────────────────────────┐
              │   OTel Collector        │
              │                         │
              │  ┌─────────────────┐    │
              │  │    Receivers    │    │  OTLP, Jaeger, Zipkin
              │  └────────┬────────┘    │
              │           │             │
              │  ┌────────▼────────┐    │
              │  │   Processors    │    │  Batch, Filter, Sample
              │  └────────┬────────┘    │
              │           │             │
              │  ┌────────▼────────┐    │
              │  │    Exporters    │    │  Jaeger, Tempo, X-Ray
              │  └─────────────────┘    │
              └───────────┬─────────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │  Jaeger │   │ Grafana │   │ AWS     │
      │         │   │  Tempo  │   │ X-Ray   │
      └─────────┘   └─────────┘   └─────────┘
```

### Collector の構成

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024

  # ヘルスチェックスパンを除外
  filter:
    spans:
      exclude:
        match_type: regexp
        attributes:
          - key: http.url
            value: .*/health.*

  # テールベースサンプリング
  tail_sampling:
    decision_wait: 10s
    policies:
      - name: errors
        type: status_code
        status_code: {status_codes: [ERROR]}
      - name: slow-traces
        type: latency
        latency: {threshold_ms: 2000}
      - name: percentage
        type: probabilistic
        probabilistic: {sampling_percentage: 10}

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true

  otlp:
    endpoint: tempo:4317

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, filter, tail_sampling]
      exporters: [jaeger, otlp]
```

---

## ベストプラクティス

### スパンの命名

```
悪い例 — 具体的すぎてカーディナリティが爆発する:
  "GET /users/8291037"             # 数百万のユニークなスパン名

良い例 — パラメータ化:
  "GET /users/{id}"

悪い例 — 汎用的すぎる:
  "database_query"

良い例 — 説明的:
  "SELECT users by id"
```

### 有用な属性

```
OTel セマンティック規約 — 標準的な属性キー:

HTTP スパン:
  http.method                    = "POST"
  http.url                       = "/api/orders"
  http.status_code               = 200
  http.request_content_length    = 1024

データベーススパン:
  db.system                      = "postgresql"
  db.name                        = "users"
  db.statement                   = "SELECT * FROM users WHERE id = ?"
  db.operation                   = "SELECT"

ビジネスコンテキスト（カスタム）:
  user.id                        = "123"
  order.id                       = "ord_456"
  tenant.id                      = "acme-corp"
```

### エラーハンドリング

```
スパンのエラーハンドリング戦略:

想定内のエラー（バリデーション、認証）:
  → スパンのステータスを ERROR に設定しメッセージを付与
  → 例外イベントは記録しない（ノイズを減らす）

想定外のエラー（パニック、インフラ障害）:
  → スパンのステータスを ERROR に設定
  → スタックトレース付きで例外イベントを記録
  → Jaeger/Tempo で赤いスパンとして表示される

スパンステータスの値:  Unset | Ok | Error
```

### ログとの相関

```
トレースコンテキストが埋め込まれた構造化ログ行:

{
  "timestamp": "2024-01-01T10:00:00.000Z",
  "level": "INFO",
  "message": "Processing order",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "service": "order-service"
}

これにより、Grafana/Loki からログ行を直接 Tempo/Jaeger のトレースビューにリンクできます。
OTel の自動計装は、trace_id/span_id をログに自動的に注入します。
```

---

## トレースの分析

### ボトルネックの発見

```
トレースタイムラインビュー:
├── API Gateway (合計: 2000ms)
│   ├── Auth (50ms) ✓
│   ├── Get User (1500ms) ← ボトルネック
│   │   ├── Cache Lookup (5ms)
│   │   ├── DB Query (1400ms) ← 根本原因
│   │   └── Serialize (10ms)
│   └── Send Response (50ms)

調査:
1. DB Query に 1400ms かかっている
2. db.statement 属性を確認
3. クエリ: SELECT * FROM users WHERE email = ?
4. email カラムのインデックスが欠落！
```

### トレースの比較

```
通常のトレース (200ms):                低速トレース (5000ms):
├── Service A (50ms)                 ├── Service A (50ms)
│   └── Cache HIT (5ms)              │   └── Cache MISS (5ms)
├── Service B (100ms)                ├── Service B (4800ms) ← 異なる
│   └── DB Query (80ms)              │   ├── DB Query (80ms)
└── Service C (50ms)                 │   └── Retry x3 (4500ms) ← リトライ！
                                     └── Service C (50ms)
```

### Jaeger / Tempo のクエリ例

```
Jaeger UI / API クエリ:

  # サービスとオペレーションでトレースを検索
  GET /api/traces?service=order-service&operation=process_order&limit=20

  # タグでトレースを検索
  GET /api/traces?service=order-service&tags={"http.status_code":"500"}

  # 2秒以上遅いトレースを検索
  GET /api/traces?service=order-service&minDuration=2s

  # 特定のトレース ID で検索
  GET /api/traces/4bf92f3577b34da6a3ce929d0e0e4736
```

```text
Grafana Tempo — TraceQL クエリ:

  # order-service の1秒超のエラースパンすべて
  {resource.service.name="order-service" && status=error && duration>1s}

  # 特定の属性を持つトレース
  {span.http.status_code=500}

  # order-service と payment-service の両方を経由するトレース
  {resource.service.name="order-service"} && {resource.service.name="payment-service"}
```

---

## トレードオフ

| 観点 | 考慮事項 |
|------|----------|
| オーバーヘッド | 1-5% のレイテンシ増加、ストレージコスト |
| サンプリング | 重要なトレースの見逃し vs. コスト |
| カーディナリティ | ユニークなタグが多すぎると高コスト |
| 完全性 | 計装されていないサービスがトレースを断絶させる |
| 複雑性 | 学習曲線、運用負荷 |

---

## 参考文献

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Google Dapper Paper](https://research.google/pubs/pub36356/)
- [Distributed Tracing in Practice](https://www.oreilly.com/library/view/distributed-tracing-in/9781492056621/)
