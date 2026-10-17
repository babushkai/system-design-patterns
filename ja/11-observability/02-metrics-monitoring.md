# メトリクスとモニタリング

> **注記:** このドキュメントは英語版からの翻訳です。最新の内容や正確な情報については、[英語版オリジナル](../../11-observability/02-metrics-monitoring.md)を参照してください。

## TL;DR

メトリクスは、時間とともに収集される数値的な測定値です。メトリクスは集約データを通じて「何が起きているか？」に答えます。これはログ（何が起きたか？）やトレース（どのように流れたか？）とは異なります。RED メソッド（Rate, Errors, Duration）と USE メソッド（Utilization, Saturation, Errors）は、包括的なモニタリングのためのフレームワークを提供します。

---

## メトリクス vs. ログ vs. トレース

```
                    メトリクス            ログ                 トレース
─────────────────────────────────────────────────────────────────────────
質問                何が起きている？     何が起きた？         どう流れた？
データ型            数値                 テキスト             スパン
カーディナリティ    低                   高                   中
ストレージコスト    低                   高                   中
クエリパターン      集約                 検索                 ID検索
例                  error_rate=0.02      "User 123 failed"    リクエストパス

必要な場面:
- アラート          ✓ 主要               時々                 まれ
- ダッシュボード    ✓ 主要               時々                 ✓
- デバッグ          時々                 ✓ 主要               ✓ 主要
- キャパシティ      ✓ 主要               まれ                 時々
```

---

## メトリクスの種類

### カウンター

累積値で、増加のみ（またはゼロにリセット）します。

```
http_requests_total

Time:    T0      T1      T2      T3      T4
Value:   100     150     225     310     400
Delta:   -       +50     +75     +85     +90

用途:
- リクエスト数
- 転送バイト数
- 完了タスク数
- 発生エラー数
```

```
# TYPE http_requests_total counter
# HELP http_requests_total Total HTTP requests.
http_requests_total{method="GET",endpoint="/api/users",status="200"} 1027
http_requests_total{method="POST",endpoint="/api/orders",status="201"} 563
http_requests_total{method="GET",endpoint="/api/users",status="500"} 12
```

### ゲージ

上下に変動する値です。

```
active_connections

Time:    T0      T1      T2      T3      T4
Value:   10      25      15      30      20

用途:
- 現在の接続数
- キュー深度
- 温度
- メモリ使用量
- アクティブユーザー数
```

```
# TYPE active_connections gauge
# HELP active_connections Current active connections.
active_connections{service="api"} 42
active_connections{service="worker"} 17
```

### ヒストグラム

設定可能なバケット全体での値の分布です。

```
http_request_duration_seconds

Buckets: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, +Inf]

Observations: 0.02, 0.03, 0.08, 0.15, 0.5, 2.0, 0.04

バケットカウント:
  le="0.01":  0
  le="0.05":  3  (0.02, 0.03, 0.04)
  le="0.1":   4  (+ 0.08)
  le="0.5":   6  (+ 0.15, 0.5)
  le="1.0":   6
  le="5.0":   7  (+ 2.0)
  le="+Inf":  7

可能にすること:
- パーセンタイル計算 (p50, p95, p99)
- 分布分析
- SLO トラッキング
```

```
# TYPE http_request_duration_seconds histogram
# HELP http_request_duration_seconds HTTP request duration in seconds.
http_request_duration_seconds_bucket{method="GET",endpoint="/api",le="0.01"} 0
http_request_duration_seconds_bucket{method="GET",endpoint="/api",le="0.05"} 3
http_request_duration_seconds_bucket{method="GET",endpoint="/api",le="0.1"} 4
http_request_duration_seconds_bucket{method="GET",endpoint="/api",le="0.5"} 6
http_request_duration_seconds_bucket{method="GET",endpoint="/api",le="1.0"} 6
http_request_duration_seconds_bucket{method="GET",endpoint="/api",le="5.0"} 7
http_request_duration_seconds_bucket{method="GET",endpoint="/api",le="+Inf"} 7
http_request_duration_seconds_sum{method="GET",endpoint="/api"} 2.82
http_request_duration_seconds_count{method="GET",endpoint="/api"} 7
```

### サマリー

ヒストグラムに似ていますが、クライアント側で分位数を計算します。

```
# TYPE http_request_duration_seconds summary
# HELP http_request_duration_seconds HTTP request duration.
http_request_duration_seconds{method="GET",quantile="0.5"} 0.042
http_request_duration_seconds{method="GET",quantile="0.9"} 0.15
http_request_duration_seconds{method="GET",quantile="0.99"} 0.48
http_request_duration_seconds_sum{method="GET"} 8734.29
http_request_duration_seconds_count{method="GET"} 51432
```

**ヒストグラム vs. サマリー:**

| 観点 | ヒストグラム | サマリー |
|------|------------|---------|
| 集約 | ✓ 集約可能 | ✗ 集約不可 |
| 分位数の精度 | 近似 | 正確 |
| サーバー負荷 | 低い | 高い |
| バケット設定 | 事前定義が必要 | 不要 |
| 推奨 | ヒストグラムを使用 | ほとんど不要 |

---

## 命名規則

### Prometheus の命名ベストプラクティス

```
Format: <namespace>_<subsystem>_<name>_<unit>

良い例:
  http_requests_total                    # カウンター — _total サフィックスを使用
  http_request_duration_seconds          # ヒストグラム — 基本単位（ミリ秒ではなく秒）
  process_memory_bytes                   # ゲージ — 基本単位（MB ではなくバイト）
  database_connections_active            # ゲージ

悪い例:
  requests                               # 曖昧すぎる
  http_requests_count                    # カウンターには _total を使用
  requestDurationMilliseconds            # フォーマットと単位が間違い
  HttpRequestDuration                    # ケースが間違い
```

### ラベルのベストプラクティス

```
良い例 — 低カーディナリティ:
  http_requests_total{method="GET", status="200", endpoint="/api/users"}

悪い例 — 高カーディナリティ（上限なし）:
  http_requests_total{user_id="12345"}   # 数百万のユニーク値！
  http_requests_total{request_id="..."}  # リクエストごとにユニーク！

目安:
  ユニークラベルの組み合わせ < 10,000
  各ラベル値のユニーク値 < 100
```

---

## RED メソッド（リクエスト駆動）

サービス（API、マイクロサービス）向け:

```
R - Rate:     リクエストスループット（リクエスト/秒）
E - Errors:   失敗リクエスト（件数またはレート）
D - Duration: レスポンスタイム分布

ダッシュボードレイアウト:
┌─────────────────────────────────────────────────────────────┐
│  Service: user-api                                          │
│                                                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ │
│  │ Request Rate    │ │ Error Rate      │ │ Latency p99   │ │
│  │     523/s       │ │     0.02%       │ │     245ms     │ │
│  │     ↑ 12%       │ │     ↓ 50%       │ │     ↑ 15%     │ │
│  └─────────────────┘ └─────────────────┘ └───────────────┘ │
│                                                             │
│  [リクエストレートの時系列グラフ]                             │
│  [レイテンシ分布ヒートマップ]                                │
│  [エンドポイント別エラーレート]                               │
└─────────────────────────────────────────────────────────────┘
```

### Prometheus クエリ (PromQL)

```text
# Rate: 1秒あたりのリクエスト数
rate(http_requests_total[5m])

# エンドポイント別レート
sum by (endpoint) (rate(http_requests_total[5m]))

# Errors: エラーレート（パーセンテージ）
sum(rate(http_requests_total{status=~"5.."}[5m]))
/
sum(rate(http_requests_total[5m])) * 100

# Duration: p99 レイテンシ
histogram_quantile(0.99,
  sum by (le) (rate(http_request_duration_seconds_bucket[5m]))
)

# Duration: 平均レイテンシ
sum(rate(http_request_duration_seconds_sum[5m]))
/
sum(rate(http_request_duration_seconds_count[5m]))
```

---

## USE メソッド（リソース指向）

リソース（CPU、メモリ、ディスク、ネットワーク）向け:

```
U - Utilization: リソースキャパシティの使用率
S - Saturation:  キューイング/バックログの度合い
E - Errors:      エラーイベント

リソース分析:
┌──────────────┬─────────────────┬─────────────────┬─────────────────┐
│   リソース    │   使用率         │   飽和度         │     エラー       │
├──────────────┼─────────────────┼─────────────────┼─────────────────┤
│ CPU          │ ビジー率 %       │ 実行キュー深度   │ -               │
│ メモリ       │ 使用率 %         │ スワップ使用量   │ OOM キル        │
│ ディスク I/O │ ビジー時間 %     │ キュー長         │ I/O エラー      │
│ ネットワーク │ 帯域幅 %         │ ソケットバックログ│ パケットエラー   │
│ コネクション │ プール使用率 %   │ 待ちキュー       │ タイムアウト     │
└──────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### 主要リソースメトリクス

```text
# CPU
# 使用率: CPU使用率パーセンテージ
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# 飽和度: ロードアベレージ / CPU数
node_load1 / count by (instance) (node_cpu_seconds_total{mode="idle"})

# メモリ
# 使用率
(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100

# 飽和度（スワップ使用量はメモリ圧迫を示す）
node_memory_SwapTotal_bytes - node_memory_SwapFree_bytes

# ディスク
# 使用率
rate(node_disk_io_time_seconds_total[5m]) * 100

# 飽和度
node_disk_io_time_weighted_seconds_total

# ネットワーク
# 使用率（インターフェース速度の把握が必要）
rate(node_network_receive_bytes_total[5m]) * 8 / 1e9  # Gbps
```

---

## ゴールデンシグナル (Google SRE)

```
Latency:     リクエストの処理時間（成功 vs. 失敗）
Traffic:     システムへの需要（リクエスト/秒、トランザクション/秒）
Errors:      失敗リクエストのレート
Saturation:  サービスの「埋まり具合」（キャパシティ使用率）

RED/USE との関係:
├── Latency   ≈ RED Duration
├── Traffic   ≈ RED Rate
├── Errors    ≈ RED Errors / USE Errors
└── Saturation ≈ USE Saturation
```

---

## 計装パターン

### Go による計装の例

```go
package main

import (
	"net/http"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/metric"
)

var meter = otel.Meter("api-server")

var (
	requestCount, _    = meter.Int64Counter("http_requests_total",
		metric.WithDescription("Total HTTP requests"))
	requestDuration, _ = meter.Float64Histogram("http_request_duration_seconds",
		metric.WithDescription("HTTP request duration in seconds"))
	inFlight, _        = meter.Int64UpDownCounter("http_requests_in_progress",
		metric.WithDescription("Requests currently being processed"))
)

func metricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attrs := metric.WithAttributes(
			attribute.String("method", r.Method),
			attribute.String("endpoint", r.URL.Path),
		)
		inFlight.Add(r.Context(), 1, attrs)
		start := time.Now()

		rw := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(rw, r)

		duration := time.Since(start).Seconds()
		requestCount.Add(r.Context(), 1, metric.WithAttributes(
			attribute.String("method", r.Method),
			attribute.String("endpoint", r.URL.Path),
			attribute.Int("status", rw.statusCode),
		))
		requestDuration.Record(r.Context(), duration, attrs)
		inFlight.Add(r.Context(), -1, attrs)
	})
}
```

### OpenTelemetry Collector パイプライン（メトリクス）

```yaml
# otel-collector-config.yaml
receivers:
  prometheus:
    config:
      scrape_configs:
        - job_name: "api-server"
          scrape_interval: 15s
          static_configs:
            - targets: ["api:8080"]
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  batch:
    timeout: 5s
    send_batch_size: 1024
  memory_limiter:
    check_interval: 1s
    limit_mib: 512

exporters:
  prometheusremotewrite:
    endpoint: "http://mimir:9009/api/v1/push"
  prometheus:
    endpoint: 0.0.0.0:8889

service:
  pipelines:
    metrics:
      receivers: [otlp, prometheus]
      processors: [memory_limiter, batch]
      exporters: [prometheusremotewrite, prometheus]
```

### ビジネスメトリクス

```
# TYPE orders_total counter
# HELP orders_total Total orders processed.
orders_total{status="completed",payment_method="credit_card"} 4521
orders_total{status="completed",payment_method="paypal"} 1203
orders_total{status="refunded",payment_method="credit_card"} 47

# TYPE order_value_dollars histogram
# HELP order_value_dollars Order value in dollars.
order_value_dollars_bucket{le="50"} 1200
order_value_dollars_bucket{le="100"} 3100
order_value_dollars_bucket{le="500"} 5400
order_value_dollars_bucket{le="+Inf"} 5771

# TYPE active_users gauge
# HELP active_users Currently active users.
active_users 342
```

---

## アラート

### アラート設計の原則

```yaml
# 良いアラートの特徴:
# - アクション可能: 誰かが何かをする必要がある
# - 関連性: 実際のユーザー影響を示す
# - 具体的: 何が問題かが明確
# - タイムリー: 過敏すぎず遅すぎない

# 悪い例: 通常動作の症状にアラート
- alert: HighCPU
  expr: cpu_usage > 70    # 繁忙サーバーでは正常

# 良い例: ユーザー影響にアラート
- alert: HighErrorRate
  expr: |
    sum(rate(http_requests_total{status=~"5.."}[5m]))
    / sum(rate(http_requests_total[5m])) > 0.01
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High error rate (> 1%)"
    description: "Error rate is {{ $value | humanizePercentage }}"
```

### SLO ベースのアラート

```yaml
# エラーバジェットの消費が速すぎる場合にアラート

# SLO: 99.9% 可用性 = 0.1% エラーバジェット
# 30日バジェット = 43.2分のエラー許容

# マルチウィンドウ、マルチバーンレートアラート
groups:
- name: slo-alerts
  rules:
  # 高速バーン: 14.4倍のバーンレートで1時間 = 月間バジェットの2%
  - alert: HighBurnRate_Fast
    expr: |
      (
        sum(rate(http_requests_total{status=~"5.."}[1h]))
        / sum(rate(http_requests_total[1h]))
      ) > (14.4 * 0.001)
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Burning error budget 14x too fast"

  # 低速バーン: 3倍のバーンレートで6時間 = 月間バジェットの3%
  - alert: HighBurnRate_Slow
    expr: |
      (
        sum(rate(http_requests_total{status=~"5.."}[6h]))
        / sum(rate(http_requests_total[6h]))
      ) > (3 * 0.001)
    for: 15m
    labels:
      severity: warning
```

### アラートルーティング

```yaml
# alertmanager.yml
route:
  receiver: 'default'
  group_by: ['alertname', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
  - match:
      severity: critical
    receiver: 'pagerduty-critical'
    continue: true
  - match:
      severity: warning
    receiver: 'slack-warnings'
  - match:
      team: database
    receiver: 'database-team'

receivers:
- name: 'pagerduty-critical'
  pagerduty_configs:
  - service_key: '<key>'

- name: 'slack-warnings'
  slack_configs:
  - channel: '#alerts'
    send_resolved: true
```

---

## ダッシュボード設計

### レイアウトの原則

```
┌─────────────────────────────────────────────────────────────────────┐
│  サービス概要ダッシュボード                                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  上段: 主要指標（現在の状態を一目で把握）                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Requests │ │  Errors  │ │ Latency  │ │ Success  │ │Saturation│  │
│  │  1.2k/s  │ │   0.1%   │ │  42ms    │ │  99.9%   │ │   34%    │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                                     │
│  中段: 時系列（トレンドとパターン）                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  リクエストレート & エラーレート（重ね表示）                    │   │
│  │  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁                                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  レイテンシ分布（ヒートマップまたはパーセンタイル）             │   │
│  │  p99: ████████████████████                                  │   │
│  │  p95: ████████████                                          │   │
│  │  p50: █████                                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  下段: 内訳（ドリルダウン詳細）                                       │
│  ┌────────────────────────┐ ┌────────────────────────────────┐     │
│  │ エンドポイント別エラー  │ │ エンドポイント別レイテンシ       │     │
│  │ /api/orders    45%     │ │ /api/search     250ms          │     │
│  │ /api/users     30%     │ │ /api/orders     120ms          │     │
│  │ /api/products  25%     │ │ /api/users       45ms          │     │
│  └────────────────────────┘ └────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### 必須ダッシュボード

```
1. サービス概要
   - 各サービスの RED メトリクス
   - 依存関係のステータス
   - 最近のデプロイメントをマーク

2. インフラストラクチャ
   - すべてのリソースの USE メトリクス
   - クラスターヘルス
   - ノードステータス

3. ビジネス KPI
   - 注文/収益
   - アクティブユーザー
   - コンバージョンレート

4. SLO ダッシュボード
   - 現在のエラーバジェット
   - バーンレート
   - 過去の SLO パフォーマンス
```

---

## アンチパターン

### カーディナリティの爆発

```
悪い例 — 上限のないラベル値:
  http_requests_total{user_id="8291037"}      # 数百万のユーザー
  http_requests_total{request_id="a1b2c3..."}  # リクエストごとにユニーク
  http_requests_total{timestamp="1710504000"}  # 無限

影響:
  - Prometheus のメモリ枯渇
  - クエリパフォーマンスの低下
  - ストレージコストの爆発

良い例 — 有界で低カーディナリティのラベル:
  http_requests_total{user_tier="pro",endpoint="/api/users",status_class="2xx"}
  # user_tier:    free, pro, enterprise  (3値)
  # endpoint:     約100ルート
  # status_class: 2xx, 3xx, 4xx, 5xx    (4値)
```

### 簡単なものを監視し、重要なものを監視しない

```
悪い例（計測が簡単）:
- CPU 使用率
- メモリ使用率
- アップタイム

これらはユーザーが満足しているかを教えてくれません！

良い例（重要なもの）:
- リクエスト成功率
- リクエストレイテンシ（ユーザー体感）
- タイプ別エラーレート
- ビジネストランザクション/秒
```

---

## 参考文献

- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [RED Method](https://grafana.com/blog/2018/08/02/the-red-method-how-to-instrument-your-services/)
- [USE Method](https://www.brendangregg.com/usemethod.html)
- [Practical Monitoring](https://www.oreilly.com/library/view/practical-monitoring/9781491957349/)
