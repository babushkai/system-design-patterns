# アラート

> **注記:** このドキュメントは英語版からの翻訳です。最新の内容や正確な情報については、[英語版オリジナル](../../11-observability/04-alerting.md)を参照してください。

## TL;DR

良いアラートは、アクション可能で、関連性があり、タイムリーです。原因（高 CPU）ではなく症状（ユーザー影響）にアラートを設定しましょう。SLO ベースのアラートは、信頼性と開発速度のバランスを取ります。すべてのアラートは、誰かを起こすか、削除されるべきです。

---

## 悪いアラートの問題

### アラート疲れ

```
月曜日 2:00 AM: "CPU > 80% on web-server-1"
月曜日 2:15 AM: "CPU > 80% on web-server-2"
月曜日 2:30 AM: "Memory > 70% on db-server"
月曜日 3:00 AM: "Disk > 60% on log-server"
...

オンコールエンジニア: *全アラートをミュートして二度寝*

火曜日: 実際の障害発生、アラートがノイズなので誰も気づかない

結果:
- アラート疲れ → アラートが無視される
- バーンアウト → 離職率上昇
- インシデント → 実際の問題の見落とし
```

### ゴールデンルール

> すべてのアラートはアクション可能であるべきです。アクションを取れないなら、アラートにしないでください。

```
すべてのアラートに対する質問:
1. 即座に人間のアクションが必要か？
2. アクションは明確か？
3. 午前3時に発火するか？
4. 閾値は意味があるか？

いずれかの答えが「いいえ」→ アラートを再検討
```

---

## 症状にアラートし、原因にアラートしない

### 症状 vs. 原因

```
原因（アラートしない）:              症状（アラートする）:
─────────────────────              ────────────────────
高 CPU 使用率         ────────►   レスポンスタイムの低下
高メモリ使用率        ────────►   ユーザーに返されるエラー
ディスクフル          ────────►   トランザクションの失敗
ネットワークパケットロス ────────►   タイムアウト
Pod の再起動          ────────►   サービスの利用不可

ユーザーは CPU を気にしません。
ユーザーはウェブサイトが遅いことを気にします。
```

### 変換の例

```yaml
# 悪い例: 原因ベースのアラート
- alert: HighCPU
  expr: cpu_usage > 80
  labels:
    severity: warning
  annotations:
    summary: "High CPU usage"

# 問題: CPU が 90% でも問題ないことがある
# 問題: CPU が 50% でもアプリが壊れていることがある

# 良い例: 症状ベースのアラート
- alert: HighErrorRate
  expr: |
    sum(rate(http_requests_total{status=~"5.."}[5m]))
    / sum(rate(http_requests_total[5m])) > 0.01
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Error rate > 1%"
    runbook: "https://wiki/runbooks/high-error-rate"
```

---

## SLO ベースのアラート

### エラーバジェットモデル

```
SLO: 月間 99.9% の可用性

エラーバジェット = 100% - 99.9% = 0.1%
30日間: 30 * 24 * 60 * 0.001 = 43.2分のエラーが許容

バジェット消費:
┌────────────────────────────────────────────────────────────────┐
│                        30日間のエラーバジェット                   │
│                                                                 │
│ 1-10日目:  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 10% 使用 (4.3分)    │
│ 10-15日目: █████░░░░░░░░░░░░░░░░░░░░░░░░░ 15% 使用 (2.2分)     │
│ 15-20日目: ████████░░░░░░░░░░░░░░░░░░░░░░ 25% 使用 (4.3分)     │
│ 20-25日目: ████████████████░░░░░░░░░░░░░░ 50% 使用 (10.8分)    │
│ インシデント: █████████████████████████████░░ 90% 使用 (17.3分) │
│                                                                 │
│ 残りのバジェット: 月末まで 4.3分                                  │
└────────────────────────────────────────────────────────────────┘
```

### バーンレート

```
バーンレート = エラーバジェットの消費速度

バーンレート 1.0 = 計画通りにバジェットを使用
バーンレート 2.0 = 2倍の速さで使用（15日でバジェット枯渇）
バーンレート 36  = 36倍の速さで使用（20時間でバジェット枯渇）

バーンレートが重要な理由:
- 午前3時にバーンレート 1 → 緊急ではない、朝まで待てる
- 午前3時にバーンレート 10 → 今すぐ誰かを起こす
```

### マルチウィンドウ・マルチバーンレートアラート

```yaml
# Google SRE が推奨
# 異なるウィンドウで異なる問題タイプを検出

# ウィンドウ 1: 高速バーン（1時間でバジェットの5%）
# 検出対象: 大規模インシデント、完全なサービス停止
- alert: ErrorBudget_FastBurn
  expr: |
    (
      # 1時間のエラーレート
      sum(rate(http_requests_total{status=~"5.."}[1h]))
      / sum(rate(http_requests_total[1h]))
    ) > (14.4 * 0.001)  # 14.4倍のバーンレート = バジェットの5%/時間
  for: 2m
  labels:
    severity: critical

# ウィンドウ 2: 低速バーン（6時間でバジェットの10%）
# 検出対象: 緩やかな劣化、部分的な障害
- alert: ErrorBudget_SlowBurn
  expr: |
    (
      # 6時間のエラーレート
      sum(rate(http_requests_total{status=~"5.."}[6h]))
      / sum(rate(http_requests_total[6h]))
    ) > (6 * 0.001)  # 6倍のバーンレート = バジェットの10%/6時間
  for: 15m
  labels:
    severity: warning

# 短いウィンドウで確認（回復した一時的なスパイクでのアラートを防止）
# 長いウィンドウで持続的な問題を表示（アラートに値する）
```

### SLO アラート設計

```
┌───────────────────────────────────────────────────────────────────┐
│            SLO ベースアラートマトリクス                               │
├──────────────┬──────────────┬──────────────┬─────────────────────┤
│ バーンレート  │ 時間ウィンドウ│ バジェット消費  │ 重大度             │
├──────────────┼──────────────┼──────────────┼─────────────────────┤
│ 14.4倍       │ 1時間        │ 2%/時間       │ 即座にページ        │
│ 6倍          │ 6時間        │ 5%/6時間      │ 営業時間中にページ  │
│ 3倍          │ 1日          │ 10%/日        │ チケット            │
│ 1倍          │ 3日          │ 10%/3日       │ レビュー            │
└──────────────┴──────────────┴──────────────┴─────────────────────┘

検出時間 vs. バジェット消費のトレードオフ:
- 高速検出 = より敏感 = 偽陽性が多い
- 低速検出 = アラート前のバジェット消費が少ない
```

---

## アラート設計のベストプラクティス

### 必須アラートコンポーネント

```yaml
- alert: PaymentServiceErrors
  # 1. 明確で具体的な名前

  expr: |
    sum(rate(http_requests_total{service="payment",status=~"5.."}[5m]))
    / sum(rate(http_requests_total{service="payment"}[5m])) > 0.01
  # 2. SLO/ビジネスインパクトに基づいた意味のある閾値

  for: 5m
  # 3. フラッピング防止のための持続時間

  labels:
    severity: critical
    team: payments
    service: payment-service
  # 4. ルーティングとグルーピング用のラベル

  annotations:
    summary: "Payment service error rate > 1%"
    description: |
      Error rate: {{ $value | humanizePercentage }}
      This may indicate payment gateway issues or database problems.
    runbook: "https://wiki.internal/runbooks/payment-errors"
    dashboard: "https://grafana/d/payments"
  # 5. 対応者向けのコンテキスト
```

### ランブックテンプレート

```markdown
# Payment Service High Error Rate

## アラートの意味
Payment API がユーザーに 1% 以上のエラーを返しています。

## 影響
- ユーザーが購入を完了できない
- 収益への影響: 障害1分あたり約 $X

## 調査手順
1. ペイメントゲートウェイのステータスを確認: https://status.stripe.com
2. データベース接続を確認:
   `kubectl logs -l app=payment -c app | grep -i database`
3. 最近のデプロイメントを確認:
   `kubectl rollout history deployment/payment`
4. 依存サービスを確認:
   - User service: https://grafana/d/user-service
   - Inventory service: https://grafana/d/inventory

## 修復手順
- ゲートウェイ障害の場合: バックアップゲートウェイを有効化（参照: /docs/failover）
- データベースの場合: レプリカへフェイルオーバー（参照: /docs/db-failover）
- 不良デプロイの場合: `kubectl rollout undo deployment/payment`

## エスカレーション
- Level 1: #payments-oncall
- Level 2: @payments-lead
- Level 3: @engineering-manager
```

---

## アラートルーティングと通知

### Alertmanager の構成

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/xxx'

route:
  receiver: 'default'
  group_by: ['alertname', 'service']
  group_wait: 30s        # 関連アラートのグルーピングを待機
  group_interval: 5m     # グループ通知の間隔
  repeat_interval: 4h    # 未解決時の再通知間隔

  routes:
    # Critical → 即座に PagerDuty
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true  # Slack にも送信

    # Warning → 営業時間中のみ Slack
    - match:
        severity: warning
      receiver: 'slack-warnings'
      mute_time_intervals:
        - nights-and-weekends

    # チーム別ルーティング
    - match:
        team: database
      receiver: 'database-team-pagerduty'

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#alerts'

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '<integration-key>'
        severity: critical
        description: '{{ .CommonAnnotations.summary }}'

  - name: 'slack-warnings'
    slack_configs:
      - channel: '#alerts-warnings'
        send_resolved: true
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'

# 非クリティカルアラートの夜間・週末サイレンス
time_intervals:
  - name: nights-and-weekends
    time_intervals:
      - weekdays: ['saturday', 'sunday']
      - times:
          - start_time: '22:00'
            end_time: '08:00'
```

### アラートグルーピング

```
グルーピングなし:
Alert: HighLatency - service=api, endpoint=/users
Alert: HighLatency - service=api, endpoint=/orders
Alert: HighLatency - service=api, endpoint=/products
Alert: HighLatency - service=api, endpoint=/cart
→ 午前3時に4回のページ

グルーピングあり (group_by: [alertname, service]):
Alert: HighLatency (4つのエンドポイントが影響)
  - /users
  - /orders
  - /products
  - /cart
→ 完全なコンテキスト付きの1回のページ
```

---

## アラートノイズの削減

### 重複排除

```python
# Alert states
FIRING = "firing"
RESOLVED = "resolved"

class AlertDeduplicator:
    def __init__(self, redis):
        self.redis = redis

    def should_notify(self, alert):
        key = f"alert:{alert.fingerprint}"
        last_state = self.redis.get(key)

        # 新規アラート
        if not last_state:
            self.redis.setex(key, 86400, FIRING)
            return True

        # 状態変化
        if last_state.decode() != alert.state:
            self.redis.setex(key, 86400, alert.state)
            return True

        # 同じ状態、通知済み
        return False
```

### 抑制ルール

```yaml
# 上流が発火している場合、下流のアラートを抑制
inhibit_rules:
  # データベースがダウンしている場合、依存サービスのアラートを出さない
  - source_match:
      alertname: 'DatabaseDown'
    target_match:
      dependency: 'database'
    equal: ['environment']

  # クラスターが異常な場合、個別 Pod のアラートを出さない
  - source_match:
      alertname: 'KubernetesClusterUnhealthy'
    target_match_re:
      alertname: 'Pod.*'
    equal: ['cluster']
```

### サイレンス

```bash
# メンテナンス用のサイレンスを作成
amtool silence add \
  --alertmanager.url=http://alertmanager:9093 \
  --author="jane@example.com" \
  --comment="Planned database maintenance" \
  --duration="2h" \
  'service=database'

# アクティブなサイレンスを照会
amtool silence query

# サイレンスを早期に期限切れにする
amtool silence expire <silence-id>
```

---

## オンコールのベストプラクティス

### ローテーション体制

```
プライマリオンコール     セカンダリオンコール
     │                     │
     │ 最初にページされる    │ 15分後にエスカレーション
     │                     │
     ▼                     ▼
┌─────────┐           ┌─────────┐
│  第1週  │ Alice     │ Alice   │ Bob
│  第2週  │ Bob       │ Bob     │ Carol
│  第3週  │ Carol     │ Carol   │ Alice
└─────────┘           └─────────┘

エスカレーションパス:
1. プライマリ (0-15分)
2. セカンダリ (15-30分)
3. チームリード (30-45分)
4. エンジニアリングマネージャー (45分以上)
```

### インシデントレスポンス

```
1. 確認（ACKNOWLEDGE）
   - 5分以内にページを確認
   - エスカレーションが止まり、対応中であることを示す

2. 評価（ASSESS）
   - ダッシュボードとランブックを確認
   - スコープと影響を判断
   - 助けが必要かを判断

3. コミュニケーション（COMMUNICATE）
   - 顧客影響がある場合はステータスページを更新
   - 重大な場合はステークホルダーに通知
   - 15-30分ごとに更新を投稿

4. 軽減（MITIGATE）
   - まずサービスの復旧に集中
   - 根本原因は安定後に調査
   - 「まずロールバック、質問は後」

5. 解決（RESOLVE）
   - サービスの復旧を確認
   - インシデントをクローズ
   - 重大な場合はポストモーテムをスケジュール
```

### ページの衛生管理

```
追跡とレビュー:
┌────────────────────────────────────────────────────────────────┐
│  週次オンコールレポート                                          │
├─────────────────────────────────────────────────────────────────┤
│  総ページ数: 12                                                 │
│  営業時間外: 4 (目標: < 2)                                      │
│  アクション可能: 8 (67%)                                        │
│  確認までの時間: 平均 3.2分                                      │
│  解決までの時間: 平均 45分                                       │
│                                                                 │
│  上位アラート:                                                  │
│  1. HighLatency - 4回 (閾値を調査)                              │
│  2. DiskSpace - 3回 (自動クリーンアップを追加)                    │
│  3. HighErrorRate - 2回 (正当な問題)                            │
│                                                                 │
│  アクションアイテム:                                             │
│  - HighLatency の閾値を調整（過敏すぎる）                         │
│  - ディスククリーンアップを自動化して DiskSpace アラートを防止      │
└─────────────────────────────────────────────────────────────────┘
```

---

## アラートのアンチパターン

### 1. すべてにアラートする

```yaml
# 悪い例: アクション不能なアラート
- alert: CPUHigh
  expr: cpu > 50  # これに対して何をすればいい？

- alert: PodsNotRunning
  expr: kube_pod_status_phase{phase!="Running"} > 0
  # デプロイメント中は Pod は通常再起動する

- alert: AnyError
  expr: increase(errors_total[1m]) > 0
  # 一部のエラーは想定内
```

### 2. 不適切な閾値

```yaml
# 悪い例: 恣意的な閾値
- alert: HighMemory
  expr: memory_usage > 70  # なぜ 70？ 何に基づいて？

# 良い例: 実際の制限に基づく閾値
- alert: HighMemory
  expr: |
    container_memory_usage_bytes
    / container_spec_memory_limit_bytes > 0.9
  # 実際の制限の 90%、10% のヘッドルームを確保
```

### 3. "for" 持続時間の欠如

```yaml
# 悪い例: 一瞬のスパイクでアラート
- alert: HighLatency
  expr: latency_p99 > 500
  # 短時間のスパイクで発火する

# 良い例: 持続的な問題のみ
- alert: HighLatency
  expr: latency_p99 > 500
  for: 5m  # 5分間持続する必要がある
```

### 4. ランブックなし

```yaml
# 悪い例: ガイダンスのないアラート
- alert: DatabaseReplicationLag
  expr: replication_lag > 10

# 良い例: ランブック付き
- alert: DatabaseReplicationLag
  expr: replication_lag > 10
  annotations:
    runbook: https://wiki/runbooks/db-replication-lag
```

---

## 監視の監視

### アラートヘルスメトリクス

```text
# Alertmanager のヘルス
up{job="alertmanager"} == 1

# アラート配信の成功率
rate(alertmanager_notifications_total{status="success"}[5m])
/ rate(alertmanager_notifications_total[5m])

# アラートから通知までの時間
histogram_quantile(0.99, alertmanager_notification_latency_seconds_bucket)

# アクティブなアラートの数
ALERTS{alertstate="firing"}
```

### デッドマンズスイッチ

```yaml
# 常に発火する「Watchdog」アラート
# 発火が止まったら監視が壊れている
- alert: Watchdog
  expr: vector(1)
  labels:
    severity: none
  annotations:
    summary: "Alerting pipeline health check"

# 外部サービス（Deadman's Snitch など）がこのアラートを期待
# 受信しない場合、外部サービスがアラートを発報
```

---

## 参考文献

- [Google SRE Book - Alerting](https://sre.google/sre-book/alerting-on-slos/)
- [My Philosophy on Alerting](https://docs.google.com/document/d/199PqyG3UsyXlwieHaqbGiWVa8eMWi8zzAn0YfcApr8Q/view)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [PagerDuty Incident Response](https://response.pagerduty.com/)
- [Atlassian Incident Management](https://www.atlassian.com/incident-management)
