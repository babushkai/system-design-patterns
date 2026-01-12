# Alerting

## TL;DR

Good alerts are actionable, relevant, and timely. Alert on symptoms (user impact) not causes (high CPU). Use SLO-based alerting to balance reliability with development velocity. Every alert should either wake someone up or be deleted.

---

## The Problem with Bad Alerting

### Alert Fatigue

```
Monday 2:00 AM: "CPU > 80% on web-server-1"
Monday 2:15 AM: "CPU > 80% on web-server-2"  
Monday 2:30 AM: "Memory > 70% on db-server"
Monday 3:00 AM: "Disk > 60% on log-server"
...

On-call engineer: *mutes all alerts, goes back to sleep*

Tuesday: Actual outage, nobody notices because alerts are noise

Result: 
- Alert fatigue → ignored alerts
- Burnout → high turnover
- Incidents → missed real problems
```

### The Golden Rule

> Every alert should be actionable. If you can't take action, don't alert.

```
Questions for every alert:
1. Does this require immediate human action?
2. Is the action clear?
3. Will this fire at 3 AM?
4. Is the threshold meaningful?

If any answer is "no" → reconsider the alert
```

---

## Alert on Symptoms, Not Causes

### Symptoms vs. Causes

```
Causes (don't alert):              Symptoms (do alert):
─────────────────────              ────────────────────
High CPU usage        ────────►   Slow response times
High memory usage     ────────►   Errors returned to users
Full disk            ────────►   Failed transactions
Network packet loss   ────────►   Timeouts
Pod restart          ────────►   Service unavailability

Users don't care about CPU.
Users care that the website is slow.
```

### Example Transformation

```yaml
# BAD: Cause-based alert
- alert: HighCPU
  expr: cpu_usage > 80
  labels:
    severity: warning
  annotations:
    summary: "High CPU usage"

# Problem: CPU can be 90% and everything is fine
# Problem: CPU can be 50% but app is broken

# GOOD: Symptom-based alert  
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

## SLO-Based Alerting

### The Error Budget Model

```
SLO: 99.9% availability per month

Error Budget = 100% - 99.9% = 0.1%
In 30 days: 30 * 24 * 60 * 0.001 = 43.2 minutes of errors allowed

Budget consumption:
┌────────────────────────────────────────────────────────────────┐
│                        30-day error budget                      │
│                                                                 │
│ Day 1-10: ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 10% used (4.3 min)   │
│ Day 10-15: █████░░░░░░░░░░░░░░░░░░░░░░░░░ 15% used (2.2 min)   │
│ Day 15-20: ████████░░░░░░░░░░░░░░░░░░░░░░ 25% used (4.3 min)   │
│ Day 20-25: ████████████████░░░░░░░░░░░░░░ 50% used (10.8 min)  │
│ Incident:  █████████████████████████████░░ 90% used (17.3 min) │
│                                                                 │
│ Remaining budget: 4.3 minutes for rest of month                │
└────────────────────────────────────────────────────────────────┘
```

### Burn Rate

```
Burn rate = rate of error budget consumption

Burn rate 1.0 = Using budget exactly as planned
Burn rate 2.0 = Using budget 2x too fast (budget gone in 15 days)
Burn rate 36 = Using budget 36x too fast (budget gone in 20 hours)

Why burn rate matters:
- Burn rate 1 at 3 AM → Not urgent, can wait until morning
- Burn rate 10 at 3 AM → Wake someone up now
```

### Multi-Window, Multi-Burn-Rate Alerts

```yaml
# Recommended by Google SRE
# Different windows catch different problem types

# Window 1: Fast burn (5% budget in 1 hour)
# Catches: Major incidents, total outages
- alert: ErrorBudget_FastBurn
  expr: |
    (
      # 1-hour error rate
      sum(rate(http_requests_total{status=~"5.."}[1h]))
      / sum(rate(http_requests_total[1h]))
    ) > (14.4 * 0.001)  # 14.4x burn rate = 5% budget/hour
  for: 2m
  labels:
    severity: critical
    
# Window 2: Slow burn (10% budget in 6 hours)  
# Catches: Gradual degradation, partial failures
- alert: ErrorBudget_SlowBurn
  expr: |
    (
      # 6-hour error rate
      sum(rate(http_requests_total{status=~"5.."}[6h]))
      / sum(rate(http_requests_total[6h]))
    ) > (6 * 0.001)  # 6x burn rate = 10% budget/6 hours
  for: 15m
  labels:
    severity: warning

# Short window confirms (prevents alert on brief spike that recovered)
# Long window shows sustained issue (worth alerting on)
```

### SLO Alert Design

```
┌───────────────────────────────────────────────────────────────────┐
│            SLO-Based Alert Matrix                                  │
├──────────────┬──────────────┬──────────────┬─────────────────────┤
│ Burn Rate    │ Time Window  │ Budget Consumed │ Severity         │
├──────────────┼──────────────┼──────────────┼─────────────────────┤
│ 14.4x        │ 1 hour       │ 2% / hour      │ Page immediately  │
│ 6x           │ 6 hours      │ 5% / 6 hours   │ Page during hours │
│ 3x           │ 1 day        │ 10% / day      │ Ticket            │
│ 1x           │ 3 days       │ 10% / 3 days   │ Review            │
└──────────────┴──────────────┴──────────────┴─────────────────────┘

Detection time vs. budget consumed trade-off:
- Fast detection = more sensitive = more false positives
- Slow detection = less budget consumed before alert
```

---

## Alert Design Best Practices

### Essential Alert Components

```yaml
- alert: PaymentServiceErrors
  # 1. Clear, specific name
  
  expr: |
    sum(rate(http_requests_total{service="payment",status=~"5.."}[5m]))
    / sum(rate(http_requests_total{service="payment"}[5m])) > 0.01
  # 2. Meaningful threshold based on SLO/business impact
  
  for: 5m
  # 3. Duration to prevent flapping
  
  labels:
    severity: critical
    team: payments
    service: payment-service
  # 4. Labels for routing and grouping
  
  annotations:
    summary: "Payment service error rate > 1%"
    description: |
      Error rate: {{ $value | humanizePercentage }}
      This may indicate payment gateway issues or database problems.
    runbook: "https://wiki.internal/runbooks/payment-errors"
    dashboard: "https://grafana/d/payments"
  # 5. Context for responders
```

### Runbook Template

```markdown
# Payment Service High Error Rate

## Alert Meaning
Payment API returning >1% errors to users.

## Impact
- Users cannot complete purchases
- Revenue impact: ~$X per minute of outage

## Investigation Steps
1. Check payment gateway status: https://status.stripe.com
2. Check database connectivity: 
   `kubectl logs -l app=payment -c app | grep -i database`
3. Check recent deployments:
   `kubectl rollout history deployment/payment`
4. Check dependent services:
   - User service: https://grafana/d/user-service
   - Inventory service: https://grafana/d/inventory

## Remediation
- If gateway down: Enable backup gateway (see: /docs/failover)
- If database: Failover to replica (see: /docs/db-failover)
- If bad deploy: `kubectl rollout undo deployment/payment`

## Escalation
- Level 1: #payments-oncall
- Level 2: @payments-lead
- Level 3: @engineering-manager
```

---

## Alert Routing and Notification

### Alertmanager Configuration

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/xxx'

route:
  receiver: 'default'
  group_by: ['alertname', 'service']
  group_wait: 30s        # Wait to group related alerts
  group_interval: 5m     # Time between grouped notifications
  repeat_interval: 4h    # Re-notify if not resolved
  
  routes:
    # Critical → PagerDuty immediately
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true  # Also send to Slack
      
    # Warnings → Slack during business hours only
    - match:
        severity: warning
      receiver: 'slack-warnings'
      mute_time_intervals:
        - nights-and-weekends
        
    # Route by team
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

# Silence overnight and weekends for non-critical
time_intervals:
  - name: nights-and-weekends
    time_intervals:
      - weekdays: ['saturday', 'sunday']
      - times:
          - start_time: '22:00'
            end_time: '08:00'
```

### Alert Grouping

```
Without grouping:
Alert: HighLatency - service=api, endpoint=/users
Alert: HighLatency - service=api, endpoint=/orders  
Alert: HighLatency - service=api, endpoint=/products
Alert: HighLatency - service=api, endpoint=/cart
→ 4 separate pages at 3 AM

With grouping (group_by: [alertname, service]):
Alert: HighLatency (4 endpoints affected)
  - /users
  - /orders
  - /products
  - /cart
→ 1 page with full context
```

---

## Reducing Alert Noise

### Deduplication

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
        
        # New alert
        if not last_state:
            self.redis.setex(key, 86400, FIRING)
            return True
        
        # State change
        if last_state.decode() != alert.state:
            self.redis.setex(key, 86400, alert.state)
            return True
        
        # Same state, already notified
        return False
```

### Inhibition Rules

```yaml
# Suppress downstream alerts when upstream is firing
inhibit_rules:
  # If database is down, don't alert on services that depend on it
  - source_match:
      alertname: 'DatabaseDown'
    target_match:
      dependency: 'database'
    equal: ['environment']
    
  # If cluster is unhealthy, don't alert on individual pods
  - source_match:
      alertname: 'KubernetesClusterUnhealthy'
    target_match_re:
      alertname: 'Pod.*'
    equal: ['cluster']
```

### Silences

```bash
# Create a silence for maintenance
amtool silence add \
  --alertmanager.url=http://alertmanager:9093 \
  --author="jane@example.com" \
  --comment="Planned database maintenance" \
  --duration="2h" \
  'service=database'

# Query active silences
amtool silence query

# Expire a silence early
amtool silence expire <silence-id>
```

---

## On-Call Best Practices

### Rotation Structure

```
Primary On-Call     Secondary On-Call
     │                     │
     │ Gets paged first    │ Escalation after 15 min
     │                     │
     ▼                     ▼
┌─────────┐           ┌─────────┐
│  Week 1 │ Alice     │ Alice   │ Bob
│  Week 2 │ Bob       │ Bob     │ Carol
│  Week 3 │ Carol     │ Carol   │ Alice
└─────────┘           └─────────┘

Escalation path:
1. Primary (0-15 min)
2. Secondary (15-30 min)
3. Team Lead (30-45 min)
4. Engineering Manager (45+ min)
```

### Incident Response

```
1. ACKNOWLEDGE
   - Acknowledge the page within 5 minutes
   - This stops escalation, shows you're working on it

2. ASSESS
   - Check dashboards and runbook
   - Determine scope and impact
   - Decide if you need help

3. COMMUNICATE
   - Update status page if customer-facing
   - Notify stakeholders if significant
   - Post updates every 15-30 minutes

4. MITIGATE
   - Focus on restoring service first
   - Root cause can wait until stable
   - "Rollback first, ask questions later"

5. RESOLVE
   - Confirm service restored
   - Close incident
   - Schedule postmortem if significant
```

### Page Hygiene

```
Track and review:
┌────────────────────────────────────────────────────────────────┐
│  Weekly On-Call Report                                          │
├─────────────────────────────────────────────────────────────────┤
│  Total pages: 12                                                │
│  After-hours: 4 (target: < 2)                                   │
│  Actionable: 8 (67%)                                            │
│  Time to acknowledge: 3.2 min avg                               │
│  Time to resolve: 45 min avg                                    │
│                                                                 │
│  Top alerts:                                                    │
│  1. HighLatency - 4 times (investigate threshold)               │
│  2. DiskSpace - 3 times (add auto-cleanup)                      │
│  3. HighErrorRate - 2 times (legitimate issues)                 │
│                                                                 │
│  Action items:                                                  │
│  - Tune HighLatency threshold (too sensitive)                   │
│  - Automate disk cleanup to prevent DiskSpace alerts            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Alerting Anti-Patterns

### 1. Alert on Everything

```yaml
# BAD: Alerts that aren't actionable
- alert: CPUHigh
  expr: cpu > 50  # What should I do about this?

- alert: PodsNotRunning
  expr: kube_pod_status_phase{phase!="Running"} > 0
  # Pods restart normally during deployments

- alert: AnyError
  expr: increase(errors_total[1m]) > 0
  # Some errors are expected
```

### 2. Wrong Thresholds

```yaml
# BAD: Arbitrary thresholds
- alert: HighMemory
  expr: memory_usage > 70  # Why 70? Based on what?

# GOOD: Threshold based on actual limits
- alert: HighMemory
  expr: |
    container_memory_usage_bytes 
    / container_spec_memory_limit_bytes > 0.9
  # 90% of actual limit, leaves 10% headroom
```

### 3. Missing "for" Duration

```yaml
# BAD: Alerts on momentary spikes
- alert: HighLatency
  expr: latency_p99 > 500
  # Will fire on any brief spike

# GOOD: Sustained issue only
- alert: HighLatency
  expr: latency_p99 > 500
  for: 5m  # Must persist for 5 minutes
```

### 4. No Runbook

```yaml
# BAD: Alert without guidance
- alert: DatabaseReplicationLag
  expr: replication_lag > 10

# GOOD: Includes runbook
- alert: DatabaseReplicationLag
  expr: replication_lag > 10
  annotations:
    runbook: https://wiki/runbooks/db-replication-lag
```

---

## Monitoring the Monitors

### Alerting Health Metrics

```promql
# Alertmanager health
up{job="alertmanager"} == 1

# Alert delivery success rate
rate(alertmanager_notifications_total{status="success"}[5m])
/ rate(alertmanager_notifications_total[5m])

# Time from alert to notification
histogram_quantile(0.99, alertmanager_notification_latency_seconds_bucket)

# Number of active alerts
ALERTS{alertstate="firing"}
```

### Dead Man's Switch

```yaml
# "Watchdog" alert that always fires
# If it stops firing, monitoring is broken
- alert: Watchdog
  expr: vector(1)
  labels:
    severity: none
  annotations:
    summary: "Alerting pipeline health check"

# External service (like Deadman's Snitch) expects this alert
# If not received, external service alerts you
```

---

## References

- [Google SRE Book - Alerting](https://sre.google/sre-book/alerting-on-slos/)
- [My Philosophy on Alerting](https://docs.google.com/document/d/199PqyG3UsyXlwieHaqbGiWVa8eMWi8zzAn0YfcApr8Q/view)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [PagerDuty Incident Response](https://response.pagerduty.com/)
- [Atlassian Incident Management](https://www.atlassian.com/incident-management)
