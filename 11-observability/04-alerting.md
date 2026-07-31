# Alert Evaluation and Notification

## TL;DR

Alerting is a stateful policy system that converts telemetry into human work. A rule evaluation is not a page: it produces alert instances; grouping combines related instances; inhibition suppresses symptoms under a known cause; silences are authorized time-bounded routing overrides; routing selects receivers and escalation; notification delivery has retries and deduplication.

Correctness depends on evaluation time, data freshness, missing-data semantics, rule revision, stable alert identity, and high-availability coordination. Two evaluators can legitimately compute the same firing state, but receivers should not page twice. Conversely, aggressive deduplication or inhibition must not hide independent incidents.

Page only for required human urgency and user/business impact. [SLOs and Error-Budget Control](./05-slos-error-budgets.md) covers burn-rate math; [Metrics Systems](./02-metrics-monitoring.md) covers signal semantics. This chapter covers alert state, grouping, inhibition, silence, routing, delivery, and human load.

---

## Alert Workload and Contract

An alert rule declares:

~~~text
rule identity and immutable revision
owner and service/tenant scope
signal query and evaluation interval
data window, freshness, and missing-data policy
condition and pending/confirmation semantics
alert identity labels
grouping and inhibition metadata
severity/urgency and required response
routing policy and escalation
runbook/dashboard/evidence links
maintenance and expiry policy
~~~

An alert instance is a stateful object keyed by a bounded fingerprint, typically rule ID plus the labels that identify one actionable failure domain. Raw error text, pod ID, request path, trace ID, and customer ID usually do not belong in the fingerprint.

### Invariants

1. The same rule revision, evaluation time, and input snapshot produce the same alert result.
2. Missing/stale input never silently becomes a numeric zero.
3. An alert fingerprint corresponds to one unit of human action.
4. Pending, firing, resolved, silenced, and inhibited remain distinct states.
5. Inhibition never alters the underlying alert’s evaluated state.
6. Silences are authorized, scoped, expiring, attributable, and auditable.
7. HA replicas may evaluate redundantly but notifications are deduplicated without losing all delivery.
8. Routing and contact configuration activate atomically under a versioned control plane.
9. Alert payloads expose only data permitted for the receiver.
10. Every page has an owner, response expectation, and retirement criteria.

## Evaluation State Machine

~~~mermaid
stateDiagram-v2
    [*] --> Inactive
    Inactive --> Pending: condition true
    Pending --> Inactive: condition false or policy-defined unknown
    Pending --> Firing: confirmation condition satisfied
    Firing --> Firing: condition remains true
    Firing --> Resolved: condition false
    Firing --> Firing: missing data under keep-firing policy
    Resolved --> Inactive: resolution notification processed
~~~

Silenced and inhibited are delivery annotations on Pending/Firing, not replacements for these states.

### Evaluation algorithm

At evaluation time $t_e$:

1. Pin the rule and routing policy revision.
2. Query the declared interval ending at $t_e$ or at an explicit data watermark.
3. Verify source coverage, freshness, and query completeness.
4. Normalize absent, stale, NaN, reset, and partial-region states.
5. Compute the condition under the rule’s typed semantics.
6. Load prior instance state by fingerprint.
7. Advance pending/firing/resolved state using elapsed evaluation time, not count of successful scheduler runs.
8. Persist new state and evidence/watermark.
9. Emit a state transition to notification processing.
10. Record evaluation duration, query cost, and failures.

If evaluations are missed, “three evaluations pending” is not the same as “condition persisted for three intervals.” Use timestamps and define how gaps affect continuity.

### Rule expression semantics

A rule query must define:

- population denominator and exclusions;
- aggregation dimensions;
- interval and alignment;
- handling of counter resets and late data;
- minimum traffic/sample sufficiency;
- regional/tenant coverage;
- comparison and hysteresis;
- no-data behavior; and
- whether the condition reflects symptom, risk, or cause.

Page primarily on user/business impact or an imminent hard limit with a known response. Cause signals such as CPU, queue, disk, or replica count are valuable diagnostic context and sometimes actionable predictors, but should not each page for the same incident.

## Missing and Stale Data

Missing data has several causes:

- the service has zero legitimate traffic;
- the instrument/resource no longer exists;
- collection or remote write failed;
- query timed out or returned partial regions;
- rule expression removed all series;
- producer schema changed;
- deployment is not yet emitting; or
- the monitoring system itself is down.

### Explicit policies

| Policy | Appropriate use | Risk |
|---|---|---|
| Treat as healthy | Optional population that truly has no work | Collection outage hides failure |
| Treat as failing | Required heartbeat/freshness signal | Planned inactivity pages |
| Keep previous state for bounded time | Short telemetry gaps | Stale firing/healthy state persists |
| Mark unknown and route separately | Most ambiguous infrastructure | Needs an independent meta-alert path |

For ratio rules, zero denominator is unknown unless the contract explicitly defines it. A missing numerator must not automatically become zero while the denominator remains.

### Dead-man signals

A continuously expected heartbeat evaluated by an independent path detects pipeline disappearance. It should traverse as much of the production alert path as possible, and an external receiver verifies its arrival. One internal rule cannot prove that its own notification transport works.

Track the data watermark and coverage alongside every alert. A “resolved” transition caused by lost telemetry should be labeled unknown, not celebrated as recovery.

## Alert Identity and Grouping

### Fingerprint

Choose labels that answer “would one responder action resolve all instances with this fingerprint?”

Common bounded identity:

~~~text
rule
service or user journey
environment
region/cell when mitigation differs
tenant tier only when response differs
severity/urgency
~~~

Exclude volatile replica/pod labels from service-level pages; include them in evidence. For node-specific hardware action, node can be the actionable identity.

Changing identity labels during a rule rollout can create one resolved old alert and one firing new alert. Migrate with shadow evaluation and explicit notification suppression.

### Grouping

Grouping batches related firing instances into one notification:

- service + alert family;
- region/cell;
- incident correlation key; or
- receiver/owner.

The group wait trades immediate notification for consolidation. The repeat interval trades reminder against fatigue. These are policy-derived from response urgency, not universal constants.

Group payloads have size/member limits. A storm group summarizes counts and top failure domains with a query link rather than embedding thousands of instances.

## Inhibition, Silences, and Routing

### Inhibition

Inhibition suppresses notification for an alert when a designated parent/cause alert is firing and labels match a declared relationship:

~~~text
region connectivity alert firing for region=A
  inhibits service dependency symptoms where region=A
~~~

Safe inhibition requires:

- parent alert is at least as urgent and routes to a responsible team;
- matching labels prove the same failure domain;
- child evaluated state remains visible;
- child can notify if parent resolves while child remains;
- maximum inhibition scope/duration; and
- tests for independent simultaneous failures.

Do not inhibit every downstream service merely because one dependency alert exists; partial routing or an unrelated service bug may coexist.

### Silences

A silence is an operator-created matcher set with:

- creator identity and authorization;
- reason/change/incident reference;
- exact bounded matchers;
- start and expiry;
- affected receivers/severity;
- review for broad or long scope; and
- audit of creation, update, early expiry, and matches.

Silence does not delete evidence or alert state. Maintenance should preferably be a versioned planned policy generated from the change system, avoiding forgotten manual muting.

### Routing

Routing maps normalized alert labels to:

- team/rotation;
- delivery channels;
- escalation stages;
- language/region;
- notification template and permitted fields;
- business-hours versus immediate behavior; and
- fallback receiver.

Every firing alert must match exactly one owned primary route or an explicit fallback that is itself monitored. Ambiguous overlapping routes and route-to-no-receiver are configuration errors.

Templates treat alert annotations as untrusted strings. Escape markup/links and prevent secrets or personal data from reaching broad chat/email receivers.

## Notification Delivery and HA

### Delivery state

~~~mermaid
stateDiagram-v2
    [*] --> TransitionQueued
    TransitionQueued --> GroupWaiting
    GroupWaiting --> Ready
    Ready --> Sending
    Sending --> Delivered
    Sending --> Retrying: transient/ambiguous failure
    Retrying --> Sending
    Retrying --> Escalated: stage deadline
    Sending --> FailedPermanent
    Delivered --> Acknowledged: human/system acknowledgement
~~~

Define “delivered”: accepted by provider, delivered to device, acknowledged by a human, and incident opened are different.

Notification attempts use a stable idempotency key:

~~~text
alert group fingerprint
state transition generation
receiver and escalation stage
routing policy revision
~~~

Provider retry after an ambiguous response may duplicate. Receivers and incident systems deduplicate this key.

### HA evaluation

Two common approaches:

- **active/active evaluation:** replicas evaluate all rules and downstream grouping/dedup removes duplicate transitions;
- **partitioned ownership with failover:** one replica/lease owns a shard, requiring fencing and fast takeover.

Active/active is simpler and tolerates evaluator loss, but duplicate queries and notifications must be controlled. Partitioning reduces cost but lease/split-brain correctness becomes critical.

Replicas need not synchronize every Pending state if their evaluations are deterministic and notification dedup works, but clock, data view, and rule revision divergence can produce different transitions. Report evaluation and active policy revision per replica.

### HA notification

A cluster can gossip/replicate notification logs, but a regional partition may cause each side to notify. Decide whether duplicate paging during partition is preferable to no page (usually yes for high urgency), then deduplicate in the incident system when connectivity returns.

Never put all notification channels behind one provider or network path for critical pages. Maintain a tested fallback with independent credentials and routing.

## Human Load and Policy Quality

Human attention is the scarce resource. Measure:

- pages per on-call hour/shift;
- unique incidents versus notifications;
- acknowledgements and escalations;
- actionable pages;
- pages requiring no action;
- duplicate/stale/resolution-only noise;
- time-to-acknowledge and time-to-mitigation;
- after-hours interruption;
- alerts without owner/runbook;
- silences and inhibition duration; and
- recurring alert families.

Do not optimize acknowledgement time alone; responders can acknowledge quickly without understanding. Tie alert review to incident outcome and qualitative feedback.

### Page, ticket, or dashboard

| Response | Channel |
|---|---|
| Human action required now to prevent/mitigate material impact | page |
| Action required within a business deadline | owned ticket/work queue |
| Trend or diagnostic context with no discrete action | dashboard/report |
| Expected automated recovery within budget | record/metric, not human notification |

An alert that never changes an operator decision is telemetry, not a page.

## Capacity and Cost Model

Assume:

- $R$ rules;
- each rule evaluates every $I_r$ seconds;
- $P_r$ is average samples/series scanned per evaluation;
- $A$ active alert instances;
- $G$ notification groups;
- $\bar{m}$ average instances per group;
- $d$ average delivery attempts per group transition; and
- $c$ configured receivers per group.

Evaluation rate:

$$
\lambda_{\text{eval}} = \sum_{r=1}^{R}\frac{1}{I_r}.
$$

Approximate query scan work:

$$
Q_{\text{scan}} =
\sum_{r=1}^{R}\frac{P_r}{I_r}.
$$

Notification attempt rate during a transition burst:

$$
\lambda_{\text{notify}}
\approx
\lambda_{\text{group-transition}} d c.
$$

State memory/storage scales with rule revisions, active fingerprints, pending history, notification log, silences, and inhibition indexes, not only rule count.

### Exceptional load

- one label mistake creates an alert per request/customer/pod;
- telemetry replay reevaluates stale windows;
- region outage fires every service rule;
- routing outage retries every notification;
- config rollout changes fingerprints;
- silence expires across a storm;
- evaluator recovery catches up missed intervals; and
- global and regional rules both page.

Bound alert instances per rule/tenant, notification group members, payload bytes, route fan-out, retry queue, and concurrent rule queries. Preserve high-urgency user-impact groups under overload and summarize the rest.

Human capacity is also bounded. If $N_p$ pages arrive in period $T$ and each needs $\bar{t}$ minutes of attention, required responder time is:

$$
H_{\text{attention}} = \frac{N_p \bar{t}}{60}
$$

person-hours, before incident work. A policy that routinely exceeds staffed attention is unsafe even if the notification system can deliver it.

## Security, Privacy, and Multi-Region Operations

Alerts may contain tenant names, customer impact, vulnerabilities, internal links, and personal on-call data.

- authenticate rule/silence/routing changes;
- separate author from approver for broad critical silences;
- scope receivers to authorized fields;
- store contact endpoints and provider credentials as secrets;
- redact query annotations and generated summaries;
- audit bulk alert and silence access;
- prevent external labels from choosing receiver/template; and
- rate-limit attacker-triggerable alert instances.

### Regional architecture

Evaluate region-local impact from regional metrics so WAN failure cannot blind the affected region. Global journey/SLO alerts consume explicit regional coverage and deduplicate related incidents.

During region partition:

- regional pages continue to regional/on-duty responders;
- global evaluator marks partial data;
- notification logs may diverge;
- incident IDs reconcile when connectivity returns;
- silences and emergency routing have defined regional authority; and
- routing/control policy uses bounded last-known-good with expiry.

A global silence must not rely on propagation faster than the incident it is meant to suppress. Show active silence revision by region.

## Failure Traces

### Missing telemetry resolves an outage

~~~text
service fails -> exporter/metrics path also fails
-> error series disappears
-> rule treats absence as zero errors
-> firing alert transitions to resolved
-> responders stand down while users still fail
~~~

**Controls:** explicit unknown policy, denominator/traffic heartbeat, data freshness/coverage in rule, external probe, and resolution hold until valid recovery evidence.

### HA replicas page twice

~~~text
notification cluster partitions
-> both evaluators see the same firing transition
-> each side lacks the other’s notification log
-> both escalate through every receiver
~~~

**Controls:** stable idempotency key at incident/receiver, preferred regional authority where safe, partition-mode marker, bounded repeats, and reconciliation.

### Inhibition hides an independent failure

~~~text
database alert fires in region A
-> broad inhibition suppresses all service alerts globally
-> unrelated auth outage in region B produces no page
~~~

**Controls:** exact failure-domain label matching, scope tests, visible inhibited state, maximum duration, and child notification after parent recovery.

### Cardinality turns one incident into thousands of pages

~~~text
new rule fingerprints on pod and raw path
-> deployment plus errors create thousands of instances
-> grouping payloads overflow and provider throttles
-> important page is delayed behind noise
~~~

**Controls:** compile-time label budget, maximum instances/group size, service-level fingerprint, priority queues, storm summary, and config rollback.

## Operating the Alert System

Track:

- rule evaluation success, duration, missed intervals, data watermark, and active revision;
- inactive/pending/firing/resolved/unknown instances;
- fingerprints and new-instance rate by bounded rule/service;
- inhibited/silenced counts and oldest/expiry;
- groups, members, wait/repeat, payload truncation, and transitions;
- notification queue age, attempt, provider acceptance, acknowledgement, escalation, and failure;
- HA replica divergence, dedup hit, partition mode, and route revision;
- unmatched/ambiguous routes and fallback usage;
- page volume, duplicate/no-action rate, and responder load; and
- external dead-man signal delivery.

Alert on the alerting system through an independent path where possible.

## Verification Strategy

| Test layer | What to prove |
|---|---|
| Rule semantics | Known time-series fixtures produce correct condition and evidence |
| State-machine tests | Pending, firing, gap, recovery, resolve, and rule migration |
| Missing-data tests | Zero traffic, scrape loss, partial region, query error, stale data, and schema removal |
| Fingerprint tests | Bounded stable identity and migration without duplicate incidents |
| Group/inhibition tests | Correct matching, storm size, independent failures, and parent recovery |
| Silence tests | Authorization, matcher scope, expiry, region propagation, and audit |
| Routing tests | Exactly one owner/fallback, template escaping, redaction, and receiver failover |
| HA tests | Evaluator crash, clock skew, split brain, notification partition, and dedup |
| Load tests | Region outage, alert cardinality explosion, silence expiry, and provider throttle |
| Human review | Every page maps to a concrete action and post-incident outcome |

Replay historical incidents and quiet periods through candidate rules. Compare pages, time-to-first-page, duplicate groups, missing incidents, and human attention, not only expression truth.

## Decision Framework

1. What human decision or mitigation does this alert request?
2. Is the signal user/business impact, imminent risk, or merely diagnostic cause?
3. Which metric/log/probe semantics and data coverage does it require?
4. What does missing, stale, partial, reset, and zero traffic mean?
5. What fingerprint equals one unit of action?
6. How long must the condition persist, and how are missed evaluations handled?
7. Which related alerts group, and which exact parent-child relation inhibits?
8. Who may silence it, at what scope and expiry?
9. Which team/region/channel owns primary and fallback delivery?
10. How do HA replicas deduplicate without creating a single point of loss?
11. What storm and human-attention budgets apply?
12. Which incident replay proves the candidate policy is better?

## Key Takeaways

1. Alerting is a stateful policy and delivery system, not a threshold in a dashboard.
2. Missing data is a typed state and can be more dangerous than a high value.
3. A fingerprint should equal one actionable failure domain.
4. Grouping combines notifications; inhibition suppresses related symptoms; silence is an authorized override.
5. Inhibited and silenced alerts remain evaluated and visible.
6. HA evaluation may be redundant, but notification transitions need stable idempotency.
7. Regional evaluation protects outage visibility; global alerts expose coverage.
8. Notification acceptance, delivery, acknowledgement, and incident creation are different states.
9. Human attention has capacity and must be measured like compute.
10. Validate rules by replaying incidents, gaps, partitions, and quiet periods.

---

## References

- [Prometheus Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/): pending/firing state and rule evaluation
- [Prometheus Alertmanager](https://prometheus.io/docs/alerting/latest/alertmanager/): grouping, inhibition, silences, routing, and HA deduplication
- [Google SRE Workbook: Monitoring](https://sre.google/workbook/monitoring/): monitoring strategy and signal design
- [Google SRE Workbook: Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/): symptom-oriented multi-window alert behavior
- [OASIS Common Alerting Protocol](https://docs.oasis-open.org/emergency/cap/v1.2/CAP-v1.2-os.html): interoperable alert message structure
- [Metrics Systems and Monitoring](./02-metrics-monitoring.md): instrument, aggregation, missing series, and query behavior
- [SLOs and Error-Budget Control](./05-slos-error-budgets.md): burn math and reliability policy
- [Incident Command and Learning](./07-incident-management.md): page-to-incident transition, command, communications, and mitigation
