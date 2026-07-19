# SLOs and Error-Budget Control

## TL;DR

A service-level objective is a decision contract over a precisely defined service-level indicator. The indicator defines eligible events or intervals, good/bad classification, exclusions, measurement point, data source, and missing-data behavior. The objective defines target, window, population, and review policy. An error budget is the allowed bad fraction under that contract—not a universal number of outage minutes.

For target $S$, the allowed error fraction is $b=1-S$. Burn rate is observed error fraction divided by $b$. A multi-window alert requires both a long window that proves material budget consumption and a short window that proves the problem is still active. Choose burn thresholds from an explicit tolerated budget-consumption fraction and response horizon, not copied constants.

Error-budget policy coordinates release and reliability work; it does not excuse known safety, security, durability, or contractual failures. Keep measurement independent enough to survive the service incident, expose data coverage, aggregate counts rather than percentages, and test the SLI against real failure scenarios.

Metrics semantics live in [Metrics Systems and Monitoring](./02-metrics-monitoring.md), notification state in [Alert Evaluation and Notification](./04-alerting.md), and incident response in [Incident Command and Learning](./07-incident-management.md). This chapter owns SLI semantics, objective windows, burn math, and budget policy.

---

## SLO Workload and Contract

An SLO document contains:

~~~text
service/journey and accountable owner
users/tenants/regions/operations in scope
SLI good, bad, eligible, and excluded definitions
measurement point and source revision
target and objective window
latency thresholds or correctness/freshness semantics
missing/late/partial data policy
aggregation across operations, tenants, and regions
burn alert and ticket policy
error-budget governance and exceptions
review date and change history
~~~

### Invariants

1. The same event cannot be both good and bad; every eligible event is classified once.
2. Exclusions are narrow, measurable, and cannot be changed retroactively to improve performance.
3. Numerator and denominator come from compatible measurement populations and intervals.
4. Missing or stale measurement is explicit, never silently good.
5. Aggregation sums good/bad/eligible counts before computing ratios.
6. The measurement point reflects user experience and survives enough failure modes to remain trustworthy.
7. Objective changes create a new version; historical reports retain their original definition.
8. Burn calculations use the target and window of the same SLO revision.
9. Error-budget policy is predetermined and owned; incident pressure does not rewrite it ad hoc.
10. Safety, security, data-loss, and legal constraints remain hard boundaries regardless of available budget.

## Designing an SLI

### Event-based indicator

For eligible events $E$ and good events $G \subseteq E$:

$$
\text{SLI}_{\text{good}} = \frac{|G|}{|E|}
$$

and observed error fraction is:

$$
e = 1 - \text{SLI}_{\text{good}}
  = \frac{|E|-|G|}{|E|}.
$$

Examples:

- request succeeds under the public contract;
- request completes below a latency threshold;
- record is processed before freshness deadline;
- read returns a correct version;
- durable object remains retrievable.

Use counters or histogram bucket counts whose reset/gap semantics are correct. Do not compute an SLO from sampled traces unless the statistical estimator and inclusion probability are part of the contract.

### Time-based indicator

For systems where state over time is the user experience:

$$
\text{SLI}_{\text{time}}
=
\frac{\text{good eligible intervals}}
{\text{eligible intervals}}.
$$

Probe interval and failure confirmation shape the result. A one-minute probe cannot faithfully measure a brief outage, and a probe from one network does not represent every region. Define probe locations, cadence, quorum/aggregation, and missing probe behavior.

### Availability

Classify outcomes at a boundary users rely on:

~~~text
eligible:
  authorized supported operations that reach the service boundary

good:
  responses satisfying documented success/acceptable-domain outcome

bad:
  unexpected server failure, timeout, malformed success, or policy-defined overload

excluded:
  invalid/unauthorized calls only when they truly are outside the service promise
~~~

Be careful with load shedding: if a valid user request is rejected because the service lacks capacity, it is often bad for availability even if rejection protected the system.

### Latency

A latency SLI is a fraction under one or more thresholds, not a percentile average:

$$
\text{SLI}_{L \le \tau}
=
\frac{\text{eligible events with latency }L \le \tau}
{\text{eligible events}}.
$$

Histograms need a boundary at $\tau$ or a known approximation error. Separate interactive and batch operations when their expectations differ. Include failed requests in a deliberate way; measuring latency only for successful responses can make a failing fast service appear excellent.

### Freshness

For data item $i$ at observation time $t$:

$$
\text{age}_i(t) = t - \text{source-event-time}_i.
$$

Good means age is within the documented threshold for the eligible dataset/population. Queue depth is not freshness; a small queue can contain one very old item, and a large queue can be processing on time.

Track unknown source time, late event, and clock uncertainty. For periodic datasets, use deadline met/missed per expected partition or run.

### Correctness and quality

Correctness may require:

- invariant/checksum comparison;
- reconciliation against authoritative state;
- shadow/reference computation;
- sampled human or automated evaluation;
- schema/contract validity; or
- business outcome.

If only a sample is evaluated, report confidence/coverage and sampling bias. A correctness SLO cannot claim full-population precision from unrepresentative labels.

### Durability

Durability concerns rare loss over long horizons and is difficult to validate from ordinary request metrics. Use:

- acknowledged writes as eligible objects/bytes;
- verified retained/restorable objects;
- integrity scrubbing and repair;
- backup restore tests;
- deletion/tombstone semantics; and
- long-window loss accounting.

Do not infer durability from API availability.

## Denominator and Exclusion Design

Most misleading SLOs fail in the denominator.

### Denominator questions

1. Does it include retries or only logical user operations?
2. Does it include rejected overload?
3. Are client cancellations eligible?
4. Are unsupported/invalid calls excluded before service work?
5. Does zero traffic mean no evidence or perfect service?
6. Are background jobs represented by expected completions rather than attempts?
7. Are tenants weighted by traffic, equally, or under separate objectives?
8. Does failover traffic remain in the same population?
9. Can a failing component stop emitting the denominator?
10. Can a policy/config change alter classification mid-window?

### Exclusions

Valid exclusions may include explicitly unsupported traffic, authorized load tests in a separate population, or periods declared outside a contractual availability schedule. Exclusions are:

- defined before measurement;
- visible as separate counters;
- bounded by reason;
- versioned;
- reviewed for growth;
- applied identically to good and total calculations; and
- not based on “we later decided the outage was unfair.”

Maintain:

$$
E_{\text{all}}
=
E_{\text{eligible}}
+ E_{\text{excluded}}
+ E_{\text{unknown}}
$$

where each term is observable. Rising exclusions or unknowns can itself violate a measurement-health objective.

## Objectives and Windows

Let target success fraction be $S$. Allowed error fraction:

$$
b = 1-S.
$$

For $N$ eligible events in the objective window, nominal bad-event budget:

$$
B_{\text{events}} = bN.
$$

This is traffic-weighted. For a time-based SLO over eligible duration $T$:

$$
B_{\text{time}} = bT.
$$

Do not convert a request-based budget into outage minutes without a traffic model; one minute at peak and one minute at idle consume different event budgets.

### Rolling versus calendar

| Window | Strength | Limitation |
|---|---|---|
| Rolling | Always reflects recent service; smooth operational signal | Every time has a different membership; accounting can be less intuitive |
| Calendar | Aligns with business/reporting period | Boundary effects and end-of-period risk |
| Fixed release window | Useful for a migration/campaign | Not a permanent reliability contract |

Report objective-window boundaries and data watermark. Late corrections require a policy: update historical results with provenance, or freeze accounting at close and report adjustments separately.

### Low traffic

With small $N$, one error creates a large ratio and statistical uncertainty is high. Options:

- extend the window;
- combine a closely related population only if user semantics match;
- use time/probe or expected-event SLI;
- page on individual critical failures;
- set a minimum-event gate plus a separate no-traffic/freshness alert; and
- report exact counts alongside percentage.

Do not hide low-volume high-value operations inside a high-volume aggregate.

## Burn Rate Mathematics

Observed burn rate:

$$
r = \frac{e}{b}
=
\frac{\text{observed bad fraction}}
{\text{allowed bad fraction}}.
$$

Interpretation:

- $r=1$: budget is consumed at the sustainable rate for the objective window;
- $r>1$: continuing behavior exhausts budget before the window ends;
- $r<1$: current behavior is within the long-run allowance.

### Budget consumption in a subwindow

For objective-window duration $T_o$ and evaluation window $W$, sustained burn $r$ consumes approximately:

$$
c = r\frac{W}{T_o}
$$

of the full objective-window budget.

Therefore choose a burn threshold from policy:

$$
r_{\text{threshold}}
=
c_{\text{action}}\frac{T_o}{W},
$$

where $c_{\text{action}}$ is the fraction of the full budget that justifies the selected response if consumed within $W$.

This makes the alert reviewable: “page if this condition consumes the selected fraction within the long window,” instead of copying a magic multiplier.

### Multi-window confirmation

For long window $W_l$ and short window $W_s$:

~~~text
alert when:
  burn(W_l) >= r_threshold
  AND burn(W_s) >= r_threshold
  AND data coverage is sufficient
~~~

The long window proves meaningful budget consumption; the short window prevents paging after recovery. Use multiple severity pairs if policy needs fast catastrophic and slower sustained detection, but each pair derives from a stated consumption/action objective.

### Event-weighted implementation

For bad counter increase $\Delta B_W$ and eligible counter increase $\Delta E_W$:

$$
\text{burn}(W)
=
\frac{\Delta B_W}
{b\Delta E_W}.
$$

Aggregate counts across instances/regions first:

$$
\text{burn}_{\text{global}}
=
\frac{\sum_j \Delta B_{W,j}}
{b\sum_j \Delta E_{W,j}}.
$$

Averaging regional percentages gives a small region the same weight as a large one and can hide or exaggerate impact. Also keep per-region/tenant-tier objectives so global traffic weighting does not hide a severe isolated population.

## Error-Budget Policy

An error budget is a control signal for balancing change risk and reliability work. Policy is agreed by service, product, and reliability owners.

### Budget state

~~~mermaid
stateDiagram-v2
    [*] --> Healthy
    Healthy --> AtRisk: forecast/policy trigger
    AtRisk --> Healthy: sustained recovery and coverage
    AtRisk --> Exhausted: remaining budget <= policy boundary
    Healthy --> Exhausted: major incident
    Exhausted --> Recovering: mitigation and policy conditions met
    Recovering --> Healthy: new window/recovery criteria
    Recovering --> Exhausted: renewed burn
~~~

Exact boundaries are service policy, not universal constants.

### Possible policy actions

- review/reduce risky release cadence;
- require canary or stronger approval;
- prioritize top recurring reliability work;
- disable optional risky features;
- restrict known high-error traffic modes;
- require incident/postmortem follow-through;
- renegotiate an impossible objective with product evidence; or
- accept a time-bounded exception with owner and expiry.

Do not freeze:

- security fixes;
- mitigations that reduce current risk;
- required compliance changes; or
- all development indiscriminately.

Policy should improve expected reliability, not punish teams for honest measurement. Teams must not gain velocity by dropping telemetry or redefining exclusions.

### Budget allocation

A journey SLO may allocate internal risk budgets among components, but component budgets do not compose by simple addition unless failure independence and traffic paths are modeled.

For serial dependencies with independent availability $S_i$, approximate success:

$$
S_{\text{journey}} \approx \prod_i S_i.
$$

Independence often fails due to shared infrastructure and correlated incidents. Use the equation for planning, then validate with end-to-end measurement.

## Data Architecture and Coverage

Compute SLOs from durable, composable counters/histograms at the user-visible boundary. Preserve:

- good/bad/eligible/excluded/unknown counts;
- source and rule/schema revision;
- interval start/end and region;
- late/corrected data;
- data completeness/coverage;
- objective version; and
- burn recording-rule version.

Precompute per-window aggregates for alert latency and query cost, but retain base counts for audit/recalculation. A percentile dashboard value is not a sufficient source for a threshold SLO.

### Independent measurement

Use more than one failure domain where needed:

- edge/load-balancer request outcomes;
- service instrumentation;
- synthetic probes from representative locations;
- durable job/event state;
- data reconciliation.

No single source is perfect. Edge metrics see user boundary but may miss internal correctness; service metrics can disappear with the service; probes sample paths; client telemetry has bias/privacy issues. State the primary accounting source and secondary validation.

## Capacity and Cost Model

Assume:

- $J$ SLO journeys;
- $D_j$ retained dimensions for journey $j$ (region, operation, tenant tier);
- $K_j$ base counters/buckets per dimension;
- $f$ samples per second;
- $W$ recording/alert windows;
- $R$ objective/reporting retention seconds; and
- $\bar{b}$ stored bytes per sample after compression/index amortization.

Base SLO series:

$$
S_{\text{SLI}} = \sum_{j=1}^{J} D_j K_j.
$$

Raw retained storage:

$$
V_{\text{SLI}} \approx S_{\text{SLI}} f R \bar{b}.
$$

Recording rules can add approximately:

$$
S_{\text{recorded}} \propto S_{\text{SLI}} W
$$

depending on retained dimensions. Multi-window alerts multiply query/evaluation work, not user traffic.

### Explicit assumptions

- Exact tenant IDs are not a dimension unless per-tenant contractual SLOs justify cardinality.
- Good/bad counts are computed before sampling.
- Histogram bucket at the latency threshold exists.
- HA source replicas are deduplicated.
- Late data correction window is bounded.
- Global reports expose missing-region coverage.

Cost includes metric ingestion/storage, rule evaluation, reports, probes, and human governance. Keep SLO telemetry small, stable, and higher priority than broad diagnostic metrics.

## Security, Privacy, and Multi-Region Operations

SLOs influence release authority and contracts; protect rule and data integrity.

- authenticate/authorize objective and exclusion changes;
- require review for denominator or target changes;
- retain immutable version history;
- prevent services from self-labeling failures as excluded;
- keep tenant/customer-level reports access controlled;
- avoid personal identifiers in SLO labels;
- sign or audit exported contractual reports;
- separate measurement admin from service deploy authority where risk requires; and
- alert on missing/changed SLO instrumentation.

### Multi-region

Compute regional/cell SLOs locally for operational response, then aggregate base counts globally with coverage:

~~~text
region result:
  good, bad, eligible, excluded, unknown
  data interval and watermark
  objective/schema revision

global:
  sum compatible counts
  report absent/incompatible regions separately
~~~

During partition, regional burn alerts continue. Global budget accounting is provisional until coverage returns. Do not fill a missing region with its last healthy ratio.

Failover changes traffic mix and denominator. Preserve source/destination region and user population so the incident cannot appear as “region A recovered” merely because all traffic moved elsewhere.

## Failure Traces

### Denominator disappears during the outage

~~~text
service crashes -> in-process request counter stops
-> edge still returns failures but SLO query uses service total
-> bad numerator and total denominator both fall to zero
-> SLO reports no eligible requests and alert clears
~~~

**Controls:** measure primary availability at surviving edge, data-coverage/heartbeat objective, unknown-not-good semantics, and cross-source reconciliation.

### Percentile aggregation hides a region

~~~text
each region exports p99 gauge
-> dashboard averages regional p99 values
-> small fast regions pull average below threshold
-> large slow region violates most users’ latency promise
~~~

**Controls:** merge histogram counts across compatible buckets, compute threshold fraction/quantile after aggregation, retain per-region SLO.

### Retry attempts inflate success

~~~text
one user operation fails twice then succeeds
-> SLI counts three attempts: two bad, one good
-> transport policy change alters SLO without user outcome changing
~~~

**Controls:** define logical-operation versus attempt population, instrument both, use user-boundary outcome for journey SLO, and expose retry cost separately.

### Exclusion policy masks overload

~~~text
capacity limit rejects valid requests
-> team labels every rejection “client error” and excludes it
-> apparent availability improves as service sheds more users
~~~

**Controls:** centrally governed outcome taxonomy, valid-demand denominator, exclusion counters/review, and edge/service cross-check.

## Operating and Reviewing SLOs

Track:

- objective version, owner, target, window, and next review;
- good/bad/eligible/excluded/unknown counts;
- data watermark, source coverage, gap, reset, and correction;
- current burn by window and budget remaining/forecast;
- per-region/operation/tenant-tier distribution;
- alert state and incident mapping;
- exclusion/unknown growth;
- measurement source divergence;
- policy state, exception, and expiry; and
- reliability work tied to budget outcomes.

Review when product behavior, architecture, traffic, measurement, or contract changes—not only on a fixed calendar.

## Verification Strategy

| Test layer | What to prove |
|---|---|
| Classification fixtures | Every protocol/domain outcome maps to good, bad, excluded, or unknown exactly once |
| Denominator tests | Retry, cancel, overload, invalid request, zero traffic, and failover follow contract |
| Metric semantics | Counter resets/gaps and histogram thresholds aggregate correctly |
| Burn math | Synthetic constant error fractions produce expected burn and budget consumption |
| Window tests | Rolling/calendar boundary, late data, missed evaluation, and short/long confirmation |
| Coverage tests | Service/collector/region loss becomes unknown and cannot resolve an incident |
| Aggregation tests | Counts aggregate before ratios; small region/tenant isolation remains visible |
| Policy tests | Budget state, exception expiry, release action, and security-fix behavior |
| Replay | Historical incidents would alert at useful time without excessive quiet-period pages |
| Governance | Versioning, approvals, audit, and report reproduction |

Run game days that fail the measurement path separately from the service. An SLO system that only works while all telemetry is healthy cannot govern reliability.

## Decision Framework

1. Which user journey or durable outcome is being promised?
2. What exact events/intervals are eligible, good, bad, excluded, and unknown?
3. Where is the primary measurement point, and which failures can make it disappear?
4. Is the SLI request-, time-, threshold-, freshness-, correctness-, or durability-based?
5. What target and window reflect product need and realistic architecture?
6. What low-traffic and missing-data behavior applies?
7. Which dimensions need separate objectives so aggregation cannot hide harm?
8. Which consumption fraction and response horizon derive each burn alert?
9. What predetermined policy follows budget risk or exhaustion?
10. Which actions remain mandatory regardless of budget?
11. How do regional results aggregate with explicit coverage and failover population?
12. Can the result be reproduced from versioned base counts and objective definition?

## Key Takeaways

1. An SLO begins with eligible/good/bad/excluded/unknown semantics, not a target percentage.
2. Measure at a boundary that reflects users and survives the failure being measured.
3. Aggregate counts before computing ratios or quantiles.
4. Missing data is unknown, not perfect service.
5. Error budget is $1-S$ of the declared population/window, not automatically outage minutes.
6. Burn rate is observed error fraction divided by allowed error fraction.
7. Derive multi-window thresholds from desired budget consumption and response horizon.
8. Error-budget policy coordinates risk; it never overrides security, safety, or data integrity.
9. Regional SLOs continue locally while global accounting exposes incomplete coverage.
10. Validate classification, measurement failure, burn math, and historical incident response.

---

## References

- [Google SRE Workbook: Implementing SLOs](https://sre.google/workbook/implementing-slos/) — SLI selection, targets, windows, and error budgets
- [Google SRE Workbook: Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/) — burn rates and multi-window alert design
- [Google SRE Workbook: Example SLO Document](https://sre.google/workbook/slo-document/) — objective specification and ownership
- [Google SRE Workbook: Example Error Budget Policy](https://sre.google/workbook/error-budget-policy/) — predetermined reliability governance
- [OpenSLO Specification](https://github.com/OpenSLO/OpenSLO) — machine-readable service-level objective model
- [The Art of SLOs](https://sre.google/static/pdf/art-of-slos-handbook-a4.pdf) — practical SLO design and failure examples
- [Metrics Systems and Monitoring](./02-metrics-monitoring.md) — counters, histograms, aggregation, gaps, and storage
- [Alert Evaluation and Notification](./04-alerting.md) — alert state, missing data, grouping, routing, and HA
- [Incident Command and Learning](./07-incident-management.md) — mitigation, communication, and evidence during budget-burning incidents
