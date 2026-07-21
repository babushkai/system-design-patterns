# Incident Command and Learning

## TL;DR

Incident response is a temporary command system for reducing harm under uncertainty. It begins when credible impact or risk is declared, not when root cause is known. A named incident commander establishes one decision log, assigns mitigation, communications, and evidence roles, controls parallel work, and repeatedly chooses the safest action that can reduce impact.

Mitigation and diagnosis run in parallel, but restoration usually has priority. Every action names owner, expected effect, blast radius, rollback, observation window, and stop condition. Recovery is proven through user-facing signals, backlog/data reconciliation, regional coverage, and removal of temporary mitigations—not by one dashboard turning green.

Postmortems reconstruct conditions and decisions without blaming individuals. Corrective actions target detection, prevention, containment, mitigation, recovery, and organizational control; each has an owner, due/review state, verification evidence, and closure decision. A postmortem without action tracking is a narrative archive.

[Alert Evaluation and Notification](./04-alerting.md) covers alert state and routing, [SLOs and Error-Budget Control](./05-slos-error-budgets.md) budget impact, and [Production Logging Architecture](./03-logging.md) evidence collection. Scope here: declaration, command, mitigation, communications, recovery, postmortems, and action governance.

---

## Incident Workload and Contract

An incident record contains:

~~~text
stable incident ID and command-system revision
declared/observed start and current state
severity with impact/urgency evidence
affected user journeys, tenants, regions, and data
incident commander and role assignments
authoritative coordination and communication channels
current facts, unknowns, hypotheses, and decisions
mitigation actions, owners, timestamps, rollback, and outcomes
telemetry/config/deployment/evidence links
next update and escalation commitments
recovery criteria and validation
postmortem/action-tracking state
~~~

### Invariants

1. Exactly one active incident commander owns priority and final coordination at a time.
2. One authoritative timeline/decision log survives chat and personnel turnover.
3. Severity is revised from observed/potential impact and urgency, not team prestige.
4. Every action has one owner; “the team” is not an assignee.
5. Mitigations name blast radius, expected signal, rollback, and observation deadline.
6. Facts, hypotheses, decisions, and unknowns remain distinguishable.
7. External communication is truthful, audience-safe, and time-stamped.
8. Evidence preservation does not block urgent harm reduction.
9. Recovery validates users, data/backlog, capacity, and temporary controls.
10. Corrective actions close only with evidence or an explicit risk-acceptance decision.

## Lifecycle and State

~~~mermaid
stateDiagram-v2
    [*] --> Suspected
    Suspected --> Declared: credible impact/risk
    Suspected --> ClosedNoIncident: disproved
    Declared --> CommandEstablished
    CommandEstablished --> Mitigating
    Mitigating --> Stabilized: impact no longer growing
    Stabilized --> Recovering: service/data restoration
    Recovering --> Monitoring: recovery criteria met
    Monitoring --> Resolved: sustained validation
    Monitoring --> Mitigating: regression
    Resolved --> Reviewing
    Reviewing --> ActionsOpen
    ActionsOpen --> Closed: actions verified or risk accepted
~~~

Security/legal incidents may fork into confidential investigation while operational command continues. The incident ID links them under access control; broad operational channels do not receive restricted evidence.

### Detection and intake

Signals include:

- user/support report;
- SLO/burn or functional alert;
- security detection;
- data reconciliation failure;
- change/deployment anomaly;
- dependency/provider notice;
- cost/capacity anomaly;
- synthetic or external observation.

The responder validates enough to choose:

1. Is impact or credible risk present?
2. Which journey/population is affected?
3. Is harm active, expanding, or latent?
4. What immediate containment is available?
5. Does coordination exceed one person/team?
6. Are security, privacy, safety, legal, or regulatory processes triggered?

Do not require root cause or a complete metric threshold to declare. Early declaration is cheap if the process supports rapid downgrade/closure.

## Severity as a Dynamic Decision

Severity determines coordination, authority, communication, and escalation. Evaluate dimensions:

| Dimension | Questions |
|---|---|
| User/business impact | Which functions, customers, revenue, obligations, or safety outcomes? |
| Scope | One request, tenant, cell, region, or global population? |
| Duration/trajectory | Stable, worsening, intermittent, or latent data corruption? |
| Workaround | Automatic, operator-assisted, degraded, or none? |
| Data/security | Loss, corruption, exposure, unauthorized action, integrity uncertainty? |
| Operational control | Can the team observe and safely mitigate? |
| External obligation | Contract, regulator, partner, public communication? |

Map these to the organization’s severity policy. Avoid universal percentage/duration thresholds; a low-volume high-value transaction or security exposure can be severe with few requests.

Reassess after each major fact:

- impact expands/contracts;
- data loss/exposure becomes possible;
- mitigation fails;
- external communication obligation begins;
- recovery stalls;
- second region/control plane is affected.

Downgrading reduces response overhead only when evidence supports it.

## Establishing Command

### Roles

| Role | Authority and responsibility |
|---|---|
| Incident commander | Sets priorities, assigns owners, accepts/halts actions, escalates, controls handoff |
| Operations/mitigation lead | Executes and coordinates technical changes |
| Communications lead | Internal/external stakeholder updates and next-update promise |
| Scribe/evidence lead | Timeline, decisions, action outcomes, links, impact evidence |
| Subject-matter responders | Investigate bounded hypotheses or systems |
| Security/privacy/legal liaison | Restricted evidence, notification obligations, chain of custody |

Small incidents can combine roles, but command and hands-on execution should separate once coordination load impairs either.

### Command bootstrap

Within the first command cycle:

1. Assign incident ID, commander, and severity.
2. Name authoritative coordination, decision log, and status destination.
3. State current user impact and confidence.
4. Freeze or constrain unrelated high-risk changes.
5. Assign one mitigation path and bounded investigations.
6. Identify missing expertise/authority and page it.
7. State next decision/update time.
8. Record recovery criteria.

This is a synchronization barrier: responders stop duplicating work and know where authority lives.

### Command loop

~~~text
observe user impact and system state
-> update facts/unknowns
-> review active actions and outcomes
-> choose next safest high-leverage action
-> assign one owner and observation deadline
-> communicate impact/current plan/next update
-> repeat until recovery criteria hold
~~~

Limit concurrent changes. Multiple uncoordinated mitigations destroy causal evidence and can interact unpredictably.

### Handoff

A commander/role handoff includes:

- current impact/severity and confidence;
- active mitigations and when to evaluate;
- facts versus leading hypotheses;
- risky/forbidden actions;
- stakeholder commitments and next update;
- access/authority gaps;
- recovery criteria; and
- explicit acceptance by the new owner.

Record handoff generation/time. “I posted notes in chat” is not an accepted transfer of command.

## Mitigation Under Uncertainty

Root-cause analysis asks why. Mitigation asks which reversible action can reduce harm now.

### Action contract

~~~text
action ID and owner
hypothesis or direct containment goal
target and exact scope
expected user/system signal
known risks and blast radius
prerequisites/approvals
execution steps/reference
rollback/compensation
start time and observation deadline
result and evidence
~~~

### Choosing an action

Rank candidates by:

- expected impact reduction;
- time to effect;
- reversibility;
- confidence/evidence;
- blast radius;
- state/data risk;
- dependencies/authority;
- ability to observe outcome; and
- interference with other recovery paths.

A useful qualitative decision score is not a magic formula, but decisions can make assumptions explicit:

$$
\text{expected net harm reduction}
\approx
P(\text{success})H_{\text{avoided}}
- P(\text{failure})H_{\text{added}}
- C_{\text{delay}}.
$$

Values may be ranges/ordinal estimates. The purpose is to expose why a broad irreversible action is or is not justified.

### Common mitigation classes

- rollback/revert recent change;
- disable feature/traffic cohort;
- shift traffic to healthy cell/region;
- shed optional or low-priority work;
- reduce concurrency/retries/fan-out;
- fail over dependency/data path under consistency policy;
- add bounded capacity;
- isolate tenant or corrupt resource;
- pause writers/consumers;
- restore/checkpoint/replay; or
- revoke credential/access.

Each is safe only under its system’s contract. A database failover can lose acknowledged writes; replay can duplicate effects; broad retry disable can expose partial failure; region shift can overload the destination.

Canonical deployment rollback mechanics live in [Deployment Strategies](../15-deployment/01-deployment-strategies.md), data migration in [Database Migrations](../15-deployment/03-database-migrations.md), and multi-region failover in [Multi-Region Architecture](../06-scaling/09-multi-region-architecture.md).

### Change control during incident

Use an incident change lane:

- commander approves or delegates approval;
- one owner executes;
- exact diff/command is recorded before or immediately after urgent execution;
- peer verifies high-risk steps when time permits;
- automation uses incident-scoped authorization;
- rollback is prepared;
- result is observed before next interacting change.

Do not paste secrets into chat/logs or disable audit globally. Break-glass access is time-bounded, least privilege, and reviewed.

## Communications

Communication reduces duplicate escalation and protects trust. Audience-specific streams:

- responder command channel;
- internal stakeholder/executive update;
- support/customer-facing status;
- partner/provider coordination;
- restricted security/privacy/legal channel.

### Update contract

~~~text
timestamp and incident ID
current confirmed impact and scope
what changed since last update
mitigation/recovery status
known limitations or workaround
next update time
contact/escalation path
~~~

Do not publish speculative root cause as fact. Say what is known, unknown, and being validated. Avoid internal topology, customer-specific details, vulnerabilities, personal data, or credentials in broad updates.

Cadence follows severity, uncertainty, and stakeholder obligation. If no new fact exists, update that investigation/mitigation continues and preserve the next-update promise.

External resolution waits until user recovery is validated and material regressions/backlog are understood. A later postmortem can explain cause more fully.

## Evidence and Timeline

### Evidence sources

- alerts, SLOs, metrics, logs, traces, profiles;
- deployment/configuration/feature-flag history;
- identity, authorization, and audit records;
- database/message/workflow state;
- provider/status communications;
- commands, changes, approvals, and access;
- support/customer reports; and
- screenshots/exports only when queryable sources may expire.

### Evidence record

For each item:

~~~text
source and access classification
event time, observed/collection time, and clock uncertainty
query/config/schema revision
region/tenant/resource scope
content digest or immutable reference
collector identity and acquisition time
retention/hold requirement
~~~

Preserve source evidence; annotations and hypotheses are separate. Redaction copies never replace restricted originals where legal/security process requires them.

### Timeline semantics

Use:

- observed start (earliest supported impact);
- detected time;
- declared time;
- command established;
- mitigation actions and outcomes;
- stabilization;
- recovery start;
- validated resolution;
- later discovered latent impact.

Do not force events into exact order when clocks/correlation cannot establish it. Record ranges and uncertainty.

Security evidence may require chain-of-custody and specialized procedures under organizational policy. Operational responders should avoid destructive cleanup until the security lead balances containment and evidence, but immediate safety/harm reduction remains primary.

## Recovery and Resolution

### Stabilized is not recovered

- **mitigating:** impact is being reduced;
- **stabilized:** impact is no longer growing;
- **recovering:** service/data/backlog is returning;
- **monitoring:** recovery criteria hold but regression risk remains;
- **resolved:** criteria sustained and command can end.

### Recovery criteria

Define at declaration and refine:

- user journey succeeds across affected populations;
- latency/error/freshness/correctness returns under agreed bounds;
- valid traffic no longer depends on emergency fallback unexpectedly;
- queues/backlogs drain within capacity;
- data integrity/reconciliation is complete or separately owned;
- failover region has headroom;
- security exposure/credentials are contained;
- telemetry coverage is trustworthy;
- temporary access/flags/rate changes are inventoried; and
- customer/status obligations are updated.

One green average can hide a failed tenant, region, or operation. Validate the exact impacted cohorts.

### Removing mitigations

Temporary changes create debt:

~~~text
mitigation
  -> owner
  -> intended expiry/exit condition
  -> rollback/removal plan
  -> residual risk
  -> verification
~~~

Remove them one at a time with observation. Some become permanent only through normal design review, testing, and ownership.

## Multi-Region and Multi-Incident Command

### Federated command

For one-region impact:

- regional lead executes local mitigation;
- global commander coordinates shared dependencies, failover, and communications;
- one incident timeline links regional substreams;
- authority for traffic/data movement is explicit.

For independent incidents, do not merge merely because they share time. For one shared cause with many symptoms, one command may reduce duplication. Criteria:

- same mitigation authority and decision set;
- evidence of causal relation;
- responder/communication overlap;
- merging will not hide distinct severity or data obligations.

### Partition

If command tooling or regions partition:

- each region can declare a local incident and operate from last known policy;
- globally unique incident IDs avoid collisions;
- local authority limits are documented;
- status updates mark connectivity/coverage;
- conflicting traffic/data actions are fenced where necessary;
- timelines reconcile without overwriting; and
- duplicate external communication has a preferred authority/fallback.

Incident tooling should have an out-of-band fallback: alternate communication, offline contacts/runbooks, and independent status publishing.

## Capacity and Human Load

### Responder capacity

Assume:

- $I$ concurrent incidents;
- $R_i$ required active roles/responders for incident $i$;
- $H$ handoffs per unit time;
- $\bar{t}_h$ handoff effort;
- $P$ pages/notifications per unit time; and
- $\bar{t}_p$ triage effort.

Active responder demand:

$$
R_{\text{demand}} = \sum_{i=1}^{I} R_i.
$$

Coordination overhead:

$$
T_{\text{coordination}}
\approx
H\bar{t}_h + P\bar{t}_p
$$

before mitigation work. During alert storms, notification reduction and incident grouping recover real engineering capacity.

Maintain:

- backup rotations and escalation;
- maximum sustainable consecutive duty under policy;
- role separation for complex incidents;
- follow-the-sun handoff;
- SMEs without making one person a permanent dependency;
- security/legal/executive contacts; and
- protected recovery time after severe incidents.

### Incident-system capacity

Let:

- $\lambda_a$ timeline actions/events per second during peak;
- $\bar{b}$ average event/evidence metadata bytes;
- $T_r$ retention;
- $r$ replication; and
- $E$ linked evidence objects.

Core timeline storage:

$$
S_{\text{timeline}}
\approx
\lambda_a \bar{b} T_r r,
$$

but evidence exports can dominate. Store immutable references/digests where source retention is guaranteed; snapshot only expiring critical evidence under authorized policy.

Size for organization-wide dependency incidents: many services declare, status subscribers surge, chat/API rate limits appear, and every responder queries telemetry simultaneously. Incident and observability tools should not share all failure/capacity domains with production.

## Failure Traces

### Two commanders create conflicting mitigations

~~~text
regional and service teams both declare command
-> one shifts traffic to region B
-> another drains region B for suspected corruption
-> neither timeline contains the other action
-> availability and evidence worsen
~~~

**Controls:** one global incident link/commander, explicit regional authority, action log, ownership, and fencing for conflicting control-plane changes.

### Reversible-looking rollback corrupts data

~~~text
recent application version is rolled back
-> new version already wrote expanded schema/state
-> old version interprets or overwrites it incorrectly
-> availability improves while latent corruption grows
~~~

**Controls:** rollback compatibility evidence, read-only/traffic containment alternative, database migration contract, canary validation, and reconciliation before resolution.

### Public update leaks restricted evidence

~~~text
internal timeline text is copied to status page
-> it includes tenant name and suspected vulnerability
-> unconfirmed/security-sensitive information spreads
~~~

**Controls:** audience-specific templates, communications owner, redaction, restricted channel, approval for security disclosures, and immutable record of published updates.

### “Resolved” ignores backlog and hidden cohort

~~~text
global error average recovers after traffic shift
-> one tenant/cell remains broken and queue backlog grows
-> command closes and temporary capacity is removed
-> delayed effects cause a second incident
~~~

**Controls:** cohort-specific recovery criteria, backlog/freshness/data validation, headroom check, monitoring state, and staged mitigation removal.

### Postmortem actions rot

~~~text
review creates many vague tasks
-> no owner, verification, or risk ranking
-> tickets age and close as “won’t do”
-> same failure recurs with prior document cited but no control changed
~~~

**Controls:** action state machine, one owner, due/review, expected control, verification evidence, recurring risk review, and explicit risk acceptance.

## Postmortems

### When to review

Trigger based on learning/risk:

- material customer/SLO/business/security impact;
- data loss/corruption or near miss;
- prolonged/complex command;
- unexpected control failure;
- repeated incident family;
- mitigation caused significant secondary harm;
- detection/response gap with high potential impact; or
- policy/regulatory requirement.

Near misses can deserve more review than routine visible outages.

### Structure

1. Executive summary and impact.
2. Detection, declaration, mitigation, and recovery metrics.
3. Timeline with uncertainty and evidence links.
4. Technical and organizational contributing conditions.
5. What worked and limited impact.
6. What failed or made response harder.
7. Counterfactual controls at prevention/detection/containment/recovery layers.
8. Corrective actions with ownership and verification.
9. Remaining/accepted risk.
10. Audience/redaction and review approvals.

Avoid “human error” as a terminal cause. Ask why the action was reasonable given interfaces, incentives, information, workload, access, and controls.

### Causal analysis

Distinguish:

- triggering event;
- latent conditions;
- detection gaps;
- amplification mechanisms;
- failed safeguards;
- coordination/authority gaps;
- recovery constraints; and
- organizational contributors.

A single root-cause label can hide the multiple conditions required for a distributed incident.

## Corrective-Action Governance

### Action types

- eliminate trigger;
- reduce probability;
- limit blast radius;
- detect earlier/more accurately;
- automate or simplify mitigation;
- improve rollback/recovery/data repair;
- improve command/communication/evidence;
- reduce recurrence across similar systems;
- retire obsolete control; or
- accept risk explicitly.

### Action state

~~~mermaid
stateDiagram-v2
    [*] --> Proposed
    Proposed --> Accepted: owner/scope/control agreed
    Proposed --> Rejected: rationale/risk decision
    Accepted --> InProgress
    InProgress --> Implemented
    Implemented --> Verified: test/production evidence
    Verified --> Closed
    Accepted --> Deferred: explicit priority/review date
    Deferred --> Accepted
    Accepted --> Superseded: replacement action linked
~~~

An action contract:

~~~text
incident/failure mode and risk
control to add/change
scope and one accountable owner
priority and due/review date
dependencies and rollout
verification method and expected evidence
residual risk
closure/risk-acceptance approver
~~~

“Improve monitoring” and “be more careful” are not verifiable actions. “Add an independent freshness signal that fails unknown on source loss, with replay test X” is.

### Portfolio review

Aggregate incidents/actions by failure mechanism, service, control layer, and recurrence. Watch:

- overdue/unowned/deferred actions;
- repeat incidents before action closure;
- action effectiveness after deployment;
- too many local fixes for a shared platform problem;
- controls that add more operational complexity than risk reduction; and
- accepted risk whose assumptions expired.

## Operating the Incident Program

Track:

- observed/detected/declared/command/mitigated/recovered/resolved times with definitions;
- impact and error-budget/data/security consequence;
- severity changes and reason;
- pages, responders, handoffs, and role gaps;
- action attempts, reversals, and mitigation effectiveness;
- communication timeliness and correction;
- evidence coverage and access;
- postmortem completion and audience;
- corrective action state/age/verification;
- recurrence and control effectiveness; and
- responder health/toil feedback.

Avoid ranking teams by raw incident count or time-to-resolve without context; it incentivizes under-declaration and premature closure.

## Verification and Exercises

| Exercise | What to verify |
|---|---|
| Declaration drill | Credible ambiguous signal becomes owned incident without root cause |
| Command handoff | New commander can state impact, actions, risks, and next update |
| Mitigation game day | Rollback, traffic shift, shed, failover, and access follow action contract |
| Tool outage | Alternate communication, contacts, runbooks, evidence, and status path work |
| Regional partition | Local authority and global reconciliation prevent conflicting actions |
| Security/privacy | Restricted channel, evidence custody, redaction, and notification escalation |
| Recovery drill | Backlog/data/cohort validation and temporary-control removal |
| Postmortem review | Timeline separates fact/hypothesis and actions are verifiable |
| Action audit | Closed items have evidence; deferred/accepted risks are still valid |
| Organization-wide scenario | Concurrent incidents, alert storm, status load, and responder capacity |

Exercise the socio-technical system, not only a failover command. Include missing dashboards, unavailable expert, incorrect hypothesis, and communication pressure.

## Decision Framework

1. Is there credible current or potential impact requiring coordinated action?
2. Who is commander, and where is the authoritative log/channel?
3. What severity follows from impact, scope, trajectory, data/security, and control?
4. Which immediate containment has highest expected harm reduction and reversibility?
5. What action owner, blast radius, rollback, observation signal, and deadline apply?
6. Which investigations can run without blocking mitigation?
7. Who needs which facts, and when is the next update?
8. What evidence is expiring/restricted and how is it preserved safely?
9. What exact user, cohort, data, backlog, capacity, and security criteria prove recovery?
10. Which temporary controls must be removed or formalized?
11. Which contributing conditions and failed safeguards explain the event?
12. Which corrective control is owned, verifiable, and worth its complexity?

## Key Takeaways

1. Declare on credible impact/risk; root cause is not a prerequisite.
2. One commander and one decision log prevent conflicting work.
3. Severity is dynamic and based on impact, urgency, data/security, and control.
4. Every mitigation has one owner, expected signal, blast radius, rollback, and observation deadline.
5. Restore service and reduce harm while bounded diagnosis proceeds in parallel.
6. Communications separate confirmed fact, uncertainty, current action, and next update.
7. Recovery validates user cohorts, backlog/data, capacity, telemetry, and temporary controls.
8. Multi-region incidents need explicit local/global authority and partition behavior.
9. Postmortems analyze multiple contributing conditions without blaming individuals.
10. Corrective actions close only with verification evidence or explicit risk acceptance.

---

## References

- [NIST SP 800-61 Rev. 3: Incident Response Recommendations](https://csrc.nist.gov/pubs/sp/800/61/r3/final) — incident response integrated with cybersecurity risk management
- [CISA Federal Government Cybersecurity Incident and Vulnerability Response Playbooks](https://www.cisa.gov/news-events/news/cisa-releases-cybersecurity-incident-and-vulnerability-response-playbooks) — standardized preparation, coordination, response, and recovery actions
- [Google SRE Workbook: Incident Response](https://sre.google/workbook/incident-response/) — command roles, structure, and response practice
- [Google SRE Workbook: Postmortem Culture](https://sre.google/workbook/postmortem-culture/) — blameless learning, review, and action tracking
- [Google SRE Book: Managing Incidents](https://sre.google/sre-book/managing-incidents/) — incident command system and communications
- [FIRST CSIRT Services Framework](https://www.first.org/standards/frameworks/csirts/csirt_services_framework_v2.1) — incident management service capabilities
- [Alert Evaluation and Notification](./04-alerting.md) — detection, paging, routing, and notification HA
- [SLOs and Error-Budget Control](./05-slos-error-budgets.md) — user impact, burn, and reliability policy
- [Production Logging Architecture](./03-logging.md) — event evidence, ordering, retention, and restricted access
- [Deployment Strategies](../15-deployment/01-deployment-strategies.md) — safe rollout, rollback, and recovery mechanics
