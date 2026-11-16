# Incident Command and Learning

> Twelve engineers are in the call, three are changing production, and nobody knows who is deciding. What structure makes the next ten minutes go better?

## TL;DR

Incident response is a temporary command system for reducing harm under uncertainty. It begins when credible impact or risk is declared, which is usually well before the cause is known. A named incident commander establishes one decision log, assigns roles for mitigation, communications, and evidence, controls parallel work, and repeatedly chooses the safest action that can reduce impact.

Mitigation and diagnosis run in parallel, and restoration usually has priority. Every action names its owner, expected effect, blast radius, rollback, observation window, and stop condition. Recovery is proven through user-facing signals, reconciliation of backlog and data, coverage of every affected region, and the removal of temporary mitigations. One dashboard turning green does not prove it.

Postmortems reconstruct conditions and decisions without blaming individuals. Corrective actions target detection, prevention, containment, mitigation, recovery, and organizational control, and each has an owner, a due or review state, verification evidence, and a closure decision. A postmortem with no tracking of its actions is a narrative archive.

[Alert Evaluation and Notification](./05-alerting.md) covers alert state and routing, [SLOs and Error-Budget Control](./04-slos-error-budgets.md) covers budget impact, and [Production Logging Architecture](./02-logging.md) covers evidence collection. This chapter covers declaration, command, mitigation, communications, recovery, postmortems, and the governance of corrective actions.

---

## Incident Workload and Contract

Most of this book concerns the behavior of software. This chapter concerns the behavior of a group of people under pressure, and the two have more in common than might be expected. A group responding to an outage is a distributed system. Its members hold different and partly stale views of the situation, they communicate over lossy channels, and they act concurrently on shared state. Left without structure, the group exhibits the failures of any uncoordinated distributed system: two people make conflicting changes, three investigate the same hypothesis while nobody examines another, and a decision made in one conversation never reaches the people in the next.

The structure that prevents this was worked out in the 1970s by fire services in California, who had found that large wildfires were being lost to confusion among the agencies fighting them, more than to any lack of equipment. Their Incident Command System gave every incident a single commander, defined roles, and a common vocabulary. Software organizations adopted it because the problem is the same one.

An incident begins with a record, which serves as the shared state the responders coordinate through:

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

The first two invariants are the familiar remedies for a distributed system with conflicting writers: a single leader and a single log. The sixth guards against a failure peculiar to people. Under stress, a plausible guess repeated a few times in a chat channel acquires the standing of an established fact, and responders then act on it. Keeping what is known separate from what is suspected, in writing, is the defense.

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

The diagram does not end at "Resolved." Three further states follow it, and an incident reaches "Closed" only when its corrective actions have been verified or the remaining risk has been formally accepted. The organization's attention naturally lapses when the service is restored, and the lifecycle is drawn this way to counter that.

Security and legal incidents may fork into a confidential investigation while operational command continues. The incident ID links the two under access control, and restricted evidence does not go to broad operational channels.

### Detection and intake

An incident may first appear through any of several channels:

- user/support report;
- SLO/burn or functional alert;
- security detection;
- data reconciliation failure;
- change/deployment anomaly;
- dependency/provider notice;
- cost/capacity anomaly;
- synthetic or external observation.

The first responder validates only enough to answer six questions:

1. Is impact or credible risk present?
2. Which journey/population is affected?
3. Is harm active, expanding, or latent?
4. What immediate containment is available?
5. Does coordination exceed one person/team?
6. Are security, privacy, safety, legal, or regulatory processes triggered?

None of the six asks for the cause. The commonest mistake at this stage is to delay declaring an incident until the problem is understood, on the feeling that a declaration without a diagnosis is premature. The cost of that delay is that for twenty minutes several people work on the problem separately, with no coordination and no record. Declaring early is cheap, provided the process makes it equally easy to downgrade or close an incident that turns out to be nothing. An organization that penalizes false alarms will get late declarations.

## Severity as a Dynamic Decision

Severity determines how much coordination, authority, communication, and escalation an incident receives. It is assessed along several dimensions:

| Dimension | Questions |
|---|---|
| User/business impact | Which functions, customers, revenue, obligations, or safety outcomes? |
| Scope | One request, tenant, cell, region, or global population? |
| Duration/trajectory | Stable, worsening, intermittent, or latent data corruption? |
| Workaround | Automatic, operator-assisted, degraded, or none? |
| Data/security | Loss, corruption, exposure, unauthorized action, integrity uncertainty? |
| Operational control | Can the team observe and safely mitigate? |
| External obligation | Contract, regulator, partner, public communication? |

These map onto the organization's severity policy. Universal thresholds expressed as a percentage of requests or a duration should be avoided, because they measure volume and severity is about consequence. A failure affecting a hundredth of a percent of requests may be trivial, or it may be every wire transfer above a certain value. A security exposure can be severe with almost no requests involved.

Severity is reassessed whenever a significant new fact arrives:

- impact expands/contracts;
- data loss/exposure becomes possible;
- mitigation fails;
- external communication obligation begins;
- recovery stalls;
- second region/control plane is affected.

It moves in both directions. An incident is downgraded when the evidence supports it, which reduces the overhead of response. It is not downgraded because the responders are tired.

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

The incident commander does not fix the problem. The commander's work is coordination: holding the overall picture, deciding priorities, and making sure every piece of work has exactly one owner. This is counterintuitive, since the person who takes command is often the most senior engineer present and the one most able to debug the system. An engineer who is deep in a log file cannot also be tracking what four other people are doing, and when the two jobs are combined, it is the coordination that lapses.

The communications role exists for a related reason. During a serious incident, executives, support staff, and customers all want to know what is happening, and without a dedicated person they ask the responders directly. Each question interrupts someone who was in the middle of a thought.

Small incidents may combine roles. Command and hands-on execution should be separated as soon as the load of coordinating begins to impair either.

### Command bootstrap

Within the first cycle of command:

1. Assign incident ID, commander, and severity.
2. Name authoritative coordination, decision log, and status destination.
3. State current user impact and confidence.
4. Freeze or constrain unrelated high-risk changes.
5. Assign one mitigation path and bounded investigations.
6. Identify missing expertise/authority and page it.
7. State next decision/update time.
8. Record recovery criteria.

This is a synchronization barrier. Before it, each responder has a private view of the situation and is acting on it. After it, everyone knows who is in charge, where decisions are recorded, and what they personally are responsible for, and duplicated work stops. Step 4 removes a confounding factor: other teams deploying changes during an incident may make it worse, and they certainly make it harder to tell which change had which effect.

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

The number of concurrent changes is kept small. When three mitigations are applied at once and the system recovers, nobody knows which of them worked, which were unnecessary, and whether any is doing harm that the improvement is masking. They may also interact in ways none would alone. Applying one change, observing its effect, and then deciding the next is slower per step and usually faster overall.

### Handoff

Long incidents outlast individual responders, and a change of commander is a moment of risk, since whatever the outgoing commander knew and did not write down is lost. A handoff includes:

- current impact/severity and confidence;
- active mitigations and when to evaluate;
- facts versus leading hypotheses;
- risky/forbidden actions;
- stakeholder commitments and next update;
- access/authority gaps;
- recovery criteria; and
- explicit acceptance by the new owner.

The last item makes it a handoff. Command has transferred when the incoming commander has said so, and the time is recorded. Notes posted in a chat channel that the next person may or may not have read do not transfer anything.

## Mitigation Under Uncertainty

The instinct of an engineer facing a broken system is to understand it. During an incident that instinct has to be held in check, because understanding and repair are different goals and the first is slower. Root-cause analysis asks why the system failed. Mitigation asks which reversible action would reduce harm now. A deployment an hour ago followed by errors is sufficient reason to roll back, and the rollback should not wait for someone to find the bug. If it works, users are served, and the bug can be found afterward at leisure.

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

The expected signal and the observation deadline together make an action falsifiable. Before acting, the owner states what should change and by when. If the deadline passes and the signal has not moved, the action did not work, and it is reversed or the hypothesis behind it is abandoned. Without this, failed mitigations accumulate in the system while responders wait to see whether they will eventually help.

### Choosing an action

Candidates are ranked by:

- expected impact reduction;
- time to effect;
- reversibility;
- confidence/evidence;
- blast radius;
- state/data risk;
- dependencies/authority;
- ability to observe outcome; and
- interference with other recovery paths.

No formula settles the choice, but writing the reasoning in one form makes the assumptions visible:

$$
\text{expected net harm reduction}
\approx
P(\text{success})H_{\text{avoided}}
- P(\text{failure})H_{\text{added}}
- C_{\text{delay}}.
$$

The values may be ranges or rough ordinal estimates. The purpose is to make clear why a broad, irreversible action is or is not justified. The middle term carries the argument for reversibility: an action that can be undone has a small cost of failure, and can therefore be tried on weaker evidence than one that cannot.

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

Each of these is safe only within the contract of the system it is applied to, and each has a characteristic way of making things worse. A database failover can lose writes that had been acknowledged. A replay can duplicate effects. Shifting traffic to another region can overload that region and turn a regional incident into a global one. The responder needs to know these hazards before reaching for the tool, which is the reason for rehearsing mitigations in advance.

Deployment rollback mechanics are in [Deployment Strategies](../14-deployment/03-deployment-strategies.md), data migration in [Database Migrations](../14-deployment/05-database-migrations.md), and multi-region failover in [Multi-Region Architecture](../06-scaling/14-multi-region-architecture.md).

### Change control during incident

Ordinary change control, with its reviews and waiting periods, is too slow for an incident. Having no control at all is how responders cause a second incident. An incident change lane sits between the two:

- commander approves or delegates approval;
- one owner executes;
- exact diff/command is recorded before or immediately after urgent execution;
- peer verifies high-risk steps when time permits;
- automation uses incident-scoped authorization;
- rollback is prepared;
- result is observed before next interacting change.

Secrets are never pasted into chat or logs, and audit is never disabled globally for convenience. Break-glass access is time-bounded, least-privilege, and reviewed afterward.

## Communications

Communication during an incident reduces duplicated escalation and protects the trust of those affected. Different audiences need different things, and each gets its own stream:

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

The next-update time is the most useful line in the template. A stakeholder who knows that another update will arrive in thirty minutes can wait for it. One who has heard nothing and has no idea when they will hear more begins asking, and escalating.

A speculative cause is not published as fact. An update says what is known, what is unknown, and what is being checked. Broad updates leave out internal topology, customer-specific details, vulnerabilities, personal data, and credentials.

Cadence follows severity, uncertainty, and obligation. When there is nothing new, an update still goes out to say that investigation or mitigation continues, and the next-update promise is kept. Silence is read as either abandonment or concealment.

External resolution is announced only after user recovery has been validated and any backlog or regression is understood. The fuller explanation of cause can wait for the postmortem.

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

The last item reflects a practical point about retention. High-resolution metrics and debug logs are often kept for days or hours. Evidence that will have expired before the postmortem is written has to be captured during the incident.

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

Source evidence is preserved as it was. Annotations and hypotheses are kept separately. A redacted copy never replaces a restricted original where legal or security process requires the original.

### Timeline semantics

The timeline distinguishes several moments that are often run together:

- observed start (earliest supported impact);
- detected time;
- declared time;
- command established;
- mitigation actions and outcomes;
- stabilization;
- recovery start;
- validated resolution;
- later discovered latent impact.

The intervals between them measure different things. From observed start to detection is the quality of monitoring. From detection to declaration is the willingness of the organization to call an incident. From declaration to mitigation is the effectiveness of the response. A single "time to resolve" blends all three and indicates nothing about which to improve.

Events are not forced into an exact order when clocks and correlation cannot establish one. Timestamps from different machines may disagree by more than the gap between two events, and the honest record is a range with its uncertainty.

Security evidence may require chain of custody and specialized procedures. Operational responders avoid destructive cleanup until the security lead has weighed containment against evidence, though immediate reduction of harm remains the first priority.

## Recovery and Resolution

### Stabilized is not recovered

The states are distinct, and confusing them causes repeat incidents:

- **mitigating:** impact is being reduced;
- **stabilized:** impact is no longer growing;
- **recovering:** service/data/backlog is returning;
- **monitoring:** recovery criteria hold but regression risk remains;
- **resolved:** criteria sustained and command can end.

### Recovery criteria

Criteria are defined at declaration and refined as the incident is understood:

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

Defining these early matters because the pressure to declare victory is strongest at the moment the main graph recovers. A global average returning to normal can hide a tenant, a region, or an operation that is still broken, and recovery has to be validated for the specific cohorts that were affected. Backlog is the other common omission. While the service was down, work accumulated in queues, and when the service returns that work arrives all at once. A system can recover and then be brought down again by its own backlog.

### Removing mitigations

Every temporary change made during the incident is a debt:

~~~text
mitigation
  -> owner
  -> intended expiry/exit condition
  -> rollback/removal plan
  -> residual risk
  -> verification
~~~

A rate limit that was lowered, a feature that was disabled, or capacity that was doubled will each, if forgotten, either cost money indefinitely or surprise someone months later who does not know why the system behaves as it does. Mitigations are removed one at a time, with observation after each. A mitigation becomes permanent only through normal design review, testing, and ownership.

## Multi-Region and Multi-Incident Command

### Federated command

For an incident confined to one region:

- regional lead executes local mitigation;
- global commander coordinates shared dependencies, failover, and communications;
- one incident timeline links regional substreams;
- authority for traffic/data movement is explicit.

Incidents that merely coincide in time are not merged, since merging them wastes the attention of people who have nothing to contribute to the other problem. Where one cause produces many symptoms, a single command reduces duplication. The criteria for merging are:

- same mitigation authority and decision set;
- evidence of causal relation;
- responder/communication overlap;
- merging will not hide distinct severity or data obligations.

### Partition

The tools of incident response can themselves fail, and they are most likely to fail during a large incident. If command tooling or regions partition:

- each region can declare a local incident and operate from last known policy;
- globally unique incident IDs avoid collisions;
- local authority limits are documented;
- status updates mark connectivity/coverage;
- conflicting traffic/data actions are fenced where necessary;
- timelines reconcile without overwriting; and
- duplicate external communication has a preferred authority/fallback.

Incident tooling needs a fallback that shares no dependencies with production: an alternate means of communication, contacts and runbooks available offline, and a status page hosted elsewhere. An organization whose chat, paging, and status page all depend on the infrastructure that is failing will be unable to coordinate its response or tell its customers anything.

## Capacity and Human Load

### Responder capacity

People are a finite resource with a capacity model like any other. Assume:

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

before any mitigation work is done. The second term is why alert storms are so damaging. If a failure sets off two hundred alerts and each takes a minute to triage, more than three hours of responder attention is consumed in establishing that they are all the same problem. Grouping notifications and suppressing dependent alerts recovers real engineering capacity at the moment it is scarcest.

The organization maintains:

- backup rotations and escalation;
- maximum sustainable consecutive duty under policy;
- role separation for complex incidents;
- follow-the-sun handoff;
- SMEs without making one person a permanent dependency;
- security/legal/executive contacts; and
- protected recovery time after severe incidents.

Fatigue is an operational risk. A responder in the fourteenth hour of an incident makes worse decisions than a fresh one, and handing over is usually better than persisting.

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

though exported evidence can dominate. Where the source's retention is guaranteed, the incident record holds immutable references or digests, and it snapshots only critical evidence that would otherwise expire.

The system is sized for an organization-wide dependency incident, in which many services declare at once, status-page traffic surges, chat and API rate limits are reached, and every responder queries the telemetry system simultaneously. Observability tools commonly slow down during a major incident, because everyone is using them at once. Incident and observability tools should not share every failure and capacity domain with production.

## Failure Traces

### Two commanders create conflicting mitigations

~~~text
regional and service teams both declare command
-> one shifts traffic to region B
-> another drains region B for suspected corruption
-> neither timeline contains the other action
-> availability and evidence worsen
~~~

Each team acted correctly on what it knew. One moved traffic into a region at the same time as the other was emptying it. This is a split brain, two leaders each believing they hold authority, and the remedy is the one used in software.

**Controls:** one global incident link/commander, explicit regional authority, action log, ownership, and fencing for conflicting control-plane changes.

### Reversible-looking rollback corrupts data

~~~text
recent application version is rolled back
-> new version already wrote expanded schema/state
-> old version interprets or overwrites it incorrectly
-> availability improves while latent corruption grows
~~~

Rollback restores the code and leaves in place whatever the code wrote. Here the errors stop, the incident appears to be over, and the old version is quietly damaging records written by the new one.

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

This is the commonest failure of an incident program, and it produces no alarm of any kind. The second postmortem cites the first, which had identified the problem correctly.

**Controls:** action state machine, one owner, due/review, expected control, verification evidence, recurring risk review, and explicit risk acceptance.

## Postmortems

### When to review

The decision to hold a review is based on how much can be learned and how much risk was revealed:

- material customer/SLO/business/security impact;
- data loss/corruption or near miss;
- prolonged/complex command;
- unexpected control failure;
- repeated incident family;
- mitigation caused significant secondary harm;
- detection/response gap with high potential impact; or
- policy/regulatory requirement.

Near misses can deserve more attention than routine visible outages. An event that almost destroyed data and was caught by luck reveals the same weakness that the real event would have, without the cost.

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

A postmortem is blameless, and the reason is practical. If the engineer who ran the command that caused the outage expects to be punished, they will describe what happened as favorably to themselves as they can, others will learn to do the same, and the organization will lose its only source of accurate information about how its systems fail. An accurate account requires that the people involved be safe in giving it.

Blamelessness also leads to better analysis. "Human error" is not accepted as a terminal cause, because it explains nothing and suggests no remedy except exhortation. The useful question is why the action seemed reasonable to a competent person at the time, given the interfaces, incentives, information, workload, access, and controls they had. If a command that deletes a production database is one character different from a command that deletes a test database, the fault lies with whoever made that possible.

### Causal analysis

The analysis distinguishes:

- triggering event;
- latent conditions;
- detection gaps;
- amplification mechanisms;
- failed safeguards;
- coordination/authority gaps;
- recovery constraints; and
- organizational contributors.

The search for a single root cause is misleading for complex systems. A serious incident in a well-defended system requires several things to go wrong together: a trigger, a latent weakness the trigger exposed, a safeguard that should have caught it and did not, and something that amplified the result. Removing any one would have prevented the incident. Naming one as "the" root cause directs all the corrective effort at it and leaves the others in place.

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

The last is a legitimate outcome. Some risks cost more to remove than they are worth, and the proper response is for someone with the authority to accept the risk to do so explicitly and on the record. An action that sits in a backlog untouched for a year is an acceptance of the same risk that nobody decided on.

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

The diagram separates "Implemented" from "Verified," and an action is closed only after the second. A control that has been built has not thereby been shown to work. Verification means demonstrating, by a test or by production evidence, that the control would catch the failure it was built for.

This also sets a standard for how actions are written. "Improve monitoring" and "be more careful" cannot be verified, so they cannot be closed honestly. "Add an independent freshness signal that reports unknown when its source is lost, with replay test X" can be.

### Portfolio review

Individual postmortems are reviewed one at a time, and patterns appear only when they are looked at together. Incidents and actions are aggregated by failure mechanism, service, control layer, and recurrence, watching for:

- overdue/unowned/deferred actions;
- repeat incidents before action closure;
- action effectiveness after deployment;
- too many local fixes for a shared platform problem;
- controls that add more operational complexity than risk reduction; and
- accepted risk whose assumptions expired.

The fourth is a signal that the organization is treating symptoms. If six teams have each written a postmortem action to add their own retry limits, the problem belongs to the platform they share.

## Operating the Incident Program

We track:

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

These metrics exist to improve the program and should not be used to rank teams. Ranking by raw incident count or by time to resolve rewards teams for not declaring incidents and for closing them early, and the organization then knows less about its reliability than it did before it started measuring.

## Verification and Exercises

Incident response is a skill, and like any skill it degrades without practice. Real incidents are too rare and too costly to be the only training.

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

An exercise should test the whole socio-technical system. Running a failover command successfully in a calm room shows that the command works. A realistic exercise includes the conditions of a real incident: a dashboard that is unavailable, an expert who cannot be reached, an early hypothesis that turns out to be wrong, and stakeholders pressing for answers.

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

- [NIST SP 800-61 Rev. 3: Incident Response Recommendations](https://csrc.nist.gov/pubs/sp/800/61/r3/final): incident response integrated with cybersecurity risk management
- [CISA Federal Government Cybersecurity Incident and Vulnerability Response Playbooks](https://www.cisa.gov/resources-tools/resources/federal-government-cybersecurity-incident-and-vulnerability-response-playbooks): standardized preparation, coordination, response, and recovery actions
- [Google SRE Workbook: Incident Response](https://sre.google/workbook/incident-response/): command roles, structure, and response practice
- [Google SRE Workbook: Postmortem Culture](https://sre.google/workbook/postmortem-culture/): blameless learning, review, and action tracking
- [Google SRE Book: Managing Incidents](https://sre.google/sre-book/managing-incidents/): incident command system and communications
- [FIRST CSIRT Services Framework](https://www.first.org/standards/frameworks/csirts/csirt_services_framework_v2.1): incident management service capabilities
- [Alert Evaluation and Notification](./05-alerting.md): detection, paging, routing, and notification HA
- [SLOs and Error-Budget Control](./04-slos-error-budgets.md): user impact, burn, and reliability policy
- [Production Logging Architecture](./02-logging.md): event evidence, ordering, retention, and restricted access
- [Deployment Strategies](../14-deployment/03-deployment-strategies.md): safe rollout, rollback, and recovery mechanics

---

## Related Chapters

- [Alert Evaluation and Notification](05-alerting.md): the page starts this process, and a badly routed page ruins the first ten minutes
- [Progressive Delivery and Deployment Strategies](../14-deployment/03-deployment-strategies.md): most incidents are changes, which is why rollback has to be a designed capability
- [Production Logging Architecture](02-logging.md): the evidence available at three in the morning, already shaped by retention and sampling
- [Multi-Region Architecture](../06-scaling/14-multi-region-architecture.md): a region failure is a decision to shed or fail over on incomplete evidence
- [Resilience Testing and Fault Injection](08-resilience-testing-and-fault-injection.md): drills rehearse the incident role set before an outage audits whether it works
