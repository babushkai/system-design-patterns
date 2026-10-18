# Threat Modeling and Abuse Case Engineering

## TL;DR

A threat model is the only security artifact that can exist before the code does, which is exactly why teams skip it and exactly why it pays. The deliverable is not a diagram and not a spreadsheet of scary possibilities; it is an ordered list of specific ways this system can be misused, each with a named owner, a named control or a written acceptance, and a test that fails when the control disappears. **Documented**: the recurring failure in post-incident reviews is not that a class of attack was unknown to the industry but that it was unknown to the person who made the decision, usually because nobody wrote down who could reach what.

Two practices carry most of the value. The first is enumerating threats against structure rather than imagination: name the data-flow elements and trust boundaries, then walk one standard taxonomy (STRIDE, or the equivalent for your environment) across them, so that coverage is a property of the method instead of the seniority in the room. The second is converting each accepted threat into an abuse case written as an executable test - actor, budget, goal, path, expected control, evidence - which turns "we thought about security" into something a reviewer can run and a regression cannot silently undo.

Threat modeling is cheap when scoped by exposure and expensive when scoped by ambition. A ninety-minute session on the newly exposed surface of one service, done before implementation details ossify, finds more exploitable defects per hour than any downstream activity in this book; a six-week enterprise-wide model produces a document that nobody can act on. **Inference**: the durable payoff is not the individual findings but the shared vocabulary - trust boundary, capability, abuse case, residual risk - that lets engineers, reviewers, and executives argue about the same object.

## What a Threat Model Is For

Strip away the tooling marketing and a threat model answers four questions in order:

| Question | Artifact that answers it | Who has to agree |
|---|---|---|
| What are we working on? | Data-flow diagram at the granularity of trust decisions, plus an assumptions list | Engineering lead, owning team |
| What can go wrong? | Enumerated threat list with a taxonomy label per entry | Security champion, reviewers from adjacent teams |
| What are we going to do about it? | Disposition per threat: mitigated by named control, transferred, or explicitly accepted with an expiry date | Product owner plus security owner |
| Did it work? | Abuse-case tests in CI, plus verification evidence for controls that cannot be automated | The team that ships the code |

Three clarifications prevent most of the ways this practice degenerates:

- A threat model is a **design activity**, not an audit. Its leverage comes from changing a design while the change is still a paragraph. Findings discovered after launch are still useful, but they arrive with a migration attached.
- It is **not** a vulnerability scan, a penetration test, or a compliance checklist. Those ask "what is broken in what exists"; a threat model asks "what will be possible once this ships", which is the question no scanner can answer because the feature is not deployed yet.
- It is **adversarial and economic**, not exhaustive. There is always an attack. The output is a judgment about which attacks are cheap enough for someone to bother with, valuable enough to target, and cheap enough to close that closing them is the right use of the sprint.

**Reference design**: keep one artifact per system, versioned next to the design doc it belongs to, with four sections in this order - scope and assumptions, data-flow diagram with trust boundaries, threat register, abuse-case index. Register entries get stable IDs so that a pull request can say "closes TM-14" and a reviewer can look up what TM-14 was in two years.

## Scoping the Model

Scope by exposure, not by org chart. The useful unit is a *trust decision surface*: one place where the system decides who is asking, what they may do, or whether to believe an input. Concretely, good scopes are "the public ordering API and its dependency calls", "the operator console and everything it can reach", "the ingest path from partner webhook to warehouse table". Bad scopes are "the platform", "all microservices", "the AWS account".

Pick the level of detail by what is still undecided:

- **Conceptual** (external systems, data classes, who owns what): right before architecture review. Answers "does this design put customer data inside a third party's blast radius?"
- **Design** (processes, stores, flows, boundaries, capabilities): the default level, and the one this chapter is about. Answers "which flow crosses a boundary without authenticating the caller's tenant?"
- **Implementation** (specific endpoints, libraries, dependency versions): right before launch, and mostly about code paths that were not planned - debug endpoints, health checks with echo, generated clients.

Write assumptions down as numbered statements with an owner and a review date: "the identity provider emits unforgeable tokens bound to audience", "operators cannot read plaintext customer fields", "partner traffic is signed". An assumption list is the cheapest artifact in security engineering because it converts an implicit trust claim into something someone can refute. Silent assumptions are where most models fail; the diagram looks complete and every box inside the perimeter is quietly trusted.

Time-box aggressively. Twenty minutes of model per hour of design review scales; a separate multi-day ceremony does not. Re-run the model when any of these change: a new externally reachable port, a new data class crossing a boundary, a new actor type (partners, agents, automated processors), a tenfold traffic change, an acquisition, or a new regulatory footprint.

## Trust Boundaries and Data-Flow Elements

Draw the system with four element types and one special line. Processes do work; data stores hold state between requests; data flows carry data across a boundary; external entities are things you do not control. The line matters more than the boxes: a **trust boundary** is a point where code stops believing what it received and has to re-establish belief, or where privilege changes.

```
  customer browser          partner callback            operator
        |                          |                        |
        v                          v                        v
   [ TLS / CDN ]============TRUST BOUNDARY #1==============(untrusted client)
        |
        v
   edge gateway --authn--> api service --authz+tenant--> order store
        |                        |                             ^
        |                        |                             |
        |                        +====TRUST BOUNDARY #2========+
        |                              (service to datastore: is the
        v                               tenant predicate inside the query?)
   [ logs, traces, metrics ]=====TRUST BOUNDARY #3=====
        (unvalidated strings become HTML, SQL, and shell in dashboards)
```

Four boundary families deserve explicit lines on every diagram:

| Boundary | Typical elevation of privilege | Belief that has to be re-established |
|---|---|---|
| Client to edge | Unauthenticated caller reaches business logic | Identity, rate entitlement, payload validity, ordering guarantees |
| Edge to service | One tenant's request executes in another's context | Authenticated principal, tenant scope, per-call authorization decision |
| Service to datastore | Application reads or writes outside its grant | Capability of the connection versus capability claimed by the end user |
| Human to control plane | Read path becomes write path through an admin tool | Whether support tooling enforces the same authorization as the API |

A fifth family appears in most modern systems: **service to service across team boundaries**, where the caller is a peer you do not review. Treat a peer service as untrusted input with better uptime. The moment one team's identity can be assumed by another team's workload, you have a confused deputy waiting for its first patient caller.

Two drawing rules keep the artifact honest. First, every flow that crosses a boundary gets a label stating what is asserted (identity, tenant, intent, integrity) and how it is asserted (signature, mTLS with pinned workload identity, HMAC over a canonical form, an out-of-band grant). Second, if you cannot name the data classes on a flow, you have not finished drawing it; "user data" is not a data class, "email address plus delivery consent flag" is.

## Spoofing and Tampering With Identity and Data

Spoofing asks "can something pretend to be an element in this diagram?" Enumerate it per element type, because the answers differ:

- **External entity**: can a caller present as a customer, a partner, or an operator? Failure modes are not exotic - tokens accepted without audience validation, a shared signing key between first-party and partner flows, a mobile client's user identifier taken from a request body, a support tool that lets an operator type a user ID.
- **Process**: can one workload run with another's identity? Shared instance metadata credentials, sidecars holding every service's token, CI runners with long-lived cloud keys, and container images pulled without digest pinning all produce spoofable processes.
- **Data flow**: can a message in transit be replayed, reordered, or redirected? Replay needs a nonce plus freshness window; redirection is caught by binding the credential to the destination (audience, channel binding, or a signature over the canonical request).
- **Data store**: less about spoofing than about who else holds credentials that reach it. A developer-accessible replica with production data is a spoofing surface for "internal service" claims.

Tampering asks whether integrity is enforced where the data crosses hands rather than where it is stored. The controls are always the same three ideas: authenticate the carrier (transport or workload identity), authenticate the payload (signature over a canonical form, not over whatever bytes the parser happened to preserve), and detect after the fact (hash chains, append-only logs, release transparency records).

Two tampering classes recur in application systems and are almost never in the original design:

1. **Canonicalization drift.** The signer hashes `{"a":1,"b":2}`, the verifier's library reorders keys or normalizes Unicode, and either the signature stops working (loud) or two different payloads hash the same after normalization (quiet and exploitable). Webhook signature schemes fail this way; so does any "we hash the JSON body" design that did not name the exact byte sequence being signed.
2. **Integrity of derived state.** Caches, search indexes, materialized views, feature stores, and embeddings are data stores whose contents came from somewhere. If the derivation is not authenticated end to end, an attacker who can write to the source or replay an old snapshot tampers with everything downstream while every individual store looks healthy.

## Repudiation and the Evidence You Need Later

Repudiation threats are the ones engineers forget because nothing breaks when they are ignored: a user denies an action, an operator denies a change, and the system cannot prove what happened. Design the evidence before the incident, because audit records cannot be backfilled.

A usable security event needs at least: a stable actor identifier (human or workload, not a display name), the action and its target, the decision that allowed it with the policy version consulted, the source of the request, a timestamp from a source the actor cannot rewrite, and an integrity property on the record itself. Append-only storage plus a periodic hash chain is the minimum for anything that may be disputed; "we have logs in the aggregator with a seven-day retention" is not.

Three traps:

- **Self-auditing.** If the actor can delete or edit their own audit trail, the trail is evidence of absence at best. Separate the write path from the read and retention paths, and put the retention policy under change control.
- **Missing negatives.** Denials are usually the interesting events: fifty failed authorization checks followed by one success describes an attack; one success alone looks like Tuesday.
- **PII in the audit log.** Audit records attract broad read access precisely because investigators need it, so they become a disclosure surface. Keep the sensitive payload out and store a reference plus a hash instead.

**Inference**: in most product teams, more repudiation risk comes from support and internal tooling than from customers. The person with a "refund anyway" button is an actor with money-moving capability whose actions are frequently unlogged; model that surface explicitly.

## Information Disclosure Paths Beyond the Database

Ask where each data class exists as a *copy*, not where it lives. A single email address typically ends up in the primary store, read replicas, backups, search index, cache, CDN logs, application logs, traces, metric labels, error reports, queue payloads, an analytics warehouse, a support ticket, and increasingly a model context or embedding store. Every copy inherits the classification of the original and usually has weaker controls than the source.

| Copy location | Disclosure path that shows up in reviews | Control that closes it |
|---|---|---|
| Logs and traces | Request body or header captured for debugging, then shipped to a third-party vendor | Field-level allowlist at capture, not redaction after the fact |
| Metrics labels | Customer identifier used as a label, becoming queryable by anyone with dashboard access | Label cardinality and content policy enforced in the pipeline |
| Cache | Cross-tenant read because the cache key omitted a tenant predicate or vary header | Key includes every authorization-relevant dimension; test it |
| Backups and snapshots | Restorable only by a vendor, or exported to a bucket with a public ACL | Encryption with customer-held keys plus an inventory of restorers |
| Model context | Prompt content retained by a provider and reused for training | Contractual retention limits plus client-side minimization |
| Error paths | Stack traces, internal hostnames, SQL fragments, object identifiers in responses | Generic error envelope; detail goes to the correlated log line only |

Two subtler classes are worth naming in every model. **Enumeration oracles** leak existence through behavior differences: pricing endpoints that answer for invalid coupon codes, password reset flows that distinguish accounts, search facets that reveal a competitor's inventory. **Timing and size channels** leak through legitimate responses: an endpoint whose latency depends on whether an identifier exists is an existence oracle regardless of its status code. Neither appears in a dependency scanner; both appear in a model that asks "what can a caller learn without ever getting a 2xx?"

## Denial of Service as a Design Decision

Availability threats are the ones most often dismissed as "just take it offline", which is how businesses lose money. Model them as amplification: attacker budget in, victim cost out. Anything with a ratio above one to one belongs in the register.

Recurring amplification patterns:

- **Unbounded per-request work.** One request that fans out to N dependents, renders an unbounded list, decompresses an archive, or triggers a full scan converts a unit of attacker effort into a unit of your capacity. The control is a stated budget per request plus a deadline that propagates.
- **Cheap-to-create expensive-to-store state.** Pending signups, unpaid carts, aborted uploads, sessions, and async jobs all cost the attacker cents and you dollars each. Rate limiting at the edge does not help once state is already created; the fix is to make unverified state cheap and expiring.
- **Retry and thundering-herd coupling.** A dependency failure that triggers client retries multiplies load exactly when capacity is gone. Model retry storms as an attack because they behave identically under both explanations.
- **Cache and index poisoning.** Forcing unique keys (attacker-controlled query strings, per-customer cache entries) converts a cache into a memory leak; unpurged invalidation lets one writer poison every reader.
- **Control-plane coupling.** If the operator path shares failure domain with the data path, an availability event becomes an unfixable availability event.

The disposition for most of these is not "add rate limiting" but "state the expected unit cost of this endpoint and enforce it", which is the same arithmetic as capacity planning. That is why the availability half of a threat model reads better with the capacity owner in the room than with a security specialist alone.

## Elevation of Privilege in Permission Graphs

Elevation of privilege is where threat models and authorization designs meet, so model it as graph reachability rather than as a list of bugs. Nodes are principals and roles; edges are "can act as", "can attach policy to", "can read secret of", "can call". An elevation path is any chain from an externally reachable node to a valuable capability, and the surprising paths are almost always through indirection:

- **Confused deputy.** A privileged service accepts a request that embeds instructions about a third party. Classic instances: passing a customer identifier to an internal API that trusts the caller, signed URLs whose scope is not bound to the recipient, image fetch endpoints that happily retrieve link-local metadata addresses.
- **Capability leakage by reference.** Long-lived tokens in URLs end up in logs, referrer headers, and browser history; object identifiers that are guessable turn a read of one record into a read of all records; "share by link" features silently mint anonymous capabilities.
- **Policy drift through the write path.** Anyone who can attach a role to a service account has that role transitively. Model the *can-modify-access* edges explicitly; they are how five support accounts become cluster administrators.
- **Human to machine boundary.** A laptop with cloud credentials is a production control plane. Developer access to production data or resources is elevation of privilege by design, accepted for convenience; write it in the register so that someone owns the acceptance rather than the fact.

Two questions find most paths in a review: "which inputs does this trusted component forward verbatim to a more privileged place?" and "who can change who is trusted, and how is that change itself authorized?" If the second answer is a ticket queue with human judgment as the only control, the model should say so plainly.

## Abuse Cases and the Adversary's Budget

A threat says what is bad; an abuse case says how a specific actor achieves it, and that difference is what makes the finding testable. Write abuse cases as requirements-shaped artifacts so they survive into the backlog next to functional stories:

```
AC-07  Actor: authenticated customer, normal client, no insider help
       Goal:  obtain another tenant's invoice PDFs
       Path:  enumerate /invoices/{id}; ids are sequential within a workspace
       Control under test: authorization check binds invoice id to caller tenant
       Evidence: automated test asserting 404 (not 403) for cross-tenant reads,
                 plus a fuzzed-id property test in CI
       Cost to attacker: ~10 minutes and a script
       Value if successful: regulated data disclosure, contractual breach
```

Attach an adversary profile, because "the attacker" is not one person and different profiles justify different controls:

| Profile | Budget | Realistic techniques | Controls that actually matter |
|---|---|---|---|
| Opportunistic scanner | Cents, automated | Credential stuffing, exposed endpoints, known CVEs within days of disclosure | Patch cadence, no default credentials, sane defaults, asset inventory |
| Automated abuser | Hundreds of dollars, botnets | Scraping, account farming, coupon testing, inventory hoarding | Rate and behavioral limits tied to identity plus cost per action |
| Targeted criminal | Thousands, patient | Phishing operators, buying stolen tokens, exploiting business logic | Phishing-resistant authentication, step-up checks on money movement, anomaly detection |
| Funded adversary | Unlimited time, scarce attention | Supply chain, zero-days, insider access, traffic analysis | Scope reduction, least privilege end to end, detection and response maturity |
| Insider or compromised operator | Legitimate credentials | Bulk export, log tampering, quiet policy edits | Dual control on sensitive exports, immutable audit, alerting on read volume |

The economic comparison is the decision. If the attack costs an adversary USD 400 in proxies and yields USD 5 of coupons, an imperfect control is fine and the sprint is better spent elsewhere; if it costs USD 20 of compute and yields account takeover at scale, no control is too expensive. Recording cost-to-execute next to value-if-successful is what lets a non-security reviewer agree with the priority without arguing about threat mythology.

## Prioritizing With Attack Trees and Kill Chains

Attack trees make prioritization arithmetic instead of opinion. Root the goal ("read all customer records"), decompose into OR branches (exploit the export endpoint, compromise an operator, restore a backup somewhere unauthorized) and AND branches (obtain credentials AND reach an internal service AND avoid detection). Assign each leaf an exploitability and cost estimate, then propagate: OR nodes take the minimum of their children, AND nodes combine multiplicatively. Two properties fall out that intuition misses. Redundant paths make a goal *easier* even when every individual path is unlikely, and hardening one branch of an AND node protects only that conjunction - which is why single-sign-on rollouts that remove passwords also have to remove the second factor's fallback or nothing was actually achieved.

For coverage rather than cost, map threats onto a tactical matrix such as MITRE ATT&CK: initial access, execution, persistence, privilege escalation, defense evasion, credential access, discovery, lateral movement, collection, exfiltration, impact. The matrix is not a risk score; it is a checklist that exposes the phases nobody modeled, which in application teams are reliably discovery and lateral movement because everyone models the front door.

Score with explicit likelihood modifiers rather than raw severity labels:

```
risk = P(exploitability) x detectability x blast radius x remediation cost
  P(exploitability): unauthenticated remote < authenticated < insider < already inside
  detectability:     0 = invisible, 1 = alert exists and is tested
  blast radius:      one tenant < one region < all tenants < regulated data
```

A finding reachable pre-authentication with no detection and all-tenant blast radius blocks a release. The same class of finding behind two authenticated boundaries with an alert on bulk reads becomes a scheduled ticket. Two-stage scores such as DREAD flatten this distinction; the multiplication above does not.

## Running the Model in Ninety Minutes

Threat modeling sessions fail for social reasons, not analytical ones. The mechanics that make them repeatable:

| Minutes | Activity | Output |
|---|---|---|
| 0-10 | State scope, goals, and what is explicitly out of scope | One sentence each, written where everyone sees it |
| 10-25 | Draw the data-flow diagram from the design doc; challenge every boundary label | Diagram with named data classes and boundaries |
| 25-45 | Silent enumeration: everyone writes threats against elements, no discussion | Independent lists (silence defeats anchoring on the loudest voice) |
| 45-70 | Merge, deduplicate, walk the taxonomy across elements to fill gaps | Threat register with taxonomy labels |
| 70-85 | Assign disposition and owner to every entry; score only the top band | Register rows with owners and dates |
| 85-90 | Name the abuse cases that become tests; schedule the re-model | Test tickets, one calendar trigger |

Facilitation rules that matter in practice:

- **No tool debate.** Choose the notation before anyone arrives; arguments about diagrams consume the session and produce nothing.
- **Invite the person who will be on call**, plus one engineer from a dependency. Outsiders find the boundary you stopped seeing.
- **Every "that would never happen" gets written down** with the name of the person who asserted it. This is the fastest way to surface silent assumptions, and being wrong later is cheap when the line is in the document.
- **Cap dispositions at three values**: mitigated by a named control, transferred to a party that accepted it in writing, or accepted by a named person with an expiry date. "Known issue" is not a disposition; it is how risk leaves the building.

A model is done when a reviewer can answer four things without asking anyone: what the system protects, who the actors are, which controls exist and where they are enforced, and which risks were accepted by whom. Anything less is a diagram with a date on it.

## Proving the Model With Purple Tests

Unverified controls decay; within two releases a "validated token" claim can silently become "parsed token". Verification is therefore part of the threat model rather than a downstream QA phase.

Tier the evidence by what the control claims:

1. **Unit and contract tests** for parsing and decision logic: reject wrong audience, wrong algorithm, expired not-before, missing tenant scope, canonicalization mismatch. These are cheap and they catch the boring majority.
2. **Access-control matrix tests** as executable tables: for each (principal, action, resource, tenant) tuple, assert allow or deny. Generate the matrix from the authorization model rather than hand-writing it, then fuzz identifiers to prove uniform denial rather than per-endpoint luck. Uniformity matters: a 403 on one endpoint and a 200 on its sibling is the standard shape of a broken authorization gate.
3. **Abuse-case integration tests** that attempt the actual path from the register against a deployed environment, asserting both the outcome and the observability side effect (the alert should fire when the attempt happens).
4. **Purple-team exercises** where an attacker mindset meets detection: obtain credentials from a laptop, move laterally with them, attempt bulk export, then check whether anyone could answer "who, from where, how many records" within an hour.
5. **Identity chaos**: rotate or expire credentials mid-request, revoke a role while a session is live, kill the identity provider, and see whether the system fails closed on write paths and fails sensibly on read paths. Most fail-open findings appear here rather than in any code review.

Feed results back into the register: every confirmed finding gets a threat ID, every closed control gets a test link, and every escape (an incident whose root cause was already in the register but accepted) triggers a re-read of who accepted it and on what assumptions. That loop is what converts a document into a program.

## Failure Traces

**Trace: the token that worked everywhere.** A service accepts a partner's ID token to authorize an internal call because both flows use the same issuer and the same signing key. Sequence: partner credentials leak in a vendor breach, the holder replays tokens against an internal endpoint, every signature check passes. Detection: none, because the requests are well-formed and authenticated. The model would have shown one boundary label that read "verified JWT" instead of "verified JWT bound to this audience and this client", and one abuse case asserting that a partner-audience token is rejected by internal APIs with a test in CI.

**Trace: the log line that read a customer's record.** A support engineer searches logs for a report ID; the search page renders raw log lines as HTML; one line contains a customer payload including an email address and a partial token. Sequence: unvalidated string crosses into a rendering boundary that nobody drew, then into a second one when the same logs feed a dashboard. The trust-boundary diagram names this flow explicitly, and the control is capture-time field allowlisting rather than a redaction regex someone forgot to update when the payload gained a new field.

**Trace: the fetch endpoint that found the metadata service.** A feature lets users import a document by URL. Validation checks the scheme and blocks localhost. Sequence: attacker supplies a link-local metadata address, receives temporary cloud credentials scoped to the instance role, then enumerates what the role can reach. The threat register catches this as "confused deputy: trusted service forwards caller-supplied destination to a more privileged place" with two dispositions: egress allowlist per destination class, and workload identity that cannot mint broad cloud credentials. Both are design-time decisions; neither survives as an afterthought.

**Trace: the discount oracle.** A pricing endpoint returns different latency and payload size for valid versus invalid promotion codes. Sequence: automated caller enumerates codes by measuring responses, harvests the valid set, applies them to a small fraction of orders, stays under fraud thresholds because each individual order looks legitimate. Detection: none, because nothing failed. The model treats this as an information disclosure path through a side channel with a blast radius bounded by promotion budget - accepted risk in one company, urgent in another - and the decision is recorded rather than discovered.

**Trace: the audit trail that could not survive discovery.** An operator changes a retention policy, exports a data set, then restores the previous configuration. Sequence: the audit store uses the same permissions as the operational database, so the actor can edit history; the export path is a read replica with no separate alerting. The repudiation section of the model asked exactly one question - "can the actor alter the record of their own action?" - and an honest answer would have cost two engineer-weeks instead of a legal discovery problem.

## Verification Strategy

Track whether the practice works rather than whether it happened:

- **Coverage**: fraction of externally reachable surfaces with a current model, weighted by data class rather than by endpoint count. A read-only public catalog does not deserve the same effort as a money-moving write path.
- **Test density on accepted risk**: every mitigated threat should point at a test that fails if the control is removed; every accepted risk should have a detection or an expiry date. Both are countable from the register.
- **Escape analysis**: for each security incident, was the root cause in the register? A yes means the disposition was wrong and belongs to whoever accepted it; a no means the model's scope or taxonomy was wrong. This single review is what improves models; more sessions do not.
- **Time-to-evidence**: during an exercise, how long until someone can name actor, source, and volume? It is the only availability-independent measure of whether the repudiation controls work.
- **Age distribution**: findings older than their agreed remediation window are a queue-health signal for leadership, because security debt ages into incident debt at a rate nobody models correctly.

Review questions that surface weak models fast: show me the boundary where privilege changes; show me one abuse case and the test that enforces it; name a risk you accepted and who signed it; what changed since the last model?

## Decision Framework

| Situation | Cheapest useful model | Depth |
|---|---|---|
| New externally reachable API | One-hour session on the request path plus dependency calls | Design level, 8-15 threats, abuse cases for the top three |
| New tenant boundary or partner integration | Boundary-focused session with the integration owner | Data-flow level, access matrix as a test artifact |
| Third-party component or acquired product | Supply-chain and trust-assumption pass on what it is allowed to reach | Assumptions list plus explicit capability grants |
| Regulated data entering a new store or pipeline | Copy-inventory model: every place the data class lands | Disclosure-heavy, retention and access per copy |
| Legacy system with no model and limited time | Twenty-minute exposure pass: what is reachable, by whom, with what capability | Conceptual level, register only, no diagram |
| Post-incident | Reverse-engineered model from the attack path actually used | One page, plus the taxonomy gap it exposed |

Reject a model that cannot change a decision. If nothing in the artifact will alter a design, a test, or an acceptance before launch, it was written for an auditor and should be replaced by the checklist that serves them directly.

## Key Takeaways

1. Model structure, not imagination: enumerate trust boundaries and data-flow elements, then walk one taxonomy across them so coverage does not depend on who is in the room.
2. The boundary line is the artifact. Every flow that crosses it needs a label stating what is asserted and how, or the diagram is decoration.
3. A threat becomes real when it becomes an abuse case with an actor, a budget, a path, and a test; until then it is a mood.
4. Prioritize with economics: cost to execute against value if successful, adjusted by detectability and blast radius. Severity labels alone hide both terms.
5. Attack trees expose the two things intuition misses - redundant paths lower risk only when every branch is closed, and hardening one leg of an AND conjunction protects nothing else.
6. Enumeration oracles and timing channels disclose data without a single successful response; model them because no scanner will.
7. Elevation of privilege is graph reachability through indirection: confused deputies, capability leakage by reference, and whoever can modify who is trusted.
8. Every disposition must be mitigated by a named control, transferred in writing, or accepted by a named person with an expiry date; "known issue" is how risk leaves the building.
9. Verification is part of the model: access matrices generated from the authorization policy, abuse cases run against deployed environments, and identity chaos to prove fail-closed behavior.
10. Scope by exposure and time-box to ninety minutes; a model that cannot change a decision should be deleted rather than maintained.

## References

- Shostack, A. "Threat Modeling: Designing for Security." Wiley, 2011. The four-question loop, element-by-element enumeration (the source of STRIDE's application method), and micro-hardening around high-value assets.
- Swiderski, F., and Snyder, W. "Threat Modeling." Microsoft Press, 2004. Practical integration of modeling into a development schedule with artifacts teams can actually maintain.
- Schneier, B. "Attack Trees: A Method for Constructing Security Models." Dr. Dobb's Journal, 1999; extended in ACM Transactions on Information and System Security, 2000. AND/OR decomposition and cost propagation from goal to leaves.
- Howard, J., and LeBlanc, D. "Writing Secure Code," 2nd ed. Microsoft Press, 2003. STRIDE as a classification of the eight properties trust claims depend on.
- Anderson, R. "Security Engineering: A Guide to Building Dependable Distributed Systems," 3rd ed. Wiley, 2020. Adversary economics, why controls must be cheaper than attacks, and human factors in perimeter design.
- Chua, T.-L. "A Philosophy of Software Security." 2023. Why security work should be organized around preventing specific bad outcomes rather than accumulating best-practice badges.
- MITRE ATT&CK knowledge base. Tactical coverage matrix used here for phase coverage, not scoring.
- OWASP Application Security Verification Standard (ASVS) 5.0 and OWASP Web Security Testing Guide. Requirement catalogs that map well onto abuse-case acceptance criteria.
- NIST SP 800-161r1, "Cybersecurity Supply Chain Risk Management Practices for Systems and Organizations." 2022. Supplier and component trust assumptions as first-class model entries.
- NIST SP 800-189, "Resilient Computer System Design." 2020. Availability-oriented framing of adversarial and accidental failure as one design problem.
- Katz, D., Reddy, S., Barkovich, B., Chojnacki, D., Lu, L., Srivatsan, D., et al. "BBox: Applying Security Chaos Engineering to Serverless Execution Context Creation." ACM CCS 2020. Executable security testing of isolation claims rather than argument about them.

---

## Related Chapters

- [API Threat Boundaries and Abuse Resistance](04-api-security.md): the enforcement half of every abuse case written here
- [Authorization at Scale](07-authorization-patterns.md): turns elevation-of-privilege paths into a decision model you can test
- [Zero-Trust Service and Workload Architecture](05-zero-trust-architecture.md): supplies the workload identity that most boundary labels quietly assume
- [Authentication Systems](01-authentication-fundamentals.md): where spoofing threats become token validation rules with real failure modes
- [Cryptographic Key and Data-Protection Architecture](06-encryption.md): the controls that make tampering and disclosure threats survivable when keys leak
- [Rate Limiting](../06-scaling/05-rate-limiting.md): the primary control for the automated-abuser profile and its economics
- [Multi-Tenant Isolation and Tenant Lifecycle Architecture](../06-scaling/12-multi-tenancy.md): tenant isolation is the boundary most abuse cases in this book try to cross
- [Incident Command and Learning](../11-observability/07-incident-management.md): what evidence an investigation needs, so repudiation threats stay testable
