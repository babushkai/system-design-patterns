# Repository Context and Policy Plane

## TL;DR

A repository context system is a control plane, not a collection of prompt files. It acquires facts and policies from multiple authorities, authenticates their origin, resolves scope and precedence, compiles an immutable effective snapshot, distributes that snapshot to runtimes, and records which revision governed each decision. The model may receive a human-readable projection of the result, but natural-language awareness is not enforcement. Actions are authorized again at a deterministic policy-enforcement point.

The hard problems are the same ones found in configuration, authorization, and software-supply-chain systems: ambiguous ownership, conflicting inheritance, stale replicas, rollback attacks, partial activation, tenant leakage, schema evolution, and untrusted content that impersonates instructions. Treat source authority, semantic trust, repository revision, tenant, scope, expiry, and provenance as typed fields. Never infer them from a filename or from where text happened to appear in a prompt.

This chapter owns repository context artifacts and the policy control plane. [Context Management](../17-llm-systems/08-context-management.md) owns request-time selection, compaction, and token allocation. [Tool and Runtime Contracts](./02-coding-agent-tool-design.md) owns capabilities and action enforcement. [Quality Engineering with AI Agents](./05-quality-engineering-with-ai-agents.md) owns review and code-quality gates.

---

## The Boundary: Source, Policy Snapshot, and Prompt Projection

Three objects are often collapsed into “the context,” but they have different correctness properties:

1. **Source artifacts** are repository facts, constraints, decisions, and overlays in their authored form.
2. **The effective policy snapshot** is a deterministic, immutable compilation for one tenant, repository, revision, task class, path set, and environment.
3. **The prompt projection** is a bounded, model-readable view of relevant facts and obligations.

Only the second object is suitable as an authorization input. The third helps the model propose compliant work; it cannot prevent a compromised or confused model from proposing something forbidden.

~~~mermaid
flowchart LR
    ORG[Organization policy] --> ING[Ingest and normalize]
    REPO[Trusted repository policy] --> ING
    FACTS[Repository facts and ADRs] --> ING
    TASK[Authorized task overlay] --> ING
    ING --> REG[(Versioned artifact registry)]
    REG --> COMP[Policy compiler]
    COMP --> BUNDLE[(Signed immutable bundles)]
    BUNDLE --> DIST[Distribution and activation]
    DIST --> LOCAL[Local evaluator and cache]
    LOCAL --> PROJ[Prompt-visible projection]
    PROJ --> MODEL[Model planner]
    MODEL --> PROPOSAL[Proposed action]
    LOCAL --> PEP[Policy-enforcement point]
    PROPOSAL --> PEP
    PEP -->|permit plus obligations| TOOL[Tool runtime]
    PEP -->|deny or indeterminate| STOP[No effect]
~~~

The architecture has two planes:

- The **control plane** validates, compiles, signs, distributes, revokes, and observes policy snapshots.
- The **decision plane** evaluates a pinned snapshot close to the action and returns *permit*, *deny*, *not applicable*, or *indeterminate*, plus obligations such as approval or stronger isolation.

The runtime must preserve these invariants:

- Every task attempt names exactly one effective policy-snapshot digest.
- Every governed action decision records that digest and the normalized resource identity.
- A narrower source cannot grant authority its signer does not possess.
- Activation is atomic: a runtime uses the old complete snapshot or the new complete snapshot, never a mixture.
- Untrusted repository or retrieved text can be evidence but cannot silently become policy.
- Tenant and trust-domain identity participate in every registry, cache, bundle, and decision key.

## Context Artifact Types

Not all context should obey one merge law or failure policy.

| Artifact kind | Examples | Semantic role | Typical freshness | On conflict |
|---|---|---|---|---|
| Enforceable policy | prohibited effects, required approval, path ownership, data-handling rules | Deterministic authorization or obligation | Must satisfy explicit validity and revocation policy | Apply declared combining algorithm; ambiguity is indeterminate |
| Repository fact | languages, service boundaries, build entry points, generated paths | Describes the target revision | Bound to a repository tree or release | Same-authority successor wins; unrelated claims remain a visible conflict |
| Decision record | architecture decision, exception rationale, deprecation | Explains why a constraint exists | Valid over an explicit revision/time interval | New record must supersede a named predecessor |
| Schema or contract | API schema, event schema, database ownership map | Machine-checkable interface evidence | Bound to artifact digest/version | Reject incompatible or unresolved versions |
| Procedure pointer | test command identifier, runbook, migration procedure | Locates an operation; does not grant permission to run it | Validate against the target revision | Missing or stale procedure fails visibly |
| Task overlay | requested scope, temporary branch, approved exception | Narrows or specializes one task | Attempt- or workflow-step lifetime | Cannot widen organization/repository authority |
| Derived projection | compiled summary, path index, rendered instruction block | Optimized view of canonical artifacts | Rebuilt when any dependency changes | Never authoritative without source lineage |
| Untrusted evidence | source comments, issue text, fixtures, external documents | Data the model may analyze | Snapshot and label at acquisition | Cannot participate in instruction precedence |

An artifact envelope separates metadata from payload:

~~~json
{
  "artifact_id": "repo-policy/build-and-release",
  "kind": "enforceable_policy",
  "schema_version": "policy-artifact.v3",
  "tenant": "tenant-a",
  "trust_domain": "engineering.example",
  "repository": "repo-7f4c",
  "authority": {
    "principal": "group:release-owners",
    "class": "repository_owner",
    "namespaces": ["release", "paths/services/api"]
  },
  "scope": {
    "paths": ["services/api/**"],
    "task_classes": ["implementation", "release"],
    "environments": ["development", "staging", "production"]
  },
  "source": {
    "revision": "git-object-or-registry-revision",
    "content_digest": "sha256:...",
    "supersedes": "sha256:..."
  },
  "validity": {
    "not_before": "2026-07-01T00:00:00Z",
    "expires_at": "2026-10-01T00:00:00Z",
    "revocation_epoch": 42
  },
  "sensitivity": "internal",
  "payload_ref": "cas://sha256:...",
  "attestations": ["in-toto://sha256:..."]
}
~~~

The envelope is illustrative, not a universal file format. The important design is that authority, scope, identity, validity, and provenance are validated independently of the prose or policy body.

### Authority and trust are different dimensions

A signature proves control of a key, not correctness. A repository location proves where bytes were found, not who was authorized to govern production. Evaluate at least:

- **Authenticity:** which principal produced or approved the artifact?
- **Authorization:** which policy namespaces and scopes may that principal govern?
- **Integrity:** do the bytes match the signed or content-addressed digest?
- **Freshness:** is this revision current, expired, superseded, or revoked?
- **Confidentiality:** may this task, model endpoint, and log sink receive the content?
- **Semantic reliability:** is the artifact enforceable policy, reviewed fact, generated view, or untrusted evidence?

These fields prevent two dangerous shortcuts: “signed means safe” and “inside the repository means instruction.”

### Authority is a lattice, not one global list

An organization security owner may govern network egress but not a module’s naming convention. A module owner may refine style under its directory but cannot weaken an organization-wide secret boundary. A task owner may narrow files for one attempt but cannot self-authorize a production deployment.

Represent authority as *(principal, namespace, scope, allowed operations)*. Specificity matters only after authorization. “Closest file wins” is acceptable for a formatting default when the closer owner is allowed to set it; it is not a security combining algorithm.

## Scoping, Inheritance, and Precedence

### Normalize the evaluation target first

Policy evaluation starts from canonical identities, not user-supplied strings:

~~~text
tenant and trust domain
repository identity, not only checkout directory
base/source revision and target tree digest
workspace-overlay digest
normalized repository-relative path
task and attempt identity
task class and requested effect
environment and deployment target
authenticated actor and workload identity
~~~

Resolve dot segments, Unicode normalization, case rules for the repository filesystem, mount boundaries, and symbolic links before matching selectors. A path that appears to be *docs/guide.md* but resolves outside the workspace must not inherit *docs/*** permissions.

Repository scope and filesystem scope are separate. Two tenants can check out a repository at the same local path; two worktrees can hold different commits; one task can have an uncommitted overlay. None may share an effective-policy cache entry unless the complete identity matches.

### Selectors are typed predicates

Useful selectors include path prefix or glob, language, artifact kind, task class, environment, effect class, data sensitivity, branch protection state, and repository revision range. Each selector language needs specified matching and normalization semantics. Do not mix regex, shell glob, ignore-file, and URL-prefix behavior behind one field named *pattern*.

A rule applies only when every required predicate is known and true. A missing security-relevant attribute produces *indeterminate*, not a guessed default. Optional descriptive context can instead be *not applicable*.

### Deterministic resolution algorithm

For evaluation input $x$, let $C(x)$ be artifacts that are authenticated, authorized for their namespace, valid, tenant-compatible, revision-compatible, and scope-matching. Resolve them in this order:

1. Validate envelope schema, signature/attestation chain, digest, and dependency closure.
2. Reject or quarantine artifacts outside the authenticated tenant and trust domain.
3. Resolve explicit supersession lineages and revocations; timestamps alone do not establish succession.
4. Filter by normalized repository, revision, path, task, environment, and effect.
5. Group rules by policy namespace and semantic type.
6. Apply the combining algorithm declared for that namespace.
7. Treat unresolved contradictions or missing mandatory attributes as compilation/evaluation errors.
8. Emit an explanation graph: selected rules, rejected candidates, override/exception edges, obligations, and source digests.

Different semantics require different combiners:

| Semantic type | Combining rule |
|---|---|
| Safety denial | Deny overrides permit unless a separately authorized exception policy explicitly covers the same resource and effect |
| Additive obligation | Union requirements; satisfying one does not cancel another |
| Default preference | Higher authorized authority wins; within equal authority, narrower scope wins; equal-specificity disagreement is an error |
| Singular fact | Explicit successor in the same lineage wins; otherwise expose multiple values as conflict |
| Bounded exception | Exact subject/resource/effect match, named parent rule, expiry, reason, and authorized approver required |
| Untrusted evidence | Never combined as instruction or permission |

The four-valued decision avoids unsafe Boolean coercion:

- *permit*: all required attributes were known and the action is allowed.
- *deny*: an applicable rule forbids the action.
- *not applicable*: no rule in this policy namespace governs the action.
- *indeterminate*: evaluation could not establish a safe result because input, policy, dependency, or evaluator state was invalid.

For a governed effect, *indeterminate* normally fails closed. It must not become *permit* because a client library catches an exception and returns its language’s zero value.

### Worked conflict trace

Assume:

- organization policy denies production changes without a recorded approval;
- repository policy allows release tasks to propose deployments;
- a database-module policy requires a migration owner for schema effects; and
- the task overlay limits work to application code.

A proposed production schema migration receives:

~~~text
repository allow-to-propose       -> applicable
organization approval obligation -> applicable and unsatisfied
module owner obligation          -> applicable and unsatisfied
task path scope                   -> does not include schema path

effective result: DENY
explanation: task_scope_mismatch
also required if resubmitted in scope: production_approval, migration_owner
~~~

The repository rule did not override organization policy, and the task overlay narrowed rather than expanded authority. The explanation is data, so clients do not have to reverse-engineer precedence from concatenated prose.

## Provenance and Revision Pinning

### Bind policy to the subject it governs

A branch name is mutable. “Latest policy” is not replayable. Pin:

- immutable repository object/tree identity;
- workspace-overlay digest and included path list;
- source-artifact digests;
- compiler and evaluator compatibility versions;
- compiled bundle digest;
- organization security/revocation epoch; and
- dependency/schema digests.

The task’s context manifest should be sufficient to reconstruct selection without embedding every sensitive payload:

~~~json
{
  "context_snapshot": "sha256:effective-snapshot",
  "repository_tree": "vcs-object:...",
  "workspace_overlay": "sha256:...",
  "policy_bundle": "sha256:...",
  "revocation_epoch": 42,
  "compiler": "policy-compiler.v5",
  "artifacts": [
    {"id": "org/security", "digest": "sha256:..."},
    {"id": "repo/architecture", "digest": "sha256:..."}
  ],
  "excluded": [
    {"id": "fixture/prompt.txt", "reason": "untrusted_evidence"}
  ]
}
~~~

Every model request can reference this manifest, while every effect decision records the policy bundle and the exact action input. This separates reproducibility from verbose prompt logging.

### The base-policy/head-code split

A proposed change must not govern its own review. If a pull request modifies repository policy, using the head revision’s new rule immediately lets an untrusted change grant itself network access, hide paths, or relax approval.

Maintain two revisions:

- **Authority revision:** protected base or approved control-plane revision from which enforceable policy is loaded.
- **Subject revision:** code and proposed policy changes being analyzed.

The head policy is evidence and a proposed migration until approved. Descriptive facts that genuinely changed in the head can be read with that provenance, but they must not silently acquire enforcement authority. After merge and policy deployment, a new trusted authority revision becomes eligible for later tasks.

### Provenance through compilation

Compilation should emit an attestation linking:

~~~text
source artifact digests
authorized signer/approver identities
source repository and revision
compiler identity and digest
schema and dependency versions
test/validation result digests
output bundle digest
build timestamp and validity interval
~~~

Content addressing detects accidental substitution; signatures bind an identity; authorization establishes whether that identity may publish; provenance explains transformation. All are needed.

Task pinning has one intentional exception: emergency revocation. A task pinned to bundle generation 41 must not keep an authority revoked at security epoch 42. Record that the task was invalidated, require re-evaluation under a new snapshot, and never mutate its historical decision records in place.

## Policy Compilation

### Structured source and human-readable projection

Natural language is useful for rationale, examples, and model guidance. It is a poor representation for an enforceable effect boundary because ambiguity is resolved probabilistically.

Use:

- typed policy/schema for permissions, denials, obligations, selectors, expiry, and exceptions;
- versioned facts for architecture and repository metadata;
- prose for rationale and non-security guidance; and
- a deterministic renderer that produces prompt-visible explanations from the same effective snapshot.

AI may assist policy authoring, but it should produce a proposed structured artifact that undergoes schema validation, semantic review, and policy tests. Do not ask a model at action time to translate prose into the security decision that authorizes its own tool call.

### Compilation pipeline

~~~mermaid
flowchart TB
    SRC[Versioned source artifacts] --> AUTH[Authenticate and authorize publisher]
    AUTH --> PARSE[Parse and schema validate]
    PARSE --> NORM[Normalize identities, selectors, and units]
    NORM --> DEPS[Resolve pinned dependencies]
    DEPS --> STATIC[Conflict, reachability, and safety analysis]
    STATIC --> TEST[Policy conformance tests]
    TEST --> OPT[Index or partially evaluate stable inputs]
    OPT --> EMIT[Emit immutable bundle + explanation metadata]
    EMIT --> SIGN[Sign/attest manifest]
    SIGN --> REG[(Bundle registry)]
~~~

Compilation rejects:

- an unknown or incompatible schema version;
- a publisher outside its policy namespace;
- unresolved dependencies or floating production dependencies;
- selector syntax with implementation-dependent meaning;
- two equal-authority rules that conflict without a combining rule;
- an exception that lacks parent rule, reason, approver, scope, or expiry;
- a derived artifact without source lineage; and
- a bundle whose evaluator compatibility range excludes the target fleet.

The compiled result contains both decision data and an explanation index. Optimization must preserve semantics; compare optimized and unoptimized decisions over a conformance corpus before activation.

### Policy compilation is not repository quality gating

Policy tests establish the policy plane’s own semantics: selection, combination, authorization, compatibility, and failure behavior. Whether an agent-produced implementation passes tests, security review, or merge criteria belongs to [Quality Engineering with AI Agents](./05-quality-engineering-with-ai-agents.md). The policy plane may return an obligation naming a required gate, but it does not duplicate that gate’s implementation.

## Distribution and Atomic Activation

### Immutable bundles and desired state

Publish immutable bundles addressed by digest. A small desired-state record maps a tenant/repository/environment to an approved digest, minimum evaluator version, validity interval, and revocation epoch. Updating desired state uses compare-and-swap so two publishers cannot silently overwrite one another; HTTP transports can use strong entity tags and conditional requests.

Runtimes pull or watch desired state, download missing content, and then:

1. authenticate the distribution service and bundle signer;
2. verify digest, signature, tenant, dependency closure, and anti-rollback metadata;
3. validate evaluator compatibility;
4. load the entire bundle into a staging evaluator;
5. execute activation smoke/conformance probes;
6. atomically swap the active pointer; and
7. report desired and active revisions plus any error.

Persist the last-known-good complete bundle for restart. Never overwrite it until the new snapshot is verified and active. Multiple source bundles may be compiled into one composite manifest, but their activation unit must be atomic.

### Fleet rollout

A control-plane release can change many decisions at once. Use:

- offline decision diff between old and candidate bundles over recorded, redacted inputs;
- shadow evaluation that records candidate outcomes without enforcing them;
- canary activation by explicit runtime cohort;
- convergence monitoring by desired versus active digest; and
- one-step rollback to a still-valid signed bundle.

A rollback is a new authorized desired-state transition, not permission to accept any lower version served by the network. Monotonic generations, expiry, trusted metadata, and revocation state protect against freeze and rollback attacks.

### Staleness and failure semantics

“Use cache on error” is not one safe policy. Define behavior by artifact and effect:

| Condition | Descriptive/read-only work | Governed local mutation | External or privileged effect |
|---|---|---|---|
| Refresh failed; last-known-good is valid and within its staleness budget | Continue and expose stale age | Continue only if policy explicitly permits cached evaluation | Usually deny or require a fresh authoritative check |
| Bundle expired | Omit affected facts or mark unavailable | Deny governed mutation | Deny |
| Revocation epoch advanced beyond local bundle | May inspect without effects if allowed by bootstrap policy | Deny and refresh | Deny and invalidate outstanding approval/capability |
| First startup with no verified bundle | Bootstrap-safe metadata only | Deny | Deny |
| Evaluator returns indeterminate | Show missing context if non-sensitive | Deny | Deny |
| Repository or overlay digest changed after decision | Reassemble context | Re-evaluate before mutation | Re-evaluate and reacquire any bound approval |

Staleness budgets derive from risk and recovery objectives. A style preference can survive a longer outage than a credential-revocation rule. Keep time from a trusted source where expiry is security-relevant, and account for clock uncertainty explicitly.

## Tenant Isolation and Trust Domains

Multi-tenant policy infrastructure handles sensitive repository structure, organizational rules, and decision inputs. Logical labels added after lookup are insufficient isolation.

### Isolation invariants

- Tenant and trust domain are part of every primary key, content-address namespace, cache key, watch stream, object-store prefix, encryption context, and log route.
- The authenticated workload identity selects tenant; request payloads do not self-assert it.
- A tenant publisher can govern only delegated namespaces and cannot attach itself to a different tenant’s bundle.
- Global baselines are immutable referenced artifacts or separately compiled copies, not mutable rows accidentally joined across tenants.
- Encryption keys, retention, export, and deletion follow the tenant’s policy.
- Shared evaluator processes must prove memory/cache separation; higher-risk tenants receive process or workload isolation.
- Decision logs and explanation traces are authorized independently from decision APIs.

Trust-domain federation is explicit. Importing another domain’s baseline requires a configured trust relationship, pinned trust material, namespace mapping, and a rule describing whether the foreign domain may constrain, supply defaults, or grant. Authentication of a foreign signer alone does not give it local policy authority.

### Cache safety

An effective-policy cache key includes at least:

~~~text
tenant + trust domain + repository identity
authority revision + subject tree + overlay digest
task class + environment + normalized scope set
bundle digest + revocation epoch + evaluator semantic version
~~~

Omitting a field can turn a performance optimization into cross-tenant disclosure or stale authorization. Negative decisions need the same isolation. Do not reuse a prompt projection merely because its rendered text hash matches; its sensitivity, provenance, and permitted recipients can differ.

## Context Injection Threats

Repository content is untrusted input when the repository, branch, dependency, issue, or generated file can be influenced by someone outside the policy authority. Prompt-shaped text is not rare: documentation quotes instructions, test fixtures contain adversarial strings, and source comments may intentionally discuss attacks.

### Separate channels before rendering

Classify every acquired object as:

- trusted system/operator policy;
- authorized repository policy;
- reviewed descriptive fact;
- task/user instruction within granted authority;
- untrusted evidence; or
- derived content with source lineage.

The renderer preserves these labels structurally. Untrusted evidence is quoted or delimited as data and cannot be promoted by phrases such as “system message,” “ignore previous rules,” or a filename that resembles a policy artifact. If the downstream model API flattens channels, the enforcement plane must still ignore the flattened prose and use the typed snapshot.

| Threat | Example | Control |
|---|---|---|
| Instruction impersonation | source comment tells the agent to upload secrets | Evidence classification; capability enforcement independent of model text |
| Policy-file spoofing | nested dependency contains a familiar policy filename | Resolve only registered sources under an authorized repository/trust identity |
| Self-modification | task edits policy then tries to use the relaxed rule | Authority-revision pinning; proposed policy cannot govern its own task |
| Path confusion | symlink or case alias moves a file into a more permissive scope | Canonicalize and confine path before selector evaluation and again before effect |
| Unicode/markup ambiguity | confusable rule ID or hidden rendered text | Normalize identifiers; parse canonical bytes; inspect rendered and raw forms |
| Stale generated context | summary still says an API exists after refactor | Dependency digests, rebuild invalidation, source links, freshness telemetry |
| Retrieval poisoning | external runbook contains prompt injection | Authorized retrieval, evidence-only trust class, source provenance |
| Secret smuggling | context artifact embeds credentials as “example configuration” | Classification/DLP at ingestion and rendering; secret references instead of values |
| Explanation exfiltration | denial trace reveals hidden organization rules | Audience-specific explanations and field-aware redaction |

Prompt-injection detection is defense in depth, not a proof of safety. The decisive controls are least-privilege tools, normalized authorization, sandboxing, tenant isolation, and effect receipts, covered in [Tool and Runtime Contracts](./02-coding-agent-tool-design.md).

## Versioning and Migrations

Track four versions independently:

1. **Artifact schema version** — fields and validation rules for source artifacts.
2. **Policy semantic version** — meaning of selectors, operators, outcomes, and combiners.
3. **Bundle revision/digest** — immutable compiled content.
4. **Compiler/evaluator version** — implementation that must preserve the declared semantics.

A syntactically additive field can be semantically breaking if an old evaluator ignores a new mandatory denial condition. Unknown security-critical fields fail compilation or activation; forward-compatible readers may ignore only fields explicitly declared informational.

### Migration sequence

1. Define old-to-new semantics and downgrade behavior.
2. Add conformance fixtures readable by both implementations.
3. Deploy evaluators that can understand old and new policy semantics.
4. Dual-compile and differentially evaluate representative inputs.
5. Classify every decision delta as intended, corrected, or regression.
6. Canary the new bundle and monitor activation plus decision deltas.
7. Switch authors to the new schema.
8. Retire old semantics only after no active or resumable task depends on them.
9. Preserve immutable historical bundles and evaluators, or a verified replay adapter, for audit.

Policy exceptions are data with their own lifecycle. Require owner, justification, exact scope/effect, parent rule, creation revision, expiry, and renewal history. An exception without expiry becomes an undocumented fork of policy.

### Rollback and revocation

Operational rollback selects a previously tested bundle through a new signed desired-state revision. Security revocation raises an epoch or denies a signer/artifact and invalidates affected tasks. Never “roll back” by lowering the anti-rollback counter or extending an expired bundle locally.

## Testing and Verification

Policy-plane testing is deterministic wherever possible.

| Layer | What to verify |
|---|---|
| Schema tests | Required fields, unknown critical fields, canonical encoding, limits, compatibility |
| Authorization tests | Publisher can govern only delegated tenant, namespace, and scope |
| Selector tests | Path normalization, symlinks, case behavior, Unicode, revision and environment predicates |
| Combining tests | Deny/permit/indeterminate behavior, specificity ties, exception bounds, additive obligations |
| Property tests | Narrower task overlays never widen authority; reordering source artifacts does not change results; tenant substitution never preserves a cache hit |
| Golden manifest tests | Candidate sources produce the expected effective snapshot and explanation graph |
| Differential tests | Old/new compiler, optimized/unoptimized evaluator, and pre/post-migration decisions agree except for reviewed deltas |
| Adversarial tests | Prompt-shaped evidence, spoofed filenames, malicious metadata, cyclic dependencies, oversized artifacts |
| Distribution tests | Corrupt/truncated bundle, wrong tenant, bad signature, rollback/freeze, partial download, watch gap |
| Fault tests | Control-plane outage, expired cache, clock skew, revocation during task, evaluator crash during activation |
| Isolation tests | Cross-tenant artifact/cache/log attempts and concurrent cleanup/export |
| Load tests | Compile bursts, fleet reconnect, hot repository, decision latency, memory pressure, explanation size |

Important properties can be expressed as invariants:

~~~text
same normalized input + same bundle digest + same evaluator semantics
    => same decision and explanation rule IDs

task_scope_2 is narrower than task_scope_1
    => permitted_effects(task_scope_2) is a subset of permitted_effects(task_scope_1)

revocation_epoch_2 > revocation_epoch_1
    => a decision under epoch_1 cannot authorize a new effect under epoch_2

tenant_a != tenant_b
    => no effective-snapshot, projection, or decision cache entry is shared
~~~

Line coverage is weak evidence for policy correctness. Prefer decision-table coverage, boundary values, mutation tests of allow/deny conditions, and property-based generation over the attribute space. Record which rule outcomes and combining branches were exercised.

The prompt-visible projection can also be evaluated for faithful rendering and model comprehension, but model compliance is probabilistic. Deterministic enforcement tests remain the safety oracle; implementation-quality evaluation stays in Chapter 19.05.

## Observability and Audit

### Control-plane status

Track:

- desired, downloaded, verified, staged, and active bundle digests;
- last successful fetch and activation, current stale age, and expiry horizon;
- compiler/evaluator versions and compatibility failures;
- fleet convergence by tenant, repository, environment, and runtime cohort;
- compile queue time, activation latency, download bytes, cache hit/miss, and reconnect rate;
- invalid signature, rollback, wrong-tenant, schema, dependency, and conflict failures; and
- last-known-good use and fail-closed transitions.

A process can be alive while running the wrong policy. Readiness for governed effects should include successful activation of an acceptable bundle, not merely an open port.

### Decision records

For each governed action, record:

~~~text
decision ID and timestamp
tenant, actor/workload, task and attempt
normalized action and resource identity
repository tree and workspace-overlay digest
policy bundle digest and revocation epoch
outcome, obligations, and matched rule IDs
evaluator semantic version
enforcement-point result and tool receipt reference
~~~

Store sensitive inputs by protected reference or digest where possible. Decision logs can contain repository paths, user identity, denial rationale, and policy-derived secrets; apply field-aware redaction, tenant-specific retention, access control, and deletion policy before export.

### Metrics need interpretation

A rising deny rate can mean an attack, a newly effective policy, a broken planner affordance, or incorrect scope normalization. A falling deny rate can mean improvement or a missing policy. Correlate changes with bundle activation, task mix, and indeterminate rate.

Avoid unbounded labels such as raw repository path, rule text, user prompt, commit ID, or decision ID in metrics. Keep those in sampled/secured traces or indexed audit records. Metrics use bounded tenant tiers, policy namespaces, result classes, evaluator cohorts, and normalized error codes.

## Capacity Planning

The control plane has three distinct workloads: artifact compilation, bundle distribution, and decision evaluation.

Let:

- $R$ be active repository/tenant scopes;
- $A_r$ be source artifacts for scope $r$;
- $B_r$ be compiled bundle bytes;
- $N_r$ be evaluator instances consuming scope $r$;
- $U_r$ be bundle activations per unit time;
- $Q_r$ be policy decisions per unit time;
- $L_r$ be retained decision-record bytes per decision after redaction; and
- $T_r$ be decision-record retention time.

Approximate retained storage is:

$$
S \approx S_{\text{source revisions}} + S_{\text{bundle revisions}}
  + \sum_r Q_r L_r T_r.
$$

Naive point-to-point distribution bandwidth is:

$$
D_{\text{naive}} = \sum_r U_r B_r N_r.
$$

Content-addressed deduplication, conditional fetches, regional fan-out, and shared immutable organization layers reduce transfer, but must not weaken tenant authorization. Measure reconnect bursts: after a control-plane or regional outage, $N_r$ instances can fetch simultaneously even when normal $U_r$ is low.

### Compiler capacity

Compilation cost follows dependency closure, selector indexes, conflict analysis, and test corpus—not source line count. Cache immutable dependency results by digest and rebuild only affected composites. Bound:

- artifact and dependency-graph size;
- compilation CPU/memory/time;
- concurrent builds per tenant;
- explanation-index growth; and
- queued revisions, coalescing superseded desired states where audit permits.

Do not activate an untested “fast path” when the queue grows. Backpressure publishers, preserve the current valid bundle, and expose delayed convergence.

### Evaluator capacity

Per-instance memory includes active and staged bundles, indexes, decision caches, explanation metadata, and transient evaluation state:

$$
M_{\text{instance}} \approx B_{\text{active, expanded}} + B_{\text{staged, expanded}}
  + M_{\text{indexes}} + M_{\text{cache}} + M_{\text{runtime}}.
$$

Serialized bundle size can significantly understate expanded evaluator memory. Benchmark load, steady state, activation overlap, and garbage collection using realistic policy/data shapes.

Decision latency is part of action latency. Size for the required percentile under the joint distribution of selector complexity, input size, explanation mode, cache state, and update concurrency. Keep a bounded local evaluator near the enforcement point when network policy-service latency or availability would violate the action contract; central management can still distribute and observe its bundles.

### Sharding and hot spots

Shard registries and compile queues by authenticated tenant/repository identity. Popular organization baselines and large monorepos create hot dependency nodes; immutable content can be cached broadly, while authorization metadata and tenant-specific composites remain isolated. Watch fan-out needs bounded queues and resynchronization tokens so a slow runtime cannot retain an unbounded update history.

## Failure Modes

### Tampered or unauthorized bundle

**Failure:** a mirror, compromised publisher, or wrong-tenant registry entry supplies valid-looking policy.

**Detection:** digest/signature failure, signer lacks namespace authority, tenant mismatch, or provenance chain is incomplete.

**Response:** reject before staging, keep a still-valid last-known-good snapshot, emit a security event, and do not “temporarily” bypass verification.

### Freeze or rollback attack

**Failure:** the distributor repeatedly serves an old signed bundle whose policy is weaker.

**Detection:** generation below the recorded floor, expired metadata, stale desired state, or revocation epoch mismatch.

**Response:** deny governed effects that require freshness; refresh through an authenticated path. A signature on stale bytes is not sufficient.

### Partial activation

**Failure:** organization policy updates while repository policy remains old, producing a combination that was never compiled or tested.

**Detection:** composite manifest/dependency digest mismatch.

**Response:** atomic pointer swap only after the full dependency closure loads and validates.

### Precedence drift across evaluators

**Failure:** different runtime versions interpret specificity, globs, unknown fields, or indeterminate differently.

**Detection:** conformance-corpus disagreement and evaluator-version telemetry.

**Response:** block incompatible activation, roll forward/back the evaluator cohort, and keep semantic version separate from bundle syntax.

### Policy self-modification

**Failure:** an agent changes a policy source and attempts a newly allowed effect in the same task.

**Detection:** subject policy digest differs from pinned authority revision.

**Response:** treat the change as a proposal; require the independent policy publication workflow and a new task snapshot.

### Stale repository facts

**Failure:** a derived context still references removed commands, owners, schemas, or service boundaries.

**Detection:** dependency digest mismatch, failed fact validation, high missing-path rate, or source revision outside validity.

**Response:** rebuild from canonical sources; omit stale optional facts visibly and fail procedures that cannot be resolved.

### Context injection succeeds at the model layer

**Failure:** untrusted source text persuades the model to propose a forbidden action.

**Detection:** policy denial, anomalous proposal telemetry, or adversarial evaluation.

**Response:** no effect occurs because the enforcement point evaluates typed action data. Investigate and improve classification/rendering without broadening capabilities.

### Cross-tenant cache or log leak

**Failure:** a cache key omits tenant/repository identity or a shared log export contains another tenant’s decision input.

**Detection:** isolation canary, authorization audit, tenant-tag mismatch, or data-loss alert.

**Response:** disable affected cache/export, rotate exposed secrets where relevant, invalidate derived artifacts, and perform tenant-scoped incident response.

### Control-plane outage

**Failure:** runtimes cannot fetch desired state or report status.

**Detection:** fleet fetch failures, increasing stale age, missing status heartbeats.

**Response:** continue only under artifact-specific last-known-good rules; fail closed at expiry/revocation boundaries; stagger recovery to avoid a reconnect storm.

### Policy complexity explosion

**Failure:** broad iteration, large data joins, or explanation expansion drives latency and memory beyond the enforcement SLO.

**Detection:** compile/evaluation percentile regression, allocation growth, cache churn, or activation OOM.

**Response:** index stable attributes, precompute safe partial results, split independent policy namespaces, bound explanations, and re-run semantic equivalence tests after optimization.

## Decision Framework

### Classify each artifact

Ask in order:

1. Is this enforceable policy, reviewed fact, rationale, procedure, task overlay, derived view, or untrusted evidence?
2. Who is authorized to publish this kind for this tenant, namespace, repository, and path?
3. What immutable subject revision and validity interval does it describe?
4. Can it grant, deny, add obligations, supply a default, or only inform?
5. What combines it with parent, peer, and narrower artifacts?
6. What happens on contradiction, missing attributes, expiry, revocation, or distribution failure?
7. Which principals, model endpoints, evaluators, and logs may receive its content?
8. Which evidence proves compilation, activation, decision, and enforcement?

If those questions do not have typed answers, the artifact is documentation—not a reliable policy input.

### Choose a deployment topology

| Topology | Appropriate when | Primary cost/risk |
|---|---|---|
| Repository-local resolution | One trusted team, low-risk local effects, limited central governance | Weak fleet visibility, revocation, and cross-repository consistency |
| Central decision service | Inputs require globally current policy and network availability meets the action contract | Service latency/availability on every decision and sensitive input centralization |
| Centrally managed, locally evaluated bundles | Large fleet, low-latency enforcement, offline tolerance with explicit staleness rules | Distribution, convergence, evaluator skew, and cache isolation |
| Hybrid organization baseline + repository/module sources | Organization controls safety while domain owners control local conventions | Most capable and usually most complex precedence/provenance model |

The hybrid is common, but it is safe only when delegated namespaces and combining rules are explicit. Copying several files into a prompt is not a hybrid control plane.

### Decide whether a rule belongs in the prompt

- If violation can cause an unauthorized or irreversible effect, enforce it outside the model and optionally explain it in the prompt.
- If it describes architecture needed to form a correct plan, include the authorized, revision-matched fact through the runtime context assembler.
- If it is a code-quality acceptance criterion, reference the quality pipeline rather than reimplementing it as policy prose.
- If it defines a tool request or effect contract, keep it in the tool registry/runtime.
- If it is untrusted material under analysis, render it as evidence and never as an instruction.

### Production readiness review

A policy plane is not ready until it can demonstrate:

- deterministic effective-policy compilation and explanation;
- protected authority/subject revision separation;
- signed/content-addressed bundles with rollback and freeze protection;
- atomic activation and last-known-good recovery;
- defined stale, expiry, revocation, and indeterminate behavior per effect;
- tenant-isolated registries, caches, evaluators, and logs;
- differential migration and evaluator conformance testing;
- decision-to-tool-receipt audit linkage;
- fleet convergence and capacity evidence; and
- recovery exercises for control-plane loss and mass refresh.

## Key Takeaways

1. Repository context is a versioned control-plane input; the prompt is only one derived view.
2. Authenticity, authorization, integrity, freshness, confidentiality, and semantic trust are separate checks.
3. Authority and specificity are independent. A closer or narrower source cannot weaken a rule it has no authority to govern.
4. Pin authority revision, subject revision, overlay, bundle digest, evaluator semantics, and revocation epoch.
5. A proposed policy change cannot authorize its own task.
6. Compile structured policy deterministically, distribute immutable bundles, and activate the dependency closure atomically.
7. Define last-known-good, expiry, revocation, and indeterminate behavior by effect risk.
8. Treat repository and retrieved text as potentially adversarial evidence; capabilities and policy enforcement remain outside the model.
9. Tenant identity belongs in every storage, cache, distribution, decision, and audit boundary.
10. Test semantic decisions, precedence properties, migrations, isolation, faults, and load—not arbitrary file length.

---

## References

- [EditorConfig Specification](https://spec.editorconfig.org/) — a stable example of hierarchical discovery, explicit roots, matching, and precedence semantics
- [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12) — machine-readable artifact schema and validation vocabulary
- [CUE Language Specification](https://cuelang.org/docs/reference/spec/) — constraint unification and conflict-as-bottom semantics
- [OASIS XACML 3.0 Core Specification](https://docs.oasis-open.org/xacml/3.0/xacml-3.0-core-spec-en.pdf) — authorization decisions and explicit policy-combining algorithms
- [NIST SP 800-162: Guide to Attribute Based Access Control](https://csrc.nist.gov/pubs/sp/800/162/upd2/final) — subject, object, action, environment, and policy attribute model
- [Open Policy Agent: Bundles](https://www.openpolicyagent.org/docs/management-bundles) — signed policy/data distribution and bundle activation
- [Open Policy Agent: Status](https://www.openpolicyagent.org/docs/management-status) — desired/active revision and activation-failure reporting
- [Open Policy Agent: Policy Testing](https://www.openpolicyagent.org/docs/policy-testing) — deterministic policy tests and coverage
- [Open Policy Agent: Decision Logs](https://www.openpolicyagent.org/docs/management-decision-logs) — decision identity, policy metadata, audit, and redaction
- [The Update Framework Specification](https://theupdateframework.github.io/specification/) — trusted metadata, expiry, rollback, freeze, and key-compromise resilience
- [in-toto Attestation Framework Specification](https://github.com/in-toto/attestation/blob/main/spec/README.md) — provenance statements linking subjects to claims
- [SLSA v1.2 Specification](https://slsa.dev/spec/v1.2/) — build provenance and supply-chain integrity model
- [RFC 9110: HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110.html) — strong validators and conditional requests for cache validation and compare-and-swap
- [SPIFFE Specifications](https://spiffe.io/docs/latest/spiffe-specs/) — workload identity, trust domains, and trust-bundle distribution
- [OWASP LLM Prompt Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html) — prompt-injection threat patterns and defense in depth
- [Git Revision Parsing](https://git-scm.com/docs/git-rev-parse) — resolving mutable revision expressions to object identities
- [NIST SP 800-218: Secure Software Development Framework](https://csrc.nist.gov/pubs/sp/800/218/final) — protected development artifacts, provenance, and secure change practices
- [Context Management](../17-llm-systems/08-context-management.md) — request-time context materialization, budgeting, compaction, and memory
- [Tool and Runtime Contracts for Coding Agents](./02-coding-agent-tool-design.md) — capability issuance, policy enforcement, sandboxing, and effect receipts
- [Quality Engineering with AI Agents](./05-quality-engineering-with-ai-agents.md) — implementation review, testing, and quality gates
