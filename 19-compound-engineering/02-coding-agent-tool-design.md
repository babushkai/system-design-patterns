# Tool and Runtime Contracts for Coding Agents

## TL;DR

A production tool is the versioned protocol between a probabilistic planner and the systems it can affect. It requires typed arguments, normalized resource identity, authorization, bounded execution, cancellation semantics, structured results, provenance, redaction, and a durable receipt. Use the smallest primitives that preserve intent—read, search, patch, execute, inspect, and invoke a scoped connector—and expose dangerous composition to policy.

Scope: the tool registry, request/effect protocol, runtime boundary, and tool-specific failures. [Platform Fundamentals](./01-compound-engineering-fundamentals.md) defines task, isolation, scheduling, and approval; [Context Management](../17-llm-systems/08-context-management.md) covers runtime selection; [Repository Context and Policy](./03-agent-context-engineering.md) covers repository policy artifacts.

---

## A Tool Call Is a Protocol Message

The model proposes a tool name and arguments. The runtime must assume both can be malformed, stale, adversarially influenced, or inconsistent with current task state.

A generic request envelope should identify:

```json
{
  "protocol_version": "tool-call.v1",
  "task_id": "task_7f2",
  "attempt_id": "attempt_3",
  "logical_effect_id": "effect_apply_patch_12",
  "tool": "repository.apply_patch",
  "tool_version": "2.1.0",
  "arguments": {
    "workspace_id": "ws_91",
    "expected_revision": "sha256:...",
    "patch": "..."
  },
  "capability_id": "cap_...",
  "deadline": "2026-07-18T20:30:00Z",
  "idempotency_key": "task_7f2:patch:12"
}
```

The runtime adds the identity and capability fields; the model must not be trusted to supply them. The response separates execution status from domain outcome:

```json
{
  "receipt_id": "receipt_b84",
  "status": "succeeded",
  "started_at": "...",
  "finished_at": "...",
  "normalized_action": {
    "workspace_id": "ws_91",
    "paths": ["src/auth/session.ts"],
    "operation": "patch"
  },
  "result": {
    "revision": "sha256:...",
    "changed_paths": ["src/auth/session.ts"],
    "hunks_applied": 2
  },
  "artifacts": [],
  "redactions": [],
  "error": null
}
```

An HTTP 200 or process exit code zero does not always mean the requested domain effect occurred. A search may be truncated, a patch may apply to an unexpected revision, a deployment API may accept work asynchronously, and a browser click may land on a different element after the page changed. Result schemas need domain-specific evidence.

### Contract invariants

- Unknown fields and unsupported versions fail explicitly according to compatibility policy.
- Resource identifiers are normalized before authorization and execution.
- The authorization decision binds the normalized action, not the model’s raw string.
- A deadline covers queueing and execution; it is not reset silently by retries.
- Results identify truncation, partial success, ambiguity, and asynchronous acceptance separately.
- Receipts are immutable and content-address large outputs rather than embedding unbounded text.
- Retrying a logical effect is either idempotent or enters reconciliation before another attempt.

---

## Registry, Schema, and Compatibility

The tool registry is a control-plane artifact. Each entry includes:

```text
tool identity and semantic version
request and response schemas
effect class and reversibility
required capability dimensions
default timeout and maximum output
supported cancellation mode
idempotency and reconciliation contract
redaction rules
runtime implementation digest
owner and deprecation state
```

Pin the registry revision when a task starts. Changing a description or schema mid-run changes the planner’s action space and can invalidate earlier approval. A registry rollout should be canaried like any other runtime release.

### Schema evolution

Prefer additive compatible changes: optional request fields with defined defaults and additional response fields that old consumers ignore. Renaming a field, changing units, widening an effect, or altering default scope is a breaking change even if the JSON still parses.

Semantic differences deserve a new tool version. For example, changing `delete(path)` from “move to workspace trash” to “permanent recursive removal” cannot hide behind implementation deployment. Keep old and new versions concurrently only for a bounded migration window, then reject new tasks that request the retired version.

Descriptions help planning but are not policy. The effect class and capability requirements are structured registry fields enforced independently from natural language.

---

## Capability Model

A capability grant should answer:

```text
who:       tenant, user/service principal, task, attempt
what:      tool and operation
where:     workspace, repository, path set, host, API account
when:      issuance, expiry, deadline, cancellation epoch
how much:  calls, bytes, CPU, money, rate
under what conditions: approval, target revision, environment, policy version
```

The runtime attenuates capabilities when delegating. A review subtask can receive repository reads and test-result inspection without inheriting patch, network, or merge privileges. Capabilities are non-transferable across tasks and attempts unless an explicit durable workflow transition reissues them.

Authorization occurs after normalization. A grant for `workspace/src/**` must not be bypassable through `..`, symlinks, alternate path encodings, case folding, hard links, archive extraction, or a race between validation and open. Prefer directory file descriptors and operating-system primitives that resolve beneath an already-open root; re-check object identity at the point of effect.

Network policy similarly authorizes resolved destinations and protocols, not merely a user-supplied URL string. Redirects, DNS rebinding, proxy configuration, IPv4/IPv6 aliases, and cloud metadata addresses belong in the threat model.

---

## Repository Read and Search Tools

Read tools appear harmless but can leak secrets, flood context, and create inconsistent views.

### Snapshot consistency

Every read and search result identifies the workspace revision it observed. A multi-step inspection should either use an immutable snapshot or acknowledge that files may change between calls. If a patch changes the workspace, subsequent search results carry the new revision; the runtime must not present cached results from the old revision as current.

### Bounded output

Support line, byte, match, depth, and file-count limits. Truncation is a typed result with continuation state, not an ellipsis that the planner may mistake for end-of-file. Continuation tokens bind query, revision, tenant, and sort order so they cannot be reused against another snapshot.

Search should expose:

- literal, regular-expression, filename, symbol, and structural modes;
- ignored/binary/generated-file policy;
- deterministic ordering;
- matched ranges and source revision;
- count-only and path-only modes to control output;
- timeout and truncation metadata.

Do not make the model parse thousands of lines to find a three-line match. Tool quality changes system behavior because it controls how much relevant evidence reaches the planner.

### Secret boundaries

Classify paths before returning content. Deny or redact credential files, private keys, environment stores, browser profiles, package-manager credentials, and platform metadata by policy. A read-only task does not need secret access merely because a build would.

---

## Patch and File-Mutation Tools

Prefer an intent-preserving patch operation over whole-file replacement for existing files. A patch request includes the expected source revision or surrounding context and fails on ambiguity. It should never guess which of several identical regions the model intended.

The mutation transaction is:

1. Normalize and authorize every affected path.
2. Verify the workspace revision or preimage hashes.
3. Stage the patch in memory or a temporary file inside the workspace.
4. Validate syntax and path constraints where cheap and deterministic.
5. Atomically replace each file where the filesystem permits.
6. Compute the resulting diff and revision digest.
7. Persist the receipt before returning success.

Multi-file mutation is not automatically atomic. If a tool can fail after changing only some files, its result must enumerate applied and unapplied changes, and the platform should restore the preimage or mark the workspace for repair. A Git-backed workspace can use the index or tree objects as a transaction boundary without committing prematurely.

Whole-file creation remains useful for new generated artifacts. It still needs size limits, newline/encoding policy, executable-bit control, and protection against writing through links or outside the workspace.

### Deletion and rename

Deletion is workspace-local until integration, which makes it reversible. The tool should distinguish deleting a tracked path, removing an untracked generated artifact, and recursively removing a directory. Renames preserve intent better than delete-plus-add and improve review, but correctness depends on content identity, not Git’s heuristic rename display.

---

## Shell Execution

Shell access is an interpreter for other capabilities. Even without direct network tools, a shell may invoke `curl`, a package manager, a compiler plugin, or code from the repository. Treat the reachable executable and filesystem set as part of the grant.

### Request contract

Use an argument vector when shell syntax is unnecessary. If a shell is required, make the shell and mode explicit. The request declares:

- working directory rooted in the workspace;
- environment allow-list and redacted variables;
- stdin mode;
- wall-clock and idle timeout;
- CPU, memory, process, file, and output limits;
- network profile;
- expected artifact paths when known;
- whether a pseudo-terminal is required.

Avoid constructing commands by interpolating untrusted strings. Tool adapters should pass arrays to process APIs and quote only at a deliberate shell boundary.

### Process lifecycle

Cancellation first prevents new child creation, then signals the process group, waits a bounded grace period, and force-terminates remaining processes. Descendants that escape the process group must still be contained by the sandbox or cgroup. Completion means all relevant output is drained and the runtime has accounted for background processes; returning while a server continues with inherited credentials violates task isolation.

### Output protocol

Capture stdout and stderr separately with timestamps or sequence numbers when ordering matters. Stream bounded previews to the planner and store large output as an artifact. Preserve the final exit status, terminating signal, timeout reason, resource usage, and truncation point. A command killed for output overflow differs from a test failure.

### Build and test execution

Repository code is untrusted executable input. Dependency installation, test discovery, compiler plugins, hooks, and generated build scripts all execute inside the sandbox. Use immutable base images, pinned toolchains, scoped caches, egress policy, and no ambient platform credentials. A “review-only” task must not automatically run code unless its workload contract allows execution.

---

## Browser and External Connector Tools

External systems introduce mutable state that Git cannot roll back.

### Structured connectors

Prefer a domain adapter over generic browser automation when an API exists. The adapter validates account and tenant server-side, exposes stable resource IDs, models asynchronous operations, supports idempotency keys, and returns provider receipts. A model-supplied `account_id` is never sufficient authorization.

Classify connector operations as read, draft, publish, send, delete, deploy, charge, or equivalent. Approval binds the normalized resource, destination, and effect class. If a redirect or refreshed page changes those, the action requires a new decision.

### Browser state

A browser tool records tab/frame identity, URL, navigation generation, element locator strategy, and a screenshot or accessibility-tree digest when needed. DOM indexes and coordinates are ephemeral; clicking “element 12” after navigation is unsafe. Re-resolve against the expected page state immediately before the effect.

Downloads are untrusted artifacts. Uploads are data-egress effects. Clipboard, browser profiles, authenticated cookies, local storage, and extension APIs all belong to the capability boundary.

### Ambiguous completion

If a publish request times out, query by idempotency key or provider operation ID before retrying. If no authoritative query exists, surface an ambiguous state for human reconciliation. “Probably failed” is not a safe basis for sending another email, opening another pull request, or triggering another deployment.

---

## Subtask and Agent Invocation

A subtask tool creates another durable task or attempt with:

- a bounded objective and expected output schema;
- an immutable input/context revision;
- attenuated capabilities;
- child budgets and deadline;
- cancellation linkage;
- ownership of files, artifacts, or analysis scope;
- a result receipt and evidence references.

The child returns conclusions and artifacts, not an unbounded transcript. The parent must validate the result against the expected schema and current task state. If two children can mutate the same path, the platform has created a merge protocol and must model conflicts rather than hoping scheduling prevents them.

Avoid recursive delegation without depth, fanout, and total-budget bounds. A child task waiting on its parent while the parent waits on the child is a workflow deadlock, not a reasoning problem.

---

## Runtime Isolation

Isolation layers address different failures:

| Layer | Protects against | Does not by itself protect against |
|---|---|---|
| Git branch/worktree | Accidental workspace interference | Malicious processes, network access, host filesystem |
| Container + namespaces | Filesystem/process/network separation | Kernel compromise, misconfigured mounts, ambient credentials |
| MicroVM | Stronger tenant/kernel boundary | Authorized but unsafe external effects |
| Capability-scoped adapters | Excess resource/effect authority | Compromise of the adapter or underlying account |
| Approval gate | Unwanted high-impact action | Misleading evidence or overly broad approval |

Compose layers based on threat and workload. Strong sandboxing does not replace tool authorization; a perfectly isolated process can still call an authorized production API destructively.

Runtime images and tool adapters are supply-chain artifacts. Pin by digest, generate provenance, inventory dependencies, scan and canary releases, and retain a fast rollback path. Model and runtime versions roll independently so a regression can be isolated.

---

## Reliability Semantics

| Condition | Required result |
|---|---|
| Invalid arguments | Typed validation error; no effect intent |
| Policy denial | Denial reason and policy version; no execution |
| Queue deadline exceeded | Expired before execution; no effect |
| Runtime timeout before request sent | Safe failure; retry under remaining deadline |
| Timeout after request may have been sent | Ambiguous; reconcile before retry |
| Partial multi-file change | Enumerated partial result or restored preimage; workspace marked if repair needed |
| Output truncated | Successful/failed execution plus explicit truncation and artifact continuation |
| Cancellation during effect | No new work; receipt/reconciliation for the in-flight operation |
| Worker loss after success | Recover receipt by logical effect ID or inspect authoritative resource |
| Tool version unavailable on resume | Migrate task explicitly or resume with pinned compatible implementation |

Tool errors are data for planning, but repeated failures need orchestration limits. The model should not be able to turn a permission denial into an infinite retry loop or a costly alternate route.

---

## Observability and Audit

Record request schema version, normalized action, capability and policy decision, runtime digest, queue/execute time, resource usage, result category, receipt, artifacts, redactions, and reconciliation state. Arguments and output pass through field-aware redaction before durable logging.

Monitor:

- validation and policy-denial rates by tool version;
- latency and saturation by queue and runtime class;
- timeout, cancellation, and forced-termination rates;
- ambiguous-effect age and reconciliation success;
- output truncation and artifact volume;
- sandbox escapes or denied filesystem/network attempts;
- patch conflict and preimage-mismatch rates;
- connector effects by destination and approval class;
- version rollout regressions.

High denial rates can mean attack, poor planner affordances, or an overly narrow tool. Diagnose rather than automatically broadening permission.

---

## Verification and Fault Injection

Test tool adapters as security- and transaction-critical services:

- schema fuzzing, unknown fields, boundary sizes, invalid Unicode, and numeric overflow;
- path traversal, symlink and rename races, archive extraction, and case-normalization collisions;
- process trees that ignore signals, fork repeatedly, fill output, disk, or memory, and keep inherited descriptors;
- network redirects, DNS changes, proxy bypass, metadata endpoints, and exfiltration attempts;
- crash after intent, during effect, after effect, and before receipt persistence;
- duplicate idempotency keys and concurrent refresh of the same logical effect;
- stale capability, cancelled task, expired deadline, and policy-version changes;
- tool registry upgrade/downgrade and durable-task resume;
- redaction canaries in inputs, outputs, exceptions, traces, and artifacts.

Run conformance suites against every tool implementation. A mock that always returns success does not validate cancellation, ambiguity, partial effects, or sandbox behavior.

---

## Decision Framework

Use a purpose-built structured tool when the operation is common, sensitive, or externally stateful. Use shell execution for composable workspace-local engineering tasks when a sandbox can contain the closure. Use browser automation when no stable API exists and accept the higher state-observation and ambiguity burden. Create a subtask only when scope, ownership, budget, and result can be bounded.

Fewer orthogonal primitives are easier to secure and teach than dozens of overlapping convenience tools. But one universal shell or browser tool makes policy blind to intent. The useful middle ground is a small primitive set plus domain adapters for irreversible systems.

---

## Key Takeaways

- Treat every tool call as a versioned, authorized, bounded protocol with a durable receipt.
- Normalize resources before authorization and bind capabilities to task, attempt, scope, budget, deadline, and policy.
- Model partial and ambiguous completion explicitly; retry only after idempotency or reconciliation is established.
- Repository reads need snapshot identity and truncation semantics; mutations need preimage checks and transactional repair.
- Shell, browser, and connector tools amplify authority and require different isolation and effect controls.
- Tool quality determines what evidence reaches the planner, but policy and the runtime—not descriptions—enforce safety.

---

## References

- [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12)
- [Open Container Initiative Runtime Specification](https://github.com/opencontainers/runtime-spec)
- [SLSA v1.2 Specification](https://slsa.dev/spec/v1.2/)
- [in-toto Attestation Framework](https://github.com/in-toto/attestation)
- [NIST SP 800-218: Secure Software Development Framework](https://csrc.nist.gov/pubs/sp/800/218/final)
- [Git Worktree Documentation](https://git-scm.com/docs/git-worktree)
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)
- [Coding Agent Platform Fundamentals](./01-compound-engineering-fundamentals.md)
- [Workflow Effect Protocols](../18-workflow-job-systems/06-retry-idempotency-compensation.md)
