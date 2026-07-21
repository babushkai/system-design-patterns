# Edge Gateway and API Mediation

## TL;DR

An edge gateway is a policy-enforcement and traffic-mediation boundary between untrusted clients and internal services. It terminates transport, canonicalizes the request, selects a versioned route, authenticates the caller, obtains an authorization decision, applies admission controls, and forwards or composes a bounded upstream request. The gateway is not the owner of business state and must not become a hidden monolith.

The hard design questions are dependency behavior and blast radius. What happens when token keys, authorization, rate-limit state, discovery, or configuration distribution is stale? Can a route update become active before its upstream exists? Does an aggregation endpoint fit inside one end-to-end deadline? Can retries multiply one client request into a fleet incident? Can one region admit traffic safely when a global dependency is unavailable?

Keep gateway configuration immutable and revisioned, make stage ordering explicit, enforce request normalization before security decisions, and attach every upstream effect to the authenticated identity, route revision, deadline, and retry budget that governed it.

Scope: edge mediation, route compilation, dependency failure, and bounded request composition. [Authentication Systems](../10-security/01-authentication-fundamentals.md), [OAuth 2.0 and OpenID Connect](../10-security/02-oauth2-openid-connect.md), [Authorization Patterns](../10-security/07-authorization-patterns.md), [API Security](../10-security/04-api-security.md), [Rate Limiting](../06-scaling/05-rate-limiting.md), and [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md) cover the linked mechanisms.

---

## System Boundary and Invariants

The gateway maps the public transport contract to internal destinations; each service retains its domain authorization and invariants. Gateway admission does not confer downstream trust.

~~~mermaid
flowchart LR
    CLIENT[Untrusted client] --> EDGE[Transport termination and normalization]
    EDGE --> ROUTE[Compiled route match]
    ROUTE --> AUTHN[Authentication interface]
    AUTHN --> AUTHZ[Authorization interface]
    AUTHZ --> ADMIT[Quota, concurrency, and size admission]
    ADMIT --> MEDIATE[Transform, aggregate, or forward]
    MEDIATE --> DISC[Service discovery]
    DISC --> UP[Upstream service]

    CFG[Gateway control plane] --> SNAP[Signed immutable route snapshot]
    SNAP --> EDGE
    IDP[Identity/key plane] --> AUTHN
    PDP[Policy decision point] --> AUTHZ
    LIMIT[Rate-limit state] --> ADMIT
~~~

### Invariants

1. **One normalized request:** routing, authentication, authorization, signing, caching, and forwarding agree on the request target and header semantics.
2. **Deny before effect:** no body forwarding or upstream side effect occurs before required identity and policy decisions.
3. **Untrusted headers stay untrusted:** identity and client-address headers from the public side are removed or overwritten.
4. **One active configuration snapshot:** all route dependencies become active atomically or the previous valid snapshot remains.
5. **Bounded work:** request bytes, decompressed bytes, header count, streams, aggregation fan-out, retries, and response buffering have limits.
6. **End-to-end deadline:** each internal attempt receives only its allocated portion of the client or server deadline.
7. **Policy revision travels with the request:** downstream services can audit the authenticated principal and decision context without trusting client-supplied metadata.
8. **Failure mode is per route:** identity, authorization, quota, and discovery outages do not fall back through one global “open” switch.
9. **Region independence is explicit:** a regional gateway knows which control-plane state it may use locally and which effects require global freshness.
10. **Business invariants remain downstream:** a gateway can reject unauthorized traffic but cannot replace domain validation or transactional consistency.

## Request Processing Pipeline

Order is a security and correctness property:

~~~text
1. accept connection under listener and tenant limits
2. parse transport with one strict protocol interpretation
3. canonicalize authority, path, query, headers, and client identity boundary
4. select an immutable route and request schema
5. enforce size/decompression/content-type limits needed before body processing
6. authenticate credentials
7. authorize principal + action + normalized resource + context
8. reserve quota/concurrency/cost budget
9. construct the upstream request or aggregation plan
10. propagate deadline, trace, trusted identity context, and idempotency metadata
11. execute bounded attempts
12. normalize the response, release reservations, and emit an audit outcome
~~~

Changing this order can create bypasses. Authorizing a raw path before dot-segment or percent-decoding normalization lets the gateway and upstream disagree about the resource. Decompressing before a compressed-size and expanded-size policy invites memory exhaustion. Charging a rate-limit key derived from an unverified token allows attackers to choose another principal’s bucket.

### Strict message boundaries

The gateway and every next hop must agree on:

- request-target and authority normalization;
- duplicate and conflicting length/framing fields;
- header combination and case rules;
- hop-by-hop field removal;
- percent-encoding and path normalization;
- transfer/content encoding;
- trailer acceptance;
- maximum header and body sizes; and
- HTTP version translation.

Reject ambiguity instead of choosing a “helpful” interpretation. HTTP request smuggling exists when two parsers disagree about where one message ends and the next begins. Reusing a parsed, typed request object is safer than forwarding an almost-raw byte stream through a different parser.

### Trusted forwarding context

At the public boundary:

1. remove client-supplied internal identity, policy, forwarding, and trace fields;
2. derive the peer address from the authenticated connection and trusted proxy chain;
3. construct new forwarding metadata under a documented trust policy;
4. attach workload-authenticated gateway identity to the upstream connection; and
5. bind principal/context metadata to that connection or sign it when it crosses an independent trust boundary.

A header named *user-id* is not an identity proof. Downstream services authorize the authenticated gateway workload and validate the integrity, audience, expiry, and provenance of delegated caller context.

## Routing as Compiled Policy

### Route key and action

A route may match:

- listener and tenant;
- normalized authority;
- method;
- normalized path template;
- protocol/content type;
- API version;
- authenticated caller class;
- bounded header or query predicates; and
- rollout cohort.

Its action names:

- upstream service identity and protocol;
- request/response schema;
- authentication and authorization policy;
- quota/cost class;
- timeout and retry policy identifiers;
- transformation or aggregation plan;
- cache policy;
- observability/audit class; and
- rollout and fail behavior.

Keep named policy objects separate from route instances so a change to one timeout or authorization policy has an explicit dependency graph and blast radius.

### Deterministic precedence

“First route in a file wins” turns file ordering into security policy. Compile routes into a deterministic match structure and reject ambiguous overlaps.

A useful precedence order is:

1. exact authority before wildcard authority;
2. exact method before any-method;
3. static path segment before typed parameter before catch-all;
4. more constrained content/version predicate before less constrained;
5. explicit priority only as a reviewed tie-breaker.

The compiler should detect:

- unreachable routes shadowed by broader rules;
- two equally specific routes with different security policies;
- a public route overlapping an authenticated route;
- path templates that normalize to the same resource;
- rewrite loops;
- upstream references with no compatible discovery resource;
- transformation schemas incompatible with either side; and
- a retry policy attached to an operation not declared safe for it.

### Route revision is request metadata

Each accepted request records the active configuration digest and route ID. This enables:

- replaying why a request selected an upstream;
- comparing old and candidate routing in shadow mode;
- attributing failures to a rollout;
- determining which requests saw a revoked policy; and
- measuring version/deprecation usage without high-cardinality path labels.

## Authentication and Authorization Interfaces

Authentication establishes a principal; authorization decides whether that principal may perform the normalized action on the normalized resource.

### Authentication interface

The gateway may validate a credential locally using a pinned, refreshed key set or call an authoritative validation service. The result is typed:

~~~text
principal identity
credential issuer and audience
authentication method and assurance
tenant and subject attributes
issued, expiry, and validation times
credential/token identifier where policy allows logging
validation source and key/config revision
~~~

The interface must distinguish:

- invalid credential;
- expired or not-yet-valid credential;
- unknown issuer/key;
- validator unavailable;
- key/configuration stale;
- caller not applicable to this route; and
- successful authentication with bounded claims.

Do not convert validator unavailability into “anonymous” and then match a public fallback route. Route selection that depends on identity may require a preliminary route class followed by authenticated refinement, with ambiguity rejected.

Local validation reduces request-path dependency but needs secure key distribution, issuer/audience checks, revocation semantics, and maximum staleness. Introspection gives fresher centralized state but puts latency and availability on every request. The canonical protocol and threat analysis live in [OAuth 2.0 and OpenID Connect](../10-security/02-oauth2-openid-connect.md) and [JWT Tokens](../10-security/03-jwt-tokens.md).

### Authorization interface

The policy decision input includes:

~~~text
tenant and authenticated principal
normalized route action and resource attributes
request method and content classification
client/workload and network context
target service and environment
relevant resource version or ownership attributes
policy revision and requested obligations
~~~

The result is not merely Boolean:

- **permit** with obligations, such as field filtering, approval, or stronger audit;
- **deny** with an audience-safe reason;
- **not applicable** when the policy set does not govern the request; or
- **indeterminate** when inputs, policy, or evaluator state are incomplete.

For protected routes, indeterminate normally denies. A gateway may cache decisions only when the cache key includes all security-relevant input, policy revision, tenant, and expiry. Resource-level authorization that depends on current domain state often remains in the service; the gateway can enforce coarse route permission and transmit verified context.

See [Authorization Patterns](../10-security/07-authorization-patterns.md) for policy models and [Zero Trust Architecture](../10-security/05-zero-trust-architecture.md) for workload-to-workload enforcement.

## Admission and Rate-Limit Dependencies

The gateway is a natural admission point, but the rate-limit algorithm is only one component. The full algorithmic treatment is in [Rate Limiting](../06-scaling/05-rate-limiting.md).

### Hierarchical budgets

Admission may constrain:

~~~text
edge listener and source network
tenant or account
authenticated principal or API client
route and cost class
region and upstream service
concurrent expensive operations
response or egress bytes
~~~

Charge a stable, verified identity. IP-based limits are a coarse abuse signal, not an account quota. One request may consume weighted cost rather than one unit; an aggregation or export endpoint should not cost the same as a cached lookup.

### Local and global state

| Design | Strength | Limitation |
|---|---|---|
| Gateway-local counter | Low latency and survives central outage | Limit is multiplied by gateway replicas and regions |
| Central synchronous decision | Strong shared quota | Adds latency and a request-path dependency |
| Leased regional allowance | Bounded global overshoot with local fast path | Requires allocation, expiry, and return/rebalancing logic |
| Asynchronous accounting | High availability | Enforcement is delayed; suitable for billing/alerting, not hard admission |

For a hard global limit $Q$ split into regional leases $q_r$, preserve:

$$
\sum_r q_r + q_{\text{unallocated}} \le Q + O_{\text{allowed}}
$$

where $O_{\text{allowed}}$ is the explicitly accepted overshoot bound. Expired or partitioned regions cannot silently mint additional allowance.

### Dependency failure policy

Define per route:

| Limiter state | Public read | Authenticated mutation | High-cost/privileged operation |
|---|---|---|---|
| Fresh local allowance | Enforce | Enforce | Enforce |
| Central service unavailable; valid lease remains | Enforce locally | Enforce locally if policy permits | Usually enforce locally with strict reserved allowance |
| Lease expired or state unknown | Bounded emergency local limit may fail open | Usually fail closed or queue | Fail closed |
| Counter rejected or malformed | Fail closed for governed key | Fail closed | Fail closed |

Never remove all admission because the limiter is unhealthy: that sends maximum load to already stressed dependencies. A conservative local emergency limit is often safer than unrestricted fail-open or complete global outage.

Return machine-readable overload semantics and a truthful retry hint only when the server can estimate recovery. Clients still need jitter and a retry budget.

## Request Aggregation and Backends for Frontends

Aggregation trades client round trips for gateway fan-out. It is justified when the composition is presentation-oriented, has a clear owner, and can be expressed as a bounded dependency graph. It is not a distributed transaction coordinator.

### Compile an execution DAG

~~~mermaid
flowchart LR
    REQ[Client request] --> A[Account summary]
    A --> B[Entitlements]
    A --> C[Recent orders]
    B --> D[Recommendations]
    C --> OUT[Response assembler]
    D --> OUT
~~~

For each node define:

- dependency and whether it can run concurrently;
- input source and schema;
- deadline slice;
- retry eligibility and budget;
- required versus optional result;
- maximum items and bytes;
- cache and consistency requirement;
- cancellation behavior; and
- error mapping.

The critical-path budget must fit inside the end-to-end deadline:

$$
T_{\text{edge}} +
\max_{\text{path }p}\left(\sum_{i \in p} T_i\right) +
T_{\text{assemble}} +
T_{\text{margin}}
\le T_{\text{request}}.
$$

Allocating the full remaining deadline independently to every child allows late work to continue after the response is doomed. Propagate cancellation and cap both concurrent children and queued aggregation work.

### Partial results are a contract

Choose one of:

- all required components or an error;
- stable partial response with per-component status;
- stale cached component under a stated freshness contract; or
- asynchronous operation resource for work that cannot fit.

Do not return HTTP success with silently missing security-sensitive or monetary fields. Do not map every child error to one generic gateway error; preserve stable problem types while redacting internal topology.

### Consistency boundary

Parallel reads can observe different upstream revisions. If the response requires one coherent snapshot, use a domain-owned read model, version token propagated across services, or a purpose-built aggregation service backed by consistent data. The edge cannot manufacture transactional consistency from unrelated APIs.

Keep reusable domain composition behind a service API. A channel-specific BFF may format, filter, and batch for its client, but business workflows and durable state transitions belong downstream.

## Deadlines, Retries, and Amplification

A gateway retry is an additional upstream load multiplier. If each layer makes up to $a_i$ attempts, worst-case attempts from one client request are:

$$
A_{\text{max}} = \prod_i a_i.
$$

The gateway should normally be the single retry owner for the edge-to-upstream hop, with downstream services receiving the remaining deadline and a bounded attempt budget.

Retry only when all are true:

- the failure class is plausibly transient;
- the operation is safe or protected by a correctly scoped idempotency mechanism;
- request body replay is bounded and available;
- sufficient deadline remains;
- the retry budget has capacity;
- a different endpoint can improve the outcome; and
- the upstream has not signaled overload that retries would worsen.

Never retry ambiguous non-idempotent effects merely because the connection closed before a response. The upstream may have committed. Use the domain’s idempotency contract, described in [Idempotency](../01-foundations/08-idempotency.md), and the transport policy from [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md).

Circuit breakers, concurrency limits, and load shedding protect the upstream; their canonical design is in [Circuit Breakers](../06-scaling/06-circuit-breakers.md) and [Backpressure](../06-scaling/07-backpressure.md).

## Protocol Mediation and Streaming

Protocol translation must preserve semantics, not only syntax:

| Boundary | Required mapping |
|---|---|
| HTTP/JSON to RPC | method safety, deadline, status/problem type, metadata allowlist, numeric/null semantics |
| RPC to HTTP | canonical status, headers, streaming behavior, trailers, cancellation |
| WebSocket or server stream | upgrade/auth lifetime, backpressure, idle timeout, reconnect/resume contract |
| Event submission | acceptance versus completion, idempotency, correlation, asynchronous failure |

Avoid buffering by default. Streaming request and response bodies preserve memory and latency, but only if security inspection, transforms, compression, retries, and error mapping also support streaming. A feature that requires full-body buffering needs an explicit maximum expanded size and per-tenant memory budget.

If the gateway changes a signed body, content encoding, or canonical path, signature verification must occur at the layer that still has the signed representation, and any new upstream signature must cover the transformed representation.

## Configuration Control Plane

Gateway configuration is executable policy. A malformed route can expose a private service or black-hole a region.

### Compilation and activation

~~~mermaid
flowchart LR
    SRC[Versioned route and policy sources] --> VALID[Schema, authority, and semantic validation]
    VALID --> GRAPH[Dependency graph and conflict analysis]
    GRAPH --> TEST[Route and policy conformance tests]
    TEST --> BUNDLE[Immutable signed snapshot]
    BUNDLE --> DIST[Regional distribution]
    DIST --> STAGE[Gateway staging]
    STAGE --> ACTIVE[Atomic active pointer]
~~~

The snapshot includes route tables, referenced policy digests, upstream service identities, transformation schemas, certificate/key references, and compatibility metadata. Secrets themselves follow a separate least-privilege distribution path.

Activation sequence:

1. authenticate source and publisher;
2. verify digest, signature, tenant, and schema;
3. ensure the dependency closure is present;
4. compile match structures and transformations off the request path;
5. run local probes against the staged snapshot;
6. atomically swap the active pointer;
7. report received, validated, active, and rejected revisions; and
8. retain a still-valid last-known-good snapshot.

An acknowledgement that syntax validated is not proof that every route is reachable or every upstream is healthy. Monitor both configuration activation and request outcomes.

### Safe rollout

- static analysis for overlap, exposure, rewrite, and dependency errors;
- offline route replay against a redacted request corpus;
- shadow match and policy comparison;
- explicit canary gateway cohort;
- per-route outcome and latency comparison;
- convergence checks across regions;
- automatic halt on security-policy divergence; and
- authorized rollback to a known compatible snapshot.

Route and application rollout need coordination. A new route may safely precede a backward-compatible service, but a route that requires a new operation waits until sufficient compatible upstream capacity is discoverable. Removal reverses the order: stop routing and drain usage before removing upstream support.

## Multi-Region Edge Architecture

~~~mermaid
flowchart TB
    CLIENT[Clients] --> GLOBAL[Global traffic steering]
    GLOBAL --> GWA[Region A gateway cell]
    GLOBAL --> GWB[Region B gateway cell]
    GWA --> SA[Region A services]
    GWB --> SB[Region B services]
    CFG[Global policy source] --> CA[Region A config replica]
    CFG --> CB[Region B config replica]
    CA --> GWA
    CB --> GWB
~~~

Keep the request path regional where possible:

- locally cached verification keys with expiry and revocation policy;
- local policy evaluation or bounded decision cache;
- regional rate-limit allowance;
- regional service discovery;
- regional telemetry buffering; and
- immutable configuration replicas.

A globally synchronous dependency can turn a regional edge into a global outage. When global freshness is mandatory—revoked credentials, hard cross-region financial quota, singleton operation—make that dependency explicit and fail according to the operation’s risk.

### Failover is more than routing

Before sending a tenant to another region verify:

- identity and authorization state is available and current;
- rate-limit allowance cannot be double-spent beyond policy;
- upstream data is present at the required consistency;
- keys and secrets are usable;
- the destination has admitted failover capacity;
- residency and network policies allow it; and
- client affinity/session semantics are preserved or intentionally reset.

Use cells or independent gateway fleets to limit configuration and overload blast radius. See [Multi-Region Architecture](../06-scaling/09-multi-region-architecture.md) and [Cell-Based Architecture](../06-scaling/11-cell-based-architecture.md).

## Capacity Planning

Let:

- $\lambda$ be admitted client requests per second;
- $L$ be mean gateway residence time;
- $F$ be mean upstream fan-out per request;
- $A$ be mean attempts per upstream call including retries;
- $b_{\text{in}}$ and $b_{\text{out}}$ be mean client request/response bytes;
- $u_{\text{in}}$ and $u_{\text{out}}$ be mean upstream bytes per call; and
- $m_{\text{buffer}}$ be mean buffered bytes per in-flight request.

Little’s Law gives expected in-flight requests:

$$
C \approx \lambda L.
$$

Expected upstream attempt rate is:

$$
\lambda_{\text{up}} \approx \lambda F A.
$$

Approximate application-layer byte rate is:

$$
B \approx \lambda(b_{\text{in}} + b_{\text{out}})
  + \lambda F A(u_{\text{in}} + u_{\text{out}}).
$$

Buffer memory is at least:

$$
M_{\text{buffers}} \approx C m_{\text{buffer}},
$$

before connection buffers, TLS state, route indexes, caches, compression, traces, and aggregation intermediates.

### Benchmark the real cost centers

- TLS handshakes and resumption ratio;
- authentication signature verification or introspection latency;
- policy and rate-limit evaluation;
- request parsing and normalization;
- compression/decompression expanded bytes;
- transformation and serialization;
- aggregation fan-out and response assembly;
- upstream connection pools and stream limits; and
- log/trace export backpressure.

Size by route class and percentile, not one mean request. Large uploads, streaming responses, and aggregation endpoints need separate concurrency pools so they cannot exhaust small-request capacity.

### Overload hierarchy

Shed work before memory or queues become unbounded:

1. reject new expensive work by verified tenant/route priority;
2. stop optional aggregation children;
3. disable speculative retries and hedges;
4. reduce per-connection stream concurrency;
5. shed low-priority accepted connections; and
6. preserve health, configuration, identity rotation, and drain control traffic.

Autoscaling is slower than a burst and can amplify cold-start dependencies. Maintain admission headroom and test scale-out while configuration, keys, and discovery streams reconnect. See [Auto-Scaling](../06-scaling/08-auto-scaling.md).

## Failure Modes and Traces

### Authorization service outage becomes a bypass

~~~text
policy call times out -> middleware catches generic exception
-> request gets empty policy result -> empty result coerces to permit
-> protected upstream receives unauthorized effect
~~~

**Controls:** four-valued decision, fail-closed protected route, bounded cached decision keyed by policy revision, and explicit emergency policy rather than exception handling.

### Rate limiter fails open into an overload

~~~text
central limiter slows -> gateways bypass limits
-> upstream saturates -> latency triggers retries
-> gateway connection and memory queues fill
~~~

**Controls:** leased/local conservative allowance, concurrency admission, retry suppression, and per-route dependency mode.

### Retry amplification after a partial upstream failure

~~~text
one gateway request fans out to five calls
-> each call retries twice -> downstream also retries
-> one client request creates dozens of attempts
~~~

**Controls:** one retry owner per hop, end-to-end attempt budget, propagated deadline, idempotency, and overload-aware retry denial.

### Route update activates before its dependencies

~~~text
route snapshot references new upstream -> gateway activates route
-> discovery/schema/policy resource has not arrived
-> requests return errors or fall through to a broader route
~~~

**Controls:** dependency-closure manifest, staging, atomic activation, no security fallback route, and canary convergence.

### Aggregation tail consumes the deadline

~~~text
optional child hangs -> assembler waits for full child timeout
-> required result completes but client deadline expires
-> abandoned child work continues upstream
~~~

**Controls:** critical-path budgeting, optional-child cutoff, cancellation propagation, bounded concurrency, and stable partial-result contract.

### Header spoofing crosses the trust boundary

~~~text
client sends internal principal header -> gateway appends verified principal
-> downstream reads the first or last duplicate differently
-> attacker becomes another user
~~~

**Controls:** strip protected fields, construct one canonical value, reject duplicates, bind context to authenticated gateway connection, and downstream allowlist.

### Parser disagreement enables request smuggling

~~~text
edge and upstream disagree on framing or normalized path
-> authorization applies to request A
-> upstream parses hidden request B on the reused connection
~~~

**Controls:** strict RFC-conformant parsing, reject ambiguous framing, canonical typed forwarding, compatible protocol stacks, and differential parser tests.

### Global dependency defeats regional isolation

~~~text
one identity/introspection region fails
-> every gateway region blocks on it
-> healthy service regions become unreachable
~~~

**Controls:** regional replicas or local validation, bounded freshness, explicit global-only operations, cell isolation, and dependency-aware readiness.

### Buffering a stream exhausts memory

~~~text
new inspection feature buffers entire decompressed body
-> slow large uploads occupy per-request memory
-> process reaches memory limit before request count alarm fires
~~~

**Controls:** streaming inspection, compressed and expanded size limits, separate pool, backpressure, and memory-based admission.

## Observability and Audit

Every request trace should make stage boundaries and governing revisions visible:

~~~text
request and trace ID
gateway cell and active config digest
route ID and API version
authenticated principal class and auth/key revision
authorization result, policy revision, and obligations
rate-limit key class, decision source, and remaining lease state
discovery revision and selected upstream
deadline, attempts, cancellation, and response problem type
bytes buffered/streamed and aggregation child outcomes
~~~

Metrics include:

- connection accepts/rejects, protocol/parser errors, and handshake cost;
- route matches, unmatched requests, ambiguity rejections, and shadow deltas;
- authentication/authorization outcomes by bounded reason;
- limiter latency, local/central mode, lease exhaustion, and fail behavior;
- per-route admitted, queued, shed, in-flight, latency, bytes, and problem types;
- upstream attempts, retry suppression, cancellations, and connection outcomes;
- aggregation fan-out, critical-path child, partial results, and abandoned work;
- received/validated/active config revision and fleet convergence; and
- region failover volume and dependency freshness.

Do not place raw principal, token, path parameter, idempotency key, or cursor into metric labels. Use secured structured logs and sampling. Redact credentials before any request logging, including error paths.

## Verification Strategy

| Test layer | What to verify |
|---|---|
| Parser differential | Edge and upstream agree on message framing, authority, path, headers, and encoding |
| Route property tests | Match is deterministic; security routes cannot be shadowed; rewrites terminate |
| Auth contract tests | Invalid, stale, unavailable, unknown-key, deny, and indeterminate stay distinct |
| Trust-boundary tests | Client cannot inject forwarding, identity, policy, or trace authority |
| Limiter fault tests | Central outage, expired lease, partition, counter rollback, and hot key follow route policy |
| Aggregation model tests | DAG dependencies, deadlines, cancellation, required/optional semantics, and byte limits |
| Retry tests | Attempt budget and idempotency classification prevent multiplicative retries |
| Configuration tests | Missing dependency, incompatible schema, bad signature, partial rollout, and rollback |
| Region tests | Loss of global config, identity, quota, discovery, and telemetry has documented local behavior |
| Load tests | Handshake storm, hot route, large body, slow stream, aggregation fan-out, and config reconnect |
| Security tests | Smuggling, traversal, duplicate headers, cache-key confusion, credential leakage, and policy bypass |

Replay a redacted production route corpus through old and candidate snapshots. Compare route, auth policy, quota class, upstream, timeout, and transformation—not only final status.

## Decision Framework

### What belongs at the gateway?

Place a concern at the gateway when it depends on the public transport boundary, must be consistent before traffic enters the system, and can be implemented without owning domain state:

- transport normalization and public TLS;
- coarse authentication and route authorization;
- global/tenant admission and request-size limits;
- stable API routing and compatibility mediation;
- channel-specific bounded aggregation;
- public observability and audit correlation.

Keep it downstream when it requires current business state, transactional invariants, reusable domain orchestration, or service-specific persistence.

### Architecture questions

1. Which client protocols and trust boundaries terminate here?
2. What exact normalized request does every policy decision see?
3. Which identity and authorization decisions can be local, cached, or unavailable?
4. Which limits are local, regional, or global, and what overshoot is acceptable?
5. Which routes may fail open, fail closed, queue, degrade, or serve stale data?
6. What is the maximum fan-out, attempt count, body expansion, and buffered memory?
7. How are route dependencies compiled and activated atomically?
8. Can each region process safe traffic without a global synchronous dependency?
9. How does a route change coordinate with upstream rollout and drain?
10. Can an incident responder reconstruct the exact route and policy revision?

## Key Takeaways

1. A gateway is an edge policy-enforcement point, not merely a reverse proxy.
2. Normalize once before route, authentication, authorization, caching, signing, and forwarding.
3. Strip untrusted identity/forwarding fields and bind delegated context to an authenticated trust boundary.
4. Authentication and authorization interfaces need explicit unavailable, stale, and indeterminate outcomes.
5. Limiter failure should fall back to a reviewed conservative policy, not unlimited traffic.
6. Aggregation needs a compiled DAG, critical-path deadline, cancellation, byte limit, and explicit partial-result contract.
7. Retry ownership and attempt budgets prevent multiplicative amplification.
8. Activate route configuration with its complete policy, schema, and upstream dependency closure.
9. Regional gateways should keep request-path dependencies regional unless global freshness is essential.
10. Capacity is driven by in-flight time, fan-out, attempts, bytes, cryptography, and buffering—not request rate alone.

---

## References

- [RFC 9110: HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110) — methods, status codes, intermediaries, conditional requests, and retry semantics
- [RFC 9112: HTTP/1.1](https://www.rfc-editor.org/rfc/rfc9112) — message framing and request-smuggling considerations
- [RFC 9113: HTTP/2](https://www.rfc-editor.org/rfc/rfc9113) — streams, connection errors, flow control, and intermediary behavior
- [RFC 9457: Problem Details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc9457) — machine-readable error representation
- [RFC 6750: OAuth 2.0 Bearer Token Usage](https://www.rfc-editor.org/rfc/rfc6750) — bearer-token transport and threat requirements
- [RFC 7662: OAuth 2.0 Token Introspection](https://www.rfc-editor.org/rfc/rfc7662) — authoritative token validation interface
- [RFC 7239: Forwarded HTTP Extension](https://www.rfc-editor.org/rfc/rfc7239) — standardized forwarding metadata and trust considerations
- [W3C Trace Context](https://www.w3.org/TR/trace-context/) — interoperable trace propagation and mutation rules
- [OpenAPI Specification](https://spec.openapis.org/oas/latest.html) — machine-readable HTTP API contracts
- [Authentication Systems](../10-security/01-authentication-fundamentals.md) — session, credential, and authenticator design
- [Authorization Patterns](../10-security/07-authorization-patterns.md) — policy decision and enforcement models
- [API Security](../10-security/04-api-security.md) — public API threat model and controls
- [Rate Limiting](../06-scaling/05-rate-limiting.md) — algorithms, distributed counters, fairness, and quotas
- [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md) — deadlines, retry budgets, and overload safety
- [API Design and Evolution](./04-api-design-patterns.md) — public resource, error, concurrency, pagination, and compatibility contracts
