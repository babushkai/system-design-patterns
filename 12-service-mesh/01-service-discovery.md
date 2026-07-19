# Service Discovery and Control-Plane State

## TL;DR

Service discovery is a distributed state system that maps a logical service identity to a changing set of usable endpoints. The difficult part is not finding an IP address. It is deciding who may publish an endpoint, distinguishing process life from request readiness, distributing changes without gaps, fencing restarted instances, draining long-lived connections, and defining what clients do when discovery is stale or unavailable.

A production design separates authoritative registration from client-local observations. The control plane publishes versioned endpoint snapshots; clients combine those snapshots with local connection and outlier state. A failed probe by one client must not globally evict a healthy instance, while an expired lease or authoritative drain must eventually remove it everywhere.

DNS, a registry client, and a discovery-aware proxy are delivery mechanisms with different caching and rollout semantics. Choose one only after defining the endpoint record, consistency target, stale-state policy, failure domains, and capacity model.

This chapter owns endpoint lifecycle and discovery-state distribution. [Load Balancing](../06-scaling/01-load-balancing.md) owns request-selection algorithms, [DNS and Connection Management](../06-scaling/13-dns-and-connection-management.md) owns resolver and connection-cache mechanics, and [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md) owns retry budgets.

---

## The Discovery Contract

A discovery result is not “all live processes.” It is the set of endpoints that a named consumer is currently allowed to consider for a named operation.

~~~text
resolve(
  tenant,
  trust_domain,
  service_identity,
  port_or_protocol,
  consumer_identity,
  consumer_region_and_zone,
  policy_revision
) -> endpoint_snapshot
~~~

The result can vary by tenant, network reachability, protocol, rollout cohort, locality, and authorization. Treat the lookup key as typed data. A string such as *payments* is not globally unique and must not accidentally cross a namespace or trust boundary.

### Core invariants

1. **Authenticated ownership:** only an authorized registrar may create or mutate an endpoint for a service identity.
2. **Unique incarnation:** a restarted process cannot renew or deregister the lease of an older incarnation.
3. **Monotonic observation:** within one stream, clients never apply an older revision after a newer one.
4. **Complete replacement:** a snapshot is atomic; absence has defined deletion semantics.
5. **Bounded staleness:** every client can report how old its last authoritative state is and what it will do after the budget expires.
6. **Explicit serving state:** liveness, readiness, draining, and protocol health are not collapsed into one Boolean.
7. **Safe emptiness:** an empty endpoint set is a first-class result, not permission to reuse an arbitrarily old address.
8. **Drain before removal:** planned termination stops new assignment before transport teardown.
9. **Local evidence stays local:** connection failures and passive outlier detection do not silently rewrite global registry truth.
10. **Tenant isolation:** storage keys, watch streams, caches, metrics, and logs include the authenticated tenant and trust domain.

## Endpoint Records Are State Machines

An endpoint record needs enough information to reject stale updates and support safe routing:

| Field | Purpose |
|---|---|
| Service key | Tenant, trust domain, namespace, logical service, and named port/protocol |
| Instance identity | Stable identity for one scheduled workload, not merely host and port |
| Incarnation or fencing token | Distinguishes restarts that reuse an instance name or address |
| Addresses | Network family, address, port, transport, and reachability scope |
| Locality | Region, zone, failure domain, and optional priority tier |
| Serving state | Starting, ready, degraded, draining, not-serving, or unknown |
| Capacity metadata | Relative weight or bounded concurrency signal, with provenance |
| Lease metadata | Issued revision, renewal deadline, and authoritative expiry |
| Revision | Monotonic stream revision or immutable snapshot digest |
| Provenance | Registrar identity, source object revision, and policy revision |

Do not use arbitrary application metadata as a routing language. Every routing-relevant field needs a schema, an authorized publisher, cardinality limits, and compatibility rules. Otherwise a misspelled zone, unbounded tag, or attacker-controlled label changes routing behavior.

### A lifecycle with fenced transitions

~~~mermaid
stateDiagram-v2
    [*] --> Registered: authenticated create
    Registered --> Starting: address allocated
    Starting --> Ready: readiness established
    Ready --> Degraded: partial capability or reduced weight
    Degraded --> Ready: recovery threshold met
    Ready --> Draining: termination or maintenance intent
    Degraded --> Draining: termination intent
    Draining --> NotServing: drain deadline or zero inflight
    Starting --> NotServing: startup failed
    Ready --> NotServing: authoritative health failure
    NotServing --> [*]: lease expires or record removed
~~~

Each mutation carries the instance incarnation and expected record revision. A delayed heartbeat from incarnation 7 must not revive incarnation 8’s drained record. Deletion is also fenced: an old shutdown hook cannot delete a replacement that reused the logical instance name.

## Registration and Source Authority

### Choose the registrar deliberately

There are three common sources:

- **Orchestrator registration:** the scheduler or platform controller derives endpoints from desired and observed workload state.
- **Self-registration:** the process acquires a lease and renews it.
- **Adapter registration:** a controller translates another source of truth, such as a VM inventory or external registry.

Orchestrator registration is attractive when the platform already authenticates workload placement and can observe lifecycle transitions. Self-registration can represent application readiness directly but creates a bootstrap dependency: the process needs identity, network access, retry behavior, and lease logic before it is discoverable. An adapter is useful during migration, but its source revision and lag must remain visible.

Do not accept the union of sources without precedence rules. If a scheduler says an endpoint is terminating while a self-registrar says ready, the scheduler normally owns lifecycle and the application owns readiness within that lifecycle. Encode the ownership split rather than resolving by last write.

### Registration protocol

A safe create or renew path is:

1. Authenticate the workload or controller.
2. Authorize it for the exact service identity and namespace.
3. Canonicalize addresses and reject addresses outside permitted network scopes.
4. Allocate or verify a unique incarnation token.
5. Compare-and-swap the record against the expected revision.
6. Commit the record and lease in the same authoritative transaction.
7. Publish a new service-set revision.
8. Return the committed revision and renewal deadline.

Registration is idempotent for the tuple *(service, instance, incarnation, requested mutation)*. Retrying after a lost response must return the committed result, not create a second endpoint.

### Leases are failure detectors, not clocks of truth

A lease bounds how long an endpoint can remain registered without renewed evidence. It cannot distinguish a dead process from a partitioned but healthy process. The lease duration trades failure-removal latency against renewal load and sensitivity to pauses:

$$
R_{\text{renew}} \approx \frac{E}{T_{\text{renew}}}
$$

where $E$ is leased endpoints and $T_{\text{renew}}$ is the renewal interval. Size for synchronized reconnect and pause recovery, not only the average.

Use server-authoritative expiry and tolerate bounded clock uncertainty. Renew before expiry with jitter. A failed renewal changes the publisher’s confidence; it does not justify continuing to serve forever. On lease loss, a safety-critical workload should stop accepting effects if it can no longer prove that clients have a current route or identity.

## Health Semantics

“Healthy” hides several independent questions:

| Signal | Question | Routing consequence |
|---|---|---|
| Startup | Has initialization completed? | Do not send normal traffic yet |
| Liveness | Is the process making progress? | Restart may be appropriate; not a direct global routing vote |
| Readiness | Can this instance accept new requests now? | Remove or reduce new assignment |
| Dependency readiness | Which operation classes remain usable? | Prefer capability-specific routing over one global Boolean |
| Drain state | Is termination planned? | Stop new work while allowing admitted work to finish |
| Passive transport evidence | Is this client seeing connection failures? | Local ejection or connection repair |
| Synthetic end-to-end health | Does a user path work? | Alert and diagnose; do not automatically blame one endpoint |

### Health checks must not become the outage

An active probe should be cheap, bounded, and representative of the serving loop. A deep check that synchronously calls every dependency creates correlated failure: one database incident marks every frontend unready, drains the fleet, and converts a partial degradation into total unavailability.

Prefer:

- liveness that verifies local progress without external dependencies;
- readiness that reflects ability to admit work and can expose operation-specific degradation;
- hysteresis or success/failure thresholds to avoid state flapping;
- probe jitter and bounded concurrency;
- separate probe traffic budgets; and
- a minimum serving floor or controlled fail-open rule only where explicitly justified.

Health status needs an observation timestamp, observer identity, and reason. “Unknown because the health checker is partitioned” is different from “not serving because the application rejected traffic.”

### Authoritative and local health

The registry may publish scheduler state, readiness, and lease validity. Each client or proxy also has local evidence: connection refusal, timeout, protocol reset, or success. Keep both:

~~~text
eligible endpoint
  = authoritative endpoint is serving
  AND endpoint is reachable from this client scope
  AND local ejection window has not excluded it
~~~

Local ejection must have bounded duration and probing rules. A client should not report a single failure as global truth; doing so creates feedback loops in which one faulty zone removes healthy endpoints for every zone. Conversely, the control plane may aggregate statistically significant multi-observer evidence through an explicit health authority.

## State Distribution: Snapshot, Watch, and Resynchronization

The registry’s write path and the client’s read path have different scaling shapes. Writes track endpoint churn; reads can involve every client watching many services.

~~~mermaid
flowchart LR
    SRC[Registrars and health authorities] --> LOG[(Authoritative state and revision log)]
    LOG --> MAT[Per-service materializer]
    MAT --> FAN[Regional fan-out]
    FAN --> C1[Client cache]
    FAN --> C2[Proxy cache]
    FAN --> DNS[DNS publication]
    C1 --> POOL1[Connection pools]
    C2 --> POOL2[Connection pools]
~~~

### Correct watch protocol

A watch is not just a stream of arrays. It needs:

- an initial snapshot and its revision;
- ordered deltas or complete replacement snapshots;
- explicit deletion;
- a sequence or opaque resume token;
- heartbeat or progress notification;
- detection of compaction and stream gaps;
- a resnapshot path;
- per-client subscription identity and authorization; and
- client acknowledgement or applied-revision telemetry where rollout safety matters.

A robust client state machine:

1. Load a locally persisted, still-permitted last-known-good snapshot if available.
2. Request a snapshot at or after a known revision.
3. Atomically install the complete snapshot.
4. Apply only contiguous deltas scoped to that snapshot lineage.
5. On a gap, compaction error, incompatible schema, or ambiguous deletion, stop delta application and resnapshot.
6. Keep the old complete state until the replacement is validated.
7. Report received, validated, and active revisions separately.

Do not merge fragments from independent streams unless the protocol defines a consistent composition. A route pointing to a newly named cluster before its endpoints arrive can create a transient black hole. Aggregated or dependency-aware delivery can sequence related resources; otherwise the client must warm dependencies before activation.

### Stale configuration is an explicit mode

Every cache entry carries:

~~~text
source revision
received time and last successful refresh
validity or lease horizon
schema version
tenant/service key
provenance and signature where applicable
~~~

Define behavior by operation risk:

| Condition | Read-only/idempotent request | Stateful mutation | Privileged or irreversible effect |
|---|---|---|---|
| Watch disconnected; snapshot within stale budget | Continue and expose stale age | Continue only if policy permits | Usually require fresh control-plane/authorization state |
| Snapshot beyond stale budget | Limited fail-open only with explicit design | Fail closed or controlled queue | Fail closed |
| No snapshot at first startup | Return unavailable | Return unavailable | Return unavailable |
| Empty authoritative set | Return no endpoint; apply bounded fallback if specified | Do not revive expired endpoints | Do not revive expired endpoints |
| Invalid update | Keep valid prior snapshot and report rejection | Same | Same |

Last-known-good improves availability only while the system can explain why that state remains safe. It must not defeat endpoint revocation, tenant migration, certificate expiry, or a completed drain.

## DNS, Registry, and Proxy Discovery

### DNS is a cache hierarchy

DNS integrates with almost every runtime and can publish addresses or SRV-style target/port records. Its operational semantics include authoritative TTL, recursive resolver caching, local stub caching, negative caching, library behavior, and connection reuse. Changing a record does not close an existing connection.

Low TTL does not mean instant convergence; it increases query load and still interacts with minimum TTLs, resolver behavior, and long-lived pools. Serving stale DNS data can improve resilience during authoritative failure, but the permitted stale window must align with endpoint lifecycle and revocation risk.

### Registry APIs expose richer state

A registry can carry endpoint identity, locality, readiness, drain state, revision, and watch semantics. It also adds a client library or local-agent dependency, authorization surface, and high-fan-out stream workload. Clients must implement snapshot recovery and stale behavior correctly.

### A proxy centralizes the consumer

Server-side discovery moves resolution and selection into a gateway, load balancer, node proxy, or workload proxy. Applications get a stable local or virtual destination; the proxy owns watches and connection pools. This reduces language-specific client complexity but makes proxy availability, resource usage, and rollout correctness part of every request path.

| Mechanism | State richness | Update model | Client burden | Common risk |
|---|---:|---|---:|---|
| DNS A/AAAA | Address only | TTL/cache refresh | Low | stale answers plus persistent connections |
| DNS SRV or service records | Target, port, priority/weight | TTL/cache refresh | Medium | partial client support and ambiguous metadata |
| Registry query/watch | Typed endpoint state | snapshot plus delta/watch | High | gap handling and per-client divergence |
| Discovery-aware proxy | Typed state hidden behind stable destination | proxy-managed stream | Low in app, high in platform | concentrated blast radius and proxy saturation |

Hybrid designs are normal: DNS locates regional gateways, gateways consume a registry, and mesh proxies receive richer endpoint resources. Document which layer is authoritative and where caching occurs.

For resolver and connection-pool behavior, see [DNS and Connection Management](../06-scaling/13-dns-and-connection-management.md).

## Endpoint Draining and Connection Lifecycle

Deregistration alone is insufficient because clients may cache endpoints and transports may multiplex long-lived requests.

### Planned termination sequence

~~~mermaid
sequenceDiagram
    participant O as Orchestrator
    participant E as Endpoint
    participant R as Registry
    participant C as Clients/proxies

    O->>E: termination intent
    E->>R: publish draining with incarnation
    R-->>C: new revision: no new assignment
    C->>C: stop creating connections and new streams
    E->>E: reject new work; finish or checkpoint admitted work
    E->>C: protocol drain signal where supported
    E->>R: not-serving or drain deadline reached
    R-->>C: endpoint removed
    O->>E: terminate after bounded grace
~~~

The grace window must cover discovery propagation, load-balancer convergence, transport drain, and application completion:

$$
T_{\text{grace}} \ge
T_{\text{publish}} +
T_{\text{fanout}} +
T_{\text{client-activate}} +
T_{\text{request-drain}} +
T_{\text{margin}}
$$

Measure these terms. A fixed sleep copied between systems is not evidence.

Clients should distinguish:

- **new endpoint assignment:** exclude draining endpoints immediately;
- **new streams on an existing multiplexed connection:** stop according to protocol drain semantics;
- **in-flight requests:** allow bounded completion where safe;
- **long-lived sessions:** transfer, reconnect, checkpoint, or terminate under an explicit product contract.

Unplanned failure skips drain, so timeouts, local outlier detection, and retries remain necessary. Those mechanisms must obey the retry-budget rules in [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md).

## Multi-Region and Failure Domains

A global registry that returns every endpoint to every client creates latency, data-sovereignty, and correlated-failure problems. Prefer hierarchical state:

~~~text
global service identity and policy
  -> regional service membership
      -> zonal endpoints
          -> client-local reachability and connection state
~~~

Global state changes slowly; endpoint churn remains regional. A client normally receives local endpoints plus explicitly configured failover priorities. Cross-region failover requires compatible data state, authorization, capacity, and dependency reachability; discovery alone cannot make a region safe.

During partition:

- regional writers must not both claim a globally unique singleton without fencing;
- region-local endpoint changes may continue under delegated authority;
- clients keep a bounded last-known-good regional view;
- failover is a policy transition with capacity admission, not “try every region”; and
- reconciliation preserves incarnation and source revisions instead of last-write-wins timestamps.

See [Multi-Region Architecture](../06-scaling/09-multi-region-architecture.md) for data and traffic failover and [Cell-Based Architecture](../06-scaling/11-cell-based-architecture.md) for blast-radius containment.

## Capacity Planning

Let:

- $E$ be registered endpoints;
- $S$ be logical services;
- $W_s$ be active watchers of service $s$;
- $C_s$ be endpoint changes per second for service $s$;
- $D_s$ be average encoded delta bytes;
- $P_s$ be complete snapshot bytes; and
- $H$ be health or lease updates per second.

Naive direct fan-out is:

$$
F_{\text{events}} = \sum_s C_s W_s
$$

and delta bandwidth is approximately:

$$
B_{\text{delta}} = \sum_s C_s W_s D_s.
$$

This shape, not registry write throughput alone, usually drives scale. Regional fan-out, subscription indexes, coalescing superseded endpoint updates, and content-aware deltas reduce work. They must preserve ordered deletion and resnapshot semantics.

### Size the exceptional paths

Normal churn hides the largest loads:

- a zone evacuation changes many endpoints together;
- control-plane recovery reconnects every watcher;
- a deployment flips readiness across a large cohort;
- lease-renewal jitter is lost after process restart;
- one popular service has far more watchers than the median;
- a compacted log forces complete snapshots; and
- DNS cache expiry aligns clients into a query wave.

Bound per-tenant subscriptions, snapshot size, metadata cardinality, outstanding stream bytes, health-probe concurrency, and resnapshot rate. Use fair queues so one hot service cannot starve unrelated control-plane updates.

### Client memory and connection effects

Client cost is more than endpoint bytes:

$$
M_{\text{client}} \approx
M_{\text{snapshots}} +
M_{\text{indexes}} +
M_{\text{local-health}} +
M_{\text{connection-pools}}.
$$

Large endpoint sets can create excessive connection pools, TLS handshakes, health checks, and load-balancer state even when the encoded snapshot is small. Limit active connections independently from discovered endpoints and avoid probing every endpoint from every client.

## Failure Modes and Traces

### Crash without deregistration

~~~text
endpoint dies -> no shutdown hook -> record remains ready
-> clients attempt connections -> local failures rise
-> lease or authoritative health eventually expires record
~~~

**Controls:** leases, passive local ejection, bounded connection timeout, jittered retry, and removal-latency SLO. Never depend on graceful deregistration for crash detection.

### Restart reuses an instance identity

~~~text
incarnation 7 pauses -> scheduler starts incarnation 8
-> delayed renewal from 7 arrives -> registry accepts by instance name
-> old address/state overwrites the replacement
~~~

**Controls:** monotonic incarnation token, compare-and-swap mutation, and fenced delete/renew.

### Watch gap silently loses a deletion

~~~text
client has revision 40 -> stream disconnects
-> revision 41 removes endpoint -> log compacts
-> client resumes at 42 without resnapshot
-> deleted endpoint remains locally forever
~~~

**Controls:** opaque resume token, compaction error, contiguous sequence validation, complete resnapshot, and active-revision telemetry.

### Health checker causes fleet eviction

~~~text
shared dependency slows -> deep readiness probes fail everywhere
-> all endpoints become not-ready -> no user request can reach degraded path
-> retries and probes intensify dependency load
~~~

**Controls:** shallow local readiness, operation-specific capability, hysteresis, probe budget, and a reviewed minimum-serving policy.

### Drain races with connection reuse

~~~text
registry removes endpoint -> clients stop selecting it
-> existing HTTP/2 connection remains open
-> new streams continue until process termination
-> requests reset mid-effect
~~~

**Controls:** draining state before deletion, protocol drain signal, connection-pool integration, admission stop, and measured grace.

### Registry partition creates divergent truth

~~~text
region A cannot reach authority -> keeps last-known-good
-> region B revokes endpoint -> A continues privileged traffic
~~~

**Controls:** risk-specific stale budget, regional delegated authority, revocation channel, expiry, and fail-closed behavior for privileged effects.

### Empty-set retry storm

~~~text
bad rollout publishes zero endpoints
-> every client retries discovery and requests
-> registry and recovering service receive synchronized load
~~~

**Controls:** last-known-good only where safe, exponential backoff with jitter, negative caching, retry budget, admission control, and a rollout invariant that detects unexpected empty sets.

## Observability

Observe the control plane, client convergence, and request outcome together.

### Control-plane signals

- registration creates, renewals, expiries, fenced mutations, and unauthorized attempts;
- service-set revision, endpoint counts by serving state and locality;
- change-log age, compaction, snapshot generation time, and fan-out queue depth;
- watch connects, resumes, gaps, resnapshots, outstanding bytes, and slow consumers;
- health transition reasons, flaps, probe load, and observer coverage; and
- desired versus active revision by client cohort.

### Client signals

- active snapshot revision and age;
- endpoints received, eligible, locally ejected, draining, and connected;
- resolution failures separated from connection, TLS, protocol, and application failures;
- connection reuse after drain;
- stale-mode entry and duration; and
- requests attempted against an endpoint after authoritative removal.

Avoid endpoint ID, raw address, and service revision as unbounded metric labels. Put them in sampled traces or structured logs. Correlate a request with service key, discovery revision, selected endpoint incarnation, connection age, and retry attempt.

For general signal and alert design, see [Metrics Systems and Monitoring](../11-observability/02-metrics-monitoring.md) and [SLOs and Error-Budget Control](../11-observability/05-slos-error-budgets.md).

## Verification Strategy

| Test layer | What to prove |
|---|---|
| State-machine tests | Only authorized lifecycle transitions occur; old incarnations cannot mutate new ones |
| Property tests | Applying a valid snapshot twice is idempotent; contiguous deltas equal a full snapshot |
| Model tests | Registration, expiry, drain, replacement, and compaction preserve invariants under reordering |
| Contract tests | Every registrar and client agrees on schema, deletion, unknown fields, and serving states |
| Fault injection | Lost renewals, partitions, pauses, clock skew, watch gaps, corrupt snapshots, and partial fan-out |
| Drain tests | No new assignment after drain revision; bounded in-flight completion; long-lived streams follow policy |
| Health tests | Dependency incidents do not accidentally evict the entire serving fleet |
| Isolation tests | Tenant substitution cannot read, publish, watch, cache, or log another tenant’s endpoints |
| Load tests | Hot service, deployment burst, zone evacuation, reconnect storm, and full resnapshot |
| Replay tests | Recorded revisions reconstruct the endpoint view used by a failed request |

Test last-known-good and fail-closed paths deliberately. They are usually exercised only during an incident, when discovering an unbounded cache or missing resnapshot path is too late.

## Decision Framework

Ask in this order:

1. **What is the logical identity?** Include tenant, namespace, protocol, and trust domain.
2. **Who owns each field?** Separate scheduler lifecycle, application readiness, security policy, and local observations.
3. **What is the endpoint state machine?** Define incarnation, lease, drain, and deletion.
4. **What consistency is required?** Specify propagation and staleness budgets by operation risk.
5. **How do clients recover a gap?** Require snapshot, sequence, resume, compaction, and atomic activation semantics.
6. **Where does selection run?** Application, library, node proxy, workload proxy, or gateway.
7. **What happens to connections?** Discovery updates must reach connection pools and long-lived streams.
8. **What are the failure domains?** Keep endpoint churn regional and failover explicit.
9. **What is the fan-out shape?** Size the hot service and reconnect storm, not only averages.
10. **Can a request be reconstructed?** Record the active discovery revision and endpoint incarnation.

### Mechanism choice

| Requirement | Bias |
|---|---|
| Broad compatibility, simple address lookup, modest churn | DNS with documented TTL and connection behavior |
| Rich readiness/locality/drain state and capable clients | Registry snapshot plus watch |
| Polyglot applications with centralized traffic policy | Discovery-aware local or regional proxy |
| Public or cross-region entry point | Stable DNS to gateway/load-balancer tier, richer discovery behind it |
| Safety-critical revocation or tenant-aware policy | Authenticated typed registry/config distribution; DNS alone is insufficient |

## Key Takeaways

1. Discovery is versioned distributed state, not an address lookup helper.
2. Endpoint identity needs an incarnation token so delayed renewals and deletes are fenced.
3. Liveness, readiness, capability, drain, and client-local health are different signals.
4. Watches require snapshot, ordered change, explicit deletion, gap detection, and resynchronization.
5. Last-known-good is safe only within an explicit risk-based stale budget.
6. DNS TTL expiry does not close existing connections; discovery and connection lifecycle must be designed together.
7. Planned termination publishes drain before removal and measures the complete convergence window.
8. Keep endpoint churn regional; cross-region failover also needs data, identity, policy, and capacity readiness.
9. Fan-out, reconnect storms, and connection pools usually dominate capacity.
10. Record the discovery revision and endpoint incarnation used for every diagnosable request.

---

## References

- [RFC 2782: A DNS RR for Specifying the Location of Services](https://www.rfc-editor.org/rfc/rfc2782) — SRV priority, weight, port, and target semantics
- [RFC 6763: DNS-Based Service Discovery](https://www.rfc-editor.org/rfc/rfc6763) — service-instance discovery using DNS records
- [RFC 8767: Serving Stale Data to Improve DNS Resiliency](https://www.rfc-editor.org/rfc/rfc8767) — bounded stale-answer behavior
- [RFC 9665: Service Registration Protocol for DNS-Based Service Discovery](https://www.rfc-editor.org/rfc/rfc9665) — lease-based registration and TTL/lease separation
- [gRPC Health Checking Protocol](https://grpc.io/docs/guides/health-checking/) — standard service and watch health semantics
- [Kubernetes EndpointSlice](https://kubernetes.io/docs/concepts/services-networking/endpoint-slices/) — scalable endpoint representation, conditions, and topology
- [xDS Transport Protocol](https://www.envoyproxy.io/docs/envoy/latest/api-docs/xds_protocol) — snapshot/delta subscriptions, version, nonce, ACK/NACK, and dependency ordering
- [Load Balancing](../06-scaling/01-load-balancing.md) — endpoint-selection algorithms and load signals
- [DNS and Connection Management](../06-scaling/13-dns-and-connection-management.md) — DNS cache layers, rebinding, and connection lifetime
- [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md) — retry budgets and deadline propagation
- [Multi-Region Architecture](../06-scaling/09-multi-region-architecture.md) — regional routing, data placement, and failover
