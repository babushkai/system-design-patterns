# Service Mesh Data and Control Planes

## TL;DR

A service mesh moves transport concerns (workload authentication, encryption, authorization enforcement, discovery consumption, connection management, retries, and telemetry) into a uniform data plane managed by a control plane. The value is consistent policy across polyglot workloads. The cost is a new distributed system on every request path.

The data plane must keep serving safely when the control plane is unavailable, yet it must not use expired identity, revoked trust, or arbitrarily stale authorization forever. Configuration delivery needs versioned resources, dependency-aware activation, ACK/NACK and applied-state telemetry. Workload identity needs attestation, short-lived credentials, overlapping rotation, trust-bundle transition, and connection renewal.

Sidecars give per-workload isolation and attribution but multiply fixed cost. Node or shared “ambient” proxies amortize resources but enlarge failure and noisy-neighbor domains. Kernel/eBPF interception can reduce hops and improve visibility, but it does not eliminate user-space L7 processing, identity, policy distribution, or correctness risks. Choose the topology per policy layer and workload class rather than treating one deployment model as universally superior.

This chapter covers east-west connectivity planes and failure behavior. [Service Discovery](./01-service-discovery.md) covers endpoint truth, [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md) covers retry mathematics, and [Zero Trust Architecture](../10-security/05-zero-trust-architecture.md) covers the broader security model.

---

## Mesh Boundary and Invariants

~~~mermaid
flowchart TB
    subgraph ControlPlane[Control plane]
        INTENT[Service, identity, and policy intent] --> COMP[Compile resource graph]
        COMP --> DIST[Versioned distribution]
        STATUS[Client status and rollout telemetry] --> COMP
    end

    subgraph DataPlane[Data plane]
        IN[Inbound traffic] --> PEP[Workload-local policy enforcement]
        APP[Application] --> OUT[Outbound traffic policy]
        OUT --> DISC[Endpoint and connection selection]
        DISC --> REMOTE[Remote workload]
        PEP --> APP
    end

    DIST --> PEP
    DIST --> OUT
    PEP --> STATUS
    OUT --> STATUS
    ID[Workload identity plane] --> PEP
    ID --> OUT
~~~

The control plane is not in the normal request path. It compiles intent into resources, distributes them, observes client state, and coordinates rollout. The data plane accepts connections and requests using an active local snapshot.

### Core invariants

1. **Authenticated workload identity:** policy binds to attested workload identity, not mutable IP address or client-supplied header.
2. **Encryption is not authorization:** successful mTLS proves peer identity and channel protection; an explicit policy still decides allowed actions.
3. **Complete active dependency graph:** listener, route, cluster, endpoint, policy, and secret references are present and compatible before use.
4. **Atomic local activation:** a proxy never serves from a half-applied configuration.
5. **Last-known-good is bounded:** configuration may remain usable through a control-plane outage only within its policy and credential validity.
6. **No hidden retry owner:** every attempt is charged to an end-to-end budget.
7. **Traffic attribution survives sharing:** node/shared data planes identify the originating workload before policy evaluation.
8. **Bypass is explicit:** traffic outside capture coverage is denied, separately governed, or intentionally exempt.
9. **Tenant isolation:** configuration, identity, policy, caches, telemetry, and shared proxy memory remain tenant-scoped.
10. **Observed state is distinguishable:** received, validated, acknowledged, warmed, and active revisions are not conflated.

## Data Plane Responsibilities

A mesh data plane may provide:

- workload-to-workload authentication and encrypted transport;
- L4 and L7 authorization enforcement;
- service discovery and locality-aware endpoint consumption;
- connection pooling, protocol negotiation, and draining;
- bounded timeouts, retries, and outlier detection;
- request metadata normalization and propagation;
- traffic splitting and migration;
- metrics, access records, and trace propagation; and
- local overload protection.

It should not own:

- domain authorization that requires current business state;
- distributed transactions or workflow semantics;
- arbitrary application transformations;
- durable business queues;
- database consistency; or
- the truth of whether an endpoint exists.

When mesh policy needs a domain fact on every request, either propagate a verifiable bounded claim, place enforcement in the service, or build a deliberately available policy dependency. Hiding synchronous business lookups inside a proxy filter creates an opaque critical path.

### Request path

~~~text
outbound application call
  -> identify source workload
  -> normalize destination and protocol
  -> authorize egress
  -> resolve eligible endpoints
  -> acquire connection under deadline and pool limits
  -> authenticate peer and validate trust domain
  -> transmit with bounded retry policy
  -> remote data plane authenticates source
  -> authorize inbound action
  -> forward to destination workload
~~~

Every transition has a failure outcome. “Transparent” interception must not mean invisible semantics.

## Control-Plane Resource Distribution

### Resource graph

An xDS-style control plane distributes typed resources rather than one unstructured configuration file:

~~~text
listener/capture resource
  -> route or network policy
      -> logical cluster/service
          -> endpoint assignment
  -> transport security policy
      -> workload credential and trust bundle
  -> authorization and telemetry policy
~~~

Each resource has a stable name, schema/type version, content version, tenant, intended workload selector, dependencies, and provenance. The client is authorized only for its scoped resource set; a node serving multiple tenants must not receive one tenant’s secrets merely because it hosts another tenant’s workload.

### Snapshot and delta protocols

State-of-the-world delivery replaces a complete subscribed resource set. Delta delivery changes named resources and carries explicit removals. Delta reduces bandwidth for large sets but makes deletion, resumption, and compaction semantics critical.

A correct streaming protocol includes:

- authenticated client identity and node/workload attributes;
- explicit subscriptions;
- response version and nonce;
- ACK or NACK tied to that response;
- typed validation errors;
- resume state after reconnect;
- resource TTL/expiry where appropriate;
- dependency ordering or aggregation;
- flow control and slow-client policy; and
- server visibility into last accepted and active state.

An ACK proves only what the protocol defines. Syntax-valid and accepted does not necessarily mean warmed, connected, or serving. Export active-state telemetry or a readiness probe separately.

### Warming and atomic activation

Before an update becomes active:

1. verify publisher, tenant, type version, and content integrity;
2. parse and semantically validate every resource;
3. resolve the complete dependency closure;
4. load referenced certificates and trust bundles;
5. build route and policy indexes off the worker path;
6. initialize mandatory clusters, endpoints, or filters;
7. run local conformance probes;
8. atomically swap the active graph; and
9. report accepted and active versions.

Keep the prior complete graph if validation fails. Do not partially accept a security-sensitive snapshot unless per-resource atomicity and cross-resource consistency are explicitly modeled.

### Eventual consistency and dependency order

Independent streams can temporarily disagree:

~~~text
route update names cluster-v2
-> route arrives before cluster-v2
-> request matches route with unresolved destination
~~~

Controls include:

- aggregate related resources on one ordered stream;
- embed dependency versions in a snapshot manifest;
- stage and warm missing dependencies;
- publish dependencies before references and remove references before dependencies;
- use on-demand resource fetch only with a bounded fail behavior; and
- reject fallback to an unintended broader cluster.

Endpoint churn can remain a separate high-rate stream if the cluster identity and deletion semantics are stable.

## Workload Identity and mTLS Rotation

### Identity lifecycle

~~~mermaid
sequenceDiagram
    participant W as Workload
    participant A as Local identity agent
    participant CA as Identity authority
    participant P as Data plane
    participant R as Remote peer

    W->>A: local workload API call
    A->>A: attest caller attributes
    A->>CA: request/renew identity
    CA-->>A: short-lived identity + trust bundles
    A-->>P: streamed credential update
    P->>P: validate and stage new key/cert
    P->>R: new connections use rotated identity
    P->>P: drain old connections before old identity expires
~~~

The identity presented on the wire should name the workload and trust domain, not its host. Issuance is based on attested runtime attributes under an authorized registration policy. A local workload API must identify its caller through an out-of-band mechanism such as operating-system peer credentials; accepting a requested identity from the workload payload defeats attestation.

### Rotation windows

For credential lifetime $T_{\text{cert}}$, rotate with enough margin for issuance, distribution, clock uncertainty, and connection replacement:

$$
T_{\text{rotate-start}} +
T_{\text{issue}} +
T_{\text{distribute}} +
T_{\text{drain}} +
T_{\text{clock-margin}}
< T_{\text{cert}}.
$$

Use overlapping validity:

1. receivers trust old and new issuing material;
2. senders receive new credentials;
3. new connections present new identity;
4. long-lived old connections drain or reauthenticate;
5. fleet telemetry proves migration;
6. old credentials and trust material expire or are revoked.

Rotating a file on disk is not enough. A proxy may retain old TLS contexts and existing connections. Track credential loaded time, not-after horizon, connections by identity generation, and trust-bundle generation.

### Trust-bundle rotation

Root or trust-bundle rotation is a distributed compatibility migration. Removing an old root before every peer trusts the new root partitions the mesh. Keeping a compromised root indefinitely preserves the attack.

Use:

- signed/versioned trust bundles;
- overlapping roots for planned migration;
- explicit federation between trust domains;
- minimum accepted bundle generation;
- emergency revocation behavior;
- connection renewal; and
- convergence telemetry by workload cohort.

Federation authenticates foreign workload identities; it does not automatically authorize them. Local policy maps foreign trust domains and identity paths to allowed services and actions.

### Identity failure behavior

| Condition | Existing connection | New connection |
|---|---|---|
| Identity service unavailable; credential still valid | Continue if policy allows | Use valid credential |
| Credential near expiry | Prefer drain and refresh | Avoid creating connection that cannot satisfy operation horizon |
| Credential expired | Terminate or reject according to protocol safety | Reject |
| Trust bundle stale but valid | Continue within stale policy | Validate under accepted bundle generation |
| Issuer/root revoked | Invalidate affected authorization and reconnect | Reject revoked chain |
| Caller cannot be attested | No identity upgrade | Reject identity issuance |

Failing open to plaintext after mTLS failure is a protocol downgrade and normally forbidden.

## Enforcement Layers

### L4 and L7 policy

L4 policy can bind source identity to destination identity, port, and transport. It works for opaque protocols and shared infrastructure but cannot distinguish HTTP methods, RPC operations, or resource paths.

L7 policy can enforce operation-aware rules after parsing the application protocol. It adds parser compatibility, CPU and memory cost, encrypted payload termination, schema evolution, and a larger attack surface.

Use the lowest layer that expresses the policy:

| Requirement | Enforcement location |
|---|---|
| Workload A may connect to database B | L4 identity policy |
| Caller may invoke one RPC method | L7 proxy or service |
| Caller may update only resources it owns | Domain service with current state |
| Payload field must satisfy business invariant | Domain service |
| Public client authentication | Edge gateway plus downstream defense |

The mesh should attach verified peer identity through a protected local channel when the application needs it. The service must distinguish this from client-controlled headers.

### Capture coverage and bypass

Traffic interception can use:

- explicit proxy addresses or application integration;
- namespace/network redirection;
- node routing;
- transparent socket interception; or
- kernel hooks and eBPF programs.

Define coverage for:

- inbound and outbound TCP;
- UDP and other protocols;
- loopback and same-node traffic;
- host-network or privileged workloads;
- direct IP, virtual IP, and DNS destinations;
- startup before proxy readiness;
- shutdown after proxy drain;
- init, debug, and ephemeral processes; and
- traffic created by the proxy itself.

The platform should test for bypass, not assume an injected component captures everything. Network policy can make the proxy path mandatory; otherwise a compromised workload may connect directly around L7 authorization.

## Deployment Topologies

### Comparison

| Topology | Isolation and attribution | Fixed resource cost | Failure domain | Upgrade model | Best fit |
|---|---|---:|---|---|---|
| Application library | Exact in-process identity | Per process/library | Application process | Language release | Homogeneous fleet, narrow features |
| Per-workload sidecar | Strong workload boundary | Highest fixed replica cost | One workload/pod | Coordinated proxy injection/rollout | Polyglot, strict isolation, rich L7 |
| Node proxy | Requires trustworthy source attribution | Amortized per node | Many workloads on node | Daemon/node rollout | High pod density, mostly L4/shared L7 |
| Shared namespace or “ambient” proxy | Intermediate | Shared across workload group | Namespace/segment | Independent shared tier | Mixed workloads with selective L7 |
| Kernel/eBPF plus user-space policy | Strong kernel visibility; identity mapping is critical | Low per-flow kernel overhead plus agents | Node/kernel program | Kernel/agent rollout | L3/L4 capture, telemetry, acceleration |

### Sidecar

Advantages:

- resource and failure isolation follows one workload;
- clear source identity and per-workload policy;
- independent connection pools and overload state;
- easy support for different trust or protocol requirements.

Costs:

- one proxy runtime and configuration stream per workload replica;
- duplicated listeners, clusters, pools, certificates, and telemetry buffers;
- application/proxy startup and shutdown ordering;
- fleet-wide injection and version skew;
- more local hops and context switches.

Sidecar lifecycle must be native to the workload scheduler where possible: start before application traffic, remain through application drain, and terminate after forwarding completes. A generic helper container that exits too early breaks request drain.

### Node or shared proxy

Advantages:

- amortized memory, connections, and control-plane streams;
- independent upgrades from application pods;
- fewer local processes and potentially fewer duplicate upstream pools.

Costs:

- one proxy fault affects many workloads;
- per-tenant fair scheduling and memory isolation become mandatory;
- source workload attribution must be unspoofable;
- policy and secret scoping are harder;
- shared pools can leak identity or authorization context if keys are incomplete.

A shared connection can be reused across callers only when transport identity, destination, policy, tenant, protocol settings, and upstream authorization semantics permit it.

### Kernel/eBPF-assisted data plane

Kernel programs can classify, redirect, load-balance, and observe traffic early with low per-packet overhead. They are constrained by verifier rules, kernel compatibility, map capacity, upgrade safety, and limited application-protocol context. Complex L7 parsing, TLS termination, request retries, and rich transformations still require a user-space component or application.

Treat eBPF as a placement option for specific data-plane functions, not a synonym for a complete mesh. Verify behavior across kernel versions and make program/map rollback as rigorous as proxy configuration rollout.

### Layered topology

A practical system can combine:

~~~text
kernel or node layer: capture, workload attribution, L3/L4 policy, basic load balancing
shared or per-workload L7 layer: HTTP/RPC auth, retries, telemetry, transformations
application: business authorization and invariants
~~~

This avoids paying L7 cost for opaque or simple traffic while keeping enforcement at the layer with sufficient semantics.

## Discovery, Connections, and Draining

The data plane consumes endpoint state but does not own it. Preserve the endpoint revision and incarnation described in [Service Discovery](./01-service-discovery.md).

### Connection pools outlive endpoint snapshots

When an endpoint becomes draining or removed:

- stop assigning new requests;
- stop creating connections;
- signal drain on multiplexed protocols;
- bound new streams on old connections;
- allow or cancel in-flight work according to operation semantics;
- close connections after a measured deadline; and
- never revive the endpoint solely because a pooled connection still exists.

Pool keys include destination service, endpoint, source identity, tenant, transport policy, protocol options, and relevant authorization context. Omitting identity from a shared pool key can present the wrong certificate or reuse a connection under the wrong caller policy.

### Local outlier detection

Local failure evidence can temporarily eject an endpoint for one proxy. It should:

- use bounded ejection and recovery probes;
- distinguish connection, TLS, protocol, overload, and application errors;
- avoid ejecting the last endpoint without an explicit policy;
- cap total ejected capacity;
- account for low-traffic statistical uncertainty; and
- remain separate from authoritative registry health.

The full selection algorithms live in [Load Balancing](../06-scaling/01-load-balancing.md).

## Retries, Timeouts, and Retry Amplification

Meshes make retries easy to configure and therefore easy to multiply. Suppose a caller and two proxy/service layers each allow $a_1$, $a_2$, and $a_3$ attempts:

$$
A_{\text{worst}} = a_1 a_2 a_3.
$$

A declarative retry policy does not know whether a non-idempotent application operation committed before a reset. The route contract must classify method safety and idempotency, and the application must participate when deduplication is required.

### Mesh retry invariants

- one owner controls retries for each hop;
- all layers share an end-to-end deadline and attempt budget;
- per-try timeout leaves time to process a response;
- retries consume a bounded global or route budget;
- overload and explicit refusal suppress retries;
- request bodies are replayed only within bounded memory/disk policy;
- retry to another endpoint does not violate session or data locality; and
- telemetry exposes original request versus attempts.

See [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md) and [Idempotency](../01-foundations/08-idempotency.md).

## Control-Plane Outage and Stale State

Separate configuration classes because they expire differently:

| State class | Examples | Safe through outage? |
|---|---|---|
| Static bootstrap | control-plane address, trust anchor, local admin listener | Yes, narrowly scoped and protected |
| Routing/discovery | service clusters, endpoints, traffic split | Bounded last-known-good; endpoint/drain risk applies |
| Authorization | identity allow/deny and operation policy | Bounded by policy revision/expiry; privileged effects may require freshness |
| Identity credential | private key and certificate | Only until cryptographic expiry/revocation |
| Trust bundle | issuer roots and federation state | Only while accepted generation remains valid |
| Telemetry config | sampling/export target | Data plane should continue if export fails |

### Data-plane behavior

1. Continue with a complete last-known-good snapshot if its class-specific validity permits.
2. Reject malformed or incompatible updates without discarding valid active state.
3. Expose control-plane disconnect duration and active-state age.
4. Keep serving data traffic independently from telemetry export.
5. Preserve config and identity refresh resources during overload.
6. Fail readiness for *new governed traffic* when required policy or identity expires, not necessarily kill the entire process.
7. Reconnect with jitter and resume state; avoid a fleet thundering herd.

At first startup with no verified policy, expose only a minimal bootstrap-safe surface. “Control plane unavailable” must not mean “accept all traffic.”

## Safe Policy and Proxy Rollout

### Configuration rollout

- compile and validate the resource graph centrally;
- replay a representative request corpus through old and candidate policy;
- compare allow/deny, route, retry, timeout, identity, and telemetry changes;
- distribute to a canary cohort;
- require received, accepted, warmed, and active telemetry;
- watch request outcomes and NACK reasons by proxy build;
- expand by failure domain, not random individual process only; and
- retain an authorized compatible rollback.

### Data-plane binary rollout

Configuration compatibility is bidirectional. A new control-plane resource may be ignored or rejected by an old proxy; a new proxy may validate an old resource more strictly.

Maintain:

- declared resource type compatibility range;
- conformance suite shared by control and data planes;
- N-minus-one and N-plus-one interoperability tests;
- unknown-field and unknown-enum policy;
- proxy build in ACK/NACK and decision telemetry;
- canary with representative protocols and tenant policies; and
- rollback that does not require an unsupported config downgrade.

### Policy migration

For a deny-to-allow change, shadow the candidate and review new permits before enforcement. For an allow-to-deny change, identify affected traffic and provide a migration window unless it is emergency revocation. Never evaluate a proposed policy change using authority granted only by that same unapproved change.

## Performance and Capacity

The mesh adds work per byte, connection, request, policy check, and configuration stream.

Let:

- $\lambda$ be application requests per second;
- $A$ be mean attempts including retries;
- $b$ be mean transferred application bytes;
- $C$ be concurrent connections;
- $P$ be data-plane instances;
- $R$ be configuration resources per instance;
- $U$ be relevant resource updates per second; and
- $W$ be subscribed instances affected by an update.

Data-plane request processing rate is approximately:

$$
\lambda_{\text{proxy}} = 2\lambda A
$$

for an outbound and inbound proxy hop, before extra gateways or shared tiers. Application-layer processed byte rate is at least:

$$
B_{\text{proxy}} \approx 2\lambda A b,
$$

excluding encryption framing, telemetry, retries with different body sizes, and protocol translation.

### Fixed versus shared cost

For per-workload sidecars:

$$
M_{\text{sidecar fleet}} \approx
\sum_{p=1}^{P}
\left(
M_{\text{runtime},p}
+ M_{\text{config},p}
+ M_{\text{pools},p}
+ M_{\text{telemetry},p}
\right).
$$

A shared proxy amortizes runtime/configuration but needs capacity for aggregate connections, tenants, and failure isolation:

$$
M_{\text{shared}} \approx
M_{\text{runtime}} +
\sum_t(M_{\text{policy},t}+M_{\text{pools},t}+M_{\text{queues},t}).
$$

Do not compare only average memory. Compare tail latency, connection reuse, noisy-neighbor isolation, restart blast radius, and control-plane stream count.

### Control-plane fan-out

Naive update delivery work is:

$$
F_{\text{updates}} \approx U W.
$$

Endpoint churn for a popular service and fleet reconnect after outage dominate averages. Use regional fan-out, delta resources, subscription indexes, update coalescing, bounded queues, and fair tenant scheduling. A slow proxy must resnapshot rather than retain an unbounded delta backlog.

### Connection and TLS costs

Measure:

- active and idle connections per source/destination/identity;
- multiplexed streams and head-of-line effects;
- handshake rate, resumption, and certificate verification cost;
- connection churn during endpoint or credential rotation;
- policy evaluation and L7 parsing CPU;
- buffer memory under slow readers/writers;
- telemetry serialization/export;
- local-hop copies, context switches, and kernel/user transitions; and
- activation overlap while old and staged configs coexist.

High pod density can make duplicated sidecar pools far more expensive than request processing. Shared pools reduce connections only if identity and policy boundaries allow reuse.

## Failure Modes and Traces

### Invalid configuration creates cohort divergence

~~~text
control plane publishes version 84
-> new proxies accept it, old proxies NACK unknown requirement
-> traffic policy differs by proxy build
-> rollout dashboard shows only “delivered”
~~~

**Controls:** compatibility range, conformance tests, accepted/active telemetry by build, decision diff, and halted rollout on NACK or semantic divergence.

### Control-plane outage outlives authorization state

~~~text
proxies keep last-known-good routes
-> authorization policy expires or is revoked
-> routing remains available but privileged traffic uses stale permit
~~~

**Controls:** separate validity per resource class, revocation epoch, privileged fail-closed mode, and visible stale age.

### Credential rotates but connections do not

~~~text
proxy loads new certificate -> existing multiplexed connections stay open
-> old credential expires or root is removed
-> long-lived calls reset unpredictably
~~~

**Controls:** identity generation on connection, proactive drain, overlapping trust, operation-horizon check, and connection convergence telemetry.

### Root rotation partitions the mesh

~~~text
issuer switches to new root -> some receivers lack new bundle
-> new connections fail only across certain cohorts
-> retries amplify handshakes and obscure cause
~~~

**Controls:** trust-first rollout, overlap, bundle generation telemetry, canary cross-cohort matrix, and retry suppression for deterministic TLS failure.

### Mesh retries amplify overload

~~~text
upstream latency rises -> proxy per-try timeout fires
-> retries select other endpoints -> caller library also retries
-> attempts consume remaining healthy capacity
~~~

**Controls:** one retry owner, end-to-end budget, overload-aware ejection/admission, and original-request versus attempt metrics.

### Shared node proxy becomes a noisy-neighbor boundary

~~~text
one tenant opens many slow streams
-> shared buffers and connection table fill
-> unrelated workloads on node lose connectivity
~~~

**Controls:** per-tenant/workload quotas, fair queues, memory accounting, priority control traffic, and node-level blast-radius testing.

### Capture bypass avoids policy

~~~text
workload connects by direct address or uncovered protocol
-> packet bypasses user-space L7 proxy
-> expected authorization and audit never run
~~~

**Controls:** coverage inventory, mandatory network path, egress network policy, bypass tests, and service-side authorization for critical effects.

### Telemetry export blocks data traffic

~~~text
collector unavailable -> access-log buffer grows
-> proxy blocks worker or exhausts memory
-> request path fails although policy and upstream are healthy
~~~

**Controls:** bounded async buffers, sampling/drop policy, disk isolation where justified, and telemetry health separate from data-plane readiness.

### Endpoint removal does not drain pooled connections

~~~text
discovery removes endpoint -> pool keeps existing connection
-> new streams continue to terminating workload
-> application exits and resets effects in progress
~~~

**Controls:** endpoint incarnation in pool, drain integration, protocol shutdown signal, and measured grace period.

## Observability

### Control plane

- source, compiled, desired, delivered, ACK/NACK, warmed, and active revisions;
- client build, resource schema compatibility, reconnect, resume, and resnapshot;
- compilation latency, dependency-graph size, fan-out queue, slow consumers, and rejected resources;
- policy decision deltas between old and candidate;
- credential and trust-bundle convergence;
- stale-state age and resource expiry horizon; and
- rollout coverage by region, zone, tenant, and topology.

### Data plane

- requests versus attempts, stage latency, bytes, and active streams;
- source/destination workload identities and policy rule IDs in secured traces;
- mTLS version, identity generation, handshake/resumption outcome, and expiry horizon;
- authorization permit/deny/indeterminate by bounded policy namespace;
- discovery revision, selected endpoint incarnation, local ejection reason, and drain state;
- pool size, queue, connection age, and reuse by safe bounded class;
- retry suppression, deadline exhaustion, reset/cancel, and overload shed;
- proxy CPU, memory, buffers, kernel drops, and telemetry queue; and
- bypass/unencrypted traffic detections.

Metrics avoid raw workload identity, path, certificate serial, endpoint address, and policy text as labels. Join high-cardinality evidence in secured logs or traces.

## Verification Strategy

| Test layer | What to prove |
|---|---|
| Resource schema tests | Unknown critical fields, type versions, dependencies, and tenant scopes reject safely |
| Control/data conformance | Compiler and every supported proxy build agree on policy decisions and routes |
| Distribution tests | ACK/NACK, reconnect, delta removal, compaction, gap, warming, and atomic activation |
| Identity tests | Caller attestation, key protection, credential rotation, trust overlap, expiry, and revocation |
| Capture tests | All intended protocols, loopback, same-node, startup, shutdown, debug, and direct-address paths |
| Authorization tests | Peer authentication cannot bypass L4/L7/domain policy; indeterminate fails as designed |
| Retry tests | Attempts remain within budget under resets, overload, timeouts, and nested callers |
| Drain tests | Removed endpoint gets no new streams and long-lived work follows the declared contract |
| Isolation tests | Shared proxy cannot cross tenant identity, policy, secret, pool, cache, or telemetry state |
| Performance tests | Connection churn, TLS rotation, large bodies, slow streams, L7 parse, and config activation overlap |
| Failure tests | Control-plane loss, identity loss, collector loss, node-proxy crash, bad update, and regional partition |
| Kernel matrix | eBPF/program behavior, map pressure, upgrade, rollback, and fallback across supported kernels |

Test policy on the actual intercepted representation. Unit tests of a declarative rule do not prove that traffic reaches the enforcement point or that source attribution is authentic.

## Decision Framework

### Should you deploy a mesh?

A mesh is justified when several are true:

- many services and languages need uniform workload identity and transport policy;
- teams repeatedly implement incompatible discovery, TLS, retry, or telemetry logic;
- east-west authorization must be centrally governed and locally enforced;
- traffic migration requires fleet-wide, observable rollout;
- the platform can operate a highly available control plane and data plane;
- the added latency, resource cost, and debugging complexity are measured and acceptable.

Avoid or narrow it when:

- a small homogeneous system can use a well-maintained library;
- only ingress needs mediation;
- workloads are so latency/resource constrained that proxy cost dominates;
- the organization cannot operate identity rotation and config rollout safely; or
- policy requires business state available only inside services.

### Choose topology by layer

1. What traffic and protocols must be captured?
2. Which policies need L4, L7, or domain semantics?
3. What is the smallest trustworthy source-attribution boundary?
4. What failure radius is acceptable: workload, namespace, node, or region?
5. Can connections be shared across identities and tenants?
6. What fixed per-workload overhead is affordable?
7. What kernel/platform compatibility must be supported?
8. How will identity and trust rotate across long-lived connections?
9. How stale may routing, authorization, and credentials become independently?
10. Can every request be tied to active config, policy, identity, and endpoint revisions?

Use sidecars where per-workload L7 isolation is essential, shared/node layers where amortization and L4 policy dominate, and kernel assistance for capture/acceleration where its semantic limits are acceptable. Hybrid is a design, not a failure to choose.

## Key Takeaways

1. A mesh is a distributed data plane plus a configuration and identity control plane.
2. ACK, warmed, and active are different states; safe rollout observes all three.
3. Resources activate as a compatible dependency graph, not independent text fragments.
4. mTLS authenticates and protects a channel; authorization still decides permitted actions.
5. Credential rotation must replace TLS contexts and long-lived connections before expiry.
6. Last-known-good routing can outlive a control-plane outage; expired identity or revoked policy cannot.
7. Sidecars maximize per-workload isolation but multiply fixed cost; shared topologies trade cost for larger failure domains.
8. eBPF improves capture and L3/L4 processing but does not replace L7 semantics, identity, or control-plane correctness.
9. Mesh retries must share one deadline and attempt budget with callers and services.
10. Test capture, attribution, isolation, rotation, stale state, and configuration convergence, not only happy-path routing.

---

## References

- [xDS Transport Protocol](https://www.envoyproxy.io/docs/envoy/latest/api-docs/xds_protocol): typed resources, state-of-the-world/delta delivery, ACK/NACK, TTL, and consistency
- [SPIFFE Standards](https://spiffe.io/docs/latest/spiffe-specs/): workload identities, verifiable identity documents, trust domains, bundles, and federation
- [SPIFFE Workload API](https://spiffe.io/docs/latest/spiffe-specs/spiffe_workload_api/): caller identification, streaming credentials, and rotation delivery
- [RFC 8446: TLS 1.3](https://www.rfc-editor.org/rfc/rfc8446): TLS authentication, key establishment, and connection security
- [Kubernetes Sidecar Containers](https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/): scheduler-native sidecar lifecycle semantics
- [Linux Kernel BPF Documentation](https://docs.kernel.org/bpf/): verifier, maps, program types, and kernel execution model
- [RFC 9113: HTTP/2](https://www.rfc-editor.org/rfc/rfc9113): multiplexed streams, flow control, connection errors, and graceful shutdown
- [Service Discovery](./01-service-discovery.md): endpoint lifecycle, watches, stale state, and draining
- [Zero Trust Architecture](../10-security/05-zero-trust-architecture.md): identity-centric security and continuous authorization
- [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md): deadline and retry-budget design
- [Distributed Tracing and Telemetry Pipelines](../11-observability/01-distributed-tracing.md): trace propagation and cross-service causality
- [Backpressure](../06-scaling/07-backpressure.md): bounded queues, flow control, and overload behavior
