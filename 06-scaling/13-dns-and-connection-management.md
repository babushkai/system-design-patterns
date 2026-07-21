# DNS and Connection Management

DNS publishes names; resolvers turn those names into time-bounded answers; connection pools turn one answer into routing state that can outlive every DNS cache involved. Production cutovers fail when teams reason about only the authoritative record. The operational unit is the entire chain: authoritative publication, recursive and client caches, address-family selection, connection creation, pool lifetime, and endpoint drain.

Scope: **DNS delegation and cache semantics, positive and negative TTLs, stale-answer policy, client resolution, dual-stack connection racing, pool/address lifecycle, DNS cutovers, and connection draining**. [Service Discovery](../12-service-mesh/01-service-discovery.md) owns endpoint authority, registration, health, watch streams, and control-plane state. [Load Balancing](./01-load-balancing.md) owns how a ready endpoint is selected and weighted. [Network Transport Internals](./14-network-transport-internals.md) owns TCP, TLS, QUIC, congestion control, and packet-level failure mechanics. [Multi-Region Architecture](./09-multi-region-architecture.md) owns regional authority and failover policy.

## Primary Evidence and Scope

| Primary evidence | What it establishes | Boundary |
|---|---|---|
| RFC 1034 and RFC 1035 (1987) | DNS namespace, delegation, iterative/recursive resolution, RRsets, caching, and wire protocol foundations | Later RFCs update details; an implementation must follow the current standards set, not these two documents alone |
| RFC 2308 (1998) | NXDOMAIN and NODATA are cacheable; the enclosing SOA controls negative-cache lifetime | Does not make transient server failure equivalent to nonexistence |
| RFC 8767 (2020) | A recursive resolver may serve expired data under a bounded stale-answer policy during refresh or authority failure | Availability feature with an explicit freshness cost, not permission to cache forever |
| RFC 8305 (2017) | Happy Eyeballs v2 races IPv6/IPv4 candidates to reduce delay from a broken address family | Client connection-establishment algorithm, not endpoint health checking |
| RFC 9460 (2023) | SVCB and HTTPS records can publish alternative endpoints and connection parameters with defined alias/service modes | Clients that do not implement the records continue through their legacy path |
| Meta engineering report, October 2021 | Backbone loss and withdrawn DNS advertisements made an otherwise global name system a shared-fate dependency | One documented incident, not a universal DNS architecture |

## Resolution and connection contract

Before choosing a TTL or pool size, define:

| Field | Required answer |
|---|---|
| **Name and purpose** | User-facing origin, API, internal service, failover alias, discovery name, validation token, or delegation? |
| **Authority** | Which zone and provider are authoritative, and how is ownership changed or recovered? |
| **Record contract** | Which RR types, targets, ports, priorities, service parameters, and DNSSEC expectations may clients consume? |
| **Cache contract** | Positive TTL, negative TTL, minimum/maximum client cache, serve-stale window, and cache-flush capability? |
| **Resolver population** | Enterprise/VPC/public recursors, local caching daemons, stub resolvers, runtimes, libraries, and proxies? |
| **Failure result** | Use a fresh answer, bounded stale answer, alternate name, cached connection, queue, or fail closed? |
| **Connection contract** | Protocol, pool key, concurrency per connection, idle timeout, maximum age, keepalive, and drain signal? |
| **Cutover objective** | Time to admit the new endpoint, stop new work on the old endpoint, and terminate old connections? |
| **Security boundary** | DNSSEC validation, split-horizon view, egress policy, rebinding defense, and change authorization? |
| **Evidence** | Which resolvers and client versions must demonstrate convergence before the old path is removed? |

“TTL 60” does not answer this contract. It says how long a normally behaving cache may treat one received RRset as fresh. It does not close a socket, bound an explicitly configured stale-answer window, ensure all authoritative replicas have the update, or force an application runtime to ask again.

## The end-to-end state machine

DNS has a publication control path and a resolution data path. Connection reuse creates a second data path after resolution.

~~~mermaid
flowchart LR
    subgraph Control[Publication and endpoint control]
        O[Change controller] --> A[Authoritative zone]
        H[Endpoint readiness] --> O
        O --> D[Drain controller]
    end

    subgraph Resolution[Resolution path]
        C[Client runtime] --> S[OS stub or local cache]
        S --> R[Recursive resolver]
        R --> P[Root and parent delegation]
        R --> A
    end

    subgraph Connections[Retained routing state]
        C --> X[Address-family and candidate racing]
        X --> Q[Connection pool]
        Q --> E[Serving endpoint]
    end

    A -. RRset plus TTL .-> R
    R -. cached answer .-> S
    S -. addresses and service parameters .-> C
    D -. stop new work and close gracefully .-> E
~~~

The normal data path should continue using valid cached answers and established healthy connections if the DNS publication API is unavailable. That separation is valuable only while revocation and drain remain bounded: a permanent client cache or unbounded connection age silently converts a transient control-plane outage into inability to move traffic.

## State, authority, and invariants

The relevant state spans several owners.

**Authoritative publication state:**

```text
zone and owner name
RRset type, class, values, TTL, and routing policy if provider-specific
SOA fields including serial, SOA TTL, and MINIMUM
delegation NS set, glue, and parent-side TTL
DNSSEC keys, signatures, delegation signer state, and rollover phase
change revision, actor, approval, and authoritative-replica rollout status
endpoint readiness evidence and intended removal/drain deadline
```

**Resolver/client cache state:**

```text
query name, type, class, and policy scope
answer/referral/negative result and validation status
received time, remaining fresh TTL, and source
stale-until bound and stale-serving reason
last refresh result and retry schedule
```

**Connection-pool state:**

```text
pool key: scheme/authority/proxy/security identity as applicable
remote address, port, protocol, and resolution generation
created time, last-used time, idle deadline, and maximum-age deadline
active streams/requests, queued demand, and capacity
ready, suspect, draining, or closed state
endpoint and certificate identity evidence
```

**Reference-design invariants:**

1. An endpoint is not published or admitted until it is ready for the advertised protocol, identity, data, and expected load.
2. Every cache result is distinguishable as fresh positive, fresh negative, bounded stale, transient failure, or policy rejection; applications do not collapse them into “host missing.”
3. A DNS change never counts as revocation of an established connection. Revocation has an application, proxy, transport, firewall, or credential mechanism.
4. Every connection derived from a movable name has a bounded lifetime or an explicit drain/re-resolution signal.
5. Removing the old endpoint waits for both the measured DNS convergence population and the declared connection-drain bound, or intentionally accepts the remaining failures.
6. A resolver outage cannot create unbounded synchronous lookup queues or unbudgeted retries.
7. Parent delegation, authoritative zone, TLS identity, and network reachability are changed in a sequence with at least one working end-to-end path.
8. DNSSEC signing and delegation state never expose an interval in which validating clients see an unverifiable chain unless an explicit, tested insecure transition is the policy.
9. Rollback preserves the old endpoint, credentials, data compatibility, and capacity until the rollback horizon closes.

## DNS resolution mechanics

### Delegation, referrals, and glue

A stub resolver usually asks a recursive resolver for a complete answer. On a cache miss, the recursive resolver follows referrals from the root to a top-level domain and then to the authoritative zone. A referral names authoritative servers with `NS` records. When an authoritative server's name is inside the delegated child, the parent supplies address **glue** to break the circular dependency.

Delegation is state in two administrative domains: the parent and the child. Updating the child zone's `NS` RRset does not update the registrar/parent delegation, and the two TTLs can differ. A nameserver migration therefore overlaps old and new servers long enough for parent referrals, glue, child data, and resolver caches to converge.

The SOA serial helps authoritative secondary servers decide whether to transfer a newer zone. It is not a cache-invalidation broadcast to recursive resolvers. A serial increment does not make clients discard a still-fresh RRset.

### RRsets and aliases

DNS caches an RRset—records with the same owner, type, and class—as a unit. Important contracts include:

- `A` and `AAAA` publish IPv4 and IPv6 addresses. Multiple addresses are candidates; record order is not a portable health or weighting contract.
- `CNAME` makes one owner an alias to another canonical name. Resolution follows the alias chain, and each RRset in the chain has its own TTL and failure surface.
- `SRV` publishes service, protocol, priority, weight, port, and target for clients that implement that service's SRV contract. It is not automatically honored by generic HTTP clients.
- `SVCB` and `HTTPS` can publish aliases, alternative endpoints, ports, protocol negotiation hints, and address hints. RFC 9460 defines mandatory-key behavior so clients do not silently ignore parameters declared essential.
- `NS`, `SOA`, `DS`, `DNSKEY`, and signature records participate in authority and DNSSEC rather than ordinary endpoint selection.

Provider-specific alias-at-apex, geographic, latency, weighted, and health-check features are not one standard DNS algorithm. Document their consistency, health source, resolver-location behavior, and failover limits. Endpoint choice after a client has candidates belongs to [Load Balancing](./01-load-balancing.md).

### Positive, negative, and transient results

A positive answer remains fresh for the remaining TTL received from its upstream cache. A downstream cache must not restart the original authoritative TTL; it receives a decremented value.

RFC 2308 distinguishes:

- **NXDOMAIN:** the queried name does not exist.
- **NODATA:** the name exists, but no record of the requested type exists.

Both can be negatively cached using the enclosing SOA. The negative TTL is bounded by the lower of the SOA RR's TTL and its `MINIMUM` field. This makes “query the future name, then create it” a real rollout hazard: early clients can retain nonexistence after the record is published.

Timeout, `SERVFAIL`, validation failure, and connection refusal are not proof of nonexistence. Applications that convert every resolution error to NXDOMAIN can cache an authority outage as a fabricated deletion. Preserve result class through the resolver API and retry only within a bounded budget.

### Expiry and serve-stale

At TTL expiry, a resolver normally refreshes before treating data as fresh. RFC 8767 permits serving expired data when refresh fails or while refresh is attempted, subject to implementation policy and a stale-retention bound. This can keep a service reachable during an authoritative outage, but it can also keep a retired address in use after its nominal TTL.

Define stale policy by name class:

| Name class | Typical policy question |
|---|---|
| Public read endpoint | Is bounded stale reachability safer than hard failure? |
| Credential/revocation endpoint | Could stale data bypass a security action? |
| Internal service | Can the endpoint remain valid and fenced through the stale window? |
| Newly created validation name | Will stale NXDOMAIN delay issuance or ownership proof? |
| Disaster-recovery alias | Does stale data point to a site that is unavailable or no longer authoritative? |

Last-known-good data is useful only if the old endpoint is deliberately kept safe for that duration. “Our resolver may serve stale for 24 hours” and “we terminate the old site after 10 minutes” are incompatible policies.

### DNS transport is part of availability

DNS commonly starts over UDP, but larger answers may be truncated and retried over TCP; DNSSEC increases response size. Standards require implementations to support TCP, and EDNS(0) advertises larger UDP capabilities. Firewalls that allow UDP/53 but block TCP/53 create size-dependent failures that resemble random names or networks being broken.

Encrypted resolver transports such as DNS-over-TLS and DNS-over-HTTPS change the path, certificate dependency, connection reuse, and observability boundary; they do not change authoritative TTL semantics. Inventory which resolver is actually used in containers, browsers, service meshes, VPNs, and host runtimes before diagnosing “DNS.”

## Cache hierarchy and convergence model

There may be caches in the recursive resolver, node-local daemon, OS, runtime, HTTP proxy, and application. Some APIs return only addresses and hide TTLs, so a runtime may apply its own fixed cache policy. Measure behavior for the exact runtime version; do not infer it from the zone setting.

For cache population $i$, let:

- $F_i$ be its maximum remaining fresh lifetime at the moment of change;
- $S_i$ be its permitted stale extension;
- $P_{\mathrm{auth}}$ be authoritative-provider publication delay;
- $P_{\mathrm{obs}}$ be the observation/health-check interval.

A useful planned DNS-answer horizon is:

$$
D_{\mathrm{DNS}} \ge P_{\mathrm{auth}} + \max_i\left(F_i + S_i\right) + P_{\mathrm{obs}}
$$

This is an engineering bound, not a DNS protocol guarantee. Unknown resolvers, noncompliant caches, disconnected clients, and application pinning create an unbounded tail. Decide which client population the SLO covers and retain a rollback path for the rest.

Lowering a TTL is itself a cached change. If an RRset currently has TTL 3,600 seconds, publish the lower TTL and wait at least the old cache horizon before depending on the new value. Lowering it five minutes before a cutover cannot recall copies that were fetched ten minutes earlier with 3,600 seconds of freshness.

### Authoritative query-rate estimate

Let $R$ be the number of independently caching recursive populations querying one RRset and $T$ its TTL. Under smooth, continuously active demand, a first lower-bound estimate is:

$$
\lambda_{\mathrm{auth,RRset}} \approx \frac{R}{T}
$$

**Illustrative:** 12 million clients aggregated behind about 24,000 independently caching resolver populations, with TTL 60 seconds, produce roughly `24,000 / 60 = 400` authoritative refreshes/s for one continuously requested RR type—not 200,000/s. That estimate omits cold caches, client subnet partitioning, several record types, alias-chain lookups, retry bursts, multiple anycast sites, and synchronized expiry.

Low TTL increases dependency on resolver and authoritative availability. Cache flush, restart, popular-name expiry, or reconnect storms make refreshes bursty, so capacity-test misses, DNSSEC responses, TCP fallback, and provider/API failure independently of average QPS.

## Address-family selection and connection establishment

A client can receive several IPv6 and IPv4 candidates with no reliable signal that every path works. Serially attempting all IPv6 addresses before IPv4 can turn one broken family into seconds of delay. RFC 8305's Happy Eyeballs v2 guidance builds an ordered candidate list and starts later alternatives after a short, adaptive delay so a working family can win without permanently preferring IPv4.

The winner is connection-local evidence, not a global declaration that one address family is bad. Cache family/path success carefully and expire it; otherwise a brief IPv6 incident can pin all future connections to IPv4 after recovery. Test partial black holes, slow handshakes, NAT64/DNS64 environments, and mixed `A`/`AAAA` rollout.

Resolution success and connection success are separate telemetry events. An address can resolve correctly while routing, firewall, TLS identity, protocol negotiation, or application readiness fails.

## Connection pools are routing caches

Once a connection is established, DNS is usually no longer consulted for requests on that connection. HTTP/2 and HTTP/3 can carry many concurrent streams; databases and other protocols may hold sessions for hours. A DNS migration therefore has two independent convergence questions:

1. When will new resolutions stop returning the old target?
2. When will connections created from old answers stop carrying work?

A pool should declare:

- its exact key, including authority, proxy, transport, security identity, and protocol constraints;
- maximum active and idle connections, queue capacity, and fairness;
- maximum streams or in-flight operations per connection;
- connect, handshake, request, idle, and maximum-age deadlines;
- endpoint/address generation and re-resolution behavior;
- reaction to `GOAWAY`, connection-close, endpoint drain, certificate rotation, and resolver failure;
- retry semantics for work whose connection fails with an ambiguous outcome.

Do not key a security-sensitive pool by raw IP alone. Different names can resolve to one address while requiring different TLS identities, authorization, tenancy, proxies, or protocol policy. HTTP connection coalescing is valid only when the protocol, certificate, origin, and client implementation permit it.

### Pool capacity from concurrency

For admitted request rate $\lambda$ and mean time $W$ occupying an upstream stream, Little's Law gives mean active streams:

$$
L_{\mathrm{streams}} = \lambda W
$$

If one healthy connection safely supports $m$ concurrent streams at the target SLO, a first estimate is:

$$
C_{\mathrm{busy}} \ge \left\lceil \frac{\lambda W}{m} \right\rceil
$$

This is an average consistency check. Size for burst distribution, head-of-line effects, connection failure, endpoint skew, and the measured latency knee. A large pool is not free headroom: it moves the queue into the server and can multiply a downstream incident across every caller.

Fleet-wide connection count matters more than one process's default. For $I$ caller instances and per-dependency pool size $p_d$:

$$
C_{\mathrm{outbound,max}} = I \sum_d p_d
$$

**Illustrative:** 8,000 instances each allowed 40 connections to each of six dependencies can create `1.92 million` outbound connections. If one backend has a 120,000-connection safe envelope, a locally reasonable default is globally impossible. Budget pools per dependency and topology; use multiplexing, client subsetting, proxies, or admission control deliberately.

If $C$ active connections have maximum age $A$, steady replacement alone creates approximately:

$$
\lambda_{\mathrm{new}} \ge \frac{C}{A}
$$

At `C = 1.92 million` and `A = 10 minutes`, planned aging creates about 3,200 new connections/s before scale events or failures. Add jitter to age deadlines so a rollout or process start does not synchronize handshakes. [Network Transport Internals](./14-network-transport-internals.md) covers handshake, port, NAT, and congestion ceilings.

### Idle timeout, maximum age, and liveness

These controls solve different problems:

- **Idle timeout** reclaims an unused connection. Set the client to retire a connection before an intermediary/server is known to discard it, while accepting that distributed timers still race.
- **Maximum age** bounds how long address, route, credential, and endpoint-generation decisions remain pinned even under continuous use.
- **Keepalive/probe** discovers some dead paths; it is not an application-health proof and consumes capacity.
- **Request deadline** bounds one operation, not the pool wait unless the wait shares the same cancellation budget.
- **Drain deadline** bounds graceful retirement; after it, the system explicitly resets, migrates, or abandons remaining sessions.

Validating a connection only before borrowing it cannot prove that it remains alive for the next write. Protocols must handle close/reset at every operation boundary and distinguish safe replay from ambiguous completion.

## DNS and endpoint cutover protocol

Treat a DNS change as a rollout with compatibility and rollback state.

~~~mermaid
stateDiagram-v2
    [*] --> Inventory
    Inventory --> TargetReady: resolver/pool behavior measured
    TargetReady --> TTLPrepared: target capacity and identity verified
    TTLPrepared --> DualPublished: old cache horizon elapsed
    DualPublished --> NewPreferred: new-path probes and load healthy
    NewPreferred --> OldDraining: new connections no longer choose old
    OldDraining --> OldRemoved: connection/drain horizon elapsed
    OldRemoved --> TTLNormalized: rollback observation window elapsed
    TTLNormalized --> [*]

    TTLPrepared --> RolledBack: target unhealthy
    DualPublished --> RolledBack: errors or capacity regression
    NewPreferred --> RolledBack: old remains compatible
~~~

1. **Inventory:** enumerate authoritative providers and replicas, delegation, positive/negative TTLs, stale policy, resolvers, runtime caches, proxies, pool lifetime, long-lived sessions, and firewall/TLS dependencies.
2. **Prepare target:** provision data, capacity, certificates, protocol support, observability, and independent health probes. Test using the target address while sending the real hostname/SNI.
3. **Prepare TTL:** if faster convergence is required, lower the relevant RRset/alias-chain TTLs and wait the previous full cache horizon. Do not casually lower delegation TTLs during a nameserver move without parent coordination.
4. **Dual publish or shift gradually:** add the new target or use the provider's documented routing control. Preserve the old target and bidirectional data/schema compatibility.
5. **Verify by population:** query every authoritative server and representative public, enterprise, VPC, node-local, and application resolvers. Measure connections and goodput at both targets; `dig` from one laptop is not coverage.
6. **Stop new old-path work:** remove/deweight the old target after the new path is healthy. Mark the endpoint draining in discovery/LB layers as well; DNS alone cannot notify existing clients.
7. **Drain connections:** send protocol-appropriate retirement signals, stop accepting new work, bound long-lived streams, and wait a declared maximum-age/drain interval before forced close.
8. **Remove and normalize:** delete the old target only after residual DNS queries and old-address connections meet the policy. Restore ordinary TTLs after rollback no longer requires rapid steering.

If the last old answer can remain available for $D_{\mathrm{DNS}}$, and a connection created from it can live for $A_{\mathrm{conn}}$, a conservative cutover horizon is:

$$
D_{\mathrm{cutover}} \ge D_{\mathrm{DNS}} + A_{\mathrm{conn}} + D_{\mathrm{drain\text{-}observation}}
$$

The terms can overlap for many clients, but the sum covers the client that obtains the old address at the end of the DNS horizon and then creates a maximum-age connection. If `A_conn` is unbounded, DNS cannot provide a finite traffic-removal SLO.

Rollback is asymmetric after old data, certificates, routes, or capacity are removed. Keep the old path write-compatible and provisioned until the rollback horizon closes; “put the record back” is not a rollback plan if resolvers cache both directions and clients retain new connections.

## Specialized failure traces

### DNS changed; old endpoint still receives writes

1. The operator replaces address A with B at TTL 60 seconds.
2. All measured resolvers return B after two minutes.
3. A client keeps an HTTP/2 connection to A for six hours and continues writing.
4. A is repurposed or its data stops replicating.

Bound connection age, drain A explicitly, and fence old writes before repurposing. Resolver convergence proves only the source for future connections.

### A new record is born into negative cache

1. A deployment probes `new.api.example` before publication and receives NXDOMAIN with a 30-minute negative TTL.
2. The record is created and authoritative probes succeed.
3. The deployment's recursive resolver continues returning cached NXDOMAIN.

Publish before consumers query, use an existing indirection name, or wait/flush the known negative cache. Never retry NXDOMAIN at request rate; that does not bypass a shared recursive cache.

### Resolver outage becomes a thread and retry outage

1. A resolver becomes slow rather than failing immediately.
2. Request handlers perform synchronous resolution with a long OS timeout.
3. Worker threads and connection slots fill; callers retry, multiplying lookups.
4. Healthy established connections are abandoned by aggressive retry/recreate logic.

Use bounded asynchronous resolution, request coalescing, valid cached/last-known-good policy, redundant resolver paths, and a retry budget. Preserve healthy pooled connections during a publication-plane incident.

### Low TTL creates synchronized dependency load

1. A popular RRset uses TTL 5 seconds for “fast failover.”
2. Recursive/node caches expire together after restart or cache flush.
3. Authoritative QPS, TCP fallbacks, and client connection attempts spike.
4. Slow answers trigger more retries and the intended failover mechanism collapses.

Choose TTL from measured convergence and authority capacity, distribute/jitter client refresh where the API permits it, and test cold-cache bursts. Fast steering that requires seconds belongs in a health-aware routing/discovery layer designed for it.

### DNS and network share one fate

Meta's October 2021 incident report describes backbone configuration changes that disconnected data centers. DNS servers then could not reach healthy data centers and withdrew their BGP advertisements, making authoritative DNS unreachable from the Internet and complicating recovery.

The lesson is structural: redundant DNS software is not independent if nameservers, control access, BGP advertisements, credentials, or management tooling depend on the same failing network. Draw and test the recovery path from outside the normal control plane.

### One address family black-holes

1. Both `AAAA` and `A` are valid in DNS.
2. A client's IPv6 route silently drops SYNs while IPv4 works.
3. A serial connector waits for IPv6 timeout before trying IPv4 on every request burst.

Use a tested Happy Eyeballs implementation, retain per-path telemetry, and repair IPv6 rather than permanently deleting it based on one client population.

### Graceful shutdown leaves long-lived streams

1. Readiness fails and ordinary requests drain.
2. WebSocket, gRPC, database, or subscription streams remain for hours.
3. The orchestrator reaches its grace deadline and kills the process, causing a reconnect storm.

Define an application-level reconnect/migration signal, stagger drain, budget destination capacity and authentication rate for reconnect, and impose a maximum stream lifetime where the product allows it.

### Partial delegation or DNSSEC transition

1. Some parent/authoritative servers publish the new nameserver or key state while others publish the old state.
2. Resolver cache histories select different validation chains.
3. Only validating clients in particular paths receive `SERVFAIL`.

Use the RFC-prescribed overlap/rollover sequence, query every authoritative server and validating vantage point, and stop the rollout on inconsistent chains. Disabling validation in clients hides the safety failure and creates a different incident.

## Security and abuse resistance

DNSSEC authenticates DNS data origin and denial-of-existence through a chain of trust; it does not encrypt names, prove endpoint health, or replace TLS/application authentication. A validating failure must remain distinguishable from NXDOMAIN and ordinary timeout. Protect key rollovers, registrar accounts, delegation changes, and emergency access with separation of duties, strong authentication, audit logs, and out-of-band recovery.

Additional controls:

- Use resolvers with source-port/query-ID randomization and current spoofing defenses; restrict unauthorized recursion and zone transfer.
- Decide whether DoT/DoH is required, prohibited, or intercepted in managed networks; otherwise clients can silently bypass split-horizon and monitoring policy.
- Treat DNS answers as untrusted input for SSRF-sensitive clients. Validate the resolved address class at connect time, pin the validated address for that attempt, reapply policy after redirects and re-resolution, and enforce network egress controls. A hostname allowlist alone is vulnerable to rebinding.
- Use fully qualified internal names and controlled search lists. Search suffixes and high `ndots` settings can leak partial names, multiply queries, and turn a short typo into several external lookups.
- Keep public and private views intentionally consistent for shared names. Split-horizon mistakes can send internal credentials to public endpoints or make incident probes observe a different system from users.
- Bind TLS identity and application authorization to the intended hostname/service, not merely to an IP returned by DNS.
- Rate-limit change APIs, require idempotent revisions, and make emergency override explicit. A compromised DNS account is a global traffic and certificate-validation capability.

## Observability and operational evidence

Observe each layer separately and correlate by name, RR type, resolver class, address, endpoint generation, and change revision.

**Authoritative DNS:**

- RRset/config revision, SOA serial, replica rollout lag, and audit actor;
- query rate, response code, latency, truncation, TCP/UDP, DNSSEC size/signing/validation signals;
- delegation and glue consistency from external vantage points;
- routing-policy and health-check state where provider-specific features are used.

**Recursive/client resolution:**

- fresh hit, negative hit, stale hit, miss, refresh, timeout, and validation failure;
- end-to-end resolution latency by resolver and runtime version;
- returned TTL/answer set, cache age, and alias-chain result where exposed;
- `A`/`AAAA` candidate order and winning family/path;
- request-coalescing, retry, and resolver queue depth.

**Connections and endpoints:**

- pool size, active/idle/draining state, queued waiters, wait latency, and rejection;
- connection creation rate, reuse ratio, handshake result/latency, maximum-age and idle retirement;
- remote address and resolution/endpoint generation, connection-age distribution, and residual old-target traffic;
- active streams, `GOAWAY`/close/reset reasons, reconnect storm rate, and ambiguous operation outcomes;
- endpoint goodput, saturation, and certificate/protocol negotiation by old/new target.

Fleet aggregates can hide the exact population still using an old answer. Cutover dashboards need a denominator: percentage of known resolver populations converged, percentage of new connections on the new generation, and count/age of remaining old-generation connections.

## Verification strategy

### Deterministic component tests

Use a controllable clock and fake resolver/authority to test:

- positive TTL decrement and expiry, negative NXDOMAIN/NODATA caching, and no fabricated negative cache for transient failure;
- bounded serve-stale and refresh coalescing;
- CNAME/SVCB chains with different TTLs, unsupported mandatory SVCB keys, truncated responses, and TCP retry;
- address-list changes, partial `A`/`AAAA` failures, cancellation, and deadline propagation;
- pool reuse, maximum age, idle race, `GOAWAY`, drain, reconnect, and ambiguous write handling;
- cache and pool generation behavior across backward/forward clock changes without trusting wall time for ordering.

### Integration and compatibility tests

Run the supported runtime/library versions against the actual recursive resolver stack. Verify whether APIs expose TTL, how long positive and negative results are cached, whether search lists expand queries, and how proxy/mesh layers resolve. Test DNSSEC validation, parent delegation, certificate/SNI, IPv4/IPv6, EDNS sizing, UDP loss, TCP/53, DoT/DoH policy, and split-horizon views.

### Cutover and failure drills

At production-like load:

1. lower then restore TTL while measuring authoritative and recursive load;
2. publish a dual target and prove new connections move while existing connections remain bounded;
3. serve stale during authoritative loss, then retire the old endpoint safely;
4. inject NXDOMAIN, NODATA, `SERVFAIL`, validation failure, slow resolution, packet loss, and one-family black holes;
5. drain endpoints with maximum-length requests/streams and measure reconnect/authentication capacity;
6. fail a resolver site, authoritative provider, DNS control account, parent delegation path, and management network independently;
7. restore an old zone/config snapshot and prove revision checks prevent accidental regression.

The rollback drill is complete only when traffic, connections, identity, and data all return—not when one resolver shows the old address again.

## Decision framework

1. Which names are human-stable identities, coarse traffic controls, discovery records, or security/validation records?
2. Where are the parent, authoritative, recursive, node, runtime, proxy, and application caches, and what are their measured positive, negative, and stale bounds?
3. Which clients implement `SRV`, SVCB/HTTPS, DNSSEC, and Happy Eyeballs semantics actually being published?
4. What resolver population and cold-cache burst must the authoritative service sustain at the chosen TTL?
5. Which established connections can outlive DNS freshness, and what maximum-age or explicit drain bounds them?
6. What is the fleet-wide connection, handshake, port/NAT, and backend concurrency budget?
7. How are new endpoints verified for protocol, certificate, data, and load before publication?
8. What measured evidence permits stopping old connections and removing the old endpoint?
9. Are DNS, routing, control access, credentials, and recovery tooling independent enough to survive the named failure?
10. Can operators roll forward and backward across delegation, DNSSEC, DNS answers, connections, and data compatibility without an unverifiable interval?

## Primary references

- [RFC 1034, *Domain Names—Concepts and Facilities*](https://www.rfc-editor.org/rfc/rfc1034)
- [RFC 1035, *Domain Names—Implementation and Specification*](https://www.rfc-editor.org/rfc/rfc1035)
- [RFC 2308, *Negative Caching of DNS Queries*](https://www.rfc-editor.org/rfc/rfc2308)
- [RFC 4033, *DNS Security Introduction and Requirements*](https://www.rfc-editor.org/rfc/rfc4033)
- [RFC 5452, *Measures for Making DNS More Resilient against Forged Answers*](https://www.rfc-editor.org/rfc/rfc5452)
- [RFC 6891, *Extension Mechanisms for DNS (EDNS(0))*](https://www.rfc-editor.org/rfc/rfc6891)
- [RFC 7766, *DNS Transport over TCP—Implementation Requirements*](https://www.rfc-editor.org/rfc/rfc7766)
- [RFC 8305, *Happy Eyeballs Version 2*](https://www.rfc-editor.org/rfc/rfc8305)
- [RFC 8767, *Serving Stale Data to Improve DNS Resiliency*](https://www.rfc-editor.org/rfc/rfc8767)
- [RFC 9460, *Service Binding and Parameter Specification via the DNS*](https://www.rfc-editor.org/rfc/rfc9460)
- [Google SRE, *Load Balancing at the Frontend*](https://sre.google/sre-book/load-balancing-frontend/)
- [Meta Engineering, *More details about the October 4 outage* (October 2021)](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/)
