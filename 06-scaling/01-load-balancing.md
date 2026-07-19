# Load Balancing

Load balancing is the controlled assignment of work to eligible capacity. The difficult part is not choosing between round robin and least connections. It is keeping a fast, distributed data plane aligned with changing backend authority, then preventing health reactions, retries, drains, and long-lived connections from turning a small fault into a fleet-wide overload.

This chapter owns the **L4 and L7 balancing data planes, endpoint eligibility, health and ejection, connection skew, and the controller state that programs routing**. [Partitioning Strategies](../02-distributed-databases/05-partitioning-strategies.md) owns placement algorithms for durable key spaces. [DNS and Connection Management](./13-dns-and-connection-management.md) owns discovery, resolver caching, and connection-pool lifetimes. [Multi-Region Architecture](./09-multi-region-architecture.md) owns regional authority and failover policy.

## Primary Evidence and Scope

| Primary evidence | What it establishes | Boundary |
|---|---|---|
| Google, *Maglev* (NSDI 2016) | ECMP distributes traffic to a software L4 fleet; each instance uses a consistent lookup table and connection tracking | A historical Google design, not a universal topology or present-day capacity claim |
| Meta Engineering, Katran post (May 2018) | An XDP/eBPF L4 forwarding plane, ECMP ingress, Maglev-style hashing, direct-server-return constraints, and operational instrumentation | Describes the open-sourced 2018 design and its stated deployment context |
| Google SRE book, load balancing chapters (2016) | Backend subsetting, partial/stale client knowledge, lame-duck draining, and the risk of distributing overload | Operational experience, not a configuration prescription |
| Envoy architecture documentation (accessed July 2026) | Current documented policies include weighted round robin, power-of-two least request, ring hash, Maglev, active health checks, passive outlier ejection, and circuit-breaker limits | Product semantics can change; pin and test the deployed release |
| Kubernetes Service and EndpointSlice documentation (accessed July 2026) | A control plane maintains changing endpoint sets; readiness and termination are explicit endpoint conditions | One concrete service-discovery API, not the only controller model |

These sources show mechanisms. They do not justify copying another operator's thresholds, endpoint counts, or health intervals.

## Workload and routing contract

Define the unit being assigned before selecting an algorithm. An L4 balancer normally assigns a transport flow, while an L7 balancer can assign an HTTP request or RPC stream. A TCP connection may carry one request, thousands of sequential requests, or many concurrent HTTP/2 streams; equality of connections therefore does not imply equality of work.

Record at least:

| Contract field | Questions that must be answered |
|---|---|
| **Protocol unit** | Packet, 5-tuple flow, QUIC connection ID, HTTP request, RPC stream, or WebSocket? |
| **Demand shape** | Requests/s, new connections/s, concurrent connections, packets/s, bits/s, request-cost distribution, and burst envelope? |
| **Affinity** | Must one session, cache key, tenant, or ordered stream stay on one backend? For how long? |
| **Eligibility** | Which readiness, locality, version, authorization, and capacity conditions admit a backend? |
| **Failure behavior** | When may the balancer retry, fail over, degrade, or reject? Which operations are safe to replay? |
| **Change budget** | How much connection churn, cache remapping, and traffic shift may one rollout or failure cause? |
| **SLO boundary** | Does the latency SLO include proxy queueing, TLS, retries, and upstream time? It should. |

**Reference design:** make the admission contract explicit: “accept only while the selected priority has a bounded request and connection budget; otherwise return a classified overload result.” A balancer redistributes admitted work. It cannot manufacture backend capacity.

## State, authority, and invariants

The controller's authoritative state is more than a list of IP addresses:

- service or virtual-IP identity and listener protocol;
- endpoint identity, locality, weight, version, and capacity class;
- endpoint condition such as `WARMING`, `READY`, `DRAINING`, `DEGRADED`, or `EJECTED`;
- routing policy and affinity-key schema;
- active-check evidence and passive-ejection state;
- per-priority connection, pending-request, retry, and ejection budgets;
- configuration epoch, rollout status, and last known safe version.

The data plane holds a compiled, cached projection of that state plus local observations: connection tables, request counters, latency samples, passive errors, and queue occupancy. Those observations are useful but are not durable service authority.

**Reference-design invariants:**

1. A new unit of work is sent only to an endpoint eligible under one complete configuration epoch.
2. All packets in a stateful L4 flow reach the same backend until the flow expires or an explicit recovery protocol takes over.
3. A draining endpoint receives no new assignment, but existing work gets its declared drain semantics.
4. Endpoint ejection never removes more capacity than the surviving admission budget can tolerate.
5. An old configuration cannot silently overwrite a newer epoch or resurrect a removed endpoint.
6. Retried work remains within a global attempt budget and preserves the operation's idempotency contract.
7. Loss of the controller does not erase the last known safe data-plane configuration.
8. Tenant or session affinity never bypasses endpoint eligibility or authorization.

## Data plane and control plane

~~~mermaid
flowchart LR
    C[Clients]
    R[Network routing<br/>anycast / ECMP]
    L4[L4 fleet<br/>flow selection]
    L7[L7 fleet<br/>TLS + request policy]
    B[Eligible backends]
    D[Discovery and health]
    P[Policy controller<br/>versioned desired state]
    O[Metrics and events]

    C --> R --> L4 --> L7 --> B
    D --> P
    P -->|epoch N snapshot / delta| L4
    P -->|epoch N snapshot / delta| L7
    L4 --> O
    L7 --> O
    B --> O
    O --> D
~~~

### L4: assign flows without application semantics

An L4 data plane selects using network-visible fields such as source/destination addresses, ports, protocol, and sometimes a QUIC connection ID. It may forward by destination NAT, tunnel/encapsulate to the backend, or use direct server return (DSR), where responses bypass the balancer.

**Documented:** Maglev's 2016 design used routers and ECMP to spread packets across balancer instances, then a consistent lookup table and connection tracking to preserve backend selection. Meta's 2018 Katran design similarly described ECMP ingress, XDP/eBPF forwarding, Maglev-style hashing, and DSR. The two publications demonstrate viable designs; they do not prove that XDP or DSR is necessary for every L4 workload.

The forwarding mode changes the contract:

- **NAT/full proxy:** symmetric traffic simplifies accounting and address translation but makes both directions consume balancer state and bandwidth.
- **Tunneling/DSR:** removes the return-path bottleneck but requires backend VIP handling, MTU discipline, and asymmetric-path observability.
- **Stateless deterministic forwarding:** reduces per-flow state, but every active balancer must compute a compatible result from compatible endpoint maps.
- **Connection tracking:** can preserve existing mappings through a table update, but its memory, expiry, replication, and failover semantics become capacity constraints.

L4 can prove that a TCP port accepts a connection. It cannot know whether a particular HTTP route is semantically healthy.

### L7: assign requests with application context

An L7 proxy terminates or parses the application protocol. It can route by host, path, method, header, tenant, RPC method, or authenticated identity; pool upstream connections; enforce request budgets; and distinguish local transport failures from application responses.

The added control costs CPU, memory, TLS key operations, parsing, buffering, and a larger security surface. L7 also changes the unit of fairness. One downstream HTTP/2 connection can multiplex requests to several upstream connections, so downstream connection count is no longer a useful backend-load proxy.

**Inference:** a common layered design uses a wide, inexpensive L4 tier to absorb flows and a smaller L7 tier to make request-aware decisions. Whether both tiers are needed depends on protocol and scale. Adding a proxy hop without a contract merely adds failure and queueing points.

### Publishing controller state

**Reference design:** publish an immutable snapshot or ordered delta stream with a monotonic epoch. A data-plane instance validates the full candidate, builds lookup structures off the hot path, then atomically swaps from epoch `N` to `N+1`. It reports acknowledgement and rejection reason. Keep the last safe snapshot for controller outages and rollback.

Partial configuration is dangerous. A listener from the new epoch combined with endpoints from the old epoch can route a request to a backend that does not implement the selected protocol. Validate referential integrity, endpoint uniqueness, weight bounds, certificate availability, and minimum viable capacity before activation.

Controller freshness must not be confused with endpoint liveness. If discovery stalls, a healthy last-known endpoint set may be safer than an empty one. If endpoint identities are security-revoked, fail-stale may be unacceptable. Specify the policy per failure class.

## Selection algorithms and their information cost

No algorithm balances “load” without defining what load means and when it is observed.

| Policy | Useful when | Hidden failure mode |
|---|---|---|
| Round robin / random | Interchangeable endpoints and narrow request-cost distribution | Equal assignment counts can produce unequal CPU or completion time |
| Weighted round robin | Capacity classes are measured and stable | Stale or aspirational weights overload a weaker class |
| Least active requests | Service times vary and the proxy sees request completion | Counts ignore remaining work; slow failures can herd traffic elsewhere |
| Power of two choices | Large pools need an O(1) approximation to least-loaded selection | Samples are only as good as local, possibly stale counters |
| Latency/load-aware | Backends expose trustworthy, timely cost signals | Feedback delay and correlated observations can oscillate or herd |
| Hash affinity | A cache/session benefit exceeds skew and remapping cost | One hot key pins overload to one endpoint; membership churn moves keys |
| Locality/priority | Cross-zone or cross-region cost and latency matter | A failed locality can transfer more load than the receiver can admit |

**Documented:** current Envoy documentation describes equal-weight least request as sampling candidate hosts and choosing the one with fewer active requests, and separately documents weighted, ring-hash, and Maglev policies. This establishes available mechanisms, not a default recommendation.

For uniform independent assignment of `m` requests to `n` endpoints, the expected count is `m/n`, but finite-sample variance remains. More importantly, production requests are not uniform. Measure per-route service demand and the hottest affinity key. The canonical treatment of stable hashing and movement is in [Partitioning Strategies](../02-distributed-databases/05-partitioning-strategies.md).

## Health, degradation, and ejection

Health is a claim about eligibility for a particular class of work:

- **Startup/readiness:** dependencies, configuration, and warm state are sufficient to accept new work.
- **Active check:** an independent probe reaches a representative path.
- **Passive observation:** real traffic produces local failures, application failures, latency, or resets.
- **Drain/lame duck:** the process is alive but intentionally closed to new work.
- **Degraded:** the endpoint can serve a reduced class or lower priority, not full traffic.

**Documented:** Envoy distinguishes active health checking from passive outlier detection and can classify locally originated transport errors separately from externally originated application errors. Kubernetes EndpointSlice exposes readiness and termination-related endpoint conditions. These distinctions are valuable because “port open,” “request failed,” and “safe for new work” are different facts.

**Reference design:** use a state machine with evidence and hysteresis, not a boolean derived from one probe. Require enough observations for the traffic volume, cap simultaneous ejection by locality, and reduce weight before complete ejection when uncertainty is high. A recovered endpoint enters slow start so cold caches and lazy connections do not receive an instantaneous full share.

Never let a shared dependency failure make every application instance look individually bad. The detector should preserve failure provenance. If all endpoints begin returning the same dependency error at once, ejecting them cannot create a healthy destination.

## Connections, draining, and skew

Balancing happens only when the data plane gets a choice. Adding a backend does not redistribute established TCP connections, WebSockets, database sessions, or QUIC connections. Removing an endpoint from new-flow selection likewise does not finish its existing work.

Let `λc` be new connections/s and `Wc` mean connection lifetime. Under the stationarity assumptions of Little's Law:

```text
mean concurrent connections Lc = λc × Wc
```

If `λc = 120,000/s` and `Wc = 120 s`, the fleet holds about `14.4 million` concurrent connections on average. Six perfectly balanced balancers would average `2.4 million` each; after losing one, the surviving mean is `2.88 million` before accounting for reconnection bursts or skew. Memory, file descriptors, conntrack, TLS state, and keepalive timers must all survive that envelope.

Connection age creates historical skew. Suppose four backends each hold 250,000 long-lived sockets and four new backends are added. New-flow balancing may be perfectly even while the old four retain almost all message traffic. Operations need connection-age histograms, work per connection, and a bounded rebalance policy. Forced disconnects can create a synchronized reconnect storm; randomized expiry or application-level session handoff is safer.

**Reference-design drain protocol:**

1. mark the endpoint `DRAINING` at a new epoch and stop new assignment;
2. advertise protocol-specific shutdown (`Connection: close`, HTTP/2 GOAWAY, or application signal) where applicable;
3. wait for in-flight requests, streams, and leases within a deadline;
4. close remaining work according to its retry/idempotency contract;
5. remove the endpoint only after both controller acknowledgement and data-plane drain telemetry agree.

QUIC connection IDs and address migration require a QUIC-aware routing contract; hashing only the UDP 4-tuple can move a migrated connection to the wrong backend. See [Network Transport Internals](./14-network-transport-internals.md).

## Capacity, headroom, and cost

Size each independent resource, then take the largest fleet requirement. A packet balancer can exhaust packets/s before bits/s; an L7 proxy can exhaust CPU, TLS operations, connection memory, or upstream pools before network bandwidth.

### Illustrative L4 envelope

Assume `80 Gbit/s` ingress with a measured mean wire size of `900 bytes`:

```text
packet_rate = 80e9 / (900 × 8) = 11.1 million packets/s
```

If fault-injection benchmarks show one node can sustain `4 million packets/s`, but the operational target is 60% to retain burst and interrupt headroom, useful planned capacity is `2.4 million packets/s/node`. Five nodes meet the normal mean only barely; six are required for one-node loss because `5 × 2.4 = 12 million packets/s`. Repeat the model for bytes/s, encapsulation overhead, both directions when symmetric, and the actual packet-size distribution.

### Illustrative L7 envelope

Assume `150,000 requests/s` and measured proxy CPU demand of `0.20 ms/request` at the real TLS and filter mix:

```text
CPU demand = 150,000 × 0.00020 = 30 CPU-seconds/second
useful CPU per 8-core node at 50% target = 4 CPU-seconds/second
base nodes = ceil(30 / 4) = 8
```

To survive one node while retaining that target requires `N` such that `(N - 1) × 4 >= 30`, so `N = 9`. A rollout that temporarily removes another node requires ten or a separately justified relaxation. This calculation is illustrative; benchmark request-size, cipher, logging, and route distributions instead of copying `0.20 ms`.

The cost model includes more than instance price:

```text
total cost = balancer compute + data processing/egress + control plane
           + cross-zone traffic + certificates/keys + observability
           + reserved failure and rollout headroom
```

High utilization is not automatically economical if it turns one failure into overload. Compare cost at the required SLO and named failure, not at the healthy-fleet average.

## Overload and specialized failure traces

### Correlated ejection collapses the pool

1. A shared database slows, so every backend begins timing out on one expensive route.
2. Passive detectors count those application failures as endpoint defects.
3. Instances eject different subsets; surviving endpoints receive more traffic.
4. Their queues grow, health probes time out, and the pool collapses despite no host-specific fault.

Preserve local-versus-application failure provenance, cap ejection, use route-aware degradation, and enforce [backpressure](./07-backpressure.md) and [circuit-breaker](./06-circuit-breakers.md) budgets.

### A stale epoch resurrects a drained endpoint

1. Epoch 81 marks backend `b7` draining for a schema-incompatible rollout.
2. One balancer misses the update and continues routing with epoch 80.
3. `b7` processes new requests after the migration boundary, creating mixed semantics.

Backends can reject assignments carrying an obsolete route epoch during a strict transition. Alert on data-plane epoch lag and never permit configuration downgrade.

### Scale-out does not move long-lived load

1. A messaging fleet doubles its backends during an event.
2. Nearly all clients keep existing WebSockets to the old half.
3. New-connection metrics look balanced; message CPU and outbound bytes remain concentrated.
4. Operators add more empty nodes while the old nodes fail.

Measure work per connection and connection age. Rebalance gradually with randomized reconnects or application-level handoff, bounded by authentication and reconnect capacity.

### Failover transfers overload

1. A locality with 40% of traffic is removed.
2. The controller immediately renormalizes all weight onto localities sized for their prior shares.
3. Retries arrive with the displaced original traffic and saturate the survivors.
4. Health reactions eject them in turn.

Failover is an admission decision: compute surviving capacity first, shed lower-priority work, and use a [retry budget](./10-retries-timeouts-hedging.md). A load balancer must be allowed to reject traffic that has no safe destination.

### Health checks are green while real requests fail

1. A TCP probe connects successfully.
2. The application route needs a broken credential or downstream dependency.
3. L4 keeps assigning flows because transport is healthy.

Use layered checks and real-request success telemetry. Do not make a deep probe so expensive or correlated that the probe itself causes failure.

## Operations, migration, and rollback

Introduce a new balancer or policy as a controlled traffic migration:

1. replay recorded metadata and property-test deterministic decisions offline;
2. shadow the new decision without forwarding and compare eligibility, locality, and remapping;
3. canary one failure domain with a small, reversible share;
4. ramp by connection starts as well as requests/s, because old flows remain on the old path;
5. exercise endpoint addition, drain, controller loss, rollback, and one-zone loss during the ramp;
6. retain old listeners, certificates, and route state until the maximum connection/drain horizon expires.

A policy rollback does not automatically restore old affinity. Quantify how many keys or sessions remap in both directions. Avoid changing the hash-key schema and backend membership in the same step.

Runbooks should cover frozen discovery, mass ejection, certificate failure, conntrack exhaustion, one-sided DSR loss, configuration rejection, reconnect storm, and overload with no eligible endpoint. Operators need an explicit “hold last safe config,” “freeze ejection,” and “shed priority” action with audit logs.

## Security boundaries

The balancer is often an authentication and trust boundary:

- terminate TLS only where private-key access, rotation, and audit are controlled;
- authenticate the balancer to upstreams when the network is not the identity boundary;
- strip untrusted forwarding and identity headers, then add canonical versions once;
- bind tenant or route metadata to authenticated context, not caller-supplied affinity alone;
- authorize controller writes and sign or authenticate configuration distribution;
- constrain admin, stats, and health endpoints separately from public listeners;
- rate-limit handshakes and expensive parsing before they exhaust the L7 tier;
- scrub secrets and personal data from access logs while preserving traceability.

DSR and preserved source addresses change firewall and return-path assumptions. Test spoofing, fragmented packets, malformed HTTP, request smuggling across parser differences, TLS-key compromise, and a malicious controller update. [API Security](../10-security/04-api-security.md) owns the broader application controls.

## Observability and verification

Separate control-plane convergence from data-plane outcomes.

| Plane | Required signals |
|---|---|
| Controller | desired and acknowledged epoch, update age, rejected configs, endpoint-state transitions, weight/ejection changes |
| L4 | packets and bytes by VIP/backend, new and active flows, flow-table occupancy, drops, fragments, encapsulation/return-path failures |
| L7 | requests, active/pending/retried work, route and backend latency, local versus upstream errors, TLS and pool saturation |
| Distribution | per-backend work share, CPU-normalized share, skew, hottest affinity key, locality share, remap rate |
| Lifecycle | readiness-to-first-traffic delay, slow-start progress, connection ages, drain duration, forced closes |
| User outcome | success, latency, and overload rejection by route, tenant, locality, and protocol |

Test at four layers:

- **Properties:** ineligible endpoints are never chosen; epoch never regresses; deterministic L4 implementations agree; weights converge within measured error.
- **Protocol:** half-close, reset, idle expiry, HTTP/2 GOAWAY, QUIC migration, malformed headers, and retry/idempotency behavior.
- **Skewed load:** variable service time, elephant connections, a hot affinity key, cold endpoints, and nonuniform clients.
- **Faults:** controller partition, delayed discovery, correlated application errors, one balancer/endpoint/zone loss, DSR return-path loss, and reconnect bursts.

Success is not “the proxy stayed up.” It is bounded admitted load, correct routing authority, predictable remapping, and preserved end-user SLOs through the declared failures.

## Decision framework

1. Is the assignable unit a flow or a request, and can one unit contain unbounded work?
2. Which application semantics justify L7 parsing, and which traffic can remain L4?
3. What state is authoritative for eligibility, and how is a stale data plane fenced?
4. Which signal represents remaining backend work rather than merely assignment count?
5. What affinity benefit is required, and what is the hottest key plus membership-remap cost?
6. How many endpoints may health logic eject while the fleet still survives its named failure?
7. How are established connections drained or rebalanced when membership changes?
8. Which resource—packets, bits, handshakes, CPU, memory, connections, or upstream concurrency—sets capacity?
9. Can controller loss, rollback, and a bad configuration preserve the last safe data plane?
10. When no safe backend exists, which traffic is rejected rather than redistributed into a cascade?

## Primary references

- [Eisenbud et al., *Maglev: A Fast and Reliable Software Network Load Balancer* (NSDI 2016)](https://research.google/pubs/maglev-a-fast-and-reliable-software-network-load-balancer/)
- [Meta Engineering, *Open-sourcing Katran, a scalable network load balancer* (May 2018)](https://engineering.fb.com/2018/05/22/open-source/open-sourcing-katran-a-scalable-network-load-balancer/)
- [Google SRE, *Load Balancing in the Datacenter* (2016)](https://sre.google/sre-book/load-balancing-datacenter/)
- [Google SRE, *Load Balancing at the Frontend* (2016)](https://sre.google/sre-book/load-balancing-frontend/)
- [Envoy, *Supported load balancers*](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/load_balancers)
- [Envoy, *Health checking*](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/health_checking)
- [Envoy, *Outlier detection*](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/outlier)
- [Envoy, *Circuit breaking*](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/circuit_breaking)
- [Kubernetes, *Service* and EndpointSlice behavior](https://kubernetes.io/docs/concepts/services-networking/service/)
- [Mitzenmacher, Richa, and Sitaraman, *The Power of Two Random Choices* (2001)](https://www.eecs.harvard.edu/~michaelm/postscripts/handbook2001.pdf)
- [RFC 9000, *QUIC: A UDP-Based Multiplexed and Secure Transport* (May 2021)](https://www.rfc-editor.org/rfc/rfc9000)
- [Little, *A Proof for the Queuing Formula: L = λW* (1961)](https://pubsonline.informs.org/doi/10.1287/opre.9.3.383)
