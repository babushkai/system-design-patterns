# Network Transport Internals

## TL;DR

Transport is not a transparent pipe. It is a set of state machines that spend round trips, limit bytes in flight, recover loss, authenticate peers, and compete for CPU and queue capacity. A useful design review therefore separates five questions:

1. **How many network flights precede useful work?** On an established encrypted connection, a small request and its first response byte usually cost about one path RTT plus server time. Under the assumptions below, new TCP plus TLS 1.3 costs about three RTTs, new QUIC about two, and accepted QUIC 0-RTT about one. QUIC 0-RTT is **resumption with prior state**, not cold first contact.
2. **How many bytes may be in flight?** Congestion control protects the path; transport flow control protects receiver buffers; HTTP and application limits protect different resources. The smallest limit wins.
3. **What does loss block?** HTTP/2 removes HTTP/1.1 response ordering across requests, but TCP still delivers one ordered byte stream. HTTP/3 uses independent QUIC streams, while congestion control and connection-level flow control remain shared.
4. **Which guarantees survive retries and path changes?** A transport acknowledgment does not prove an application commit. TLS early data can be replayed. QUIC migration requires path validation and routing continuity; it is optional and can be disabled.
5. **Where is the bottleneck?** On Linux, packet rate, softirq work, socket queues, encryption, copies, and application scheduling can each dominate. Kernel bypass is a measured response to a localized packet-path bottleneck, not a default architecture.

Transport analysis covers handshake flights, congestion and flow control, loss recovery, PMTU, protocol behavior, and host packet capacity. [DNS and Connection Management](./13-dns-and-connection-management.md) covers freshness, pools, and draining; [Backpressure](./07-backpressure.md) covers resource bounds; [Retries, Timeouts, and Hedging](./10-retries-timeouts-hedging.md) and [Idempotency](../01-foundations/08-idempotency.md) cover ambiguous outcomes.

---

## Scope, Terms, and Assumptions

Use the following models only within their stated boundaries.

| Symbol | Meaning | Unit and boundary |
|---|---|---|
| $R_0$ | Unloaded path round-trip time | Seconds; propagation, transmission, and baseline processing, excluding application work |
| $D_q$ | Additional queueing delay observed during load | Seconds; may exist in hosts, NICs, switches, middleboxes, or the remote service |
| $R$ | Measured RTT for the modeled exchange | $R = R_0 + D_q$ |
| $S$ | Server time until the first response byte is ready | Seconds; includes the server queues chosen for the measurement |
| $B$ | Bottleneck bit rate available to this traffic | Bits per second, not interface line rate unless the flow can use all of it |
| $W$ | Effective bytes allowed in flight | Bytes; the minimum applicable congestion and flow-control limit |
| $P$ | Application payload remaining after first byte | Bytes |
| $G$ | Measured application goodput | Payload bytes per second after protocol overhead and contention |

Unless a subsection says otherwise, the latency ledger assumes:

- one client and one server path, no forward proxy, no packet loss, no reordering, and no handshake retry;
- a small request that fits in the first permitted application flight and a response whose first byte fits in one packet;
- TLS 1.3 as revised by RFC 9846, no TCP Fast Open, no DNS lookup, and no client-side pool wait;
- QUIC v1 with TLS integrated into the QUIC handshake;
- enough receive, stream, and application capacity that flow control does not delay the first byte;
- timestamps measured from the first transport packet sent by the client, not from a user action.

Real requests add resolver time, address racing, proxy hops, scheduler delay, certificate-chain serialization, packet loss, QUIC Retry, TLS HelloRetryRequest, request upload, response transfer, and application queueing. Treat the equations as a decomposition tool, not as an SLO prediction.

## Layered Data Path and Ownership

An end-to-end request crosses several control loops. Collapsing them into one word such as “network” makes the wrong team tune the wrong limit.

```text
caller
  │ admission, deadline, retry and idempotency policy
  ▼
HTTP
  │ request streams, header compression, stream concurrency
  ▼
TLS over TCP                    QUIC
  │ identity, records           │ TLS handshake + streams + loss recovery
  ▼                             ▼
TCP                             UDP
  │ ordered bytes, flow         │ datagrams and checksum
  │ control, congestion         │
  └──────────────┬──────────────┘
                 ▼
          IP routing and PMTU
                 ▼
      host queues → NIC → path → peer
```

| Concern | Primary owner in this book | Transport-facing contract |
|---|---|---|
| DNS caching, endpoint selection, Happy Eyeballs, pool reuse, maximum connection age, idle races, and drain | [DNS and Connection Management](./13-dns-and-connection-management.md) | Supplies a reachable endpoint and decides when a connection may be reused or replaced |
| Backend routing, affinity, and QUIC connection-ID routing | [Load Balancing](./01-load-balancing.md) | Keeps packets for a live connection on a backend that owns its state |
| Admission, bounded queues, consumer capacity, and overload shedding | [Backpressure](./07-backpressure.md) | Prevents application work from accumulating even when the transport remains writable |
| Deadlines, retries, hedges, and ambiguous outcomes | [Retries, Timeouts, and Hedging](./10-retries-timeouts-hedging.md) | Interprets transport errors without inventing an application result |
| Encryption policy, certificate lifecycle, and secret rotation | [Encryption](../10-security/06-encryption.md) | Provides identities, certificates, ticket keys, ECH keys, and rotation boundaries |
| Transport layer | Handshake flights, congestion and flow control, loss recovery, PMTU, protocol behavior, and host packet capacity | Exposes measured state and failure signals to the layers above |

Congestion control is not application admission control. It estimates a safe sending rate for a network path; it does not know whether a database is saturated, whether one tenant is unfair, or whether a request should have been rejected.

## Latency Budget: Count Flights, Then Add Work

### First-byte ledger

For the assumptions above, the approximate first-byte latency is:

| Connection state | Packet-flight interpretation | Approximate first response byte |
|---|---|---:|
| Reused TCP plus established TLS | Request crosses, server works, first byte returns | $R + S$ |
| New TCP plus full or 1-RTT resumed TLS 1.3 | TCP handshake, TLS handshake, request/response | $3R + S$ |
| New TCP plus accepted TLS 1.3 early data, without TCP Fast Open | TCP handshake, then early request in ClientHello and response | $2R + S$ |
| New QUIC 1-RTT connection | Combined QUIC/TLS handshake, then request/response | $2R + S$ |
| Resumed QUIC with accepted 0-RTT | Early request crosses with the first client flight, then response | $R + S$ |

The new TCP plus TLS path can be visualized as:

```text
time      client                                      server
0         SYN ---------------------------------------->
1/2 R                                                receive SYN
R         <------------------------------------ SYN + ACK
R         ACK + TLS ClientHello ----------------------->
2 R       <------------------- TLS ServerHello...Finished
2 R       TLS Finished + HTTP request ----------------->
3 R + S   <----------------------------- first response byte
```

The final TCP ACK can carry the TLS ClientHello, and the client's TLS Finished can carry application data, but propagation still leaves the three-RTT first-byte path shown above.

For QUIC 1-RTT, the transport and TLS handshakes share flights:

```text
time      client                                      server
0         QUIC Initial + TLS ClientHello ------------->
R         <------------ Initial/Handshake + TLS Finished
R         Handshake completion + HTTP request -------->
2 R + S   <----------------------------- first response byte
```

A server-sent QUIC Retry or a TLS HelloRetryRequest adds a flight. Loss can add packet-threshold recovery or a probe timeout. A large certificate flight can require several datagrams and be constrained by anti-amplification. These are observable branches, not constants to hide inside “QUIC is one RTT.”

### Worked example with explicit assumptions

Let $R = 80\text{ ms}$ and $S = 10\text{ ms}$, with cached DNS, no loss, no Retry, no upload time, and a one-packet first response. The model gives:

| State | Calculation | Modeled first byte |
|---|---:|---:|
| Reused encrypted connection | $80 + 10$ | $90\text{ ms}$ |
| New TCP plus TLS 1.3 | $3(80) + 10$ | $250\text{ ms}$ |
| New QUIC 1-RTT | $2(80) + 10$ | $170\text{ ms}$ |
| Accepted QUIC 0-RTT resumption | $80 + 10$ | $90\text{ ms}$ |

This example does **not** say QUIC always saves 80 ms. An established H2 connection has already paid its setup cost; an H3 attempt that loses UDP connectivity can be slower because fallback has to occur; and an early-data rejection requires the client to replay after handshake completion.

### Completion time

For a streamed response, a first approximation is:

$$
L_{\mathrm{complete}} \approx L_{\mathrm{first\ byte}} + \frac{P}{G}
$$

This assumes the server can stream immediately and that $G$ is measured for the same path, protocol, payload distribution, and concurrency. If the request body must arrive before work begins, add its upload time. If the response is flow-control blocked, application-limited, or shares a congested connection, derive $G$ from that state rather than interface speed.

## Bandwidth-Delay Product, Windows, and Capacity

### Bytes required in flight

For bottleneck rate $B$ in bits per second and unloaded RTT $R_0$ in seconds:

$$
\mathrm{BDP}_{\mathrm{bytes}} = \frac{B R_0}{8}
$$

A $1\text{ Gbit/s}$ path with $R_0 = 0.1\text{ s}$ has a BDP of $12.5\text{ MB}$ using decimal units. A single long-lived flow needs roughly that much usable in-flight capacity to fill the path. More buffering is not automatically better: excess queue occupancy adds $D_q$ and can create bufferbloat.

For TCP:

$$
W_{\mathrm{TCP}} = \min(\mathrm{cwnd}, \mathrm{rwnd})
$$

For one QUIC stream, the applicable ceiling also includes the remaining connection and stream credit:

$$
W_{\mathrm{QUIC\ stream}} =
\min(\mathrm{cwnd}_{\mathrm{connection}},
     \mathrm{credit}_{\mathrm{connection}},
     \mathrm{credit}_{\mathrm{stream}})
$$

Ignoring headers, loss, pacing granularity, and application stalls:

$$
G \leq \frac{W}{R}
$$

The equations identify different remedies:

- low congestion window: inspect startup, loss, ECN, congestion-control state, and whether the sender is application-limited;
- low receiver or QUIC flow-control window: inspect read rate and window-update policy at the peer;
- sufficient windows but low goodput: inspect pacing, packet loss, CPU, packet size, application production, and competing flows;
- full throughput with high RTT: inspect queue occupancy and AQM rather than increasing buffers.

### Initial window and idealized startup

RFC 6928 permits, but does not require, a TCP sender to use this upper bound for its initial congestion window:

$$
\mathrm{IW}_{\max} =
\min(10\,\mathrm{SMSS},
     \max(2\,\mathrm{SMSS}, 14600\text{ bytes}))
$$

An implementation or route may begin lower. If a lossless, ACK-clocked slow start doubles the usable congestion window once per RTT and the application always has data, the approximate number of growth rounds needed to reach a target window is:

$$
k =
\max\left(0,
\left\lceil
\log_2\left(\frac{\mathrm{target\ window}}{\mathrm{IW}}\right)
\right\rceil\right)
$$

Using $12.5\text{ MB}$ and $14.6\text{ kB}$ gives about ten ideal growth rounds. This is a model of one startup phase, not a promise: ACK ratios, pacing, application gaps, receiver limits, HyStart-style exits, loss, and idle restart change the trajectory. RFC 5681 also allows reducing the congestion window after a sufficiently long idle period, so “socket still open” does not prove “startup already paid.”

### Scope of the Mathis model

For a long-lived Reno-like flow in congestion avoidance, independent random loss probability $p$, sufficient receive window, and no timeout-dominated recovery, the Mathis model is:

$$
G \approx \frac{C \cdot \mathrm{MSS}}{R\sqrt{p}}
$$

$C$ depends on acknowledgment and implementation behavior. If $C=1$, $\mathrm{MSS}=1460$ bytes, $R=80$ ms, and $p=0.001$, the illustrative result is about $0.58$ MB/s, or $4.6$ Mbit/s. It is a useful sensitivity check for Reno-style behavior; it is **not** a universal TCP ceiling and should not be used to predict CUBIC, BBR, short flows, burst loss, or timeout-heavy paths.

### Packet-rate and queue models

Bit rate alone hides small-packet cost. For mean on-wire packet size $L_{\mathrm{wire}}$ bytes:

$$
\mathrm{pps} = \frac{B}{8L_{\mathrm{wire}}}
$$

If a measured host path consumes $c_{\mathrm{packet}}$ CPU cycles per packet, the lower-bound core budget at frequency $f$ and target utilization $u$ is:

$$
\mathrm{cores} \geq
\frac{\mathrm{pps}\,c_{\mathrm{packet}}}{f u}
$$

For an explicitly illustrative model, $10$ Gbit/s, $1000$ mean on-wire bytes, $2000$ measured cycles per packet, a $3$ GHz CPU, and a $65\%$ utilization target imply:

$$
\mathrm{pps}=1.25\text{ Mpps}, \qquad
\mathrm{cores}\geq
\frac{1.25\times10^6 \times 2000}
     {3\times10^9 \times 0.65}
\approx 1.28
$$

That is not a hardware benchmark. The cycles must be measured for the actual direction, encryption, copy path, offload settings, NUMA placement, packet-size distribution, and application work. A design should round up and preserve failure and burst headroom.

Similarly, a queue holding $Q$ bytes behind a drain rate of $B/8$ bytes per second contributes at least:

$$
D_q \approx \frac{8Q}{B}
$$

This simple drain-time relationship explains why a “healthy, full” queue can violate latency objectives without dropping packets.

## TCP Connection and Recovery State

### Establishment and close

TCP's three-way handshake synchronizes sequence-number state and demonstrates a bidirectional path so delayed segments from an older incarnation are not mistaken for the new connection.

```text
client: CLOSED → SYN-SENT → ESTABLISHED
server: LISTEN → SYN-RECEIVED → ESTABLISHED
```

SYN backlog pressure is a consequence of the server's half-open state. SYN cookies are one overload and attack mitigation: they encode enough state in the SYN-ACK sequence number to postpone allocation. They are not the reason the handshake exists, and their feature tradeoffs depend on implementation.

TCP Fast Open can carry application data during connection establishment, but it changes replay and middlebox assumptions and is outside this baseline. Validate it on the real client, server, and network population before counting a saved flight.

Close is also stateful:

- `FIN` closes one direction after earlier bytes; TCP can be half-closed.
- `RST` aborts the byte stream and can discard unread data.
- `TIME_WAIT` prevents delayed old segments from colliding with a reused tuple and permits retransmission of the final ACK. Which endpoint accumulates it depends on close behavior.

Port budgets, maximum age, graceful HTTP draining, and who initiates close are connection-lifecycle policy and belong in [DNS and Connection Management](./13-dns-and-connection-management.md).

### Three different meanings of “sent”

For a mutating RPC, keep these boundaries separate:

1. the application copied bytes into a local socket;
2. the peer transport acknowledged those bytes;
3. the peer application committed the operation and durably recorded the response or idempotency result.

A timeout or reset between any two boundaries creates an ambiguous application outcome. TCP reliability proves ordered delivery to the peer transport, not transaction commit. The retry layer must use an operation identifier and a durable deduplication contract where duplicate effects are unacceptable.

### Nagle, delayed acknowledgments, and small writes

Nagle's algorithm withholds some small sends while earlier data remains unacknowledged, reducing tiny-packet overhead. A receiver may delay an ACK to coalesce work or acknowledge multiple full-sized segments. RFC 5681 says a delayed ACK should be generated for at least every second full-sized segment and delayed by less than $500$ ms; it does not establish a universal $40$ ms timer.

Their interaction can add a timer-sized pause to request patterns that emit several small dependent writes. For latency-sensitive framed RPC:

- batch a complete application frame where possible;
- test `TCP_NODELAY` rather than assuming every library sets it;
- compare payload and packet-size distributions before and after;
- retain buffering when throughput and packet efficiency matter more than per-message latency.

`TCP_NODELAY` is a workload choice, not a blanket correction for every TCP application.

### Loss-recovery ladder

TCP implementations can recover loss through several evidence paths:

1. ACK and SACK information can trigger fast retransmission and recovery when later data proves a gap.
2. RACK uses transmission time and acknowledgment of later packets to infer loss under reordering more robustly than duplicate-ACK counting alone.
3. Tail Loss Probe sends a probe near the tail when there may be no later data to generate loss evidence.
4. If evidence remains insufficient, the retransmission timeout uses smoothed RTT and RTT variance and backs off after expiration.

RFC 6298 specifies a one-second initial RTO and a one-second lower bound for its standards algorithm. A particular Linux release may use a different effective minimum, often lower, but that value is not a portable protocol guarantee. Do not diagnose a latency shelf as “the 200 ms TCP RTO” from shape alone; correlate it with packet traces, `TCP_INFO`, and kernel-version behavior.

QUIC does not copy TCP's RTO state machine. RFC 9002 combines packet-number and time-threshold loss detection with a Probe Timeout (PTO). A PTO sends ack-eliciting probes to make progress; its expiration does not by itself mean every outstanding packet is declared lost.

## Congestion Control and Queue Behavior

### Loss-based control, AQM, and ECN

CUBIC, standardized in RFC 9438, grows its congestion window as a cubic function around the previous congestion point. It is commonly configured as the Linux default, but distribution, kernel, route, and per-socket settings must be checked rather than assumed. Reno-compatible behavior remains relevant for fallback and fairness.

Loss-based senders usually discover a bottleneck by increasing in-flight data until they observe congestion. Deep unmanaged buffers can convert the signal from loss into queueing delay: throughput looks high while interactive RTT rises. Active Queue Management such as FQ-CoDel attempts to bound queue sojourn and separate flows. Explicit Congestion Notification marks congestion instead of forcing a drop when every relevant endpoint and device supports it.

In datacenters, synchronized many-to-one responses can fill a shallow switch queue faster than end hosts react. ECN-based controls such as DCTCP, pacing, fan-out limits, and response desynchronization address different parts of that incast problem. Host transport tuning cannot recover packets discarded by an already-overrun switch queue.

### What BBR does and does not claim

BBR models a path using delivery-rate and RTT observations and attempts to operate near an estimated bandwidth-delay product. Current BBRv3 work also responds to packet loss; describing BBR as “ignoring loss” is incorrect. The algorithm deliberately probes, so queue, loss, and coexistence behavior must be measured with the actual bottleneck.

As reviewed in July 2026, `draft-ietf-ccwg-bbr-06` is an active Internet-Draft intended for Experimental publication, not an RFC. Its ECN response is not yet fully specified. The public Linux code selected by a generic `bbr` congestion-control name must not be assumed to implement that draft's BBRv3 behavior; upstream and vendor kernels can expose different generations and patches.

A defensible CUBIC-versus-BBR experiment records:

| Dimension | Required comparison |
|---|---|
| Path | RTT range, loss and reorder pattern, bottleneck rate, policers, AQM, and ECN mode |
| Workload | Flow duration, object-size distribution, connection reuse, concurrent flow count, and application-limited fraction |
| Efficiency | Goodput, completion time by object size, retransmission/loss, and CPU |
| Queue health | Unloaded RTT versus RTT under load, queue drops/marks, and p95/p99 delay |
| Coexistence | Same-algorithm and mixed CUBIC/BBR fairness, including different RTT populations |
| Deployment | Exact kernel, module, draft/implementation generation, pacing qdisc, and rollback setting |

Choose from the result for a named path population. “High RTT means BBR” is a hypothesis, not a decision rule.

## TLS 1.3, Resumption, and Early Data

### Current protocol baseline

RFC 9846 is the current TLS 1.3 specification and obsoletes RFC 8446 while retaining TLS 1.3's basic architecture. A full handshake authenticates the server, negotiates keys, and normally completes in one network round trip. Most handshake messages from ServerHello onward are encrypted; the initial ClientHello remains observable unless ECH is successfully used.

Encrypted ClientHello is split across RFC 9848, which defines configuration discovery using DNS SVCB/HTTPS records, and RFC 9849, which defines ECH. ECH protects a ClientHelloInner containing sensitive values such as the real server name, while an outer ClientHello remains visible. It requires compatible client, resolver/config distribution, frontend, and origin behavior. Fallback and retry behavior are part of the privacy threat model; “TLS 1.3 hides SNI” is false without successful ECH.

### Full, resumed, and 0-RTT paths

| TLS mode | Prior state | Application data timing | Forward-secrecy nuance |
|---|---|---|---|
| Full TLS 1.3 | None | After one-RTT handshake | Ephemeral (EC)DHE provides forward secrecy when used as specified |
| PSK resumption with (EC)DHE | Ticket/PSK | After one-RTT resumed handshake | Fresh asymmetric exchange protects handshake/application traffic against later PSK disclosure |
| PSK-only resumption | Ticket/PSK | After one-RTT resumed handshake | Does not provide forward secrecy for that resumed connection |
| 0-RTT early data | Ticket/PSK plus remembered configuration | Sent with ClientHello | Early data lacks full forward-secrecy and replay guarantees; later 1-RTT traffic can still use PSK plus (EC)DHE |

Resumption normally reduces authentication work and handshake bytes; it does not automatically remove a network flight. Zero-RTT is an application profile layered on resumption. A first-time client has no ticket and cannot use it.

### Replay-safe early-data protocol

Idempotent syntax is not enough. A `GET` can increment a counter, consume a one-time token, expose freshness-sensitive data, or trigger billing. Cross-region frontends may also have anti-replay stores that do not share state quickly enough.

A safe deployment specifies:

1. **Eligibility:** an explicit route and method allowlist based on application semantics, with early data disabled by default.
2. **Binding:** authenticate the ticket's tenant, frontend, protocol, age, and configuration; limit ticket lifetime and early-data bytes.
3. **Replay control:** use the strongest anti-replay mechanism the availability design permits, and document its regional consistency and fail-open/fail-closed behavior.
4. **Effect deduplication:** carry a durable operation key when a repeated request must converge to one business result.
5. **Rejection:** if early data is not accepted, process it only after handshake confirmation or return HTTP `425 Too Early` where RFC 8470 applies.
6. **Client behavior:** retry rejected early data after the handshake under the normal deadline and retry budget; never report success merely because bytes were sent.
7. **Telemetry:** distinguish attempted, accepted, rejected, replay-blocked, retried, and duplicate-effect outcomes.

Session-ticket encryption keys and ECH keys need versioned rotation, bounded retention, and regional blast-radius decisions. A leaked ticket key can expose or mint resumable state within its scope. It does not retroactively defeat every full TLS 1.3 session: the consequence depends on PSK-only versus PSK-plus-(EC)DHE and whether the data was 0-RTT.

## HTTP/1.1, HTTP/2, and HTTP/3

The three versions change where concurrency and ordering live; none removes shared resource limits.

| Property | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---|---|---|---|
| Transport | Usually one ordered TCP byte stream per connection | One ordered TCP byte stream carrying many HTTP streams | QUIC connection carrying many QUIC streams |
| Request concurrency | Pipelining is permitted, but responses remain ordered; clients often use multiple connections instead | Multiplexed frames across streams | Multiplexed streams without TCP connection-wide delivery ordering |
| Header compression | No shared dynamic compression in the base protocol | HPACK | QPACK |
| Loss effect | Delays the affected TCP connection and any responses behind ordered bytes | Missing TCP bytes prevent delivery to all H2 streams beyond the gap | Delivery ordering is per QUIC stream, but connection congestion control still reduces aggregate sending |
| Flow control | TCP receive window | TCP plus H2 connection and stream windows | QUIC connection and stream credit plus H3 behavior |
| Concurrency limit | Client/server policy; no protocol-mandated “six connections” | Peer advertises `SETTINGS_MAX_CONCURRENT_STREAMS` | QUIC transport stream limits and HTTP/3 control |

### Correctly locating head-of-line blocking

HTTP/1.1 pipelining allows multiple requests before prior responses finish, but responses must be returned in request order. A slow first response can therefore block later responses at the HTTP layer. The commonly cited per-origin browser connection count is an implementation policy, not an HTTP/1.1 invariant.

HTTP/2 frames independent HTTP streams, eliminating that response-order dependency. However, all frames travel through one TCP sequence space. A missing TCP segment prevents the receiver from delivering later bytes to H2 even if they contain an unrelated stream.

HTTP/3 maps requests to QUIC streams. Loss on stream A does not impose in-order delivery on stream B. Three shared dependencies remain:

- the congestion controller budgets packets for the entire connection;
- connection-level flow-control credit is shared in addition to per-stream credit;
- QPACK can block a field section that references dynamic-table inserts not yet received. The peer's blocked-stream limit bounds this dependency rather than eliminating it.

Critical H3 control or QPACK stream failure can close the connection. “QUIC streams fail independently” is therefore too strong; the precise guarantee is independent ordered delivery within streams.

### Stream-limit queues are client-visible capacity

In HTTP/2, `SETTINGS_MAX_CONCURRENT_STREAMS` is advertised by the peer and can change. RFC 9113's initial value is unlimited; when an endpoint sends a limit, the specification recommends allowing at least 100, but 100 is not a universal default. HTTP/3 has analogous peer-advertised stream limits at the QUIC layer.

When a client reaches a limit, new calls can queue locally while the server's request queue appears empty. Measure:

- connections and active streams per connection;
- configured and peer-advertised stream limits;
- time waiting for stream capacity separately from DNS, connect, TLS, and server time;
- flow-control-blocked time separately from stream-limit wait;
- connection churn introduced by clients that open extra connections to escape a limit.

The response is not automatically “raise the limit.” More streams increase memory, application concurrency, and the burst presented to dependencies. Coordinate it with admission control.

## QUIC State, Flow Control, Loss, and Migration

### Packet spaces and handshake path

QUIC carries TLS handshake bytes in CRYPTO frames and maintains separate packet-number spaces for Initial, Handshake, and application data. Initial protection prevents trivial corruption but its keys are derivable from public connection inputs; it does not make the first flight confidential from an observer. Handshake and 1-RTT keys come from TLS. Zero-RTT uses resumption state and has the replay properties described above.

The states operators need to distinguish are:

```text
new
 ├─ Initial sent
 │   ├─ Retry received → token-bearing Initial sent
 │   └─ server Initial/Handshake received
 ├─ handshake confirmed
 ├─ 1-RTT application traffic
 ├─ optional path validation / migration
 ├─ draining
 └─ closed or stateless reset observed

resumed
 ├─ Initial + optional 0-RTT application data
 ├─ early data accepted or rejected
 └─ handshake confirmed → 1-RTT application traffic
```

Log handshake branches rather than reducing them to one latency histogram. Retry, version negotiation, 0-RTT rejection, loss by packet-number space, and certificate-flight pressure imply different corrections.

### Congestion and flow control

QUIC has:

- congestion control over ack-eliciting packets for the connection/path;
- connection-level flow control, advanced with `MAX_DATA`;
- per-stream flow control, advanced with `MAX_STREAM_DATA`;
- limits on how many streams a peer may open;
- receiver and application consumption that determines when credit is returned.

A blocked stream can be caused by its own credit, the connection's credit, the peer's stream-count limit, congestion, application scheduling, or QPACK. Export those states separately. Raising `MAX_STREAM_DATA` cannot fix a full `MAX_DATA` limit; raising both cannot fix congestion; opening more streams cannot fix an application that is not reading.

### Loss detection and shared congestion

QUIC packet numbers increase across retransmissions, avoiding TCP's ambiguity about whether an ACK refers to an original or retransmitted sequence range. ACK ranges show gaps. RFC 9002 declares loss using packet and time thresholds and uses PTO probes when acknowledgments do not arrive.

Stream bytes lost in one packet can be retransmitted in another packet without delaying ordered delivery on other streams. Nevertheless, a congestion event reduces the connection's sending budget, so all active streams can see lower throughput. If tenant isolation or independent congestion fate matters, one shared connection may be the wrong multiplexing boundary.

### Address validation and amplification

Before validating a client's address, a QUIC server is limited to sending at most three times the bytes it has received from that address. This bounds reflection amplification but can constrain a large certificate flight. A server can validate with a token, including Retry, at the cost of another client/server flight.

Endpoints must support QUIC datagram payloads of at least 1200 bytes, and UDP datagrams carrying Initial packets are padded to at least 1200 bytes. This gives the server amplification budget and establishes a conservative starting size. It does **not** prove that larger datagrams will traverse the path.

### Connection IDs and migration

QUIC connection IDs allow packets to identify a connection independently of the current IP/port tuple. They support NAT rebinding and path change, but continuity is conditional:

1. the endpoint has usable peer-issued connection IDs;
2. active migration has not been disabled for the relevant behavior;
3. the endpoint validates the new path with PATH_CHALLENGE/PATH_RESPONSE;
4. the load-balancing tier routes the connection ID to state that can process it;
5. congestion and PMTU state are handled safely for the new path.

NAT rebinding is not the same as a proven better path. Until validation, amplification restrictions apply on the new address. Implementations may reset or conservatively reinitialize congestion state when path characteristics change. A QUIC-unaware load balancer that hashes only the new five-tuple can route migrated packets to a backend with no connection state.

Connection IDs also create privacy and security obligations. Stable or structured IDs can link path changes or expose topology. Routing information should be authenticated or encrypted according to the load-balancer design; reset tokens must be unpredictable; retired IDs must stop routing after a bounded drain.

### Passive observability is intentionally limited

QUIC encrypts frames and protects packet numbers, but some long-header fields and connection IDs remain observable. The spin bit is optional: each endpoint independently chooses whether to expose it, and RFC 9000 requires disabling it on a fraction of connections. It is not negotiated and cannot be treated as a complete RTT signal.

Use endpoint metrics and sampled implementation event traces for authoritative state. Many implementations emit qlog-shaped events, but the qlog main schema remains an evolving Internet-Draft; record schema and implementation version with every trace.

## Path MTU and Datagram Sizing

The path MTU is the largest IP packet that can cross the route without fragmentation. Encapsulation by VPN, IPsec, VXLAN, or another tunnel reduces the inner packet budget.

### IPv4, IPv6, and DPLPMTUD differ

- **IPv4 PMTUD:** a sender commonly sets Don't Fragment. A router that cannot forward the packet returns ICMP Destination Unreachable, Type 3 Code 4, “fragmentation needed.” Blocking that signal can create a black hole. IPv4 fragmentation remains possible when DF is clear, but depending on in-path fragmentation is fragile and inefficient.
- **IPv6 PMTUD:** routers do not fragment transit packets. The source can fragment using the Fragment extension header after learning a limit; ICMPv6 Packet Too Big carries the path signal.
- **Datagram PLPMTUD:** RFC 8899 uses transport/application probes and confirmation rather than relying only on ICMP. QUIC can perform DPLPMTUD and should still react to ICMP when available.

QUIC's 1200-byte minimum is a starting interoperability floor, not immunity from PMTU failure. After handshake, a sender that jumps to a larger datagram can still black-hole if Packet Too Big messages are filtered and probe logic is absent or faulty.

### Diagnostic signature and controlled mitigations

A classic trace is:

```text
small handshake packets pass
→ sender emits packet larger than effective tunnel MTU
→ router drops it and returns ICMP/PTB
→ firewall discards the control message
→ sender retransmits at the same size
→ small health checks stay green while large transfers stall
```

Verify the claim with packet-size-dependent tests in both directions, IPv4 and IPv6, ICMP counters/captures, retransmission or PTO state, and a controlled MTU reduction. TCP MSS clamping at a tunnel boundary can be a pragmatic local mitigation, but it must include all encapsulation overhead and does not repair non-TCP datagrams. The durable fix is correct PMTU signaling plus packetization-layer probing.

## Linux Host Data Path and Measurable Limits

This section is Linux-specific. Other kernels expose different queues, APIs, and offloads.

### Receive and transmit paths

A common receive path is:

```text
NIC DMA → RX ring → MSI-X / interrupt moderation → NAPI poll
→ optional native XDP before skb allocation
→ skb + GRO → netfilter / optional conntrack → IP → TCP or UDP
→ socket receive queue → epoll readiness → read/recvmsg → application
```

A common transmit path is:

```text
application write/sendmsg
→ socket and TCP/UDP/QUIC processing
→ netfilter / optional conntrack → qdisc
→ GSO/TSO and checksum offload as supported
→ driver TX ring → NIC
```

TLS changes where encryption and copying occur. `sendfile` can avoid a userspace copy for suitable file-to-socket paths. kTLS can move supported TLS record processing into the kernel, but availability depends on kernel, cipher, direction, socket state, and possibly NIC offload. It is not a universal “zero-copy TLS” switch.

### What the common mechanisms actually do

| Mechanism | Useful effect | Boundary or trap |
|---|---|---|
| RSS/RPS/RFS | Distribute receive processing or steer flows across CPUs | One ordered flow normally remains constrained by its queue/CPU; poor NUMA placement can erase gains |
| Interrupt moderation and NAPI | Amortize interrupt work with polling batches | Larger batches can add latency; NAPI budget exhaustion appears as backlog/drop pressure |
| GRO/GSO/TSO | Amortize per-packet stack work by coalescing or segmenting | Packet captures can show synthetic sizes; disabling offloads for diagnosis changes the workload |
| `epoll` | Reports readiness for registered file descriptors | `maxevents` limits events returned per call, not registered sockets; edge-triggered consumers must drain until `EAGAIN` |
| `io_uring` | Batches submission/completion through shared rings and supports registered resources | It does not remove protocol work; SQ polling can consume a dedicated CPU and changes security/operational assumptions |
| `sendmmsg`/`recvmmsg` | Batch datagram syscalls | Benefits depend on batch size and latency tolerance |
| conntrack | Stateful filtering/NAT when configured | Adds per-flow state, hash work, and a capacity limit; it is not on every path |
| XDP/eBPF | Early verified drop, redirect, or metadata logic | Feature set and driver mode matter; complex transport/application semantics still live elsewhere |
| AF_XDP | Delivers selected frames to userspace with reduced stack work | Requires queue ownership, memory/ring management, and operational tooling |
| DPDK | Poll-mode userspace ownership of NIC queues | Pins cores, changes isolation and observability, and transfers network-stack responsibilities to the application |

### A measurement-first escalation path

1. **Characterize demand:** packets/s, bits/s, requests/s, connection churn, mean and p99 packet size, encryption mode, direction, and burst duration.
2. **Locate the queue:** NIC ring drops, NAPI/softnet backlog, conntrack, qdisc, socket buffer, event-loop delay, application queue, or remote flow control.
3. **Profile cycles:** measure IRQ/softirq, protocol, crypto, copy, allocator, and application CPU with the production offload and NUMA layout.
4. **Use ordinary batching and affinity:** tune RSS queues, CPU/NUMA placement, socket buffers within BDP requirements, syscall batching, and supported offloads.
5. **Canary one structural change:** XDP for early packet policy/redirect, AF_XDP for a selected userspace data path, or DPDK only when full queue ownership is justified.
6. **Retest failure behavior:** routing changes, NIC reset, queue saturation, rollout rollback, packet capture, security policy, and on-call diagnostics.

Useful Linux evidence includes `ss -ti` and `TCP_INFO` for TCP state, `nstat` for stack counters, `tc -s qdisc` for queue statistics, `ethtool -S` for NIC counters, `/proc/net/softnet_stat` for receive backlog pressure, CPU profiles, and targeted eBPF probes. Field names and semantics are kernel/version dependent; dashboards should record that version.

### Connection-state capacity

Memory gives another explicit lower-bound model:

$$
N_{\mathrm{connections}} \leq
\frac{M_{\mathrm{state\ budget}}}
     {m_{\mathrm{state\ per\ connection}}}
$$

Measure $m_{\mathrm{state\ per\ connection}}$ for idle, active, TLS, H2/H3 stream, retransmission, and application metadata states. The practical ceiling is the minimum of memory, file descriptors, ephemeral ports/NAT mappings, conntrack entries, accept queues, stream limits, crypto CPU, and downstream concurrency. [DNS and Connection Management](./13-dns-and-connection-management.md) turns those host facts into pool and fleet budgets.

## Observability That Can Falsify a Transport Hypothesis

Start with a latency decomposition, not a generic “network time” span:

$$
L =
L_{\mathrm{DNS/pool}}
+ L_{\mathrm{connect}}
+ L_{\mathrm{crypto}}
+ L_{\mathrm{stream\ wait}}
+ L_{\mathrm{request/response}}
+ L_{\mathrm{application}}
$$

The terms can overlap under address racing, coalescing, or early data, so the trace schema should include timestamps and branches rather than summing incompatible client library fields blindly.

| Layer | Measurements | Question it can answer |
|---|---|---|
| Path | unloaded and under-load RTT, loss, reordering, ECN marks, PMTU/probe result, IPv4/IPv6, network class | Is the path slower, queued, lossy, or size-dependent? |
| TCP | congestion algorithm/version, `cwnd`, receiver window, bytes in flight, retransmits, SACK/RACK evidence, RTO, pacing/delivery rate, application-limited flag | Is the sender congestion-, flow-, loss-, or application-limited? |
| TLS | full/resumed handshakes, selected group/cipher, certificate bytes, alerts, ticket age, ECH outcome, early-data attempt/accept/reject/retry | Which handshake branch and security policy was exercised? |
| HTTP/2 | negotiated protocol, active/maximum streams, client stream wait, connection/stream flow-control wait, resets and GOAWAY | Is queuing before the server caused by an H2 connection limit? |
| QUIC/HTTP/3 | version, Retry, packet-space loss/PTO, 0-RTT outcome, connection/stream blocked time, QPACK blocked time, migration and path validation, datagram size | Is latency from handshake, loss, shared credit, compression dependency, or path change? |
| Linux/NIC | pps and bit rate, softirq CPU, NAPI budget/backlog, ring drops, GRO/GSO state, qdisc backlog/drop/mark, socket drops, conntrack occupancy | Which host stage is saturated? |
| Application | admission wait, handler queue, dependency time, commit/dedup result, response production/read rate | Is “network backpressure” actually application overload? |

Do not label every connection ID, source address, or packet number as a metric dimension. Keep bounded aggregates by service, region, protocol, network class, and error branch; retain sampled per-connection event traces for diagnosis. Packet captures, TLS key logs, connection IDs, URLs, and qlog-style traces can contain identifiers or decryptable material and require access control, minimization, and retention limits.

## Failure Traces and Root-Cause Boundaries

### Early data accepted twice across regions

```text
t0  region A issues a resumption ticket
t1  client sends an early-data operation through edge A
t2  retry, replay, or attacker sends the same bytes through edge B
t3  A and B consult region-local anti-replay state and both accept
t4  business effect executes twice
```

The TLS layer cannot infer business equivalence. Reject the route from early data, use replay state with a documented consistency scope, and/or converge effects through a durable operation key. A transport or HTTP idempotency label without a backing state transition is not a control.

### PMTU black hole hidden by health checks

Handshakes and health checks fit below the effective tunnel MTU, while data packets do not. ICMP/PTB is filtered, so TCP retransmits or QUIC reaches repeated PTO without learning a usable size. Prove it by varying packet size and IP family. Fix signaling and DPLPMTUD; use MSS clamping only at a boundary you control.

### UDP blocked and HTTP/3 silently falls back

A client has cached H3 availability, but a VPN or enterprise firewall drops UDP. Depending on client racing and fallback behavior, the request eventually uses H2/TCP and succeeds. Error rate stays flat while connect latency and mobile tail latency change. Alert on protocol attempt, success, fallback reason, and fallback delay by network class, not only the final protocol.

### Migration reaches the wrong backend

```text
client changes Wi-Fi → cellular
→ source tuple changes but destination CID remains valid
→ five-tuple load balancer selects a different backend
→ backend has no QUIC connection state
→ packets are dropped or trigger a stateless reset
```

Either provide CID-aware routing or shared/forwarded connection state, or disable/limit migration. Test NAT rebinding and genuine address change separately because their validation and routing paths can differ.

### H2 stream ceiling looks like a slow server

The peer advertises a finite stream limit. Calls wait inside the client pool while server concurrency and CPU remain low. Export stream-acquisition wait and peer settings. Raising the limit may shift the bottleneck downstream; size it with application admission.

### Tail loss waits for probe or timeout

The final data packet is lost and no later packet supplies gap evidence. RACK/TLP or QUIC PTO may recover it before a TCP RTO, depending on implementation and traffic. Diagnose with loss-recovery events and packet sequence, not a fixed latency number. A hedge can reduce caller tail latency but also adds load and does not repair the transport fault.

### Bufferbloat looks like “we need more bandwidth”

During a bulk transfer, throughput approaches the bottleneck rate, loss remains low, and RTT climbs far above $R_0$. That is queue occupancy. Compare under-load RTT, ECN/AQM signals, and qdisc/device queues; increasing a host socket buffer can make the symptom worse.

### Offload or queue mapping regression

A kernel, NIC firmware, VM type, or configuration change disables GRO/GSO or maps too many hot flows to one RX queue. Bit rate may be unchanged while pps work and one softirq CPU saturate. Compare offload state, packet-size observations, per-queue counters, CPU affinity, and cycles per packet to a known-good host before moving to kernel bypass.

## Security and Abuse Boundaries

| Surface | Threat | Required control |
|---|---|---|
| TLS 0-RTT | Replay, duplicated effects, weaker early-data forward secrecy | Route allowlist, bounded ticket/early-data scope, anti-replay design, operation deduplication, rejection/retry telemetry |
| Tickets and ECH configuration | Key theft, over-wide blast radius, stale configuration | Versioned keys, short bounded acceptance, secure distribution, staged rotation, emergency revocation |
| QUIC address validation | Reflection amplification | Three-times pre-validation limit, validated tokens/Retry where justified, rate limiting, source/path monitoring |
| Connection IDs | Linkability, topology disclosure, forged routing input | Unpredictable IDs or protected routing encoding, rotation/retirement, authenticated LB contract |
| Stateless reset | Forged connection termination | Unpredictable reset tokens, secret rotation, bounded retained state |
| H3 fallback | Downgrade to a permitted but weaker operational path | Define allowed protocol floor, observe fallback cause, keep TLS policy consistent, avoid fail-open authentication changes |
| PMTU control traffic | Forged or filtered ICMP/PTB | Validate quoted flow context, rate limit safely, retain DPLPMTUD, do not blanket-block required ICMP |
| Diagnostics | PII, traffic secrets, reusable connection metadata | Sampling, redaction, encryption, access control, short retention, audited TLS key-log use |
| Kernel fast path | Bypass of firewall, conntrack, audit, or namespace policy | Threat-model the exact hook/queue, reproduce policy deliberately, test rollback and isolation |

Encryption protects contents, not all metadata. Packet size, timing, IP addresses, some QUIC header fields, and connection IDs can remain visible. ECH narrows ClientHello exposure but does not hide the destination IP or traffic pattern.

## Migration and Rollout Protocol

### Establish a comparable baseline

Before changing transport, record by region and network class:

- DNS/pool, connect, crypto, stream-wait, first-byte, and completion latency;
- H1/H2/H3 attempt and success mix, reused versus new connections, and resumption/0-RTT outcomes;
- RTT under idle and load, loss/reordering, ECN, PMTU, retransmission/PTO, and fallback;
- goodput and completion time by object size;
- CPU cycles/request and cycles/packet, pps, NIC/softnet/qdisc drops, and memory per connection/stream;
- error, duplicate-effect, and deadline rates.

Freeze exact client, kernel, congestion algorithm, library, TLS, and QUIC versions in the experiment record.

### HTTP/3 rollout

1. Verify origin semantics over H3, including cancellation, upload, trailers, GOAWAY, QPACK limits, and graceful drain.
2. Test IPv4, IPv6, NAT rebinding, address migration, VPN, enterprise firewall, mobile handoff, UDP block/throttle, reordering, burst loss, and reduced MTU.
3. Advertise H3 to a small population with a deliberately bounded advertisement lifetime; cached Alt-Svc or HTTPS records can make rollback non-instant.
4. Keep H2/TCP available and measure attempt-to-fallback delay, not merely final success.
5. Guard on first-byte/completion tails, fallback, CPU, PTO/loss, QPACK/flow-control block, and origin correctness.
6. During rollback, stop new advertisement, continue serving already informed clients through a drain window, and verify protocol mix converges.

### Congestion-control rollout

Canary by a controlled server pool or socket population. Hold routing and workload constant, test mixed algorithms at the real bottleneck, and include AQM/ECN and policer variants. Roll back on fairness, queueing-delay, loss, or CPU regression even if aggregate goodput improves. Record the implementation generation; “BBR” alone is not reproducible.

### Host fast-path rollout

Start with a flame graph and queue evidence showing that ordinary kernel packet processing is the constrained resource. Preserve a control group with identical hardware. For XDP, AF_XDP, or DPDK, verify policy parity, routing convergence, NIC/queue reset, NUMA failover, observability, packet capture, overload behavior, and emergency bypass. A faster happy path with a weaker recovery path is not a production improvement.

## Decision Framework

| Observed problem | First discriminating evidence | Candidate action | Do not infer |
|---|---|---|---|
| New cross-region requests are slow | Flight timestamps for DNS, connect, TLS/QUIC, request, and first byte | Reuse connections; reduce avoidable handshakes; evaluate regional termination or safe resumption | That all latency is server time or that 0-RTT works for first contact |
| Long transfer underfills a high-BDP path | $W/R$, receiver/stream credit, delivery rate, application-limited state | Size windows from measured BDP; inspect loss/pacing; compare congestion control experimentally | That interface line rate is available goodput |
| RTT rises under load with little loss | $R_0$ versus loaded RTT, qdisc/AQM/ECN and bottleneck queues | Bound queues, enable suitable AQM/ECN, pace, isolate flows | That larger buffers or more socket memory help |
| H2 calls wait before reaching server | Client stream-acquisition and flow-control wait | Adjust connection/stream policy with admission budget | That `MAX_CONCURRENT_STREAMS` is always 100 |
| Loss on one request delays unrelated H2 work | TCP loss trace across multiplexed streams | Evaluate H3 or separate connection fate for affected clients | That H3 removes shared congestion |
| Mobile path changes break connections | QUIC path-validation/CID routing trace or TCP reconnect trace | H3 with validated migration and CID-aware routing, plus fallback | That connection IDs guarantee migration |
| Small transfers work and large ones stall | Packet-size sweep, ICMP/PTB, DPLPMTUD state | Repair PMTU signaling/probing; clamp TCP MSS only at controlled tunnels | That QUIC's 1200-byte Initial solves later PMTU |
| Host CPU saturates at modest bit rate | pps, cycles/packet, per-queue softirq, offload state | Batch, rebalance, restore offloads; XDP/AF_XDP if packet path remains dominant | That a bit-rate benchmark represents small packets |
| Tail shelf follows loss | Packet/SACK/RACK/TLP/RTO or QUIC PTO events | Repair path/recovery; then evaluate deadline/hedge policy | A universal 200 ms TCP timer |
| Early-data optimization risks duplicate writes | Attempt/accept/replay/dedup trace and route semantics | Reject early data or build scoped replay plus effect deduplication | That an HTTP method name proves replay safety |

## Verification and Fault-Injection Checklist

### Protocol correctness

- Capture one full TCP plus TLS 1.3 handshake, one resumed 1-RTT handshake, an accepted and rejected early-data attempt, one QUIC 1-RTT handshake, and a QUIC Retry path. Confirm the flight ledger against timestamps.
- Verify ECH success and fallback with the actual DNS SVCB/HTTPS configuration and frontend; prove SNI exposure differs as expected without logging secrets.
- Send a mutating request, cut the connection after peer transport acknowledgment but before the application response, and prove the retry/idempotency contract returns one business result.
- Exercise H2 and H3 stream limits, connection and stream flow-control exhaustion, cancellation, GOAWAY/drain, and QPACK blocked-stream bounds.

### Impaired-path matrix

- Inject delay, random loss, burst loss, and reordering independently; distinguish fast recovery, RACK/TLP, TCP RTO, QUIC loss threshold, and PTO.
- Lower MTU at each encapsulation boundary; test IPv4 DF/ICMP Type 3 Code 4, IPv6 Packet Too Big, blocked control messages, and DPLPMTUD convergence.
- Block and throttle UDP, then measure H3 fallback success and delay.
- Rebind NAT ports and change addresses while requests are active; verify path validation, CID routing, congestion reset behavior, and fallback.
- Load a bottleneck until RTT rises; compare queue delay, AQM/ECN, fairness, goodput, and application tail latency.

### Host capacity

- Sweep packet size and connection churn independently of bit rate; calculate expected pps and compare measured cycles/packet.
- Saturate RX and TX separately; inspect per-queue NIC drops, NAPI/softnet backlog, qdisc, socket drops, crypto, copy, and event-loop delay.
- Toggle one supported offload at a time in a controlled environment and explain packet-capture changes.
- Exhaust file descriptors, ports/NAT, conntrack, socket memory, stream state, and application concurrency separately; verify each has an alert and bounded rejection mode.
- Repeat after kernel, NIC firmware, transport library, congestion algorithm, or TLS library upgrades.

The test passes only when the chosen hypothesis is visible in transport state and the application result remains correct. “No request failed” is insufficient if fallback, replay, queueing, or CPU cost silently moved.

## Key Takeaways

1. **State the latency ledger's assumptions.** Reused encrypted transport is approximately one RTT to first byte; new TCP plus TLS 1.3 is approximately three; new QUIC 1-RTT is approximately two; QUIC 0-RTT is resumed prior state and still replay-constrained.
2. **The smallest window wins.** Congestion window, receive window, QUIC connection credit, QUIC stream credit, HTTP stream limits, and application capacity solve different problems.
3. **QUIC removes connection-wide ordered delivery, not every shared fate.** Congestion, connection credit, critical streams, QPACK dependencies, CPU, and backend routing remain shared.
4. **Recovery timers are implementation and state dependent.** RACK/TLP, QUIC PTO, and TCP RTO must be identified from events rather than a memorized latency shelf.
5. **TLS 1.3 is not synonymous with hidden SNI or safe zero-RTT.** ECH must succeed, PSK modes have different forward-secrecy properties, and replay controls must reach the business effect.
6. **The 1200-byte QUIC floor is not PMTU discovery.** IPv4, IPv6, tunnels, ICMP/PTB, and DPLPMTUD still need explicit testing.
7. **Measure packets and queues, not only bits.** A line-rate capacity plan without packet-size distribution, cycles/packet, queue delay, and connection state is incomplete.
8. **Keep ownership boundaries intact.** Transport protects path and receiver state; connection lifecycle, request admission, retry semantics, and durable deduplication remain higher-layer designs.

## Primary Evidence and Scope

| Claim area | Primary source | Scope used here |
|---|---|---|
| Current TLS 1.3 handshake, PSK modes, early-data forward secrecy and replay | [RFC 9846, *The Transport Layer Security (TLS) Protocol Version 1.3*](https://www.rfc-editor.org/rfc/rfc9846.html) | Normative current TLS 1.3 behavior; it obsoletes RFC 8446 |
| ECH configuration and protocol | [RFC 9848, *Bootstrapping TLS Encrypted ClientHello with DNS Service Bindings*](https://www.rfc-editor.org/rfc/rfc9848.html); [RFC 9849, *TLS Encrypted ClientHello*](https://www.rfc-editor.org/rfc/rfc9849.html) | Normative discovery and ClientHello privacy behavior |
| HTTP treatment of early data | [RFC 8470, *Using Early Data in HTTP*](https://www.rfc-editor.org/rfc/rfc8470.html) | Normative `Early-Data` and `425 Too Early` behavior; application replay safety remains deployment-specific |
| TCP state and core semantics | [RFC 9293, *Transmission Control Protocol*](https://www.rfc-editor.org/rfc/rfc9293.html) | Normative handshake, sequence, close, and byte-stream behavior |
| TCP congestion, delayed ACK, initial window, RTO, and RACK-TLP | [RFC 5681](https://www.rfc-editor.org/rfc/rfc5681.html), [RFC 6928](https://www.rfc-editor.org/rfc/rfc6928.html), [RFC 6298](https://www.rfc-editor.org/rfc/rfc6298.html), [RFC 8985](https://www.rfc-editor.org/rfc/rfc8985.html) | Normative algorithms and bounds; a kernel can choose supported extensions and implementation parameters |
| CUBIC | [RFC 9438, *CUBIC for Fast and Long-Distance Networks*](https://www.rfc-editor.org/rfc/rfc9438.html) | Standard algorithm, not evidence that a given host selected it |
| Reno loss sensitivity | [Mathis et al., *The Macroscopic Behavior of the TCP Congestion Avoidance Algorithm*](https://doi.org/10.1145/263932.264023) | Analytic approximation under the stated assumptions |
| BBRv3 | [IETF CCWG BBR Internet-Draft](https://datatracker.ietf.org/doc/draft-ietf-ccwg-bbr/); [upstream Linux `tcp_bbr.c`](https://github.com/torvalds/linux/blob/master/net/ipv4/tcp_bbr.c) | Draft algorithm versus public upstream implementation; neither proves a vendor kernel's generation or result |
| QUIC transport, TLS mapping, and recovery | [RFC 9000, *QUIC: A UDP-Based Multiplexed and Secure Transport*](https://www.rfc-editor.org/rfc/rfc9000.html), [RFC 9001](https://www.rfc-editor.org/rfc/rfc9001.html), [RFC 9002](https://www.rfc-editor.org/rfc/rfc9002.html) | Normative QUIC v1 state, flow control, migration, security, loss, PTO, and congestion baseline |
| HTTP versions and QPACK | [RFC 9112, *HTTP/1.1*](https://www.rfc-editor.org/rfc/rfc9112.html), [RFC 9113, *HTTP/2*](https://www.rfc-editor.org/rfc/rfc9113.html), [RFC 9114, *HTTP/3*](https://www.rfc-editor.org/rfc/rfc9114.html), [RFC 9204, *QPACK*](https://www.rfc-editor.org/rfc/rfc9204.html) | Normative ordering, stream, settings, fallback, and header-compression behavior |
| IPv4/IPv6 PMTU and datagram probing | [RFC 1191](https://www.rfc-editor.org/rfc/rfc1191.html), [RFC 8200](https://www.rfc-editor.org/rfc/rfc8200.html), [RFC 8201](https://www.rfc-editor.org/rfc/rfc8201.html), [RFC 8899](https://www.rfc-editor.org/rfc/rfc8899.html) | Normative signaling, fragmentation responsibilities, and DPLPMTUD |
| Linux scaling and offloads | [Linux kernel networking scaling documentation](https://docs.kernel.org/networking/scaling.html), [segmentation offloads](https://docs.kernel.org/networking/segmentation-offloads.html), [kTLS](https://docs.kernel.org/networking/tls.html), and [AF_XDP](https://docs.kernel.org/networking/af_xdp.html) | Linux mechanisms; the labeled capacity models require local measurement |
| qlog-shaped diagnostics | [IETF QUIC qlog main-schema draft](https://datatracker.ietf.org/doc/draft-ietf-quic-qlog-main-schema/) | Useful implementation telemetry with draft/version caveat, not a stable RFC guarantee |

## Related Patterns

- [DNS and Connection Management](./13-dns-and-connection-management.md): resolver behavior, address selection, connection pooling, draining, and invisible ceilings
- [Load Balancing](./01-load-balancing.md): L4/L7 boundaries, affinity, and QUIC connection-ID routing
- [CDN Architecture](./04-cdn-architecture.md): regional termination and last-mile protocol deployment
- [Backpressure](./07-backpressure.md): bounded application queues and overload control
- [Retries, Timeouts, and Hedging](./10-retries-timeouts-hedging.md): deadline budgets and ambiguous transport outcomes
- [Idempotency](../01-foundations/08-idempotency.md): durable effect deduplication
- [Encryption](../10-security/06-encryption.md): certificate, key, and rotation policy
- [Polling, SSE, and WebSockets](../07-real-time/01-polling.md): long-lived connection behavior
