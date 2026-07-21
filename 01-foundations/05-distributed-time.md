# Distributed Time

## TL;DR

“What time is it?” hides several different system requirements. Human timestamps need civil time; timeouts need elapsed duration; causal replication needs a happens-before relation; deterministic logs need a total-order tie-breaker; externally consistent databases need timestamps with a proven uncertainty bound. No single clock supplies all five. Wall clocks can step, slew, lie, or be delayed in transit. Monotonic clocks measure local duration but cannot be compared across machines. Lamport clocks preserve causal precedence but cannot detect concurrency. Vector clocks detect concurrency at metadata cost. Hybrid logical clocks combine physical proximity with causal monotonicity but do not manufacture a TrueTime-style uncertainty guarantee. Choose a time domain per field and API, persist that choice, and design the failure behavior for clocks outside their assumed bound.

---

## Scope: Time Coordinates, Not Coordination by Timeout

Clock semantics cover event ordering, causal metadata, and time-synchronization failure; they do not make timeout-based ownership safe.

- [Consistency Models](04-consistency-models.md) owns the client-visible ordering guarantee.
- [Leader Election](../02-distributed-databases/09-leader-election.md) and [Distributed Locks](09-distributed-locks.md) own epochs and fencing.
- [Conflict Resolution](../02-distributed-databases/04-conflict-resolution.md) owns merge policy once concurrent versions are identified.
- [Distributed Scheduling and Timers](../18-workflow-job-systems/03-distributed-cron-scheduling.md) owns durable timer dispatch.

A lease may use time as one input, but a resource must reject stale owners using a monotonic epoch or equivalent authority. Better clock synchronization reduces uncertainty; it does not turn “my timeout expired” into proof that another process stopped acting.

---

## Begin With the Required Relation

| Requirement | Question | Appropriate primitive | Invalid shortcut |
|---|---|---|---|
| Civil timestamp | When did a human-observable event occur? | UTC wall clock plus zone/format metadata | Local time without offset |
| Elapsed duration | Has 300 ms passed on this process? | Monotonic clock | Subtracting wall-clock readings |
| Causal order | Could event A have influenced B? | Lamport/vector/HLC metadata | Sorting wall timestamps |
| Concurrency detection | Are versions incomparable? | Vector/version/dotted version vector | Lamport counter alone |
| Deterministic total order | Which event wins a stable tie? | Logical value plus stable node/sequence tie-breaker | Claiming the order represents causality |
| External consistency | Did committed order respect real-time precedence? | Consensus plus bounded time uncertainty and commit protocol | HLC without an uncertainty source |
| Expiration | May an authority still accept this object? | Authority-owned wall-time policy, often with grace/fencing | Client clock decides validity |

Store the semantic type, not merely an integer called `timestamp`. A schema with `occurred_at_utc`, `observed_monotonic_delta`, `causal_version`, and `lease_epoch` is harder to misuse than four anonymous `int64` fields.

---

## Physical Clocks Are Estimates

A machine clock estimates UTC using an oscillator and periodic observations of reference clocks. Three quantities matter:

- **offset:** current difference between local clock and the reference;
- **frequency error/drift:** how quickly that offset changes;
- **uncertainty:** a bound or confidence interval for the unknown true time.

If a free-running oscillator has fractional frequency error $\rho$, uncertainty grows approximately with time since the last trusted synchronization:

$$
\epsilon(t) \gtrsim \epsilon_0 + \rho t + \epsilon_{path}
$$

where $\epsilon_0$ is uncertainty at synchronization and $\epsilon_{path}$ captures measurement/network error. This is a model for budgeting, not a universal hardware specification. Temperature, virtualization pauses, host migration, oscillator quality, and asymmetric paths all affect the terms.

### NTP measures offset through a network path

For one NTP exchange:

```text
client sends       t1  -------------------->  t2 server receives
client receives    t4  <--------------------  t3 server sends
```

The standard symmetric-path estimates are:

$$
\theta = \frac{(t_2-t_1) + (t_3-t_4)}{2}
$$

$$
\delta = (t_4-t_1) - (t_3-t_2)
$$

`theta` estimates offset and `delta` estimates round-trip network delay excluding server processing. The hidden assumption is roughly symmetric one-way delay. If the outbound path takes 2 ms and the return path takes 80 ms, the exchange cannot tell whether the asymmetry came from the network or the clocks. Multiple diverse sources, filtering, and disciplined oscillators reduce error; they do not eliminate this identifiability limit.

### Slew and step have different application failures

A time daemon can correct offset by changing clock rate (**slew**) or by discontinuously changing the displayed time (**step**). Slewing preserves local monotonic-looking wall time at the cost of taking time to converge. A step can make wall time repeat or jump forward.

Applications must tolerate both unless their platform contract forbids stepping after boot. Never derive elapsed duration from `CLOCK_REALTIME`/wall time. Use the operating system's monotonic clock and understand its suspend semantics: some monotonic sources exclude system sleep, while boot-time sources may include it. Monotonic values are process/host-local coordinates and must not be serialized for comparison on another host.

### UTC, leap seconds, and smear domains

UTC occasionally represents an inserted leap second. Implementations may step, repeat, pause, or smear that adjustment over a window. Two systems using different smear policies can disagree during the window even when both are functioning as configured. Record instants in an unambiguous UTC-based format, retain the source policy for forensic-grade timing, and do civil-time conversion at presentation boundaries. Calendar arithmetic (“next local 09:00”) is not duration arithmetic (“after 24 hours”), especially across daylight-saving transitions.

---

## Synchronization Is a Security Boundary

An attacker who moves a host's time can affect certificate validation, token expiry, log correlation, replay windows, signed artifacts, scheduler decisions, and database timestamp assumptions.

Operate time synchronization as infrastructure:

1. use multiple independent, authorized sources and monitor source selection;
2. authenticate exchanges where supported—Network Time Security (NTS) protects NTP client/server authentication and packet integrity;
3. restrict control/query modes and prevent the service from becoming an amplification endpoint;
4. alert on offset, root dispersion/uncertainty, reachability, frequency correction, source changes, and unsynchronized state;
5. define a degraded mode when uncertainty exceeds the consumer's bound.

Authentication does **not** prevent delay attacks. A man-in-the-middle can delay a valid packet without modifying it. A time-sensitive protocol must include path uncertainty in its bound and fail closed, degrade precision, or switch authority when that bound is exceeded.

Future-dated client values are another attack surface. A single HLC timestamp far in the future can force peers to advance their logical/physical component, damaging retention and conflict policy. Validate remote timestamps against an allowed skew envelope, quarantine offenders, and never let an untrusted client directly set an authoritative commit timestamp.

---

## Causality: The Relation We Can Actually Prove

Lamport's happens-before relation, written $a \rightarrow b$, is the transitive closure of:

1. events ordered within one process;
2. a message send before its corresponding receive; and
3. transitivity—if $a \rightarrow b$ and $b \rightarrow c$, then $a \rightarrow c$.

Events for which neither $a \rightarrow b$ nor $b \rightarrow a$ holds are concurrent in the model. “Concurrent” does not mean simultaneous wall time; it means the system has no causal evidence relating them.

This distinction drives replication. If version B descends from A, B can supersede A. If A and B are concurrent, silently choosing the later wall timestamp may discard a valid update; the system needs a domain merge or explicit winner rule.

---

## Lamport Clocks: Causality-Preserving Scalars

Each process maintains integer $L$:

```text
local event or send:  L := L + 1; attach L
receive message m:    L := max(L, m.L) + 1
```

This satisfies the **clock condition**:

$$
a \rightarrow b \implies L(a) < L(b)
$$

The converse is false. `L(a) < L(b)` does not prove that `a` caused `b`; two independent processes can generate comparable counters. Lamport values are excellent for extending a causal partial order into a deterministic total order, for example `(L, node_id, local_sequence)`, but that total order is a chosen serialization, not observed causality.

### Operational rules

- Persist the counter or pair it with a boot/session epoch if values survive restart.
- Use a fixed-width representation with an explicit overflow policy.
- Authenticate the sender when a remote value can advance local state.
- Do not treat a Lamport value as an age, deadline, or UTC timestamp.

---

## Vector Clocks: Detecting Concurrent Versions

A vector clock maps participant identifiers to counters. On a local event, a process increments its component. On receive/merge, it takes the component-wise maximum and then advances its own component.

Vector $V$ dominates $W$ when every component is greater than or equal and at least one is greater:

$$
W < V \iff (\forall i, W_i \le V_i) \land (\exists j, W_j < V_j)
$$

If neither vector dominates, the versions are concurrent.

```text
A = {east: 4, west: 2}
B = {east: 3, west: 5}

A does not dominate B; B does not dominate A
-> retain siblings or invoke a domain merge
```

The hard problem is not the comparison; it is membership. Naive vectors grow with every writer identity ever observed. Replica replacement, mobile clients, and elastic fleets make that unbounded.

Practical variants change the identity scope:

- **version vectors** track durable replica/actor identities rather than requests;
- **dotted version vectors** separate one new event (“dot”) from the causal context, representing siblings more compactly;
- version-vector-with-exceptions or interval schemes compress contiguous event ranges;
- retirement protocols prune an actor only after every relevant replica has learned that its epoch is dead.

Pruning without such a protocol can manufacture false concurrency or, worse, make a genuinely concurrent update appear dominated and lose data. The merge policy itself belongs in [Conflict Resolution](../02-distributed-databases/04-conflict-resolution.md).

---

## Hybrid Logical Clocks: Physical Proximity Plus Causal Order

An HLC timestamp is commonly represented as `(physical, logical)`. On a local event, the node reads its wall clock. If physical time advanced, it uses the new physical value and resets the logical counter; otherwise it increments the counter. On receive, it takes the maximum of local physical, remote physical, and current wall time, then advances the logical part enough to order the receive after the send.

The result has two valuable properties:

1. causal precedence implies HLC order; and
2. when physical clocks remain within their operating bound, the physical component stays close to wall time.

That enables causally ordered change feeds, snapshot coordinates, and retention partitions that remain intelligible to operators. It avoids vector-clock growth but, like a Lamport clock, cannot detect concurrency from scalar comparison alone.

An HLC is **not** an uncertainty interval and does not by itself guarantee external consistency. It can preserve causality even while the underlying wall clock is wrong. A protocol that needs “transaction B began after A returned, therefore B's timestamp is later” needs consensus/commit rules plus a validated bound on physical-time uncertainty.

### HLC failure containment

- cap or reject remote physical components beyond the accepted future-skew window;
- include a node/epoch tie-breaker when a total order must be unique;
- monitor logical-counter growth—sustained high values can indicate a stuck/future clock;
- define encoding overflow and rollback behavior before deployment;
- retain original event time separately when HLC is an ingestion/order coordinate.

---

## Bounded-Uncertainty Time and Commit Wait

TrueTime-style APIs return an interval `[earliest, latest]` that is asserted to contain actual time, rather than pretending one scalar is exact. Let uncertainty width be $\epsilon = latest-earliest$.

A simplified externally consistent commit protocol is:

1. reach consensus on the transaction and hold the required concurrency-control state;
2. choose commit timestamp $s$ no earlier than the current interval's `latest` bound;
3. wait until the time service proves `earliest > s`;
4. expose the commit and release state.

The **commit wait** ensures that when a caller observes completion, real time has passed the assigned timestamp. A later transaction can therefore receive a later timestamp, preserving real-time precedence. Commit latency includes uncertainty: poor synchronization is now a database tail-latency problem by design.

The guarantee depends on the entire trusted system—reference clocks, oscillator holdover, cross-checking, daemon/kernel path, uncertainty calculation, and fail-safe behavior. Copying the API shape without that infrastructure gives a decorative interval, not the proof.

---

## Design Patterns by Time Domain

### Timeouts and deadlines

Within one process, compute `deadline = monotonic_now + duration` and pass a **remaining duration** across RPC boundaries. An absolute wall deadline is useful for cross-service tracing and policy, but each hop should bound it against local monotonic elapsed time. This prevents a backward wall-clock step from extending work indefinitely.

### IDs

Time-sortable identifiers improve index locality and observability, but timestamp bits do not ensure uniqueness or causal order. Include sufficient randomness or a node/sequence component; specify behavior when the clock moves backward; never authorize access or establish “latest write” solely from ID order.

### Expiration and retention

An authoritative service evaluates persisted expiry using its clock policy and grace window. Clients may display a countdown using monotonic time but do not decide validity. For destructive retention, separate “eligible after time T” from a durable deletion workflow with legal holds, replication confirmation, and audit state.

### Logs and traces

Record wall time for human correlation, a per-process monotonic offset for local duration, trace/span causality, and an ingestion/commit coordinate where available. A log sorter can present a best-effort timeline, but it must not invent causal certainty when uncertainty intervals overlap.

### Leases

Time can bound how long an authority intends a lease to remain valid. It cannot stop a paused old holder from resuming. Every grant carries a monotonically increasing fencing token, and the protected resource accepts only the greatest token seen. See [Distributed Locks](09-distributed-locks.md).

---

## Capacity and Error Budgets for Time

Treat uncertainty as a consumed budget. For a consumer that tolerates maximum error $E_{max}$:

$$
E_{path} + E_{sync} + E_{holdover} + E_{software} \le E_{max}
$$

Measure each term or conservatively bound it. A certificate check may tolerate more error than a high-frequency event sequencer; a commit-wait database converts a larger bound directly into latency.

Time service capacity includes request QPS, source diversity, fanout hierarchy, recovery after source loss, and thundering-herd behavior at boot. Avoid making every host depend directly on one remote public source. Use a controlled hierarchy or provider-supported local endpoint, diversify upstream references, and stagger startup/resynchronization.

For causal metadata, budget bytes per object/event and merge work. A vector with $R$ active identities costs $O(R)$ space and comparison time; an HLC costs $O(1)$ but loses concurrency detection. That is an architectural trade, not merely a serialization choice.

---

## Failure Modes

### Backward wall-clock step extends a timeout

A worker computes `now() - started_at < timeout`. The wall clock steps backward, so the job runs beyond its safety window. Use monotonic elapsed time locally; persist a separate authority-owned wall deadline for restart recovery.

### Mixed leap-smear domains reorder events

One region smears UTC while another steps. Wall timestamps diverge during the event and a last-write-wins policy drops an update. Use causal/version metadata for conflict decisions and record the time-source policy for forensic interpretation.

### VM pause breaks an assumed lease bound

A lease holder is paused longer than the lease, resumes with unchanged memory, and writes. Clock correctness is irrelevant: its authority expired elsewhere. The storage service rejects its stale fencing epoch.

### Authenticated NTP is delayed

Packets pass integrity checks but an attacker delays one direction. Offset becomes biased. Monitor path delay/dispersion, use diverse paths and sources, and enter the defined uncertainty-exceeded mode. NTS authenticates the peer and packets; it cannot prove prompt delivery.

### Future HLC poisons a cluster

One compromised client sends a timestamp months ahead. Peers advance, newly written data appears future-dated, and TTL/retention logic misbehaves. Accept time-order metadata only from authenticated peers, enforce maximum future skew, and separate untrusted event time from authoritative HLC.

### Replica-identity pruning loses causality

A retired vector-clock component is deleted before an offline replica returns. Its old update can no longer be recognized as an ancestor/concurrent sibling. Use epoch-scoped membership and a retirement barrier acknowledged by all replicas whose data can re-enter.

---

## Observability and Incident Evidence

Time telemetry must answer both “is the daemon synchronized?” and “can this application still uphold its promise?” Track:

- offset estimate, uncertainty/root dispersion, round-trip delay, frequency correction, stratum/source, and last good update;
- source changes, falseticker/outlier decisions, NTS failures, and unsynchronized duration;
- wall-clock steps and process/VM suspend intervals;
- HLC logical component, rejected future timestamps, and maximum peer skew;
- vector width, retired identities, sibling/concurrency rate, and merge outcomes;
- commit-wait duration or other application-visible uncertainty cost;
- event-time versus ingestion-time lag for data pipelines.

Preserve the raw synchronization observations and configuration around an incident. Correcting the clock erases the symptom but not invalid decisions already made under the bad interval.

---

## Verification and Fault Injection

Test clock semantics as protocols, not helper functions:

- step wall time backward and forward while monotonic time advances;
- slew slowly, suspend/resume the process, and reboot with lost local state;
- inject asymmetric delay, source disagreement, loss, and an authenticated-but-delayed peer;
- cross a leap/smear window and daylight-saving boundary;
- deliver duplicate, reordered, and concurrent causal messages;
- introduce a future HLC and verify quarantine rather than cluster-wide advance;
- retire and later revive a vector-clock actor;
- verify that stale lease holders are rejected by epoch even with a “valid” local clock.

Property tests should assert the clock condition: whenever the harness creates a send/receive causal edge, the receive timestamp orders after the send. Vector tests additionally assert that incomparable vectors are never reported as ancestors. For bounded-time protocols, test the fail-safe path when the uncertainty service cannot maintain its advertised bound.

---

## Decision Framework

1. **Is the value for display/audit?** Use UTC wall time with explicit source and uncertainty where material.
2. **Is it a local duration or timeout?** Use a monotonic clock.
3. **Must descendants order after ancestors?** Use Lamport or HLC metadata.
4. **Must the system detect concurrent versions?** Use vectors/dotted versions or preserve an explicit dependency graph.
5. **Must timestamps respect completed-before-started real time?** Require an externally consistent database/time service; HLC alone is insufficient.
6. **Does time grant authority?** Add an authority-owned epoch/fencing check.
7. **What happens outside the clock bound?** Define fail-closed, degraded, or alternate-authority behavior before production.

---

## Key Takeaways

1. Wall time, duration, causality, total order, and external consistency are different contracts.
2. NTP estimates offset under path assumptions; authentication cannot eliminate delay uncertainty.
3. Lamport clocks preserve causality in one direction but do not detect concurrency.
4. Vector clocks detect concurrency at identity/metadata cost; safe pruning is a membership protocol.
5. HLC keeps a scalar close to physical time while preserving causal order, but is not a bounded-time proof.
6. Timeouts do not revoke authority. Fencing epochs protect resources from paused or partitioned old owners.
7. Clock uncertainty must appear in capacity, latency, observability, and failure policy.

---

## References

- Leslie Lamport, [*Time, Clocks, and the Ordering of Events in a Distributed System*](https://www.microsoft.com/en-us/research/publication/time-clocks-ordering-events-distributed-system/), 1978.
- D. Mills et al., [RFC 5905: Network Time Protocol Version 4](https://www.rfc-editor.org/rfc/rfc5905.html), IETF, 2010.
- D. Reilly, H. Stenn, and D. Sibold, [RFC 8633: Network Time Protocol Best Current Practices](https://www.rfc-editor.org/rfc/rfc8633.html), IETF, 2019.
- D. Franke et al., [RFC 8915: Network Time Security for NTP](https://www.rfc-editor.org/rfc/rfc8915.html), IETF, 2020.
- Sandeep S. Kulkarni, Murat Demirbas, Deepak Madappa, Bharadwaj Avva, and Marcelo Leone, [*Logical Physical Clocks and Consistent Snapshots in Globally Distributed Databases*](https://cse.buffalo.edu/tech-reports/2014-04.pdf), 2014.
- James C. Corbett et al., [*Spanner: Google's Globally-Distributed Database*](https://research.google.com/archive/spanner-osdi2012.pdf), OSDI, 2012.
- Paulo Sérgio Almeida, Carlos Baquero, and Victor Fonte, [*Interval Tree Clocks: A Logical Clock for Dynamic Systems*](https://gsd.di.uminho.pt/members/cbm/ps/itc2008.pdf), 2008.
