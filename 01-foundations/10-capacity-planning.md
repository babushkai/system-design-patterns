# Capacity Planning and Back-of-the-Envelope Estimation

## TL;DR

Estimation is the step that comes before every pattern in this book: a few memorized constants (the latency ladder, per-node throughput classes, seconds-per-day), powers-of-ten arithmetic with stated assumptions, and two pieces of queueing math — **Little's law** (`concurrency = throughput × latency`) and the **utilization curve** (latency explodes as you approach saturation, which is *why* the practical ceiling is ~70–80% and why headroom is a feature, not waste). With those, you can size a design in five minutes, kill a bad one in two, and explain mathematically why retry storms, full connection pools, and autoscaling-at-90% all end the same way. Capacity *planning* is the same math run as a process: forecast from leading indicators, hold an explicit headroom policy, and load-test with an open-loop generator past saturation — because closed-loop tests and coordinated omission hide exactly the collapse you're testing for.

---

## The Numbers You Memorize

Order-of-magnitude constants, 2026 edition. Precision is not the point — knowing that two designs differ by 100× is.

### The latency ladder

| Operation | Time | Mnemonic |
|---|---|---|
| L1 cache reference | ~1 ns | |
| Main memory reference | ~100 ns | RAM is 100× L1 |
| Compress 1KB (snappy-class) | ~2 µs | |
| NVMe SSD random read | ~20–100 µs | SSD is ~1000× RAM |
| Read 1MB sequentially from memory | ~10–50 µs | |
| Read 1MB sequentially from NVMe | ~200 µs–1 ms | |
| Round trip within a datacenter / AZ | ~0.5 ms | the RPC floor |
| Round trip cross-AZ | ~1–2 ms | |
| HDD seek | ~2–10 ms | disks are mechanical |
| Round trip US coast-to-coast | ~60–70 ms | |
| Round trip US ↔ Europe / US ↔ Asia | ~80–150+ ms | physics; see [Multi-Region](../06-scaling/09-multi-region-architecture.md) |

Three conclusions fall straight out of the table: memory beats disk by 10³ (the reason [caching](../04-caching/01-cache-strategies.md) works), one cross-region call costs more than a hundred intra-DC calls (the reason chatty cross-region protocols die), and any request that fans out to N sequential RPCs has a latency floor of N × 0.5ms before anyone does any work (the reason for parallel fan-out and [hedging](../06-scaling/10-retries-timeouts-hedging.md)).

### Throughput classes per node

| Component (one well-tuned node) | Order of magnitude |
|---|---|
| Postgres/MySQL, simple indexed queries | ~5–50K QPS |
| Redis / in-memory KV | ~100K–1M ops/s |
| Kafka, per broker | ~100s of MB/s |
| Stateless API service (JSON, light work) | ~1–10K RPS per core-ish node |
| NIC | 10–100 Gbps (1.25–12.5 GB/s) |
| Single TCP+TLS handshake | 1–3 RTTs before byte one |

### Calendar arithmetic

- **1 day ≈ 86,400 s ≈ 10⁵ s** (the single most-used constant)
- 1 month ≈ 2.6M s; 1 year ≈ 31.5M s ≈ π × 10⁷ s
- 1M requests/day ≈ **12 RPS average**; 1B/day ≈ 12K RPS
- Daily-traffic rule: if X million DAU each do Y actions, average RPS ≈ `X × Y × 10` (because 10⁶/10⁵ = 10)

---

## The Method

1. **Clarify what you're sizing** — reads or writes, average or peak, steady-state or burst. Most estimation arguments are two people sizing different things.
2. **Decompose** into the four meters: request rate, storage, bandwidth, memory (cache).
3. **Round brutally** to powers of ten; keep one significant figure. Track units explicitly — the classic error is a silent KB/MB or bits/bytes slip (network is bits; storage is bytes; factor of 8 has sunk many designs).
4. **State assumptions out loud** ("assume 1 post per DAU per day, 10 reads per post, 200KB median image"). The assumptions are the review surface; the arithmetic is mechanical.
5. **Sanity-check from a second direction** — if the answer implies 400 Postgres shards or 0.3 servers, one of your assumptions is the story.

### Worked example: photo-sharing service

> 100M DAU; each posts 0.5 photos/day and views 50; photo median 200KB + 20KB of thumbnails; metadata 1KB/photo; 5-year retention.

```
Writes:   100M × 0.5 / 10⁵ s        ≈ 500 uploads/s avg   → ×3 peak ≈ 1,500/s
Reads:    100M × 50  / 10⁵ s        ≈ 50K views/s avg     → ×3 peak ≈ 150K/s
          read:write ≈ 100:1 → design is read-path-dominated (cache + CDN problem)

Storage:  50M photos/day × 220KB    ≈ 11 TB/day  ≈ 4 PB/year  ≈ 20 PB over 5y
          → object storage + lifecycle tiers, not a database ([Object Storage](../03-storage-engines/08-object-storage.md))
Metadata: 50M/day × 1KB ≈ 50 GB/day ≈ 18 TB/year → sharded DB territory in year one

Bandwidth (egress, peak): 150K views/s × 200KB ≈ 30 GB/s ≈ 240 Gbps
          → CDN is not optional; origin sees only misses ([CDN](../06-scaling/04-cdn-architecture.md))

Cache:    80/20 rule — 20% of today's content serves 80% of reads.
          Hot set ≈ 20% × (last ~7 days × 11TB) ≈ 15 TB → a small cluster of
          memory-heavy cache nodes, not "cache everything"
```

Ten lines, and the architecture's spine is decided: CDN-fronted object storage, a sharded metadata store, a ~15TB cache tier, and a write path that one decent queue absorbs. That's what estimation is *for* — the patterns in the rest of this book are the implementation details of conclusions you reach here.

---

## Little's Law: The One Formula

> **L = λ × W** — items in the system = arrival rate × time each spends.

Rearranged, it sizes almost everything concurrent:

- **Server concurrency:** 5,000 RPS × 0.2s latency = **1,000 in-flight requests**. With 200 per node, that's 5 nodes *at zero headroom* — see below for why you provision 8.
- **Connection pools:** a service doing 2,000 QPS against a DB at 5ms/query holds 2,000 × 0.005 = **10 busy connections**; a pool of 20 covers bursts, a pool of 500 is a misconfiguration that will melt the database during an incident ([connection management](../06-scaling/13-dns-and-connection-management.md)).
- **Queue depth / consumer sizing:** to drain 10K msg/s at 50ms each you need ≥ 500 concurrent consumers; the backlog when you fall behind grows at (arrival − service) rate, and Little's law converts any backlog into user-visible delay: 1M queued ÷ 10K/s = 100s of lag ([Backpressure](../06-scaling/07-backpressure.md)).
- It also runs in reverse as a **diagnostic**: if concurrency ballooned but throughput didn't, latency grew — something downstream is slow, and your thread pool is the symptom.

## The Utilization Curve: Why Headroom Exists

For a service with random arrivals, waiting time scales like:

```
W ≈ S / (1 − ρ)        S = service time, ρ = utilization

ρ:      50%   70%   80%   90%   95%   99%
W/S:     2×  3.3×    5×   10×   20×  100×
```

Latency versus utilization is a **hockey stick**: the difference between 70% and 90% utilization is not "20% more efficient," it's **3× worse latency**, and variance (real traffic is burstier than the math's best case; heavy-tailed service times make it worse) moves the cliff left. Everything follows:

- **The 70–80% ceiling** isn't folklore — it's the knee of the curve. Run hotter and p99 detonates before average CPU looks scary.
- **Autoscaling must trigger well below the knee** (and scaling takes minutes — during which the curve is doing the 90→99% segment to you).
- **Retry storms have a formula now:** retries add λ exactly when ρ→1, which divides (1−ρ) toward zero — the [metastable failure](../06-scaling/10-retries-timeouts-hedging.md) mechanism in one fraction.
- **Headroom is a product feature**: failover capacity ([static stability](../06-scaling/09-multi-region-architecture.md) — a 2-region pair must run ≤50% each), deploy surges, and the gap between "incident" and "blip" all live in (1−ρ).
- Utilization math is also why **one big queue beats per-node queues** (pooled capacity absorbs variance) and why isolating noisy tenants ([cells, shuffle sharding](../06-scaling/11-cell-based-architecture.md)) is about protecting *everyone else's* ρ.

### Peak factors

Provision for peak, pay for average ([FinOps](../11-observability/06-finops-cost-engineering.md)):

| Pattern | Peak ÷ average |
|---|---|
| Global consumer diurnal | 1.5–2.5× |
| Single-region business hours | 3–5× |
| Media/social events | 5–20× spikes |
| Flash sales, ticket drops | 10–100× — pre-warm, queue at the door, [shed](../06-scaling/05-rate-limiting.md) |
| Synchronized clients (cron at :00, TTL herds) | self-inflicted — add jitter |

---

## Load Testing Without Lying to Yourself

The two errors that invalidate most benchmark numbers:

1. **Closed-loop generators hide collapse.** A closed-loop tool (N virtual users, each waits for a response before sending again) *automatically slows down* when the system slows down — the load adapts to the victim, and you measure a polite system that never sees queueing. Real users are **open-loop**: arrivals don't care about your latency. Test with open-loop, constant-arrival-rate generators (wrk2-style, k6 arrival rates) to see the true curve.
2. **Coordinated omission.** If the generator stalls during a server pause, it *fails to send* the requests that would have suffered most, then reports the survivors' latencies. A 10s server stall can vanish from p99 entirely. Use tools that correct for it (HdrHistogram-based: wrk2, k6 with correction) and sanity-check: max latency should be visible in the percentiles' tail, not suspiciously absent.

What a real test plan includes: ramp **past saturation** to find the actual knee (the number capacity planning needs), hold at the knee to observe degradation mode (graceful shedding vs collapse), then **drop the load and verify recovery** — a system that stays degraded after the spike has a metastable region that production will find. Test with production-shaped data (hot keys, big tenants, cold caches), and prefer shadow/replayed production traffic for realism ([migration shadowing](../15-deployment/06-migration-strategies.md) uses the same machinery).

---

## Capacity Planning as a Process

Estimation sizes the design; planning keeps it sized:

- **Forecast from leading indicators, not resource graphs.** Tie capacity to business metrics (DAU, tenants, orders/day) via your measured unit costs — "each 1M DAU = +120 RPS peak = +2 nodes + 0.4TB cache" — so the plan moves when sales does, not after CPU does.
- **Write the headroom policy down:** e.g., "every tier ≤ 60% at observed peak; survives one AZ loss and one deploy surge simultaneously; one [cell](../06-scaling/11-cell-based-architecture.md) evacuable at all times." Then alert on *headroom remaining*, not utilization — "weeks until knee at current growth" is the metric that triggers procurement and sharding projects while they're still calm.
- **Respect lead times.** Autoscaling handles minutes; quota increases, new shards ([resharding is a migration](../15-deployment/03-database-migrations.md)), GPU capacity, and new regions take weeks-to-months. The plan's job is to start those clocks early.
- **Re-validate the knee quarterly** — code changes move it; the load test from last year describes last year's system.

---

## Cheat Sheet

```
86,400 s/day ≈ 10⁵        1M/day ≈ 12 RPS         1B/day ≈ 12K RPS
RAM 100ns · NVMe 50µs · DC RTT 0.5ms · region RTT 1ms · continent 80ms
concurrency = RPS × latency (Little)         pool ≈ λ×W + headroom
W ≈ S/(1−ρ): 80% → 5×, 90% → 10×             ceiling ≈ 70–80%
peak = 2–5× average (events: 10×+)           provision peak, pay average
open-loop load tests; correct coordinated omission; test past the knee + recovery
```

---

## References

- [Latency Numbers Every Programmer Should Know](https://colin-scott.github.io/personal_website/research/interactive_latency.html) — the Dean/Norvig table, kept current and interactive
- [The SRE Workbook, ch. 12: Non-Abstract Large System Design](https://sre.google/workbook/non-abstract-design/) — estimation as Google teaches it
- [How NOT to Measure Latency](https://www.infoq.com/presentations/latency-response-time/) — Gil Tene; coordinated omission, the canonical talk
- [Open Versus Closed: A Cautionary Tale](https://www.usenix.org/legacy/event/nsdi06/tech/full_papers/schroeder/schroeder.pdf) — NSDI '06; why generator loop model changes everything
- *Systems Performance* (Brendan Gregg) — the USE method and the measurement discipline underneath all of this
- [wrk2](https://github.com/giltene/wrk2) / [k6 arrival-rate executors](https://grafana.com/docs/k6/latest/using-k6/scenarios/executors/) — open-loop, omission-corrected load generation
