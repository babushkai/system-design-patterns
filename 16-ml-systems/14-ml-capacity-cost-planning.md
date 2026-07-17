# ML Capacity and Cost Planning

## TL;DR

ML capacity planning is ordinary capacity planning with two expensive twists: accelerators are costly indivisible resources, and model work has high variance. The math is still first principles: request rate, service time, concurrency, utilization, memory, bandwidth, and queueing. But the bottlenecks differ by workload. Online inference is usually governed by p99 latency, batch wait, model load time, accelerator memory, and feature-fetch fanout. Batch inference is governed by throughput and cost per prediction. Training is governed by accelerator-hours, data I/O, checkpointing, and cluster scheduling. The central rule is **size against the bottleneck resource and the percentile that the product experiences, not the average metric the dashboard happens to show.** For GPU serving, that usually means queue depth, batch latency, GPU memory, and utilization — not CPU.

---

## The Units That Matter

Every ML capacity estimate starts with four units:

```text
requests/second       incoming prediction or training examples
seconds/request       service time per item or batch
bytes/request         features, embeddings, prompts, outputs
accelerator memory    model weights + activations + batch/KV/cache state
```

From these, derive the rest:

```text
concurrency = arrival_rate × latency          (Little's law)
throughput_per_replica = batch_size / batch_service_time
replicas = peak_rps / safe_throughput_per_replica
cost_per_prediction = hourly_cost / predictions_per_hour
```

The numbers do not need to be exact at first. They need to be explicit. A wrong estimate with visible assumptions can be corrected. An architecture with no estimate discovers its costs in production.

---

## Capacity Is a Feasible Region, Not One Number

A replica does not have a single capacity. It has a **capacity envelope** bounded simultaneously by compute, memory, memory bandwidth, host preprocessing, network, feature dependencies, and the latency SLO. A configuration that sustains 2,000 predictions/s may be invalid because its p99 is 300 ms; a configuration meeting 50 ms at batch one may be economically invalid because it uses only 8% of the accelerator. The useful capacity is the highest offered load for which *all* constraints remain true.

```mermaid
flowchart LR
    WORK["Workload distribution<br/>arrival, shape, priority"] --> QUEUE["Admission + queue"]
    QUEUE --> HOST["Host preprocess<br/>CPU + memory"]
    HOST --> DEVICE["Accelerator<br/>FLOPs + HBM + bandwidth"]
    DEVICE --> POST["Postprocess + log"]
    FEATURES["Feature/cache dependencies"] --> HOST
    QUEUE -. constrained by .-> SLO["Latency / deadline SLO"]
    DEVICE -. constrained by .-> COST["Unit-cost budget"]
    FEATURES -. constrained by .-> AVAIL["Availability budget"]
```

Benchmark the envelope as a surface, not a point. Sweep concurrency, batch wait, maximum batch, input-shape bucket, precision, model variant, and hardware. For each cell record achieved throughput, p50/p95/p99 queue and service time, memory high-water mark, errors, quality delta, power if available, and unit cost. Mark a cell feasible only if it meets every guardrail. The serving controller then chooses among feasible cells according to the product objective: minimum cost, minimum latency, or maximum goodput.

The workload input to this exercise is a distribution. At minimum retain per-route arrival series at one-second or finer buckets, input shapes, feature count, model choice, output size, cacheability, tenant/priority, and correlation with external events. Independent averages erase the worst interaction. A promotion may simultaneously increase model compute by 20%, reduce cache hit rate, and expand the eligible population during the daily peak. Replaying joined production traces catches that combination; multiplying separate averages does not.

---

## Serving Regimes Have Different Capacity Goals

The first capacity decision is not hardware. It is serving regime.

| Regime | Goal | Main constraint | Typical metric |
|---|---|---|---|
| Online synchronous | Answer inside user request | p99 latency | p99, timeout rate, queue wait |
| Online async | Process soon after event | queue lag | end-to-end lag, backlog |
| Batch scoring | Finish by deadline cheaply | throughput/cost | predictions/hour, cost/run |
| Training | Produce artifact by deadline | accelerator-hours + I/O | time-to-train, cost/run |

A daily churn score does not need online GPU serving. A checkout fraud decision does. Moving work to the cheapest regime that satisfies freshness is the highest-leverage cost optimization in ML systems.

---

## Online Inference Sizing

For online inference, start with the latency budget and peak traffic.

Example:

```text
Peak traffic: 12,000 RPS
End-to-end p99 budget: 120 ms
Budget allocation:
  ingress/auth         10 ms
  feature fetch        35 ms
  inference            45 ms
  postprocess/logging  15 ms
  network/headroom     15 ms
```

If one model replica can process a batch of 32 in 20 ms, its theoretical throughput is:

```text
32 / 0.020s = 1,600 predictions/s
```

But theoretical throughput is not safe throughput. At high utilization, queueing delay explodes. If the safe target is 65% utilization:

```text
safe_throughput = 1,600 × 0.65 ≈ 1,040 predictions/s
replicas = 12,000 / 1,040 ≈ 12 replicas
```

Then add failure and deploy headroom. If the service must survive one replica or one AZ loss and support rolling deploys, 12 is not enough. The final number may be 16-20 replicas depending on topology.

The key is that batch throughput and latency budget interact. Larger batches raise throughput but add queue wait. A capacity plan that uses max batch throughput without modeling batch wait is optimistic by design.

---

## Dynamic Batching Capacity

Dynamic batching has two knobs: max batch size and max wait. Capacity planning must account for both.

```text
max_batch_size = 32
max_wait = 5 ms
batch_compute_time = 20 ms
```

Under heavy load, batches fill quickly, so the service approaches 32/20ms throughput. Under low load, requests may wait the full 5ms and run smaller batches, lowering utilization. Under bursty load, queues oscillate and p99 can jump.

A simple sizing rule:

```text
per_replica_capacity ≈ effective_batch_size / (batch_compute_time + average_batch_wait)
```

But p99 planning must use tail wait, not average wait. If max wait is 5ms and burst queueing adds 30ms before the request even enters the batch, the inference budget is blown even though GPU utilization looks healthy.

Monitor:

- queue depth,
- queue wait p50/p95/p99,
- effective batch size,
- batch compute time,
- timeout/drop rate,
- GPU utilization and memory.

CPU is not a saturation signal for GPU-bound inference.

---

## Queueing, Goodput, and the Utilization Knee

Throughput counts completed work. **Goodput** counts work completed within its SLO and correctness constraints. If a fleet accepts 15,000 RPS but 30% time out after spending accelerator cycles, throughput can look impressive while useful capacity is only 10,500 RPS and the wasted work worsens the queue. Capacity tests should plot goodput against offered load and choose an operating point before the knee where queue delay rises sharply.

For an idealized M/M/1 queue, mean queueing time is:

```text
utilization ρ = arrival_rate λ / service_rate μ
mean time in system W = 1 / (μ - λ)
```

The model is not a faithful description of batched accelerator service, but it teaches the invariant: as `λ` approaches `μ`, delay diverges. Real ML service is worse in several ways—service time varies with input shape, batches couple requests, downstream fanout adds correlated tails, and retries add endogenous traffic. Kingman's approximation for a single general queue exposes why variance matters:

```text
Wq ≈ (ρ / (1 - ρ)) × ((Ca² + Cs²) / 2) × E[S]

Ca = coefficient of variation of inter-arrival time
Cs = coefficient of variation of service time
E[S] = mean service time
```

Two fleets with the same average rate and service time can need very different headroom because one has bursty arrivals or long-tailed inputs. Shape bucketing reduces `Cs`; token/input limits bound it; admission control prevents one pathological request from occupying a batch and stretching every neighbor's latency. This is why “keep GPU utilization at 90%” is not a universal efficiency target. The correct target is the measured utilization immediately before SLO goodput bends, with margin for estimation error and failure.

Retries must enter the model as load. If a dependency has 5% ambiguous failures and every request retries twice without a budget, offered work becomes approximately `λ × (1 + 0.05 + 0.05²)` before synchronized retry waves and timeouts are considered. At saturation, the extra work lowers success probability and creates more retries—a positive feedback loop. Capacity protection therefore includes retry budgets, deadlines propagated to queues, and rejection before expensive work when the remaining deadline cannot be met.

---

## Memory Sizing: The Hard Limit

Accelerator memory is a hard capacity boundary. If a batch does not fit, the replica crashes or rejects work. Estimate memory before throughput.

```text
total_memory = model_weights
             + runtime_overhead
             + activation_memory(batch_size, input_shape)
             + cache_memory
             + fragmentation_headroom
```

For classical neural inference, activation memory grows with batch size and input size. For LLMs, KV cache grows with sequence length and concurrent tokens. For recommendation systems, embedding tables may dominate memory, and feature caches may sit outside the model server.

The memory capacity question is:

```text
max_safe_batch = floor((device_memory - weights - overhead - headroom) / per_item_activation)
```

Use headroom. Memory fragmentation, variable input sizes, and framework workspaces make exact fits dangerous. A batch size that works in a benchmark may OOM under production input shape variance.

A worked sizing example makes the hard limit visible:

```text
device_memory              = 24 GiB
model_weights              = 7.5 GiB
runtime/framework overhead = 1.5 GiB
workspace/fragmentation    = 2.0 GiB
safety headroom            = 3.0 GiB
available_for_activations  = 24 - 7.5 - 1.5 - 2.0 - 3.0 = 10 GiB
per_item_activation_p95    = 220 MiB
max_safe_batch_p95         = floor(10 GiB / 220 MiB) ≈ 46
```

If production input size has a long tail, use p99 or an explicit max input policy:

```text
per_item_activation_p99 = 410 MiB
max_safe_batch_p99      = floor(10 GiB / 410 MiB) ≈ 24
```

The correct batch size may be 24, not 46, because an online service must survive the tail. Another valid design is to cap or bucket input sizes: small inputs use batch 46, large inputs route to a separate pool with batch 16. The anti-pattern is benchmarking on average inputs and discovering at p99 traffic that the model server OOMs.

---

## Feature Fetch Fanout Can Dominate Inference

Many ML serving systems are not model-bound. They are feature-bound.

A request that fetches 40 features from 5 stores can spend more time on network I/O than on inference. Capacity planning must include feature-store QPS and fanout:

```text
prediction_rps = 12,000
feature_groups_per_prediction = 5
online_feature_qps = 60,000 group reads/s
```

If each group read fans out internally to multiple partitions, the backend load is higher. Add cache hit rates:

```text
backend_qps = prediction_rps × feature_groups × (1 - cache_hit_rate)
```

At 90% hit rate:

```text
12,000 × 5 × 0.10 = 6,000 backend reads/s
```

At 50% hit rate:

```text
12,000 × 5 × 0.50 = 30,000 backend reads/s
```

A cache hit-rate regression can therefore create a 5× backend load increase with no traffic growth. Feature fetch capacity belongs in the inference plan.

---

## Cold Start and Warm Capacity

Model replicas have load time. Large models may take tens of seconds or minutes to load weights into memory. Capacity that is not warm is not immediately available.

If demand can spike by 5,000 RPS in one minute and a replica takes two minutes to become ready, reactive autoscaling cannot save the SLO. The capacity must already be warm or predicted early enough.

```text
warmup_time = 120s
scale_signal_lead_time must be > 120s
```

This changes scale-down policy. Ordinary web services may scale down aggressively. Model services often keep warm pools because the cost of cold-start latency is higher than the cost of idle capacity.

A capacity plan should state:

- minimum warm replicas,
- scale-up signal and threshold,
- expected warmup time,
- scale-down delay,
- cold-start user impact,
- whether scale-to-zero is allowed.

For latency-critical large models, scale-to-zero is usually a false economy.

---

## Batch Inference Sizing

Batch scoring is throughput planning. The question is: can the job finish before its deadline at acceptable cost?

Example:

```text
Users to score: 200M
Deadline: 4 hours
Required throughput: 200M / (4 × 3600) ≈ 13,900 predictions/s
```

If one worker scores 2,000 predictions/s safely:

```text
workers = 13,900 / 2,000 ≈ 7 workers
```

Add straggler and retry headroom; provision 10. Then estimate cost:

```text
10 workers × 4 hours × $3/hour = $120 per run
```

Batch inference often becomes I/O-bound. Reading features, writing predictions, and shuffling candidate sets can dominate model compute. Measure throughput by pipeline stage:

```text
read features → preprocess → inference → write outputs
```

The slowest stage sets throughput. If inference is 50,000/s but writes are 10,000/s, buying more GPUs does nothing.

---

## Training Cost Planning

Training cost is accelerator count times wall-clock time plus storage and orchestration overhead.

```text
training_cost = accelerator_count × hours × hourly_rate
              + CPU/memory overhead
              + storage/read costs
              + experiment multiplier
```

The experiment multiplier is what surprises teams. One training run may be cheap; hyperparameter search multiplies it.

```text
base run: 8 GPUs × 6h × $4/h = $192
50 trials with early stopping at 40% average length:
  50 × 0.4 × $192 = $3,840
weekly search: ≈ $200K/year
```

Cost planning should be attached to the training pipeline. Every run records estimated cost, actual cost, dataset size, accelerator type, utilization, and reason. Cost anomalies often reveal pipeline bugs: data doubled, cache missed, distributed training slowed, spot instances stopped being used, or hyperparameter search exploded.

---

## Training Throughput and I/O

Accelerator utilization during training is often limited by data I/O. If GPUs wait for batches, cost burns without learning.

Estimate data demand:

```text
batch_size = 1024 examples
example_size = 200 KB
steps_per_second = 20
data_rate = 1024 × 200KB × 20 ≈ 4 GB/s
```

If the storage path cannot deliver 4 GB/s sustained, the GPUs idle. Fixes include:

- sharding data into appropriately sized files,
- sequential formats for full scans,
- local NVMe caching,
- prefetching and parallel data loaders,
- co-locating compute with storage,
- avoiding per-example remote reads.

Training capacity is therefore not only GPU quota. It is GPU + storage bandwidth + CPU preprocessing + network.

---

## Distributed Training Scaling Efficiency

Adding GPUs does not linearly reduce training time. Communication grows.

```text
speedup_N = single_gpu_time / N_gpu_time
scaling_efficiency = speedup_N / N
```

If 8 GPUs give 6× speedup, efficiency is 75%. If 64 GPUs give 20× speedup, efficiency is 31%; cost per useful training step may be worse than a smaller job.

A capacity plan should measure throughput per dollar, not only time-to-train. Sometimes the right answer is fewer GPUs for longer because queue wait and communication overhead make the larger job wasteful. Sometimes the product deadline justifies the waste. Make the trade explicit.

---

## Multi-Tenant GPU Clusters

A shared ML cluster is a scheduling system. Without policy, one team can starve others.

Required controls:

| Control | Purpose |
|---|---|
| Quotas | Bound team spend and usage |
| Fair share | Allocate idle capacity without permanent monopolies |
| Priority classes | Let production retraining preempt experiments |
| Gang scheduling | Start distributed jobs all at once or not at all |
| Checkpointing | Make preemption safe |
| Idle detection | Kill jobs holding GPUs without progress |
| Budget alerts | Stop runaway searches |

Gang scheduling is crucial. A distributed job needing 8 GPUs cannot make progress with 7. Partial allocation can deadlock the cluster: jobs hold some GPUs while waiting for the rest. All-or-nothing allocation avoids this.

A multi-tenant cluster also needs a fairness contract, not only quotas:

```yaml
gpu_cluster_policy:
  quota_window: weekly
  teams:
    search:
      guaranteed_gpu_hours: 1200
      max_burst_gpus: 64
      budget_usd: 18000
    fraud:
      guaranteed_gpu_hours: 800
      max_burst_gpus: 32
      budget_usd: 12000
  priority_classes:
    production_retraining:
      preemptible: false
      max_queue_wait: 30m
    launch_blocking_experiment:
      preemptible: true
      max_queue_wait: 4h
    exploratory_notebook:
      preemptible: true
      max_runtime: 6h
  preemption:
    require_checkpoint_age_below: 15m
    grace_period: 5m
  idle_detection:
    gpu_utilization_below: 5%
    duration: 20m
    action: notify_then_kill
```

Fair-share scheduling should answer two questions separately: what capacity is guaranteed when the cluster is busy, and who may opportunistically use idle capacity when it is not. Without this distinction, either the cluster sits idle under rigid quotas or one team squats on shared GPUs and calls it efficiency.

---

## Cost per Prediction

Serving cost should be reduced to cost per prediction or cost per 1,000 predictions.

```text
cost_per_prediction = hourly_instance_cost / predictions_per_hour
```

Example:

```text
GPU instance: $4/hour
Throughput: 1,000 predictions/s = 3.6M/hour
Cost: $4 / 3.6M = $0.0000011 per prediction
```

At 10% utilization:

```text
effective throughput: 360K/hour
Cost: $4 / 360K = $0.000011 per prediction
```

A 10× utilization drop creates a 10× unit-cost increase. This is why batching, consolidation, and right-sizing matter more for ML serving than for many ordinary services.

Include the full serving stack, not only the model server:

```text
unit_cost = (model_server_cost
           + feature_store_incremental_cost
           + cache_cost
           + logging/storage_cost
           + monitoring_cost) / predictions
```

Example:

```text
model servers:       16 replicas × $4/h      = $64/h
feature store load:                           = $18/h
prediction logging:  12K rps × 2KB × storage  = $6/h
monitoring/vendor:                            = $4/h
total:                                        = $92/h
predictions/hour:    12,000 × 3600            = 43.2M
cost/prediction:     $92 / 43.2M              ≈ $0.00000213
cost/1K predictions:                           ≈ $0.00213
```

This view catches misleading optimizations. Halving model GPU cost while doubling feature-store load is not a win; lowering p99 by adding a cache that costs more than the saved compute may or may not be justified depending on product value.

---

## CPU Versus GPU Crossover

GPU is not automatically cheaper or faster. The crossover depends on model size, traffic, latency budget, and achievable batching.

Use this comparison:

```text
CPU fleet cost at required p99 and RPS
vs
GPU fleet cost at required p99 and RPS, including warm idle capacity
```

Small models at low traffic often belong on CPU. A GPU at 5% utilization is usually worse than a few CPU nodes. Quantization, distillation, pruning, and compilation can shift the crossover by making CPU viable or reducing GPU size.

The hardware decision should be per model, not platform-wide ideology.

---

## Headroom Policy

ML services need explicit headroom for:

- burst traffic,
- model load/warmup time,
- rolling deployments,
- shadow/canary traffic,
- dependency degradation,
- AZ or node failure,
- larger-than-normal inputs,
- batch jobs overlapping online peaks.

A sample policy:

```text
Online inference:
  steady peak GPU utilization ≤ 65%
  queue wait p99 ≤ 10ms
  tolerate one replica loss per pool
  keep previous production model warm during rollout

Training cluster:
  reserve 20% for production retraining
  experimental jobs preemptible
  distributed jobs require gang scheduling
```

Headroom is a product feature. It is what turns a burst from an outage into a blip.

---

## Failure Domains and Effective Fleet Capacity

Fleet size is constrained by placement. Suppose steady peak requires 24 replicas and the service spans three zones. Placing eight per zone has zero zone-failure headroom: losing one zone leaves 16. To survive one-zone loss without violating safe utilization, each remaining zone must be able to carry half the required load, so the fleet needs at least 12 per zone, or 36 total, before accounting for rolling deploys. If a deploy temporarily removes 10% of replicas, the policy needs still more or must prohibit deploys near peak.

This is why a flat “30% headroom” rule can be wrong. Derive headroom from named events:

```text
required_healthy_capacity = peak_goodput / target_utilization
provisioned_capacity >= required_healthy_capacity
                       + largest_failure_domain
                       + simultaneous_maintenance_or_rollout
```

Correlated resources matter. Twelve model replicas in one zone backed by a feature cache in another do not form two independent failure domains. GPU type may be unavailable in the failover region; a previous-model fallback may use the same incompatible feature version; spot nodes may all be reclaimed together. Draw the full dependency and quota topology, then test the loss, not only the replica count.

Long procurement or quota lead times turn forecasting into architecture. Record a base, peak, launch, and failure scenario over the hardware lead-time horizon. Separate committed baseline from burst capacity and specify which demand can be delayed, downgraded, routed to an alternate model/hardware class, or rejected. Forecast error is inevitable; irreversible capacity with no degradation ladder makes it expensive.

## Admission Control and Degradation Economics

When offered load exceeds feasible capacity, the system must choose which work not to do. An unbounded queue makes that choice implicitly and badly: requests expire after consuming memory and perhaps compute. A bounded queue plus deadline-aware admission rejects work before its cost is sunk. Priority should be based on product semantics—checkout decisions before dashboard refreshes—not merely arrival order.

Common degradation steps are: serve a smaller/distilled model, reuse a bounded-staleness prediction, omit expensive feature groups, switch personalization to a deterministic baseline, move synchronous work to async, or shed low-priority tenants according to an explicit fairness policy. Each step changes quality and possibly risk, so it is a versioned policy with evaluation evidence, not an infrastructure improvisation. A fraud system may fail closed for a high-risk amount and fall back to rules for a low-risk one; a recommender can safely return popular items. Capacity and product policy meet at this boundary.

Measure the economic result as cost per **successful decision satisfying the SLO and quality policy**. Cheap outputs that time out, violate a guardrail, or are discarded by a downstream deadline are not savings. This denominator also makes fallback trade-offs visible: a smaller model may increase raw prediction volume per dollar but create enough business loss to be more expensive overall.

---

## Open-Loop Load Testing for ML Serving

Closed-loop load tests understate tail latency. In a closed-loop test, each client sends the next request only after receiving a response, so when the service slows down the test automatically reduces offered load. Real traffic does not politely slow down when your queue grows.

Use open-loop load generation: send requests at a fixed arrival schedule, independent of responses, and measure queueing collapse.

```yaml
load_test:
  mode: open_loop
  endpoint: fraud_authorization
  traffic_shape:
    warmup: 5m at 2k_rps
    steady: 15m at 12k_rps
    burst: 2m at 18k_rps
    recovery: 10m at 12k_rps
  request_mix:
    p50_feature_count: 35
    p95_feature_count: 60
    p99_large_input_share: 2%
  success_criteria:
    p99_latency_ms: <=120
    queue_wait_p99_ms: <=10
    timeout_rate: <=0.1%
    gpu_oom_count: 0
    feature_store_error_rate: <=0.05%
    fallback_rate: <=1%
  compare:
    baseline_model: fraud_classifier:v41
    candidate_model: fraud_classifier:v42
```

The load test should preserve production-like feature fanout, cache hit rate, input-size distribution, and logging path. A synthetic test that sends tiny feature vectors to a warmed model server measures a benchmark, not the production system.

Open-loop testing also exposes autoscaling truth. If replicas take 120 seconds to warm and the burst lasts 90 seconds, autoscaling is irrelevant for that burst; only warm headroom or admission control can protect p99.

---

## Failure Modes

**CPU-based autoscaling for GPU inference** scales too late or never because CPU is not the bottleneck. Defense: scale on queue depth, queue wait, GPU utilization, memory, and timeout rate.

**Average-latency planning** misses p99 violations caused by batching, fanout, and queueing. Defense: size against p95/p99 and test burst traffic.

**Cold-start capacity illusion** counts replicas that are starting but not yet serving. Defense: warm pools and predictive scale-up with warmup-time awareness.

**Feature-store bottleneck** buys more model servers while feature fetch is saturated. Defense: include feature QPS, cache hit rate, and fanout in capacity math.

**GPU OOM under variance** tests average input size but production sends long or large inputs. Defense: memory sizing with worst-case input bounds and headroom.

**Training GPU starvation** pays for accelerators that wait on data loading. Defense: measure data throughput, prefetch, shard correctly, and cache locally.

**Hyperparameter cost explosion** multiplies a cheap base run into a large bill. Defense: budget per search, early stopping, trial caps, and cost logging.

**Cluster deadlock from partial allocation** gives distributed jobs some but not all GPUs. Defense: gang scheduling.

**Shadow traffic overload** doubles inference and feature load during rollout. Defense: sample shadow, isolate resources, and include rollout overhead in headroom.

---

## Decision Framework

For any ML workload, answer:

1. Is this online, async, batch, or training? What freshness or deadline is required?
2. What is peak arrival rate, and what percentile latency or completion deadline matters?
3. What is the bottleneck: model compute, memory, feature fetch, storage I/O, network, or scheduler?
4. What is safe per-replica throughput at target utilization, not benchmark maximum?
5. How much warm capacity is required given model load time?
6. What is cost per prediction or cost per training run at expected utilization?
7. What headroom is required for failure, deploys, canaries, and bursts?
8. Which metric triggers scale-up before the user-visible metric breaks?
9. What happens when capacity is exhausted — queue, shed, fallback, or fail closed?
10. How will estimates be validated under open-loop, burst-shaped load?

A plan that cannot answer these is not capacity planning; it is hoping the cloud bill and p99 are friendly.

---

## Key Takeaways

1. ML capacity planning is still request rate, service time, concurrency, utilization, memory, bandwidth, and queueing — but the bottleneck resource is often accelerator memory or feature I/O.
2. Choose the serving regime first; batch is cheaper than online when freshness allows it.
3. Online inference sizing must use p99 latency, queue wait, safe utilization, and warmup time, not average throughput.
4. Dynamic batching trades latency for utilization; measure effective batch size and tail queue wait.
5. Accelerator memory is a hard limit; model weights, activations, caches, and input variance determine safe batch size.
6. Feature fetch fanout can dominate inference and must be included in QPS and latency budgets.
7. Training cost is accelerator-hours multiplied by experiment count; hyperparameter search is often the real bill.
8. Training throughput often bottlenecks on data I/O, not GPU FLOPs.
9. Shared GPU clusters need quotas, fair share, priority, checkpointing, gang scheduling, and idle-job enforcement.
10. Cost per prediction should include feature stores, logging, caching, and monitoring, not only model-server instances.
11. Open-loop load testing is required for tail-latency confidence; closed-loop tests hide queue collapse.
12. Cost per prediction rises directly when utilization falls; right-size hardware and measure CPU/GPU crossover.

---

## References

1. [Capacity Planning and Back-of-the-Envelope Estimation](../01-foundations/10-capacity-planning.md)
2. [Model Serving](./03-model-serving.md)
3. [Training Pipelines](./05-training-pipelines.md)
4. [The Tail at Scale](https://research.google/pubs/pub40801/) — Dean & Barroso, 2013
5. [Clipper: A Low-Latency Online Prediction Serving System](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw) — Crankshaw et al., NSDI 2017
6. [NVIDIA Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
7. [FinOps and Cost Engineering](../11-observability/06-finops-cost-engineering.md)
