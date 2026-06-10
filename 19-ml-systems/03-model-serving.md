# Model Serving

## TL;DR

Model serving turns trained artifacts into production predictions. The design space is shaped by latency, throughput, model size, hardware, feature freshness, rollout safety, and observability. A good serving system can load model versions, route traffic, batch requests, enforce timeouts, explain failures, and roll back independently from application code.

---

## Serving Modes

| Mode | Latency | Throughput | Use case |
|---|---:|---:|---|
| Batch scoring | Minutes to hours | Very high | Daily recommendations, churn scores |
| Online synchronous | Milliseconds to seconds | Medium | Fraud, ranking, personalization |
| Online asynchronous | Seconds to minutes | High | Enrichment, review queues |
| Streaming inference | Milliseconds to seconds per event | High | Abuse detection, anomaly detection |
| Edge inference | Local | Device-bound | Offline apps, privacy-sensitive features |

---

## Online Serving Architecture

```mermaid
flowchart LR
    CLIENT["Application"] --> GW["Prediction gateway"]
    GW --> FEAT["Feature fetch"]
    FEAT --> ROUTER["Model router"]
    ROUTER --> M1["Model server v1"]
    ROUTER --> M2["Model server v2"]
    ROUTER --> FALL["Fallback policy"]

    M1 --> LOG["Prediction log"]
    M2 --> LOG
    FEAT --> LOG
    LOG --> MON["Monitoring"]
```

The prediction log is essential. It should capture request metadata, model version, feature values or references, prediction, latency, and later label joins.

---

## Latency Budget

```text
Total p99 budget: 100 ms

Network ingress       10 ms
Auth/routing           5 ms
Feature lookup        25 ms
Model inference       40 ms
Post-processing       10 ms
Logging/egress        10 ms
```

If feature lookup consumes the whole budget, optimizing the model will not fix the user experience. Budget each step before choosing serving hardware.

---

## Model Versioning and Routing

```mermaid
flowchart TD
    REQ["Request"] --> R{"Routing policy"}
    R -->|"95%"| CHAMP["Champion model"]
    R -->|"5%"| CANARY["Canary model"]
    R -->|"shadow"| SHADOW["Shadow model"]
    CHAMP --> RESP["Response"]
    CANARY --> RESP
    SHADOW --> LOG["Log only"]
```

Common routing policies:

- Champion/challenger: compare production model against candidate.
- Canary: send small live traffic to candidate and watch guardrails.
- Shadow: run candidate without affecting response.
- Segment routing: send a model to a region, tenant, device class, or risk tier.
- Fallback: route to simpler model or rules when the primary path fails.

---

## Batching

Batching improves throughput but can increase tail latency.

| Strategy | Strength | Risk |
|---|---|---|
| No batching | Predictable latency | Low hardware utilization |
| Fixed batch | Simple capacity planning | Waits for batch to fill |
| Dynamic batching | Better utilization under variable load | More complex p99 behavior |
| Continuous batching | High GPU utilization for large models | Scheduler complexity |

Use batching when the model is compute-heavy and requests can wait briefly. Avoid it for extremely tight latency budgets unless the serving framework gives strong p99 controls.

---

## Autoscaling

Autoscale on serving-specific signals, not only CPU:

- Request rate.
- Queue depth.
- Inference latency.
- GPU utilization and memory.
- Model load time.
- Feature lookup latency.
- Timeout rate.

Large models make scale-from-zero risky because cold start can take minutes. Keep warm capacity for latency-critical models.

---

## Failure Modes

### Model Load Failure

A new artifact cannot be loaded because of incompatible runtime, missing dependency, wrong tensor shape, or corrupt artifact.

Mitigation: validate artifacts before promotion, use staged rollout, keep previous model loaded until the new model passes health checks.

### Feature Fetch Timeout

The model server is healthy but upstream feature retrieval fails.

Mitigation: enforce strict timeouts, define fallback features, use cached features when safe, and measure feature-store availability separately.

### Tail Latency Collapse

Average latency is fine, but p99 rises during bursts because queues grow faster than workers drain them.

Mitigation: queue limits, load shedding, admission control, separate pools for expensive models, and capacity tests at expected burst size.

### Silent Wrong Model

The service deploys a valid model artifact that belongs to the wrong dataset, segment, or feature schema.

Mitigation: require model cards or metadata checks, schema compatibility gates, artifact hashes, and model-version logging on every prediction.

---

## Deployment Patterns

| Pattern | Use when | Watch out for |
|---|---|---|
| Blue-green | Need fast rollback of whole model service | Double capacity |
| Canary | Want gradual live validation | Weak signal at low traffic |
| Shadow | Need compare without user impact | Shadow feature load can still affect dependencies |
| Multi-armed bandit | Optimization objective is measurable quickly | Can exploit short-term proxy metrics |
| Rules fallback | Model can fail open or fail closed safely | Rule path may drift from model path |

---

## Operational Metrics

| Layer | Metrics |
|---|---|
| Request | QPS, p50/p95/p99 latency, timeout rate, error rate |
| Queue | Queue depth, wait time, dropped requests |
| Model | Inference time, model load time, version, memory usage |
| Hardware | CPU/GPU utilization, GPU memory, accelerator errors |
| Features | Lookup latency, freshness, miss rate |
| Quality | Online guardrails, delayed labels, drift, calibration |

---

## Key Takeaways

1. Model serving is a production service with model-specific failure modes.
2. Roll out model artifacts independently from application code.
3. Prediction logs are required for monitoring, debugging, and retraining.
4. Batching improves throughput but must be managed against p99 latency.
5. Always design fallback behavior before deployment.

---

## References

1. [TensorFlow Serving: Flexible, High-Performance ML Serving](https://arxiv.org/abs/1712.06139)
2. [KServe Documentation](https://kserve.github.io/website/)
3. [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
4. [Hidden Technical Debt in Machine Learning Systems](https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
