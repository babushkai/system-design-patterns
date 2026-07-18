# GPU Inference Internals

## TL;DR

LLM inference contains two resource regimes behind one API. *Prefill* processes many prompt tokens in parallel and often approaches the compute or attention-kernel roof; *decode* advances each active sequence by one token and is commonly constrained by weight and KV-cache traffic. The useful model is the roofline bound:

$$
t_{step} \ge \max\left(\frac{F}{\eta_c C_{peak}},
                         \frac{D_{HBM}}{\eta_b B_{HBM}},
                         \frac{D_{link}}{B_{link}}\right),
$$

where $F$ is useful work, $D$ is data movement, and the $\eta$ terms capture the fact that an implementation never sustains every datasheet peak simultaneously. Batching, quantization, paged KV memory, specialized attention kernels, speculation, parallelism, and phase disaggregation change one or more terms in that bound. Capacity decisions should begin with this ledger and end with measured goodput on the real prompt, output, cache-hit, and concurrency distribution.

---

## The Hardware Ledger

Record five quantities for the exact accelerator and form factor: usable HBM capacity, sustained HBM bandwidth, dense tensor-core throughput at the deployed precision, collective/link bandwidth for the chosen topology, and power. Do not mix SXM and PCIe variants or sparse and dense tensor-core figures. Vendor tables often mark structured-sparsity throughput with a footnote; a dense model receives the dense rate.

As a worked example, NVIDIA publishes 80 GB HBM3, 3.35 TB/s peak HBM bandwidth, and 900 GB/s aggregate NVLink bandwidth for H100 SXM. Its published BF16 and FP8 tensor-core figures include structured sparsity; the corresponding dense arithmetic rates are half those values. H200 retains the Hopper compute architecture while increasing HBM capacity and bandwidth, illustrating why memory-system changes can improve decode even when peak arithmetic is unchanged. These are example inputs, not timeless constants; attach the datasheet revision and measured sustained efficiencies to a capacity model.

The number that organizes everything else is **machine balance**: peak compute divided by memory bandwidth.

```text
H100 example, dense BF16: 989e12 FLOPS / 3.35e12 B/s ≈ 295 FLOPs per byte
H100 example, dense FP8:  1979e12 / 3.35e12          ≈ 590 FLOPs per byte
```

A workload below machine balance cannot saturate both peak arithmetic and memory bandwidth. The roofline identifies the limiting resource, while measured efficiency terms account for kernel shape, occupancy, cache behavior, launch overhead, synchronization, and power clocks. The bound is diagnostic, not a latency prediction by itself.

The H100-to-H200 example separates two benefits: more capacity admits larger models, batches, or contexts; more bandwidth raises the roof for bandwidth-bound decode. Neither guarantees proportional application speedup when collectives, KV traffic, host scheduling, or compute dominate.

---

## Roofline: Why Decode Is Memory-Bound

A transformer forward pass performs roughly `2 × P` FLOPs per token, where `P` is the parameter count (each parameter participates in one multiply and one accumulate). The bytes side depends on batch size, because weights are read from HBM *once per forward pass* and shared across every sequence in the batch.

**Decode at batch size 1.** Generating one token for one sequence:

```text
Llama-3.3-70B in FP8: weights ≈ 70 GB

FLOPs  = 2 × 70e9            = 140 GFLOPs
Bytes  = 70e9 (all weights)  + KV cache (see next section)
Arithmetic intensity ≈ 140e9 / 70e9 = 2 FLOPs/byte

Machine balance (H100, FP8): 590 FLOPs/byte  →  intensity is ~295× below balance
```

The GPU spends its time streaming weights, not multiplying. The tokens/sec ceiling follows directly:

```text
Single-stream ceiling = bandwidth / bytes per token
                      = 3.35 TB/s / 70 GB ≈ 48 tokens/sec
```

The quotient is an optimistic weight-only upper bound, not a guaranteed 48 tokens/s. KV reads, embedding/output layers, non-matrix operations, allocator metadata, kernel inefficiency, and clocks add traffic or reduce sustained bandwidth. An observed result above the naive bound signals that an assumption changed—weights were shared across a batch, compressed, cached at another level, sparsely activated, or not all bytes were read—not that the roofline was violated.

**Decode at batch size B.** The same weight read now serves $B$ sequences: FLOPs scale with $B$, while weight bytes do not. With $b_w$ bytes per stored weight, weight-only arithmetic intensity is $2B/b_w$ FLOPs per byte. On the stated H100 example this reaches machine balance around $B\approx295$ for both dense BF16 and dense FP8. This is not an admission target: KV reads, activation traffic, collectives, padding, and the latency SLO alter or preclude that crossover.

**Prefill.** Processing $N$ prompt tokens performs approximately `2 × P × N` dense-model FLOPs while amortizing weight reads across tokens. Weight-only intensity therefore grows roughly with $N$. Short or skinny prefills may remain launch- or bandwidth-limited, while long prompts introduce attention work and activation traffic that the simple parameter-count model omits. The important asymmetry is empirical: prefill and decode often occupy different roofline regions, so co-scheduling policy must control their interference.

Speculative decoding is the same arithmetic exploited from another angle: a small draft model proposes k tokens, and the large model *verifies all k in one forward pass* — one weight read amortized over k tokens, exactly like batching, but within a single stream. That is why speculation helps most at low batch (spare compute everywhere) and fades at high batch (compute already spoken for).

---

## The KV-Cache Ledger

Attention requires every past token's key and value vectors. Engines cache those vectors so each new decode step projects only the new token instead of re-running the prior prefix. With caching, dense attention for one new token still scans a prefix of length $S$, so attention work is linear in $S$ per step and quadratic across a generated sequence. Re-running the full dense transformer over the growing prefix at every step adds repeated projection work and can make total attention work cubic in generated length. The cache therefore removes redundant computation; it does not make dense attention linear over the whole generation. It also competes with weights for both HBM *capacity* and HBM *bandwidth*.

Per-token size for a grouped-query attention (GQA) model:

```text
KV bytes/token = 2 (K and V) × layers × kv_heads × head_dim × bytes/param

Llama-3-70B (80 layers, 8 KV heads, head_dim 128, BF16):
  2 × 80 × 8 × 128 × 2 ≈ 320 KB per token
  → a 128K-context sequence holds ≈ 40 GB of aggregate KV across its replica.
```

The logical amount is sharded according to the serving parallelism; the BF16 weights themselves already require multiple 80 GB devices. Capacity planning therefore applies the same placement contract to both weight and KV shards rather than treating 40 GB as a stand-alone single-device deployment.

(The platform-level admission, paging, and fragmentation consequences are developed in [Model Serving](../16-ml-systems/03-model-serving.md); this chapter owns the byte equation.)

The capacity story is well known; the *bandwidth* story is the one that surprises. During decode, attention reads prior KV state in addition to weights. Across a sharded replica, an illustrative logical batch of 32 sequences at 16K tokens with the sizing above contains `32 × 16,384 × 320 KB ≈ 168 GB` of aggregate KV, compared with about 70 GB of aggregate FP8 weights. The physical traffic per device depends on tensor/sequence parallelism, KV layout, kernels, and cache hierarchy, but the context-dependent term does not amortize like dense weights. GQA, latent KV representations, and lower-precision KV can therefore change throughput as well as capacity. Prefix reuse avoids recomputing the shared prefill; it does not eliminate decode-time reads of resident KV.

---

## Kernels: FlashAttention and the Memory Hierarchy

The roofline logic recurses one level down. Inside the GPU, the hierarchy is HBM (TB/s, tens of GB) → L2 (~50 MB) → SRAM/shared memory per streaming multiprocessor (~228 KB on H100, ~20 TB/s aggregate). Kernels are fast when they keep intermediate results in SRAM and touch HBM once.

Naive attention materializes the N×N score matrix in HBM:

```text
One layer, one head, N = 8,192, BF16:
  S = QKᵀ: 8192² × 2 B ≈ 134 MB written to HBM, read back for softmax,
  written again, read again for ×V — ≈ 0.5 GB of HBM traffic per head,
  × 64 heads × 80 layers ≈ multiple TB per single forward pass. Unusable.
```

FlashAttention tiles Q, K, and V into on-chip memory and computes softmax incrementally, avoiding materialization of the full $N\times N$ score and probability matrices in HBM. The attention arithmetic remains quadratic in sequence length for dense attention; the gain comes from lower HBM I/O and better tiling, not from changing the algorithm to linear-time attention. Hardware-specific versions further overlap matrix and softmax work and support lower-precision paths.

Two other kernel-level facts matter operationally:

- **Kernel launch and host scheduling are decode taxes.** A decode iteration invokes many small operations, so host launch latency can become visible at small batches. Graph capture and operator fusion amortize that control overhead, but dynamic shapes, adapters, and sampling features can reduce graph reuse.
- **Kernel compatibility is part of model qualification.** Paged-KV layout, GQA/MLA shape, quantization, speculation, and accelerator generation determine which fused path is available. A fallback kernel may be correct yet move substantially more data, so record kernel selection in performance traces.

---

## Batching Economics

For dense decode, a first-order step model is

$$
D_{step}(B,S) \approx W + B\,S\,K + D_{activation},
\qquad F_{step}(B) \approx 2PB,
$$

where $W$ is weight bytes, $B$ active sequences, $S$ their effective context length, and $K$ KV bytes per token. Weight traffic amortizes with $B$, while per-sequence KV traffic does not. Total throughput initially rises quickly, then approaches a compute, KV-bandwidth, or scheduler limit; per-stream inter-token latency can deteriorate throughout.

Continuous batching keeps the active set populated by admitting and retiring sequences at iteration boundaries. It improves utilization but introduces scheduling choices around priority, adapter locality, KV residency, and long prefills. The economic objective is accelerator cost divided by SLO-compliant output tokens, including rejected, preempted, and cancelled work. Hourly accelerator price and sustained efficiency are deployment inputs, not constants suitable for a universal example.

Benchmark across concurrency until the goodput curve reaches its knee. The selected operating point must leave headroom for length variance, failures, rollout surge, and cache misses; the maximum-throughput point is usually an overload condition for an interactive service.

---

## Quantization: Change the Byte and Compute Ledger

Lower precision reduces stored bytes only to the extent that scales, zero points, grouping metadata, padding, and dequantization work permit. A native low-precision tensor-core path can reduce both memory traffic and arithmetic time; weight-only quantization primarily changes weight traffic and adds dequantization; KV quantization changes the context-dependent term. Consequently, the same artifact can accelerate low-batch decode while providing little gain—or a regression—for compute-bound prefill.

For a $b$-bit weight representation, the ideal weight footprint is $Pb/8$, but capacity planning uses the serialized artifact plus runtime workspaces. The ideal bandwidth speedup from halving $W$ is bounded by the fraction of step time attributable to weight traffic. Once KV, collectives, sampling, or compute dominates, Amdahl's law caps the gain.

The governing rule from [Model Serving](../16-ml-systems/03-model-serving.md) applies with extra force here: aggregate language-model loss can hide task- and slice-specific quantization damage. Precision, calibration data, grouping, outlier handling, kernels, and workload all affect the result. Treat each quantized model/runtime tuple as a new artifact and qualify task quality, safety, latency, memory, and fallback behavior ([LLM Evaluation](./10-llm-evaluation.md)).

---

## Parallelism: TP, PP, EP

When the model outgrows one GPU — in capacity or in required ceiling — you split it. The three axes have sharply different communication profiles:

**Tensor parallelism (TP)** slices layer operations across devices and introduces frequent collectives on the token-critical path. It is effective only while the selected topology's collective latency and bandwidth remain small relative to local compute and memory work. TP usually stays within the fastest coherent or switched accelerator domain. The ideal local weight-traffic reduction is divided by TP degree; collective and synchronization costs prevent linear scaling.

**Pipeline parallelism (PP)** assigns contiguous layer blocks to stages and transfers activations across boundaries. It tolerates slower links than TP but introduces bubbles, stage imbalance, and additional buffering. Decode's token dependency makes microbatching harder than in training. Large deployments often compose TP inside a fast domain with PP across domains, but the degrees follow the measured topology and model partition rather than a fixed node size.

**Expert parallelism (EP)** distributes mixture-of-experts parameters and routes tokens through an all-to-all exchange. Only active experts perform arithmetic for a token, but inactive expert weights still consume distributed capacity. Small batches underutilize experts; large batches expose routing skew and communication. Capacity and latency models therefore need total parameters, active parameters per token, expert placement, routing distribution, and straggler behavior—not only the advertised active-parameter count.

---

## Disaggregation: Splitting Prefill from Decode

Prefill and decode can interfere when co-scheduled: a large prefill occupies compute and memory resources while active decodes wait. Chunked prefill limits the blocking interval. *Disaggregated serving* instead places phases on separate pools and transfers the prompt KV state between them.

The transfer is justified only when saved phase interference and independent placement exceed serialization, network, admission, and cache-management costs:

$$
t_{transfer} = \frac{D_{KV}}{\eta_n B_{network}} + t_{setup}.
$$

Layer-wise streaming can overlap transfer with decode startup. Tiered KV storage can avoid recomputation for reusable prefixes, but moves the bottleneck to lookup, network/storage bandwidth, eviction, and security-domain-aware cache identity.

Disaggregation also permits heterogeneous pools when one phase benefits more from compute and the other from bandwidth/capacity. It adds queues, failure boundaries, KV routing, and stranded-capacity risk, so compare it with co-located chunked prefill under the same workload. Sizing the pools is covered in [ML Capacity & Cost Planning](../16-ml-systems/14-ml-capacity-cost-planning.md); the operator view is in the [LLM Inference Platforms case study](../08-case-studies/13-llm-inference-platforms.md).

---

## Constrained Decoding Has a Separate Control Cost

Structured outputs mask invalid next tokens according to grammar state. Efficient runtimes compile grammars, cache token masks, and overlap mask construction with model execution; deterministic spans may be advanced without a full model step. The residual cost depends on schema compilation, tokenizer interaction, grammar ambiguity, batch diversity, CPU/GPU synchronization, and cache reuse. Measure cold and warm paths separately, and key compiled artifacts by trusted schema and tokenizer digests. A highly dynamic schema can make control-plane compilation visible even when per-token masking is small.

---

## Measuring It: The Metrics That Map to the Roofline

Each serving metric corresponds to a hardware regime, which is what makes them diagnostic rather than decorative:

- **TTFT (time to first token)** — includes queue and prefill; diagnose it against prompt length, arithmetic intensity, attention shape, launch/communication overhead, and prefix reuse rather than assuming one bottleneck.
- **TPOT / ITL (time per output token / inter-token latency)** — exposes decode scheduling and the active weight/KV/collective ledger; bandwidth often dominates at low batch, while compute, KV traffic, or communication can dominate elsewhere.
- **Throughput (tok/s per GPU)** — the amortization metric; meaningless without the latency it was bought at.
- **Goodput** — qualified completions or tokens per unit time within the declared SLO; interpret it with offered load, rejection, cost, and headroom.

Benchmark with the workload's joint prompt/output/context/concurrency distribution and its real cache policy. Useful tools can replay length distributions and report percentile TTFT/ITL; standardized suites provide cross-system reference points. Fixed-length prompts hide scheduler interference and KV pressure, assumed cache-hit rates hide cold-path capacity, and means hide user-visible stalls. Use open-loop load to expose queueing and report both admitted and rejected work.

---

## Failure Modes

**KV pressure cascades.** Memory fills with cached sequences → the scheduler preempts one, evicting its KV → the sequence later resumes with a full re-prefill → which adds compute load and evicts someone else. Symptom: throughput collapses and TTFT spikes under load that "should fit." Diagnose preemption and recomputation, tighten admission or residency, and compare added capacity with a separately qualified lower-precision KV/runtime path.

**TP across a slow interconnect.** Tensor parallelism over a link whose collectives are slow relative to local work puts frequent layer-level communication on the token-critical path; the deployment remains correct at a fraction of expected throughput. Map TP and PP only after measuring the actual collective and activation-transfer schedule on each topology.

**The quantization eval gap.** The INT4 model matches perplexity, passes the smoke test, ships — and a week later math-heavy or code-heavy traffic shows a regression no serving metric caught. Quality gates must be task evals, run per quantization artifact, not per model family.

**Throughput tuning past the SLO.** Benchmarks reward batch sizes and chunk sizes that a p99 inter-token SLO forbids. Any tok/s figure quoted without its latency percentile is a red flag in a design review.

**Long-context surprise.** A feature raises average context from 4K to 64K; per-token KV traffic grows 16×, the decode ceiling drops, and the fleet sized for 4K saturates. Context-length distribution is a first-class capacity input, same as QPS.

**Straggler experts.** In MoE serving, a hot expert (or a slow GPU hosting one) gates every token in the all-to-all; p99 latency degrades fleet-wide. Requires per-expert load metrics and redundant placement of hot experts.

---

## Decision Framework

| Situation | Reach for |
|---|---|
| Single-stream latency matters most | Reduce weight/KV bytes, evaluate speculation, and use only enough parallelism to meet the bound |
| Throughput/cost matters most | Continuous batching near the SLO knee; compare dense and MoE fleet economics |
| Long contexts dominate | Smaller KV representations, qualified KV precision, prefix reuse, and bandwidth/capacity headroom |
| Model fits one GPU | Capacity does not require TP; compare one device with TP only if aggregated bandwidth or compute improves the target latency enough to pay collective overhead |
| Model exceeds one accelerator but fits a fast link domain | Benchmark TP within that domain |
| Model exceeds one fast link domain | Compose TP/PP from measured collective and activation-transfer costs; add EP for MoE |
| Mixed prefill/decode interference | Compare co-located chunking with disaggregation; split only when reduced interference and independent placement exceed KV transfer, extra queues, and stranded capacity |
| Deciding whether to buy compute or bandwidth | Use phase-specific roofline and goodput benchmarks; do not infer from peak FLOPS alone |

---

## Key Takeaways

1. **Prefill and decode often occupy different roofline regions, but neither has one universal bottleneck.** Compute useful work, bytes, and communication for the actual shapes before trusting a benchmark.
2. **`bandwidth ÷ bytes per step` is an upper bound, not a benchmark result.** Include weights, KV, activations, collectives, metadata, and sustained-efficiency factors.
3. **Batching, speculation, and quantization can improve useful work per byte through different mechanisms and with different acceptance, control, and quality costs.**
4. **KV cache is a bandwidth problem, not just a capacity problem.** Long contexts increase the byte term; GQA, latent representations, and qualified lower-precision KV can reduce it when the runtime supports them.
5. **Parallelism follows the interconnect**: TP needs frequent fast collectives, PP trades bandwidth for bubbles, and EP adds routed all-to-all plus skew.
6. **Goodput is a primary capacity objective, not a complete report.** State offered load and SLOs, then report goodput with rejection, cost, headroom, bottleneck metrics, means, and tail percentiles.
7. **Quantization is a new model and runtime artifact.** Its speedup follows Amdahl's law, and its release requires task, safety, and serving evaluation.

## References

- Williams, Waterman & Patterson — *Roofline: An Insightful Visual Performance Model* (2009)
- Dao et al. — *FlashAttention* (2022), *FlashAttention-2* (2023); Shah et al. — *FlashAttention-3* (2024)
- Kwon et al. — *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, 2023)
- Zhong et al. — *DistServe: Disaggregating Prefill and Decoding* (2024)
- Qin et al. — *Mooncake: A KVCache-centric Disaggregated Architecture* (2024)
- DeepSeek-AI — *DeepSeek-V3 Technical Report* (2024)
- Leviathan et al. — *Fast Inference via Speculative Decoding* (2023)
- Dong et al. — *XGrammar: Flexible and Efficient Structured Generation* (2024)
- [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/) and [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/) — worked-example hardware inputs and sparsity footnotes
- MLPerf Inference results (mlcommons.org); vLLM, SGLang, TensorRT-LLM documentation
- [Attention & Transformers](../09-whitepapers/15-attention-transformers.md) — the architecture this chapter serves
