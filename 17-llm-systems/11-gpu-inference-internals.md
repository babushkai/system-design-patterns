# GPU Inference Internals

## Overview {#tl-dr}

A model server processes a request in two phases. During prefill, it processes prompt tokens in parallel and can approach its compute limit. During ordinary autoregressive decode, each step advances an active sequence by one token. Reading weights and cached keys and values often limits that step. The roofline bound relates these limits:

$$
t_{step} \ge \max\left(\frac{F}{\eta_c C_{peak}},
                         \frac{D_{HBM}}{\eta_b B_{HBM}},
                         \frac{D_{link}}{B_{link}}\right),
$$

Here $F$ counts floating-point operations. Divide it by the sustained compute rate to estimate compute time. The other terms estimate the time to move bytes through device memory and across the interconnect. The slowest term gives a lower bound on step time. For example, if memory traffic dominates decode, buying more peak compute alone may do little for the gap between tokens. Check that prediction under the expected concurrency and context lengths.

---

## The Hardware Ledger

Start with one accelerator SKU and one deployment topology. Record usable memory after runtime initialization, sustained memory bandwidth, dense compute throughput at the deployed precision, collective bandwidth and latency, and power. A different form factor can change several inputs at once.

For an H100 SXM, NVIDIA lists 80 GB of HBM3, 3.35 TB/s of peak HBM bandwidth, and 900 GB/s of aggregate NVLink bandwidth. Its headline BF16 and FP8 tensor-core figures include structured sparsity. Use half those figures for this dense example. [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)

Dividing peak arithmetic throughput by peak memory bandwidth gives the arithmetic intensity needed to reach both limits:

```text
H100 example, dense BF16: 989e12 FLOPS / 3.35e12 B/s ≈ 295 FLOPs per byte
H100 example, dense FP8:  1979e12 / 3.35e12          ≈ 590 FLOPs per byte
```

At lower intensity, moving bytes takes longer than executing the arithmetic in this idealized model. Compare that prediction with measured kernel time and memory traffic. Small shapes, launch stalls, and collectives can leave both resources underused.

H200 illustrates the difference between memory capacity and bandwidth. Extra capacity can admit a larger batch. Extra bandwidth can shorten a decode step dominated by memory reads. Measure each effect separately when deciding whether a hardware change will meet a latency target.

---

## Roofline: Why Decode Is Memory-Bound

For a dense model with about $P$ active weights in its projections, count roughly $2P$ multiply-add FLOPs per token. Attention over the retained context adds work. As a first estimate, assume the weights stream from HBM once per batch step and all sequences share that read.

**One active sequence.** With one byte per weight, use the following idealized budget:

```text
Llama-3.3-70B in FP8: weights ≈ 70 GB

FLOPs  = 2 × 70e9            = 140 GFLOPs
Bytes  = 70e9 (all weights)  + KV cache (see next section)
Arithmetic intensity ≈ 140e9 / 70e9 = 2 FLOPs/byte

Machine balance (H100, FP8): 590 FLOPs/byte  →  intensity is ~295× below balance
```

If every decode step reads 70 GB of weights on one H100 SXM, the weight-only bandwidth ceiling is

```text
Single-stream ceiling = bandwidth / bytes per token
                      = 3.35 TB/s / 70 GB ≈ 48 tokens/sec
```

This gives about 48 tokens/s before KV reads, launch overhead, other operations, and bandwidth losses. It also assumes the weights, cache, and workspaces fit on that device. A faster measurement calls for checking the assumed bytes read per output token, the number of devices, and whether proposals allowed several tokens to advance in one step.

**A batch of $B$ sequences.** Each step now produces $B$ tokens. With $b_w$ bytes per weight, weight-only intensity rises to $2B/b_w$. For dense FP8 the H100 balance above implies $B\approx295$. Dense BF16 reaches a similar crossover because its weights are twice as wide and its peak arithmetic rate is about half. KV traffic and per-stream deadlines may bind much earlier.

**Prefill.** Processing $N$ new positions together brings approximately $2PN$ projection FLOPs. The same weights serve many positions. Long contexts add attention work, and short prompts can leave the device underoccupied. Profile both phases with the prompt and context lengths the scheduler actually groups together.

**Speculative decode.** A drafter proposes several future tokens. The target evaluates candidate positions together and can advance more than one token in a verification step. Speedup depends on accepted tokens per round, draft cost, and how verification competes with other sequences. Measure delivered tokens per second at the required concurrency. The original speculative sampling algorithm preserves the target distribution under its acceptance and correction rules. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

---

## The KV-Cache Ledger

Full causal attention uses past keys and values when computing the next position. Caching them avoids projecting the old positions again. A new token at context length $S$ still attends over $S$ positions. Across $N$ newly generated tokens after a prompt of length $S_0$, the number of past positions visited scales with $NS_0+N(N-1)/2$ per head.

Count both K and V using the precision of the cache elements. For full attention with identical grouped-query layers:

```text
KV bytes/token = 2 × layers × kv_heads × head_dim × bytes/cache element

Llama 3.1 70B, BF16 KV:
2 × 80 × 8 × 128 × 2 = 327,680 bytes per token = 320 KiB
131,072 tokens × 320 KiB = 40 GiB of logical KV per sequence
```

The 128K example uses Llama 3.1. Its 70B model supports that context size. Place the weights, each rank's actual KV shards or replicas, and workspaces before estimating concurrent sequences. [Meta Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)

For models that mix full attention with sliding-window attention, calculate retained bytes per layer. A full-attention layer retains the entire prefix. A sliding-window layer needs only its configured window during decode. Sum the layer estimates, then measure allocated pages under load before setting admission limits. Block rounding and padding can increase the physical allocation. [vLLM hybrid KV cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)

Paged allocation, prefix sharing, and fragmentation change the physical footprint. Distinguish logical bytes needed by the sequences from physical pages reserved by the allocator. [Model Serving](../16-ml-systems/03-model-serving.md)

Now take 32 unrelated sequences at 16,384 tokens each with the same BF16 cache. Their aggregate KV is $32\times16{,}384\times320$ KiB, or 160 GiB (about 172 GB). That exceeds the roughly 70 GB of FP8 weights in the earlier example. This is a logical footprint. Measure actual HBM traffic because sharding, cache hits, shared prefixes, and the attention kernel determine how often each device reads a page. Prefix reuse saves work during prefill. Subsequent decode still attends to earlier positions. [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

---

## Kernels: FlashAttention and the Memory Hierarchy

A conventional attention implementation can write the score matrix to HBM, read it for softmax, then write and read the probabilities for the value product. For a square 8,192-position BF16 matrix, each such transfer is 128 MiB per head. Four transfers across 64 heads and 80 layers would move about 2.5 TiB for those intermediates alone. This illustration assumes full matrices and separate operations.

FlashAttention instead computes tiles in on-chip memory and updates the softmax statistics as tiles arrive. Dense attention still performs quadratic arithmetic in sequence length. Avoiding the full intermediate matrix in HBM changes the time spent moving data. [FlashAttention](https://arxiv.org/abs/2205.14135)

A latency regression after a runtime change can come from kernel selection even when the weights are identical. Compare traces for the same shapes, attention layout, precision, and adapter. Check whether the fused kernel ran, how much HBM traffic it generated, and whether graph capture still covered common decode shapes. If host launch gaps dominate, tuning device bandwidth will have little effect.

---

## Batching Economics

Estimate bytes moved by a full-attention dense decode batch and the projection FLOPs for that batch:

$$
D_{step}(B,S) \approx W + B\,S\,K + D_{activation},
\qquad F_{step}(B) \approx 2PB,
$$

Here $W$ counts weight bytes, $B$ active sequences, $S$ mean retained length, and $K$ KV bytes per cached token. The estimate assumes one logical read of each sequence's past keys and values, uniform KV shape, and no sharing across sequences. Add attention FLOPs for long contexts. Use hardware counters to determine how much traffic reached HBM.

Continuous batching replaces completed sequences between steps. Raising the batch cap can increase output throughput while lengthening each step for every stream. At a fixed arrival trace, sweep the cap and record offered requests, successful completions within the SLO, and the longest content gaps. Divide accelerator spend during the run by qualified completions to include time spent on failed work.

Repeat near the selected operating point with longer prompts, cache loss, and reduced fleet capacity. Keep enough room below saturation for the required failure case.

---

## Quantization: Change the Byte and Compute Ledger

A $b$-bit weight representation has a raw footprint of $Pb/8$ bytes. Measure resident weights, scales, zero points, padding, and runtime workspaces after load. Check that the deployed hardware has an efficient kernel for the actual shape and precision.

Weight-only quantization can reduce low-batch decode time even when prefill gains little. KV quantization addresses a different term that grows with retained context. If weight reads consume half the baseline step and their time falls by half, the new step takes 75% of baseline, a speedup of about 1.33× under otherwise identical conditions.

Qualify the specific model, calibration data, precision, and kernel revision together. Test long-context retrieval, code, structured output, and other high-value slices for quality as well as speed. Keep a rollback target that fits the deployed hardware. [Model Serving](../16-ml-systems/03-model-serving.md) · [LLM Evaluation](./10-llm-evaluation.md)

---

## Parallelism: TP, PP, EP

Choose a partition by asking what crosses the link on each generated token.

**Tensor parallelism** splits layer operations across devices. Each rank reads a smaller weight shard, then exchanges intermediate results through frequent collectives. Measure collective latency at the message sizes used by decode. Aggregate link bandwidth alone says little about the cost of many small synchronizations.

**Pipeline parallelism** assigns layers to stages and sends activations to the next stage. A single sequence follows the stages in order. Concurrent microbatches can fill the pipeline if the stages are balanced. Compare idle time at each stage with transfer time before increasing pipeline depth.

**Expert parallelism** routes tokens to sharded experts in a mixture-of-experts model. Total expert weights determine memory capacity. The activated experts and their token counts determine work in one step. Inspect the busiest rank and time in token exchange. Uneven routing can delay a whole batch even when average expert utilization is low.

---

## Disaggregation: Splitting Prefill from Decode

If long prefills delay active decode, first measure how much chunking restores stream latency in one pool. Splitting the phases gives each its own queue and capacity policy. It also moves KV onto the decode worker's critical path.

For $D_{KV}$ bytes over an effective path bandwidth $\eta_n B_{network}$, a simple model for a complete handoff is

$$
t_{transfer} = \frac{D_{KV}}{\eta_n B_{network}} + t_{setup}.
$$

For example, 4 GiB transferred at a sustained aggregate 40 GiB/s takes 100 ms before setup or queueing. Overlap can hide some of that time from the client. Parallel shard transfers can also compete for the same switch. Measure the time until the decode worker can actually advance and the bytes still in flight after cancellation.

Compare chunked and split pools under the same hardware budget and arrival trace. Include a failed handoff and a period of skewed phase demand. A split pool that scales well at the average mix may strand capacity when most new work is decode. [ML Capacity & Cost Planning](../16-ml-systems/14-ml-capacity-cost-planning.md) · [LLM Inference Platforms](../08-case-studies/13-llm-inference-platforms.md)

---

## Constrained Decoding Has a Separate Control Cost

A grammar-constrained decoder tracks which next tokens are legal. Compilation and token-mask construction can add CPU time before the GPU advances. Some engines can skip sampling for deterministic spans. Compare cold-schema, warm-schema, and mixed-schema requests at the same output length. Key compiled artifacts by schema, tokenizer, and relevant runtime options. Report compilation and per-step overhead separately. [XGrammar](https://arxiv.org/abs/2411.15100)

---

## Measuring It: The Metrics That Map to the Roofline

Use the client trace to choose where to look, then use the device trace to test the explanation.

| Observation | Next measurement |
|---|---|
| First visible content is late | Separate queue age, uncached prefill time, and transport. Compare within the same prompt-length band. |
| Streams pause between updates | Compare client content gaps with engine step latency. Check competing prefills, active KV length, collectives, and buffering. |
| More tokens per GPU but fewer users finish | Count rejected, cancelled, and late requests at the same offered arrival rate. |
| One kernel or topology wins a synthetic test | Replay production-shaped arrivals with the same warm and cold prefix mix. Measure quality, latency, and spend per qualified completion. |

Issue benchmark arrivals independently of completions when testing overload. A client that waits for each answer before issuing another request reduces offered load as latency rises. Include cold starts and burst lengths in the report so the next operator can reproduce the result.

---

## Multi-Tenant Isolation and Accelerator State

A running replica holds tenant-derived data in KV pages, activations, host buffers, and optional storage tiers. Trace where each copy goes during normal operation, transfer, drain, and crash collection.

Bind a cache handle to its access domain and allocation generation. Check both before dereferencing a page. When a block returns to the allocator, fence old references before assigning it to another tenant. Restrict profiler output and dumps that can contain payloads.

Choose the hardware boundary the threat model requires. Supported MIG instances partition selected compute and memory resources. Driver, management privileges, networking, and telemetry still need separate controls. Verify the exact SKU, partition mode, and teardown procedure before accepting workloads from a different trust domain. [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/)

Exercise reassignment as a complete lifecycle:

```text
admit tenant/security domain
  -> allocate qualified device or partition
  -> load verified model/runtime artifacts
  -> serve with domain-bound KV and telemetry
  -> drain and fence new work
  -> delete/demote cache objects and revoke directory generations
  -> perform the platform's qualified reset/sanitization transition
  -> reassign only after verification
```

Retain evidence that references were revoked and that the required reset or sanitization completed. Apply the same expiry and access policy to off-device copies. A failed teardown leaves the device unavailable for reassignment until an operator can establish its state. [Agent Inference](./12-agent-inference.md) · [LLM Infrastructure](./05-llm-infrastructure.md)

---

## Failure Modes

| Symptom | Likely mechanism | Test and response |
|---|---|---|
| Queue age rises with recomputed tokens | KV eviction creates more prefill work, which displaces active sessions | Correlate preemptions with resumed prefixes. Cap admitted growth and check that useful completions recover. |
| Adding a TP rank slows decode | Small collectives dominate the token path | Profile message sizes and per-rank wait. Try fewer ranks or a faster link domain. |
| Quantized rollout is fast but users retry | A difficult quality slice regressed | Replay affected tasks against the prior artifact and roll back on the product gate. |
| Long prompts change a stable fleet's latency | Full-attention KV reads grow with retained length | Compare traffic by context band. A 4K-to-64K increase scales that layer's logical KV read by 16×. Measure whole-step impact before buying capacity. |
| One expert stalls many requests | Tokens concentrate on a rank or its link slows | Compare per-expert load and rank duration. Test rebalancing or replication where memory allows. |
| Cache hits return inconsistent output after worker restart | An old handle may refer to a reassigned block | Disable affected reuse, check generation and ownership validation, and exercise eviction and restart before re-enabling it. |

---

## Decision Framework

| Design decision | Evidence required |
|---|---|
| Keep a model on one device or introduce TP | Establish memory fit and single-stream latency first. Include added collective time when comparing rank counts. |
| Increase batching | Show more completions per accelerator at the same arrival mix, quality gate, and tail-latency targets. |
| Change precision or KV format | Demonstrate savings in the limiting byte term, kernel support, and acceptable quality for long contexts. |
| Split a model across link domains | Compare measured TP collectives with pipeline transfers and idle stages. Include expert exchange for MoE. |
| Separate prefill from decode | Show that saved interference and independent scaling pay for KV handoff and the extra queues under both typical and skewed demand. |

---

## Key Takeaways

1. Estimate projection FLOPs and bytes moved for the actual phase and batch. Use sustained measurements to find which limit the runtime reaches.
2. Count KV elements with the deployed layer layout and cache precision. Allocate per rank and leave room for active sequences to grow.
3. Keep a client-visible latency metric next to each optimization. A change that raises aggregate throughput can also lengthen every stream gap.
4. Qualify the whole serving revision. Precision, kernels, topology, scheduler, and cache policy jointly determine cost, quality, and recovery behavior.

## References

- [Williams, Waterman, and Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (Communications of the ACM, 2009)](https://doi.org/10.1145/1498765.1498785)
- [Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)](https://arxiv.org/abs/2205.14135)
- [Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023)](https://arxiv.org/abs/2307.08691)
- [Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* (2024)](https://arxiv.org/abs/2407.08608)
- [Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- [Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving* (OSDI 2024)](https://arxiv.org/abs/2401.09670)
- [Qin et al., *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving* (2024)](https://arxiv.org/abs/2407.00079)
- [DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437)
- [Leviathan, Kalman, and Matias, *Fast Inference from Transformers via Speculative Decoding* (ICML 2023)](https://arxiv.org/abs/2211.17192)
- [Dong et al., *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models* (2024)](https://arxiv.org/abs/2411.15100)
- [MLCommons, *MLPerf Inference: Datacenter* results](https://mlcommons.org/benchmarks/inference-datacenter/)
- [vLLM documentation](https://docs.vllm.ai/)
- [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/) and [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/): worked-example hardware inputs and sparsity footnotes
- [NVIDIA Multi-Instance GPU User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/): supported hardware partitioning, memory paths, deployment, reset, and management boundaries
- [Attention & Transformers](../09-whitepapers/15-attention-transformers.md): the architecture this chapter serves

---

## Related Chapters

- [Model Serving](../16-ml-systems/03-model-serving.md): the serving layer that treats this as a black box and inherits its latency distribution
- [Agent Inference](12-agent-inference.md): multi-step workloads break the assumptions made here about request lifetime
- [ML Capacity and Cost Planning](../16-ml-systems/14-ml-capacity-cost-planning.md): the capacity plan that consumes the constants measured in this chapter
- [Attention Is All You Need](../09-whitepapers/15-attention-transformers.md): the paper that fixes the arithmetic these schedulers exist to amortize
