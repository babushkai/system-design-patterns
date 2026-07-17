# Distributed Training Internals

## TL;DR

Training a large model is a supercomputing problem wearing an ML costume. Two constraints force distribution: the model's training state doesn't fit on one accelerator (a 70B-parameter model needs ~1.1 TB just for weights, gradients, and optimizer state), and the compute doesn't fit in one lifetime (the same model trained on 15T tokens is ~500 GPU-years — you run 8,000 GPUs for three weeks or you don't run at all). Everything in this chapter follows from how you split those two things: **data parallelism** replicates the model and averages gradients (cheap until the all-reduce and the batch-size ceiling bite), **ZeRO/FSDP** shards the training state across the replicas, **tensor parallelism** splits individual matrix multiplies (needs NVLink-class bandwidth, so it stays inside a node), and **pipeline parallelism** splits layers into stages (pays a bubble that shrinks with microbatch count). Real jobs compose all of them, and the composition is dictated by the memory hierarchy of the cluster, not by preference. The operational reality on top: a synchronous job moves at the speed of its slowest worker, at 10,000+ GPUs *something* fails every few hours (Llama 3's team logged 466 interruptions in 54 days), and the metric that summarizes all of it — MFU, the fraction of theoretical FLOPs you actually achieve — hovers around 40% for well-run large jobs. This chapter builds the arithmetic for each piece: memory ledgers, collective-communication costs, bubble fractions, checkpoint intervals, and where the missing 60% goes.

The orchestration layer above this — pipelines, retraining, reproducibility — is [Training Pipelines](./05-training-pipelines.md); the sibling problem of extracting FLOPs from a single accelerator at inference time is [GPU Inference Internals](../17-llm-systems/11-gpu-inference-internals.md); cluster-level sizing and cost is [ML Capacity & Cost Planning](./14-ml-capacity-cost-planning.md).

---

## Why Distribute: The Two Ledgers

### The memory ledger

Training state is far bigger than the model. With standard mixed-precision Adam, every parameter carries:

```
bf16 weights                     2 bytes
bf16 gradients                   2 bytes
fp32 master weights              4 bytes
fp32 Adam momentum (m)           4 bytes
fp32 Adam variance (v)           4 bytes
────────────────────────────────────────
                                16 bytes per parameter

  7B model:   112 GB  of state  — doesn't fit an 80 GB H100
 70B model:   1.1 TB            — doesn't fit a NODE of 8 H100s (640 GB)
405B model:   6.5 TB            — needs ≥ 82 GPUs before ONE token is processed
```

And that's before **activations** — the intermediate tensors saved during the forward pass for use in the backward pass, which scale with `batch × sequence_length × hidden × layers` and routinely exceed the parameter state at long sequence lengths. (The standard mitigation, *activation checkpointing*, discards most activations and recomputes them during backward — spending roughly 30% more FLOPs to cut activation memory several-fold. Memory and compute are fungible, and large-model training constantly trades one for the other.)

### The compute ledger

Training FLOPs are well-approximated by `6 × parameters × tokens` (forward ≈ 2·P·T, backward ≈ 2× forward):

```
70B params × 15T tokens × 6 ≈ 6.3 × 10²⁴ FLOPs

One H100: ~990 TFLOP/s peak (bf16, dense). At a realistic 40% MFU:
  ~4 × 10¹⁴ FLOP/s achieved
  6.3e24 / 4e14 ≈ 1.6 × 10¹⁰ GPU-seconds  ≈  500 GPU-YEARS

  On 8,192 GPUs: ~22 days.  On 8 GPUs: ~63 years.
```

The ledgers make the design space concrete: you need tens of GPUs just to *hold* a large model's training state, and thousands to train it in calendar time. The rest of this chapter is about splitting the state and the work so those thousands of GPUs help instead of waiting on each other.

---

## Data Parallelism: Replicate the Model, Average the Gradients

The baseline strategy: every worker holds a full model replica, processes a different slice of each batch, and workers average gradients before stepping — which keeps all replicas bit-for-bit identical and makes N workers mathematically equivalent to one worker with an N×-larger batch.

The averaging is an **all-reduce**, and its cost is worth knowing exactly because it recurs every single step. The bandwidth-optimal ring algorithm has each of N workers send and receive

```
2 × (N-1)/N × D  ≈  2D bytes        (D = gradient bytes, N large)

70B model, bf16 gradients: D = 140 GB → each GPU moves ~280 GB per step.
On a 50 GB/s effective inter-node link: ~5.6 seconds — likely LONGER
than the compute step itself.
```

Three mechanisms keep this from being fatal:

- **Overlap**: gradients for late layers are ready while early layers are still doing backward. Frameworks bucket gradients and launch all-reduce on each bucket as it completes, hiding communication behind computation. A well-tuned job hides most of the 2D; a poorly-tuned one serializes compute-then-communicate and loses a third of its throughput to the network.
- **Gradient accumulation**: run k micro-batches locally, sync once. Divides communication frequency by k at the cost of a k×-larger effective batch — which is only free if you *wanted* a bigger batch (see below).
- **Hierarchical topology**: reduce within a node over NVLink (~900 GB/s) first, then across nodes over the much slower fabric — the ring's cost is set by its slowest link, so you build the rings to match the hardware.

### The ceiling nobody escapes: batch size

Data parallelism's scaling limit is usually not the network — it's *optimization*. N workers means an N×-larger global batch, and beyond a model/dataset-dependent **critical batch size**, larger batches stop reducing the number of steps needed: you burn more FLOPs for the same learning progress. The gradient-noise-scale work (McCandlish et al.) formalizes it; the practical symptom is that scaling from 1k to 4k GPUs makes each *step* 4× bigger but no longer makes *training* meaningfully faster. Learning-rate scaling and warmup (Goyal et al.'s linear-scaling rule) push the ceiling but don't remove it. This is why pure data parallelism runs out at some point even with a perfect network — and why the other parallelisms exist even for models that would technically fit on one device.

---

## ZeRO / FSDP: Shard the State Across the Replicas

Plain data parallelism is memory-obtuse: N replicas hold N identical copies of 16 bytes/param. ZeRO (and PyTorch's FSDP, the same idea) removes the redundancy in three escalating stages:

```
Per-GPU memory (P params, N data-parallel workers):

  Plain DP:        16P                    all state replicated
  ZeRO-1:          4P  + 12P/N            optimizer state sharded
  ZeRO-2:          2P  + 14P/N            + gradients sharded
  ZeRO-3 / FSDP:         16P/N            + parameters sharded

  70B on 64 GPUs with ZeRO-3: 1.12 TB / 64 ≈ 17.5 GB per GPU — fits,
  with room for activations.
```

The mechanics of stage 3: each layer's parameters live sharded; just before a layer runs (forward and again in backward), workers **all-gather** that layer's weights, use them, and immediately free them; gradients are **reduce-scattered** so each worker keeps only its shard. Communication rises from the 2D of plain DP to ~3D — a 1.5× tax paid for an N-fold memory reduction, and like DP's all-reduce it overlaps: prefetch the next layer's all-gather while computing the current one.

The design sensibility to absorb: **ZeRO treats aggregate cluster memory as one pool and pays bandwidth to pretend it's local.** It composes with data parallelism trivially (it *is* data parallelism, memory-optimized) and is the default answer for models in the 7B–70B range. Its limit is that the model's *layers* still execute on every GPU — when a single layer's working set or the activation traffic outgrows the node, you need parallelisms that split the computation itself.

---

## Tensor and Pipeline Parallelism: Split the Computation

### Tensor parallelism: split the matrices

Tensor parallelism (Megatron-style) splits individual weight matrices across GPUs — each holds a column- or row-slice, computes a partial matmul, and an all-reduce assembles the result. The brutal property: that all-reduce happens **per layer, per microbatch, in the critical path** — it cannot be hidden behind other compute the way DP's gradient sync can.

```
Consequence: TP lives or dies on interconnect latency+bandwidth.
  NVLink within a node:   ~900 GB/s  → TP works
  InfiniBand across nodes: ~50 GB/s  → TP dies

Rule that follows: TP degree ≤ GPUs per node (8, typically).
TP is not a scaling strategy — it's a "make the layer fit and keep
per-GPU matmuls large" strategy, confined to one node.
```

### Pipeline parallelism: split the layers

Pipeline parallelism assigns contiguous layer blocks to *stages* on different nodes; activations flow forward stage-to-stage and gradients flow back. Stage-to-stage traffic is just one activation tensor per microbatch — tiny compared to TP's chatter — so PP is the parallelism that crosses slow links happily.

Its tax is the **bubble**: stages idle while the pipeline fills and drains. With p stages and m microbatches per batch:

```
bubble fraction = (p − 1) / (m + p − 1)

  p=8,  m=8:    47% of the schedule is idle  — catastrophic
  p=8,  m=64:   ~10%                          — acceptable
  p=8,  m=256:  ~3%                           — good

So PP demands many microbatches — which is the same resource the
batch-size ceiling limits. Deep pipelines + modest global batch =
bubbles you cannot schedule away. (1F1B scheduling and interleaved
stages reduce peak activation memory and shave the bubble, but the
(p−1)/(m+p−1) shape is the invariant to reason with.)
```

### Composing them: 3D parallelism

Real large-model jobs use all three, and the composition follows the hardware hierarchy rather than taste:

```
  TP  innermost — needs NVLink        → within the node, degree ≤ 8
  PP  next      — tolerates slow links → across nodes, until it fits
  DP  outermost — embarrassingly parallel across the rest (often with
                  ZeRO-1 sharding the optimizer inside each replica)

Example, 70B on 512 H100s (64 nodes):
  TP=8 (one node) × PP=4 (4 nodes = one model replica of 32 GPUs)
  × DP=16 replicas
  Per-GPU: 1/32 of the model's layers × 1/8 of each matrix. Fits with
  activation headroom; global batch = 16 × microbatches × micro size.
```

The knobs interact: raising PP eases memory but demands more microbatches (bubble) which raises the global batch toward its ceiling; raising TP shrinks per-GPU matmuls until they're too small to saturate the tensor cores. Finding the optimum is an afternoon of arithmetic plus a day of profiling, and it is genuinely worth it — published configs (Megatron-LM, Llama 3) are the right starting points, not the right answers, for your cluster's fabric.

```mermaid
flowchart TB
    subgraph DC["Training cluster"]
      subgraph DP0["Data-parallel replica 0"]
        subgraph PP0["Pipeline stage 0 / node 0"]
          T00["TP rank 0"] --- T01["TP rank 1"] --- T07["TP rank 7"]
        end
        subgraph PP1["Pipeline stage 1 / node 1"]
          T10["TP rank 0"] --- T11["TP rank 1"] --- T17["TP rank 7"]
        end
        PP0 -->|"activations / gradients"| PP1
      end
      subgraph DP1["Data-parallel replica 1"]
        R1["same TP × PP shape"]
      end
      DP0 <-->|"gradient reduce / state sharding"| DP1
    end
```

The diagram is a topology contract. High-frequency tensor collectives stay on the node's fastest fabric; pipeline boundaries cross nodes with one activation stream; data-parallel synchronization spans replicas and is overlapped with backward. A scheduler that assigns the right number of accelerators but maps these dimensions onto the wrong links can make a mathematically sensible plan unusable.

## Collective Cost and Topology Mapping

Collective communication follows an `α–β` model:

```text
time(message_bytes M) ≈ α × synchronization_rounds + β × bytes_transferred

α = per-round latency and software synchronization cost
β = inverse effective bandwidth (seconds/byte)
```

Large gradient buckets are bandwidth-bound, so ring all-reduce—with many rounds but near-optimal bytes—is effective. Tiny per-layer tensor-parallel messages can be latency-bound, so a tree or topology-specific collective may win despite moving different bytes. “The fabric is 400 Gb/s” is not an input to the model until benchmarked through the actual collective, message-size range, rank count, NIC/GPU direct-memory path, and concurrent traffic. Effective bandwidth can be far below line rate because PCIe topology, NUMA placement, switch oversubscription, routing, protocol overhead, or another job shares the path.

Model a hierarchical all-reduce as two collectives, not one average link: reduce-scatter inside each node, inter-node collective among node representatives or shards, then all-gather inside the node. The slow fabric sees less traffic and the fast fabric absorbs the local expansion. Modern libraries choose algorithms dynamically, but the training team still needs per-bucket traces: a library cannot fix ranks placed across a split PCIe tree or an oversubscribed rack.

Overlap is also a dependency graph, not a checkbox. A gradient bucket can launch only when every parameter in it has a gradient; a single late layer delays the whole bucket. Buckets that are too large expose little overlap, while buckets that are too small pay `α` repeatedly and increase launch overhead. Parameter order, bucket construction, backward graph, and network streams together determine whether the advertised communication is actually hidden. The decisive profile is a timeline showing kernels, collectives, input, and idle gaps on every rank—not a single utilization percentage.

## Sequence, Context, and Expert Parallelism

Long sequences create an activation problem even when parameters fit. **Sequence parallelism** shards operations whose inputs can be partitioned along tokens—layer normalization, dropout, and parts of attention/MLP processing—across tensor-parallel ranks, avoiding replicated activations. **Context parallelism** goes further and partitions the attention sequence itself. Each rank owns a span of queries and must obtain the keys/values needed to attend across the full context, commonly through ring-style exchange. The memory reduction is close to the context-parallel degree, but communication grows with activation size and the attention algorithm must preserve causal masking and numerical equivalence across partitions.

The placement rule mirrors TP: context exchange happens every layer, so keep the group on fast links when possible. It is attractive when context length, rather than parameter state, is the limiting dimension; it is wasteful for short sequences where communication and smaller kernels dominate. Sequence packing matters too. Padding every example to the longest sequence in a batch can burn most FLOPs on padding, while naive packing can let tokens attend across document boundaries or leak examples. The attention mask and position IDs are part of training correctness, not loader optimization details.

Mixture-of-experts models add **expert parallelism**. A router sends each token to a small top-k subset of experts, and ranks exchange token activations with an all-to-all before and after expert computation:

```text
tokens per rank
  → router probabilities
  → top-k expert assignments
  → all-to-all dispatch
  → local expert MLPs
  → all-to-all combine
  → weighted output
```

Unlike dense TP, communication volume depends on routing and can become imbalanced. One popular expert receives more tokens, fills its capacity, and becomes both a straggler and a quality problem if overflow tokens are dropped. Capacity factor reserves slots above the expected uniform load; auxiliary load-balancing loss encourages even routing; expert replication places hot experts more than once; token dropping or rerouting defines overload semantics. Each choice couples systems and learning behavior. A tight capacity factor saves memory but changes which tokens receive expert computation. An aggressive balancing loss protects throughput but may make the model choose less appropriate experts. Monitor per-expert token counts, dropped/rerouted tokens, dispatch bytes, all-to-all time, and expert utilization alongside training loss.

For sparse models, distinguish **active parameters per token** from total stored parameters. Compute scales mainly with active experts, but memory and checkpoint size scale with total experts, and all-to-all scales with routed activation traffic. Capacity plans that use only active parameter count underestimate model load and recovery; plans that use total count for FLOPs overestimate compute. The two ledgers diverge more sharply than in a dense model.

## Precision, Numerics, and Distributed Correctness

Mixed precision is part of the parallel system. `bf16` has fp32's exponent range and usually avoids loss scaling; `fp16` has a narrower range and commonly needs dynamic loss scaling; fp8 paths require per-tensor or block scaling, delayed scale updates, and kernels/hardware that implement the assumed format. Master weights and optimizer states may remain fp32 even when matmuls use lower precision. Reductions accumulate values from many ranks, so accumulation dtype and reduction order affect error.

Floating-point addition is not associative: `(a+b)+c` can differ from `a+(b+c)`. Changing world size, collective algorithm, bucket order, compiler fusion, or pipeline schedule can change rounding and therefore the exact trajectory. Reproducibility must declare its level: bitwise replay on identical topology, statistically equivalent training within tolerances, or only comparable final evaluation. Promising bitwise identity while allowing the collective tree and kernel set to change is incoherent.

Numerical safety signals need rank-level provenance. Track loss scale, non-finite gradients, gradient norm before and after clipping, parameter/update norms, activation extrema, router statistics, and optimizer state health. A global mean can hide one rank producing corrupt values just before all-reduce spreads them everywhere. Sample per-rank checksums or known-answer kernels during burn-in, quarantine hardware with repeatable deviations, and record the exact topology and library versions in the training run. “Loss became NaN” is the final symptom; the useful signal is which tensor, rank, and step first violated its expected range.

---

## The Parts That Aren't Matrix Math

### The input pipeline must outrun the GPUs

A training step consumes `global_batch × seq_len` tokens; the storage and preprocessing path has to deliver that every step, forever:

```
8,192 GPUs × ~50K tokens/s/GPU ≈ 4 × 10⁸ tokens/s ≈ several GB/s of
decompressed, tokenized, shuffled data — sustained for weeks.
```

This is a storage-systems problem ([Training Pipelines](./05-training-pipelines.md) covers formats and sharding); the distributed-training-specific requirements are **determinism and resumability** — every worker must see a disjoint shard, in an order that is exactly reproducible so a job restarted from a checkpoint continues the *same* data sequence (skipping consumed samples) rather than re-sampling. Data-order bugs are among the nastiest in the field: they show up as irreproducible loss curves, silent sample duplication, or eval contamination, weeks after the fact.

### Stragglers: the max() over workers

A synchronous step completes when the *slowest* participant finishes. At scale this transforms rare slowness into constant slowness:

```
One GPU has a 1-in-1000-steps slow event (thermal throttle, ECC retry,
noisy neighbor on the NIC, background daemon):
  1 GPU:      0.1% of steps affected
  8,192 GPUs: essentially EVERY step waits for someone's slow event —
  the fleet moves at its collective p99.9.
```

Defenses are unglamorous and essential: pre-flight burn-in to evict weak hardware, tight monitoring of per-rank step times (the histogram's outlier *is* the job's speed), topology-aware placement so one oversubscribed switch doesn't slow a whole ring, and hot spares to swap in rather than debug in place. Asynchronous SGD — the 2012-era answer — traded this problem for stale gradients and worse convergence; modern practice is synchronous training plus ruthless straggler elimination.

### Failure math and checkpointing

Component failures are Poisson; fleets are large; multiply:

```
If one GPU fails on average every ~5 years:
  16,384 GPUs → a hardware failure every ~2.7 hours of wall clock.

Llama 3 405B (Meta, 2024): 54 days on 16K H100s, 466 job
interruptions — one every ~2.8 hours — 78% attributed to hardware
(GPUs, HBM, NICs, cables). This is the NORMAL operating regime.

Synchronous training means one failure stops all 16K GPUs. Recovery =
restore from last checkpoint. Expected loss per failure:
  (checkpoint_interval / 2) × fleet — plus restart time × fleet.

Optimal checkpoint interval (Young/Daly):
  τ* ≈ sqrt(2 × δ × MTBF)      δ = time to write a checkpoint
  δ = 5 min, MTBF = 2.7 h  →  τ* ≈ 52 min.
  But drive δ down to 30 s (async, sharded) → τ* ≈ 16 min, and the
  expected goodput loss per failure drops ~3×.
```

That last line is why modern checkpointing is an engineering topic of its own: **sharded** (every rank writes its own state slice in parallel — a 6.5 TB state written by 16K ranks is manageable; funneled through rank 0 it's an outage), **asynchronous** (snapshot to host memory in seconds, drain to [object storage](../03-storage-engines/08-object-storage.md) in the background while training continues), and increasingly **peer-redundant** (recover a lost rank's state from neighbors' memory rather than storage). Checkpoint frequency stops being a dial you set timidly and becomes cheap insurance.

The end-to-end health metric is **goodput**: the fraction of wall-clock GPU-time spent on steps that contributed to the final model — after subtracting restarts, replayed work, stalls, and initialization. Mature large-job teams report goodput improvements (85→95%+) worth millions of dollars, achieved entirely with the unglamorous machinery above.

---

## MFU: The One Number That Summarizes Everything

**Model FLOPs Utilization** — achieved useful FLOPs (`6·P·tokens/sec`) divided by the fleet's theoretical peak — is the honest scoreboard, immune to batch-size games and hardware-generation confusion:

```
MFU = (6 × P × tokens_per_second) / (N_gpus × peak_FLOPs_per_gpu)

Reference points:
  50-57%   exceptional (dense LLM, tuned Megatron-class stack, PaLM/
           Llama-3-scale engineering)
  35-45%   good, typical for well-run large jobs
  20-30%   common in practice — something is eating a third of the fleet
  <20%     the job has a bug wearing a performance costume

Where the missing fraction goes (the audit order):
  1. data stalls        — input pipeline can't feed (check first, it's
                          the cheapest fix and the most common)
  2. communication      — unoverlapped all-reduce/all-gather time
  3. pipeline bubbles   — (p−1)/(m+p−1), by construction
  4. kernel inefficiency— small matmuls (TP too high), missing fused
                          kernels (FlashAttention et al.)
  5. stragglers/restarts— the max() tax and the failure tax (goodput)
```

MFU is also the negotiation currency: "we need 2× the GPUs" and "we can raise MFU from 25% to 40%" are the same sentence to a budget, and the second one is usually cheaper ([ML Capacity & Cost Planning](./14-ml-capacity-cost-planning.md)).

---

## Failure Modes

**The NCCL stall that looks like a hang.** One rank dies or one NIC flaps; every other rank blocks inside a collective, GPUs pinned at 100% utilization doing nothing. Without watchdogs the job hangs silently until a human notices. Defenses: collective timeouts (NCCL watchdog), per-rank liveness heartbeats *outside* the training loop, and automated restart-from-checkpoint — at a failure every ~3 hours, recovery must be reflexive, not paged.

**Dataloader starvation misdiagnosed as GPU problems.** Step time is high, GPU utilization graphs look busy (they show occupancy, not useful work), and the team tunes kernels for a week. The tell: profile one step; if the GPU waits on the input queue, the fix is in CPU count, decode/tokenize throughput, or storage bandwidth. Always audit item #1 before item #4.

**Silent divergence after restart.** Resume mishandles the data-order cursor, the LR schedule, or RNG state; the loss curve looks *plausible* but the run is no longer the run you think it is — discovered at eval time, weeks later. Treat resume as a tested code path: restart a small job mid-run and diff loss curves bit-for-bit against an uninterrupted control.

**Loss spikes and bad numerics at scale.** bf16 training with large batches occasionally spikes; a single corrupted node (bad HBM producing NaNs or — worse — *wrong numbers without NaNs*, silent data corruption) can poison a global all-reduce. Defenses: gradient-norm clipping and monitoring, per-rank gradient-norm outlier detection (the corrupt rank is visible *before* the loss shows it), skip-batch-on-spike policies, and periodic known-answer tests on suspect hardware.

**Topology-blind placement.** The scheduler grants 512 GPUs scattered across the datacenter; rings cross oversubscribed spine links; all-reduce runs at a fraction of node-local speed. Gang scheduling with topology awareness (whole racks/pods, [ML Capacity & Cost Planning](./14-ml-capacity-cost-planning.md)) is a first-order throughput factor, not an infra nicety.

**Cargo-culted parallelism configs.** A 3D config tuned for one fabric (NVLink+NDR InfiniBand) transplanted onto slower cloud networking inverts its trade-offs — TP across nodes, PP with too few microbatches. The published config encodes someone else's hardware; re-derive from your own bandwidth numbers.

---

## Decision Framework

| Situation | Reach for |
|---|---|
| Model + optimizer state fits on one GPU | Plain DDP; add gradient accumulation before adding machinery |
| State exceeds one GPU, fits the cluster ÷ N | ZeRO/FSDP (stage 2, then 3) — the 7B–70B workhorse |
| Single layer / activation working set exceeds a node's memory even sharded | Add TP (≤ node size), then PP across nodes |
| Hundreds of nodes, model spans many | Full 3D: TP≤8 innermost, PP to fit, DP (+ZeRO-1) outermost |
| Long-context training blowing activation memory | Activation checkpointing first; sequence/context parallelism second |
| MoE model | Expert parallelism (all-to-all); budget fabric for it explicitly |
| Scaling DP but steps-to-converge stopped improving | You've hit the critical batch size — more GPUs won't help; change the parallelism split or accept the wall |
| Job at ≥ thousands of GPUs | Async sharded checkpointing, NCCL watchdogs, hot spares, per-rank step-time monitoring — before scaling, not after the first 3 a.m. hang |
| Deciding cluster size / budget | Do the 6PT arithmetic and target MFU first ([capacity planning](./14-ml-capacity-cost-planning.md)); GPU count is an output, not an input |

---

## Key Takeaways

1. **Two ledgers force distribution**: 16 bytes/param of training state (memory) and 6·P·T FLOPs (compute). Do this arithmetic before any architecture discussion — GPU count is an output.
2. **Data parallelism scales until the batch-size ceiling**, not just the network — beyond the critical batch size, more replicas buy bigger steps, not faster training.
3. **ZeRO/FSDP is data parallelism with the redundancy removed** — 16P/N memory for ~1.5× communication, and the default answer below the "one layer doesn't fit a node" threshold.
4. **TP is confined to the node, PP crosses nodes, and the bubble is (p−1)/(m+p−1)** — the 3D composition is dictated by the interconnect hierarchy, not preference.
5. **Synchronous training moves at the fleet's p99.9** — straggler elimination, burn-in, and topology-aware placement are throughput features.
6. **At 10K+ GPUs, failure every few hours is the normal regime** — Young/Daly sizes the checkpoint interval, and driving checkpoint *cost* down (sharded, async) is worth more than any single kernel optimization.
7. **MFU is the scoreboard, goodput is the uptime** — audit data stalls before communication before bubbles before kernels; ~40% MFU is good, and the gap from 25% to 40% is usually cheaper than 60% more GPUs.
8. **Resume is a correctness feature** — data cursors, RNG, and LR schedules must survive restarts bit-for-bit, and the only way to know is to test the restart path like production code.

---

## References

- Shoeybi, M., et al. (2019). *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*. (Tensor parallelism.)
- Rajbhandari, S., et al. (2019). *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*. SC.
- Huang, Y., et al. (2019). *GPipe*; Narayanan, D., et al. (2019/2021). *PipeDream* and *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (3D parallelism, 1F1B, MFU-style accounting).
- Goyal, P., et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. (Linear scaling rule.)
- McCandlish, S., et al. (2018). *An Empirical Model of Large-Batch Training*. (Gradient noise scale / critical batch size.)
- Chowdhery, A., et al. (2022). *PaLM: Scaling Language Modeling with Pathways*. (MFU definition and reference numbers.)
- Grattafiori, A., et al. (2024). *The Llama 3 Herd of Models*. (16K-GPU operations: 466 interruptions/54 days, failure taxonomy, MFU at scale.)
- Jiang, Z., et al. (2024). *MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs*. NSDI.
- Young, J. W. (1974) / Daly, J. T. (2006). Optimal checkpoint interval analyses.
- Zhao, Y., et al. (2023). *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*. VLDB.
- NCCL documentation: *Collective Operations* (ring/tree algorithms and their cost models).
