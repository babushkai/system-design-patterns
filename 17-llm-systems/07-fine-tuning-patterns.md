# Fine-Tuning Patterns

## TL;DR

Fine-tuning changes a model artifact's conditional behavior. It is justified when a versioned training intervention produces a measured quality, latency, cost, privacy, or ownership benefit over the best prompt/retrieval/tool baseline, and when the organization can operate the resulting data and release lifecycle. It is a poor store for mutable or permissioned facts; retrieval and source-of-truth tools own those.

The system is larger than the optimizer: data rights and lineage, contamination-resistant evaluation, base/tokenizer/template compatibility, checkpoint recovery, privacy and safety testing, artifact registration, quantization qualification, adapter placement, rollback, and refresh. Parameter-efficient methods reduce trainable state but do not remove activation, data, serving, or governance costs. The durable asset is a reproducible behavior dataset and evaluation contract; any individual base model or adapter will age.

---

## When to Fine-Tune (and When Not To)

The question arrives as "should we fine-tune?" and should be answered by walking a ladder in cost-of-ownership order:

**1. Prompting and deterministic interfaces.** Establish the strongest qualified baseline using concise instructions, representative examples, structured outputs, tools, and the intended reasoning budget. Prompting has low iteration latency but nonzero inference-token, evaluation, and maintenance cost. Its economics depend on the deployed model and measured prefix reuse, not a universal cached-token ratio.

**2. Retrieval and source-of-truth tools.** If the gap is mutable knowledge, authorization-sensitive data, exact calculation, or evidence, supply it at inference. Retrieval can publish corrections and enforce current ACLs independently of weight refresh. Fine-tuned weights do not provide record-level attribution or immediate deletion semantics; training on sensitive facts expands the governance boundary (see [ML risk and governance](../16-ml-systems/09-ml-risk-governance.md)).

**3. Fine-tuning.** What remains is *behavior and economics*, and here tuning is genuinely strong:

| Motivation | Why tuning wins | Typical shape |
|---|---|---|
| Style/format lock-in | Stable examples can move repeated behavioral guidance into weights | Curated SFT, compared with schema/prompt baseline |
| Distillation for cost/latency | A smaller model may reproduce bounded teacher behavior on the target distribution | Teacher/corrected data plus student SFT and coverage tests |
| Deep domain adaptation | A measured representation or task gap remains after supplying evidence | Continued pretraining and/or SFT, compared with retrieval baseline |
| Reliability on a narrow task | Repeated behavior can be learned from governed corrections | SFT on representative successes, corrected failures, and abstentions |
| Latency floors | Shorter prompts (no few-shot block) and smaller models cut TTFT | SFT replacing the prompt scaffolding |

Treat the tune as a product line rather than a one-off job. The proposal should include the baseline experiment, dataset and legal basis, qualification suite, serving artifact, break-even volume, owner, rollback, deletion response, and refresh trigger. If the second run cannot be reproduced from sealed inputs, the first checkpoint is a stranded experiment rather than an operable model.

### Frame the tune as a falsifiable intervention

A training proposal should state the target behavior, affected workload slices, incumbent baseline, acceptable regressions, and expected serving benefit. For example: “increase valid tool selection on multilingual return requests from 91% to 97% while factual-answer accuracy drops no more than 0.5 points; reduce median input tokens by 1,200 and cost per successful case by 30%.” That statement determines the data, evaluation, and rollout. “Make the model know our business better” does not.

The artifact under test is the complete inference configuration:

$$
Behavior = f(base\ revision, adapter, chat\ template, system\ prompt,
             tools, decoding, retrieval, runtime).
$$

A tuned checkpoint cannot be qualified independently of its template and serving stack. A tokenizer change, altered stop token, different quantized base, or new tool schema can erase the apparent gain even if adapter bytes are unchanged.

---

## The Method Landscape

### Supervised Fine-Tuning (SFT)

Supervised fine-tuning continues training on `(input → desired output)` pairs with prompt-token loss masked so only the completion is learned. Subsequent methods differ in pair construction and which parameters move.

### LoRA and QLoRA: Parameter-Efficient Mechanics

Full fine-tuning updates every selected base parameter and therefore holds or shards weights, gradients, optimizer state, and activations. **LoRA** freezes a base matrix $W$ and learns a low-rank update $BA$:

```text
h = Wx + (α/r)·B·A·x

For W in R^(d_out × d_in):
  full trainable parameters = d_out × d_in
  LoRA trainable parameters = r × (d_in + d_out)
```

The reduction depends on layer dimensions, target modules, and rank. **QLoRA** stores the frozen base in a quantized representation while training adapter parameters at a higher compute precision. It reduces base-weight memory, but activations, temporary dequantization state, adapter optimizer state, sequence length, and runtime kernels still determine whether a configuration fits. The QLoRA paper's single-device result is an existence proof for a specific stack, not a capacity guarantee for every model or sequence length.

Choose target modules and rank through ablation. Low rank can underfit a complex distribution shift; excessive rank raises memory and can overfit without improving the product metric. A merged adapter removes runtime adapter selection but creates a separate full artifact. An unmerged adapter preserves composability and multi-tenant serving but adds compatibility, cache, scheduling, and isolation concerns.

At the end you either **merge** the adapter into the base weights (zero serving overhead, one artifact) or keep it separate — which enables the serving pattern below.

### Preference Optimization: DPO and Friends

SFT teaches the model what target outputs look like; preference methods fit relative judgments where no single reference completion captures the objective. Online RLHF introduces rollout, reward-model, policy, reference, and value/optimizer state plus reward-hacking and stability risks. **DPO** trains directly on `(prompt, chosen, rejected)` triples relative to a reference policy, avoiding online policy rollouts and a separately served reward model in the optimization loop. This simplifies one class of preference training but does not make pair collection, reference choice, regularization, or evaluation trivial. Other objectives encode different assumptions about feedback and policy drift; choose from the label contract rather than framework defaults.

Policy optimization with verifiable rewards uses executable or otherwise checkable outcomes instead of a learned preference score. It is attractive where rollouts can be sandboxed and reward cannot be cheaply gamed—formal answers, tests, or constrained environments. It introduces rollout infrastructure, variance, credit assignment, environment versioning, and reward-hacking risk. Compare it with rejection sampling or SFT on verified successes before operating an online RL loop.

For a prompt $x$, preferred response $y_w$, rejected response $y_l$, policy $\pi_\theta$, and frozen reference $\pi_{ref}$, DPO increases the relative log-probability margin of the preferred pair:

$$
\mathcal{L}_{DPO} = -\log \sigma \left(\beta
\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
- \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]\right).
$$

The reference term restrains drift from the starting policy; $\beta$ controls that trade-off. The equation also exposes a data requirement: a preference pair must express a meaningful distinction. If `chosen` and `rejected` differ in several dimensions—correctness, verbosity, tone, and formatting—the model cannot know which preference to learn. Capture rubric dimension and annotator rationale, balance response position, include close calls, and measure label agreement.

Preference optimization can exploit artifacts of the data collector. If annotators prefer longer answers, the policy learns length; if a judge model produced both labels and later grades the release, the apparent improvement can be self-agreement. Run dimension-specific human evaluation and objective checks outside the preference pipeline.

### Continued pretraining and domain adaptation

Continued pretraining uses next-token learning over an unlabeled domain corpus before instruction tuning. It can improve vocabulary, syntax, and latent domain patterns, but it is more likely than LoRA-SFT to shift general capability and memorize source text. Corpus licensing, PII removal, deduplication, holdout contamination, and general-capability regression become central. A practical sequence is domain-adaptive pretraining → SFT on task demonstrations → optional preference optimization, with evaluation after each stage so a harmful stage can be removed.

Do not confuse corpus perplexity improvement with product success. A model can predict legal text better without becoming safer or more accurate at legal question answering. Perplexity diagnoses representation fit; task and safety evaluations determine release value.

### Distillation

Distillation transfers behavior from a teacher or ensemble into a student on a bounded distribution. The student need not match the teacher outside that support, so coverage and abstention are part of the contract. Teacher generations should retain prompt, sampling, model, evidence, and verification provenance; self-grading by the same teacher is a correlated filter, not independent truth. Use executable checks, expert correction, or a calibrated judge where possible, and verify that data licenses and provider terms permit the intended training and deployment.

The break-even condition is volume-dependent:

$$
N_{break-even} =
\frac{C_{data}+C_{training}+C_{eval}+C_{platform}+C_{refresh}}
{C_{baseline/task}-C_{student/task}},
$$

provided the student satisfies every quality and safety gate. Include failed tasks and fallback-to-teacher cost in both per-task terms.

---

## Data Is the Product

The training corpus and its lineage determine what intervention can be reproduced, inspected, deleted, and refreshed. Model and data are both release inputs; neither is a commodity when its exact revision changes behavior.

**Curation controls signal.** Small high-quality instruction sets can outperform much larger noisy sets for some adaptation tasks, as LIMA illustrates, but sample complexity depends on base capability, task diversity, and desired coverage. Track marginal gain as examples are added and retain hard negative, abstention, and rare-policy slices instead of optimizing row count.

**Mine production with selection controls.** Real traces expose the served distribution, but feedback is biased by exposure, user action, and incumbent behavior. Sample both successes and failures, retain assignment/exposure metadata, correct outputs through a governed label process, and keep a locked evaluation path. The [label-system](../16-ml-systems/10-label-ground-truth-systems.md) chapter covers adjudication and sampling.

**Deduplicate, decontaminate, split honestly.** Near-duplicate examples silently overweight their pattern; eval examples leaking into training data produce the classic too-good-to-be-true validation score ([leakage](../16-ml-systems/05-training-pipelines.md), in fine-tuning clothes). Split by *entity or scenario*, not by row, when generalization to new entities is the goal.

**Version the dataset like a release artifact.** Snapshot, hash, and record the exact training set with the produced model — a tuned model whose data can't be reconstructed can't be debugged, audited, or legally defended. The full argument lives in [dataset management](../16-ml-systems/11-dataset-management-versioning.md); fine-tuning inherits all of it, plus a sharper privacy edge: PII in training data can resurface verbatim in generations, so scrubbing happens *before* training, not in the output filter.

**Format for the target.** Chat-tuned bases encode template assumptions through roles and special tokens; a training/serving mismatch silently changes the learned sequence distribution. Pin tokenizer, chat template, role boundaries, loss mask, and serving-time instruction contract as one compatibility tuple.

### Dataset contract and lineage

Each example needs more than prompt and completion. Store a stable example ID, source and consent basis, creation method, annotator or teacher provenance, rubric, language/domain slice, policy version, quality status, and content hash. For conversation data, preserve role boundaries and tool-call/result pairing. A generated tool result that never came from the real tool should be marked synthetic; otherwise the model learns impossible API behavior.

Publish training data as immutable manifests. A manifest references exact shards and transformations, records exclusions and tombstones, and is sealed before training. The run records the manifest digest rather than a mutable table query. Dataset construction code, tokenizer, chat template, truncation, packing, loss mask, and sampling weights are part of lineage.

### Mixture design

The effective training distribution is controlled by sampling, not raw row counts. Let dataset $D_j$ have sampling weight $w_j$. The training objective is approximately

$$
\mathcal{L} = \sum_j w_j E_{(x,y)\sim D_j}[\ell(x,y)],
\qquad \sum_j w_j = 1.
$$

A small high-weight slice can dominate learning; a huge low-weight safety slice may barely appear. Define weights from product importance and regression risk, then log realized example and token proportions. Sampling by examples overweights long conversations in tokens; sampling by tokens changes exposure again. Report both.

Include negative and abstention behavior deliberately. A dataset containing only answerable requests teaches the model to answer everything. Tool-use data must include when not to call a tool, ambiguous arguments requiring clarification, permission denial, transient error, irreversible-action approval, and successful recovery.

### Splitting and contamination

Random row splits are often invalid. Split by customer, case, document, repository, time window, or underlying scenario so near-duplicate turns cannot cross boundaries. Maintain a provenance graph and similarity-based contamination scan across pretraining, SFT, preference, and eval datasets. Exact hashes catch copies; MinHash or embedding neighbors catch paraphrased and templated duplicates, which require reviewed thresholds.

Use a time-forward test when the product faces drift. Keep a locked qualification set that prompt and training authors cannot inspect freely, plus a rotating production-derived set that reflects new failures. No single split serves both unbiased model selection and ongoing incident learning.

---

## Training Mechanics Worth Knowing

Learning rate, schedule, effective token batch, sequence length, packing, loss mask, target modules, rank, precision, and training duration interact with model family and data. Copying one recipe across bases is not reproducibility. Run a small design of experiments around a documented baseline and select on held-out product metrics, not minimum training loss.

Train and validation loss diagnose optimization fit but not behavior alone. Falling train loss with worsening held-out loss suggests overfitting or split leakage; flat curves can indicate optimizer scale, frozen targets, masking errors, or weak data signal; stable loss can coexist with regression on rare safety or tool slices. Monitor slice metrics and general-capability probes at checkpoints, then retain the earliest checkpoint on the product Pareto frontier.

Training frameworks and hosted APIs package mechanics but do not remove the [training-pipeline](../16-ml-systems/05-training-pipelines.md) contract: immutable inputs, versioned environment, deterministic-enough replay, checkpoint integrity, data cursor recovery, metrics, and artifact lineage.

### Memory accounting

For full Adam training, budget model parameters, gradients, two optimizer moments, master weights where used, activations, temporary buffers, and communication workspaces. A rough per-parameter state can exceed 16 bytes before activations. Sharding strategies distribute different terms:

- data parallelism replicates model and optimizer state;
- ZeRO/FSDP shards optimizer state, then gradients, then parameters by stage;
- tensor and pipeline parallelism split model computation when one sharded unit still does not fit;
- activation checkpointing trades recomputation for activation memory;
- sequence/context parallelism addresses long-sequence activation pressure.

LoRA reduces trainable optimizer state, but the frozen base, activations, and forward/backward compute remain. For a linear layer $W \in \mathbb{R}^{d_{out}\times d_{in}}$, LoRA adds

$$
r(d_{in}+d_{out})
$$

trainable parameters instead of $d_{in}d_{out}$. Total memory depends on target modules, rank, sequence length, batch, quantization, checkpointing, and runtime kernels; “fits on one GPU” is a measured configuration, not a property of QLoRA itself.

### Sequence construction and loss semantics

Train on tokenized sequences exactly as the serving template will produce them. Mask padding. For assistant behavior, usually mask system/user/tool-result tokens and train on assistant tokens; for continued pretraining, train all tokens. Tool calls may need separate weighting from prose so a large natural-language corpus does not drown the small structured-action signal.

Packing short examples removes padding waste, but attention and position IDs must prevent one example from leaking into the next. Truncation policy is a model behavior decision: dropping the end can remove the answer; dropping the beginning can remove instructions; silently discarding long examples biases the trained distribution toward easy short cases. Measure truncation by slice and construct windows intentionally.

### Reproducible training and recovery

A run manifest pins code/container, base and tokenizer digests, dataset manifest, sampling weights, template, optimizer/scheduler, precision, distributed topology, seed, and library/kernel versions. Exact bitwise reproduction across accelerator counts is rarely realistic, but statistical reproduction should be: repeated runs land within a declared metric tolerance.

Checkpoint model/adapter, optimizer, scheduler, random states, data-loader cursor, and consumed-sample counts consistently. On distributed failure, restore the whole logical step; mixing a new data cursor with old optimizer state changes the run. Validate checkpoints by loading and generating, not only by successful upload.

Monitor loss by data slice, token type, and sequence length. A single aggregate loss can fall while the rare tool or safety examples regress. Gradient norms, skipped steps, overflow, learning-rate progression, tokens/sec, padding ratio, and host/input stalls distinguish numeric failure from data-pipeline underutilization.

---

## Serving Tuned Models

The deployment decision interacts with the tuning method more than teams expect:

**Merged model.** Fold the LoRA update into the base and publish one qualified artifact. This removes per-request adapter selection but duplicates weight storage and rollout capacity per variant. It fits a small stable variant set with enough traffic to justify dedicated replicas.

**Multi-adapter serving.** Keep adapters separate and resolve one or more compatible adapters per request over a shared base. This amortizes base weights across a sparse variant catalog, but adapter residency, cold load, batching compatibility, per-tenant isolation, and cache thrash become control-plane concerns. Compare the measured adapter working set with HBM/host capacity before assuming one pool can serve every tenant variant.

**Champion/challenger release.** Register lineage to the exact data, base, tokenizer, template, optimizer, and serving artifact; gate against the qualified prompting/retrieval baseline; shadow effects safely; canary by stable session assignment; and retain a tested rollback target. An adapter is compatible with exact base weights and target modules, not merely a marketing family name. A base or runtime change creates a new candidate that must be re-qualified and may require retraining.

### Adapter control plane

An adapter registry records adapter digest, compatible base digest and tokenizer, target modules, rank/scaling, training/eval lineage, tenant or product owner, policy status, and lifecycle state. The gateway resolves a logical variant to `(base, adapter, template, decoding)` atomically. A request cannot mix an adapter with a merely similar base family.

Multi-adapter serving adds a cache and scheduler. Popular adapters remain resident; cold adapters load from signed storage; requests may batch together only if the runtime preserves correct adapter application. Admission includes adapter memory and load deadline. Per-tenant quotas prevent one customer with many variants from thrashing the adapter cache. Eviction and prefetch use observed request frequency and size, not count alone.

Never accept arbitrary user-uploaded adapters into a trusted process without validation. Adapters can drastically alter safety behavior and may target unexpected modules. Scan formats, disallow executable serialization, verify signatures and compatibility, isolate untrusted variants, and re-run policy evaluations.

### Quantization and compilation qualification

Quantizing or compiling a tuned model creates a new release candidate. Small numeric changes can affect narrow decision boundaries, structured output, or long-context behavior. Evaluate the exact serving artifact and kernel stack, compare output distributions by slice, and retain the unquantized reference for diagnosis. Calibration data for post-training quantization must represent the tuned workload and comply with the same data governance.

## Evaluation, Release, and Feedback

Evaluation begins before data collection so annotations match the intended metrics. Separate task correctness, factuality, calibration/abstention, tool trajectory, format, style, safety, and general capability. A weighted “quality” score can hide a catastrophic regression; use per-dimension gates for non-negotiable properties and a product utility function only for tradeable ones.

Repeatedly testing candidate checkpoints against one set creates selection overfitting. Keep a development set for iteration, a locked qualification set for release, and a final human comparison on sampled production-like traffic. Use paired examples and confidence intervals. For preference-shaped behavior, randomize response order and blind annotators to model identity.

Shadow traffic is safe only when tuned tool calls do not execute effects. Compare proposed trajectories or use sandbox tools. Canary rollout pins session assignment, records the full inference configuration, and watches delayed outcomes long enough to catch user corrections and escalation. Rollback moves an alias to the prior qualified tuple; it must not depend on rebuilding weights during the incident.

Production feedback is not automatically a label. Thumbs-up is selection-biased; a user may accept a wrong answer; successful tool completion may reflect forgiving downstream systems. Store feedback with exposure and outcome context, sample corrections for expert labeling, and prevent a newly tuned model from training directly on its own unchecked outputs. The [feedback-loop](../16-ml-systems/04-model-monitoring.md) risk is strongest when generated data dominates future training.

## Privacy, Safety, and Model Governance

Training changes the model artifact's data-governance boundary. Minimize and redact data before it enters the immutable training snapshot; restrict raw trace access; preserve consent and licensing fields; and define retention for examples, checkpoints, and intermediate caches. Output filters cannot reliably undo memorization.

Membership and extraction tests are risk signals, not proofs that a model contains or lacks one record. Limit verbatim repeated sequences, deduplicate aggressively, and use privacy-preserving training such as differential privacy only when its quantified privacy/utility trade-off matches the threat model. Legal deletion may require excluding the record from future runs and retraining or using an approved unlearning procedure; deleting the source row does not modify existing weights.

Safety tuning does not replace runtime policy. Distribution shift, adversarial input, a changed system prompt, or a new tool surface can bypass learned refusal. The serving layer still enforces authorization, sandboxing, rate limits, and action approvals. Red-team both the base and adapter because tuning can create new jailbreak or data-exfiltration behavior.

---

## Failure Modes

**Tuning what prompting solves.** A quarter of engineering for what a rewritten prompt plus five examples achieves. Defense: the prompting baseline is a mandatory pre-experiment, and the tune must beat it on the eval set to ship.

**Knowledge baked into weights.** Facts go stale, can't be cited, can't be deleted; the model confidently recites last year. Defense: knowledge → RAG; behavior → weights.

**Data leakage and contamination.** Eval examples (or near-duplicates) in the training set produce a mirage of quality. Defense: hash-based dedup across train/eval, entity-level splits, and suspicion of any dramatic jump.

**Catastrophic forgetting.** The narrow tune erodes general capability nobody thought to test. Defense: general-capability probe suite in the gate, mixed general data in training.

**Style drift laundered as success.** The tuned model *sounds* more on-brand, and a judge model rewards the confidence while factuality quietly drops. Defense: separate evals for correctness and style; never a single "quality" score.

**Unreproducible artifact.** Great model, unknown data, departed author. Defense: the reproducibility contract — data snapshot hash, base model + revision, config, seed — enforced at registry time, exactly as for any [model registry](../16-ml-systems/13-model-registry-metadata.md) entry.

**Stranded adapter.** The base model family moves two generations; the adapter is welded to obsolete weights and the data pipeline to regenerate it was never built. Defense: budget the refresh pipeline, not the one-off run; keep the training set (the durable asset) in better shape than the checkpoint.

**Template mismatch.** Training uses one chat template or tool serialization and serving uses another. The model appears inexplicably worse despite correct weights. Defense: version the complete inference tuple and test the rendered token sequence.

**Preference shortcut.** DPO learns that preferred responses are longer, more confident, or contain a recurring phrase. Defense: control pair differences, balance artifacts, and evaluate rubric dimensions independently.

**Packing leakage.** Tokens from one packed example attend to another or loss masks include user/tool text. Defense: test attention boundaries, position IDs, labels, and decoded training batches as first-class pipeline artifacts.

**Adapter cache thrash.** Per-tenant variants repeatedly load and evict, making TTFT unpredictable and storage a hot dependency. Defense: admission-aware adapter residency, prefetch, quotas, and a measured fallback policy.

**Unchecked feedback loop.** The model's own outputs become tomorrow's training truth, amplifying a mistake. Defense: provenance, expert correction, holdout evaluation, and limits on synthetic-data share by slice.

---

## Decision Framework

*Did a strong prompt with examples and schema enforcement fail, measurably, on an eval set?* If it wasn't tried, or there's no eval set, the answer to "should we fine-tune" is no — not yet.

*Is the gap knowledge or behavior?* Knowledge → retrieval. Behavior (style, format, task skill, tool reliability) → tuning is on the table.

*What's the unit economics goal?* First compare a smaller existing target, routing, prompt/context reduction, quantization, batching, and caching under the same quality constraints. Distillation is justified when a student can reproduce the bounded behavior more economically after including teacher generation, training, qualification, fallback, serving, and refresh cost. Its break-even model uses verified successful volume, not raw tokens alone.

*One model or many variants?* Per-tenant/per-task variants → LoRA + multi-adapter serving; single flagship behavior → merge and serve plain.

*Can you fund the loop?* Data refresh, re-training on base-model upgrades, eval maintenance, rollback capacity. A fine-tune is a product line, not a project.

---

## Key Takeaways

1. Qualify prompting, deterministic tools, and retrieval before training; a tune must produce a measured system-level gain over that baseline.
2. Use weights for stable behavior and representation changes, not as the authority for mutable or permissioned facts.
3. LoRA/QLoRA reduces trainable and base-weight state under explicit module, rank, precision, and sequence assumptions; it does not eliminate activation, evaluation, serving, or governance cost.
4. Preference and verifiable-reward objectives encode different feedback contracts and failure modes; choose them from label and verifier semantics rather than fashion.
5. Distillation is a bounded-distribution compression program whose business case depends on verified quality, fallback rate, volume, and refresh cost.
6. Data lineage, mixture weights, loss masks, contamination controls, and split semantics determine what behavior the run actually learns.
7. Multi-adapter serving trades duplicated bases for residency, compatibility, scheduling, isolation, and cache-management complexity.
8. A tuned model is a model release: registry lineage, offline gates including the prompting baseline, canary rollout, and re-qualification whenever the base model moves.

---

## References

1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
2. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
3. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
4. [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022 (RLHF)
5. [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al., 2022
6. [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al., 2023
7. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — DeepSeek, 2025 (GRPO, verifiable rewards)
8. [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285) — Sheng et al., 2023
9. [TRL — Transformer Reinforcement Learning](https://huggingface.co/docs/trl) — SFT/DPO/GRPO tooling
10. [Distilling Step-by-Step!](https://arxiv.org/abs/2305.02301) — Hsieh et al., 2023
