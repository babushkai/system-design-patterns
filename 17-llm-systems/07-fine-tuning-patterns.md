# Fine-Tuning Patterns

## Overview {#tl-dr}

Fine-tune when a trained model delivers a measurable benefit over your best prompt, retrieval, or tool-based baseline and your team can maintain the training data and release process. Keep changing facts and access-controlled records in systems that can enforce updates, deletions, and permissions at request time.

Suppose an adapter improves classification in offline tests. Before rollout, run the same cases with the tokenizer, template, quantization, and adapter loader that will serve traffic. Check the gain on held-out examples and measure latency when several adapters share a worker. Keep the previous serving configuration available until the candidate passes both checks.

---

## When to Fine-Tune (and When Not To)

Start with the defect you can reproduce and the cheapest intervention that could fix it.

**Prompt or interface.** If the model chooses a valid action but formats it incorrectly, test a schema and validator. Compare prompts with representative examples under the intended model, reasoning budget, and cache behavior.

**Evidence or tool.** If the answer depends on current stock, access rights, or an exact calculation, read the source at request time. The owner can then correct or revoke the record independently of a model deployment. Training on the same data expands the privacy boundary. [ML risk and governance](../16-ml-systems/09-ml-risk-governance.md)

**Training.** Consider changing weights when the model still makes a recurring behavioral error with the right evidence, or when a smaller model could deliver the same result at lower cost.

| Motivation | What to measure | Candidate intervention |
|---|---|---|
| Style/format lock-in | Stable examples can move repeated behavioral guidance into weights | Curated SFT, compared with schema/prompt baseline |
| Distillation for cost/latency | A smaller model may reproduce bounded teacher behavior on the target distribution | Teacher/corrected data plus student SFT and coverage tests |
| Deep domain adaptation | A measured representation or task gap remains after supplying evidence | Continued pretraining and/or SFT, compared with retrieval baseline |
| Reliability on a narrow task | Repeated behavior can be learned from governed corrections | SFT on representative successes, corrected failures, and abstentions |
| First-token latency | Time saved by removing examples, including on warm-cache requests | Compare tuned and prompted models on the actual traffic mix |

Give the project an owner for data refresh, qualification, rollback, and deletion requests. Include those costs in the proposal. A checkpoint you cannot reproduce or replace leaves the service dependent on one set of weights.

### Frame the tune as a falsifiable intervention

For example, propose raising valid tool selection on multilingual returns from 91% to 97% while limiting factual accuracy loss to 0.5 percentage points. If the business case also requires saving 1,200 median input tokens and 30% per successful case, test those outcomes on the same traffic. Agree on the decision thresholds before choosing the training data.

Record the full configuration for each comparison.

$$
Behavior = f(base\ revision, adapter, chat\ template, system\ prompt,
             tools, decoding, retrieval, runtime).
$$

A serving rollout can change tokenization, stop behavior, or tool formats while the adapter bytes stay identical. Replay the same labeled requests through both rendered configurations to isolate the change.

---

## The Method Landscape

### Supervised Fine-Tuning (SFT)

Supervised fine-tuning fits a next-token objective on examples of desired behavior. Choose which tokens receive loss. A prompt-completion dataset can train on completion tokens, and a chat dataset can restrict loss to assistant turns when the template marks those spans. Inspect the labels from an actual batch before starting an expensive run. [TRL SFT trainer](https://huggingface.co/docs/trl/en/sft_trainer)

### LoRA and QLoRA: Parameter-Efficient Mechanics

Full fine-tuning stores or shards gradients and optimizer state for the trainable base weights. LoRA freezes the base matrix $W$ and learns a low-rank update $BA$.

```text
h = Wx + (α/r)·B·A·x

For W in R^(d_out × d_in):
  full trainable parameters = d_out × d_in
  LoRA trainable parameters = r × (d_in + d_out)
```

Rank and target layer dimensions determine how many parameters train. QLoRA also stores the frozen base in quantized form and computes updates at higher precision. Measure peak memory at the required batch and sequence length. Activations, temporary buffers, and adapter optimizer state still consume device memory.

Compare ranks and target modules on the hard workload slices. Stop increasing rank when held-out performance stops improving. A larger adapter costs more to train, store, and load.

Merging $BA$ into $W$ removes the separate adapter operation at inference. Keep adapters separate when the fleet needs to switch variants without loading another full base. Qualify a merged or requantized artifact on its actual serving kernels.

### Preference Optimization: DPO and Friends

Demonstrations specify an answer to imitate. A preference pair records which of two answers the reviewer prefers for the same prompt. Direct Preference Optimization (DPO) uses those pairs and a reference model without online rollouts during fitting. Compare its annotation cost with the cost of collecting corrected demonstrations. Online RLHF adds live sampling, a reward signal, and policy optimization.

For outcomes a program can check, compare training on verified successes with optimizing a policy through repeated rollouts. The latter also requires a versioned environment, sampling capacity, and tests for ways to earn reward while failing the task.

For prompt $x$, chosen answer $y_w$, and rejected answer $y_l$, the DPO loss compares their log probabilities under the policy and frozen reference.

$$
\mathcal{L}_{DPO} = -\log \sigma \left(\beta
\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
- \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]\right).
$$

The coefficient $\beta$ scales the preference margin against the reference. Label the reason for each choice. If the better answer is always longer, the trainer has little evidence to distinguish correctness from length. Include close pairs and counterexamples, randomize display order, and measure annotator agreement.

Evaluate preferred style separately from factuality. If the same judge produced the training labels and scores the checkpoint, compare both candidates in a blinded human assessment and run an independent correctness check.

### Training with verifiable rewards

Group relative policy optimization (GRPO) samples several completions per prompt and compares rewards within the group. With a binary verifier, both all-pass and all-fail groups have zero reward variation. Inspect them separately. The former may be too easy. The latter may expose an impossible task, a verifier defect, or a capability gap. Test these explanations before buying more rollouts. Qualify the resulting policy with independent checks. [TRL GRPO trainer](https://huggingface.co/docs/trl/en/grpo_trainer)

### Continued pretraining and domain adaptation

Continued pretraining learns from unlabeled domain text. Test it when vocabulary, syntax, or representation appears to limit downstream performance even with the right evidence available. Deduplicate and screen the corpus, hold out sensitive material, and run general-capability checks after each stage. Follow with supervised or preference tuning only if the task still calls for it.

Lower perplexity on domain text says the model predicts that text better. Put the checkpoint through the product task before approving release.

### Distillation

A student learns a bounded behavior from teacher outputs, expert corrections, or both. Preserve the teacher revision, prompt, sampling settings, and verification results. Test where the student should abstain or escalate. Check the rights to train on and deploy the data before generating a large corpus.

Compare both systems over the same operating period and workload. With equal qualified completion rates, the number of incoming tasks needed to recover fixed costs is

$$
N_{break-even} =
\frac{C_{data}+C_{training}+C_{eval}+C_{platform}+C_{refresh}}
{C_{baseline/task}-C_{student/task}},
$$

where each variable cost includes retries and fallbacks per incoming task. For illustration, 12,000 in fixed costs and a saving of 0.03 per task break even at 400,000 tasks. If fallback expense cuts the saving to 0.01, the requirement rises to 1.2 million tasks. A zero or negative saving has no finite payback. If success rates differ, compare total cost for the same number of verified outcomes and keep the quality gates in force.

---

## Data Is the Product

Preserve enough information to reconstruct the training population and explain the inclusion or exclusion of each example.

**Curate for the target behavior.** Plot held-out gain as you add reviewed examples. Include hard negatives, rare policy cases, and legitimate refusals. The number of records alone says little about coverage.

**Sample actual traffic.** Collect successes as well as failures. Record which users saw which model and how examples were selected. Have an independent reviewer correct labels before admission. [Label systems](../16-ml-systems/10-label-ground-truth-systems.md)

**Separate related cases.** Assign one conversation or customer scenario to one split when testing generalization to unseen scenarios. Check near duplicates across splits. [Training pipelines](../16-ml-systems/05-training-pipelines.md)

**Version the inputs.** Record the immutable dataset manifest with the model. Remove sensitive material before creating the training snapshot, and retain enough lineage to locate affected artifacts after a correction. [Dataset management](../16-ml-systems/11-dataset-management-versioning.md)

**Match production formatting.** Pin role boundaries, tokenizer, chat template, tool serialization, and loss mask. Decode a training sample and compare it with the request the server actually sends.

### Dataset contract and lineage

Give each example a stable ID, source, authorized use, content hash, annotation provenance, policy version, and workload slice. Preserve tool-call/result pairing. Label simulated tool responses so an evaluator can distinguish them from observed behavior.

Seal exact shard references and transformation versions in a manifest. Include exclusions, tombstones, tokenizer, template, truncation, packing, sampling, and loss configuration. A mutable query cannot reproduce last month's run after the source changes.

### Mixture design

Sampling long support conversations and short classification tasks equally by example can give them very different token exposure. For dataset weights $w_j$ the nominal mixture is

$$
\mathcal{L} = \sum_j w_j E_{(x,y)\sim D_j}[\ell(x,y)],
\qquad \sum_j w_j = 1.
$$

Log realized examples, tokens, and tokens that actually receive loss by source. A dataset can dominate the batch by token count while contributing little supervised signal after masking.

Include clarification, abstention, denied access, tool failure, and successful recovery. An all-success dataset provides no examples of when the model should stop or request help.

### Splitting and contamination

Match the split to deployment. Hold out a customer to test generalization to new customers, or hold out later dates to test drift. Look for shared source documents, repeated scenarios, and paraphrases across accessible training and evaluation sets. For an opaque base corpus, record the residual uncertainty.

Keep a frozen qualification set away from training authors. Use a separate rotating set for incident-driven iteration. Once authors tune against a failure case, treat it as a regression test.

---

## Training Mechanics Worth Knowing

Choose a small experiment around a pinned baseline. Vary learning rate, training duration, sequence handling, and adapter size deliberately. Record effective supervised tokens per update so two runs with different padding or masking remain interpretable.

Falling training loss with rising held-out loss warrants an overfitting or distribution-shift check. Cross-split leakage can make validation look deceptively good. For a flat loss curve, inspect trainable parameters, label masks, gradient norms, and the learning rate before changing architecture.

Require the same evidence from a hosted run as from a local one wherever the provider exposes it. Record immutable inputs, environment, checkpoint integrity, recovery point, and the exact artifact evaluated. [Training pipelines](../16-ml-systems/05-training-pipelines.md)

### Memory accounting

Calculate memory separately for weights, gradients, Adam moments, optional master weights, activations, temporary buffers, and collectives. For example, two-byte weights, two-byte gradients, two four-byte moments, and four-byte master weights total 16 bytes per parameter before activations. Confirm which copies the implementation keeps.

- replicated data parallelism keeps a full local copy of model and optimizer state
- ZeRO stages progressively partition optimizer states, gradients, and parameters. FSDP memory depends on the sharding strategy and temporary all-gathers
- tensor and pipeline parallelism split computation when the model or one layer exceeds a device
- activation checkpointing trades recomputation for activation memory
- sequence/context parallelism addresses long-sequence activation pressure.

LoRA needs optimizer state for the adapter parameters while training still runs forward and backward computations through the frozen network. For a layer $W \in \mathbb{R}^{d_{out}\times d_{in}}$, the extra trainable parameter count is

$$
r(d_{in}+d_{out})
$$

Measure peak memory at the longest admitted sequence, the intended microbatch, and the actual precision and checkpointing settings.

### Sequence construction and loss semantics

Render a training dialogue and its production equivalent through the same tokenizer. Inspect system, user, assistant, tool, termination, and padding labels. A conversation ending in a long tool result may contain almost no supervised assistant tokens after truncation.

Packing reduces padding. For independent examples, verify their attention boundaries and position handling, and mask labels that would train a transition across examples. Position IDs alone cannot enforce isolation. Measure which languages, tools, and sequence lengths lose supervision under truncation.

### Reproducible training and recovery

Pin the dataset, sampling, model, tokenizer, code, kernel versions, topology, seed, precision, and schedule. Define the tolerance within which a repeat run should reproduce held-out outcomes.

Write model, optimizer, schedule, random state, and data cursor at the same logical step. Test recovery by interrupting a short run and resuming. Verify the next examples consumed, next learning rate, and checkpoint metrics against an uninterrupted run. Generating from restored weights tests inference only.

Report loss and supervised token counts by slice alongside throughput, overflow, skipped updates, and host stalls. For a rare action regression, check whether its examples were sampled and labeled before changing the optimizer.

---

## Serving Tuned Models

Choose how the serving fleet will identify, load, and retire each variant.

**Merged weights.** A popular stable variant can justify dedicated replicas and avoid a separate adapter operation. Budget storage and warm rollback capacity for the full model.

**Shared base.** Keep multiple compatible adapters over a common base when variants see sparse traffic. Measure cold-load latency, memory demand, and batch behavior using the observed tenant mix.

**Release.** Bind the adapter to the exact base, tokenizer, template, and tool versions. Shadow proposed actions without executing them, then canary with stable session assignment. Keep a qualified previous artifact available throughout rollout.

### Adapter control plane

Resolve every request to an immutable base, adapter, and configuration. The registry records the target modules, rank, scale, training lineage, permitted tenant, owner, and release status. Reject a mismatched base digest at load time.

Reserve memory and check the remaining request deadline before a cold adapter load. Measure load time and eviction by adapter and tenant. If a burst of rare variants displaces a hot one, test tenant quotas or a separate cold pool and measure both cohorts.

Validate shape, target modules, artifact format, and trusted signatures before loading an adapter. Use a non-executable serialization format and isolate untrusted variants. Test behavior under the policy the deployment will enforce.

### Quantization and compilation qualification

Treat quantization or compilation as a distinct release. If a new variant regresses, replay the same inputs through the reference and deployed kernels with identical tokenization. For methods that need calibration data, check its coverage by language, sequence length, and tool format.

## Evaluation, Release, and Feedback

Choose separate release gates for the failures users cannot absorb, such as an incorrect action or unauthorized disclosure. Track style, format, latency, calibrated abstention, and task success separately so one improvement cannot conceal a failed gate.

Tune on development data, lock the candidate, then evaluate on withheld cases. Pair incumbent and candidate on the same inputs and report uncertainty. Blind the evaluator and randomize display order for preference tests.

Shadow traffic with tool writes disabled. On canary traffic, record both assigned and served revisions and wait for delayed corrections. If requests routed to a tuned variant show worse results, check fallback exposure and the exact rendered template before blaming weights. Roll back the serving alias while diagnosing a failed release.

A positive rating can accompany a wrong answer. A successful tool call can act on the wrong resource. Sample production outcomes for independent review before feeding them into another tune. Track how much of each training slice came from the model itself. [Feedback loops](../16-ml-systems/04-model-monitoring.md)

## Privacy, Safety, and Model Governance

Document what the team may collect, train on, retain, and deploy. Minimize personal data before snapshotting and restrict access to raw traces, checkpoints, and intermediate caches.

Track which runs consumed a record. A source deletion leaves earlier checkpoints intact. Follow the applicable deletion process for those artifacts. Extraction or membership tests can reveal exposure, but a negative sample cannot certify erasure. For differential privacy, evaluate the privacy budget and utility trade-off on the actual training configuration.

Keep access checks, rate limits, isolation, and action approval in the runtime. Test the tuned model with changed prompts and tools, then verify that the external controls still hold when the model proposes an invalid action.

---

## Failure Modes

| What goes wrong | How to test it | Release decision |
|---|---|---|
| The improvement disappears against a stronger prompt | Pair the tune with the best serving baseline on the same cases | Keep the cheaper qualified system. |
| A model repeats a superseded fact | Correct the source and replay the case | Fetch mutable facts at request time. |
| Validation jumps while production stays flat | Trace documents, customers, and near duplicates across splits | Rebuild the holdout before trusting the score. |
| Style improves as factual or general capability declines | Score each dimension with independent checks | Block the failing slice and examine data mix. |
| The model trained successfully but produces wrong tool calls in serving | Compare actual prompt token IDs, delimiters, tool schemas, and loss masks | Fix the first divergence and replay the same cases. |
| A preference model rewards verbosity | Match answer lengths and randomize position in a blinded test | Recollect ambiguous pairs and judge correctness separately. |
| Training improves only when examples share a packed sequence | Inspect the attention boundary and first predicted label of each example | Fix packing and requalify the resulting checkpoint. |
| An adapter misses first-token deadlines | Separate load wait from queue and prefill time for warm and cold requests | Change residency or routing and measure displaced tenants. |
| A base upgrade strands the adapter | Restore the pinned base, rebuild from its manifest, and exercise rollback | Retain the compatible deployment until a replacement passes. |
| Generated errors recur in later releases | Follow teacher and feedback lineage across dataset versions | Quarantine unchecked examples and independently relabel affected slices. |

---

## Decision Framework

Start with labeled tasks and a qualified incumbent. If new evidence or a deterministic interface resolves the failure, measure that path in production conditions. If a stable behavior gap remains, run the smallest training experiment that could change the decision.

For a cheaper student, include teacher fallback and refresh in the break-even calculation. For many tenant variants, compare cold-load latency and isolation against dedicated replicas. Approve rollout only when the exact serving artifact passes the agreed slices, fits the operating budget, and has a tested rollback path.

---

## Key Takeaways

1. Identify the recurring behavior or cost gap with a deployed-quality baseline.
2. Treat source data, masks, sampling, and split boundaries as part of the learned behavior. Verify them on real batches.
3. Select the objective from the feedback available. Demonstrations, preferences, and verifiable rewards answer different questions.
4. Measure memory and latency on the actual base, adapter, precision, sequence length, and tenant mix.
5. Qualify the serving artifact and keep enough lineage to recover, roll back, and retrain.

---

## References

1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): Hu et al., 2021
2. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314): Dettmers et al., 2023
3. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290): Rafailov et al., 2023
4. [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155): Ouyang et al., 2022 (RLHF)
5. [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073): Bai et al., 2022
6. [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206): Zhou et al., 2023
7. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948): DeepSeek, 2025 (GRPO, verifiable rewards)
8. [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285): Sheng et al., 2023
9. [TRL - Transformers Reinforcement Learning](https://huggingface.co/docs/trl): SFT/DPO/GRPO tooling
10. [Distilling Step-by-Step!](https://arxiv.org/abs/2305.02301): Hsieh et al., 2023

---

## Related Chapters

- [Prompt Engineering](06-prompt-engineering.md): how to qualify the prompt baseline that a trained model must beat
- [Retrieval-Augmented Generation Systems](04-rag-patterns.md): the usual first alternative: put the knowledge in retrieval rather than in weights
- [Training Pipelines](../16-ml-systems/05-training-pipelines.md): a fine-tune is a training run, with every lineage obligation that implies
- [Dataset Management and Versioning](../16-ml-systems/11-dataset-management-versioning.md): an instruction dataset you cannot version cannot be retrained or audited
- [LLM Evaluation and Observability](10-llm-evaluation.md): telling a real capability gain apart from evaluation set contamination
