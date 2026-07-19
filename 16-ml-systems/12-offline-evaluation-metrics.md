# Offline Evaluation and Metric Design

## TL;DR

Offline evaluation is the cheapest place to reject a bad model and the most dangerous place to believe a good-looking one. It measures behavior on a frozen approximation of the world, using labels and splits that may be biased, delayed, leaked, stale, or misaligned with the product objective. The central discipline is to treat evaluation as a measurement system, not a notebook cell: define the decision the metric supports, pin the dataset and split, separate primary metrics from guardrails, evaluate slices, quantify uncertainty, check calibration and thresholds, and never confuse offline improvement with production impact. Offline evaluation answers **"is this candidate worth exposing to live traffic?"** It does not answer **"will this improve the business?"** That final causal question belongs to online experiments.

---

## The Job of Offline Evaluation

Offline evaluation exists because live traffic is expensive and risky. Before a model reaches shadow, canary, or experiment, the team needs evidence that it is not obviously worse than the incumbent. Offline evaluation provides that first filter.

The filter has three jobs:

1. **Reject regressions cheaply** — do not spend production risk on candidates that lose on historical data.
2. **Compare model variants quickly** — choose among architectures, features, hyperparameters, and training windows.
3. **Prepare deployment decisions** — estimate threshold behavior, calibration, slice risk, and expected operating trade-offs.

The key word is *filter*. Offline evaluation is not proof. It is a necessary but insufficient gate. The offline dataset is historical, logged under old policies, labeled by imperfect processes, and often missing the outcomes the new model would have caused. A candidate can win offline and lose online because the metric measured the wrong objective, the split leaked, the logs were biased by the incumbent model, or the world changed.

The correct posture is skeptical: offline evaluation should be hard to fool, but never trusted as the final verdict.

---

## Evaluation Is a Measurement System

A trustworthy offline evaluation has the same structure as any measurement system:

```mermaid
flowchart LR
    Q["Decision question"] --> DATA["Pinned eval dataset"]
    DATA --> SPLIT["Split / holdout policy"]
    SPLIT --> METRIC["Metric computation"]
    METRIC --> SLICE["Slice analysis"]
    SLICE --> UNC["Uncertainty"]
    UNC --> GATE["Promotion gate"]
```

Each stage can invalidate the result. If the decision question is vague, the metric will optimize the wrong thing. If the dataset is mutable, the result cannot be reproduced. If the split leaks, the score is inflated. If slice analysis is missing, the average hides harm. If uncertainty is absent, noise looks like progress. If the gate is informal, teams cherry-pick.

A production evaluation report should therefore be a versioned artifact:

```yaml
candidate_model: fraud_classifier:v42
baseline_model: fraud_classifier:v41
evaluation_dataset: fraud_eval:2026-06-24.3
label: transaction_fraud:v6
split: time_holdout_2026_05
primary_metric: recall_at_0.5_percent_fpr
guardrails:
  - precision_at_current_review_capacity
  - false_positive_rate_by_country
  - p99_inference_latency
uncertainty: bootstrap_95_ci
result: blocked
reason: japan_slice_fpr_regression
```

This is the evaluation analogue of a deployment contract. It makes the decision reproducible and reviewable.

An evaluation report has two identities. The **measurement definition** pins label semantics, split, metric implementation, slice predicates, uncertainty method, baseline-selection rule, and gate policy. The **measurement realization** pins the candidate and baseline hashes, evaluation manifest, runtime image, random seeds, and computed outputs. A result is comparable across runs only when the definition is compatible; reusing the metric name after changing its code or population creates false history.

Publication should be transactional in the same way as a [dataset version](./11-dataset-management-versioning.md#publication-protocol-a-dataset-version-is-a-transaction). Workers write metric shards and per-example predictions into a private attempt. A reducer checks expected shards, unique example ownership, candidate/baseline alignment, and numeric validity, then seals one report hash. The registry attaches that immutable report and the gate decision in one idempotent transition. A timed-out evaluator queries the report hash before retrying; it never “passes” a model from a partial directory.

## Labels Mature on a Clock

Labels are often delayed, censored, or selectively observed. A chargeback may arrive weeks after a transaction; churn is not knowable until an inactivity window closes; repayment outcomes exist only after a loan matures; moderation correctness may be reviewed only for appealed decisions. Evaluating recent examples before their outcomes mature systematically treats unresolved positives as negatives.

For every example define at least:

```text
prediction_time
label_observation_deadline = prediction_time + outcome_window
label_as_of                 = snapshot time of the label source
label_status                = observed | unresolved | censored | excluded
```

The evaluation watermark is the latest prediction time for which the required outcome window has closed and late-label policy has been applied. “Evaluate on last week” is invalid if the metric needs a month of outcome. Report both event count and independently sampled unit count after maturity filters; a large row count can still contain little information when many rows belong to the same entity.

Selective labels require a causal argument. A fraud label may be observed mainly for reviewed transactions, and repayment is observed only for approved applicants. Evaluating the candidate on those observed outcomes estimates performance on the incumbent policy's selected population, not on all decisions the candidate would make. Propensity weighting or doubly robust estimators can reduce bias when logging propensities and overlap assumptions are credible, but they can have extreme variance and cannot recover outcomes for actions with no support. The report must name the target population, observation mechanism, overlap diagnostics, and any clipped weights. When support is missing, the honest verdict is “not identifiable offline,” followed by controlled exploration or an online experiment—not a more elaborate metric.

---

## Start With the Decision, Not the Metric

Metrics are easy to compute and hard to choose. The correct metric depends on the product decision the model supports.

A fraud model does not merely classify fraud. It decides whether to allow, review, or block transactions under a review-team capacity constraint. A recommender does not merely rank clicked items. It chooses a slate that should improve long-term satisfaction without collapsing diversity. A medical risk model does not merely maximize AUC. It prioritizes patients for scarce intervention capacity.

This means the metric should match the decision surface:

| Decision type | Better metric family | Why |
|---|---|---|
| Binary action under fixed threshold | Precision, recall, FPR/FNR at threshold | Measures actual operating point |
| Review queue with fixed capacity | Precision@K, recall@K, lift@K | Capacity is the constraint |
| Ranking list | NDCG, MAP, MRR, recall@K | Position matters |
| Probability used by downstream policy | Calibration, Brier score, log loss | Probability meaning matters |
| Rare-event detection | PR-AUC, recall at fixed FPR | ROC-AUC can hide poor precision |
| Regression with asymmetric costs | Quantile loss, weighted error | Over- and under-prediction differ |
| Recommendation slate | Offline rank metrics + diversity/coverage | Set quality matters |

The anti-pattern is choosing the metric because it is standard. AUC is standard; it may be irrelevant. If the business can review only 10,000 cases per day, the metric that matters is quality in the top 10,000, not average ranking over every case. If the model output is used as a calibrated risk estimate, ranking metrics are insufficient because the probability value itself is a contract.

A design review should force metric choice into a table like this:

| System | Decision constraint | Primary metric | Guardrails | Diagnostics |
|---|---|---|---|---|
| Fraud review | 20K reviews/day, low false-negative tolerance | Precision@20K and recall at fixed FPR | false-positive rate by country, review SLA, chargeback dollars | PR curve, score deciles, feature drift |
| Credit underwriting | Legal/financial decision, calibrated risk | Expected cost under policy and calibration error | adverse-action reason quality, protected-slice disparity, appeal rate | ROC/PR-AUC, monotonicity checks |
| Search ranking | Position-sensitive relevance | NDCG@K or MRR | latency, coverage, diversity, zero-result rate | recall@K, per-query-class metrics |
| Recommendation slate | Long-term satisfaction, exploration | online experiment primary; offline NDCG/recall as filter | diversity, creator/item exposure, churn proxy | candidate recall, novelty, popularity bias |
| Forecasting | Asymmetric over/under cost | pinball loss or weighted absolute error | capacity stockout/overstock rate | residuals by horizon, seasonality error |
| Abuse moderation | Harm minimization with appeal path | recall at bounded false-positive/appeal load | protected-slice FPR, reviewer load, reversal rate | confusion matrix by policy class |

The primary metric should map to the production decision. Guardrails protect constraints the primary metric would happily violate. Diagnostics explain failures but should not silently become promotion criteria after results are known.

---

## Accuracy Is Usually the Wrong Metric

Accuracy is seductive because it is simple. It is also wrong for many production ML systems.

In imbalanced domains, accuracy mostly measures the majority class. If 99.9% of transactions are legitimate, a model that predicts "legitimate" for everything is 99.9% accurate and useless. Fraud, abuse, churn, medical diagnosis, security, and incident detection are all dominated by rare positive classes where accuracy hides the event of interest.

Even balanced accuracy can hide cost asymmetry. A false positive in fraud may send a legitimate customer to review; a false negative may lose money. A false positive in content moderation may censor speech; a false negative may expose users to harm. These errors do not have equal cost. The evaluation metric should encode the trade-off explicitly, or at least report the trade-off curve so a human can choose.

Confusion matrices are useful because they force this accounting:

```text
                 Predicted positive   Predicted negative
Actual positive       TP                    FN
Actual negative       FP                    TN
```

From this, derive the quantities the product actually cares about: false positives per day, missed fraud dollars, review queue load, appeal volume, blocked-user rate, intervention capacity. Metrics become meaningful when they are translated into operational consequences.

---

## Thresholds Are Part of the Model

Many models emit scores, but products take actions. The threshold or policy mapping score to action is part of the decision system and must be evaluated with the model.

A model can improve AUC while becoming worse at the current threshold. A model can be better calibrated but shift score distribution so that the old threshold doubles the review queue. A model can improve recall by flooding humans with false positives. Therefore every evaluation should report both threshold-free metrics and operating-point metrics.

```text
Threshold-free:
  ROC-AUC, PR-AUC, log loss, NDCG

Operating-point:
  precision at threshold 0.92
  recall at 0.5% FPR
  review volume per day
  false positives per 10K users
  expected cost under policy_v9
```

For high-impact systems, evaluate a **policy curve**:

| Threshold | Review volume/day | Precision | Recall | Estimated cost |
|---|---:|---:|---:|---:|
| 0.70 | 80,000 | 0.08 | 0.92 | high ops cost |
| 0.85 | 25,000 | 0.21 | 0.78 | balanced |
| 0.95 | 4,000 | 0.61 | 0.41 | misses too much |

This curve is more useful than one headline metric because it shows the trade space. It also protects against threshold migration bugs during deployment: if `v42` has a different score distribution than `v41`, the old threshold may not mean the old action rate.

Threshold migration should be explicit. If the incumbent policy reviews the top 25,000 transactions/day, the candidate threshold should often be selected by matching capacity, not by reusing the old numeric cutoff:

```text
incumbent threshold 0.85 on v41 → 25,000 reviews/day
candidate score distribution shifted upward
reusing 0.85 on v42 → 41,000 reviews/day  (ops overload)
capacity-matched v42 threshold → 0.91 for 25,000 reviews/day
```

A deployment bundle should therefore record both the model and policy:

```yaml
threshold_migration:
  baseline_model: fraud_classifier:v41
  candidate_model: fraud_classifier:v42
  old_policy: fraud_policy:v9
  migration_rule: match_review_capacity
  target_capacity_per_day: 25000
  selected_threshold: 0.91
  expected_precision: 0.27
  expected_recall: 0.74
  blocked_if:
    precision_delta_ci_low_below: 0
    fpr_slice_regression: true
```

A model version without its threshold policy is not a production decision system; it is only a score generator.

---

## Calibration: When the Number Must Mean Probability

A ranking model only needs ordering. A risk model often needs calibrated probabilities. If a fraud model outputs 0.8, downstream systems may interpret that as "80% fraud probability" and price review, blocking, or reserves accordingly. If the true rate among 0.8-scored examples is 30%, the model may rank well but mislead every policy that consumes it.

Calibration asks whether predicted probabilities match observed frequencies:

```text
Among examples scored 0.70-0.80, is the positive rate roughly 75%?
Among examples scored 0.90-1.00, is the positive rate roughly 95%?
```

Common metrics include Brier score, expected calibration error, calibration curves, and log loss. Calibration must be evaluated by slice because a model can be calibrated globally and miscalibrated for a country, device, tenant, or protected group.

Calibration also drifts when base rates change. A model trained when fraud prevalence was 1% may over- or under-estimate probabilities when fraud prevalence is 3%, even if ranking remains decent. This is why monitoring prediction distributions and delayed labels matters after deployment.

A concrete calibration report bins predictions and compares predicted risk to observed frequency:

| Score bin | Count | Avg predicted | Observed positive rate | Gap | Action |
|---|---:|---:|---:|---:|---|
| 0.00-0.10 | 800,000 | 0.04 | 0.05 | +0.01 | pass |
| 0.10-0.30 | 160,000 | 0.18 | 0.16 | -0.02 | pass |
| 0.30-0.60 | 35,000 | 0.43 | 0.31 | -0.12 | recalibrate |
| 0.60-1.00 | 5,000 | 0.78 | 0.52 | -0.26 | block for high-risk policy |

This table is more actionable than a single Brier score. It shows where the probability contract breaks and whether the break occurs near the thresholds that drive decisions. For regulated or financial decisions, calibration should also be reported by pre-declared slices; global calibration can hide a group-specific overestimate or underestimate.

The scalar summary of that table is expected calibration error — the traffic-weighted average of the per-bin gaps — and computing it makes clear how little machinery is involved:

```python
def ece(y_true, y_score, n_bins=10):
    bins = np.minimum((y_score * n_bins).astype(int), n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        gap = abs(y_score[mask].mean() - y_true[mask].mean())
        err += (mask.sum() / len(y_true)) * gap
    return err
```

Note the trap visible in the arithmetic: the huge low-score bin dominates the weighted average, so the headline ECE is 0.017 — "well calibrated" — while the high-score bins that actually drive blocking decisions are off by 12 and 26 points. For decision systems, report ECE *and* the per-bin table restricted to the score region where the policy acts.

When calibration is broken but ranking is fine, the fix is a post-hoc recalibrator — a small monotone function fitted on a held-out calibration split (never on training data), shipped as part of the model bundle:

```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(scores_calib, labels_calib)     # held-out split, not train
p_calibrated = calibrator.predict(scores_prod)
```

Isotonic regression can fit flexible monotone distortions but may overfit when calibration data is sparse in the score region that matters. Platt scaling imposes a smoother logistic shape and can have lower variance when that shape is adequate. There is no universal sample-count crossover: select the method on an untouched calibration/evaluation split using the operating region and slices the policy consumes. Either way the calibrator is a versioned artifact with the same lifecycle as a threshold policy: evaluated by slice, monitored against mature labels, and rolled back with the model. A calibrator fitted under one base rate can become wrong when prevalence changes — which is why the prediction-distribution and delayed-label monitors in [model monitoring](./04-model-monitoring.md) are also the calibrator's health checks.

---

## Ranking Evaluation: Positions, Candidates, and Missing Counterfactuals

Ranking metrics evaluate ordered lists, not independent examples. NDCG, MRR, MAP, recall@K, hit rate, and precision@K all encode position: an item at rank 1 matters more than the same item at rank 20.

The hidden issue is candidate availability. Offline ranking evaluation typically uses logged impressions or sampled negatives. That means the evaluation only knows about items the previous system retrieved or showed. A new retrieval model may find excellent candidates the old system never logged; offline evaluation may not credit it. Conversely, sampled negatives may be too easy, inflating metrics.

For recommenders and search systems, a serious offline report should state:

1. candidate source used for evaluation,
2. negative sampling strategy,
3. whether exposure and position bias are corrected,
4. whether metrics are computed per user/query then averaged,
5. coverage and diversity metrics, not only relevance.

A ranking metric without candidate-set context is incomplete. If the candidate generator changed, ranking evaluation over the old candidate set answers the wrong question.

NDCG, the workhorse, rewards placing high-relevance items early, with a logarithmic position discount — computing one small example removes the mystique:

```text
Query with graded relevances; model ranks items in order [3, 2, 0, 1]  (rel of each shown item)

DCG@4  = 3/log2(2) + 2/log2(3) + 0/log2(4) + 1/log2(5) = 3.00 + 1.26 + 0 + 0.43 = 4.69
Ideal ranking [3, 2, 1, 0]:
IDCG@4 = 3.00 + 1.26 + 0.50 + 0 = 4.76
NDCG@4 = 4.69 / 4.76 ≈ 0.985
```

Two computation conventions silently change results across teams and libraries: whether gains are linear (`rel`) or exponential (`2^rel − 1`), and whether queries with no relevant items are skipped or scored zero. An "NDCG improvement" between two runs that changed convention is a bug, not a result — the evaluation harness should pin both choices in the metric's definition the same way a label definition is versioned.

---

## Leakage: The Evaluation That Certifies a Broken Model

Leakage happens when evaluation examples contain information that would not be available at prediction time. It is worse than ordinary data bugs because it makes the model look better.

Common leakage sources:

- random splits for time-dependent problems,
- joining latest feature values instead of point-in-time values,
- using post-action fields as features,
- duplicates across train and test,
- same user or entity appearing in both train and test when generalization to new entities matters,
- labels or label proxies included as features,
- preprocessing fit on the full dataset before splitting.

A good evaluation pipeline runs leakage checks automatically:

```text
- no feature availability_time > prediction_time
- no entity overlap for entity-disjoint split
- no duplicate content hashes across train/test
- no suspicious single-feature AUC near 1.0
- preprocessing fit only on train split
- label columns excluded from feature registry
```

The suspicious-feature check is crude but useful: if one feature alone makes the model nearly perfect in a hard domain, assume leakage until proven otherwise. Real-world prediction is rarely that easy.

---

## Slice Evaluation: Averages Lie

Aggregate metrics hide regressions. A candidate can improve overall AUC while harming new users, a small country, a protected class, a large tenant, or high-value transactions. If that slice is important, the aggregate is not a defense.

Slices should be pre-declared, not discovered only after a metric looks good. Common slices:

- geography and language,
- device and platform,
- new vs returning users,
- traffic source,
- tenant or merchant,
- item/content category,
- amount or risk band,
- protected or regulated groups where legally and ethically appropriate.

The challenge is multiple comparisons. If you inspect hundreds of slices, some will regress by chance. The solution is not to avoid slices; it is to separate **guardrail slices** from **exploratory slices**. Guardrail slices are pre-registered and can block promotion. Exploratory slices generate hypotheses and require confirmation.

A useful report format:

| Slice | Baseline metric | Candidate metric | Delta | CI | Gate |
|---|---:|---:|---:|---:|---|
| all traffic | 0.812 | 0.821 | +0.009 | [+0.005,+0.013] | pass |
| new users | 0.744 | 0.731 | -0.013 | [-0.022,-0.004] | fail |
| JP | 0.801 | 0.797 | -0.004 | [-0.009,+0.001] | watch |

The point is to make harm visible before deployment, when fixing it is cheapest.

---

## Uncertainty: Do Not Ship Noise

Offline metrics are estimates. A reported AUC of 0.812 is not a fact about the universe; it is an estimate on a finite sample. If the candidate scores 0.813, the difference may be noise.

Evaluation should report confidence intervals or uncertainty estimates. Bootstrap resampling is often sufficient: resample users or entities, recompute the metric, and report the distribution of deltas. The resampling unit matters. For recommender systems, resample users, not rows, because interactions from one user are correlated. For marketplace experiments, resample markets or clusters when those are the independent units.

```text
candidate - baseline PR-AUC = +0.004
95% bootstrap CI = [-0.001, +0.009]
verdict = inconclusive, not pass
```

A minimal bootstrap algorithm is simple enough to make part of the evaluation contract:

```text
unit = customer_id                    # not row_id if rows are correlated
for b in 1..1000:
    sample units with replacement
    compute metric(candidate, sample)
    compute metric(baseline, sample)
    store delta_b
ci_95 = percentile(delta, [2.5, 97.5])
pass if ci_95.lower > minimum_practical_effect
```

The `minimum_practical_effect` matters. With very large evaluation sets, tiny deltas can be statistically confident and operationally irrelevant. A candidate that improves AUC by 0.0002 but adds 20ms p99 latency, larger serving cost, and migration risk should not pass just because the p-value is impressive.

The promotion gate should evaluate the delta and its uncertainty, not only the point estimate. A small noisy win is not a win. This discipline prevents model teams from shipping random variation as progress.

Compare candidate and baseline on the **same resampled units**. A paired delta preserves the correlation between their errors and is usually much more precise than computing two independent intervals and subtracting endpoints. For time series or spatially correlated populations, iid bootstrap is still optimistic; use blocks or the natural cluster that preserves dependence. For very rare failures, report the count and exposure behind a rate so a narrow-looking interval does not hide a brittle approximation.

Model selection adds another layer of uncertainty. If a team trains many candidates and promotes the best score on one holdout, that holdout has become part of training even though gradients never touched it. Keep a final lockbox for the last comparison, limit who can inspect per-example errors, and record the search that selected the candidate. Repeated training seeds estimate optimization variance; repeated data windows estimate temporal variance. Neither substitutes for the other. A gate can require a robust statistic across seeds or windows when training instability is material, rather than letting one lucky run represent the model family.

Slice multiplicity is a policy choice. Pre-declared safety slices may each have a hard non-regression margin. Exploratory scans can control false discovery or rank findings for confirmation, but should not retroactively redefine the primary gate. Hierarchical shrinkage can stabilize tiny-slice estimates, provided the report still exposes raw counts and the model assumptions. “No significant harm” on an underpowered slice is absence of evidence, not evidence of safety.

---

## Baselines: Beat the Right Thing

Every evaluation needs baselines. The strongest baseline is the current production model evaluated on the same dataset. Without it, the team cannot distinguish absolute quality from improvement.

Useful baselines include:

1. **Current production model** — the real incumbent.
2. **Simple heuristic** — catches overengineered models that barely beat rules.
3. **Previous training run with same code** — estimates training variance.
4. **Ablation models** — measure whether a feature group actually helps.
5. **Oracle-ish upper bound** where available — estimates room for improvement.

The heuristic baseline is underrated. If a complex ML system barely beats "rank by popularity" or "review transactions above amount threshold," it may not justify its operational cost. ML should earn complexity.

---

## Cost-Sensitive Evaluation

Many production decisions have asymmetric and nonuniform costs. A false positive on a USD 5 transaction is not the same as a false positive on a USD 5,000 transaction. A false negative for severe abuse is not the same as a false negative for mild spam.

Cost-sensitive evaluation translates confusion-matrix cells into business impact:

```text
expected_cost = FP_count × cost_FP
              + FN_count × cost_FN
              + review_count × cost_review
              + latency_cost
              + fairness_or_policy_penalties
```

The exact numbers may be uncertain, but writing them down forces the trade-off into the open. It also reveals when a metric is misaligned. Optimizing recall without review cost may choose a model the operations team cannot run. Optimizing revenue without complaint cost may choose a model users hate.

Cost models should be versioned because product policy changes. A model approved under `cost_policy_v3` may not be approved under `v4`.

## Evaluation Execution at Scale

Large evaluation is a distributed join followed by aggregation. The join key is example identity; candidate prediction, baseline prediction, label, weight, slice attributes, and decision context must refer to the same observation. Joining by row position or allowing independent filters on candidate and baseline creates a silent paired-population bug. The reducer should account for every manifest row:

```text
N_manifest = N_scored + N_explicitly_excluded + N_failed
N_failed must satisfy the gate's failure policy
candidate_example_ids = baseline_example_ids for paired metrics
```

Metric aggregation must match the mathematical statistic. Counts and sums merge exactly; AUC, quantiles, top-K ranking, and per-query NDCG generally cannot be obtained by averaging shard-level answers. Either collect sufficient statistics with a proven merge rule or repartition by the metric's unit. For ranking, all candidates for a query belong to one reducer. For global top-K, local top-K sets can be merged only when each shard covers a disjoint population under the same score ordering. Approximate algorithms must record their error bound and configuration as part of the metric definition.

The harness owns deterministic preprocessing and model invocation, not the model-training notebook. It loads artifacts by hash, validates the serving signature, captures runtime and hardware, bounds malformed outputs, and stores enough per-example results to explain aggregate movement under access controls. A small “golden” corpus with hand-checked outputs catches metric-code and serialization regressions; metamorphic tests check invariants such as permutation independence for classification metrics or score-monotonicity for threshold sweeps. The harness itself is production software because a bug in it can promote every bad candidate consistently.

## Governance and Evaluation Integrity

The evaluation set is a high-value secret in any system where teams can iterate against its answers. Unrestricted access produces test-set overfitting; in high-impact settings it can also expose sensitive labels or protected attributes. Separate permission to submit predictions from permission to read per-example outcomes. Return the minimum report needed for iteration, log queries and slice creation, rate-limit repeated lockbox evaluation, and rotate or refresh holdouts when their information has leaked into development.

Promotion evidence should be tamper-evident and attributable: immutable candidate/baseline hashes, evaluation manifest, metric-definition hash, code/runtime digest, report hash, gate policy version, actor or workload identity, and decision timestamp. Protected-attribute slices may require a controlled evaluation service that releases aggregates subject to minimum-count and privacy policy while preserving an audit path for authorized reviewers. This is a separation-of-duties problem, not a reason to omit safety analysis.

An attacker can target the evaluator as well as the model—inject easy examples, suppress hard labels, manipulate weights, or exploit NaN handling so a guardrail disappears. Fail closed on missing required metrics, non-finite values, unexpected population loss, or unsigned inputs. A result with compromised measurement provenance is not “inconclusive”; it is invalid.

---

## Offline-to-Online Gap

Offline metrics fail to predict online impact for structural reasons:

1. **Logged-policy bias** — the evaluation data was generated by the old model.
2. **Feedback loops** — the new model changes future data.
3. **Proxy mismatch** — offline label is not the real product goal.
4. **Distribution shift** — production traffic has moved.
5. **Interference** — users/items/markets affect each other.
6. **Implementation skew** — serving computes features differently.

This is why offline evaluation should gate exposure, not replace online experiments. The handoff should be explicit:

```text
Offline pass → shadow for runtime safety → canary for operational guardrails → A/B for causal impact
```

A team that ships directly from offline metrics is assuming away the entire reason production ML is hard.

The handoff is a typed claim manifest, not a checklist. It records the target population and logging policy, label cutoff/maturity and censoring rule, feature and candidate-set provenance, estimand, metric and slice uncertainty, threshold/calibration policy, and the claims the offline design cannot identify. Deployment consumes this manifest to choose the next evidence boundary: shadow for runtime parity, canary for bounded live-path safety, and randomized or otherwise identified measurement for causal product impact. A missing or unsupported claim remains explicit rather than being converted into a generic pass.

---

## Failure Modes

**Metric mismatch** optimizes what is easy to label rather than what matters: clicks instead of satisfaction, approvals instead of repayment, reports instead of true abuse. Defense: metric hierarchy with primary, guardrail, and diagnostic metrics tied to the product decision.

**AUC worship** celebrates threshold-free ranking improvement while the operating threshold, review capacity, or calibrated probability behavior worsens. Defense: report operating-point and policy metrics.

**Leaky evaluation** certifies a broken model because future information entered features, splits, or preprocessing. Defense: point-in-time joins, honest splits, duplicate checks, and suspicious-feature audits.

**Average hides harm** ships a model that improves aggregate quality while degrading an important slice. Defense: pre-registered slice guardrails and uncertainty-aware deltas.

**Noise shipped as progress** promotes tiny metric changes without confidence intervals or training-variance checks. Defense: bootstrap CIs, repeated seeds, and minimum practical effect thresholds.

**Old-candidate-set evaluation** penalizes a new retrieval system because evaluation only includes items the old system found. Defense: evaluate retrieval and ranking separately and state candidate-set construction.

**Offline-online surprise** happens when a model wins offline and loses online because logs were biased or the proxy metric was wrong. Defense: treat offline as a filter and require progressive rollout plus experiment for impact.

**Immature-label optimism** counts unresolved delayed outcomes as negatives and makes the newest window look best. Defense: outcome-window watermarks, explicit censoring state, and label-as-of snapshots.

**Selective-label blindness** evaluates only cases the incumbent chose to review or approve, then generalizes to the whole population. Defense: state the observation policy and target population, inspect propensity overlap, and declare unsupported decisions non-identifiable offline.

**Distributed metric corruption** averages non-mergeable shard metrics or compares candidate and baseline on different examples. Defense: exact accounting, paired example IDs, and metric-specific sufficient-statistic or repartition logic.

**Holdout becomes training data** occurs when repeated candidate and slice queries adapt development to the test set. Defense: submission/read separation, query audit, a final lockbox, and periodic holdout refresh.

**Fail-open evaluator** drops NaN, missing, or unauthorized shards and still emits a passing summary. Defense: closed population accounting and typed invalid-report state; absence of a required metric is a gate failure.

---

## Decision Framework

First classify what the evidence can identify:

| Evidence condition | Defensible conclusion | Required next step |
|---|---|---|
| Mature labels, honest split, same decision support as production | Comparative offline quality on the represented population | Shadow/canary for implementation and runtime risk |
| Logged under a different but overlapping policy with valid propensities | Policy-value estimate with estimator assumptions and uncertainty | Sensitivity analysis, then constrained online validation |
| Outcomes observed only for selected actions with poor overlap | Quality on the selected subset only | Exploration, additional labeling, or causal experiment |
| Retrieval/candidate universe changed | Component evidence, not end-to-end ranking quality | Evaluate retrieval coverage and validate live behavior |
| Proxy metric differs from the product objective | Safety/filter evidence only | Online experiment for causal product impact |
| Metric definition, population, or report integrity changed unexpectedly | No valid comparison | Repair and republish measurement before promotion |

Then verify the decision bundle:

1. The production action, target population, label clock, and generalization question are explicit.
2. Candidate and incumbent are evaluated on the same immutable, point-in-time-correct examples using compatible definitions.
3. Threshold, calibration, capacity, and cost are evaluated at the policy's operating region.
4. Required slices have declared margins, enough evidence for the intended claim, and visible raw counts.
5. The paired delta accounts for sampling, clustering, training seeds, and selection where those sources matter.
6. The report is complete, immutable, attributable, and linked to the registry transition it controls.
7. Remaining offline-to-online assumptions are assigned to shadow, canary, or experiment rather than silently accepted.

Offline evaluation that answers these well is a trustworthy gate. Offline evaluation that cannot answer them is a chart, not evidence.

---

## Key Takeaways

1. Offline evaluation is a cheap rejection filter, not proof of production impact.
2. Treat evaluation as a measurement system with pinned data, declared metrics, slice analysis, uncertainty, and a gate.
3. Start with the decision the model supports; choose metrics that reflect the operating constraint.
4. Accuracy is often useless for imbalanced or asymmetric-cost systems.
5. Thresholds and policies are part of the model's behavior and must be evaluated together with the artifact.
6. Calibration matters whenever downstream systems interpret scores as probabilities.
7. Ranking metrics require candidate-set and exposure context; otherwise they answer an incomplete question.
8. Leakage makes bad models look excellent; point-in-time correctness and honest splits are non-negotiable.
9. Averages hide harmed slices; pre-register guardrail slices and report uncertainty.
10. Bootstrap confidence intervals should use the true independent unit and a minimum practical effect, not only a p-value.
11. Threshold migration is part of deployment; match the policy constraint instead of blindly reusing numeric cutoffs.
12. Offline wins must pass through shadow, canary, and experiments before being treated as production wins.
13. Delayed and selective labels define what can be learned offline; unresolved outcomes and unsupported actions are not negatives.
14. Distributed evaluation must preserve paired example identity and use aggregation valid for the statistic.
15. The evaluator is a governed production system: publish atomically, fail closed, protect holdouts, and attest evidence.

---

## References

1. [Rules of Machine Learning: Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml) — Zinkevich
2. [Hidden Technical Debt in Machine Learning Systems](https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — Sculley et al., 2015
3. [The ML Test Score: A Rubric for ML Production Readiness](https://research.google/pubs/pub46555/) — Breck et al., 2017
4. [Data Validation for Machine Learning](https://mlsys.org/Conferences/2019/doc/2019/167.pdf) — Breck et al., 2019
5. [Trustworthy Online Controlled Experiments](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/6A3B263E7114E81B95669A95B219C1D8) — Kohavi, Tang & Xu, 2020
6. [Offline Evaluation for Recommender Systems](https://dl.acm.org/doi/10.1145/1864708.1864721) — recommender evaluation and bias context
7. [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://arxiv.org/abs/1811.12808) — Raschka, 2018
8. [Counterfactual Risk Minimization: Learning from Logged Bandit Feedback](https://proceedings.mlr.press/v37/swaminathan15.html) — Swaminathan & Joachims, 2015
9. [Unbiased Learning-to-Rank with Biased Feedback](https://doi.org/10.1145/3018661.3018699) — Joachims, Swaminathan & Schnabel, 2017
10. [Online Experiments](./08-online-experiments.md)
