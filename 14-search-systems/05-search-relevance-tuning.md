# Search Relevance Tuning

## TL;DR

Search relevance tuning is the iterative process of improving search quality through query understanding, evaluation metrics, and experimentation. It involves understanding user intent, measuring quality with metrics like NDCG and MRR, running A/B tests, and continuously iterating based on feedback. Good relevance tuning requires both offline analysis and online experimentation.

---

## The Relevance Engineering Loop

### Continuous Improvement Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    Relevance Engineering Loop                    │
│                                                                 │
│                        ┌──────────────┐                         │
│                        │   MEASURE    │                         │
│                        │              │                         │
│                        │ • Metrics    │                         │
│                        │ • User       │                         │
│                        │   feedback   │                         │
│                        └──────┬───────┘                         │
│                               │                                 │
│               ┌───────────────┼───────────────┐                │
│               │               │               │                │
│               ▼               │               │                │
│        ┌──────────────┐       │       ┌──────────────┐         │
│        │   ANALYZE    │       │       │    DEPLOY    │         │
│        │              │       │       │              │         │
│        │ • Failure    │       │       │ • A/B test   │         │
│        │   analysis   │       │       │ • Gradual    │         │
│        │ • Query      │       │       │   rollout    │         │
│        │   segments   │       │       │              │         │
│        └──────┬───────┘       │       └──────▲───────┘         │
│               │               │              │                 │
│               │               │              │                 │
│               ▼               │              │                 │
│        ┌──────────────┐       │       ┌──────────────┐         │
│        │  HYPOTHESIZE │───────┴──────►│    BUILD     │         │
│        │              │               │              │         │
│        │ • Root cause │               │ • Feature    │         │
│        │ • Solutions  │               │   changes    │         │
│        │              │               │ • Model      │         │
│        │              │               │   updates    │         │
│        └──────────────┘               └──────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Query Understanding

### Query Classification

```python
class QueryClassifier:
    """
    Classify queries to apply different ranking strategies
    """
    
    def classify_intent(self, query):
        """
        Query intent types:
        - Navigational: User wants specific site ("facebook login")
        - Informational: User wants information ("how to cook pasta")
        - Transactional: User wants to do something ("buy iphone")
        - Local: User wants nearby results ("pizza near me")
        """
        features = self.extract_features(query)
        return self.intent_model.predict(features)
    
    def classify_complexity(self, query):
        """
        Query complexity:
        - Head: High volume, well-understood ("weather")
        - Torso: Medium volume, some ambiguity ("python")
        - Tail: Low volume, specific ("python asyncio connection pool timeout")
        """
        query_freq = self.get_query_frequency(query)
        
        if query_freq > 10000:  # Per day
            return "head"
        elif query_freq > 100:
            return "torso"
        else:
            return "tail"
    
    def detect_modifiers(self, query):
        """
        Detect query modifiers that affect ranking
        """
        modifiers = {
            'freshness': self.has_freshness_intent(query),  # "latest", "2024"
            'location': self.extract_location(query),        # "in tokyo"
            'price': self.has_price_intent(query),          # "cheap", "under $50"
            'comparison': self.has_comparison(query),        # "vs", "versus"
            'review': self.wants_reviews(query),            # "review", "rating"
        }
        return modifiers

# Example usage
classifier = QueryClassifier()

queries = [
    "facebook",                          # Navigational, head
    "how to learn python",              # Informational, head
    "buy macbook pro 2024",             # Transactional, torso
    "best italian restaurant downtown", # Local, transactional
    "asyncio semaphore timeout error",  # Informational, tail
]

for query in queries:
    intent = classifier.classify_intent(query)
    complexity = classifier.classify_complexity(query)
    mods = classifier.detect_modifiers(query)
    print(f"{query}: {intent}, {complexity}, {mods}")
```

### Query Rewriting

```python
class QueryRewriter:
    """
    Improve queries for better matching
    """
    
    def expand_synonyms(self, query):
        """
        Add synonyms to improve recall
        
        "cheap hotel" → "cheap OR budget OR affordable hotel OR accommodation"
        """
        tokens = tokenize(query)
        expanded = []
        
        for token in tokens:
            synonyms = self.synonym_dict.get(token, [])
            if synonyms:
                expanded.append(f"({token} OR {' OR '.join(synonyms)})")
            else:
                expanded.append(token)
        
        return ' '.join(expanded)
    
    def fix_spelling(self, query):
        """
        Correct typos while preserving intent
        
        "pythn tutrial" → "python tutorial"
        """
        corrections = []
        for token in tokenize(query):
            if token not in self.vocabulary:
                suggestion = self.spell_checker.correct(token)
                if self.is_confident(token, suggestion):
                    corrections.append(suggestion)
                else:
                    corrections.append(token)
            else:
                corrections.append(token)
        
        return ' '.join(corrections)
    
    def segment_query(self, query):
        """
        Identify meaningful segments
        
        "new york pizza" → ["new york", "pizza"] not ["new", "york", "pizza"]
        """
        # Use statistical phrase detection
        tokens = tokenize(query)
        segments = []
        i = 0
        
        while i < len(tokens):
            # Try longer segments first
            for length in range(min(4, len(tokens) - i), 0, -1):
                candidate = ' '.join(tokens[i:i+length])
                if candidate in self.known_phrases or self.is_entity(candidate):
                    segments.append(candidate)
                    i += length
                    break
            else:
                segments.append(tokens[i])
                i += 1
        
        return segments
    
    def remove_stopwords(self, query, aggressive=False):
        """
        Remove stopwords, but carefully
        
        "the who" should not become "who" (band name)
        "to be or not to be" should stay intact (famous quote)
        """
        if query.lower() in self.protected_phrases:
            return query
        
        tokens = tokenize(query)
        if aggressive:
            return ' '.join(t for t in tokens if t.lower() not in self.stopwords)
        else:
            # Keep stopwords that affect meaning
            return ' '.join(
                t for t in tokens 
                if t.lower() not in self.stopwords or self.is_meaningful_stopword(t, tokens)
            )
```

### Query Relaxation

```python
def relax_query_if_no_results(original_query, search_func):
    """
    Progressively relax query to find results
    """
    relaxation_strategies = [
        # Level 1: Try exact query
        lambda q: q,
        
        # Level 2: Remove quotes (phrase → terms)
        lambda q: q.replace('"', ''),
        
        # Level 3: Remove filters
        lambda q: remove_filters(q),
        
        # Level 4: Remove less important terms
        lambda q: keep_important_terms(q, top_k=3),
        
        # Level 5: Spell correction
        lambda q: spell_correct(q),
        
        # Level 6: Synonym expansion
        lambda q: expand_synonyms(q),
        
        # Level 7: Semantic search (vector)
        lambda q: f"~semantic:{q}",
    ]
    
    for i, strategy in enumerate(relaxation_strategies):
        relaxed = strategy(original_query)
        results = search_func(relaxed)
        
        if results:
            if i > 0:
                log_relaxation(original_query, relaxed, i)
            return results, relaxed, i
    
    return [], original_query, len(relaxation_strategies)

# Example
query = '"exact phrase match" site:example.com filetype:pdf'

# Level 0: No results
# Level 1: "exact phrase match" → No results  
# Level 2: exact phrase match → No results
# Level 3: exact phrase match → Found 5 results
```

---

## Offline Evaluation

### Building a Test Collection

```python
class RelevanceTestCollection:
    """
    Curated set of queries with judged results
    """
    
    def __init__(self):
        self.queries = []  # List of test queries
        self.judgments = {}  # query → {doc_id: relevance_label}
    
    def add_query(self, query, judged_docs):
        """
        Add a query with relevance judgments
        
        judgments: {doc_id: grade}
        grades: 0=Bad, 1=Fair, 2=Good, 3=Excellent, 4=Perfect
        """
        self.queries.append(query)
        self.judgments[query] = judged_docs
    
    def sample_queries_for_judging(self, query_log, n_samples=1000):
        """
        Sample representative queries for human judging
        """
        # Stratified sampling by query frequency
        head_queries = sample_by_frequency(query_log, 'head', n_samples // 3)
        torso_queries = sample_by_frequency(query_log, 'torso', n_samples // 3)
        tail_queries = sample_by_frequency(query_log, 'tail', n_samples // 3)
        
        return head_queries + torso_queries + tail_queries
    
    def pool_documents_for_judging(self, query, systems, k=100):
        """
        Pool top results from multiple systems for judging
        
        Ensures we judge documents that any system might return
        """
        pooled = set()
        for system in systems:
            results = system.search(query, top_k=k)
            pooled.update(doc.id for doc in results)
        
        return list(pooled)

# Judging guidelines
JUDGING_GUIDELINES = """
4 - Perfect: Exact answer to query, authoritative source
3 - Excellent: Highly relevant, comprehensive answer
2 - Good: Relevant, addresses query but not completely
1 - Fair: Marginally relevant, tangentially related
0 - Bad: Not relevant, spam, or broken
"""
```

### Computing Metrics

```python
import numpy as np
from collections import defaultdict

class RelevanceMetrics:
    
    @staticmethod
    def precision_at_k(retrieved, relevant, k):
        """
        Fraction of top-k that are relevant
        """
        retrieved_k = retrieved[:k]
        relevant_in_k = sum(1 for doc in retrieved_k if doc in relevant)
        return relevant_in_k / k
    
    @staticmethod
    def recall_at_k(retrieved, relevant, k):
        """
        Fraction of relevant docs in top-k
        """
        retrieved_k = set(retrieved[:k])
        return len(retrieved_k & relevant) / len(relevant)
    
    @staticmethod
    def average_precision(retrieved, relevant):
        """
        Average of precision@k for each relevant doc
        """
        if not relevant:
            return 0
        
        precisions = []
        relevant_found = 0
        
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                relevant_found += 1
                precisions.append(relevant_found / i)
        
        return sum(precisions) / len(relevant)
    
    @staticmethod
    def ndcg_at_k(retrieved, judgments, k):
        """
        Normalized Discounted Cumulative Gain
        """
        def dcg(rels):
            return sum(
                (2 ** rel - 1) / np.log2(i + 2)
                for i, rel in enumerate(rels)
            )
        
        # Actual relevance scores
        actual_rels = [judgments.get(doc, 0) for doc in retrieved[:k]]
        
        # Ideal relevance scores
        ideal_rels = sorted(judgments.values(), reverse=True)[:k]
        
        if dcg(ideal_rels) == 0:
            return 0
        
        return dcg(actual_rels) / dcg(ideal_rels)
    
    @staticmethod
    def err(retrieved, judgments, max_grade=4):
        """
        Expected Reciprocal Rank
        
        Models user stopping after finding satisfying result
        """
        p_stop = 0
        err_value = 0
        
        for i, doc in enumerate(retrieved, 1):
            grade = judgments.get(doc, 0)
            # Probability this result satisfies user
            p_satisfy = (2 ** grade - 1) / (2 ** max_grade)
            
            # ERR contribution
            err_value += (1 - p_stop) * p_satisfy / i
            
            # Update stopping probability
            p_stop += (1 - p_stop) * p_satisfy
        
        return err_value

def evaluate_system(system, test_collection, metrics=['ndcg@10', 'map', 'mrr']):
    """
    Evaluate a search system on test collection
    """
    results = defaultdict(list)
    
    for query in test_collection.queries:
        retrieved = system.search(query, top_k=100)
        judgments = test_collection.judgments[query]
        relevant = {doc for doc, grade in judgments.items() if grade >= 2}
        
        if 'ndcg@10' in metrics:
            results['ndcg@10'].append(
                RelevanceMetrics.ndcg_at_k(retrieved, judgments, 10)
            )
        
        if 'map' in metrics:
            results['map'].append(
                RelevanceMetrics.average_precision(retrieved, relevant)
            )
        
        if 'mrr' in metrics:
            rr = 0
            for i, doc in enumerate(retrieved, 1):
                if doc in relevant:
                    rr = 1 / i
                    break
            results['mrr'].append(rr)
    
    # Aggregate
    return {metric: np.mean(values) for metric, values in results.items()}
```

### Failure Analysis

```python
class FailureAnalyzer:
    """
    Analyze where and why search fails
    """
    
    def find_failures(self, system, test_collection, threshold=0.5):
        """
        Find queries where system performs poorly
        """
        failures = []
        
        for query in test_collection.queries:
            retrieved = system.search(query, top_k=10)
            judgments = test_collection.judgments[query]
            ndcg = RelevanceMetrics.ndcg_at_k(retrieved, judgments, 10)
            
            if ndcg < threshold:
                failures.append({
                    'query': query,
                    'ndcg': ndcg,
                    'retrieved': retrieved,
                    'judgments': judgments
                })
        
        return failures
    
    def categorize_failure(self, failure):
        """
        Categorize the type of failure
        """
        query = failure['query']
        retrieved = failure['retrieved']
        judgments = failure['judgments']
        
        relevant_docs = {d for d, g in judgments.items() if g >= 2}
        perfect_docs = {d for d, g in judgments.items() if g >= 4}
        
        # Check different failure modes
        if not any(d in judgments for d in retrieved):
            return "RECALL_FAILURE"  # Good docs not in index or not retrieved
        
        if perfect_docs and not any(d in perfect_docs for d in retrieved[:3]):
            return "RANKING_FAILURE"  # Perfect doc exists but ranked low
        
        if len(query.split()) > 5:
            return "LONG_QUERY"  # Complex query not handled well
        
        if self.has_ambiguous_intent(query):
            return "AMBIGUOUS_QUERY"  # Query has multiple interpretations
        
        if self.is_entity_query(query) and not self.entity_matched(query, retrieved):
            return "ENTITY_RECOGNITION"  # Failed to recognize entity
        
        return "OTHER"
    
    def generate_failure_report(self, failures):
        """
        Generate actionable failure report
        """
        categories = defaultdict(list)
        
        for failure in failures:
            category = self.categorize_failure(failure)
            categories[category].append(failure)
        
        report = []
        for category, items in sorted(categories.items(), key=lambda x: -len(x[1])):
            report.append({
                'category': category,
                'count': len(items),
                'percentage': len(items) / len(failures) * 100,
                'examples': items[:5],  # Top 5 examples
                'suggested_fixes': self.suggest_fixes(category, items)
            })
        
        return report
    
    def suggest_fixes(self, category, examples):
        """
        Suggest fixes for each failure category
        """
        suggestions = {
            'RECALL_FAILURE': [
                "Check if relevant documents are indexed",
                "Review tokenization and stemming",
                "Add synonym expansion",
                "Consider query relaxation"
            ],
            'RANKING_FAILURE': [
                "Review ranking features",
                "Check if BM25 parameters are tuned",
                "Add document quality signals",
                "Consider learning-to-rank model"
            ],
            'LONG_QUERY': [
                "Implement query segmentation",
                "Add phrase matching",
                "Consider semantic similarity"
            ],
            'AMBIGUOUS_QUERY': [
                "Add query classification",
                "Implement result diversification",
                "Consider personalization"
            ],
            'ENTITY_RECOGNITION': [
                "Improve NER model",
                "Add entity synonyms to index",
                "Consider knowledge graph"
            ]
        }
        
        return suggestions.get(category, ["Investigate further"])
```

---

## Online Experimentation (A/B Testing)

### Experiment Design

```python
class SearchExperiment:
    """
    A/B test for search changes
    """
    
    def __init__(self, name, hypothesis, metric_targets):
        self.name = name
        self.hypothesis = hypothesis
        self.metric_targets = metric_targets  # e.g., {'ctr': 0.02, 'ndcg': 0.01}
        self.variants = {}
        
    def add_variant(self, name, config, allocation):
        """
        Add experiment variant
        
        allocation: Percentage of traffic (0-100)
        """
        self.variants[name] = {
            'config': config,
            'allocation': allocation,
            'users': set(),
            'impressions': 0,
            'clicks': 0,
            'metrics': defaultdict(list)
        }
    
    def assign_user(self, user_id):
        """
        Deterministically assign user to variant
        
        Use hash for consistent assignment
        """
        hash_val = hash(f"{self.name}:{user_id}") % 100
        
        cumulative = 0
        for name, variant in self.variants.items():
            cumulative += variant['allocation']
            if hash_val < cumulative:
                variant['users'].add(user_id)
                return name
        
        return 'control'  # Fallback
    
    def log_impression(self, user_id, query, results, variant):
        """
        Log search impression
        """
        self.variants[variant]['impressions'] += 1
        
        # Log for detailed analysis
        self.log({
            'timestamp': time.time(),
            'user_id': user_id,
            'query': query,
            'variant': variant,
            'results': [r.id for r in results[:10]],
            'scores': [r.score for r in results[:10]]
        })
    
    def log_click(self, user_id, query, doc_id, position, variant):
        """
        Log click event
        """
        self.variants[variant]['clicks'] += 1
        
        self.log({
            'timestamp': time.time(),
            'user_id': user_id,
            'query': query,
            'variant': variant,
            'clicked_doc': doc_id,
            'position': position,
            'event': 'click'
        })

# Example experiment setup
experiment = SearchExperiment(
    name="bm25_k1_tuning",
    hypothesis="Increasing k1 from 1.2 to 1.5 will improve relevance for long documents",
    metric_targets={'ndcg@10': 0.02, 'ctr': 0.01}
)

experiment.add_variant('control', {'bm25_k1': 1.2}, allocation=50)
experiment.add_variant('treatment', {'bm25_k1': 1.5}, allocation=50)
```

### Statistical Analysis

```python
import scipy.stats as stats
import numpy as np

class ExperimentAnalyzer:
    
    def analyze_experiment(self, experiment, min_samples=1000):
        """
        Analyze A/B test results
        """
        control = experiment.variants['control']
        treatment = experiment.variants['treatment']
        
        if control['impressions'] < min_samples:
            return {'status': 'INSUFFICIENT_DATA'}
        
        results = {}
        
        # CTR analysis
        control_ctr = control['clicks'] / control['impressions']
        treatment_ctr = treatment['clicks'] / treatment['impressions']
        
        ctr_lift = (treatment_ctr - control_ctr) / control_ctr
        ctr_pvalue = self.proportion_test(
            control['clicks'], control['impressions'],
            treatment['clicks'], treatment['impressions']
        )
        
        results['ctr'] = {
            'control': control_ctr,
            'treatment': treatment_ctr,
            'lift': ctr_lift,
            'p_value': ctr_pvalue,
            'significant': ctr_pvalue < 0.05
        }
        
        # Per-query metrics (need logged data)
        for metric in ['ndcg@10', 'mrr']:
            if metric in control['metrics']:
                control_vals = control['metrics'][metric]
                treatment_vals = treatment['metrics'][metric]
                
                _, pvalue = stats.ttest_ind(control_vals, treatment_vals)
                
                results[metric] = {
                    'control': np.mean(control_vals),
                    'treatment': np.mean(treatment_vals),
                    'lift': (np.mean(treatment_vals) - np.mean(control_vals)) / np.mean(control_vals),
                    'p_value': pvalue,
                    'significant': pvalue < 0.05
                }
        
        return results
    
    def proportion_test(self, c1, n1, c2, n2):
        """
        Two-proportion z-test
        """
        p1 = c1 / n1
        p2 = c2 / n2
        p_pooled = (c1 + c2) / (n1 + n2)
        
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z = (p1 - p2) / se
        
        return 2 * (1 - stats.norm.cdf(abs(z)))
    
    def calculate_sample_size(self, baseline_rate, mde, alpha=0.05, power=0.8):
        """
        Calculate required sample size for experiment
        
        mde: Minimum Detectable Effect (relative)
        """
        effect = baseline_rate * mde
        
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate + effect
        p_avg = (p1 + p2) / 2
        
        n = (
            (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) +
             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        ) / (effect ** 2)
        
        return int(np.ceil(n))

# Example analysis
analyzer = ExperimentAnalyzer()

# Calculate required sample size
# Baseline CTR: 5%, want to detect 5% relative improvement
sample_size = analyzer.calculate_sample_size(
    baseline_rate=0.05,
    mde=0.05,  # 5% relative = 0.25% absolute
    alpha=0.05,
    power=0.8
)
print(f"Need {sample_size} samples per variant")  # ~62,000
```

### Interleaving

```python
class InterleavedExperiment:
    """
    Interleaving: More sensitive than A/B testing
    
    Show results from both systems interleaved,
    measure which system's results get more clicks
    """
    
    def interleave_team_draft(self, results_a, results_b, k=10):
        """
        Team Draft interleaving
        
        Alternately pick from each team, avoid duplicates
        """
        interleaved = []
        team_assignments = {}  # doc_id → 'A' or 'B'
        
        ptr_a, ptr_b = 0, 0
        turn = 'A'  # Could randomize first pick
        
        while len(interleaved) < k:
            if turn == 'A':
                while ptr_a < len(results_a):
                    doc = results_a[ptr_a]
                    ptr_a += 1
                    if doc.id not in team_assignments:
                        interleaved.append(doc)
                        team_assignments[doc.id] = 'A'
                        break
                turn = 'B'
            else:
                while ptr_b < len(results_b):
                    doc = results_b[ptr_b]
                    ptr_b += 1
                    if doc.id not in team_assignments:
                        interleaved.append(doc)
                        team_assignments[doc.id] = 'B'
                        break
                turn = 'A'
        
        return interleaved, team_assignments
    
    def credit_click(self, clicked_doc_id, team_assignments):
        """
        Credit click to appropriate team
        """
        return team_assignments.get(clicked_doc_id)
    
    def analyze_interleaving(self, impressions):
        """
        Analyze interleaving results
        
        impressions: list of (team_assignments, clicks)
        """
        wins_a = 0
        wins_b = 0
        ties = 0
        
        for assignments, clicks in impressions:
            clicks_a = sum(1 for c in clicks if assignments.get(c) == 'A')
            clicks_b = sum(1 for c in clicks if assignments.get(c) == 'B')
            
            if clicks_a > clicks_b:
                wins_a += 1
            elif clicks_b > clicks_a:
                wins_b += 1
            else:
                ties += 1
        
        total = wins_a + wins_b + ties
        
        # Binomial test (excluding ties)
        decisive = wins_a + wins_b
        if decisive > 0:
            p_value = stats.binom_test(wins_a, decisive, 0.5)
        else:
            p_value = 1.0
        
        return {
            'wins_a': wins_a,
            'wins_b': wins_b,
            'ties': ties,
            'win_rate_a': wins_a / total if total > 0 else 0,
            'win_rate_b': wins_b / total if total > 0 else 0,
            'p_value': p_value,
            'winner': 'A' if wins_a > wins_b else ('B' if wins_b > wins_a else 'TIE')
        }
```

---

## Tuning Techniques

### BM25 Parameter Tuning

```python
from sklearn.model_selection import ParameterGrid
import numpy as np

def tune_bm25_parameters(search_index, test_collection, param_grid=None):
    """
    Grid search for BM25 parameters
    
    k1: Controls term frequency saturation (1.0-2.0)
    b: Controls document length normalization (0.0-1.0)
    """
    if param_grid is None:
        param_grid = {
            'k1': [0.5, 0.75, 1.0, 1.2, 1.5, 2.0],
            'b': [0.0, 0.25, 0.5, 0.75, 1.0]
        }
    
    best_params = None
    best_ndcg = 0
    results = []
    
    for params in ParameterGrid(param_grid):
        search_index.set_bm25_params(**params)
        
        # Evaluate
        ndcgs = []
        for query in test_collection.queries:
            retrieved = search_index.search(query, top_k=10)
            judgments = test_collection.judgments[query]
            ndcg = RelevanceMetrics.ndcg_at_k(retrieved, judgments, 10)
            ndcgs.append(ndcg)
        
        avg_ndcg = np.mean(ndcgs)
        results.append({'params': params, 'ndcg': avg_ndcg})
        
        if avg_ndcg > best_ndcg:
            best_ndcg = avg_ndcg
            best_params = params
    
    return best_params, results

# Typical findings:
# - k1=1.2-1.5 works well for most corpora
# - b=0.75 is a good default (standard BM25)
# - Lower b (0.3-0.5) for collections with high length variance
# - Higher k1 (1.5-2.0) when term frequency is important
```

### Field Boosting

```python
def tune_field_boosts(search_index, test_collection):
    """
    Tune how much to weight different fields
    """
    # Grid search over field weights
    field_weights = {
        'title': [1.0, 2.0, 3.0, 5.0],
        'body': [1.0],  # Keep as baseline
        'description': [0.5, 1.0, 1.5],
        'tags': [0.5, 1.0, 2.0]
    }
    
    best_weights = None
    best_ndcg = 0
    
    for title_w in field_weights['title']:
        for desc_w in field_weights['description']:
            for tag_w in field_weights['tags']:
                weights = {
                    'title': title_w,
                    'body': 1.0,
                    'description': desc_w,
                    'tags': tag_w
                }
                
                ndcg = evaluate_with_weights(search_index, test_collection, weights)
                
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_weights = weights
    
    return best_weights

# Example Elasticsearch query with field boosting
BOOSTED_QUERY = {
    "query": {
        "multi_match": {
            "query": "python tutorial",
            "fields": [
                "title^3",      # Title matches worth 3x
                "body",         # Body is baseline
                "description^1.5",
                "tags^2"
            ],
            "type": "best_fields",
            "tie_breaker": 0.3
        }
    }
}
```

### Function Score Tuning

```python
def tune_freshness_boost(search_index, test_collection):
    """
    Tune how much to boost recent documents
    """
    # Different decay functions
    decay_configs = [
        # No freshness boost
        None,
        
        # Exponential decay (half-life of 30 days)
        {
            'type': 'exp',
            'scale': '30d',
            'decay': 0.5,
            'weight': 1.0
        },
        
        # Gaussian decay (peak at origin, smooth falloff)
        {
            'type': 'gauss',
            'scale': '60d',
            'decay': 0.5,
            'weight': 1.5
        },
        
        # Linear decay
        {
            'type': 'linear',
            'scale': '90d',
            'decay': 0.5,
            'weight': 2.0
        }
    ]
    
    results = []
    for config in decay_configs:
        ndcg = evaluate_with_freshness(search_index, test_collection, config)
        results.append({'config': config, 'ndcg': ndcg})
    
    return max(results, key=lambda x: x['ndcg'])

# Elasticsearch function_score example
FRESHNESS_BOOST_QUERY = {
    "query": {
        "function_score": {
            "query": {"match": {"content": "python"}},
            "functions": [
                {
                    "exp": {
                        "date": {
                            "origin": "now",
                            "scale": "30d",
                            "decay": 0.5
                        }
                    },
                    "weight": 1.5
                },
                {
                    "field_value_factor": {
                        "field": "popularity",
                        "modifier": "log1p",
                        "factor": 0.1
                    }
                }
            ],
            "score_mode": "sum",
            "boost_mode": "multiply"
        }
    }
}
```

---

## Monitoring and Alerting

### Key Metrics to Monitor

```python
class SearchMonitor:
    """
    Monitor search quality in production
    """
    
    def __init__(self):
        self.metrics_store = MetricsStore()
    
    def track_search(self, query, results, latency, user_id=None):
        """
        Track metrics for each search
        """
        metrics = {
            # Performance
            'latency_ms': latency,
            'result_count': len(results),
            
            # Quality signals
            'zero_results': len(results) == 0,
            'top_score': results[0].score if results else 0,
            'score_gap': self.score_gap(results),  # Gap between #1 and #2
            
            # Query characteristics
            'query_length': len(query.split()),
            'query_type': classify_query(query),
        }
        
        self.metrics_store.record(metrics)
    
    def track_engagement(self, query, clicked_position, dwell_time):
        """
        Track user engagement
        """
        metrics = {
            'clicked': True,
            'click_position': clicked_position,
            'dwell_time_seconds': dwell_time,
            'satisfied': dwell_time > 30,  # Proxy for satisfaction
        }
        
        self.metrics_store.record(metrics)
    
    def get_dashboard_metrics(self, time_range='1h'):
        """
        Aggregate metrics for dashboard
        """
        data = self.metrics_store.get_range(time_range)
        
        return {
            # Performance
            'p50_latency': np.percentile(data['latency_ms'], 50),
            'p99_latency': np.percentile(data['latency_ms'], 99),
            'qps': len(data) / time_range_seconds,
            
            # Quality
            'zero_result_rate': np.mean(data['zero_results']),
            'ctr': np.mean(data['clicked']),
            'avg_click_position': np.mean(data['click_position']),
            'satisfaction_rate': np.mean(data['satisfied']),
            
            # Trends (compare to previous period)
            'ctr_change': self.calculate_change('ctr', time_range),
            'zero_result_change': self.calculate_change('zero_result_rate', time_range),
        }

# Alert thresholds
ALERT_THRESHOLDS = {
    'zero_result_rate': {'warning': 0.05, 'critical': 0.10},
    'p99_latency': {'warning': 500, 'critical': 1000},  # ms
    'ctr': {'warning_drop': 0.10, 'critical_drop': 0.20},  # Relative change
}

def check_alerts(metrics, thresholds):
    """
    Check if any metrics breach thresholds
    """
    alerts = []
    
    for metric, limits in thresholds.items():
        value = metrics.get(metric)
        
        if value is None:
            continue
        
        if 'critical' in limits and value >= limits['critical']:
            alerts.append({
                'severity': 'CRITICAL',
                'metric': metric,
                'value': value,
                'threshold': limits['critical']
            })
        elif 'warning' in limits and value >= limits['warning']:
            alerts.append({
                'severity': 'WARNING',
                'metric': metric,
                'value': value,
                'threshold': limits['warning']
            })
    
    return alerts
```

### Automated Quality Checks

```python
class QualityChecker:
    """
    Automated checks for search quality regressions
    """
    
    def __init__(self, golden_queries):
        """
        golden_queries: List of (query, expected_top_results)
        """
        self.golden_queries = golden_queries
    
    def run_golden_query_check(self, search_func):
        """
        Check if expected results still appear in top positions
        """
        failures = []
        
        for query, expected in self.golden_queries:
            results = search_func(query, top_k=10)
            result_ids = [r.id for r in results]
            
            for expected_doc, expected_position in expected:
                if expected_doc not in result_ids:
                    failures.append({
                        'query': query,
                        'expected_doc': expected_doc,
                        'expected_position': expected_position,
                        'actual_position': None,
                        'error': 'MISSING'
                    })
                else:
                    actual_pos = result_ids.index(expected_doc)
                    if actual_pos > expected_position + 2:  # Allow some variance
                        failures.append({
                            'query': query,
                            'expected_doc': expected_doc,
                            'expected_position': expected_position,
                            'actual_position': actual_pos,
                            'error': 'POSITION_DROP'
                        })
        
        return failures
    
    def run_regression_test(self, old_system, new_system, test_queries):
        """
        Compare new system against old system
        """
        regressions = []
        improvements = []
        
        for query in test_queries:
            old_results = old_system.search(query)
            new_results = new_system.search(query)
            
            old_ndcg = calculate_ndcg(old_results)
            new_ndcg = calculate_ndcg(new_results)
            
            delta = new_ndcg - old_ndcg
            
            if delta < -0.1:  # Significant regression
                regressions.append({
                    'query': query,
                    'old_ndcg': old_ndcg,
                    'new_ndcg': new_ndcg,
                    'delta': delta
                })
            elif delta > 0.1:  # Significant improvement
                improvements.append({
                    'query': query,
                    'old_ndcg': old_ndcg,
                    'new_ndcg': new_ndcg,
                    'delta': delta
                })
        
        return {
            'regressions': regressions,
            'improvements': improvements,
            'net_change': np.mean([r['delta'] for r in regressions + improvements])
        }
```

---

## Best Practices

```
Query Understanding:
□ Implement spell correction with confidence thresholds
□ Use query classification for different ranking strategies  
□ Segment queries to identify entities and phrases
□ Have a query relaxation strategy for zero results

Offline Evaluation:
□ Build representative test collection (head + torso + tail)
□ Use graded relevance judgments (not just binary)
□ Report multiple metrics (NDCG, MRR, MAP)
□ Segment analysis by query type and complexity

Online Experimentation:
□ Calculate required sample size before launching
□ Use proper randomization (user-level, not request-level)
□ Guard against novelty effects (run for 2+ weeks)
□ Consider interleaving for sensitivity

Monitoring:
□ Track leading indicators (zero results, latency)
□ Set up alerts with clear ownership
□ Run automated golden query checks
□ Regular manual search quality reviews

Iteration:
□ Prioritize fixes by impact × effort
□ Document all changes and their effects
□ Build regression test suite over time
□ Establish a regular tuning cadence
```

---

## References

- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) - Stanford NLP
- [Relevance Engineering](https://opensourceconnections.com/blog/tag/relevance-engineering/)
- [Interleaving Methods for Search Evaluation](https://dl.acm.org/doi/10.1145/2505515.2505662)
- [How Not To Sort By Average Rating](https://www.evanmiller.org/how-not-to-sort-by-average-rating.html)
- [Controlled Experiments on the Web](https://www.exp-platform.com/Documents/GuideControlledExperiments.pdf)
