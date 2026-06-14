# LLM Infrastructure

## TL;DR

Production LLM infrastructure spans model serving (continuous batching, PagedAttention, prefix caching, speculative decoding, prefill/decode disaggregation), caching (provider prompt caching first, semantic caching with care), structured outputs via constrained decoding, evaluation, cost optimization (cascades, quantization, batch tiers), and guardrails. The serving layer is dominated by two regimes — compute-bound prefill and memory-bandwidth-bound decode — and almost every optimization in the modern stack (vLLM, SGLang, TensorRT-LLM) is about keeping GPUs busy across that asymmetry. Measure TTFT, inter-token latency, and goodput under SLO, not just requests per second.

---

## The Infrastructure Challenge

```mermaid
graph TD
    CH["LLM INFRASTRUCTURE CHALLENGES"]
    CH --> LAT["LATENCY<br/>Model loading<br/>Inference<br/>Network"]
    CH --> COST["COST<br/>Token costs<br/>GPU compute<br/>Overprovisioning"]
    CH --> REL["RELIABILITY<br/>Rate limits<br/>API failures<br/>Model updates"]
    CH --> SAF["SAFETY<br/>Harmful output<br/>Prompt injection<br/>Data leakage"]
    CH --> EVAL["EVALUATION<br/>Quality drift<br/>Regressions<br/>A/B testing"]
    CH --> SC["SCALE<br/>Traffic spikes<br/>Multi-region<br/>Concurrency"]
```

---

## Model Serving Architecture

### Basic Serving Infrastructure

```mermaid
graph TD
    LB["Load Balancer"]
    LB --> GW1["Gateway Server<br/>(FastAPI)"]
    LB --> GW2["Gateway Server<br/>(FastAPI)"]
    LB --> GW3["Gateway Server<br/>(FastAPI)"]

    GW1 & GW2 & GW3 --> RQ["Request Queue<br/>(Redis)"]

    RQ --> W1["Inference Worker<br/>(GPU Node)<br/>vLLM / TGI"]
    RQ --> W2["Inference Worker<br/>(GPU Node)<br/>vLLM / TGI"]
    RQ --> W3["Inference Worker<br/>(GPU Node)<br/>vLLM / TGI"]
```

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import uuid

app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    model: str = "llama-3-70b"
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = False

class CompletionResponse(BaseModel):
    id: str
    choices: List[dict]
    usage: dict

class LLMGateway:
    """Gateway service for LLM requests."""
    
    def __init__(self, config):
        self.request_queue = RequestQueue(config.redis_url)
        self.model_router = ModelRouter(config.models)
        self.rate_limiter = RateLimiter(config.rate_limits)
        self.cache = SemanticCache(config.cache_url)
    
    async def complete(self, request: CompletionRequest, user_id: str) -> CompletionResponse:
        """Process completion request."""
        
        # Rate limiting
        if not await self.rate_limiter.allow(user_id):
            raise RateLimitExceeded()
        
        # Check cache
        cached = await self.cache.get(request.prompt, request.model)
        if cached:
            return cached
        
        # Route to appropriate model/backend
        backend = await self.model_router.route(request)
        
        # Queue request
        request_id = str(uuid.uuid4())
        result = await self.request_queue.enqueue_and_wait(
            request_id=request_id,
            backend=backend,
            request=request
        )
        
        # Cache result
        await self.cache.set(request.prompt, request.model, result)
        
        return result


class RequestQueue:
    """Manages request queuing and batching."""
    
    def __init__(self, redis_url: str):
        self.redis = Redis(redis_url)
        self.pending = {}
    
    async def enqueue_and_wait(
        self, 
        request_id: str, 
        backend: str, 
        request: CompletionRequest,
        timeout: float = 60.0
    ) -> CompletionResponse:
        """Enqueue request and wait for result."""
        
        # Create future for result
        future = asyncio.Future()
        self.pending[request_id] = future
        
        # Add to queue
        await self.redis.lpush(f"queue:{backend}", {
            "request_id": request_id,
            "request": request.dict()
        })
        
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            raise InferenceTimeout()
        finally:
            del self.pending[request_id]
    
    async def on_result(self, request_id: str, result: dict):
        """Called when inference completes."""
        if request_id in self.pending:
            self.pending[request_id].set_result(result)
```

### Continuous Batching with vLLM

```python
from vllm import LLM, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
import asyncio

class InferenceWorker:
    """Worker that runs model inference with continuous batching."""
    
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.9):
        self.engine = AsyncLLMEngine.from_engine_args(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=8192,
            max_num_seqs=256,  # Max concurrent sequences
        )
    
    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str
    ) -> str:
        """Generate completion with continuous batching."""
        
        results_generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id
        )
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        return final_output.outputs[0].text
    
    async def stream_generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str
    ):
        """Stream tokens as they're generated."""
        
        results_generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id
        )
        
        previous_text = ""
        async for request_output in results_generator:
            current_text = request_output.outputs[0].text
            new_text = current_text[len(previous_text):]
            previous_text = current_text
            
            if new_text:
                yield new_text


class BatchProcessor:
    """Processes requests in optimized batches."""
    
    def __init__(self, worker: InferenceWorker, batch_size: int = 32):
        self.worker = worker
        self.batch_size = batch_size
        self.request_buffer = asyncio.Queue()
    
    async def run(self):
        """Main processing loop."""
        while True:
            batch = await self._collect_batch()
            if batch:
                await self._process_batch(batch)
    
    async def _collect_batch(self, timeout: float = 0.05) -> list:
        """Collect requests into a batch."""
        batch = []
        
        try:
            # Wait for first request
            first = await asyncio.wait_for(
                self.request_buffer.get(),
                timeout=timeout
            )
            batch.append(first)
            
            # Collect more requests without waiting
            while len(batch) < self.batch_size:
                try:
                    req = self.request_buffer.get_nowait()
                    batch.append(req)
                except asyncio.QueueEmpty:
                    break
        except asyncio.TimeoutError:
            pass
        
        return batch
    
    async def _process_batch(self, batch: list):
        """Process a batch of requests concurrently."""
        tasks = []
        for request in batch:
            task = asyncio.create_task(
                self.worker.generate(
                    prompt=request["prompt"],
                    sampling_params=SamplingParams(**request["params"]),
                    request_id=request["id"]
                )
            )
            tasks.append((request, task))
        
        # Wait for all completions
        for request, task in tasks:
            try:
                result = await task
                await self._send_result(request["id"], result)
            except Exception as e:
                await self._send_error(request["id"], str(e))
```

### Inside a Modern Inference Engine

LLM inference has two phases with opposite hardware profiles, and nearly every serving optimization exploits that asymmetry:

```mermaid
graph LR
    subgraph PREFILL["PREFILL (compute-bound)"]
        P["Process entire prompt<br/>in parallel<br/>→ determines TTFT"]
    end
    subgraph DECODE["DECODE (memory-bandwidth-bound)"]
        D["One token per step,<br/>re-reads weights + KV cache<br/>→ determines inter-token latency"]
    end
    P -->|"KV cache<br/>(the bottleneck resource)"| D
```

**Continuous batching.** Requests join and leave the batch at *token* granularity instead of waiting for the slowest request in a static batch. This alone is why vLLM/TGI-class engines deliver an order of magnitude more throughput than naive serving — a finished sequence's slot is reused on the very next step.

**PagedAttention.** The KV cache is allocated in fixed-size blocks with an indirection table, like virtual memory pages, instead of one contiguous buffer per request. This eliminates the memory fragmentation that previously capped batch sizes, and enables KV sharing between sequences (e.g., N samples from one prompt share the prompt's blocks).

**Prefix caching (RadixAttention).** Requests that share a prompt prefix — same system prompt, same few-shot block, the entire history of an agent conversation — reuse the prefix's KV blocks instead of recomputing prefill. SGLang organizes the cache as a radix tree to maximize cross-request sharing. For agent and chat workloads, where each turn resends the whole transcript, prefix caching routinely cuts prefill work by 80–95%; it is the self-hosted counterpart of provider-side prompt caching.

**Chunked prefill.** A long prompt's prefill is split into chunks and interleaved with ongoing decode steps, so one user's 100K-token prompt doesn't stall every other user's token stream. This is the standard fix for the prefill-vs-decode interference problem within a single replica.

**Speculative decoding.** A cheap drafter (small model, extra decoding heads as in Medusa/EAGLE, or n-gram lookup) proposes k tokens; the target model verifies them in a single forward pass and accepts the longest correct run. Output is provably identical to normal decoding, but you get 2–3× lower inter-token latency when acceptance rates are high (code and structured text accept well; high-entropy creative text doesn't).

**Disaggregated prefill/decode.** At datacenter scale, prefill and decode run on *separate GPU pools* sized independently, with KV caches streamed between them over NVLink/RDMA (DistServe, Mooncake — the architecture behind Kimi's serving stack). This stops the two phases from fighting over the same SLO: you scale prefill capacity for TTFT and decode capacity for tokens/sec, and a tiered KV store (HBM → DRAM → SSD) serves as a cluster-wide prefix cache.

**Quantization.** FP8 weights+activations are near-lossless on current hardware and roughly double throughput versus BF16; INT4 weight-only (AWQ/GPTQ) halves memory again for latency-tolerant or memory-constrained deployments; KV-cache quantization (FP8) directly raises the achievable batch size, which is usually the binding constraint. Validate on your own evals — quantization loss concentrates in long-tail reasoning, not average-case perplexity.

```python
# What the knobs look like in practice (vLLM)
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,          # shard weights across 4 GPUs
    gpu_memory_utilization=0.92,     # rest is headroom for activations
    enable_prefix_caching=True,      # radix-style KV reuse across requests
    enable_chunked_prefill=True,     # protect inter-token latency
    quantization="fp8",              # weights + activations
    kv_cache_dtype="fp8",            # bigger effective batch
    speculative_config={             # draft-model speculation
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "num_speculative_tokens": 5,
    },
)
```

**The metrics that matter.** Throughput alone is meaningless for interactive serving. Track **TTFT** (time to first token — prefill + queueing), **TPOT/ITL** (time per output token), and **goodput**: requests per second that *meet their latency SLO*. A configuration that raises raw throughput 30% while pushing p95 TTFT past your SLO has negative value.

### Structured Outputs and Constrained Decoding

Production systems need valid JSON, not "mostly JSON." Modern engines enforce output structure *during decoding*: a grammar compiler (xgrammar, llguidance, Outlines) turns a JSON Schema into a token-level mask, and invalid tokens are simply never sampled. Guaranteed-valid output, near-zero overhead, no retry loops.

```python
from pydantic import BaseModel

class Triage(BaseModel):
    severity: str          # "P0" | "P1" | "P2"
    component: str
    needs_human: bool

# vLLM / SGLang / OpenAI-compatible servers accept the schema directly
response = client.chat.completions.create(
    model="local-llama",
    messages=[{"role": "user", "content": f"Triage this incident:\n{report}"}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "triage", "schema": Triage.model_json_schema(), "strict": True},
    },
)
incident = Triage.model_validate_json(response.choices[0].message.content)
```

Use schema-constrained outputs for anything a program consumes; keep free text for humans. One caution: a constrained model *will* produce schema-valid garbage rather than express uncertainty — include an explicit `"unknown"`/`needs_human` escape hatch in the schema.

### Model Routing

```python
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    endpoint: str
    max_tokens: int
    cost_per_1k_tokens: float
    latency_p50_ms: int
    capabilities: List[str]

class ModelRouter:
    """Routes requests to appropriate models."""
    
    def __init__(self, models: Dict[str, ModelConfig]):
        self.models = models
        self.health_checker = ModelHealthChecker()
    
    async def route(self, request: CompletionRequest) -> str:
        """Select best model for request."""
        
        # Filter by capabilities
        capable_models = [
            name for name, config in self.models.items()
            if self._can_handle(config, request)
        ]
        
        # Filter by health
        healthy_models = [
            name for name in capable_models
            if await self.health_checker.is_healthy(name)
        ]
        
        if not healthy_models:
            raise NoHealthyModelsAvailable()
        
        # Select based on strategy
        return await self._select_model(healthy_models, request)
    
    def _can_handle(self, config: ModelConfig, request: CompletionRequest) -> bool:
        """Check if model can handle request."""
        return (
            request.max_tokens <= config.max_tokens and
            request.model in [config.name, "auto"]
        )
    
    async def _select_model(
        self, 
        candidates: List[str], 
        request: CompletionRequest
    ) -> str:
        """Select from candidate models."""
        
        # Cost-optimized selection
        if request.model == "auto":
            return min(
                candidates,
                key=lambda m: self.models[m].cost_per_1k_tokens
            )
        
        # Specific model requested
        if request.model in candidates:
            return request.model
        
        # Fallback
        return candidates[0]


class ModelCascade:
    """Route to cheaper models first, escalate if needed."""
    
    def __init__(self, models: List[ModelConfig], quality_threshold: float = 0.7):
        # Sort by cost (cheapest first)
        self.models = sorted(models, key=lambda m: m.cost_per_1k_tokens)
        self.quality_threshold = quality_threshold
        self.quality_estimator = QualityEstimator()
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Try models in order of cost until quality threshold met."""
        
        for model in self.models:
            response = await self._call_model(model, request)
            
            # Estimate quality
            quality = await self.quality_estimator.estimate(
                request.prompt,
                response.choices[0]["text"]
            )
            
            if quality >= self.quality_threshold:
                return response
            
            # Log for analysis
            logger.info(f"Model {model.name} quality {quality} below threshold")
        
        # Return last (most capable) model's response
        return response
```

---

## Caching Strategies

### Prompt Caching (Provider-Side) — Use This First

Every major API provider caches the KV state of repeated prompt *prefixes* and bills cached tokens at roughly 10% of the fresh-input price (with cache writes at a small premium on some providers). Unlike semantic caching, this is exact, lossless, and provider-managed — there is no correctness risk, only an engineering requirement: **keep your prefixes byte-stable**.

Rules that make prompt caching work:

- Order context stable→volatile: system prompt and tool schemas first, session data next, the running conversation last.
- Append-only message history. Editing, reordering, or re-rendering earlier turns invalidates the cache from that point on.
- No timestamps, UUIDs, or per-request noise in the prefix. Inject volatile data at the end or via tool results.
- Monitor `cached_tokens` from API usage fields as a first-class metric. For agentic workloads — which resend the full transcript every turn — cache hit rate is typically the largest cost lever in the entire system, far ahead of model choice.

Most providers also offer a **batch tier** (50% discount for asynchronous, hours-latency processing) — route evals, backfills, and non-interactive pipelines there by default.

### Semantic Cache

```mermaid
graph LR
    Q["Query:<br/>What's the capital<br/>of France?"] --> EMB["Embed<br/>Query"]
    EMB --> VS["Vector<br/>Search"]
    VS --> SIM["Similar query found:<br/>France's capital?<br/>similarity: 0.95"]
    SIM --> RET["Return Cached Response:<br/>The capital of France<br/>is Paris."]
```

Benefits: handles paraphrased queries, reduces API costs significantly, sub-millisecond response for cache hits.

Caveats — semantic caching trades correctness for cost, so scope it deliberately: similarity ≥ 0.95 does **not** guarantee the same correct answer ("capital of France" vs "capital city of metropolitan France" is fine; "2023 revenue" vs "2024 revenue" is not). Partition the cache by user/tenant and model+parameter hash, keep TTLs short for anything time-sensitive, and restrict it to stateless, high-repetition query traffic — never cache across personalized context, and never use it for agent turns, where conversation state makes every request unique (that's what prompt caching is for).

```python
import hashlib
from typing import Optional
import numpy as np

class SemanticCache:
    """Cache LLM responses with semantic similarity matching."""
    
    def __init__(
        self,
        embedding_model,
        vector_store,
        kv_store,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600
    ):
        self.embedder = embedding_model
        self.vector_store = vector_store
        self.kv_store = kv_store  # Redis/Memcached
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
    
    async def get(
        self, 
        prompt: str, 
        model: str,
        params_hash: str = None
    ) -> Optional[dict]:
        """Try to get cached response."""
        
        # Try exact match first (faster)
        exact_key = self._exact_key(prompt, model, params_hash)
        exact_result = await self.kv_store.get(exact_key)
        if exact_result:
            return exact_result
        
        # Try semantic match
        prompt_embedding = await self.embedder.embed(prompt)
        
        results = await self.vector_store.search(
            embedding=prompt_embedding,
            top_k=1,
            filter={"model": model}
        )
        
        if results and results[0].score >= self.threshold:
            cache_key = results[0].metadata["cache_key"]
            return await self.kv_store.get(cache_key)
        
        return None
    
    async def set(
        self, 
        prompt: str, 
        model: str, 
        response: dict,
        params_hash: str = None
    ):
        """Cache a response."""
        
        cache_key = self._exact_key(prompt, model, params_hash)
        
        # Store response
        await self.kv_store.set(cache_key, response, ex=self.ttl)
        
        # Store embedding for semantic search
        prompt_embedding = await self.embedder.embed(prompt)
        await self.vector_store.upsert([{
            "id": cache_key,
            "embedding": prompt_embedding,
            "metadata": {
                "model": model,
                "cache_key": cache_key,
                "prompt_preview": prompt[:100]
            }
        }])
    
    def _exact_key(self, prompt: str, model: str, params_hash: str = None) -> str:
        """Generate exact match cache key."""
        content = f"{model}:{prompt}:{params_hash or ''}"
        return f"llm_cache:{hashlib.sha256(content.encode()).hexdigest()}"


class TieredCache:
    """Multi-tier caching strategy."""
    
    def __init__(self):
        self.l1_cache = InMemoryCache(max_size=1000)  # Hot cache
        self.l2_cache = RedisCache()  # Distributed cache
        self.l3_cache = SemanticCache()  # Semantic matching
    
    async def get(self, prompt: str, model: str) -> Optional[dict]:
        """Try caches in order."""
        
        # L1: In-memory (fastest)
        key = self._key(prompt, model)
        result = self.l1_cache.get(key)
        if result:
            return result
        
        # L2: Redis (fast)
        result = await self.l2_cache.get(key)
        if result:
            self.l1_cache.set(key, result)  # Populate L1
            return result
        
        # L3: Semantic (slower but handles variations)
        result = await self.l3_cache.get(prompt, model)
        if result:
            # Populate upper tiers
            self.l1_cache.set(key, result)
            await self.l2_cache.set(key, result)
            return result
        
        return None
```

### KV Cache Management

```python
class KVCacheManager:
    """Manage KV cache for efficient inference."""
    
    def __init__(self, max_cache_tokens: int = 100000):
        self.max_tokens = max_cache_tokens
        self.cache = {}  # session_id -> KVCache
        self.usage = {}  # session_id -> last_used
    
    def get_or_create(self, session_id: str, prefix_tokens: list) -> "KVCache":
        """Get existing cache or create new one."""
        
        if session_id in self.cache:
            return self.cache[session_id]
        
        # Evict if needed
        self._evict_if_needed()
        
        # Create new cache
        cache = KVCache(prefix_tokens)
        self.cache[session_id] = cache
        self.usage[session_id] = time.time()
        
        return cache
    
    def _evict_if_needed(self):
        """Evict least recently used caches."""
        total_tokens = sum(c.token_count for c in self.cache.values())
        
        while total_tokens > self.max_tokens and self.cache:
            # Find LRU session
            lru_session = min(self.usage, key=self.usage.get)
            
            total_tokens -= self.cache[lru_session].token_count
            del self.cache[lru_session]
            del self.usage[lru_session]
    
    def extend(self, session_id: str, new_tokens: list):
        """Extend existing cache with new tokens."""
        if session_id in self.cache:
            self.cache[session_id].extend(new_tokens)
            self.usage[session_id] = time.time()


class PrefixCache:
    """Cache common prompt prefixes."""
    
    def __init__(self, model_engine):
        self.engine = model_engine
        self.prefix_cache = {}  # prefix_hash -> computed_kv_cache
    
    def compute_prefix(self, prefix: str) -> str:
        """Compute and cache KV state for prefix."""
        prefix_hash = hashlib.sha256(prefix.encode()).hexdigest()[:16]
        
        if prefix_hash not in self.prefix_cache:
            # Compute KV cache for prefix
            kv_cache = self.engine.compute_kv_cache(prefix)
            self.prefix_cache[prefix_hash] = kv_cache
        
        return prefix_hash
    
    def generate_with_prefix(
        self, 
        prefix_hash: str, 
        continuation: str,
        params: dict
    ) -> str:
        """Generate using cached prefix."""
        
        if prefix_hash not in self.prefix_cache:
            raise ValueError("Prefix not found in cache")
        
        kv_cache = self.prefix_cache[prefix_hash]
        
        return self.engine.generate(
            prompt=continuation,
            kv_cache=kv_cache,
            **params
        )
```

---

## Evaluation & Testing

### Automated Evaluation Pipeline

```mermaid
graph LR
    TC["Test Cases"] --> RM["Run Model"]
    RM --> SO["Score Outputs"]
    TC --> CB["Compare to<br/>Baseline"]
    SO --> CB
    CB --> RR["Report Results<br/>Accuracy, Regressions,<br/>Latency, Cost"]
```

```python
from dataclasses import dataclass
from typing import List, Callable
import json

@dataclass
class TestCase:
    id: str
    prompt: str
    expected: str = None  # For exact match
    criteria: List[str] = None  # For LLM-as-judge
    tags: List[str] = None

@dataclass
class EvalResult:
    test_id: str
    passed: bool
    score: float
    latency_ms: float
    tokens_used: int
    details: dict

class LLMEvaluator:
    """Evaluate LLM outputs automatically."""
    
    def __init__(self, target_model, judge_model=None):
        self.target = target_model
        self.judge = judge_model or target_model
    
    async def run_eval(
        self, 
        test_cases: List[TestCase],
        eval_functions: List[Callable] = None
    ) -> List[EvalResult]:
        """Run evaluation on test cases."""
        
        results = []
        
        for test in test_cases:
            start = time.time()
            
            # Generate output
            output = await self.target.generate(test.prompt)
            
            latency = (time.time() - start) * 1000
            
            # Evaluate
            scores = {}
            
            # Exact match
            if test.expected:
                scores["exact_match"] = float(output.strip() == test.expected.strip())
            
            # LLM-as-judge
            if test.criteria:
                judge_scores = await self._llm_judge(
                    test.prompt, 
                    output, 
                    test.criteria
                )
                scores.update(judge_scores)
            
            # Custom eval functions
            if eval_functions:
                for func in eval_functions:
                    scores[func.__name__] = func(test.prompt, output)
            
            # Aggregate score
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            
            results.append(EvalResult(
                test_id=test.id,
                passed=avg_score >= 0.7,
                score=avg_score,
                latency_ms=latency,
                tokens_used=output.usage.total_tokens,
                details=scores
            ))
        
        return results
    
    async def _llm_judge(
        self, 
        prompt: str, 
        output: str, 
        criteria: List[str]
    ) -> dict:
        """Use LLM to judge output quality."""
        
        judge_prompt = f"""Evaluate this LLM output against the criteria.

Original prompt: {prompt}

Output to evaluate: {output}

Criteria to check:
{json.dumps(criteria, indent=2)}

For each criterion, score 0-1 and explain.
Return JSON with criterion names as keys."""
        
        response = await self.judge.generate(
            judge_prompt,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response)


class RegressionDetector:
    """Detect quality regressions between model versions."""
    
    def __init__(self, baseline_results: List[EvalResult]):
        self.baseline = {r.test_id: r for r in baseline_results}
    
    def compare(
        self, 
        new_results: List[EvalResult],
        threshold: float = 0.05
    ) -> dict:
        """Compare new results to baseline."""
        
        regressions = []
        improvements = []
        
        for result in new_results:
            if result.test_id not in self.baseline:
                continue
            
            baseline = self.baseline[result.test_id]
            diff = result.score - baseline.score
            
            if diff < -threshold:
                regressions.append({
                    "test_id": result.test_id,
                    "baseline_score": baseline.score,
                    "new_score": result.score,
                    "diff": diff
                })
            elif diff > threshold:
                improvements.append({
                    "test_id": result.test_id,
                    "baseline_score": baseline.score,
                    "new_score": result.score,
                    "diff": diff
                })
        
        return {
            "regressions": regressions,
            "improvements": improvements,
            "regression_rate": len(regressions) / len(new_results),
            "passed": len(regressions) == 0
        }


class ContinuousEval:
    """Run evaluations continuously in production."""
    
    def __init__(self, evaluator: LLMEvaluator, sample_rate: float = 0.01):
        self.evaluator = evaluator
        self.sample_rate = sample_rate
        self.metrics = PrometheusMetrics()
    
    async def maybe_evaluate(self, request: dict, response: dict):
        """Sample and evaluate production traffic."""
        
        if random.random() > self.sample_rate:
            return
        
        # Create test case from production request
        test = TestCase(
            id=f"prod_{request['id']}",
            prompt=request["prompt"],
            criteria=[
                "Response is helpful and relevant",
                "Response is factually accurate",
                "Response follows safety guidelines"
            ]
        )
        
        # Evaluate
        results = await self.evaluator.run_eval([test])
        result = results[0]
        
        # Record metrics
        self.metrics.record_eval_score(result.score)
        self.metrics.record_latency(result.latency_ms)
        
        # Alert on low scores
        if result.score < 0.5:
            await self._alert_low_quality(request, response, result)
```

---

## Cost Optimization

### Token Management

```python
from tiktoken import encoding_for_model

class TokenManager:
    """Manage token usage and costs.

    Never hardcode prices — they change quarterly. Load a pricing table
    from config, and model all four meters: fresh input, cached input
    (~10% of fresh), output (often 4-5x input — thinking tokens bill as
    output), and batch-tier discounts (~50%).
    """

    def __init__(self, pricing: dict[str, dict[str, float]]):
        self.pricing = pricing  # per 1M tokens, from config/pricing.yaml
        self.encoders = {}

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text."""
        if model not in self.encoders:
            self.encoders[model] = encoding_for_model(model)
        return len(self.encoders[model].encode(text))

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        batch: bool = False,
    ) -> float:
        """Estimate cost for a request."""
        p = self.pricing[model]
        fresh = input_tokens - cached_tokens

        cost = (
            fresh * p["input"]
            + cached_tokens * p.get("cached_input", p["input"] * 0.1)
            + output_tokens * p["output"]
        ) / 1_000_000

        return cost * 0.5 if batch else cost
    
    def optimize_prompt(self, prompt: str, max_tokens: int) -> str:
        """Trim prompt to fit token budget."""
        # Implementation depends on use case
        # Could use summarization, truncation, etc.
        pass


class BudgetManager:
    """Manage spending budgets."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def check_budget(
        self, 
        user_id: str, 
        estimated_cost: float
    ) -> bool:
        """Check if user has budget for request."""
        
        key = f"budget:{user_id}"
        current_spend = float(await self.redis.get(key) or 0)
        limit = await self._get_limit(user_id)
        
        return current_spend + estimated_cost <= limit
    
    async def record_spend(self, user_id: str, cost: float):
        """Record spending."""
        key = f"budget:{user_id}"
        await self.redis.incrbyfloat(key, cost)
    
    async def _get_limit(self, user_id: str) -> float:
        """Get user's spending limit."""
        # Could be from database, config, etc.
        return 100.0  # Default $100/month


class CostOptimizedPipeline:
    """Pipeline that optimizes for cost."""
    
    def __init__(self, models: List[dict]):
        # Sort models by cost (cheapest first)
        self.models = sorted(models, key=lambda m: m["cost_per_1k"])
        self.token_manager = TokenManager()
    
    async def complete(
        self, 
        prompt: str,
        quality_threshold: float = 0.7,
        max_cost: float = None
    ) -> dict:
        """Complete with cost optimization."""
        
        input_tokens = self.token_manager.count_tokens(prompt, "gpt-4")
        
        for model in self.models:
            # Check cost constraint
            estimated_cost = self.token_manager.estimate_cost(
                model["name"],
                input_tokens,
                model.get("avg_output_tokens", 500)
            )
            
            if max_cost and estimated_cost > max_cost:
                continue
            
            # Try model
            response = await self._call_model(model, prompt)
            
            # Check quality
            quality = await self._estimate_quality(prompt, response)
            
            if quality >= quality_threshold:
                return {
                    "response": response,
                    "model": model["name"],
                    "cost": estimated_cost,
                    "quality": quality
                }
        
        # Fallback to best model
        return await self._call_model(self.models[-1], prompt)
```

---

## Guardrails & Safety

### Input/Output Filtering

```mermaid
graph LR
    IN["Input"] --> IG["INPUT GUARDS<br/>Prompt injection<br/>Jailbreak detection<br/>Topic blocking<br/>Rate limiting<br/>PII redaction"]
    IG --> LLM["LLM<br/>PROCESS"]
    LLM --> OG["OUTPUT GUARDS<br/>PII detection<br/>Harmful content filter<br/>Factuality check<br/>Format validation<br/>Confidence threshold"]
    OG --> OUT["Output"]
```

```python
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import re

class Guard(ABC):
    """Base class for guardrails."""
    
    @abstractmethod
    async def check(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check text against guard.
        Returns (passed, reason if failed)
        """
        pass

class PromptInjectionGuard(Guard):
    """Detect prompt injection attempts."""
    
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"disregard\s+(previous|above|all)",
        r"you\s+are\s+now\s+(?:a|an)\s+\w+",
        r"new\s+instructions:",
        r"system\s*:\s*",
        r"<\|.*?\|>",  # Special tokens
    ]
    
    def __init__(self, llm_detector=None):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self.llm_detector = llm_detector
    
    async def check(self, text: str) -> Tuple[bool, Optional[str]]:
        # Pattern matching
        for pattern in self.patterns:
            if pattern.search(text):
                return False, "Potential prompt injection detected"
        
        # LLM-based detection for sophisticated attacks
        if self.llm_detector:
            is_injection = await self.llm_detector.detect(text)
            if is_injection:
                return False, "LLM-detected prompt injection"
        
        return True, None


class PIIGuard(Guard):
    """Detect and redact PII."""
    
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    }
    
    def __init__(self, mode: str = "block"):  # block, redact, or warn
        self.mode = mode
        self.patterns = {k: re.compile(v) for k, v in self.PII_PATTERNS.items()}
    
    async def check(self, text: str) -> Tuple[bool, Optional[str]]:
        found_pii = []
        
        for pii_type, pattern in self.patterns.items():
            if pattern.search(text):
                found_pii.append(pii_type)
        
        if found_pii:
            if self.mode == "block":
                return False, f"PII detected: {', '.join(found_pii)}"
            elif self.mode == "warn":
                return True, f"Warning: PII detected: {', '.join(found_pii)}"
        
        return True, None
    
    def redact(self, text: str) -> str:
        """Redact PII from text."""
        for pii_type, pattern in self.patterns.items():
            text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
        return text


class ContentModerationGuard(Guard):
    """Filter harmful content."""
    
    def __init__(self, moderation_api):
        self.api = moderation_api
    
    async def check(self, text: str) -> Tuple[bool, Optional[str]]:
        result = await self.api.moderate(text)
        
        if result.flagged:
            categories = [c for c, v in result.categories.items() if v]
            return False, f"Content flagged: {', '.join(categories)}"
        
        return True, None


class GuardrailsPipeline:
    """Complete guardrails pipeline."""
    
    def __init__(self):
        self.input_guards: List[Guard] = []
        self.output_guards: List[Guard] = []
    
    def add_input_guard(self, guard: Guard):
        self.input_guards.append(guard)
    
    def add_output_guard(self, guard: Guard):
        self.output_guards.append(guard)
    
    async def process(
        self, 
        input_text: str,
        llm_func: Callable
    ) -> dict:
        """Process request through guardrails."""
        
        # Input guards
        for guard in self.input_guards:
            passed, reason = await guard.check(input_text)
            if not passed:
                return {
                    "blocked": True,
                    "stage": "input",
                    "guard": guard.__class__.__name__,
                    "reason": reason
                }
        
        # LLM call
        output = await llm_func(input_text)
        
        # Output guards
        for guard in self.output_guards:
            passed, reason = await guard.check(output)
            if not passed:
                return {
                    "blocked": True,
                    "stage": "output",
                    "guard": guard.__class__.__name__,
                    "reason": reason,
                    "original_output": output  # For debugging
                }
        
        return {
            "blocked": False,
            "output": output
        }


# Usage
pipeline = GuardrailsPipeline()
pipeline.add_input_guard(PromptInjectionGuard())
pipeline.add_input_guard(PIIGuard(mode="redact"))
pipeline.add_output_guard(ContentModerationGuard(openai_moderation))
pipeline.add_output_guard(PIIGuard(mode="block"))
```

### Rate Limiting

```python
from dataclasses import dataclass
import time

@dataclass
class RateLimitConfig:
    requests_per_minute: int
    tokens_per_minute: int
    requests_per_day: int
    tokens_per_day: int

class RateLimiter:
    """Token and request rate limiting."""
    
    def __init__(self, redis_client, default_config: RateLimitConfig):
        self.redis = redis_client
        self.default_config = default_config
    
    async def check_and_consume(
        self, 
        user_id: str, 
        tokens: int,
        config: RateLimitConfig = None
    ) -> Tuple[bool, dict]:
        """Check rate limits and consume quota if allowed."""
        
        config = config or self.default_config
        now = time.time()
        minute_key = f"rl:{user_id}:minute:{int(now / 60)}"
        day_key = f"rl:{user_id}:day:{int(now / 86400)}"
        
        # Get current usage
        minute_requests = int(await self.redis.hget(minute_key, "requests") or 0)
        minute_tokens = int(await self.redis.hget(minute_key, "tokens") or 0)
        day_requests = int(await self.redis.hget(day_key, "requests") or 0)
        day_tokens = int(await self.redis.hget(day_key, "tokens") or 0)
        
        # Check limits
        if minute_requests >= config.requests_per_minute:
            return False, {"reason": "requests_per_minute", "retry_after": 60}
        
        if minute_tokens + tokens > config.tokens_per_minute:
            return False, {"reason": "tokens_per_minute", "retry_after": 60}
        
        if day_requests >= config.requests_per_day:
            return False, {"reason": "requests_per_day", "retry_after": 86400}
        
        if day_tokens + tokens > config.tokens_per_day:
            return False, {"reason": "tokens_per_day", "retry_after": 86400}
        
        # Consume quota
        pipe = self.redis.pipeline()
        pipe.hincrby(minute_key, "requests", 1)
        pipe.hincrby(minute_key, "tokens", tokens)
        pipe.expire(minute_key, 120)
        pipe.hincrby(day_key, "requests", 1)
        pipe.hincrby(day_key, "tokens", tokens)
        pipe.expire(day_key, 172800)
        await pipe.execute()
        
        return True, {
            "remaining": {
                "requests_minute": config.requests_per_minute - minute_requests - 1,
                "tokens_minute": config.tokens_per_minute - minute_tokens - tokens,
            }
        }
```

---

## Monitoring & Observability

```python
from prometheus_client import Counter, Histogram, Gauge

class LLMMetrics:
    """Prometheus metrics for LLM infrastructure."""
    
    def __init__(self):
        # Request metrics
        self.requests_total = Counter(
            "llm_requests_total",
            "Total LLM requests",
            ["model", "status"]
        )
        
        self.request_latency = Histogram(
            "llm_request_latency_seconds",
            "Request latency",
            ["model"],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
        )
        
        # Token metrics
        self.tokens_processed = Counter(
            "llm_tokens_total",
            "Total tokens processed",
            ["model", "direction"]  # input/output
        )
        
        # Cost metrics
        self.cost_total = Counter(
            "llm_cost_dollars",
            "Total cost in dollars",
            ["model"]
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            "llm_cache_hits_total",
            "Cache hits",
            ["cache_level"]
        )
        
        self.cache_misses = Counter(
            "llm_cache_misses_total",
            "Cache misses"
        )
        
        # Quality metrics
        self.eval_scores = Histogram(
            "llm_eval_score",
            "Evaluation scores",
            ["model", "eval_type"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Safety metrics
        self.guardrail_blocks = Counter(
            "llm_guardrail_blocks_total",
            "Requests blocked by guardrails",
            ["guard_type", "stage"]
        )
    
    def record_request(
        self, 
        model: str, 
        status: str, 
        latency: float,
        input_tokens: int,
        output_tokens: int,
        cost: float
    ):
        self.requests_total.labels(model=model, status=status).inc()
        self.request_latency.labels(model=model).observe(latency)
        self.tokens_processed.labels(model=model, direction="input").inc(input_tokens)
        self.tokens_processed.labels(model=model, direction="output").inc(output_tokens)
        self.cost_total.labels(model=model).inc(cost)
```

---

## Trade-offs

| Decision | Trade-off |
|----------|-----------|
| **Self-hosted vs API** | Control & cost vs complexity |
| **Cache aggressiveness** | Speed & cost vs freshness |
| **Guard strictness** | Safety vs utility |
| **Model cascading** | Cost vs latency |
| **Batch size** | Throughput vs latency |

---

## References

- [vLLM: High-throughput LLM Serving](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) — the dominant open serving engines
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — the vLLM paper
- [DistServe: Disaggregating Prefill and Decoding](https://arxiv.org/abs/2401.09670) and [Mooncake: KVCache-centric Disaggregated Architecture](https://arxiv.org/abs/2407.00079)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) and [Medusa](https://arxiv.org/abs/2401.10774) — speculative decoding
- [XGrammar: Flexible and Efficient Structured Generation](https://github.com/mlc-ai/xgrammar) / [Outlines](https://github.com/dottxt-ai/outlines) — constrained decoding
- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) / [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — standard trace/metric schema for LLM systems
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) / [Guardrails AI](https://github.com/guardrails-ai/guardrails)
- [LiteLLM](https://github.com/BerriAI/litellm) — multi-provider gateway, routing, budgets
