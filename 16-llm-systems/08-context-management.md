# Context Management Patterns

## TL;DR

Context management is the art of efficiently utilizing the limited context window of LLMs to maximize response quality while controlling costs. Key strategies include understanding token economics (input/output allocation, cost implications), handling long contexts (lost-in-the-middle problem, needle-in-haystack testing), compression techniques (summarization, LLMLingua, selective context), sliding window patterns (rolling summaries, hierarchical context), memory systems (short-term, long-term, episodic, semantic, working memory), and production optimizations (KV caching, prefix caching, context pre-computation). The choice between long context and RAG depends on update frequency, precision requirements, and cost constraints.

---

## Context Window Fundamentals

### Context Window Anatomy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW ANATOMY                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    CONTEXT WINDOW (e.g., 128K tokens)          │ │
│  │                                                                 │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │ SYSTEM PROMPT                                      ~500  │  │ │
│  │  │ "You are a helpful assistant..."                         │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  │                                                                 │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │ RETRIEVED CONTEXT / DOCUMENTS                    ~50,000 │  │ │
│  │  │ [Document chunks, RAG results, knowledge base...]        │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  │                                                                 │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │ CONVERSATION HISTORY                              ~5,000 │  │ │
│  │  │ [Previous messages, tool calls, observations...]         │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  │                                                                 │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │ CURRENT USER MESSAGE                              ~1,000 │  │ │
│  │  │ [User's current query or instruction]                    │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  │                                                                 │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │ RESERVED FOR OUTPUT                               ~4,000 │  │ │
│  │  │ [Space for model's response]                             │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  │                                                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Input Tokens: Billed at input rate                                  │
│  Output Tokens: Billed at output rate (usually 2-3x input)           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Token Counting and Limits

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import tiktoken

@dataclass
class ModelContextConfig:
    """Context configuration for different models."""
    model_name: str
    max_context_tokens: int
    max_output_tokens: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    supports_system_prompt: bool = True
    
# Model configurations (as of 2024)
MODEL_CONFIGS = {
    "gpt-4-turbo": ModelContextConfig(
        model_name="gpt-4-turbo",
        max_context_tokens=128000,
        max_output_tokens=4096,
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.03,
    ),
    "gpt-4o": ModelContextConfig(
        model_name="gpt-4o",
        max_context_tokens=128000,
        max_output_tokens=16384,
        input_cost_per_1k=0.005,
        output_cost_per_1k=0.015,
    ),
    "claude-3-opus": ModelContextConfig(
        model_name="claude-3-opus",
        max_context_tokens=200000,
        max_output_tokens=4096,
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
    ),
    "claude-3.5-sonnet": ModelContextConfig(
        model_name="claude-3.5-sonnet",
        max_context_tokens=200000,
        max_output_tokens=8192,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
    ),
    "gemini-1.5-pro": ModelContextConfig(
        model_name="gemini-1.5-pro",
        max_context_tokens=1000000,  # 1M tokens
        max_output_tokens=8192,
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.005,
    ),
    "llama-3-70b": ModelContextConfig(
        model_name="llama-3-70b",
        max_context_tokens=8192,
        max_output_tokens=4096,
        input_cost_per_1k=0.0008,
        output_cost_per_1k=0.0008,
    ),
}


class TokenCounter:
    """Count tokens for different models."""
    
    def __init__(self):
        self._encoders: Dict[str, tiktoken.Encoding] = {}
    
    def get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get or create encoder for model."""
        if model not in self._encoders:
            try:
                self._encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                self._encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self._encoders[model]
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text."""
        encoder = self.get_encoder(model)
        return len(encoder.encode(text))
    
    def count_messages_tokens(
        self, 
        messages: List[Dict], 
        model: str = "gpt-4"
    ) -> int:
        """Count tokens in chat messages with overhead."""
        encoder = self.get_encoder(model)
        
        # Token overhead per message (varies by model)
        tokens_per_message = 4  # <im_start>, role, \n, <im_end>
        tokens_per_name = -1  # If name is present
        
        total = 0
        for message in messages:
            total += tokens_per_message
            for key, value in message.items():
                total += len(encoder.encode(str(value)))
                if key == "name":
                    total += tokens_per_name
        
        total += 3  # Every reply is primed with <im_start>assistant<im_sep>
        return total
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> float:
        """Estimate cost for a request."""
        config = MODEL_CONFIGS.get(model)
        if not config:
            raise ValueError(f"Unknown model: {model}")
        
        input_cost = (input_tokens / 1000) * config.input_cost_per_1k
        output_cost = (output_tokens / 1000) * config.output_cost_per_1k
        return input_cost + output_cost


class ContextBudgetManager:
    """Manage token budget allocation."""
    
    def __init__(self, model: str, token_counter: TokenCounter):
        self.config = MODEL_CONFIGS.get(model)
        if not self.config:
            raise ValueError(f"Unknown model: {model}")
        self.counter = token_counter
        self.model = model
    
    def allocate_budget(
        self,
        system_prompt: str,
        conversation_history: List[Dict],
        reserved_output: int = None,
        min_context_space: int = 1000
    ) -> Dict[str, int]:
        """Calculate token budget allocation."""
        
        reserved_output = reserved_output or self.config.max_output_tokens
        
        # Count fixed allocations
        system_tokens = self.counter.count_tokens(system_prompt, self.model)
        history_tokens = self.counter.count_messages_tokens(
            conversation_history, self.model
        )
        
        # Calculate available space
        total_available = self.config.max_context_tokens
        used = system_tokens + history_tokens + reserved_output
        remaining = total_available - used
        
        return {
            "total_context": total_available,
            "system_prompt": system_tokens,
            "conversation_history": history_tokens,
            "reserved_output": reserved_output,
            "available_for_context": max(0, remaining),
            "utilization_percent": (used / total_available) * 100,
            "can_add_context": remaining >= min_context_space
        }
    
    def fit_context(
        self,
        documents: List[str],
        available_tokens: int,
        strategy: str = "truncate"  # truncate, prioritize, summarize
    ) -> List[str]:
        """Fit documents into available token budget."""
        
        if strategy == "truncate":
            return self._truncate_to_fit(documents, available_tokens)
        elif strategy == "prioritize":
            return self._prioritize_to_fit(documents, available_tokens)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _truncate_to_fit(
        self, 
        documents: List[str], 
        available_tokens: int
    ) -> List[str]:
        """Include documents until budget exhausted."""
        result = []
        used_tokens = 0
        
        for doc in documents:
            doc_tokens = self.counter.count_tokens(doc, self.model)
            if used_tokens + doc_tokens <= available_tokens:
                result.append(doc)
                used_tokens += doc_tokens
            else:
                # Try to include partial document
                remaining = available_tokens - used_tokens
                if remaining > 100:  # Worth including partial
                    truncated = self._truncate_text(doc, remaining)
                    result.append(truncated)
                break
        
        return result
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token budget."""
        encoder = self.counter.get_encoder(self.model)
        tokens = encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return encoder.decode(truncated_tokens) + "..."
    
    def _prioritize_to_fit(
        self,
        documents: List[str],
        available_tokens: int
    ) -> List[str]:
        """Prioritize shorter documents to maximize coverage."""
        # Sort by token count (ascending)
        doc_tokens = [
            (doc, self.counter.count_tokens(doc, self.model))
            for doc in documents
        ]
        doc_tokens.sort(key=lambda x: x[1])
        
        result = []
        used_tokens = 0
        
        for doc, tokens in doc_tokens:
            if used_tokens + tokens <= available_tokens:
                result.append(doc)
                used_tokens += tokens
        
        return result
```

---

## Long Context Handling

### The Lost in the Middle Problem

```
┌─────────────────────────────────────────────────────────────────────┐
│                  LOST IN THE MIDDLE PHENOMENON                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Attention Distribution Across Context Window:                       │
│                                                                      │
│  High │ ████                                              ████       │
│       │ ████                                              ████       │
│       │ ████                                              ████       │
│  Att  │ ████                                              ████       │
│  ent  │ ████    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    ████       │
│  ion  │ ████    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    ████       │
│       │ ████    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    ████       │
│  Low  │ ████    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    ████       │
│       └─────────────────────────────────────────────────────────     │
│         Beginning           MIDDLE              End                  │
│         (High recall)    (Low recall)     (High recall)              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Key Insight: Models attend well to beginning and end of     │    │
│  │ context but struggle with information in the middle.        │    │
│  │                                                              │    │
│  │ Mitigation strategies:                                       │    │
│  │ • Place important information at beginning or end            │    │
│  │ • Use explicit markers and structure                         │    │
│  │ • Repeat critical information                                │    │
│  │ • Use hierarchical organization                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

class ContextPosition(Enum):
    """Position priority for context placement."""
    BEGINNING = "beginning"
    END = "end"
    MIDDLE = "middle"

@dataclass
class RankedDocument:
    """Document with importance ranking."""
    content: str
    importance: float  # 0-1, higher is more important
    source: str
    position_hint: ContextPosition = ContextPosition.MIDDLE

class ContextOrganizer:
    """Organize context to mitigate lost-in-the-middle problem."""
    
    def __init__(self, token_counter: TokenCounter, model: str = "gpt-4"):
        self.counter = token_counter
        self.model = model
    
    def organize_context(
        self,
        documents: List[RankedDocument],
        available_tokens: int,
        strategy: str = "bookend"
    ) -> str:
        """Organize documents to maximize retrieval."""
        
        if strategy == "bookend":
            return self._bookend_strategy(documents, available_tokens)
        elif strategy == "importance_first":
            return self._importance_first(documents, available_tokens)
        elif strategy == "alternating":
            return self._alternating_strategy(documents, available_tokens)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _bookend_strategy(
        self,
        documents: List[RankedDocument],
        available_tokens: int
    ) -> str:
        """Place most important at beginning and end."""
        
        # Sort by importance
        sorted_docs = sorted(documents, key=lambda d: d.importance, reverse=True)
        
        if len(sorted_docs) <= 2:
            return self._join_documents([d.content for d in sorted_docs])
        
        # Split into three groups
        n = len(sorted_docs)
        beginning = sorted_docs[:n//3]
        middle = sorted_docs[n//3:2*n//3]
        end = sorted_docs[2*n//3:]
        
        # Reorganize: high importance at edges
        organized = beginning + list(reversed(middle)) + end
        
        return self._fit_and_join([d.content for d in organized], available_tokens)
    
    def _importance_first(
        self,
        documents: List[RankedDocument],
        available_tokens: int
    ) -> str:
        """Simply order by importance."""
        sorted_docs = sorted(documents, key=lambda d: d.importance, reverse=True)
        return self._fit_and_join([d.content for d in sorted_docs], available_tokens)
    
    def _alternating_strategy(
        self,
        documents: List[RankedDocument],
        available_tokens: int
    ) -> str:
        """Alternate high and low importance to distribute attention."""
        sorted_docs = sorted(documents, key=lambda d: d.importance, reverse=True)
        
        high = sorted_docs[:len(sorted_docs)//2]
        low = sorted_docs[len(sorted_docs)//2:]
        
        alternated = []
        for h, l in zip(high, low):
            alternated.extend([h, l])
        
        # Add remaining if odd number
        if len(high) > len(low):
            alternated.append(high[-1])
        
        return self._fit_and_join([d.content for d in alternated], available_tokens)
    
    def _fit_and_join(self, contents: List[str], available_tokens: int) -> str:
        """Fit contents into token budget and join."""
        result = []
        used_tokens = 0
        
        for content in contents:
            tokens = self.counter.count_tokens(content, self.model)
            if used_tokens + tokens <= available_tokens:
                result.append(content)
                used_tokens += tokens
        
        return self._join_documents(result)
    
    def _join_documents(self, documents: List[str]) -> str:
        """Join documents with clear separators."""
        return "\n\n---\n\n".join(documents)


class NeedleInHaystackTester:
    """Test model's ability to find information in long context."""
    
    def __init__(self, llm_client, token_counter: TokenCounter):
        self.llm = llm_client
        self.counter = token_counter
    
    async def run_test(
        self,
        needle: str,
        haystack_size: int,
        positions: List[float] = None,  # 0.0 to 1.0
        model: str = "gpt-4"
    ) -> dict:
        """Run needle-in-haystack test."""
        
        positions = positions or [0.0, 0.25, 0.5, 0.75, 1.0]
        results = []
        
        # Generate filler text (haystack)
        haystack = self._generate_haystack(haystack_size - len(needle))
        
        for position in positions:
            # Insert needle at position
            context = self._insert_at_position(haystack, needle, position)
            
            # Test retrieval
            prompt = f"""Context:
{context}

Question: What is the secret code mentioned in the context?
Answer with just the code, nothing else."""
            
            response = await self.llm.generate(prompt, model=model)
            
            # Check if needle was found
            found = self._check_retrieval(response, needle)
            
            results.append({
                "position": position,
                "position_name": self._position_name(position),
                "found": found,
                "response": response[:100]
            })
        
        return {
            "model": model,
            "haystack_tokens": haystack_size,
            "results": results,
            "accuracy": sum(r["found"] for r in results) / len(results)
        }
    
    def _generate_haystack(self, target_tokens: int) -> str:
        """Generate filler text."""
        # Use repetitive but coherent text
        filler = """The city of Arcadia was known for its beautiful gardens and 
        ancient libraries. Scholars from across the land would travel there to 
        study the manuscripts preserved in the Great Archive. The weather was 
        typically mild, with gentle breezes carrying the scent of jasmine through 
        the cobblestone streets. """
        
        result = ""
        while self.counter.count_tokens(result, "gpt-4") < target_tokens:
            result += filler
        
        return result
    
    def _insert_at_position(
        self, 
        haystack: str, 
        needle: str, 
        position: float
    ) -> str:
        """Insert needle at specified position (0-1)."""
        insert_idx = int(len(haystack) * position)
        return haystack[:insert_idx] + f"\n\n{needle}\n\n" + haystack[insert_idx:]
    
    def _position_name(self, position: float) -> str:
        """Convert position to human-readable name."""
        if position <= 0.1:
            return "beginning"
        elif position <= 0.4:
            return "early_middle"
        elif position <= 0.6:
            return "middle"
        elif position <= 0.9:
            return "late_middle"
        else:
            return "end"
    
    def _check_retrieval(self, response: str, needle: str) -> bool:
        """Check if needle information was retrieved."""
        # Extract the key information from needle
        # This should be customized based on your needle format
        return any(
            keyword.lower() in response.lower() 
            for keyword in needle.split()[:3]
        )
```

### Long Context vs RAG Decision Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│              LONG CONTEXT vs RAG DECISION FRAMEWORK                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Use LONG CONTEXT when:                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ + Data is static or changes infrequently                    │    │
│  │ + You need cross-document reasoning                         │    │
│  │ + Simplicity is more important than cost                    │    │
│  │ + Context size is within model limits                       │    │
│  │ + Low latency is critical (no retrieval step)               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Use RAG when:                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ + Data is large (exceeds context window)                    │    │
│  │ + Data updates frequently                                    │    │
│  │ + You need precise retrieval over vast corpus               │    │
│  │ + Cost optimization is important                            │    │
│  │ + You need source attribution/citations                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Decision Matrix:                                                    │
│                                                                      │
│              │ Small Data   │ Large Data                             │
│  ────────────┼──────────────┼───────────────                         │
│  Static      │ Long Context │ RAG or Hybrid                          │
│  Dynamic     │ RAG          │ RAG                                    │
│              │              │                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
@dataclass
class DataCharacteristics:
    """Characteristics of the data to query."""
    total_tokens: int
    update_frequency_hours: float
    requires_cross_doc_reasoning: bool
    precision_requirement: float  # 0-1
    latency_budget_ms: int

class ContextStrategySelector:
    """Select optimal context strategy."""
    
    def __init__(self, model_configs: Dict[str, ModelContextConfig]):
        self.configs = model_configs
    
    def select_strategy(
        self,
        data: DataCharacteristics,
        available_models: List[str],
        cost_sensitivity: float = 0.5  # 0-1, higher = more cost sensitive
    ) -> dict:
        """Select optimal context strategy."""
        
        recommendations = []
        
        for model in available_models:
            config = self.configs.get(model)
            if not config:
                continue
            
            # Check if data fits in context
            fits_in_context = data.total_tokens < config.max_context_tokens * 0.8
            
            # Calculate scores
            if fits_in_context:
                # Long context option
                long_context_score = self._score_long_context(
                    data, config, cost_sensitivity
                )
                recommendations.append({
                    "strategy": "long_context",
                    "model": model,
                    "score": long_context_score,
                    "estimated_cost_per_query": self._estimate_long_context_cost(
                        data, config
                    )
                })
            
            # RAG option (always available)
            rag_score = self._score_rag(data, config, cost_sensitivity)
            recommendations.append({
                "strategy": "rag",
                "model": model,
                "score": rag_score,
                "estimated_cost_per_query": self._estimate_rag_cost(data, config)
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "recommended": recommendations[0] if recommendations else None,
            "alternatives": recommendations[1:3],
            "analysis": self._generate_analysis(data, recommendations)
        }
    
    def _score_long_context(
        self,
        data: DataCharacteristics,
        config: ModelContextConfig,
        cost_sensitivity: float
    ) -> float:
        """Score long context approach."""
        score = 0.5
        
        # Bonus for static data
        if data.update_frequency_hours > 24:
            score += 0.2
        
        # Bonus for cross-doc reasoning
        if data.requires_cross_doc_reasoning:
            score += 0.15
        
        # Penalty for cost if sensitive
        utilization = data.total_tokens / config.max_context_tokens
        cost_penalty = utilization * cost_sensitivity * 0.3
        score -= cost_penalty
        
        return max(0, min(1, score))
    
    def _score_rag(
        self,
        data: DataCharacteristics,
        config: ModelContextConfig,
        cost_sensitivity: float
    ) -> float:
        """Score RAG approach."""
        score = 0.5
        
        # Bonus for dynamic data
        if data.update_frequency_hours < 24:
            score += 0.2
        
        # Bonus for large data
        if data.total_tokens > config.max_context_tokens:
            score += 0.25
        
        # Bonus for cost sensitivity
        score += cost_sensitivity * 0.15
        
        # Penalty for cross-doc reasoning
        if data.requires_cross_doc_reasoning:
            score -= 0.1
        
        return max(0, min(1, score))
    
    def _estimate_long_context_cost(
        self,
        data: DataCharacteristics,
        config: ModelContextConfig
    ) -> float:
        """Estimate cost per query for long context."""
        input_tokens = data.total_tokens + 500  # system + query overhead
        output_tokens = 500  # estimated response
        
        return (
            (input_tokens / 1000) * config.input_cost_per_1k +
            (output_tokens / 1000) * config.output_cost_per_1k
        )
    
    def _estimate_rag_cost(
        self,
        data: DataCharacteristics,
        config: ModelContextConfig
    ) -> float:
        """Estimate cost per query for RAG."""
        # RAG typically uses ~5000 tokens of context
        input_tokens = 5000 + 500
        output_tokens = 500
        
        return (
            (input_tokens / 1000) * config.input_cost_per_1k +
            (output_tokens / 1000) * config.output_cost_per_1k
        )
    
    def _generate_analysis(
        self,
        data: DataCharacteristics,
        recommendations: List[dict]
    ) -> str:
        """Generate human-readable analysis."""
        if not recommendations:
            return "No suitable strategy found."
        
        best = recommendations[0]
        return f"""Recommended: {best['strategy']} with {best['model']}
Score: {best['score']:.2f}
Est. cost/query: ${best['estimated_cost_per_query']:.4f}"""
```

---

## Context Compression

### Compression Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                   CONTEXT COMPRESSION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Original Context (100K tokens)                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Document 1 │ Document 2 │ Document 3 │ ... │ Document N     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              IMPORTANCE SCORING                              │    │
│  │  Score each sentence/chunk for relevance to query            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              SELECTIVE EXTRACTION                            │    │
│  │  Keep high-importance content, remove redundancy             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              ABSTRACTIVE SUMMARIZATION                       │    │
│  │  Optionally summarize remaining content                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Compressed Context (10K tokens)                                     │
│  ┌───────────────────────────────────────┐                          │
│  │ Key facts │ Summaries │ Critical details                    │    │
│  └───────────────────────────────────────┘                          │
│                                                                      │
│  Compression Ratio: 10x                                              │
│  Information Retention: ~90%                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class CompressionStrategy(ABC):
    """Base class for context compression strategies."""
    
    @abstractmethod
    async def compress(
        self,
        context: str,
        query: str,
        target_tokens: int
    ) -> str:
        """Compress context to target token count."""
        pass

class ImportanceBasedCompressor(CompressionStrategy):
    """Compress by scoring and selecting important content."""
    
    def __init__(
        self,
        embedding_model,
        token_counter: TokenCounter
    ):
        self.embedder = embedding_model
        self.counter = token_counter
    
    async def compress(
        self,
        context: str,
        query: str,
        target_tokens: int
    ) -> str:
        """Compress using importance scoring."""
        
        # Split into sentences
        sentences = self._split_sentences(context)
        
        # Score each sentence
        scored = await self._score_sentences(sentences, query)
        
        # Select top sentences within budget
        selected = self._select_within_budget(scored, target_tokens)
        
        # Maintain original order
        selected_sorted = sorted(selected, key=lambda x: x[2])  # by original index
        
        return " ".join([s[0] for s in selected_sorted])
    
    async def _score_sentences(
        self,
        sentences: List[str],
        query: str
    ) -> List[Tuple[str, float, int]]:
        """Score sentences by relevance to query."""
        
        # Get query embedding
        query_embedding = await self.embedder.embed(query)
        
        # Get sentence embeddings
        sentence_embeddings = await self.embedder.embed_batch(sentences)
        
        # Calculate similarity scores
        scored = []
        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            similarity = self._cosine_similarity(query_embedding, embedding)
            scored.append((sentence, similarity, i))
        
        return scored
    
    def _select_within_budget(
        self,
        scored: List[Tuple[str, float, int]],
        target_tokens: int
    ) -> List[Tuple[str, float, int]]:
        """Select top-scoring sentences within token budget."""
        
        # Sort by score descending
        sorted_scored = sorted(scored, key=lambda x: x[1], reverse=True)
        
        selected = []
        used_tokens = 0
        
        for item in sorted_scored:
            sentence_tokens = self.counter.count_tokens(item[0], "gpt-4")
            if used_tokens + sentence_tokens <= target_tokens:
                selected.append(item)
                used_tokens += sentence_tokens
        
        return selected
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


class SummarizationCompressor(CompressionStrategy):
    """Compress using LLM summarization."""
    
    def __init__(self, llm_client, token_counter: TokenCounter):
        self.llm = llm_client
        self.counter = token_counter
    
    async def compress(
        self,
        context: str,
        query: str,
        target_tokens: int
    ) -> str:
        """Compress using hierarchical summarization."""
        
        context_tokens = self.counter.count_tokens(context, "gpt-4")
        
        if context_tokens <= target_tokens:
            return context
        
        # Determine compression approach
        if context_tokens <= target_tokens * 4:
            # Single-pass summarization
            return await self._single_summarize(context, query, target_tokens)
        else:
            # Hierarchical summarization
            return await self._hierarchical_summarize(context, query, target_tokens)
    
    async def _single_summarize(
        self,
        context: str,
        query: str,
        target_tokens: int
    ) -> str:
        """Single-pass summarization."""
        
        target_words = int(target_tokens * 0.75)  # Rough token-to-word ratio
        
        prompt = f"""Summarize the following context, focusing on information relevant to answering this question: {query}

Context:
{context}

Provide a concise summary in approximately {target_words} words. Preserve key facts, numbers, and specific details that are relevant to the question."""
        
        return await self.llm.generate(prompt)
    
    async def _hierarchical_summarize(
        self,
        context: str,
        query: str,
        target_tokens: int
    ) -> str:
        """Multi-level hierarchical summarization."""
        
        # Split into chunks
        chunk_size = 3000  # tokens per chunk
        chunks = self._split_into_chunks(context, chunk_size)
        
        # First level: summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = await self._single_summarize(chunk, query, chunk_size // 4)
            chunk_summaries.append(summary)
        
        # Combine summaries
        combined = "\n\n".join(chunk_summaries)
        combined_tokens = self.counter.count_tokens(combined, "gpt-4")
        
        # If still too large, recurse
        if combined_tokens > target_tokens:
            return await self._hierarchical_summarize(combined, query, target_tokens)
        
        return combined
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately chunk_size tokens."""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        paragraphs = text.split("\n\n")
        
        for para in paragraphs:
            para_tokens = self.counter.count_tokens(para, "gpt-4")
            
            if current_tokens + para_tokens <= chunk_size:
                current_chunk += para + "\n\n"
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
                current_tokens = para_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


class LLMLinguaCompressor(CompressionStrategy):
    """Compression inspired by LLMLingua approach."""
    
    def __init__(
        self,
        small_model,  # For perplexity calculation
        token_counter: TokenCounter,
        compression_ratio: float = 0.5
    ):
        self.small_model = small_model
        self.counter = token_counter
        self.ratio = compression_ratio
    
    async def compress(
        self,
        context: str,
        query: str,
        target_tokens: int
    ) -> str:
        """Compress using token-level importance."""
        
        # Tokenize
        tokens = self._tokenize(context)
        
        # Calculate per-token perplexity (importance)
        importances = await self._calculate_token_importance(tokens, query)
        
        # Select tokens to keep
        keep_count = min(target_tokens, int(len(tokens) * (1 - self.ratio)))
        
        # Keep most important tokens
        indexed_importance = list(enumerate(importances))
        sorted_by_importance = sorted(indexed_importance, key=lambda x: x[1], reverse=True)
        
        keep_indices = set([idx for idx, _ in sorted_by_importance[:keep_count]])
        
        # Reconstruct maintaining order
        kept_tokens = [tokens[i] for i in range(len(tokens)) if i in keep_indices]
        
        return self._detokenize(kept_tokens)
    
    async def _calculate_token_importance(
        self,
        tokens: List[str],
        query: str
    ) -> List[float]:
        """Calculate importance score for each token."""
        
        # This is a simplified version
        # Real LLMLingua uses perplexity from a small LM
        
        query_tokens = set(self._tokenize(query.lower()))
        
        importances = []
        for i, token in enumerate(tokens):
            score = 0.5  # Base importance
            
            # Boost query-related tokens
            if token.lower() in query_tokens:
                score += 0.3
            
            # Boost tokens near punctuation (sentence boundaries)
            if i > 0 and tokens[i-1] in '.!?':
                score += 0.1
            if i < len(tokens) - 1 and tokens[i+1] in '.!?':
                score += 0.1
            
            # Boost named entities, numbers
            if token[0].isupper() or token.isdigit():
                score += 0.2
            
            importances.append(score)
        
        return importances
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return text.split()
    
    def _detokenize(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens."""
        return " ".join(tokens)


class HybridCompressor:
    """Combines multiple compression strategies."""
    
    def __init__(
        self,
        importance_compressor: ImportanceBasedCompressor,
        summarization_compressor: SummarizationCompressor,
        token_counter: TokenCounter
    ):
        self.importance = importance_compressor
        self.summarizer = summarization_compressor
        self.counter = token_counter
    
    async def compress(
        self,
        context: str,
        query: str,
        target_tokens: int,
        strategy: str = "auto"
    ) -> str:
        """Compress using hybrid approach."""
        
        context_tokens = self.counter.count_tokens(context, "gpt-4")
        compression_ratio = target_tokens / context_tokens
        
        if strategy == "auto":
            strategy = self._select_strategy(compression_ratio)
        
        if strategy == "importance":
            return await self.importance.compress(context, query, target_tokens)
        elif strategy == "summarize":
            return await self.summarizer.compress(context, query, target_tokens)
        elif strategy == "hybrid":
            # First reduce with importance, then summarize
            intermediate_tokens = int(target_tokens * 2)
            intermediate = await self.importance.compress(
                context, query, intermediate_tokens
            )
            return await self.summarizer.compress(
                intermediate, query, target_tokens
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _select_strategy(self, compression_ratio: float) -> str:
        """Select compression strategy based on ratio needed."""
        if compression_ratio > 0.7:
            return "importance"  # Light compression
        elif compression_ratio > 0.3:
            return "hybrid"  # Medium compression
        else:
            return "summarize"  # Heavy compression
```

---

## Sliding Window Patterns

### Sliding Window Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   SLIDING WINDOW PATTERNS                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  FIXED WINDOW WITH OVERLAP                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Window 1 │▓▓▓│ Window 2 │▓▓▓│ Window 3 │▓▓▓│ Window 4 │      │   │
│  │◄────────►│overlap│◄──────►│overlap│◄──────►│overlap│◄──────►│      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ROLLING SUMMARIZATION                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                               │   │
│  │  [Summary of older] + [Recent window] ──► [New Summary]       │   │
│  │       ▲                      │                  │             │   │
│  │       └──────────────────────┴──────────────────┘             │   │
│  │            Continuous compression                              │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  HIERARCHICAL CONTEXT                                                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Level 0: High-level summary (always included)                │   │
│  │  Level 1: Section summaries (selectively included)            │   │
│  │  Level 2: Recent detailed content (full detail)               │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │ L0 │ L1: Relevant │ L2: Full Recent Detail              │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class Message:
    """Represents a conversation message."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    importance: float = 0.5

@dataclass
class WindowConfig:
    """Configuration for sliding window."""
    max_tokens: int
    overlap_tokens: int = 500
    summary_trigger_tokens: int = None  # When to trigger summarization
    preserve_system: bool = True
    preserve_recent_turns: int = 2

class SlidingWindowManager:
    """Manage conversation with sliding window."""
    
    def __init__(
        self,
        config: WindowConfig,
        token_counter: TokenCounter,
        llm_client = None  # For summarization
    ):
        self.config = config
        self.counter = token_counter
        self.llm = llm_client
        
        self.messages: List[Message] = []
        self.summary: Optional[str] = None
        self.total_tokens = 0
    
    def add_message(self, role: str, content: str, importance: float = 0.5):
        """Add message to conversation."""
        token_count = self.counter.count_tokens(content, "gpt-4")
        
        message = Message(
            role=role,
            content=content,
            token_count=token_count,
            importance=importance
        )
        
        self.messages.append(message)
        self.total_tokens += token_count
        
        # Check if we need to slide the window
        if self.total_tokens > self.config.max_tokens:
            self._slide_window()
    
    def _slide_window(self):
        """Slide window to fit within token budget."""
        target = self.config.max_tokens - self.config.overlap_tokens
        
        # Preserve system messages and recent turns
        preserved = []
        removable = []
        
        for i, msg in enumerate(self.messages):
            if msg.role == "system" and self.config.preserve_system:
                preserved.append(msg)
            elif i >= len(self.messages) - self.config.preserve_recent_turns * 2:
                preserved.append(msg)
            else:
                removable.append(msg)
        
        # Calculate preserved tokens
        preserved_tokens = sum(m.token_count for m in preserved)
        
        if preserved_tokens >= target:
            # Need to summarize even recent content
            self._aggressive_summarize(preserved, target)
            return
        
        # Remove oldest messages until we fit
        removed = []
        while removable and (preserved_tokens + sum(m.token_count for m in removable)) > target:
            removed.append(removable.pop(0))
        
        # Generate summary of removed messages
        if removed and self.llm:
            asyncio.create_task(self._summarize_removed(removed))
        
        # Reconstruct messages
        self.messages = (
            [m for m in preserved if m.role == "system"] +
            removable +
            [m for m in preserved if m.role != "system"]
        )
        
        self._recalculate_tokens()
    
    async def _summarize_removed(self, removed: List[Message]):
        """Summarize removed messages."""
        content = "\n".join([f"{m.role}: {m.content}" for m in removed])
        
        new_summary = await self.llm.generate(f"""
Summarize this conversation segment, preserving key information:

{content}

Previous summary: {self.summary or "None"}

Provide a concise summary:""")
        
        self.summary = new_summary
    
    def get_context(self) -> List[dict]:
        """Get current context for LLM."""
        context = []
        
        # Include summary if exists
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}"
            })
        
        # Include messages
        for msg in self.messages:
            context.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return context
    
    def _recalculate_tokens(self):
        """Recalculate total tokens."""
        self.total_tokens = sum(m.token_count for m in self.messages)
        if self.summary:
            self.total_tokens += self.counter.count_tokens(self.summary, "gpt-4")


class RollingSummarizer:
    """Continuously summarize conversation as it grows."""
    
    def __init__(
        self,
        llm_client,
        token_counter: TokenCounter,
        summary_window: int = 5000,  # Tokens before triggering summary
        target_summary_size: int = 1000
    ):
        self.llm = llm_client
        self.counter = token_counter
        self.summary_window = summary_window
        self.target_size = target_summary_size
        
        self.running_summary: str = ""
        self.unsummarized: List[Message] = []
        self.unsummarized_tokens: int = 0
    
    async def add_and_maybe_summarize(
        self,
        message: Message
    ) -> Optional[str]:
        """Add message and summarize if threshold reached."""
        
        self.unsummarized.append(message)
        self.unsummarized_tokens += message.token_count
        
        if self.unsummarized_tokens >= self.summary_window:
            await self._roll_summary()
            return self.running_summary
        
        return None
    
    async def _roll_summary(self):
        """Roll unsummarized content into summary."""
        
        content = "\n".join([
            f"{m.role}: {m.content}" 
            for m in self.unsummarized
        ])
        
        prompt = f"""Update the running summary with new conversation content.

Current summary:
{self.running_summary or "No previous summary."}

New content to incorporate:
{content}

Provide an updated summary in approximately {self.target_size // 4} words.
Preserve key facts, decisions, and context needed for future conversation."""
        
        self.running_summary = await self.llm.generate(prompt)
        
        # Clear unsummarized buffer (keep last 2 messages for continuity)
        self.unsummarized = self.unsummarized[-2:]
        self.unsummarized_tokens = sum(m.token_count for m in self.unsummarized)
    
    def get_full_context(self, recent_messages: List[Message]) -> str:
        """Get summary + recent messages."""
        context_parts = []
        
        if self.running_summary:
            context_parts.append(f"Conversation summary:\n{self.running_summary}")
        
        if self.unsummarized:
            unsummarized_text = "\n".join([
                f"{m.role}: {m.content}" 
                for m in self.unsummarized
            ])
            context_parts.append(f"Recent unsummarized:\n{unsummarized_text}")
        
        recent_text = "\n".join([
            f"{m.role}: {m.content}" 
            for m in recent_messages
        ])
        context_parts.append(f"Current exchange:\n{recent_text}")
        
        return "\n\n".join(context_parts)


class HierarchicalContextManager:
    """Manage context with hierarchical summarization levels."""
    
    def __init__(
        self,
        llm_client,
        token_counter: TokenCounter,
        levels: int = 3
    ):
        self.llm = llm_client
        self.counter = token_counter
        self.levels = levels
        
        # Level 0: Global summary (always included)
        # Level 1: Section summaries (selectively included)
        # Level 2: Recent detail (full content)
        self.hierarchy: Dict[int, List[str]] = {i: [] for i in range(levels)}
        self.global_summary: str = ""
    
    async def add_content(self, content: str, level: int = 2):
        """Add content at specified level."""
        self.hierarchy[level].append(content)
        
        # Check if level needs compression
        level_tokens = self._count_level_tokens(level)
        threshold = self._level_threshold(level)
        
        if level_tokens > threshold:
            await self._compress_level(level)
    
    def _level_threshold(self, level: int) -> int:
        """Get token threshold for each level."""
        # Lower levels have smaller thresholds
        base = 2000
        return base * (2 ** level)
    
    def _count_level_tokens(self, level: int) -> int:
        """Count tokens at level."""
        return sum(
            self.counter.count_tokens(c, "gpt-4") 
            for c in self.hierarchy[level]
        )
    
    async def _compress_level(self, level: int):
        """Compress content at level, pushing summary to level above."""
        if level == 0:
            # Can't compress further, update global summary
            await self._update_global_summary()
            return
        
        content = "\n\n".join(self.hierarchy[level])
        
        summary = await self.llm.generate(f"""
Summarize this content concisely:

{content}

Preserve key facts and important details.""")
        
        # Push summary to level above
        self.hierarchy[level - 1].append(summary)
        
        # Clear current level (keep most recent)
        self.hierarchy[level] = self.hierarchy[level][-1:]
    
    async def _update_global_summary(self):
        """Update the global summary."""
        all_l0 = "\n\n".join(self.hierarchy[0])
        
        self.global_summary = await self.llm.generate(f"""
Create a comprehensive summary incorporating:

Previous summary: {self.global_summary or "None"}

New content: {all_l0}

Provide a concise but complete summary.""")
        
        self.hierarchy[0] = []
    
    def get_context(self, query: str, max_tokens: int) -> str:
        """Get hierarchical context for query."""
        context_parts = []
        used_tokens = 0
        
        # Always include global summary
        if self.global_summary:
            context_parts.append(f"Overview:\n{self.global_summary}")
            used_tokens += self.counter.count_tokens(self.global_summary, "gpt-4")
        
        # Add recent detail (level 2)
        for content in reversed(self.hierarchy[2]):
            tokens = self.counter.count_tokens(content, "gpt-4")
            if used_tokens + tokens <= max_tokens:
                context_parts.insert(1, content)  # Insert after overview
                used_tokens += tokens
        
        # Fill remaining with level 1 summaries
        remaining = max_tokens - used_tokens
        for summary in self.hierarchy[1]:
            tokens = self.counter.count_tokens(summary, "gpt-4")
            if tokens <= remaining:
                context_parts.insert(1, summary)
                remaining -= tokens
        
        return "\n\n---\n\n".join(context_parts)
```

---

## Memory Systems

### Memory Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MEMORY SYSTEM ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    WORKING MEMORY                            │    │
│  │  Current task state, active context window                    │    │
│  │  ┌─────────────────────────────────────────────────────────┐ │    │
│  │  │ System Prompt │ Retrieved Docs │ Recent Messages │ Task │ │    │
│  │  └─────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│          ┌───────────────────┼───────────────────┐                  │
│          ▼                   ▼                   ▼                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │SHORT-TERM  │     │  EPISODIC   │     │  SEMANTIC   │           │
│  │  MEMORY    │     │   MEMORY    │     │   MEMORY    │           │
│  │            │     │             │     │             │           │
│  │ Recent     │     │ Specific    │     │ Facts &     │           │
│  │ conversation│    │ interactions│     │ knowledge   │           │
│  │ Last N turns│    │ Key moments │     │ User prefs  │           │
│  │            │     │ Outcomes    │     │ Learned info│           │
│  │ In-memory  │     │ Vector DB   │     │ Knowledge   │           │
│  │ buffer     │     │ w/ metadata │     │ graph/DB    │           │
│  └─────────────┘     └─────────────┘     └─────────────┘           │
│                                                                      │
│                    ┌─────────────────┐                              │
│                    │   LONG-TERM     │                              │
│                    │    MEMORY       │                              │
│                    │                 │                              │
│                    │ Persistent      │                              │
│                    │ storage of all  │                              │
│                    │ memory types    │                              │
│                    │                 │                              │
│                    │ Vector DB +     │                              │
│                    │ Metadata store  │                              │
│                    └─────────────────┘                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import uuid

class MemoryType(Enum):
    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"

@dataclass
class MemoryItem:
    """A single memory item."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: MemoryType = MemoryType.SHORT_TERM
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)

@dataclass
class WorkingMemoryState:
    """Current working memory state."""
    system_prompt: str = ""
    active_documents: List[str] = field(default_factory=list)
    recent_messages: List[Message] = field(default_factory=list)
    current_task: Optional[str] = None
    task_context: Dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    """Manages recent conversation history."""
    
    def __init__(
        self,
        max_items: int = 20,
        max_tokens: int = 4000,
        token_counter: TokenCounter = None
    ):
        self.max_items = max_items
        self.max_tokens = max_tokens
        self.counter = token_counter or TokenCounter()
        self.buffer: List[MemoryItem] = []
    
    def add(self, content: str, role: str = "user", importance: float = 0.5):
        """Add item to short-term memory."""
        item = MemoryItem(
            content=content,
            memory_type=MemoryType.SHORT_TERM,
            importance=importance,
            metadata={"role": role}
        )
        
        self.buffer.append(item)
        self._enforce_limits()
    
    def _enforce_limits(self):
        """Enforce size limits."""
        # Enforce item limit
        while len(self.buffer) > self.max_items:
            # Remove lowest importance item (not from last 5)
            candidates = self.buffer[:-5]
            if candidates:
                min_item = min(candidates, key=lambda x: x.importance)
                self.buffer.remove(min_item)
            else:
                self.buffer.pop(0)
        
        # Enforce token limit
        total_tokens = sum(
            self.counter.count_tokens(item.content, "gpt-4")
            for item in self.buffer
        )
        
        while total_tokens > self.max_tokens and len(self.buffer) > 2:
            removed = self.buffer.pop(0)
            total_tokens -= self.counter.count_tokens(removed.content, "gpt-4")
    
    def get_recent(self, n: int = None) -> List[MemoryItem]:
        """Get recent items."""
        n = n or len(self.buffer)
        return self.buffer[-n:]
    
    def to_messages(self) -> List[Dict]:
        """Convert to message format."""
        return [
            {"role": item.metadata.get("role", "user"), "content": item.content}
            for item in self.buffer
        ]


class EpisodicMemory:
    """Stores specific interaction episodes."""
    
    def __init__(
        self,
        embedding_model,
        vector_store,
        importance_threshold: float = 0.6
    ):
        self.embedder = embedding_model
        self.store = vector_store
        self.threshold = importance_threshold
    
    async def record_episode(
        self,
        content: str,
        outcome: str,
        importance: float,
        metadata: Dict[str, Any] = None
    ):
        """Record an episode if important enough."""
        if importance < self.threshold:
            return
        
        episode = MemoryItem(
            content=content,
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            metadata={
                **(metadata or {}),
                "outcome": outcome,
                "recorded_at": datetime.now().isoformat()
            }
        )
        
        # Generate embedding
        episode.embedding = await self.embedder.embed(content)
        
        # Store
        await self.store.upsert([{
            "id": episode.id,
            "embedding": episode.embedding,
            "content": episode.content,
            "metadata": {
                **episode.metadata,
                "importance": episode.importance,
                "type": "episodic"
            }
        }])
    
    async def recall_similar(
        self,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.0
    ) -> List[MemoryItem]:
        """Recall episodes similar to query."""
        query_embedding = await self.embedder.embed(query)
        
        results = await self.store.search(
            embedding=query_embedding,
            top_k=top_k * 2,  # Get extra for filtering
            filter={"type": "episodic"}
        )
        
        # Filter by importance and convert to MemoryItems
        episodes = []
        for result in results:
            if result.metadata.get("importance", 0) >= min_importance:
                episodes.append(MemoryItem(
                    id=result.id,
                    content=result.content,
                    memory_type=MemoryType.EPISODIC,
                    importance=result.metadata.get("importance", 0.5),
                    metadata=result.metadata
                ))
            if len(episodes) >= top_k:
                break
        
        return episodes


class SemanticMemory:
    """Stores facts and learned knowledge."""
    
    def __init__(
        self,
        embedding_model,
        vector_store,
        llm_client = None
    ):
        self.embedder = embedding_model
        self.store = vector_store
        self.llm = llm_client
    
    async def store_fact(
        self,
        fact: str,
        source: str = None,
        confidence: float = 1.0,
        category: str = None
    ):
        """Store a fact in semantic memory."""
        item = MemoryItem(
            content=fact,
            memory_type=MemoryType.SEMANTIC,
            importance=confidence,
            metadata={
                "source": source,
                "category": category,
                "type": "semantic"
            }
        )
        
        item.embedding = await self.embedder.embed(fact)
        
        await self.store.upsert([{
            "id": item.id,
            "embedding": item.embedding,
            "content": item.content,
            "metadata": item.metadata
        }])
    
    async def extract_and_store_facts(self, text: str, source: str = None):
        """Extract facts from text and store them."""
        if not self.llm:
            raise ValueError("LLM client required for fact extraction")
        
        response = await self.llm.generate(f"""
Extract key facts from this text. Return as JSON array of objects with 'fact' and 'category' fields.

Text: {text}

Categories to use: person, place, event, concept, preference, other
""",
            response_format={"type": "json_object"}
        )
        
        facts = response.get("facts", [])
        
        for fact_obj in facts:
            await self.store_fact(
                fact=fact_obj["fact"],
                source=source,
                category=fact_obj.get("category", "other")
            )
    
    async def recall_facts(
        self,
        query: str,
        category: str = None,
        top_k: int = 10
    ) -> List[MemoryItem]:
        """Recall facts relevant to query."""
        query_embedding = await self.embedder.embed(query)
        
        filter_dict = {"type": "semantic"}
        if category:
            filter_dict["category"] = category
        
        results = await self.store.search(
            embedding=query_embedding,
            top_k=top_k,
            filter=filter_dict
        )
        
        return [
            MemoryItem(
                id=r.id,
                content=r.content,
                memory_type=MemoryType.SEMANTIC,
                metadata=r.metadata
            )
            for r in results
        ]


class UnifiedMemorySystem:
    """Unified interface to all memory types."""
    
    def __init__(
        self,
        embedding_model,
        vector_store,
        llm_client,
        token_counter: TokenCounter
    ):
        self.short_term = ShortTermMemory(token_counter=token_counter)
        self.episodic = EpisodicMemory(embedding_model, vector_store)
        self.semantic = SemanticMemory(embedding_model, vector_store, llm_client)
        self.working = WorkingMemoryState()
        
        self.counter = token_counter
        self.llm = llm_client
    
    async def build_context(
        self,
        query: str,
        max_tokens: int,
        include_types: List[MemoryType] = None
    ) -> str:
        """Build context from all memory types."""
        include_types = include_types or list(MemoryType)
        
        context_parts = []
        used_tokens = 0
        
        # Allocate budget across memory types
        budget_per_type = max_tokens // len(include_types)
        
        if MemoryType.WORKING in include_types:
            # Working memory first
            working_context = self._format_working_memory()
            working_tokens = self.counter.count_tokens(working_context, "gpt-4")
            if working_tokens <= budget_per_type:
                context_parts.append(("Working Memory", working_context))
                used_tokens += working_tokens
        
        if MemoryType.SHORT_TERM in include_types:
            # Short-term memory
            messages = self.short_term.to_messages()
            stm_context = "\n".join([
                f"{m['role']}: {m['content']}" for m in messages
            ])
            stm_tokens = self.counter.count_tokens(stm_context, "gpt-4")
            if used_tokens + stm_tokens <= max_tokens:
                context_parts.append(("Recent Conversation", stm_context))
                used_tokens += stm_tokens
        
        if MemoryType.SEMANTIC in include_types:
            # Relevant facts
            facts = await self.semantic.recall_facts(query, top_k=5)
            facts_context = "\n".join([f"- {f.content}" for f in facts])
            facts_tokens = self.counter.count_tokens(facts_context, "gpt-4")
            if facts_context and used_tokens + facts_tokens <= max_tokens:
                context_parts.append(("Relevant Facts", facts_context))
                used_tokens += facts_tokens
        
        if MemoryType.EPISODIC in include_types:
            # Relevant past episodes
            episodes = await self.episodic.recall_similar(query, top_k=3)
            ep_context = "\n".join([
                f"- {e.content} (Outcome: {e.metadata.get('outcome', 'N/A')})"
                for e in episodes
            ])
            ep_tokens = self.counter.count_tokens(ep_context, "gpt-4")
            if ep_context and used_tokens + ep_tokens <= max_tokens:
                context_parts.append(("Relevant Past Interactions", ep_context))
                used_tokens += ep_tokens
        
        # Format final context
        formatted = []
        for title, content in context_parts:
            formatted.append(f"## {title}\n{content}")
        
        return "\n\n".join(formatted)
    
    def _format_working_memory(self) -> str:
        """Format working memory state."""
        parts = []
        
        if self.working.current_task:
            parts.append(f"Current Task: {self.working.current_task}")
        
        if self.working.task_context:
            parts.append(f"Task Context: {self.working.task_context}")
        
        return "\n".join(parts)
    
    # Convenience methods
    def add_message(self, content: str, role: str = "user"):
        """Add message to short-term memory."""
        self.short_term.add(content, role)
    
    async def learn_fact(self, fact: str, source: str = None):
        """Add fact to semantic memory."""
        await self.semantic.store_fact(fact, source)
    
    async def record_interaction(
        self,
        content: str,
        outcome: str,
        importance: float = 0.7
    ):
        """Record important interaction."""
        await self.episodic.record_episode(content, outcome, importance)
```

---

## Conversation Management

### Message History Strategies

```
┌─────────────────────────────────────────────────────────────────────┐
│              CONVERSATION MANAGEMENT STRATEGIES                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TURN-BASED TRUNCATION                                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Keep last N turns, discard older                            │    │
│  │                                                              │    │
│  │ [Discarded] [Discarded] [Turn N-2] [Turn N-1] [Turn N]      │    │
│  │     ▼                                                        │    │
│  │ Simple but loses important early context                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  SUMMARY-BASED COMPRESSION                                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Summarize older turns, keep recent verbatim                 │    │
│  │                                                              │    │
│  │ [Summary of T1-T5] [Turn 6] [Turn 7] [Turn 8]               │    │
│  │     ▼                                                        │    │
│  │ Preserves context but adds latency/cost                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  IMPORTANCE-WEIGHTED RETENTION                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Score each turn, keep high-importance regardless of age     │    │
│  │                                                              │    │
│  │ [Important T2] [Important T5] [T7] [T8] [T9] [T10]          │    │
│  │     ▼                                                        │    │
│  │ Best context but requires importance scoring                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  HYBRID: SUMMARY + IMPORTANCE + RECENT                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ [Summary] [Key Moments] [Recent Window]                     │    │
│  │     ▼                                                        │    │
│  │ Most complete but most complex                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from abc import ABC, abstractmethod

class ConversationManager(ABC):
    """Base class for conversation management strategies."""
    
    @abstractmethod
    def add_turn(self, role: str, content: str):
        """Add a conversation turn."""
        pass
    
    @abstractmethod
    def get_context(self) -> List[Dict]:
        """Get context for LLM."""
        pass

class TurnBasedManager(ConversationManager):
    """Simple turn-based truncation."""
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.turns: List[Dict] = []
    
    def add_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})
        
        # Keep only last N turns (pairs of user/assistant)
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-(self.max_turns * 2):]
    
    def get_context(self) -> List[Dict]:
        return self.turns.copy()


class SummaryBasedManager(ConversationManager):
    """Summarize older context."""
    
    def __init__(
        self,
        llm_client,
        token_counter: TokenCounter,
        max_recent_tokens: int = 3000,
        summary_threshold: int = 5000
    ):
        self.llm = llm_client
        self.counter = token_counter
        self.max_recent = max_recent_tokens
        self.threshold = summary_threshold
        
        self.summary: str = ""
        self.recent_turns: List[Dict] = []
    
    def add_turn(self, role: str, content: str):
        self.recent_turns.append({"role": role, "content": content})
        
        # Check if we need to summarize
        recent_tokens = self._count_recent_tokens()
        if recent_tokens > self.threshold:
            asyncio.create_task(self._compress())
    
    async def _compress(self):
        """Compress older turns into summary."""
        # Split recent turns
        keep_count = 4  # Keep last 2 exchanges
        to_summarize = self.recent_turns[:-keep_count]
        self.recent_turns = self.recent_turns[-keep_count:]
        
        # Generate new summary
        content = "\n".join([
            f"{t['role']}: {t['content']}" for t in to_summarize
        ])
        
        self.summary = await self.llm.generate(f"""
Update the conversation summary:

Current summary: {self.summary or "None"}

New content:
{content}

Provide a concise updated summary preserving key information.""")
    
    def _count_recent_tokens(self) -> int:
        total = sum(
            self.counter.count_tokens(t["content"], "gpt-4")
            for t in self.recent_turns
        )
        if self.summary:
            total += self.counter.count_tokens(self.summary, "gpt-4")
        return total
    
    def get_context(self) -> List[Dict]:
        context = []
        
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}"
            })
        
        context.extend(self.recent_turns)
        return context


class ImportanceWeightedManager(ConversationManager):
    """Keep turns based on importance scores."""
    
    def __init__(
        self,
        llm_client,
        token_counter: TokenCounter,
        max_tokens: int = 8000
    ):
        self.llm = llm_client
        self.counter = token_counter
        self.max_tokens = max_tokens
        
        self.turns: List[Dict] = []  # Each dict has role, content, importance
    
    def add_turn(self, role: str, content: str, importance: float = None):
        if importance is None:
            importance = asyncio.get_event_loop().run_until_complete(
                self._score_importance(content)
            )
        
        self.turns.append({
            "role": role,
            "content": content,
            "importance": importance,
            "timestamp": datetime.now()
        })
        
        self._enforce_budget()
    
    async def _score_importance(self, content: str) -> float:
        """Score importance of a turn."""
        response = await self.llm.generate(f"""
Rate the importance of this conversation turn for future context.
Consider: decisions made, key information shared, action items, preferences expressed.

Turn: {content[:500]}

Return a score from 0.0 to 1.0 where 1.0 is critically important.
Just return the number.""")
        
        try:
            return float(response.strip())
        except:
            return 0.5
    
    def _enforce_budget(self):
        """Remove low-importance turns to fit budget."""
        # Always keep last 4 turns
        protected = self.turns[-4:]
        candidates = self.turns[:-4]
        
        # Sort by importance
        candidates.sort(key=lambda t: t["importance"], reverse=True)
        
        # Calculate budget for candidates
        protected_tokens = sum(
            self.counter.count_tokens(t["content"], "gpt-4")
            for t in protected
        )
        available = self.max_tokens - protected_tokens
        
        # Select top candidates within budget
        selected = []
        used = 0
        for turn in candidates:
            tokens = self.counter.count_tokens(turn["content"], "gpt-4")
            if used + tokens <= available:
                selected.append(turn)
                used += tokens
        
        # Sort by timestamp and combine with protected
        selected.sort(key=lambda t: t["timestamp"])
        self.turns = selected + protected
    
    def get_context(self) -> List[Dict]:
        return [
            {"role": t["role"], "content": t["content"]}
            for t in self.turns
        ]


class HybridConversationManager(ConversationManager):
    """Combines summary, importance, and recency."""
    
    def __init__(
        self,
        llm_client,
        token_counter: TokenCounter,
        config: Dict = None
    ):
        self.llm = llm_client
        self.counter = token_counter
        
        config = config or {}
        self.summary_budget = config.get("summary_budget", 1000)
        self.important_budget = config.get("important_budget", 2000)
        self.recent_budget = config.get("recent_budget", 3000)
        
        self.summary: str = ""
        self.important_turns: List[Dict] = []
        self.recent_turns: List[Dict] = []
    
    def add_turn(self, role: str, content: str):
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        }
        
        self.recent_turns.append(turn)
        self._rebalance()
    
    def _rebalance(self):
        """Rebalance content across tiers."""
        recent_tokens = self._count_tokens(self.recent_turns)
        
        if recent_tokens > self.recent_budget:
            # Move oldest from recent to important candidates
            overflow = self.recent_turns[:-6]
            self.recent_turns = self.recent_turns[-6:]
            
            # Score and potentially keep important ones
            asyncio.create_task(self._process_overflow(overflow))
    
    async def _process_overflow(self, turns: List[Dict]):
        """Process overflowed turns."""
        for turn in turns:
            importance = await self._score_importance(turn["content"])
            
            if importance > 0.7:
                # Keep in important tier
                turn["importance"] = importance
                self.important_turns.append(turn)
            else:
                # Add to summary
                await self._update_summary(turn)
        
        # Trim important tier if needed
        self._trim_important()
    
    async def _score_importance(self, content: str) -> float:
        """Score importance of content."""
        # Simplified scoring
        indicators = ["decided", "agreed", "important", "must", "should", "will"]
        score = 0.3
        for indicator in indicators:
            if indicator in content.lower():
                score += 0.1
        return min(1.0, score)
    
    async def _update_summary(self, turn: Dict):
        """Update summary with new turn."""
        self.summary = await self.llm.generate(f"""
Update summary with new content:

Current summary: {self.summary or "None"}

New turn ({turn['role']}): {turn['content'][:500]}

Provide updated summary in ~200 words.""")
    
    def _trim_important(self):
        """Trim important tier to budget."""
        tokens = self._count_tokens(self.important_turns)
        
        while tokens > self.important_budget and self.important_turns:
            # Remove lowest importance
            min_turn = min(self.important_turns, key=lambda t: t.get("importance", 0))
            self.important_turns.remove(min_turn)
            tokens = self._count_tokens(self.important_turns)
    
    def _count_tokens(self, turns: List[Dict]) -> int:
        return sum(
            self.counter.count_tokens(t["content"], "gpt-4")
            for t in turns
        )
    
    def get_context(self) -> List[Dict]:
        context = []
        
        # Add summary
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Conversation summary: {self.summary}"
            })
        
        # Add important turns (sorted by time)
        important_sorted = sorted(
            self.important_turns,
            key=lambda t: t["timestamp"]
        )
        for turn in important_sorted:
            context.append({"role": turn["role"], "content": turn["content"]})
        
        # Add recent turns
        for turn in self.recent_turns:
            context.append({"role": turn["role"], "content": turn["content"]})
        
        return context
```

---

## Multi-Document Context

### Document Context Management

```
┌─────────────────────────────────────────────────────────────────────┐
│                 MULTI-DOCUMENT CONTEXT STRATEGIES                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  DOCUMENT RANKING FOR CONTEXT                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Query ──► Retrieval ──► Ranking ──► Selection ──► Context  │    │
│  │                           │                                  │    │
│  │                    ┌──────┴──────┐                          │    │
│  │                    │ Rank by:    │                          │    │
│  │                    │ • Relevance │                          │    │
│  │                    │ • Recency   │                          │    │
│  │                    │ • Authority │                          │    │
│  │                    │ • Coverage  │                          │    │
│  │                    └─────────────┘                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  CONTEXT FUSION STRATEGIES                                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │  Sequential:  [Doc1] ──► [Doc2] ──► [Doc3]                  │    │
│  │                                                              │    │
│  │  Interleaved: [D1-chunk1] [D2-chunk1] [D1-chunk2] ...       │    │
│  │                                                              │    │
│  │  Summarized:  [Summary(D1,D2,D3)] + [Key excerpts]          │    │
│  │                                                              │    │
│  │  Structured:  ## Doc1 Title                                 │    │
│  │               [Content]                                      │    │
│  │               ## Doc2 Title                                 │    │
│  │               [Content]                                      │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
@dataclass
class Document:
    """Represents a document for context."""
    id: str
    title: str
    content: str
    source: str
    relevance_score: float = 0.0
    recency_score: float = 0.0
    authority_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class MultiDocumentContextBuilder:
    """Build context from multiple documents."""
    
    def __init__(
        self,
        token_counter: TokenCounter,
        embedding_model = None
    ):
        self.counter = token_counter
        self.embedder = embedding_model
    
    async def build_context(
        self,
        query: str,
        documents: List[Document],
        max_tokens: int,
        strategy: str = "ranked"
    ) -> str:
        """Build context from documents."""
        
        # Rank documents
        ranked = await self._rank_documents(query, documents)
        
        if strategy == "ranked":
            return self._sequential_context(ranked, max_tokens)
        elif strategy == "interleaved":
            return self._interleaved_context(ranked, max_tokens)
        elif strategy == "structured":
            return self._structured_context(ranked, max_tokens)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    async def _rank_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """Rank documents by relevance."""
        if not self.embedder:
            return documents
        
        query_embedding = await self.embedder.embed(query)
        
        for doc in documents:
            doc_embedding = await self.embedder.embed(doc.content[:1000])
            doc.relevance_score = self._cosine_similarity(
                query_embedding, doc_embedding
            )
        
        # Combined score
        for doc in documents:
            doc.metadata["combined_score"] = (
                0.6 * doc.relevance_score +
                0.2 * doc.recency_score +
                0.2 * doc.authority_score
            )
        
        return sorted(
            documents,
            key=lambda d: d.metadata.get("combined_score", 0),
            reverse=True
        )
    
    def _sequential_context(
        self,
        documents: List[Document],
        max_tokens: int
    ) -> str:
        """Build sequential context."""
        context_parts = []
        used_tokens = 0
        
        for i, doc in enumerate(documents):
            doc_text = f"[{i+1}] Source: {doc.source}\n{doc.content}"
            doc_tokens = self.counter.count_tokens(doc_text, "gpt-4")
            
            if used_tokens + doc_tokens <= max_tokens:
                context_parts.append(doc_text)
                used_tokens += doc_tokens
            else:
                # Try to fit partial
                remaining = max_tokens - used_tokens
                if remaining > 200:
                    truncated = self._truncate_to_tokens(doc_text, remaining)
                    context_parts.append(truncated)
                break
        
        return "\n\n".join(context_parts)
    
    def _interleaved_context(
        self,
        documents: List[Document],
        max_tokens: int
    ) -> str:
        """Interleave chunks from different documents."""
        # Split each document into chunks
        doc_chunks = []
        for doc in documents:
            chunks = self._split_into_chunks(doc.content, 500)
            for j, chunk in enumerate(chunks):
                doc_chunks.append({
                    "source": doc.source,
                    "chunk_idx": j,
                    "content": chunk,
                    "doc_rank": documents.index(doc)
                })
        
        # Sort to interleave: first chunk of each doc, then second, etc.
        doc_chunks.sort(key=lambda c: (c["chunk_idx"], c["doc_rank"]))
        
        context_parts = []
        used_tokens = 0
        
        for chunk in doc_chunks:
            chunk_text = f"[{chunk['source']}]: {chunk['content']}"
            tokens = self.counter.count_tokens(chunk_text, "gpt-4")
            
            if used_tokens + tokens <= max_tokens:
                context_parts.append(chunk_text)
                used_tokens += tokens
        
        return "\n\n".join(context_parts)
    
    def _structured_context(
        self,
        documents: List[Document],
        max_tokens: int
    ) -> str:
        """Build structured context with headers."""
        context_parts = []
        used_tokens = 0
        
        for doc in documents:
            header = f"## {doc.title}\nSource: {doc.source}\n"
            header_tokens = self.counter.count_tokens(header, "gpt-4")
            
            remaining = max_tokens - used_tokens - header_tokens
            if remaining <= 100:
                break
            
            content = self._truncate_to_tokens(doc.content, remaining)
            full_section = header + content
            
            context_parts.append(full_section)
            used_tokens += self.counter.count_tokens(full_section, "gpt-4")
        
        return "\n\n".join(context_parts)
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks."""
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        
        for word in words:
            if current_len + len(word) + 1 > chunk_size:
                chunks.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1
        
        if current:
            chunks.append(" ".join(current))
        
        return chunks
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token limit."""
        tokens = self.counter.count_tokens(text, "gpt-4")
        if tokens <= max_tokens:
            return text
        
        # Binary search for cut point
        ratio = max_tokens / tokens
        cut_point = int(len(text) * ratio * 0.9)  # Start conservative
        
        truncated = text[:cut_point]
        while self.counter.count_tokens(truncated, "gpt-4") > max_tokens:
            cut_point = int(cut_point * 0.9)
            truncated = text[:cut_point]
        
        return truncated + "..."
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


class SourceAttributionManager:
    """Manage source attribution in context."""
    
    def __init__(self):
        self.sources: Dict[str, Document] = {}
    
    def prepare_context_with_citations(
        self,
        documents: List[Document]
    ) -> Tuple[str, Dict[str, str]]:
        """Prepare context with citation markers."""
        context_parts = []
        citation_map = {}
        
        for i, doc in enumerate(documents):
            citation_id = f"[{i+1}]"
            citation_map[citation_id] = {
                "source": doc.source,
                "title": doc.title,
                "id": doc.id
            }
            
            # Mark content with citation
            marked_content = f"{citation_id} {doc.content}"
            context_parts.append(marked_content)
            
            self.sources[citation_id] = doc
        
        return "\n\n".join(context_parts), citation_map
    
    def extract_citations(self, response: str) -> List[Dict]:
        """Extract citations used in response."""
        import re
        
        citations = []
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, response)
        
        for match in set(matches):
            citation_id = f"[{match}]"
            if citation_id in self.sources:
                doc = self.sources[citation_id]
                citations.append({
                    "citation": citation_id,
                    "source": doc.source,
                    "title": doc.title
                })
        
        return citations
```

---

## Production Patterns

### KV Cache and Prefix Caching

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION CACHING PATTERNS                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  KV CACHE (Attention Key-Value Cache)                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │  Without KV Cache:                                           │    │
│  │  Token 1 ─► Compute K,V ─► Attention                        │    │
│  │  Token 2 ─► Compute K,V for 1,2 ─► Attention                │    │
│  │  Token 3 ─► Compute K,V for 1,2,3 ─► Attention              │    │
│  │  O(n²) computation                                           │    │
│  │                                                              │    │
│  │  With KV Cache:                                              │    │
│  │  Token 1 ─► Compute & Store K,V                             │    │
│  │  Token 2 ─► Compute K,V ─► Concat with cached ─► Attention  │    │
│  │  Token 3 ─► Compute K,V ─► Concat with cached ─► Attention  │    │
│  │  O(n) computation                                            │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  PREFIX CACHING                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │  Common Prefix: "You are a helpful assistant. Given the     │    │
│  │                  following documents: [DOC1] [DOC2]..."      │    │
│  │                                                              │    │
│  │  Request 1: [Common Prefix] + "What is X?"                  │    │
│  │  Request 2: [Common Prefix] + "Explain Y"                   │    │
│  │  Request 3: [Common Prefix] + "Compare X and Y"             │    │
│  │                                                              │    │
│  │  ┌─────────────────┐    ┌─────────────────────────────┐     │    │
│  │  │  Precomputed    │───►│ Reuse for all requests      │     │    │
│  │  │  Prefix KV      │    │ Only compute suffix KV      │     │    │
│  │  └─────────────────┘    └─────────────────────────────┘     │    │
│  │                                                              │    │
│  │  Savings: 80%+ reduction in compute for shared prefix       │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
import hashlib
from typing import Optional, Tuple
import time

class KVCacheManager:
    """Manage KV cache for inference optimization."""
    
    def __init__(
        self,
        max_cache_size_gb: float = 10.0,
        eviction_policy: str = "lru"
    ):
        self.max_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.policy = eviction_policy
        
        self.cache: Dict[str, dict] = {}
        self.access_times: Dict[str, float] = {}
        self.current_size = 0
    
    def get_or_create_cache(
        self,
        session_id: str,
        prefix_tokens: List[int] = None
    ) -> dict:
        """Get existing cache or create new one."""
        if session_id in self.cache:
            self.access_times[session_id] = time.time()
            return self.cache[session_id]
        
        # Create new cache entry
        cache_entry = {
            "kv_state": None,  # Will be populated during inference
            "token_count": len(prefix_tokens) if prefix_tokens else 0,
            "created_at": time.time()
        }
        
        # Evict if necessary
        estimated_size = self._estimate_cache_size(cache_entry["token_count"])
        while self.current_size + estimated_size > self.max_size:
            self._evict_one()
        
        self.cache[session_id] = cache_entry
        self.access_times[session_id] = time.time()
        self.current_size += estimated_size
        
        return cache_entry
    
    def update_cache(
        self,
        session_id: str,
        kv_state: Any,
        new_tokens: int
    ):
        """Update cache with new KV state."""
        if session_id not in self.cache:
            return
        
        old_size = self._estimate_cache_size(self.cache[session_id]["token_count"])
        
        self.cache[session_id]["kv_state"] = kv_state
        self.cache[session_id]["token_count"] += new_tokens
        self.access_times[session_id] = time.time()
        
        new_size = self._estimate_cache_size(self.cache[session_id]["token_count"])
        self.current_size += (new_size - old_size)
    
    def _evict_one(self):
        """Evict one cache entry based on policy."""
        if not self.cache:
            return
        
        if self.policy == "lru":
            # Evict least recently used
            oldest = min(self.access_times, key=self.access_times.get)
        elif self.policy == "oldest":
            # Evict oldest created
            oldest = min(
                self.cache,
                key=lambda k: self.cache[k]["created_at"]
            )
        else:
            oldest = next(iter(self.cache))
        
        evicted_size = self._estimate_cache_size(self.cache[oldest]["token_count"])
        del self.cache[oldest]
        del self.access_times[oldest]
        self.current_size -= evicted_size
    
    def _estimate_cache_size(self, token_count: int) -> int:
        """Estimate memory size for KV cache."""
        # Rough estimate: 2 * num_layers * hidden_dim * 2 (K+V) * bytes per float
        # For a 70B model with 80 layers, 8192 hidden: ~2.6MB per token
        bytes_per_token = 2.6 * 1024 * 1024  # Adjust based on model
        return int(token_count * bytes_per_token)


class PrefixCacheManager:
    """Cache common prompt prefixes for reuse."""
    
    def __init__(
        self,
        inference_engine,
        max_prefixes: int = 100
    ):
        self.engine = inference_engine
        self.max_prefixes = max_prefixes
        
        self.prefix_cache: Dict[str, dict] = {}
        self.prefix_hits: Dict[str, int] = {}
    
    def get_prefix_hash(self, prefix: str) -> str:
        """Generate hash for prefix."""
        return hashlib.sha256(prefix.encode()).hexdigest()[:16]
    
    async def get_or_compute_prefix(
        self,
        prefix: str
    ) -> Tuple[str, dict]:
        """Get cached prefix KV or compute it."""
        prefix_hash = self.get_prefix_hash(prefix)
        
        if prefix_hash in self.prefix_cache:
            self.prefix_hits[prefix_hash] = self.prefix_hits.get(prefix_hash, 0) + 1
            return prefix_hash, self.prefix_cache[prefix_hash]
        
        # Compute KV cache for prefix
        kv_cache = await self.engine.compute_prefix_kv(prefix)
        
        # Evict if at capacity
        if len(self.prefix_cache) >= self.max_prefixes:
            self._evict_least_used()
        
        self.prefix_cache[prefix_hash] = {
            "kv_cache": kv_cache,
            "token_count": len(prefix.split()),  # Rough estimate
            "created_at": time.time()
        }
        self.prefix_hits[prefix_hash] = 1
        
        return prefix_hash, self.prefix_cache[prefix_hash]
    
    async def generate_with_prefix(
        self,
        prefix_hash: str,
        continuation: str,
        generation_params: dict
    ) -> str:
        """Generate using cached prefix."""
        if prefix_hash not in self.prefix_cache:
            raise ValueError("Prefix not found in cache")
        
        cached = self.prefix_cache[prefix_hash]
        
        return await self.engine.generate_with_kv_cache(
            kv_cache=cached["kv_cache"],
            new_tokens=continuation,
            **generation_params
        )
    
    def _evict_least_used(self):
        """Evict least frequently used prefix."""
        if not self.prefix_cache:
            return
        
        least_used = min(self.prefix_hits, key=self.prefix_hits.get)
        del self.prefix_cache[least_used]
        del self.prefix_hits[least_used]


class ContextPrecomputeService:
    """Service for precomputing context for common scenarios."""
    
    def __init__(
        self,
        prefix_cache: PrefixCacheManager,
        token_counter: TokenCounter
    ):
        self.prefix_cache = prefix_cache
        self.counter = token_counter
        
        self.precomputed: Dict[str, str] = {}  # scenario -> prefix_hash
    
    async def precompute_scenario(
        self,
        scenario_id: str,
        system_prompt: str,
        common_context: str
    ) -> str:
        """Precompute prefix for a scenario."""
        full_prefix = f"{system_prompt}\n\nContext:\n{common_context}\n\n"
        
        prefix_hash, _ = await self.prefix_cache.get_or_compute_prefix(full_prefix)
        self.precomputed[scenario_id] = prefix_hash
        
        return prefix_hash
    
    async def generate_for_scenario(
        self,
        scenario_id: str,
        user_query: str,
        generation_params: dict = None
    ) -> str:
        """Generate using precomputed scenario prefix."""
        if scenario_id not in self.precomputed:
            raise ValueError(f"Scenario {scenario_id} not precomputed")
        
        prefix_hash = self.precomputed[scenario_id]
        
        return await self.prefix_cache.generate_with_prefix(
            prefix_hash=prefix_hash,
            continuation=f"User: {user_query}\n\nAssistant:",
            generation_params=generation_params or {}
        )


class CostOptimizedContextManager:
    """Optimize context for cost efficiency."""
    
    def __init__(
        self,
        token_counter: TokenCounter,
        compressor: HybridCompressor,
        prefix_cache: PrefixCacheManager
    ):
        self.counter = token_counter
        self.compressor = compressor
        self.prefix_cache = prefix_cache
    
    async def optimize_request(
        self,
        system_prompt: str,
        context: str,
        conversation: List[Dict],
        query: str,
        model: str,
        target_cost: float = None
    ) -> Dict:
        """Optimize request for cost."""
        
        config = MODEL_CONFIGS.get(model)
        if not config:
            raise ValueError(f"Unknown model: {model}")
        
        # Calculate current cost
        total_tokens = (
            self.counter.count_tokens(system_prompt, model) +
            self.counter.count_tokens(context, model) +
            self.counter.count_messages_tokens(conversation, model) +
            self.counter.count_tokens(query, model)
        )
        
        current_cost = (total_tokens / 1000) * config.input_cost_per_1k
        
        optimizations_applied = []
        
        # Check if prefix can be cached
        prefix = system_prompt + "\n\n" + context[:2000]
        prefix_hash, _ = await self.prefix_cache.get_or_compute_prefix(prefix)
        optimizations_applied.append("prefix_caching")
        
        # Compress context if needed
        if target_cost and current_cost > target_cost:
            target_reduction = current_cost - target_cost
            context_tokens = self.counter.count_tokens(context, model)
            
            # Calculate how much to compress
            reduction_ratio = 1 - (target_reduction / (current_cost * 0.5))
            target_context_tokens = int(context_tokens * reduction_ratio)
            
            if target_context_tokens < context_tokens:
                context = await self.compressor.compress(
                    context, query, target_context_tokens
                )
                optimizations_applied.append("context_compression")
        
        # Recalculate
        final_tokens = (
            self.counter.count_tokens(system_prompt, model) +
            self.counter.count_tokens(context, model) +
            self.counter.count_messages_tokens(conversation, model) +
            self.counter.count_tokens(query, model)
        )
        
        final_cost = (final_tokens / 1000) * config.input_cost_per_1k
        
        return {
            "optimized_context": context,
            "prefix_hash": prefix_hash,
            "original_tokens": total_tokens,
            "final_tokens": final_tokens,
            "original_cost": current_cost,
            "final_cost": final_cost,
            "savings_percent": ((current_cost - final_cost) / current_cost) * 100,
            "optimizations": optimizations_applied
        }
```

---

## Best Practices Checklist

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTEXT MANAGEMENT BEST PRACTICES                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TOKEN MANAGEMENT                                                    │
│  ☐ Always count tokens before sending requests                      │
│  ☐ Reserve buffer for output tokens (10-20% of context)             │
│  ☐ Track token usage per request for cost monitoring                │
│  ☐ Use appropriate tokenizer for each model                         │
│                                                                      │
│  CONTEXT ORGANIZATION                                                │
│  ☐ Place critical information at beginning and end                  │
│  ☐ Use clear section markers and headers                            │
│  ☐ Maintain consistent formatting across contexts                   │
│  ☐ Include source attribution for retrieval results                 │
│                                                                      │
│  MEMORY MANAGEMENT                                                   │
│  ☐ Implement appropriate memory type for use case                   │
│  ☐ Set clear retention policies for each memory tier                │
│  ☐ Regularly prune low-importance memories                          │
│  ☐ Test memory retrieval quality periodically                       │
│                                                                      │
│  COMPRESSION                                                         │
│  ☐ Choose compression strategy based on compression ratio needed    │
│  ☐ Validate information retention after compression                 │
│  ☐ Cache compressed versions when context is reused                 │
│  ☐ Monitor compression latency in production                        │
│                                                                      │
│  CACHING                                                             │
│  ☐ Identify common prefixes for prefix caching                      │
│  ☐ Implement KV cache for multi-turn conversations                  │
│  ☐ Set appropriate cache eviction policies                          │
│  ☐ Monitor cache hit rates and adjust strategies                    │
│                                                                      │
│  COST OPTIMIZATION                                                   │
│  ☐ Track cost per request and per user                              │
│  ☐ Implement token budgets for different use cases                  │
│  ☐ Use smaller models for simpler queries                           │
│  ☐ Consider caching entire responses for common queries             │
│                                                                      │
│  TESTING                                                             │
│  ☐ Run needle-in-haystack tests for long context scenarios          │
│  ☐ Test context strategies against diverse query types              │
│  ☐ Benchmark retrieval quality across context sizes                 │
│  ☐ Validate memory systems preserve critical information            │
│                                                                      │
│  MONITORING                                                          │
│  ☐ Track context utilization (tokens used / available)              │
│  ☐ Monitor compression ratios and information loss                  │
│  ☐ Alert on context overflow situations                             │
│  ☐ Log cache performance metrics                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## References

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) - Stanford research on attention patterns
- [LLMLingua: Compressing Prompts for Accelerated Inference](https://arxiv.org/abs/2310.05736) - Microsoft prompt compression
- [Extending Context Window of Large Language Models](https://arxiv.org/abs/2401.01325) - Position interpolation techniques
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) - Memory management for LLMs
- [Anthropic Context Window Research](https://www.anthropic.com/research) - Long context handling
- [vLLM: Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180) - PagedAttention and KV cache
- [RAG vs Long Context](https://arxiv.org/abs/2402.14848) - When to use each approach
- [Tiktoken](https://github.com/openai/tiktoken) - OpenAI tokenizer library
- [LangChain Memory](https://python.langchain.com/docs/modules/memory/) - Memory implementations
- [Needle in a Haystack Test](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) - Long context testing
