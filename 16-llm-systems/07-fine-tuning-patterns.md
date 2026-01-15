# Fine-Tuning Patterns

## TL;DR

Fine-tuning adapts pre-trained LLMs to specific tasks, domains, or behaviors by training on curated datasets. Key decisions include choosing between full fine-tuning, parameter-efficient methods (LoRA, QLoRA, adapters), and prompt-based approaches based on data availability, compute budget, and task requirements. Success depends on high-quality training data, proper hyperparameter selection, rigorous evaluation, and production deployment strategies including model merging and version management.

---

## When to Fine-Tune vs RAG vs Prompting

### Decision Framework

```
┌─────────────────────────────────────────────────────────────────┐
│              FINE-TUNING DECISION TREE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  START HERE                                 │ │
│  │                      │                                      │ │
│  │                      ▼                                      │ │
│  │  ┌─────────────────────────────────────┐                    │ │
│  │  │ Can prompting solve the problem?    │                    │ │
│  │  │ (few-shot, chain-of-thought, etc.)  │                    │ │
│  │  └──────────────┬──────────────────────┘                    │ │
│  │                 │                                           │ │
│  │        YES ◄────┴────► NO                                   │ │
│  │         │              │                                    │ │
│  │         ▼              ▼                                    │ │
│  │    ┌────────┐   ┌─────────────────────────────┐             │ │
│  │    │  USE   │   │ Is the knowledge static or  │             │ │
│  │    │PROMPTING│  │ needs frequent updates?     │             │ │
│  │    └────────┘   └─────────────┬───────────────┘             │ │
│  │                               │                             │ │
│  │                  FREQUENT ◄───┴───► STATIC                  │ │
│  │                     │                  │                    │ │
│  │                     ▼                  ▼                    │ │
│  │               ┌─────────┐    ┌────────────────────────┐     │ │
│  │               │ USE RAG │    │ Do you have >1000      │     │ │
│  │               └─────────┘    │ high-quality examples? │     │ │
│  │                              └───────────┬────────────┘     │ │
│  │                                          │                  │ │
│  │                               NO ◄───────┴───────► YES      │ │
│  │                               │                     │       │ │
│  │                               ▼                     ▼       │ │
│  │                        ┌───────────┐         ┌──────────┐   │ │
│  │                        │ PROMPTING │         │FINE-TUNE │   │ │
│  │                        │  or RAG   │         └──────────┘   │ │
│  │                        └───────────┘                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Approach Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                 APPROACH COMPARISON MATRIX                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CHARACTERISTIC     │ PROMPTING  │    RAG     │  FINE-TUNING    │
│  ──────────────────┼────────────┼────────────┼─────────────────│
│  Setup Time        │ Minutes    │ Hours-Days │ Days-Weeks      │
│  Data Required     │ 0-10       │ Documents  │ 1000+ examples  │
│  Compute Cost      │ Low        │ Medium     │ High            │
│  Inference Cost    │ Higher*    │ Medium     │ Lower           │
│  Knowledge Updates │ Instant    │ Fast       │ Requires retrain│
│  Behavior Changes  │ Limited    │ Limited    │ Deep changes    │
│  Latency           │ Base       │ +retrieval │ Base            │
│  Hallucination     │ Higher     │ Lower      │ Task-dependent  │
│                                                                  │
│  *Longer prompts = more tokens = higher cost                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Use Case Mapping

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class ApproachType(Enum):
    PROMPTING = "prompting"
    RAG = "rag"
    FINE_TUNING = "fine_tuning"
    HYBRID = "hybrid"

@dataclass
class UseCaseProfile:
    """Profile a use case to determine best approach."""
    name: str
    data_size: int  # Number of training examples available
    data_update_frequency: str  # "daily", "weekly", "monthly", "static"
    task_complexity: str  # "simple", "moderate", "complex"
    latency_requirement_ms: int
    budget_monthly_usd: float
    requires_domain_knowledge: bool
    requires_specific_format: bool
    requires_specific_tone: bool

class ApproachRecommender:
    """Recommend approach based on use case profile."""
    
    def recommend(self, profile: UseCaseProfile) -> ApproachType:
        """Determine best approach for use case."""
        
        # Check if fine-tuning is viable
        fine_tune_score = self._score_fine_tuning(profile)
        rag_score = self._score_rag(profile)
        prompt_score = self._score_prompting(profile)
        
        scores = {
            ApproachType.FINE_TUNING: fine_tune_score,
            ApproachType.RAG: rag_score,
            ApproachType.PROMPTING: prompt_score,
        }
        
        best = max(scores, key=scores.get)
        
        # Consider hybrid approaches
        if fine_tune_score > 0.5 and rag_score > 0.5:
            return ApproachType.HYBRID
        
        return best
    
    def _score_fine_tuning(self, profile: UseCaseProfile) -> float:
        score = 0.0
        
        # Data requirements
        if profile.data_size >= 10000:
            score += 0.3
        elif profile.data_size >= 1000:
            score += 0.2
        elif profile.data_size >= 100:
            score += 0.1
        
        # Static data is better for fine-tuning
        if profile.data_update_frequency == "static":
            score += 0.2
        elif profile.data_update_frequency == "monthly":
            score += 0.1
        
        # Complex tasks benefit from fine-tuning
        if profile.task_complexity == "complex":
            score += 0.2
        
        # Format/tone requirements
        if profile.requires_specific_format:
            score += 0.15
        if profile.requires_specific_tone:
            score += 0.15
        
        return min(score, 1.0)
    
    def _score_rag(self, profile: UseCaseProfile) -> float:
        score = 0.0
        
        # Dynamic data favors RAG
        if profile.data_update_frequency in ["daily", "weekly"]:
            score += 0.3
        
        # Domain knowledge requirement
        if profile.requires_domain_knowledge:
            score += 0.25
        
        # Lower data requirements
        if profile.data_size < 1000:
            score += 0.2
        
        # Moderate complexity
        if profile.task_complexity == "moderate":
            score += 0.15
        
        return min(score, 1.0)
    
    def _score_prompting(self, profile: UseCaseProfile) -> float:
        score = 0.0
        
        # Simplicity favors prompting
        if profile.task_complexity == "simple":
            score += 0.4
        
        # Low data available
        if profile.data_size < 100:
            score += 0.3
        
        # Low latency requirements
        if profile.latency_requirement_ms > 5000:
            score += 0.2
        
        return min(score, 1.0)


# Example use case mapping
USE_CASE_RECOMMENDATIONS = {
    "customer_support_bot": {
        "approach": ApproachType.HYBRID,
        "reason": "Fine-tune for tone/format, RAG for knowledge base"
    },
    "code_completion": {
        "approach": ApproachType.FINE_TUNING,
        "reason": "Specific format, static patterns, lots of training data"
    },
    "document_qa": {
        "approach": ApproachType.RAG,
        "reason": "Dynamic documents, factual accuracy critical"
    },
    "sentiment_analysis": {
        "approach": ApproachType.FINE_TUNING,
        "reason": "Classification task, labeled data available"
    },
    "creative_writing": {
        "approach": ApproachType.PROMPTING,
        "reason": "Style can be guided via prompts, subjective output"
    },
}
```

---

## Fine-Tuning Fundamentals

### Full Fine-Tuning vs Parameter-Efficient Methods

```
┌─────────────────────────────────────────────────────────────────┐
│              FINE-TUNING METHOD COMPARISON                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FULL FINE-TUNING                                                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  ALL PARAMETERS UPDATED                                     │ │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │ │
│  │  │ ███ │ │ ███ │ │ ███ │ │ ███ │ │ ███ │ │ ███ │ │ ███ │   │ │
│  │  │TRAIN│ │TRAIN│ │TRAIN│ │TRAIN│ │TRAIN│ │TRAIN│ │TRAIN│   │ │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘   │ │
│  │  Embed   Layer1  Layer2  Layer3  ...     LayerN   Head     │ │
│  │                                                             │ │
│  │  Pros: Maximum adaptation, best performance potential       │ │
│  │  Cons: Expensive, risk of catastrophic forgetting           │ │
│  │  Memory: 4x model size (model + gradients + optimizer)      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  PARAMETER-EFFICIENT (LoRA)                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  ONLY ADAPTER PARAMETERS UPDATED                            │ │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │ │
│  │  │     │ │ +A  │ │ +A  │ │ +A  │ │ +A  │ │ +A  │ │     │   │ │
│  │  │FROZEN│ │FROZEN│ │FROZEN│ │FROZEN│ │FROZEN│ │FROZEN│ │FROZEN│ │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘   │ │
│  │  Embed   Layer1  Layer2  Layer3  ...     LayerN   Head     │ │
│  │                  +A = Small trainable adapter               │ │
│  │                                                             │ │
│  │  Pros: Cheap, fast, no forgetting, multiple adapters        │ │
│  │  Cons: Slightly lower ceiling on performance                │ │
│  │  Memory: ~10-20% of full fine-tuning                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Training Data Requirements

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class TrainingDataRequirements:
    """Requirements for fine-tuning training data."""
    
    # Minimum samples
    minimum_samples: int = 100  # Bare minimum
    recommended_samples: int = 1000  # Good starting point
    optimal_samples: int = 10000  # Best results
    
    # Quality guidelines
    max_input_tokens: int = 2048
    max_output_tokens: int = 2048
    
    @staticmethod
    def estimate_samples_needed(task_complexity: str, base_model_capability: str) -> int:
        """Estimate training samples needed."""
        
        complexity_multiplier = {
            "simple": 1.0,      # Classification, simple extraction
            "moderate": 2.5,    # Summarization, Q&A
            "complex": 5.0,     # Code generation, multi-step reasoning
        }
        
        capability_multiplier = {
            "high": 0.5,        # GPT-4 class models
            "medium": 1.0,      # GPT-3.5, Llama-70B class
            "low": 2.0,         # Smaller models, 7B-13B
        }
        
        base = 1000
        return int(
            base * 
            complexity_multiplier.get(task_complexity, 2.5) * 
            capability_multiplier.get(base_model_capability, 1.0)
        )


@dataclass
class TrainingExample:
    """Single training example for fine-tuning."""
    instruction: str
    input: str = ""
    output: str = ""
    system_prompt: str = ""
    metadata: Dict[str, Any] = None
    
    def to_chat_format(self) -> List[Dict[str, str]]:
        """Convert to chat format for instruction tuning."""
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        user_content = self.instruction
        if self.input:
            user_content += f"\n\nInput:\n{self.input}"
        
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": self.output})
        
        return messages
    
    def to_alpaca_format(self) -> Dict[str, str]:
        """Convert to Alpaca format."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }
    
    def to_sharegpt_format(self) -> Dict[str, Any]:
        """Convert to ShareGPT format."""
        conversations = []
        
        user_content = self.instruction
        if self.input:
            user_content += f"\n\n{self.input}"
        
        conversations.append({"from": "human", "value": user_content})
        conversations.append({"from": "gpt", "value": self.output})
        
        return {
            "conversations": conversations,
            "system": self.system_prompt or "",
        }


class DataValidator:
    """Validate training data quality."""
    
    def __init__(self, tokenizer, max_tokens: int = 4096):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
    
    def validate_example(self, example: TrainingExample) -> Dict[str, Any]:
        """Validate a single training example."""
        issues = []
        
        # Check lengths
        full_text = f"{example.instruction}\n{example.input}\n{example.output}"
        token_count = len(self.tokenizer.encode(full_text))
        
        if token_count > self.max_tokens:
            issues.append(f"Total tokens ({token_count}) exceeds max ({self.max_tokens})")
        
        # Check for empty fields
        if not example.instruction.strip():
            issues.append("Empty instruction")
        if not example.output.strip():
            issues.append("Empty output")
        
        # Check for quality issues
        if len(example.output.split()) < 5:
            issues.append("Output too short (< 5 words)")
        
        # Check for repetition
        if self._has_repetition(example.output):
            issues.append("Output contains excessive repetition")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "token_count": token_count,
        }
    
    def _has_repetition(self, text: str, threshold: float = 0.3) -> bool:
        """Detect excessive repetition in text."""
        words = text.lower().split()
        if len(words) < 10:
            return False
        
        # Check for repeated n-grams
        ngrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        unique_ratio = len(set(ngrams)) / len(ngrams)
        
        return unique_ratio < threshold
    
    def validate_dataset(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Validate entire dataset."""
        results = [self.validate_example(ex) for ex in examples]
        
        valid_count = sum(1 for r in results if r["valid"])
        total_tokens = sum(r["token_count"] for r in results)
        
        issue_counts = {}
        for r in results:
            for issue in r["issues"]:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return {
            "total_examples": len(examples),
            "valid_examples": valid_count,
            "invalid_examples": len(examples) - valid_count,
            "valid_ratio": valid_count / len(examples),
            "total_tokens": total_tokens,
            "avg_tokens_per_example": total_tokens / len(examples),
            "issue_summary": issue_counts,
        }
```

### Compute Requirements

```python
@dataclass
class ComputeRequirements:
    """Estimate compute requirements for fine-tuning."""
    
    model_params_billions: float
    training_samples: int
    epochs: int
    batch_size: int
    sequence_length: int
    
    def estimate_vram_full_finetune(self) -> float:
        """Estimate VRAM for full fine-tuning (GB)."""
        # Model weights (fp16): params * 2 bytes
        model_memory = self.model_params_billions * 2
        
        # Gradients (fp16): params * 2 bytes
        gradient_memory = self.model_params_billions * 2
        
        # Optimizer states (Adam): params * 8 bytes
        optimizer_memory = self.model_params_billions * 8
        
        # Activations (rough estimate)
        activation_memory = (
            self.batch_size * 
            self.sequence_length * 
            self.model_params_billions * 0.1
        )
        
        total_gb = model_memory + gradient_memory + optimizer_memory + activation_memory
        return total_gb
    
    def estimate_vram_lora(self, lora_rank: int = 8) -> float:
        """Estimate VRAM for LoRA fine-tuning (GB)."""
        # Model weights (fp16 or 4-bit)
        model_memory = self.model_params_billions * 2
        
        # LoRA parameters (much smaller)
        # Roughly: 2 * rank * hidden_dim * num_layers
        lora_params_estimate = (
            2 * lora_rank * 4096 * 32 * 2  # Rough estimate
        ) / 1e9
        
        lora_memory = lora_params_estimate * 2  # fp16
        lora_gradients = lora_params_estimate * 2
        lora_optimizer = lora_params_estimate * 8
        
        # Activations still needed
        activation_memory = (
            self.batch_size * 
            self.sequence_length * 
            self.model_params_billions * 0.05
        )
        
        return model_memory + lora_memory + lora_gradients + lora_optimizer + activation_memory
    
    def estimate_vram_qlora(self, lora_rank: int = 8) -> float:
        """Estimate VRAM for QLoRA (4-bit quantized) fine-tuning (GB)."""
        # Model weights (4-bit): params * 0.5 bytes
        model_memory = self.model_params_billions * 0.5
        
        # LoRA in fp16
        lora_params_estimate = (2 * lora_rank * 4096 * 32 * 2) / 1e9
        lora_memory = lora_params_estimate * 2
        lora_gradients = lora_params_estimate * 2
        lora_optimizer = lora_params_estimate * 8
        
        # Reduced activations with gradient checkpointing
        activation_memory = (
            self.batch_size * 
            self.sequence_length * 
            self.model_params_billions * 0.02
        )
        
        return model_memory + lora_memory + lora_gradients + lora_optimizer + activation_memory
    
    def estimate_training_time_hours(self, tflops: float = 100) -> float:
        """Estimate training time in hours."""
        # FLOPs per token (rough): 6 * params
        flops_per_token = 6 * self.model_params_billions * 1e9
        
        # Total tokens
        total_tokens = self.training_samples * self.sequence_length * self.epochs
        
        # Total FLOPs
        total_flops = flops_per_token * total_tokens
        
        # Time in seconds
        time_seconds = total_flops / (tflops * 1e12)
        
        return time_seconds / 3600


# GPU recommendations
GPU_SPECS = {
    "A100_80GB": {"vram_gb": 80, "tflops_fp16": 312},
    "A100_40GB": {"vram_gb": 40, "tflops_fp16": 312},
    "H100_80GB": {"vram_gb": 80, "tflops_fp16": 989},
    "A10G": {"vram_gb": 24, "tflops_fp16": 125},
    "RTX_4090": {"vram_gb": 24, "tflops_fp16": 165},
    "RTX_3090": {"vram_gb": 24, "tflops_fp16": 71},
}

def recommend_hardware(
    model_params_billions: float,
    method: str = "qlora"
) -> List[str]:
    """Recommend hardware for fine-tuning."""
    
    req = ComputeRequirements(
        model_params_billions=model_params_billions,
        training_samples=1000,
        epochs=3,
        batch_size=4,
        sequence_length=2048,
    )
    
    if method == "full":
        vram_needed = req.estimate_vram_full_finetune()
    elif method == "lora":
        vram_needed = req.estimate_vram_lora()
    else:  # qlora
        vram_needed = req.estimate_vram_qlora()
    
    recommendations = []
    for gpu, specs in GPU_SPECS.items():
        if specs["vram_gb"] >= vram_needed:
            recommendations.append(f"{gpu} ({specs['vram_gb']}GB)")
    
    return recommendations
```

---

## Parameter-Efficient Fine-Tuning (PEFT)

### LoRA Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LoRA ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ORIGINAL TRANSFORMER LAYER                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  Input ──────────────────────────────────────────► Output   │ │
│  │    │                                                  ▲     │ │
│  │    │                                                  │     │ │
│  │    ▼                                                  │     │ │
│  │  ┌─────────────────────────────────────────────────┐  │     │ │
│  │  │           Pre-trained Weight Matrix (W)         │──┘     │ │
│  │  │                 d × k dimensions                │        │ │
│  │  │                    (FROZEN)                     │        │ │
│  │  └─────────────────────────────────────────────────┘        │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  WITH LoRA ADAPTERS                                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  Input ─────────────────────────────────────┬─────► Output  │ │
│  │    │                                        │         ▲     │ │
│  │    │                                        │         │     │ │
│  │    ├────────────────────────────────────────┤         │     │ │
│  │    │                                        │         │     │ │
│  │    ▼                                        ▼         │     │ │
│  │  ┌──────────────────────────┐   ┌────────────────┐    │     │ │
│  │  │   Pre-trained W (FROZEN) │   │  LoRA Adapter  │    │     │ │
│  │  │        d × k             │   │                │    │     │ │
│  │  └───────────┬──────────────┘   │  ┌─────────┐   │    │     │ │
│  │              │                  │  │ A (d×r) │───┤    │     │ │
│  │              │                  │  │TRAINABLE│   │    │     │ │
│  │              │                  │  └────┬────┘   │    │     │ │
│  │              │                  │       │        │    │     │ │
│  │              │                  │       ▼        │    │     │ │
│  │              │                  │  ┌─────────┐   │    │     │ │
│  │              │                  │  │ B (r×k) │───┤    │     │ │
│  │              │                  │  │TRAINABLE│   │    │     │ │
│  │              │                  │  └─────────┘   │    │     │ │
│  │              │                  │                │    │     │ │
│  │              │                  └───────┬────────┘    │     │ │
│  │              │                          │ × α/r      │     │ │
│  │              │                          │             │     │ │
│  │              └─────────────────►(+)◄────┘             │     │ │
│  │                                 │                     │     │ │
│  │                                 └─────────────────────┘     │ │
│  │                                                             │ │
│  │  W' = W + (α/r) × B × A                                     │ │
│  │  r = rank (typically 8-64), α = scaling factor              │ │
│  │  Trainable params: 2 × d × r (vs d × k for full)            │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### LoRA Implementation

```python
import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 8  # Rank
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.05
    target_modules: List[str] = None  # Which modules to adapt
    bias: str = "none"  # "none", "all", or "lora_only"
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default: attention projection matrices
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class LoRALayer(nn.Module):
    """LoRA adapter layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with Kaiming, B with zeros (so initial output is zero)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA contribution: x @ A^T @ B^T * scaling
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return lora_out * self.scaling


class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA adapter."""
    
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output + LoRA adaptation
        return self.linear(x) + self.lora(x)
    
    def merge_weights(self):
        """Merge LoRA weights into original layer."""
        with torch.no_grad():
            # W' = W + (alpha/r) * B @ A
            delta = (
                self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            )
            self.linear.weight.add_(delta)
        
        # Remove LoRA after merging
        self.lora = None
    
    def unmerge_weights(self):
        """Reverse merge operation."""
        with torch.no_grad():
            delta = (
                self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            )
            self.linear.weight.sub_(delta)


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Apply LoRA adapters to a model."""
    
    for name, module in model.named_modules():
        # Check if this module should have LoRA
        if any(target in name for target in config.target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA-enabled version
                parent = model
                parts = name.split('.')
                
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                
                lora_layer = LinearWithLoRA(
                    module,
                    rank=config.r,
                    alpha=config.lora_alpha,
                    dropout=config.lora_dropout,
                )
                
                setattr(parent, parts[-1], lora_layer)
    
    return model


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Get only LoRA parameters for training."""
    params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            params.append(param)
        else:
            param.requires_grad = False
    return params
```

### QLoRA Implementation

```python
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class QLoRATrainer:
    """QLoRA: Quantized LoRA for memory-efficient fine-tuning."""
    
    def __init__(
        self,
        model_name: str,
        lora_config: LoRAConfig,
        quantization_bits: int = 4,
    ):
        self.model_name = model_name
        self.lora_config = lora_config
        self.quant_bits = quantization_bits
    
    def load_model(self):
        """Load model with quantization and LoRA."""
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat4
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # Double quantization
        )
        
        # Load quantized model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)
        
        return model
    
    def print_trainable_parameters(self, model):
        """Print number of trainable parameters."""
        trainable_params = 0
        all_params = 0
        
        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {100 * trainable_params / all_params:.2f}%"
        )


# Training with QLoRA
from transformers import TrainingArguments, Trainer

def train_qlora(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
):
    """Train model with QLoRA."""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        gradient_checkpointing=True,
        report_to="tensorboard",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )
    
    trainer.train()
    
    return trainer
```

### Other PEFT Methods

```python
from dataclasses import dataclass
from typing import List
from enum import Enum

class PEFTMethod(Enum):
    LORA = "lora"
    QLORA = "qlora"
    PREFIX_TUNING = "prefix_tuning"
    PROMPT_TUNING = "prompt_tuning"
    ADAPTERS = "adapters"
    IA3 = "ia3"

@dataclass
class PrefixTuningConfig:
    """Config for prefix tuning."""
    num_virtual_tokens: int = 20
    prefix_projection: bool = True
    hidden_size: int = 512

@dataclass
class PromptTuningConfig:
    """Config for soft prompt tuning."""
    num_virtual_tokens: int = 8
    prompt_init_text: str = ""  # Optional initialization text


class PrefixTuning(nn.Module):
    """Prefix Tuning: Learn continuous prompt prefixes."""
    
    def __init__(
        self,
        config: PrefixTuningConfig,
        model_config,
    ):
        super().__init__()
        
        self.num_virtual_tokens = config.num_virtual_tokens
        self.num_layers = model_config.num_hidden_layers
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // config.num_attention_heads
        
        # Learnable prefix embeddings for each layer
        # Shape: (num_layers * 2, num_virtual_tokens, num_heads * head_dim)
        # *2 for key and value
        total_dim = self.num_layers * 2 * self.num_heads * self.head_dim
        
        if config.prefix_projection:
            # Use MLP to project to prefix
            self.prefix_encoder = nn.Sequential(
                nn.Embedding(config.num_virtual_tokens, config.hidden_size),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, total_dim),
            )
        else:
            # Direct embedding
            self.prefix_encoder = nn.Embedding(
                config.num_virtual_tokens, 
                total_dim,
            )
        
        self.prefix_tokens = torch.arange(config.num_virtual_tokens)
    
    def forward(self, batch_size: int):
        """Generate prefix key-value pairs for all layers."""
        
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        prefix_tokens = prefix_tokens.to(self.prefix_encoder[0].weight.device)
        
        # Shape: (batch_size, num_virtual_tokens, total_dim)
        prefix = self.prefix_encoder(prefix_tokens)
        
        # Reshape to (num_layers, 2, batch_size, num_heads, num_virtual_tokens, head_dim)
        prefix = prefix.view(
            batch_size,
            self.num_virtual_tokens,
            self.num_layers,
            2,
            self.num_heads,
            self.head_dim,
        )
        
        prefix = prefix.permute(2, 3, 0, 4, 1, 5)
        
        # Split into keys and values for each layer
        past_key_values = []
        for i in range(self.num_layers):
            key = prefix[i, 0]  # (batch_size, num_heads, num_virtual_tokens, head_dim)
            value = prefix[i, 1]
            past_key_values.append((key, value))
        
        return past_key_values


class PromptTuning(nn.Module):
    """Prompt Tuning: Learn soft prompt embeddings."""
    
    def __init__(
        self,
        config: PromptTuningConfig,
        tokenizer,
        model_embeddings: nn.Embedding,
    ):
        super().__init__()
        
        self.num_virtual_tokens = config.num_virtual_tokens
        embedding_dim = model_embeddings.embedding_dim
        
        # Initialize soft prompts
        if config.prompt_init_text:
            # Initialize from text
            init_ids = tokenizer.encode(
                config.prompt_init_text, 
                add_special_tokens=False
            )[:config.num_virtual_tokens]
            
            init_embeds = model_embeddings(torch.tensor(init_ids))
            
            # Pad if needed
            if len(init_ids) < config.num_virtual_tokens:
                pad_embeds = torch.randn(
                    config.num_virtual_tokens - len(init_ids),
                    embedding_dim,
                ) * 0.01
                init_embeds = torch.cat([init_embeds, pad_embeds])
            
            self.soft_prompt = nn.Parameter(init_embeds)
        else:
            # Random initialization
            self.soft_prompt = nn.Parameter(
                torch.randn(config.num_virtual_tokens, embedding_dim) * 0.01
            )
    
    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """Prepend soft prompt to input embeddings."""
        batch_size = input_embeds.shape[0]
        
        # Expand soft prompt for batch
        soft_prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Prepend to input
        return torch.cat([soft_prompt, input_embeds], dim=1)


class Adapter(nn.Module):
    """Bottleneck adapter for transformer layers."""
    
    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.up_project = nn.Linear(adapter_size, hidden_size)
        
        # Initialize for near-zero output
        nn.init.normal_(self.down_project.weight, std=0.01)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottleneck: down -> activate -> up
        down = self.down_project(x)
        activated = self.activation(down)
        up = self.up_project(activated)
        
        # Residual connection
        return x + up


# PEFT Method comparison
PEFT_COMPARISON = {
    "LoRA": {
        "trainable_params": "~0.1-1%",
        "memory_savings": "~10x",
        "best_for": "General adaptation, instruction following",
        "merge_possible": True,
    },
    "QLoRA": {
        "trainable_params": "~0.1-1%",
        "memory_savings": "~30x",
        "best_for": "Consumer hardware, large models",
        "merge_possible": True,
    },
    "Prefix Tuning": {
        "trainable_params": "~0.01%",
        "memory_savings": "~20x",
        "best_for": "Generation tasks, style transfer",
        "merge_possible": False,
    },
    "Prompt Tuning": {
        "trainable_params": "~0.001%",
        "memory_savings": "~100x",
        "best_for": "Simple classification, when data is limited",
        "merge_possible": False,
    },
    "Adapters": {
        "trainable_params": "~1-5%",
        "memory_savings": "~5x",
        "best_for": "Multi-task learning, task composition",
        "merge_possible": False,
    },
}
```

---

## Dataset Preparation

### Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                  DATASET PREPARATION PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │ Collect │───►│  Clean  │───►│ Format  │───►│ Quality │       │
│  │  Data   │    │  Data   │    │  Data   │    │ Filter  │       │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘       │
│       │              │              │              │              │
│       ▼              ▼              ▼              ▼              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │ Sources:│    │ Remove: │    │ Convert │    │ Verify: │       │
│  │ - Logs  │    │ - PII   │    │   to:   │    │ - Length│       │
│  │ - APIs  │    │ - Noise │    │ - Alpaca│    │ - Quality│      │
│  │ - Manual│    │ - Dupes │    │ - Chat  │    │ - Diverse│      │
│  │ - Synth │    │ - Errors│    │ - Custom│    │ - Balance│      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘       │
│                                                                  │
│                           │                                      │
│                           ▼                                      │
│                    ┌─────────────┐                               │
│                    │ Train/Val/  │                               │
│                    │ Test Split  │                               │
│                    │             │                               │
│                    │ 80/10/10    │                               │
│                    └─────────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Curation and Cleaning

```python
import re
import hashlib
from typing import List, Set, Dict, Any
from dataclasses import dataclass
from collections import Counter

@dataclass
class DataCleaningConfig:
    """Configuration for data cleaning."""
    min_input_length: int = 10
    max_input_length: int = 4096
    min_output_length: int = 10
    max_output_length: int = 4096
    remove_duplicates: bool = True
    dedupe_similarity_threshold: float = 0.9
    remove_pii: bool = True
    language_filter: str = None  # e.g., "en"


class DataCleaner:
    """Clean and preprocess training data."""
    
    def __init__(self, config: DataCleaningConfig):
        self.config = config
        self.seen_hashes: Set[str] = set()
    
    def clean_dataset(
        self, 
        examples: List[TrainingExample]
    ) -> List[TrainingExample]:
        """Clean entire dataset."""
        
        cleaned = []
        stats = {
            "total": len(examples),
            "passed": 0,
            "removed_length": 0,
            "removed_duplicate": 0,
            "removed_quality": 0,
        }
        
        for example in examples:
            # Apply cleaning pipeline
            result = self._clean_example(example)
            
            if result["valid"]:
                cleaned.append(result["example"])
                stats["passed"] += 1
            else:
                stats[f"removed_{result['reason']}"] = (
                    stats.get(f"removed_{result['reason']}", 0) + 1
                )
        
        print(f"Cleaning stats: {stats}")
        return cleaned
    
    def _clean_example(self, example: TrainingExample) -> Dict[str, Any]:
        """Clean single example."""
        
        # Clean text
        example.instruction = self._clean_text(example.instruction)
        example.output = self._clean_text(example.output)
        if example.input:
            example.input = self._clean_text(example.input)
        
        # Length checks
        input_len = len(example.instruction) + len(example.input)
        if input_len < self.config.min_input_length:
            return {"valid": False, "reason": "length"}
        if input_len > self.config.max_input_length:
            return {"valid": False, "reason": "length"}
        
        output_len = len(example.output)
        if output_len < self.config.min_output_length:
            return {"valid": False, "reason": "length"}
        if output_len > self.config.max_output_length:
            return {"valid": False, "reason": "length"}
        
        # Duplicate check
        if self.config.remove_duplicates:
            content_hash = self._hash_content(example)
            if content_hash in self.seen_hashes:
                return {"valid": False, "reason": "duplicate"}
            self.seen_hashes.add(content_hash)
        
        # PII removal
        if self.config.remove_pii:
            example = self._remove_pii(example)
        
        # Quality check
        if not self._quality_check(example):
            return {"valid": False, "reason": "quality"}
        
        return {"valid": True, "example": example}
    
    def _clean_text(self, text: str) -> str:
        """Clean individual text field."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize unicode
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        return text.strip()
    
    def _hash_content(self, example: TrainingExample) -> str:
        """Create hash for deduplication."""
        content = f"{example.instruction}|{example.input}|{example.output}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _remove_pii(self, example: TrainingExample) -> TrainingExample:
        """Remove PII from example."""
        patterns = {
            "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL]"),
            "phone": (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "[PHONE]"),
            "ssn": (r'\b\d{3}-\d{2}-\d{4}\b', "[SSN]"),
        }
        
        for field in ['instruction', 'input', 'output']:
            text = getattr(example, field)
            for name, (pattern, replacement) in patterns.items():
                text = re.sub(pattern, replacement, text)
            setattr(example, field, text)
        
        return example
    
    def _quality_check(self, example: TrainingExample) -> bool:
        """Check example quality."""
        
        # Check for repeated content
        words = example.output.lower().split()
        if len(words) > 10:
            word_counts = Counter(words)
            most_common_ratio = word_counts.most_common(1)[0][1] / len(words)
            if most_common_ratio > 0.3:
                return False
        
        # Check for incomplete sentences
        if not example.output.strip()[-1] in '.!?"\')':
            if len(example.output) > 100:  # Only for longer outputs
                return False
        
        return True


class DataAugmenter:
    """Augment training data."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def augment_with_paraphrases(
        self, 
        examples: List[TrainingExample],
        num_paraphrases: int = 2,
    ) -> List[TrainingExample]:
        """Generate paraphrased versions of examples."""
        
        augmented = []
        
        for example in examples:
            augmented.append(example)  # Keep original
            
            # Generate paraphrases
            paraphrases = await self._generate_paraphrases(
                example.instruction,
                num_paraphrases,
            )
            
            for i, para in enumerate(paraphrases):
                augmented.append(TrainingExample(
                    instruction=para,
                    input=example.input,
                    output=example.output,
                    metadata={**(example.metadata or {}), "augmented": True},
                ))
        
        return augmented
    
    async def _generate_paraphrases(
        self, 
        text: str, 
        num: int
    ) -> List[str]:
        """Generate paraphrases using LLM."""
        
        response = await self.llm.generate(
            prompt=f"""Generate {num} different paraphrases of this instruction.
Keep the same meaning but vary the wording.

Original: {text}

Paraphrases (one per line):""",
        )
        
        return [line.strip() for line in response.split('\n') if line.strip()][:num]


class SyntheticDataGenerator:
    """Generate synthetic training data."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def generate_from_seed(
        self,
        seed_examples: List[TrainingExample],
        num_to_generate: int,
        task_description: str,
    ) -> List[TrainingExample]:
        """Generate new examples based on seed examples."""
        
        # Format seed examples
        seed_text = "\n\n".join([
            f"Instruction: {ex.instruction}\n"
            f"Input: {ex.input}\n"
            f"Output: {ex.output}"
            for ex in seed_examples[:5]
        ])
        
        generated = []
        batch_size = 10
        
        for i in range(0, num_to_generate, batch_size):
            response = await self.llm.generate(
                system=f"You are generating training data for: {task_description}",
                prompt=f"""Based on these examples:

{seed_text}

Generate {min(batch_size, num_to_generate - i)} new, diverse examples.
Each should follow the same format but cover different scenarios.

Format each as:
---
Instruction: <instruction>
Input: <input or empty>
Output: <expected output>
---""",
            )
            
            # Parse generated examples
            parsed = self._parse_generated(response)
            generated.extend(parsed)
        
        return generated[:num_to_generate]
    
    def _parse_generated(self, response: str) -> List[TrainingExample]:
        """Parse generated examples from LLM response."""
        examples = []
        
        # Split by separator
        blocks = response.split('---')
        
        for block in blocks:
            if not block.strip():
                continue
            
            # Extract fields
            instruction = ""
            input_text = ""
            output = ""
            
            lines = block.strip().split('\n')
            current_field = None
            
            for line in lines:
                if line.startswith('Instruction:'):
                    instruction = line[12:].strip()
                    current_field = 'instruction'
                elif line.startswith('Input:'):
                    input_text = line[6:].strip()
                    current_field = 'input'
                elif line.startswith('Output:'):
                    output = line[7:].strip()
                    current_field = 'output'
                elif current_field == 'output':
                    output += '\n' + line
            
            if instruction and output:
                examples.append(TrainingExample(
                    instruction=instruction,
                    input=input_text,
                    output=output.strip(),
                    metadata={"synthetic": True},
                ))
        
        return examples
```

### Format Conversion

```python
import json
from typing import List, Dict, Any

class DatasetFormatter:
    """Convert data to various fine-tuning formats."""
    
    @staticmethod
    def to_alpaca(examples: List[TrainingExample]) -> List[Dict[str, str]]:
        """Convert to Alpaca format."""
        return [ex.to_alpaca_format() for ex in examples]
    
    @staticmethod
    def to_sharegpt(examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """Convert to ShareGPT format."""
        return [ex.to_sharegpt_format() for ex in examples]
    
    @staticmethod
    def to_openai_chat(examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """Convert to OpenAI chat fine-tuning format."""
        return [{"messages": ex.to_chat_format()} for ex in examples]
    
    @staticmethod
    def to_huggingface(
        examples: List[TrainingExample],
        tokenizer,
        max_length: int = 2048,
    ) -> Dict[str, List]:
        """Convert to HuggingFace datasets format."""
        
        input_ids = []
        attention_masks = []
        labels = []
        
        for ex in examples:
            # Format as instruction-following
            text = f"### Instruction:\n{ex.instruction}"
            if ex.input:
                text += f"\n\n### Input:\n{ex.input}"
            text += f"\n\n### Response:\n{ex.output}"
            
            # Tokenize
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            input_ids.append(encoded["input_ids"].squeeze())
            attention_masks.append(encoded["attention_mask"].squeeze())
            
            # Labels: -100 for input tokens, actual ids for output
            label = encoded["input_ids"].clone().squeeze()
            # Find where response starts
            response_start = text.find("### Response:")
            response_tokens = len(tokenizer.encode(text[:response_start]))
            label[:response_tokens] = -100
            labels.append(label)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }
    
    @staticmethod
    def create_train_val_test_split(
        examples: List[TrainingExample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, List[TrainingExample]]:
        """Split dataset into train/val/test."""
        
        import random
        random.seed(seed)
        
        # Shuffle
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        # Calculate split points
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        return {
            "train": shuffled[:train_end],
            "validation": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }
    
    @staticmethod
    def save_dataset(
        examples: List[TrainingExample],
        path: str,
        format: str = "jsonl",
    ):
        """Save dataset to file."""
        
        if format == "jsonl":
            with open(path, 'w') as f:
                for ex in examples:
                    f.write(json.dumps(ex.to_alpaca_format()) + '\n')
        
        elif format == "json":
            with open(path, 'w') as f:
                json.dump([ex.to_alpaca_format() for ex in examples], f, indent=2)
        
        elif format == "parquet":
            import pandas as pd
            df = pd.DataFrame([ex.to_alpaca_format() for ex in examples])
            df.to_parquet(path)
```

---

## Training Pipeline

### Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                   FINE-TUNING TRAINING PIPELINE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                      PREPARATION                            │ │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │ │
│  │  │  Load   │───►│  Load   │───►│ Config  │───►│  Setup  │  │ │
│  │  │  Data   │    │  Model  │    │ PEFT    │    │Optimizer│  │ │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    TRAINING LOOP                            │ │
│  │                                                             │ │
│  │  for epoch in epochs:                                       │ │
│  │    for batch in dataloader:                                 │ │
│  │      ┌────────────┐    ┌────────────┐    ┌────────────┐    │ │
│  │      │  Forward   │───►│  Compute   │───►│  Backward  │    │ │
│  │      │   Pass     │    │   Loss     │    │   Pass     │    │ │
│  │      └────────────┘    └────────────┘    └────────────┘    │ │
│  │             │                                    │          │ │
│  │             │          ┌────────────┐            │          │ │
│  │             │          │  Gradient  │◄───────────┘          │ │
│  │             │          │ Accumulate │                       │ │
│  │             │          └─────┬──────┘                       │ │
│  │             │                │                              │ │
│  │             │          ┌─────▼──────┐                       │ │
│  │             │          │  Optimizer │                       │ │
│  │             │          │   Step     │                       │ │
│  │             │          └─────┬──────┘                       │ │
│  │             │                │                              │ │
│  │             │          ┌─────▼──────┐                       │ │
│  │             └─────────►│    Log     │                       │ │
│  │                        │  Metrics   │                       │ │
│  │                        └────────────┘                       │ │
│  │                                                             │ │
│  │    Evaluate on validation set                               │ │
│  │    Save checkpoint if improved                              │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                     POST-TRAINING                           │ │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │ │
│  │  │  Merge  │───►│  Final  │───►│  Export │───►│  Deploy │  │ │
│  │  │ Weights │    │  Eval   │    │  Model  │    │         │  │ │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Training Configuration

```python
from dataclasses import dataclass, field
from typing import Optional, List
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
)

@dataclass
class FineTuningConfig:
    """Complete fine-tuning configuration."""
    
    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"
    use_flash_attention: bool = True
    
    # LoRA/PEFT
    peft_method: str = "lora"  # lora, qlora, full
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # Training
    num_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    
    # Optimization
    optimizer: str = "adamw"  # adamw, paged_adamw_8bit
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Precision
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    
    # Memory optimization
    gradient_checkpointing: bool = True
    
    # Evaluation
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


class FineTuningTrainer:
    """Complete fine-tuning trainer."""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def setup(self):
        """Setup model, tokenizer, and training components."""
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model based on PEFT method
        if self.config.peft_method == "qlora":
            self.model = self._load_qlora_model()
        elif self.config.peft_method == "lora":
            self.model = self._load_lora_model()
        else:
            self.model = self._load_full_model()
        
        # Enable optimizations
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        if self.config.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
    
    def _load_qlora_model(self):
        """Load model with QLoRA."""
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.config.use_flash_attention else None,
        )
        
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        return get_peft_model(model, peft_config)
    
    def _load_lora_model(self):
        """Load model with LoRA (fp16)."""
        from peft import LoraConfig, get_peft_model
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.config.use_flash_attention else None,
        )
        
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        return get_peft_model(model, peft_config)
    
    def _load_full_model(self):
        """Load model for full fine-tuning."""
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.config.use_flash_attention else None,
        )
    
    def create_training_args(self, output_dir: str) -> TrainingArguments:
        """Create HuggingFace TrainingArguments."""
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=self.config.gradient_checkpointing,
            optim=("paged_adamw_8bit" 
                   if self.config.peft_method == "qlora" 
                   else "adamw_torch"),
            report_to=["tensorboard", "wandb"],
            push_to_hub=False,
        )
    
    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir: str,
    ):
        """Run training."""
        
        training_args = self.create_training_args(output_dir)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            ),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold,
                ),
            ],
        )
        
        # Train
        train_result = self.trainer.train()
        
        # Save
        self.trainer.save_model()
        
        return train_result


class LearningRateSchedules:
    """Learning rate schedule utilities."""
    
    @staticmethod
    def get_schedule_info() -> Dict[str, str]:
        return {
            "cosine": "Cosine annealing - smooth decay, good default",
            "linear": "Linear decay - simple and predictable",
            "constant": "No decay - use with warmup",
            "constant_with_warmup": "Constant after warmup",
            "cosine_with_restarts": "Cosine with periodic restarts",
            "polynomial": "Polynomial decay - flexible",
        }
    
    @staticmethod
    def recommend_lr(
        model_params_billions: float,
        peft_method: str,
        dataset_size: int,
    ) -> float:
        """Recommend learning rate based on setup."""
        
        # Base LR depends on method
        if peft_method == "full":
            base_lr = 1e-5
        elif peft_method in ["lora", "qlora"]:
            base_lr = 2e-4
        else:
            base_lr = 1e-4
        
        # Adjust for model size (larger = lower LR)
        if model_params_billions > 30:
            base_lr *= 0.5
        elif model_params_billions > 10:
            base_lr *= 0.7
        
        # Adjust for dataset size (larger = can use higher LR)
        if dataset_size > 100000:
            base_lr *= 1.2
        elif dataset_size < 1000:
            base_lr *= 0.8
        
        return base_lr
```

### Preventing Catastrophic Forgetting

```python
class CatastrophicForgettingMitigation:
    """Strategies to prevent catastrophic forgetting."""
    
    @staticmethod
    def strategies() -> Dict[str, str]:
        return {
            "low_learning_rate": "Use 10-100x smaller LR than pre-training",
            "regularization": "L2 regularization to stay close to original",
            "ewc": "Elastic Weight Consolidation - protect important weights",
            "replay": "Mix in original training data distribution",
            "lora": "Only train small adapters, keep base frozen",
            "gradual_unfreezing": "Unfreeze layers progressively",
        }
    
    @staticmethod
    def create_replay_dataset(
        task_data: List[TrainingExample],
        general_data: List[TrainingExample],
        replay_ratio: float = 0.1,
    ) -> List[TrainingExample]:
        """Create dataset with replay samples."""
        
        num_replay = int(len(task_data) * replay_ratio)
        
        import random
        replay_samples = random.sample(general_data, min(num_replay, len(general_data)))
        
        combined = task_data + replay_samples
        random.shuffle(combined)
        
        return combined


class ElasticWeightConsolidation(nn.Module):
    """EWC loss for preventing forgetting."""
    
    def __init__(self, model, dataloader, importance_weight: float = 1000):
        super().__init__()
        
        self.importance_weight = importance_weight
        
        # Store reference parameters
        self.reference_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.reference_params[name] = param.data.clone()
        
        # Compute Fisher information (importance weights)
        self.fisher = self._compute_fisher(model, dataloader)
    
    def _compute_fisher(self, model, dataloader) -> Dict[str, torch.Tensor]:
        """Compute diagonal Fisher information matrix."""
        
        fisher = {name: torch.zeros_like(param) 
                  for name, param in model.named_parameters() 
                  if param.requires_grad}
        
        model.eval()
        for batch in dataloader:
            model.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Normalize
        for name in fisher:
            fisher[name] /= len(dataloader)
        
        return fisher
    
    def penalty(self, model) -> torch.Tensor:
        """Compute EWC penalty."""
        
        penalty = 0
        for name, param in model.named_parameters():
            if name in self.reference_params:
                diff = param - self.reference_params[name]
                penalty += (self.fisher[name] * diff ** 2).sum()
        
        return self.importance_weight * penalty
```

---

## Evaluation and Validation

### Evaluation Metrics

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
import numpy as np
import torch

@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    perplexity: float
    accuracy: float = None
    f1_score: float = None
    bleu_score: float = None
    rouge_scores: Dict[str, float] = None
    custom_metrics: Dict[str, float] = None
    latency_ms: float = None
    tokens_per_second: float = None


class FineTuningEvaluator:
    """Comprehensive evaluation for fine-tuned models."""
    
    def __init__(self, tokenizer, device: str = "cuda"):
        self.tokenizer = tokenizer
        self.device = device
    
    def compute_perplexity(
        self,
        model,
        eval_dataset,
        batch_size: int = 8,
    ) -> float:
        """Compute perplexity on evaluation set."""
        
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**inputs)
                
                # Mask padding tokens in loss calculation
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = inputs["labels"][..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                
                num_tokens = (shift_labels != -100).sum().item()
                total_loss += loss.item()
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def evaluate_generation(
        self,
        model,
        test_cases: List[Dict[str, str]],
        max_new_tokens: int = 256,
    ) -> List[Dict[str, Any]]:
        """Evaluate generation quality."""
        
        model.eval()
        results = []
        
        for case in test_cases:
            prompt = case["prompt"]
            expected = case.get("expected")
            
            # Generate
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt"
            ).to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            latency = time.time() - start_time
            
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            
            result = {
                "prompt": prompt,
                "generated": generated,
                "latency_ms": latency * 1000,
                "tokens_generated": len(outputs[0]) - inputs["input_ids"].shape[1],
            }
            
            if expected:
                result["expected"] = expected
                result["exact_match"] = generated.strip() == expected.strip()
            
            results.append(result)
        
        return results
    
    def compare_models(
        self,
        baseline_model,
        finetuned_model,
        eval_dataset,
        test_cases: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Compare baseline and fine-tuned models."""
        
        baseline_ppl = self.compute_perplexity(baseline_model, eval_dataset)
        finetuned_ppl = self.compute_perplexity(finetuned_model, eval_dataset)
        
        baseline_gen = self.evaluate_generation(baseline_model, test_cases)
        finetuned_gen = self.evaluate_generation(finetuned_model, test_cases)
        
        return {
            "perplexity": {
                "baseline": baseline_ppl,
                "finetuned": finetuned_ppl,
                "improvement": (baseline_ppl - finetuned_ppl) / baseline_ppl,
            },
            "generation": {
                "baseline_avg_latency": np.mean([r["latency_ms"] for r in baseline_gen]),
                "finetuned_avg_latency": np.mean([r["latency_ms"] for r in finetuned_gen]),
            },
            "detailed_results": {
                "baseline": baseline_gen,
                "finetuned": finetuned_gen,
            },
        }


class HumanEvaluation:
    """Framework for human evaluation of fine-tuned models."""
    
    @staticmethod
    def create_eval_interface(
        test_cases: List[Dict[str, str]],
        model_outputs: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """Create evaluation interface for human raters."""
        
        eval_items = []
        
        for i, case in enumerate(test_cases):
            item = {
                "id": i,
                "prompt": case["prompt"],
                "reference": case.get("expected"),
                "outputs": {},
            }
            
            # Randomize order to prevent bias
            model_names = list(model_outputs.keys())
            random.shuffle(model_names)
            
            for j, model_name in enumerate(model_names):
                item["outputs"][f"output_{j}"] = {
                    "text": model_outputs[model_name][i],
                    "_model": model_name,  # Hidden from raters
                }
            
            item["criteria"] = [
                {
                    "name": "relevance",
                    "description": "How relevant is the output to the prompt?",
                    "scale": "1-5",
                },
                {
                    "name": "coherence",
                    "description": "How coherent and well-structured is the output?",
                    "scale": "1-5",
                },
                {
                    "name": "correctness",
                    "description": "How factually correct is the output?",
                    "scale": "1-5",
                },
            ]
            
            eval_items.append(item)
        
        return eval_items
    
    @staticmethod
    def compute_inter_rater_reliability(
        ratings: List[Dict[str, Any]],
    ) -> float:
        """Compute Krippendorff's alpha for inter-rater reliability."""
        # Implementation of Krippendorff's alpha
        pass


class ABTestingFramework:
    """A/B testing for fine-tuned model deployment."""
    
    def __init__(self, model_a, model_b, traffic_split: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.results = {"a": [], "b": []}
    
    def route_request(self, request_id: str) -> str:
        """Route request to model A or B."""
        # Deterministic routing based on request_id
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        if (hash_val % 100) / 100 < self.traffic_split:
            return "a"
        return "b"
    
    async def process_request(
        self,
        request_id: str,
        prompt: str,
    ) -> Dict[str, Any]:
        """Process request with A/B testing."""
        
        variant = self.route_request(request_id)
        model = self.model_a if variant == "a" else self.model_b
        
        start_time = time.time()
        output = await model.generate(prompt)
        latency = time.time() - start_time
        
        result = {
            "request_id": request_id,
            "variant": variant,
            "output": output,
            "latency": latency,
        }
        
        self.results[variant].append(result)
        
        return result
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute A/B test statistics."""
        
        stats = {}
        
        for variant in ["a", "b"]:
            results = self.results[variant]
            if results:
                latencies = [r["latency"] for r in results]
                stats[variant] = {
                    "count": len(results),
                    "avg_latency": np.mean(latencies),
                    "p50_latency": np.percentile(latencies, 50),
                    "p95_latency": np.percentile(latencies, 95),
                }
        
        # Statistical significance test
        if len(self.results["a"]) > 30 and len(self.results["b"]) > 30:
            from scipy import stats as scipy_stats
            
            a_latencies = [r["latency"] for r in self.results["a"]]
            b_latencies = [r["latency"] for r in self.results["b"]]
            
            t_stat, p_value = scipy_stats.ttest_ind(a_latencies, b_latencies)
            stats["significance"] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }
        
        return stats
```

---

## Production Deployment

### Model Deployment Options

```
┌─────────────────────────────────────────────────────────────────┐
│                 MODEL DEPLOYMENT OPTIONS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OPTION 1: MERGED WEIGHTS                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Base Model + LoRA ──► Merged Model ──► Single Deployment   │ │
│  │                                                             │ │
│  │  Pros: Simpler serving, no adapter overhead                 │ │
│  │  Cons: Larger model, can't easily switch adapters           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  OPTION 2: DYNAMIC ADAPTER LOADING                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  ┌─────────────┐                                            │ │
│  │  │ Base Model  │◄──┬── Adapter A (Task 1)                   │ │
│  │  │  (Shared)   │   ├── Adapter B (Task 2)                   │ │
│  │  │             │   └── Adapter C (Task 3)                   │ │
│  │  └─────────────┘                                            │ │
│  │                                                             │ │
│  │  Pros: Memory efficient, multi-task support                 │ │
│  │  Cons: Slight latency overhead, complexity                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  OPTION 3: MODEL VERSIONING                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                  │ │
│  │  │  v1.0   │    │  v1.1   │    │  v2.0   │                  │ │
│  │  │(Canary) │───►│(Staging)│───►│(Prod)   │                  │ │
│  │  │   5%    │    │   10%   │    │   85%   │                  │ │
│  │  └─────────┘    └─────────┘    └─────────┘                  │ │
│  │                                                             │ │
│  │  Progressive rollout with monitoring                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Merging and Export

```python
from peft import PeftModel
from typing import Optional
import os
import shutil

class ModelMerger:
    """Merge LoRA weights into base model."""
    
    def __init__(self, base_model_name: str, adapter_path: str):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
    
    def merge_and_save(
        self,
        output_path: str,
        push_to_hub: bool = False,
        hub_repo: str = None,
    ):
        """Merge adapter weights and save full model."""
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, self.adapter_path)
        
        # Merge weights
        merged_model = model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_path)
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        tokenizer.save_pretrained(output_path)
        
        if push_to_hub and hub_repo:
            merged_model.push_to_hub(hub_repo)
            tokenizer.push_to_hub(hub_repo)
        
        print(f"Merged model saved to {output_path}")
    
    def export_to_gguf(
        self,
        merged_model_path: str,
        output_path: str,
        quantization: str = "q4_k_m",
    ):
        """Export to GGUF format for llama.cpp."""
        
        # This requires llama.cpp conversion script
        import subprocess
        
        subprocess.run([
            "python", "llama.cpp/convert.py",
            merged_model_path,
            "--outtype", "f16",
            "--outfile", f"{output_path}/model-f16.gguf",
        ])
        
        # Quantize
        subprocess.run([
            "llama.cpp/quantize",
            f"{output_path}/model-f16.gguf",
            f"{output_path}/model-{quantization}.gguf",
            quantization,
        ])


class DynamicAdapterServer:
    """Serve model with dynamic adapter loading."""
    
    def __init__(self, base_model_name: str):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.loaded_adapters = {}
    
    def load_adapter(self, adapter_name: str, adapter_path: str):
        """Load an adapter."""
        if adapter_name not in self.loaded_adapters:
            # Load adapter weights
            from peft import PeftModel
            model_with_adapter = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                adapter_name=adapter_name,
            )
            self.loaded_adapters[adapter_name] = model_with_adapter
    
    def generate(
        self,
        prompt: str,
        adapter_name: str = None,
        **kwargs,
    ) -> str:
        """Generate with specified adapter."""
        
        if adapter_name and adapter_name in self.loaded_adapters:
            model = self.loaded_adapters[adapter_name]
        else:
            model = self.base_model
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            do_sample=kwargs.get("do_sample", False),
            temperature=kwargs.get("temperature", 1.0),
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class ModelVersionManager:
    """Manage model versions for deployment."""
    
    def __init__(self, model_registry_path: str):
        self.registry_path = model_registry_path
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load version metadata."""
        versions_file = os.path.join(self.registry_path, "versions.json")
        if os.path.exists(versions_file):
            with open(versions_file) as f:
                return json.load(f)
        return {"versions": [], "current": None}
    
    def register_version(
        self,
        version: str,
        model_path: str,
        metrics: Dict[str, float],
        metadata: Dict[str, Any] = None,
    ):
        """Register a new model version."""
        
        version_info = {
            "version": version,
            "model_path": model_path,
            "metrics": metrics,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "status": "registered",
        }
        
        self.versions["versions"].append(version_info)
        self._save_versions()
    
    def promote_version(self, version: str, stage: str):
        """Promote version to a deployment stage."""
        
        for v in self.versions["versions"]:
            if v["version"] == version:
                v["status"] = stage
                if stage == "production":
                    self.versions["current"] = version
                self._save_versions()
                return
        
        raise ValueError(f"Version {version} not found")
    
    def rollback(self, target_version: str):
        """Rollback to a previous version."""
        
        # Find version
        for v in self.versions["versions"]:
            if v["version"] == target_version:
                self.versions["current"] = target_version
                self._save_versions()
                return
        
        raise ValueError(f"Version {target_version} not found")
    
    def get_current_model_path(self) -> str:
        """Get path to current production model."""
        
        current = self.versions.get("current")
        if not current:
            raise ValueError("No current version set")
        
        for v in self.versions["versions"]:
            if v["version"] == current:
                return v["model_path"]
        
        raise ValueError(f"Current version {current} not found")
    
    def _save_versions(self):
        """Save version metadata."""
        versions_file = os.path.join(self.registry_path, "versions.json")
        with open(versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
```

---

## Advanced Patterns

### Continued Pre-Training

```python
class ContinuedPreTrainer:
    """Continue pre-training on domain-specific data."""
    
    def __init__(self, model_name: str, domain_corpus_path: str):
        self.model_name = model_name
        self.corpus_path = domain_corpus_path
    
    def prepare_corpus(self, chunk_size: int = 2048) -> List[str]:
        """Prepare corpus for continued pre-training."""
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        with open(self.corpus_path) as f:
            for line in f:
                tokens = len(line.split())  # Rough estimate
                
                if current_length + tokens > chunk_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(line.strip())
                current_length += tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def train(
        self,
        chunks: List[str],
        output_dir: str,
        num_epochs: int = 1,
        learning_rate: float = 1e-5,
    ):
        """Run continued pre-training."""
        
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Lower learning rate than fine-tuning
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            save_strategy="epoch",
            bf16=True,
        )
        
        # Create dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048,
            )
        
        dataset = Dataset.from_dict({"text": chunks})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            ),
        )
        
        trainer.train()
        return trainer


class MultiTaskFineTuner:
    """Fine-tune on multiple tasks simultaneously."""
    
    def __init__(self, model_name: str, tasks: Dict[str, List[TrainingExample]]):
        self.model_name = model_name
        self.tasks = tasks
    
    def create_mixed_dataset(
        self,
        sampling_strategy: str = "proportional",
    ) -> List[TrainingExample]:
        """Create mixed dataset from multiple tasks."""
        
        if sampling_strategy == "proportional":
            # Sample proportional to task size
            all_examples = []
            for task_name, examples in self.tasks.items():
                for ex in examples:
                    ex.metadata = ex.metadata or {}
                    ex.metadata["task"] = task_name
                    all_examples.append(ex)
            
            random.shuffle(all_examples)
            return all_examples
        
        elif sampling_strategy == "equal":
            # Equal samples from each task
            min_size = min(len(ex) for ex in self.tasks.values())
            all_examples = []
            
            for task_name, examples in self.tasks.items():
                sampled = random.sample(examples, min_size)
                for ex in sampled:
                    ex.metadata = ex.metadata or {}
                    ex.metadata["task"] = task_name
                    all_examples.append(ex)
            
            random.shuffle(all_examples)
            return all_examples
        
        elif sampling_strategy == "temperature":
            # Temperature-based sampling
            temperature = 2.0
            sizes = [len(ex) for ex in self.tasks.values()]
            probs = np.array(sizes) ** (1.0 / temperature)
            probs = probs / probs.sum()
            
            all_examples = []
            target_size = sum(sizes)
            
            for _ in range(target_size):
                task_idx = np.random.choice(len(self.tasks), p=probs)
                task_name = list(self.tasks.keys())[task_idx]
                ex = random.choice(self.tasks[task_name])
                ex.metadata = ex.metadata or {}
                ex.metadata["task"] = task_name
                all_examples.append(ex)
            
            return all_examples
        
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")


class DPOTrainer:
    """Direct Preference Optimization training."""
    
    def __init__(self, model_name: str, reference_model_name: str = None):
        self.model_name = model_name
        self.reference_model_name = reference_model_name or model_name
    
    def prepare_preference_data(
        self,
        comparisons: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Prepare preference pairs for DPO training.
        
        Each comparison should have:
        - prompt: The input prompt
        - chosen: The preferred response
        - rejected: The non-preferred response
        """
        
        formatted = []
        for comp in comparisons:
            formatted.append({
                "prompt": comp["prompt"],
                "chosen": comp["chosen"],
                "rejected": comp["rejected"],
            })
        return formatted
    
    def train(
        self,
        preference_data: List[Dict[str, str]],
        output_dir: str,
        beta: float = 0.1,  # DPO temperature parameter
        num_epochs: int = 1,
        learning_rate: float = 1e-6,
    ):
        """Train with DPO."""
        
        from trl import DPOTrainer as TRLDPOTrainer, DPOConfig
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
        )
        
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.reference_model_name,
            torch_dtype=torch.bfloat16,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        dpo_config = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            beta=beta,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
        )
        
        # Create dataset
        dataset = Dataset.from_list(preference_data)
        
        trainer = TRLDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        
        trainer.train()
        return trainer


class RLHFTrainer:
    """Reinforcement Learning from Human Feedback (simplified)."""
    
    def __init__(
        self,
        model_name: str,
        reward_model_name: str,
    ):
        self.model_name = model_name
        self.reward_model_name = reward_model_name
    
    def train_reward_model(
        self,
        comparison_data: List[Dict[str, Any]],
        output_dir: str,
    ):
        """Train reward model on human comparisons."""
        
        # Reward model predicts which response is preferred
        # This is typically a classification head on top of LLM
        pass
    
    def train_with_ppo(
        self,
        prompts: List[str],
        output_dir: str,
        num_epochs: int = 1,
    ):
        """Train policy with PPO using reward model."""
        
        from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
        
        ppo_config = PPOConfig(
            output_dir=output_dir,
            learning_rate=1e-5,
            batch_size=16,
            mini_batch_size=4,
            gradient_accumulation_steps=4,
            ppo_epochs=4,
            target_kl=0.1,
        )
        
        # Load policy model with value head
        policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_name,
        )
        
        # Load reward model
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_name,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        ppo_trainer = PPOTrainer(
            model=policy_model,
            ref_model=None,  # Will create copy
            config=ppo_config,
            tokenizer=tokenizer,
        )
        
        # Training loop
        for epoch in range(num_epochs):
            for prompt_batch in batch_prompts(prompts, 16):
                # Generate responses
                response_tensors = []
                for prompt in prompt_batch:
                    inputs = tokenizer(prompt, return_tensors="pt")
                    response = policy_model.generate(**inputs, max_new_tokens=128)
                    response_tensors.append(response)
                
                # Get rewards from reward model
                rewards = []
                for prompt, response in zip(prompt_batch, response_tensors):
                    response_text = tokenizer.decode(response[0])
                    reward = self._get_reward(prompt, response_text, reward_model)
                    rewards.append(reward)
                
                # PPO step
                stats = ppo_trainer.step(
                    [tokenizer(p, return_tensors="pt").input_ids.squeeze() 
                     for p in prompt_batch],
                    response_tensors,
                    rewards,
                )
        
        return ppo_trainer
    
    def _get_reward(
        self,
        prompt: str,
        response: str,
        reward_model,
    ) -> torch.Tensor:
        """Get reward score for a response."""
        
        inputs = self.tokenizer(
            prompt + response,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            outputs = reward_model(**inputs)
            reward = outputs.logits.squeeze()
        
        return reward
```

---

## Best Practices Checklist

### Before Fine-Tuning

- [ ] Evaluate if prompting or RAG can solve the problem
- [ ] Collect and validate sufficient training data (1000+ examples)
- [ ] Clean and deduplicate training data
- [ ] Remove PII and sensitive information
- [ ] Create proper train/validation/test splits
- [ ] Establish baseline metrics with the base model
- [ ] Estimate compute requirements and budget

### During Fine-Tuning

- [ ] Start with PEFT (LoRA/QLoRA) before full fine-tuning
- [ ] Use appropriate learning rate (2e-4 for LoRA, 1e-5 for full)
- [ ] Enable gradient checkpointing for memory efficiency
- [ ] Monitor training loss and validation metrics
- [ ] Watch for overfitting (validation loss increasing)
- [ ] Save checkpoints regularly
- [ ] Use early stopping to prevent overfitting

### After Fine-Tuning

- [ ] Evaluate on held-out test set
- [ ] Compare against baseline model
- [ ] Test for regression on general capabilities
- [ ] Conduct human evaluation for subjective tasks
- [ ] Document training configuration and results
- [ ] Merge LoRA weights if deploying merged model
- [ ] Version model artifacts properly

### Production Deployment

- [ ] Set up model versioning and registry
- [ ] Implement gradual rollout (canary deployment)
- [ ] Configure monitoring and alerting
- [ ] Plan rollback procedure
- [ ] A/B test against production baseline
- [ ] Monitor for distribution drift
- [ ] Set up regular re-evaluation pipeline

---

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.15647)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - InstructGPT
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - DPO
- [The Llama 2 Technical Report](https://arxiv.org/abs/2307.09288)
- [PEFT Library Documentation](https://huggingface.co/docs/peft)
- [TRL: Transformer Reinforcement Learning](https://huggingface.co/docs/trl)
- [Axolotl Fine-tuning Framework](https://github.com/OpenAccess-AI-Collective/axolotl)
- [LLM Fine-tuning Best Practices - OpenAI](https://platform.openai.com/docs/guides/fine-tuning)
