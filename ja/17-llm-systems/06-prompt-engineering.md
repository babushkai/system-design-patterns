# プロンプトエンジニアリングパターン

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../17-llm-systems/06-prompt-engineering.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

プロンプトエンジニアリングとは、LLMへの入力を設計・最適化し、望ましい出力を信頼性高く効率的に得るための手法です。コア技法には、構造化プロンプティング（system/user/assistantメッセージ）、few-shotサンプル、連鎖推論（Chain-of-Thought）、JSONスキーマによる出力フォーマットがあります。本番システムでは、プロンプトのバージョン管理、A/Bテスト、インジェクション防御（入力サニタイズ、デリミタ、命令の階層化）、キャッシング戦略、パフォーマンスモニタリングが必要です。成功の鍵は、明確さ、コンテキスト効率、コストのバランスを取りつつ、敵対的入力に対するセキュリティを維持することです。

---

## プロンプトの構造

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PROMPT STRUCTURE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     SYSTEM MESSAGE                           │    │
│  │                                                              │    │
│  │  • Role definition and persona                               │    │
│  │  • Behavioral constraints and guidelines                     │    │
│  │  • Output format specifications                              │    │
│  │  • Available tools and capabilities                          │    │
│  │  • Examples and demonstrations                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   CONVERSATION HISTORY                       │    │
│  │                                                              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │    │
│  │  │  User    │  │Assistant │  │  User    │  │Assistant │    │    │
│  │  │ Message  │─►│ Response │─►│ Message  │─►│ Response │    │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    CURRENT USER INPUT                        │    │
│  │                                                              │    │
│  │  • User's question or instruction                            │    │
│  │  • Context or reference materials                            │    │
│  │  • Constraints for this specific request                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### メッセージタイプとロール

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None  # For tool messages
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

@dataclass
class Prompt:
    """Complete prompt structure."""
    system: str
    messages: List[Message] = field(default_factory=list)

    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to API message format."""
        result = [{"role": "system", "content": self.system}]

        for msg in self.messages:
            message_dict = {
                "role": msg.role.value,
                "content": msg.content
            }
            if msg.name:
                message_dict["name"] = msg.name
            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id

            result.append(message_dict)

        return result

    def count_tokens(self, tokenizer) -> int:
        """Estimate token count."""
        total = tokenizer.count(self.system)
        for msg in self.messages:
            total += tokenizer.count(msg.content)
        return total


# Example usage
prompt = Prompt(
    system="""You are a helpful coding assistant.
You write clean, well-documented Python code.
Always explain your reasoning before providing code.""",
    messages=[
        Message(
            role=MessageRole.USER,
            content="Write a function to check if a string is a palindrome"
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="I'll write a function that checks if a string reads the same forwards and backwards..."
        ),
        Message(
            role=MessageRole.USER,
            content="Now make it case-insensitive and ignore spaces"
        )
    ]
)
```

---

## 生成パラメータ

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GENERATION PARAMETERS                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TEMPERATURE                         TOP-P (Nucleus Sampling)        │
│  ┌─────────────────────────┐        ┌─────────────────────────┐     │
│  │ 0.0  Deterministic      │        │ 0.1  Very focused       │     │
│  │ 0.3  Conservative       │        │ 0.5  Balanced           │     │
│  │ 0.7  Balanced           │        │ 0.9  Diverse            │     │
│  │ 1.0  Creative           │        │ 1.0  All tokens         │     │
│  │ 1.5+ Very random        │        │                         │     │
│  └─────────────────────────┘        └─────────────────────────┘     │
│                                                                      │
│  FREQUENCY PENALTY                   PRESENCE PENALTY                │
│  ┌─────────────────────────┐        ┌─────────────────────────┐     │
│  │ -2.0 Encourage repetition│       │ -2.0 Encourage repeating│     │
│  │  0.0 No penalty          │       │  0.0 No penalty         │     │
│  │  1.0 Mild discouragement │       │  1.0 Mild discouragement│     │
│  │  2.0 Strong avoidance    │       │  2.0 Strong avoidance   │     │
│  │                          │       │                         │     │
│  │ Scales with frequency    │       │ Fixed penalty per token │     │
│  └─────────────────────────┘        └─────────────────────────┘     │
│                                                                      │
│  MAX TOKENS                          STOP SEQUENCES                  │
│  ┌─────────────────────────┐        ┌─────────────────────────┐     │
│  │ Limits output length    │        │ Strings that stop gen   │     │
│  │ Prevents runaway costs  │        │ e.g., "###", "\n\n"     │     │
│  └─────────────────────────┘        └─────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""

    # Sampling parameters
    temperature: float = 0.7  # 0.0 = deterministic, 1.0 = creative
    top_p: float = 1.0  # Nucleus sampling threshold
    top_k: Optional[int] = None  # Limit vocabulary to top K tokens

    # Penalty parameters
    frequency_penalty: float = 0.0  # Penalize based on frequency
    presence_penalty: float = 0.0  # Penalize any repetition
    repetition_penalty: float = 1.0  # Alternative: multiplicative penalty

    # Output control
    max_tokens: int = 1024
    stop_sequences: List[str] = None

    # Response format
    response_format: Optional[dict] = None  # {"type": "json_object"}

    def for_task(self, task_type: str) -> "GenerationConfig":
        """Get optimized config for specific task types."""
        configs = {
            "coding": GenerationConfig(
                temperature=0.2,
                top_p=0.95,
                max_tokens=2048,
                frequency_penalty=0.1
            ),
            "creative_writing": GenerationConfig(
                temperature=0.9,
                top_p=0.95,
                presence_penalty=0.6,
                frequency_penalty=0.3
            ),
            "factual_qa": GenerationConfig(
                temperature=0.0,
                top_p=1.0,
                max_tokens=512
            ),
            "summarization": GenerationConfig(
                temperature=0.3,
                top_p=0.9,
                max_tokens=256
            ),
            "classification": GenerationConfig(
                temperature=0.0,
                max_tokens=50
            ),
            "extraction": GenerationConfig(
                temperature=0.0,
                response_format={"type": "json_object"}
            )
        }
        return configs.get(task_type, self)


class TokenLimitManager:
    """Manage context window limits."""

    MODEL_LIMITS = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "llama-3-70b": 8192,
    }

    def __init__(self, model: str, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.context_limit = self.MODEL_LIMITS.get(model, 4096)

    def available_tokens(
        self,
        prompt: Prompt,
        reserved_for_output: int = 1024
    ) -> int:
        """Calculate remaining tokens for context."""
        used = prompt.count_tokens(self.tokenizer)
        return self.context_limit - used - reserved_for_output

    def truncate_messages(
        self,
        prompt: Prompt,
        max_output_tokens: int
    ) -> Prompt:
        """Truncate old messages to fit context window."""
        available = self.context_limit - max_output_tokens

        # Always keep system message
        system_tokens = self.tokenizer.count(prompt.system)
        remaining = available - system_tokens

        # Keep messages from most recent, drop oldest
        kept_messages = []
        for msg in reversed(prompt.messages):
            msg_tokens = self.tokenizer.count(msg.content)
            if remaining >= msg_tokens:
                kept_messages.insert(0, msg)
                remaining -= msg_tokens
            else:
                break

        return Prompt(system=prompt.system, messages=kept_messages)
```

---

## プロンプティング技法

### Zero-Shot vs Few-Shotプロンプティング

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ZERO-SHOT vs FEW-SHOT                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ZERO-SHOT                           FEW-SHOT                        │
│  ┌─────────────────────────┐        ┌─────────────────────────┐     │
│  │ Classify this text:     │        │ Examples:               │     │
│  │ "I love this product!"  │        │ "Great!" → positive     │     │
│  │                         │        │ "Terrible" → negative   │     │
│  │ Output: positive/negative│        │ "Okay" → neutral        │     │
│  │                         │        │                         │     │
│  │ (No examples provided)  │        │ Now classify:           │     │
│  │                         │        │ "I love this product!"  │     │
│  └─────────────────────────┘        └─────────────────────────┘     │
│                                                                      │
│  メリット:                           メリット:                        │
│  • トークン数が少ない               • より高精度                      │
│  • 反復が速い                        • フォーマットが一貫              │
│  • サンプルバイアスなし              • エッジケースに対応              │
│                                                                      │
│  デメリット:                         デメリット:                      │
│  • 予測性が低い                      • トークン数が多い               │
│  • フォーマットに従わない場合あり    • サンプル選択が重要              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class FewShotExample:
    """A single example for few-shot prompting."""
    input: str
    output: str
    explanation: Optional[str] = None

class FewShotPromptBuilder:
    """Build few-shot prompts with examples."""

    def __init__(self, task_description: str):
        self.task_description = task_description
        self.examples: List[FewShotExample] = []

    def add_example(
        self,
        input: str,
        output: str,
        explanation: str = None
    ):
        """Add an example to the prompt."""
        self.examples.append(FewShotExample(input, output, explanation))
        return self

    def build(self, query: str, include_explanations: bool = False) -> str:
        """Build the complete few-shot prompt."""
        parts = [self.task_description, "\nExamples:\n"]

        for i, ex in enumerate(self.examples, 1):
            parts.append(f"\nExample {i}:")
            parts.append(f"Input: {ex.input}")
            if include_explanations and ex.explanation:
                parts.append(f"Reasoning: {ex.explanation}")
            parts.append(f"Output: {ex.output}")

        parts.append(f"\n\nNow process this input:")
        parts.append(f"Input: {query}")
        parts.append("Output:")

        return "\n".join(parts)

    def select_examples(
        self,
        query: str,
        embedding_model,
        k: int = 3
    ) -> List[FewShotExample]:
        """Select most relevant examples using semantic similarity."""
        query_embedding = embedding_model.embed(query)

        scored_examples = []
        for ex in self.examples:
            ex_embedding = embedding_model.embed(ex.input)
            similarity = cosine_similarity(query_embedding, ex_embedding)
            scored_examples.append((similarity, ex))

        # Sort by similarity and return top k
        scored_examples.sort(reverse=True, key=lambda x: x[0])
        return [ex for _, ex in scored_examples[:k]]


# Usage example
sentiment_classifier = FewShotPromptBuilder(
    "Classify the sentiment of the following text as positive, negative, or neutral."
)

sentiment_classifier.add_example(
    input="This product exceeded my expectations!",
    output="positive",
    explanation="Expresses strong satisfaction"
)
sentiment_classifier.add_example(
    input="It broke after one day of use.",
    output="negative",
    explanation="Reports product failure"
)
sentiment_classifier.add_example(
    input="It works as described.",
    output="neutral",
    explanation="Factual statement without emotion"
)

prompt = sentiment_classifier.build("I'm never buying from this company again!")
# Output: negative
```

### Chain-of-Thought（CoT）プロンプティング

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CHAIN-OF-THOUGHT PROMPTING                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Question: If a train travels 120 miles in 2 hours, then stops       │
│  for 30 minutes, then travels 90 miles in 1.5 hours, what is         │
│  the average speed for the entire journey?                           │
│                                                                      │
│  WITHOUT CoT                         WITH CoT                        │
│  ┌─────────────────────────┐        ┌─────────────────────────┐     │
│  │                         │        │ Let me work through this│     │
│  │ Answer: 55 mph          │        │ step by step:           │     │
│  │                         │        │                         │     │
│  │ (Often incorrect)       │        │ 1. Total distance:      │     │
│  │                         │        │    120 + 90 = 210 miles │     │
│  │                         │        │                         │     │
│  │                         │        │ 2. Total time:          │     │
│  │                         │        │    2 + 0.5 + 1.5 = 4 hrs│     │
│  │                         │        │                         │     │
│  │                         │        │ 3. Average speed:       │     │
│  │                         │        │    210 / 4 = 52.5 mph   │     │
│  │                         │        │                         │     │
│  │                         │        │ Answer: 52.5 mph        │     │
│  └─────────────────────────┘        └─────────────────────────┘     │
│                                                                      │
│  トリガーフレーズ:                                                    │
│  • "Let's think step by step"                                        │
│  • "Let's work through this carefully"                               │
│  • "First, ... Then, ... Finally, ..."                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

この記事の完全な内容（Self-Consistency、ReAct、プロンプトテンプレートシステム、プロンプトインジェクション防御、バージョン管理、A/Bテスト、プロンプト最適化、キャッシング、モニタリング等）については、英語版の原文をご参照ください。コードブロックを含む全パターンは英語版と同一の内容です。

---

## トレードオフ

| 判断項目 | トレードオフ |
|----------|-----------|
| **Zero-Shot vs Few-Shot** | トークン数の節約 vs 精度と一貫性 |
| **Temperature** | 創造性 vs 一貫性 |
| **コンテキスト長** | 情報量 vs コストとレイテンシ |
| **構造化出力** | 信頼性 vs 柔軟性 |
| **プロンプト長** | 明確さ vs コスト |

---

## 参考文献

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/introduction-to-prompt-design)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves CoT Reasoning](https://arxiv.org/abs/2203.11171)
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Prompt Injection Attacks](https://arxiv.org/abs/2302.12173)
- [DSPy: Programming with Foundation Models](https://github.com/stanfordnlp/dspy)
