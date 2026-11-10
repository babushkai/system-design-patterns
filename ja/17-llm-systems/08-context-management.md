# コンテキスト管理パターン

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../17-llm-systems/08-context-management.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

コンテキスト管理とは、LLMの限られたコンテキストウィンドウを効率的に活用し、コストを制御しながら応答品質を最大化する技術です。主要な戦略には、トークンエコノミクスの理解（入出力の配分、コストへの影響）、長いコンテキストの処理（Lost-in-the-Middle問題、Needle-in-Haystackテスト）、圧縮技法（要約、LLMLingua、選択的コンテキスト）、スライディングウィンドウパターン（ローリング要約、階層型コンテキスト）、メモリシステム（短期、長期、エピソード的、意味的、作業メモリ）、本番最適化（KVキャッシング、プレフィックスキャッシング、コンテキスト事前計算）があります。ロングコンテキストとRAGの選択は、更新頻度、精度要件、コスト制約によって決まります。

---

## コンテキストウィンドウの基礎

### コンテキストウィンドウの構造

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
│  入力トークン: 入力料金で課金                                        │
│  出力トークン: 出力料金で課金（通常入力の2-3倍）                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### トークンカウントと制限

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

        tokens_per_message = 4
        tokens_per_name = -1

        total = 0
        for message in messages:
            total += tokens_per_message
            for key, value in message.items():
                total += len(encoder.encode(str(value)))
                if key == "name":
                    total += tokens_per_name

        total += 3
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
```

---

## ロングコンテキストの処理

### Lost-in-the-Middle問題

```
┌─────────────────────────────────────────────────────────────────────┐
│                  LOST IN THE MIDDLE PHENOMENON                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  コンテキストウィンドウ全体の注意力分布:                              │
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
│         先頭              中間                末尾                    │
│         (高い再現率)    (低い再現率)     (高い再現率)                 │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 重要な知見: モデルはコンテキストの先頭と末尾には注意を      │    │
│  │ 向けますが、中間の情報は見落としがちです。                   │    │
│  │                                                              │    │
│  │ 緩和戦略:                                                    │    │
│  │ • 重要な情報を先頭か末尾に配置                               │    │
│  │ • 明示的なマーカーと構造を使用                               │    │
│  │ • 重要な情報を繰り返す                                       │    │
│  │ • 階層的な構成を使用                                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

この記事の完全な内容（コンテキスト予算管理、Bookend戦略、Needle-in-Haystackテスト、コンテキスト圧縮、スライディングウィンドウパターン、メモリシステム、KVキャッシング、ロングコンテキスト vs RAG比較等）については、英語版の原文をご参照ください。コードブロックを含む全パターンは英語版と同一の内容です。

このプレフィックスキャッシングの規律をサービング側から見た鏡像 — キャッシュ認識ルーティング、フェイルオーバーコスト、フリートのKVフットプリント — は[エージェント推論](./12-agent-inference.md)にあります。

---

## ロングコンテキスト vs RAG

| 観点 | ロングコンテキスト | RAG |
|------|-----------------|-----|
| **精度** | 全コンテキスト利用可能 | 検索品質に依存 |
| **コスト** | 高い（全トークン課金） | 低い（関連チャンクのみ） |
| **レイテンシ** | 長い初回応答 | 検索+生成のオーバーヘッド |
| **更新** | コンテキスト入れ替えが必要 | インデックス更新のみ |
| **スケール** | ウィンドウサイズに制限 | 任意のコーパスサイズ |

---

## トレードオフ

| 判断項目 | トレードオフ |
|----------|-----------|
| **コンテキスト長** | 多い情報 vs コストとレイテンシ |
| **圧縮** | トークン節約 vs 情報損失 |
| **メモリ戦略** | 長期記憶の品質 vs 検索コスト |
| **キャッシュサイズ** | ヒット率 vs メモリ使用量 |
| **要約頻度** | コンテキスト効率 vs 計算コスト |

---

## 参考文献

- [Lost in the Middle (Stanford)](https://arxiv.org/abs/2307.03172)
- [LLMLingua: Prompt Compression](https://arxiv.org/abs/2310.05736)
- [Extending Context Window (RoPE Scaling)](https://arxiv.org/abs/2306.15595)
- [MemGPT: Virtual Context Management](https://arxiv.org/abs/2310.08560)
- [Anthropic Long Context](https://www.anthropic.com/news/claude-2-1-200k)
- [Google Gemini Long Context](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/)
