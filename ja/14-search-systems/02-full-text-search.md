# 全文検索

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../14-search-systems/02-full-text-search.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

全文検索は、完全一致の文字列マッチングではなく、テキストコンテンツを分析することで関連するドキュメントを見つけます。主要なコンポーネントには、テキスト分析（トークン化、ステミング）、関連性スコアリング（TF-IDF、BM25）、クエリタイプ（ブーリアン、フレーズ、ファジー）があります。Elasticsearch と Solr が主要な実装です。

---

## テキスト分析パイプライン

### 分析プロセス

```
Raw Text: "The Quick Brown Foxes are JUMPING over lazy dogs!"
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Character Filters                             │
│                                                                 │
│  • HTML stripping: <b>text</b> → text                           │
│  • Pattern replace: é → e                                        │
│  • Mapping: & → and                                              │
│                                                                 │
│  Result: "The Quick Brown Foxes are JUMPING over lazy dogs!"    │
└─────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Tokenizer                                   │
│                                                                 │
│  Split text into tokens (words)                                 │
│                                                                 │
│  Result: ["The", "Quick", "Brown", "Foxes", "are", "JUMPING",   │
│           "over", "lazy", "dogs"]                               │
└─────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Token Filters                                 │
│                                                                 │
│  1. Lowercase:   ["the", "quick", "brown", "foxes", ...]        │
│  2. Stop words:  ["quick", "brown", "foxes", "jumping", ...]    │
│  3. Stemming:    ["quick", "brown", "fox", "jump", ...]         │
│  4. Synonyms:    ["quick", "fast", "brown", "fox", ...]         │
│                                                                 │
│  Final tokens: ["quick", "brown", "fox", "jump", "lazi", "dog"] │
└─────────────────────────────────────────────────────────────────┘
```

テキスト分析パイプラインは3段階で構成されます。**文字フィルター**で HTML の除去やパターン置換を行い、**トークナイザー**でテキストをトークン（単語）に分割し、**トークンフィルター**で小文字化、ストップワード除去、ステミング、同義語展開を行います。

### Elasticsearch アナライザー設定

```json
{
  "settings": {
    "analysis": {
      "char_filter": {
        "html_strip": {
          "type": "html_strip"
        }
      },
      "tokenizer": {
        "standard": {
          "type": "standard"
        }
      },
      "filter": {
        "english_stop": {
          "type": "stop",
          "stopwords": "_english_"
        },
        "english_stemmer": {
          "type": "stemmer",
          "language": "english"
        },
        "my_synonyms": {
          "type": "synonym",
          "synonyms": [
            "quick, fast, speedy",
            "big, large, huge"
          ]
        }
      },
      "analyzer": {
        "my_english_analyzer": {
          "type": "custom",
          "char_filter": ["html_strip"],
          "tokenizer": "standard",
          "filter": [
            "lowercase",
            "english_stop",
            "english_stemmer",
            "my_synonyms"
          ]
        }
      }
    }
  }
}
```

### ステミング vs. レンマ化

```
Stemming (Porter Stemmer):
  running → run
  runner  → runner  (not always correct)
  better  → better  (doesn't handle irregulars)

  Pros: Fast, simple rules
  Cons: Can be incorrect, not linguistically aware

Lemmatization:
  running → run
  runner  → run
  better  → good

  Pros: Linguistically correct
  Cons: Slower, requires dictionary

Elasticsearch default: Stemming (configurable)
```

**ステミング**は高速でシンプルなルールベースですが、不正確な場合があります。**レンマ化**は言語学的に正しい結果を返しますが、辞書が必要でより低速です。Elasticsearch のデフォルトはステミングです（設定で変更可能）。

---

## 関連性スコアリング

### TF-IDF

```
TF (Term Frequency):
  How often does the term appear in this document?

  TF = count(term in doc) / total_terms_in_doc

IDF (Inverse Document Frequency):
  How rare is this term across all documents?

  IDF = log(total_docs / docs_containing_term)

TF-IDF = TF × IDF

Example:
  Query: "search engine"
  Doc A: "This is a search engine. The search engine is fast."
  Doc B: "This is a car engine."

  For term "search":
    TF in Doc A = 2/11 = 0.18
    TF in Doc B = 0/5 = 0
    IDF = log(2/1) = 0.30

    TF-IDF(A) = 0.18 × 0.30 = 0.054
    TF-IDF(B) = 0 × 0.30 = 0

  Doc A scores higher for "search"
```

TF（出現頻度）はドキュメント内でその用語がどの程度出現するかを測定します。IDF（逆文書頻度）は全ドキュメントに対してその用語がどの程度希少かを測定します。TF-IDF はこの2つの積です。

### BM25（Best Matching 25）

```
BM25 improves on TF-IDF with:
- Saturation: Diminishing returns for repeated terms
- Document length normalization

Formula:
  score(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))

Where:
  f(qi,D) = term frequency in document
  |D| = document length
  avgdl = average document length
  k1 = term frequency saturation (typically 1.2)
  b = length normalization (typically 0.75)

TF-IDF vs BM25:
┌────────────────────────────────────────────────────────────────┐
│  Score                                                         │
│    │                                                           │
│    │    TF-IDF (linear)                                        │
│    │         /                                                 │
│    │        /                                                  │
│    │       /    BM25 (saturating)                             │
│    │      /   ═══════════════                                 │
│    │     / ═══                                                │
│    │    /══                                                   │
│    │  ═/                                                      │
│    │ /                                                        │
│    └────────────────────────────────────────────► Term Freq   │
│                                                                │
│  BM25: 10 occurrences doesn't score 10x more than 1           │
└────────────────────────────────────────────────────────────────┘
```

BM25 は TF-IDF を改良し、用語の繰り返しに対する飽和（収穫逓減）とドキュメント長の正規化を追加しています。10回の出現が1回の10倍のスコアにはなりません。

### フィールドブースティング

```json
// Boost matches in title higher than body
{
  "query": {
    "multi_match": {
      "query": "search engine",
      "fields": ["title^3", "body^1", "tags^2"]
    }
  }
}

// title matches weighted 3x
// tags matches weighted 2x
// body matches weighted 1x (baseline)
```

タイトルでのマッチに3倍の重み、タグに2倍の重み、本文に1倍の重み（基準）を設定できます。

---

## クエリタイプ

### マッチクエリ（分析あり）

```json
// Query is analyzed the same way as indexed content
{
  "query": {
    "match": {
      "content": "running fast"
    }
  }
}

// Analyzed to: ["run", "fast"] (stemmed)
// Finds documents containing "run" OR "fast"
// Documents with both terms score higher
```

クエリはインデックスされたコンテンツと同様に分析されます。「running fast」は「run」「fast」にステミングされ、いずれかの用語を含むドキュメントが検索されます。

### タームクエリ（分析なし）

```json
// Exact match - no analysis
{
  "query": {
    "term": {
      "status": "published"
    }
  }
}

// Use for keyword fields, IDs, enums
// DON'T use for text fields (won't match analyzed content)
```

完全一致で分析は行われません。キーワードフィールド、ID、列挙型に使用します。テキストフィールドには使用しないでください（分析済みコンテンツとマッチしません）。

### ブーリアンクエリ

```json
{
  "query": {
    "bool": {
      "must": [
        // All must match (AND)
        { "match": { "title": "elasticsearch" } }
      ],
      "should": [
        // At least one should match, boosts score (OR)
        { "match": { "content": "tutorial" } },
        { "match": { "content": "guide" } }
      ],
      "must_not": [
        // Must not match (NOT)
        { "term": { "status": "draft" } }
      ],
      "filter": [
        // Must match but doesn't affect score
        { "range": { "date": { "gte": "2024-01-01" } } },
        { "term": { "category": "tech" } }
      ]
    }
  }
}
```

`must` はすべて一致する必要があり（AND）、`should` は少なくとも1つが一致するとスコアがブーストされ（OR）、`must_not` は一致してはならず（NOT）、`filter` は一致する必要がありますがスコアに影響しません。

### フレーズクエリ

```json
// Terms must appear in order
{
  "query": {
    "match_phrase": {
      "content": "quick brown fox"
    }
  }
}

// Matches: "The quick brown fox jumped"
// Doesn't match: "brown quick fox" (wrong order)
// Doesn't match: "quick fox" (missing "brown")

// With slop (allowed gaps)
{
  "query": {
    "match_phrase": {
      "content": {
        "query": "quick fox",
        "slop": 2  // Allow up to 2 words between
      }
    }
  }
}
// Matches: "quick brown fox" (1 word between)
```

用語は順序通りに出現する必要があります。`slop` パラメーターで用語間の許容ギャップを指定できます。

### ファジークエリ

```json
// Matches with edit distance (typo tolerance)
{
  "query": {
    "fuzzy": {
      "title": {
        "value": "elasticsaerch",  // Typo
        "fuzziness": 2  // Allow 2 edits (insert, delete, substitute)
      }
    }
  }
}

// Matches: "elasticsearch" (2 character swaps)

// Auto fuzziness based on term length
{
  "query": {
    "match": {
      "title": {
        "query": "elasticsaerch",
        "fuzziness": "AUTO"
        // 0-2 chars: exact match
        // 3-5 chars: 1 edit
        // >5 chars: 2 edits
      }
    }
  }
}
```

編集距離（タイポ許容）を使用してマッチングします。`AUTO` ファジネスでは、用語の長さに基づいて自動的に許容編集数が決まります。

---

## オートコンプリートとサジェスト

### Edge N-gram によるプレフィックスマッチング

```json
// Index configuration
{
  "settings": {
    "analysis": {
      "filter": {
        "edge_ngram": {
          "type": "edge_ngram",
          "min_gram": 1,
          "max_gram": 20
        }
      },
      "analyzer": {
        "autocomplete": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase", "edge_ngram"]
        },
        "autocomplete_search": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase"]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "autocomplete",
        "search_analyzer": "autocomplete_search"
      }
    }
  }
}

// "elasticsearch" indexed as:
// ["e", "el", "ela", "elas", "elast", "elasti", "elastic", ...]

// Search for "elast" → matches "elasticsearch"
```

### Completion サジェスター

```json
// Index with completion field
{
  "mappings": {
    "properties": {
      "suggest": {
        "type": "completion"
      }
    }
  }
}

// Index document
{
  "title": "Elasticsearch Tutorial",
  "suggest": {
    "input": ["elasticsearch", "elastic search", "es tutorial"],
    "weight": 10
  }
}

// Query
{
  "suggest": {
    "title-suggest": {
      "prefix": "elast",
      "completion": {
        "field": "suggest",
        "fuzzy": {
          "fuzziness": 1
        }
      }
    }
  }
}
```

### Search-as-you-type

```json
// Multi-field for different match types
{
  "mappings": {
    "properties": {
      "title": {
        "type": "search_as_you_type"
      }
    }
  }
}

// Creates sub-fields:
// title           - standard analysis
// title._2gram    - shingle (2 word combinations)
// title._3gram    - shingle (3 word combinations)
// title._index_prefix - edge ngrams

// Query
{
  "query": {
    "multi_match": {
      "query": "quick br",
      "type": "bool_prefix",
      "fields": [
        "title",
        "title._2gram",
        "title._3gram"
      ]
    }
  }
}
```

`search_as_you_type` フィールドタイプは、標準分析、シングル（2語・3語の組み合わせ）、edge n-gram のサブフィールドを自動的に作成します。

---

## ファセット検索

### ファセット用アグリゲーション

```json
// Search with facets
{
  "query": {
    "match": { "content": "laptop" }
  },
  "aggs": {
    "categories": {
      "terms": { "field": "category.keyword" }
    },
    "brands": {
      "terms": { "field": "brand.keyword" }
    },
    "price_ranges": {
      "range": {
        "field": "price",
        "ranges": [
          { "to": 500 },
          { "from": 500, "to": 1000 },
          { "from": 1000, "to": 2000 },
          { "from": 2000 }
        ]
      }
    },
    "avg_price": {
      "avg": { "field": "price" }
    }
  }
}

// Response includes:
{
  "hits": { ... },
  "aggregations": {
    "categories": {
      "buckets": [
        { "key": "Electronics", "doc_count": 150 },
        { "key": "Computers", "doc_count": 120 }
      ]
    },
    "brands": {
      "buckets": [
        { "key": "Apple", "doc_count": 45 },
        { "key": "Dell", "doc_count": 38 }
      ]
    },
    "price_ranges": {
      "buckets": [
        { "key": "*-500", "doc_count": 80 },
        { "key": "500-1000", "doc_count": 100 }
      ]
    }
  }
}
```

### ファセット独立性のための Post-Filter

```json
// Problem: Filtering by brand hides other brands in facet
// Solution: post_filter applies AFTER aggregations

{
  "query": {
    "match": { "content": "laptop" }
  },
  "aggs": {
    "all_brands": {
      "global": {},  // Ignores query filter
      "aggs": {
        "brands": {
          "filter": {
            "match": { "content": "laptop" }  // Reapply base query
          },
          "aggs": {
            "brand_names": {
              "terms": { "field": "brand.keyword" }
            }
          }
        }
      }
    }
  },
  "post_filter": {
    "term": { "brand.keyword": "Apple" }
  }
}

// Results filtered to Apple
// But brand facet shows all brands matching "laptop"
```

ブランドでフィルタリングすると他のブランドがファセットから消えてしまう問題を解決するために、`post_filter` はアグリゲーションの後に適用されます。

---

## ハイライト

```json
{
  "query": {
    "match": { "content": "elasticsearch tutorial" }
  },
  "highlight": {
    "fields": {
      "content": {
        "pre_tags": ["<mark>"],
        "post_tags": ["</mark>"],
        "fragment_size": 150,
        "number_of_fragments": 3
      }
    }
  }
}

// Response:
{
  "hits": {
    "hits": [
      {
        "_source": { ... },
        "highlight": {
          "content": [
            "This <mark>tutorial</mark> explains how <mark>Elasticsearch</mark> works...",
            "Learn <mark>Elasticsearch</mark> in this comprehensive <mark>tutorial</mark>..."
          ]
        }
      }
    ]
  }
}
```

検索結果の中でマッチした用語をハイライト表示できます。フラグメントサイズやフラグメント数を制御可能です。

---

## パフォーマンス最適化

### マッピング最適化

```json
{
  "mappings": {
    "properties": {
      // Don't index fields you won't search
      "internal_id": {
        "type": "keyword",
        "index": false  // Not searchable, just stored
      },

      // Disable norms for filtering-only fields
      "status": {
        "type": "keyword",
        "norms": false
      },

      // Use appropriate types
      "count": {
        "type": "integer"  // Not text!
      },

      // Limit field length
      "description": {
        "type": "text",
        "ignore_above": 10000
      }
    }
  }
}
```

検索しないフィールドはインデックスしない、フィルタリング専用フィールドでは norms を無効化する、適切なタイプを使用するなどの最適化が重要です。

### クエリ最適化

```python
# 1. Use filters for non-scoring criteria
{
  "query": {
    "bool": {
      "must": { "match": { "title": "search" } },
      "filter": [  # Cached, no scoring
        { "term": { "status": "published" } },
        { "range": { "date": { "gte": "2024-01-01" } } }
      ]
    }
  }
}

# 2. Limit returned fields
{
  "query": { ... },
  "_source": ["title", "date"],  # Only fetch needed fields
  "size": 10
}

# 3. Use search_after for deep pagination
# First request
{
  "query": { ... },
  "sort": [
    { "date": "desc" },
    { "_id": "asc" }
  ],
  "size": 10
}

# Subsequent requests
{
  "query": { ... },
  "search_after": ["2024-01-15", "doc_id_123"],  # Last result's sort values
  "sort": [
    { "date": "desc" },
    { "_id": "asc" }
  ],
  "size": 10
}

# 4. Pre-warm cache with warmers or eager loading
{
  "index": {
    "queries": {
      "cache": {
        "enabled": true
      }
    }
  }
}
```

1. スコアリング不要な条件にはフィルターを使用します（キャッシュされます）。
2. 返却フィールドを必要なものだけに制限します。
3. ディープページネーションには offset ではなく `search_after` を使用します。
4. ウォーマーやイーガーローディングでキャッシュを事前にウォームアップします。

---

## アーキテクチャパターン

### 読み取り負荷の高い検索クラスター

```
                          ┌──────────────────┐
                          │   Application    │
                          └────────┬─────────┘
                                   │
                          ┌────────┴─────────┐
                          │   Load Balancer  │
                          └────────┬─────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
    ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
    │ Coordinating  │      │ Coordinating  │      │ Coordinating  │
    │    Node       │      │    Node       │      │    Node       │
    │  (no data)    │      │  (no data)    │      │  (no data)    │
    └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
    ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
    │   Data Node   │      │   Data Node   │      │   Data Node   │
    │  Shard 0 (P)  │      │  Shard 1 (P)  │      │  Shard 2 (P)  │
    │  Shard 1 (R)  │      │  Shard 2 (R)  │      │  Shard 0 (R)  │
    └───────────────┘      └───────────────┘      └───────────────┘

P = Primary, R = Replica
```

コーディネーティングノードはクエリの解析と結果のマージを行い、データノードはシャードを格納して検索を実行します。P はプライマリ、R はレプリカです。

### インデクシングパイプライン

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Source    │────►│   Kafka     │────►│   Logstash  │
│  Database   │     │   Queue     │     │  /Ingest    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               │ Bulk API
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │   Elasticsearch     │
                                    │   Cluster           │
                                    └─────────────────────┘
```

利点:
- インデクシングをソースシステムから分離できます
- バックプレッシャーを処理できます（キューがバッファリング）
- Logstash/Ingest パイプラインで変換できます
- バルクインデクシングで効率化できます

---

## ベストプラクティス

```
Index Design:
□ Use appropriate analyzers for each field
□ Don't index fields you won't search
□ Use keyword type for exact match/aggregations
□ Set reasonable shard count (not too many)

Query Design:
□ Use filters for non-scoring criteria
□ Limit result size and returned fields
□ Use search_after for deep pagination
□ Cache frequent queries

Operations:
□ Monitor query latency and indexing rate
□ Set up slow query logging
□ Plan for index lifecycle (rollover, delete)
□ Test with production-like data volume
```

**インデックス設計**: 各フィールドに適切なアナライザーを使用し、検索しないフィールドはインデックスしません。完全一致/アグリゲーションにはキーワードタイプを使用し、適切なシャード数を設定します。

**クエリ設計**: スコアリング不要な条件にはフィルターを使用し、結果サイズと返却フィールドを制限します。ディープページネーションには `search_after` を使用し、頻繁なクエリをキャッシュします。

**運用**: クエリレイテンシーとインデクシングレートを監視し、スロークエリログを設定します。インデックスライフサイクル（ロールオーバー、削除）を計画し、本番に近いデータ量でテストします。

---

## 参考文献

- [Elasticsearch: The Definitive Guide](https://www.elastic.co/guide/en/elasticsearch/guide/current/index.html)
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
- [Lucene in Action](https://www.manning.com/books/lucene-in-action-second-edition)
- [Relevant Search](https://www.manning.com/books/relevant-search)
