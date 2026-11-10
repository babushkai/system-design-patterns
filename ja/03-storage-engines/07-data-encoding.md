# データエンコーディング

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../03-storage-engines/07-data-encoding.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

エンコーディングとは、データと時間の間の契約です。永続的な境界 — ディスク、ネットワーク、キュー — を越えるすべてのバイトは、それを書いたコードより長生きし、まだ存在しないコードによって読まれます。つまりエンコーディングは、ひとつの名前をまとった2つの問題です: *パフォーマンス*の問題（JSONは約100MB/sでパースされペイロードを3倍にする。Protobufは5〜10倍良い。ゼロコピーフォーマットはパース自体をスキップする）と、*互換性*の問題（昨年のコンシューマは明日のプロデューサの出力を読めるか？逆は？）。障害を引き起こすのは互換性の問題のほうです。それは機械的なルール — Protobufのフィールドタグ、Avroのreader/writerスキーマ解決、reserved番号の規律 — に支配されており、成熟したシステムではスキーマレジストリが、破壊的変更を午前3時に発見する代わりにCI時点で拒否します。本章ではワイヤフォーマットとその形の理由、フォーマットごとの進化ルール、レジストリの運用、そしてあらゆる会社で再発する障害モード（JSONの数値精度、タグの再利用、requiredフィールドの罠、データレイクのスキーマドリフト）を扱います。

---

## 問題: データはコードより長生きする

```
In-memory object:                       Must become bytes for:
  User {                                  - disk (storage engines, backups)
    id: 123,                              - network (RPC, APIs)
    name: "Alice",                        - queues (events that sit for days)
    emails: [...]                         - other languages entirely
  }

And the reverse, years later, by code that has been deployed
hundreds of times since the bytes were written.
```

すべてのエンコーディング設計を駆動する非対称性: **書き込みは1回、1つのスキーマの下で起こる。読み取りは永遠に、これから存在するすべてのスキーマの下で起こる。** ローリングデプロイは古いコードと新しいコードが同じトピックとテーブルに対して*同時に*動くことを意味するので、互換性は移行イベントではなく恒常的な運用条件です:

```
Backward compatibility:  NEW code reads OLD data
  (required to deploy consumers first, and to read historical data —
   missing fields must have defaults)

Forward compatibility:   OLD code reads NEW data
  (required to deploy producers first — unknown fields must be
   skippable without breaking)

Full compatibility = both = deploy in any order. Rolling deploys and
multi-team ownership effectively demand full compatibility.
```

---

## テキストフォーマット: JSONとその鋭い刃

```json
{"id": 123, "name": "Alice", "emails": ["a@example.com"], "balance": 99.99}
```

JSONは組織間の境界で勝利しました — 自己記述的で、普遍的で、目視でデバッグできる。そのコストはよく知られています（バイナリの3〜5倍のバイト数、レコードごとに繰り返されるキー名、約100MB/sのパーススループット、バイナリデータにはbase64）。あまり知られていないのは、実際のインシデントを引き起こす鋭い刃のほうです:

```
Number precision: JSON has ONE number type — IEEE-754 double.
  Integers above 2⁵³ silently lose precision in JavaScript and many
  parsers. Twitter's snowflake IDs (64-bit) famously broke JS clients:
    {"id": 10765432100123456789}  → parsed as ...456780
  Fix: serialize 64-bit IDs as strings ("id_str"), or use a
  big-integer-aware parser. This bug ships to production constantly.

No schema = no contract: a "compatible" change is whatever your
  most fragile consumer's hand-written parsing tolerates. null vs
  absent-field vs "" distinctions differ per language and library.

Duplicate keys, key ordering, NaN/Infinity: all
  implementation-defined. Canonicalization (for signing/hashing JSON)
  is a minefield — see JCS (RFC 8785) before hashing JSON.
```

XMLが生き残るのは、そのスキーマ機構（XSD、名前空間）やドキュメント指向が本当に使われている場所です — SOAP時代のエンタープライズ、設定、文書。データレコード用途では完全に劣位です。YAMLは人間が設定を編集するためのもので、マシン間のデータには決して使いません（暗黙の型付け: `no` は `false` に、`3.10` は `3.1` にパースされます）。

---

## バイナリフォーマット: 3つの異なる賭け

### Protobuf: ワイヤ上のタグ番号

```protobuf
message User {
  int32 id = 1;                //  ← the "= 1" is the wire identity
  string name = 2;
  repeated string emails = 3;
}
```

```
Wire format: a stream of (tag, wire-type, value) triples
  field 1 (id=123):    08 7B          — tag+type: 1 byte, varint value
  field 2 ("Alice"):   12 05 41 6c 69 63 65   — tag, length, bytes

Varint: 7 bits per byte, high bit = continuation.
  0-127 → 1 byte; 300 → 2 bytes. Small numbers are nearly free.
  (sint32/64 add ZigZag: maps -1→1, 1→2 so negatives stay small.)

The consequences of tags-on-wire:
  + unknown tags are SKIPPABLE (wire type tells the length)
    → forward compatibility is structural, not optional
  + field NAMES are compile-time only — renaming a field is free
  - the tag number IS the field's identity forever
    → reusing a tag misinterprets old data with the new field's type:
      silent corruption, the worst failure mode in this chapter
```

### Avro: タグなし、スキーマが仕事をする

```json
{"type": "record", "name": "User", "fields": [
  {"name": "id", "type": "int"},
  {"name": "name", "type": "string"}
]}
```

```
Wire format: just concatenated values, in schema order. No tags,
no field names, no lengths where the type implies them.
  → the smallest payloads of any general format
  → but UNREADABLE without the writer's exact schema

Reading = schema resolution: reader supplies ITS schema, runtime
matches fields BY NAME against the writer's schema:
  writer has extra field  → reader skips it (forward compat)
  reader has extra field  → default value fills it (backward compat,
                            IF a default was declared — enforced!)
  renamed field           → aliases: ["old_name"]

The bet: schemas are cheap to distribute (embedded in files for
batch: one schema, a million rows; registry ID for streaming: 5 bytes
per message). Where that holds — Kafka, data lakes — Avro excels.
Where it doesn't — ad-hoc RPC — Protobuf's self-delimiting fields win.
```

### ゼロコピー: FlatBuffersとCap'n Proto

```
Traditional: bytes → parse → object graph → access
Zero-copy:   bytes → access (fields read directly from the buffer
             via offsets; "parse time" is zero)

Use when: you read millions of records and touch 2 fields each
  (games, ML feature pipelines, mmap'd files, IPC).
Costs: larger payloads than protobuf (alignment, offset tables),
  awkward mutation, less ecosystem. Wrong default for APIs.
```

MessagePack/CBORは「バイナリJSON」です — スキーマレスで40〜60%小さく高速。JSONの柔軟性を無駄少なく使いたいときに適します（CBORはIoT/COSEの標準）が、JSONの契約不在の問題を受け継ぎます。

---

## スキーマ進化: フォーマットごとのルール

### Protobuf

```protobuf
message User {
  reserved 2, 5;                 // tags retired FOREVER
  reserved "name", "age";        // names too (prevents tooling confusion)
  int32 id = 1;
  string email = 3;
  string phone = 4;
}
```

```
Safe:                                Unsafe:
  add field with NEW tag               reuse a tag number — silent corruption
  remove field + reserve its tag       change a field's type (mostly)
  rename a field (names ≠ wire)        change int32 ↔ sint32 (different encoding!)
  optional → repeated (scalars)        anything with required (proto2 — which is
                                       why proto3 removed required entirely)
```

`required` の教訓は世代をまたぎます: 存在必須のフィールドは決して削除できません。どこかのリーダーがメッセージを拒否するからです。成熟したスキーマ文化はすべて「全部optional、どこでもデフォルト、バリデーションはコードで」に収束しました — 制約はデプロイ可能なアプリケーション層に属します。ワイヤフォーマットはデプロイできません。

### Avro

```
Safe:                                Requires care:
  add field WITH default               add field without default: breaks
  remove field that HAD default          backward compat (registry rejects)
  rename via aliases                   type promotions: int→long→float→double
  reorder fields (matched by name)       OK; the reverse is not
```

### JSON

機械的なルールは存在しません — 進化の規律は慣習です: 追加のみの変更、JSON Schemaを使うなら `additionalProperties: true`、バージョンエンベロープ（`{"v": 2, ...}`）またはバージョン付きエンドポイント、そしてレジストリが強制するはずのものを代替するコントラクトテスト。

---

## スキーマレジストリ: 互換性を強制可能にする

レジストリは進化ルールを部族の知恵からビルド時のゲートに変えます:

```
Producer:  register schema (once) → get 4-byte schema ID
           send [magic 0x0][schema_id: 4B][avro payload]
Consumer:  read ID → fetch schema (cached — registry is NOT on the
           hot path) → resolve against its own reader schema

The enforcement: registering a schema that violates the subject's
compatibility mode is REJECTED — the incompatible producer fails in
CI/deploy, not in the consumer at runtime.
```

```
Compatibility modes (Confluent vocabulary):
  BACKWARD (default): new schema can read data written by the last
    schema. Deployment order: consumers first.
  FORWARD:  last schema can read data written by the new schema.
    Deployment order: producers first.
  FULL:     both. Deploy in any order.
  *_TRANSITIVE: checked against ALL registered versions, not just the
    last — REQUIRED if a topic/lake retains data older than one schema
    generation (i.e., almost always the right choice; the non-transitive
    defaults are a common gap: v3 is compatible with v2, v2 with v1,
    but v3 cannot read v1's data still sitting in the topic).

API sketch:
  POST /subjects/{subject}/versions            → register, returns {"id": 42}
  POST /compatibility/subjects/{s}/versions/latest → pre-flight check
  PUT  /config/{subject}                       → set mode per subject
```

Protobufのエコシステムは同じゲートを（レジストリの有無にかかわらず）`buf breaking`（CIでベースラインに対するスキーマリンティング）から得ます。原則は同一です — **破壊的変更はコンシューマではなくビルドを失敗させるべき**。

---

## パフォーマンス: ベンチマークが実際に言っていること

| フォーマット | シリアライズ | デシリアライズ | サイズ（1KB JSON基準） | スキーマ |
|--------|-----------|-------------|--------------|--------|
| JSON | ~100 MB/s | ~200 MB/s | 100% | 不要 |
| JSON + gzip | ~50 MB/s | ~80 MB/s | ~40% | 不要 |
| MessagePack | ~400 MB/s | ~800 MB/s | ~65% | 不要 |
| Protobuf | ~800 MB/s | ~1.5 GB/s | ~30% | 必要 |
| Avro | ~600 MB/s | ~1 GB/s | ~28% | 必要 |
| FlatBuffers | ~1.5 GB/s | ゼロコピー | ~45% | 必要 |

*オーダーレベルの目安。言語とデータの形で変動します。JS/Pythonでは差は縮まり（JSONパーサはCで書かれ、protobufはそうでないことが多い）、Go/Java/C++/Rustでは広がります。*

スキーマ付きバイナリが勝つ理由: ワイヤ上にフィールド名がない（代わりにタグ/位置）、型の推測がない（スキーマがコンパイル時に型を固定）、小さな数値にはvarint、リフレクションではなく生成コード。それが問題にならない場合: 50 req/sのサービスはJSONに意味のあるコストを払っていません — エンコーディング最適化が報われるのは*パイプライン*スケール（ホップごとのコスト × ホップ数 × メッセージレート）か*ストレージ*スケール（数十億行 — その場合はそもそも[Parquet/カラムナ](./06-column-storage.md)にいるべきで、そこでは同じエンコーディング — 辞書、RLE、ビットパッキング — がカラム単位で適用されます）。

---

## システムの中でエンコーディングが住む場所

```
External API:      JSON (REST/GraphQL) — humans and unknown clients
Internal RPC:      Protobuf over gRPC — contracts + codegen + speed
Event streaming:   Avro (or Protobuf) + Schema Registry on Kafka
                   — messages outlive deploys; registry is the contract
Analytics at rest: Parquet/ORC — columnar; schema evolution rules of
                   the TABLE FORMAT (Iceberg/Delta) apply on top
Storage engines:   internal record formats (slotted pages, LSM entries)
                   — see B-Trees / LSM chapters; WAL records must be
                   versioned too (recovery reads old-format records
                   after an upgrade!)
Config:            YAML/JSON — human-edited, schema-validated in CI
```

再発するアンチパターン: 高ファンアウトの内部サービス間JSON（パースコスト × 10ホップは実際のp99に効く）。独自バイナリフォーマット（「ドキュメントは後で」— 進化・ツーリング・デバッグを永遠に自分で所有することになる）。スキーマバージョンを添付しないデータベースカラム内のprotobufブロブ（4000万行目をデコードするのはどの `.proto` ？）。あらゆるフォーマットでのrequiredフィールド最大主義。

---

## 障害モード

**タグ/フィールド番号の再利用（Protobuf）。** `string nickname = 5;` を削除し、後から `int64 team_id = 5;` を追加すると、古いメッセージのnicknameのバイト列がチームIDとしてデコードされます — エラーなし、ゴミデータ。`reserved` が存在する理由であり、スキーマレビューがタグ番号を追加専用として扱うべき理由です。`buf breaking` は機械的にこれを捕捉します。

**JSONの64ビット整数。** 2⁵³を超えうるIDはすべて文字列として運ぶこと。障害は、doubleにパースするどこかのコンシューマでの無音の丸めです — 「2人の別ユーザーが同じIDになった」として数週間後に発見されがちです。

**デフォルトなしのAvroフィールド。** プロデューサチームがデフォルトなしのフィールドを追加し、互換性NONE（またはレジストリなし）でプッシュ: 古いreaderスキーマを持つすべてのコンシューマが解決時に例外を投げます。これを不可能にするのがレジストリの仕事のすべてです。1世代を超えて不可能に保つためにTRANSITIVEモードを使ってください。

**データレイクのスキーマドリフト。** JSONをレイクにストリームしてバッチごとにスキーマ推論すると、型が行ったり来たりするカラムができます（`user_id`: 月曜のファイルではint、火曜ではstring）。数か月後、混ざった履歴の上でクエリが失敗します。修正: schema-on-write（レジストリでインジェスト時に強制）と、カラムの同一性と型昇格を所有するテーブルフォーマット（Iceberg/Delta）。

**WAL/スナップショットのフォーマットアップグレード。** ストレージエンジンとステートフルサービスはアップグレード後に*自分自身の*古いフォーマットを読まなければなりません — リカバリは前のバイナリが書いたレコードをリプレイします（[WAL](./04-write-ahead-logging.md)）。すべての永続レコードヘッダにバージョンを付け、新規状態だけでなくN−1・N−2が書いたデータでアップグレードパスをテストしてください。

**圧縮との混同。** エンコーディングと圧縮は別のレイヤです: Protobufはコンパクトですが、zstdでさらに約2倍圧縮されます。JSON+zstdは高いCPUコストでprotobufサイズに近づけます。「バイナリが必要だ」とも「gzipで十分だ」とも結論する前に、エンドツーエンド（CPU+バイト+レイテンシ）で計測を。

---

## 意思決定フレームワーク

| 要件 | 選択 |
|-------------|--------|
| 公開/外部API | JSON（RESTまたはGraphQL）。64ビットIDは文字列で |
| 内部サービスRPC | Protobuf + gRPC、CIで `buf breaking` |
| Kafka / イベントストリーム | AvroまたはProtobuf + Schema Registry、FULL_TRANSITIVE |
| 分析ストレージ | Iceberg/Delta配下のParquet/ORC — 一度エンコードし、カラムナで |
| 長寿命の保存ブロブ | スキーマバージョンヘッダを埋め込んだものなら何でも |
| ホットなIPC / mmap、少数フィールドのみ参照 | FlatBuffers / Cap'n Proto |
| スキーマレスでJSONより小さく | CBORまたはMessagePack（JSONの契約ギャップを受け入れる） |
| 人間が編集 | YAML/JSON + CIでのスキーマ検証。マシン間には決して使わない |
| ペイロードの署名/ハッシュ | まず正規形（JSONならJCS）にするか、生バイトに署名 |

---

## 重要なポイント

1. **エンコーディングは第一に互換性の問題、第二にパフォーマンスの問題** — データはコードより長生きし、ローリングデプロイは「古が新を読む」「新が古を読む」を移行イベントではなく恒常的な運用条件にする。
2. **ワイヤ上の同一性はフォーマットごとに異なる**: Protobuf = タグ番号（再利用禁止。名前は自由）、Avro = スキーマ解決による名前（デフォルト必須）、JSON = 最も脆いコンシューマがやること次第。
3. **全部optional、どこでもデフォルト** — ワイヤレベルの `required` は、成熟したスキーマ文化がすべて閉じた一方通行のドア。バリデーションはアプリケーションコードで。
4. **進化を機械的に強制する** — TRANSITIVE互換性付きスキーマレジストリ、またはCIでの `buf breaking`。破壊的変更はコンシューマチームを呼び出すのではなくビルドを失敗させるべき。
5. **JSONの数値はdouble** — 64ビットIDは文字列で運ぶ。さもなければ無音で壊れる。
6. **スキーマ付きバイナリはJSONの5〜10倍速く3倍小さい**。それが効くのはパイプラインとストレージのスケールであり、50 req/sのサービスではない。
7. **ゼロコピーフォーマットはサイズと使い勝手をパース時間ゼロと交換する** — 少数フィールドの読み取り重視アクセスには正しく、APIのデフォルトには誤り。
8. **ストレージフォーマットにもバージョンを** — WAL、スナップショット、DBカラムのブロブは将来のバイナリに読まれる。すべてのレコードにスキーマ/バージョンヘッダを。

---

## 参考文献

- Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 4 "Encoding and Evolution" — 決定版の扱い。
- Protocol Buffers documentation: *Encoding* (varint/wire format) and *Proto Best Practices* (reserved, field numbering).
- Apache Avro specification: *Schema Resolution*.
- Confluent Schema Registry documentation: *Compatibility Types* (incl. transitive modes).
- `buf` documentation: *Breaking Change Detection*.
- RFC 8785: *JSON Canonicalization Scheme (JCS)*; RFC 8949: *CBOR*.
- FlatBuffers and Cap'n Proto design documents (zero-copy layouts).
