# ブルームフィルタ

> この記事は英語版から翻訳されました。最新版は[英語版](/03-storage-engines/05-bloom-filters.md)をご覧ください。コードブロック・数式・図は原文のまま維持しています。

## TL;DR

ブルームフィルタは、片側誤差（one-sided error）を持つ集合メンバーシップ判定の仕組みです。「確実に存在しない」または「おそらく存在する」と答え、偽陰性は決して発生しません。偽陽性率1%あたり要素ごとに約10ビット — 情報理論上の下限は約6.6ビット — で実現できるため、LSMツリー型ストレージエンジンのほぼすべてのディスク読み取りの手前に置かれています。しかし教科書的な説明（「ビット配列とk個のハッシュ」）は、本番環境で本当に重要な部分を隠しています。偽陽性率は「予算」であり、フィルタが設計容量を超えて詰め込まれると急激に劣化します（100万キー用に設計したフィルタに500万キーを入れると偽陽性率は5%ではなく約83%になります）。k回のプローブは、ブロック化レイアウトを使わない限りk回のランダムなDRAMキャッシュミスです。LSMの全レベルに一律のbits-per-keyを割り当てるのは証明可能に誤った配分です（Monkey）。そして不変データに対しては、ブルームフィルタはもはや最良のフィルタではありません — RibbonフィルタやBinary Fuseフィルタはエントロピー下限に20〜30%近づきます。本章では数学を正直に導出し、フィルタが真価を発揮するLSM読み取りパスを歩き、モダンなフィルタの系譜と、フィルタが静かに壊れるパターンを扱います。

---

## フィルタが存在する理由: 高くつくのは「存在しない」ルックアップ

B-tree は「キーが存在しない」ことを安価に証明できます。ルートからリーフへの1回の降下（大部分はキャッシュ済み）で、リーフが不在を証明します。LSMツリー（[LSMツリー](./02-lsm-trees.md)）にはそれができません — キーは*どの*ランにも存在しうるため、不在の証明にはすべての候補SSTableの確認が必要です。

典型的なレベル型LSMで候補数を数えてみましょう:

```
Leveled compaction, fanout 10, 7 levels:
  L0: up to 4 overlapping files   → 4 candidate runs
  L1–L6: one sorted run each      → 6 candidate runs
  Total: ~10 runs may contain any given key

Point lookup for a MISSING key, no filters:
  10 runs × (index block + data block read) ≈ 10–20 block reads
  At ~100μs per cached-miss NVMe read: 1–2ms to learn "not found"

Same lookup with a 1% FPR filter per run:
  Expected disk reads = 10 runs × 0.01 = 0.1 reads
  → 100–200× reduction in read amplification for negative lookups
```

そして、「存在しない」ルックアップ（negative lookup）は直感より多くのワークロードを支配しています。重複排除（「このイベントIDは見たことがあるか？」）、insert-if-absent や一意性チェック、キャッシュフィルのプローブ、キーが存在*しない*ことを確認してから作成する書き込みパス。これらのワークロードでは「ミス」こそが共通ケースであり、フィルタは共通ケースを「全レベルを読む」から「何も読まない」に変換します。

保持すべきフレーミングはこれです: **ブルームフィルタは「ディスクに行く価値はあるか？」という問いに対する前払いの答え**です。それ以外のすべて — サイジング、ハッシング、レイアウト — は、その前払いの答えがいくらかかるか、どのくらいの頻度で嘘をつくかの話です。

---

## 構造と片側誤差

ブルームフィルタは `m` ビットのビット配列と `k` 個のハッシュ関数です。挿入は `k` ビットをセットし、クエリは `k` ビットを確認します。

```
Insert "hello":                    Query "world":
  h₁("hello") = 3                    h₁("world") = 5 → bit 5 is 0 ✗
  h₂("hello") = 7                    → "definitely not present"
  h₃("hello") = 12                   (stop at first zero bit)
  set bits 3, 7, 12
                                   Query "hello":
┌─────────────────────────────┐      all of bits 3, 7, 12 are 1
│ 0 0 0 1 0 0 0 1 0 0 0 0 1 0 │      → "probably present"
└─────────────────────────────┘
        ↑       ↑         ↑
        3       7         12
```

誤差の非対称性は偶然ではなく構造的なものです:

- **偽陰性なし**: ビットはセットされるだけで、クリアされることはありません。挿入された要素のビットは永遠に1なので、その要素へのクエリが0を見ることはありえません。（これが標準ブルームフィルタが削除をサポートできない理由でもあります — ビットをクリアすると、同じ位置にハッシュされた*別の*要素の証拠を消してしまい、偽陰性を製造する恐れがあります。）
- **偽陽性**: 一度も挿入されていない要素の `k` ビットすべてが、他の要素の挿入の合成によって既にセットされていることがあります。フィルタは「おそらく存在する」と言い、ディスク読み取りが発生し、何も見つかりません。これは誤った答えではなく無駄な仕事です — ただし、周囲のシステムが「おそらく」を権威ではなくヒントとして扱っている限りにおいて（[障害モード](#障害モード)参照）。

---

## 数学を正直に導出する

### 偽陽性確率

`n` 個の要素を `k` 個のハッシュで `m` ビットに挿入した後、特定のビットがまだ0である確率:

```
P(bit = 0) = (1 - 1/m)^(kn) ≈ e^(-kn/m)
```

偽陽性には、プローブされた `k` ビットすべてが1である必要があります:

```
p ≈ (1 - e^(-kn/m))^k
```

教科書が省略する2つの正直な注意点。第一に、この式はビットの占有を独立として扱いますが実際は独立ではなく、正確な偽陽性率（Bose et al., 2008）はこの式よりわずかに*高く*なります。ただし現実的なサイズのフィルタ（mが数百万以上）では差は無視できます。第二に、この式はk個のハッシュ関数が真に一様かつ独立であることを仮定しています — 実際のフィルタはこれを近似しており（[ハッシング](#ハッシング-実装が実際に壊れる場所)参照）、近似が悪いと実際の偽陽性率が理論値を上回る形で現れます。

### 最適なk、そして「最適」の姿

`m/n` を固定して `p` を `k` について最小化すると:

```
k* = (m/n) × ln 2  ≈ 0.693 × bits-per-key

At the optimum:
  - exactly half the bits are set (load factor 1/2 — maximum entropy per bit)
  - p = 2^(-k*) = 0.6185^(m/n)
```

逆算すると、目標偽陽性率 `p` を達成するには:

```
m/n = log₂(1/p) / ln 2 = 1.44 × log₂(1/p) bits per key

p = 1%    → 1.44 × 6.64 ≈  9.6 bits/key, k = 7
p = 0.1%  → 1.44 × 9.97 ≈ 14.4 bits/key, k = 10
p = 0.01% → 1.44 × 13.3 ≈ 19.2 bits/key, k = 13
```

要素ごとに約4.8ビット追加するごとに、偽陽性率が1桁改善します。この式に*含まれない*ものに注目してください: 要素のサイズです。64バイトのキー用のフィルタも4KBのドキュメント用のフィルタも、1%なら同じ9.6ビット/要素です — フィルタが格納するのはハッシュの痕跡であってデータではありません。

### 44%の税金、そしてポストブルームフィルタが存在する理由

情報理論は下限を定めます: 偽陽性率 `p` でメンバーシップに答えるいかなる構造も、要素あたり最低 `log₂(1/p)` ビットを必要とします。ブルームの `1.44 × log₂(1/p)` は従って**最適値より44%上** — 1970年に設計され、ビット配列以外何も必要としないほど単純な構造の代償です。後述するモダンなフィルタ（cuckoo、ribbon、xor/binary fuse）は、可変性か構築速度のどちらかを差し出す代わりに、この44%を取り戻すために存在します:

```
Bits per key at ~1% FPR (log₂(1/p) = 6.64 → theoretical floor 6.64 bits):

  Standard Bloom       9.6 bits   (1.44× floor)   dynamic inserts
  Blocked Bloom       ~10.5 bits  (1.6× floor)    dynamic, 1 cache miss
  Cuckoo filter       ~10.1 bits  (1.5× floor)    dynamic + deletion
  Ribbon filter       ~7.0 bits   (1.05–1.1×)     static, slow build
  Binary fuse filter  ~7.5 bits   (1.13×)         static, fast build
```

### 過充填カーブ: 偽陽性率は穏やかにではなく残酷に劣化する

この数学の最も運用上重要な帰結: 偽陽性率を決めるのは計画上の `n` ではなく*実際の* `n` です。100万キー・1%用に設計したフィルタ（m = 9.59Mビット、k = 7）に挿入を続けると:

```
actual n     kn/m     fraction of bits set     actual FPR
─────────────────────────────────────────────────────────
1.0M (as planned)  0.73        52%                 1.0%
1.5M               1.09        66%                 5.8%
2.0M               1.46        77%                16%
3.0M               2.19        89%                44%
5.0M               3.65        97%                83%
```

2倍の過充填で設定値の16倍悪化し、5倍ではゴミの83%を承認します — メモリを払った*上で*ディスク読み取りも払うことになります。この障害は無音です: 何のエラーも出ず、「確実に存在しない」が静かに「ディスクを確認しに行け」に変わるにつれてレイテンシがじわじわ上がるだけです。構造的な防御は2つ: 予想成長の1.5〜2倍でサイジングすること、そして長寿命のグローバルフィルタが設計点を超えて成長するアーキテクチャよりも、フィルタが不変アーティファクトごとに存在するアーキテクチャ（SSTableごとに1フィルタ、フラッシュ時に正確なキー数からサイジング — LSMエンジンが定常状態でこの問題に当たらない理由がこれです）を選ぶことです。

---

<a id="ハッシング-実装が実際に壊れる場所"></a>

## ハッシング: 実装が実際に壊れる場所

### ダブルハッシング: 2つの価格でk個のハッシュ

キーごとに7個の独立した高品質ハッシュを計算するのは無駄です。標準的なトリック（Kirsch & Mitzenmacher, 2006）は、2つのベースハッシュから `k` 個すべてのプローブを導出します:

```
h_i(x) = h₁(x) + i × h₂(x)   (mod m),  for i = 0..k-1
```

これは漸近的な偽陽性率を証明可能に保存します。実装上の罠が2つ:

- **h₂ = 0 の縮退**: `h₂(x) ≡ 0 (mod m)` の場合、k個のプローブすべてが同じビットに当たり、そのキーの実効kは1になります。（mが2のべき乗なら）`h₂` を奇数に強制するか、二次項 `+ i²` を追加します。
- **実用上は64ビットハッシュ1回で十分**: 上位・下位32ビットに分割してh₁とh₂にします。キーあたりxxHash/Murmur3の評価1回で済みます。

### ハッシュの選択

```
Good:  xxHash (fastest), MurmurHash3 (ubiquitous), CityHash/FarmHash
Bad:   MD5/SHA-256 — cryptographic strength buys nothing here, costs 10–20× cycles
       Language-default hashCode() — weak avalanche, correlated outputs,
         and often deliberately randomized per-process (breaks serialized filters!)
       h(x) mod m with structured keys — clustering
```

最後の括弧内は現実のバグクラスです: プロセスごとにシードされるハッシュ（Pythonの `hash()` など。Javaの `String.hashCode` は安定ですが、他の多くは安定ではありません）で構築したフィルタをシリアライズし、別のプロセスでロードすると、すべてのクエリが誤ったビットをプローブします — フィルタは自分が含んでいるキーに対して「確実に存在しない」を返します。**永続化されたフィルタは、ハッシュ関数・シード・ビットインデックス導出をシリアライゼーションフォーマットの一部として固定しなければなりません。**

### 32ビットハッシュは予想より早く機能しなくなる

32ビットハッシュでは、`m` が 2³² に近づくとビットインデックスが繰り返し始めます。しかし被害はもっと早く始まります: RocksDBのレガシーなblock-basedフィルタは単一の32ビットハッシュからプローブを導出しており、フィルタあたり数百万キーを超えると衝突構造が実際の偽陽性率に*床*を作りました — bits/keyを約14以上に増やしても、式が0.1%を約束する一方で実際の偽陽性率はほとんど改善しませんでした。モダンなfull/blockedフィルタ実装が64ビットハッシングに移行したのは、まさにこれを修正するためです。フィルタが100万キーを超える可能性があるなら、64ビットハッシュを使ってください。

### 敵対的な入力

公開された鍵なしハッシュを持つブルームフィルタはオラクルです: フィルタにクエリできる（または実装を知っているだけの）攻撃者は、プローブが既にセットされたビットに着地するキーを計算でき、無制限の偽陽性を製造できます — その一つひとつがディスク読み取り、つまりフィルタが守るはずだったまさにそのパスへの安価なリクエスト増幅攻撃です。入力が攻撃者に制御されうる場所（クローラのURL重複排除、ユーザーごとのレートリミットフィルタ、スパム既読フィルタ）では、2011年のhash-flooding攻撃後のハッシュテーブルと全く同じように、デプロイメントごとの秘密鍵を持つ鍵付きハッシュ（SipHash）を使ってください。

---

## メモリ階層: ブロック化ブルームフィルタ

教科書のコストモデルはブルームクエリをO(k)回の「操作」と言います。実際のコストモデルはキャッシュミスです。標準フィルタのk回のプローブは、数MB〜数GBのフィルタ全体に散らばるk個の独立したランダム位置です — プローブ1回がDRAMキャッシュミス1回です:

```
Standard bloom, k = 7, filter >> L3 cache:
  Negative query (probe until first 0): ~2 probes average → ~2 misses
  Positive/false-positive query:         all 7 probes     → ~7 misses
  At ~100ns per DRAM miss: 200–700ns per query

For comparison: the entire rest of a memtable lookup can be ~100–200ns.
The "free" in-memory check can dominate the in-memory path.
```

**ブロック化ブルームフィルタ**（Putze, Sanders & Singler, 2007）はこれを修正します: 最初のハッシュが64バイトのキャッシュラインサイズのブロックを1つ選択し、k個のビットすべてを*そのライン内で*セット/プローブします。

```
Query = exactly 1 cache miss, regardless of k.
Within the line, probing 7 bits is a handful of register ops —
and with SIMD, all probes resolve in a couple of instructions.

The price: keys are no longer spread evenly. Some blocks get
overloaded (Poisson variance across blocks), raising FPR.
Cost ≈ 0.5–1 extra bits/key to hit the same FPR as standard bloom.
```

これはエキゾチックな変種ではなく、本番エンジンが実際にデプロイしているものです。RocksDBの `format_version=5` フィルタ（RocksDB 6.6以降で利用可能）は、ブロック化されたSIMDフレンドリーな「fast local bloom」です — クエリあたり1キャッシュミスが、あらゆる読み取り重視ベンチマークで30%の空間節約に勝ったため、このレイアウトが選ばれました。モダンなストレージエンジンで「bloom filter」を見たら、ブロック化されていると考えてください。

---

## LSM読み取りパスの内側

ここがフィルタがメモリ代を稼ぐ場所です。RocksDBの語彙を使ったメカニクス（CassandraとHBaseはノブが違うだけで構造は同じです）:

### フィルタの住処と確認のタイミング

```
Point lookup Get(key):
  1. memtable        — optional memtable bloom (prefix-based) skips probing
  2. immutable memtables
  3. L0 files, newest first — each: check filter block → maybe read data
  4. L1..Lmax        — binary search file boundaries, one candidate file
                       per level: check filter block → maybe read data

Filter blocks are stored per-SSTable (in the table's metadata),
fetched through the block cache like any other block, and — if
cache_index_and_filter_blocks=true — subject to eviction like any
other block. For very large SSTables, partitioned filters split the
filter into a two-level structure so only the needed partition loads.
```

内面化する価値のある帰結が2つ。第一に、ルックアップの*集計*偽陽性コストは**ランごとの偽陽性率の合計**です — 各1%のランが10個あれば、存在しないキーのルックアップの約10%はどこかでディスクに触れます。第二に、ブロックキャッシュから追い出されたフィルタブロックは、ディスク読み取りを節約できるようになる*前に*ディスクから読まれなければなりません。メモリ圧の下ではフィルタは純粋なオーバーヘッドに反転しえます（障害モード参照）。

### 全キー vs プレフィックスフィルタ — そしてレンジスキャンが何も得られない理由

ブルームフィルタは正確なハッシュ値についての質問にしか答えられません。レンジスキャン `[a, b)` は連続体について尋ねます — 有限個の完全一致プローブでは覆えないため、**標準ブルームフィルタはレンジクエリには無力**です（Cassandraがスキャン専用テーブルで `bloom_filter_fp_chance = 1.0` によるフィルタ無効化を許しているのは、まさにこの理由です）。

部分的な例外が**プレフィックスブルームフィルタ**です。プレフィックス抽出器を定義し（例: 先頭8バイト = ユーザーID）、プレフィックスに対してフィルタを構築すると、*1つのプレフィックス内での*イテレータシーク（「ユーザーXの全イベント」）がフィルタを参照できます。RocksDBはこれを `prefix_extractor` とmemtableブルーム（`memtable_prefix_bloom_size_ratio`）として公開しており、MyRocksはインデックスプレフィックススキャンで多用しています。トレードオフ: イテレータはプレフィックス境界を越えないと約束しなければならず（`prefix_same_as_start`）、さもなければ結果は静かに間違います — フィルタによるスキップは、クエリが本当にフィルタされたドメイン内に留まる場合にのみ健全です。真のレンジフィルタリングには新しい構造（SuRF、Rosetta、レンジ分割されたfence pointer）が存在しますが、まだどれもデフォルトにはなっていません。

### Monkey: 一律のbits-per-keyは誤った配分

あらゆるエンジンのデフォルト — 「すべてのSSTableに10 bits/key」 — は証明可能に準最適です。洞察（Dayan, Athanassoulis & Idreos, *Monkey*, SIGMOD 2017）: ルックアップコストは**ラン横断の偽陽性率の合計**である一方、`n` キーのランで偽陽性率 `p` を達成するメモリコストは `∝ n·ln(1/p)` です。固定メモリ予算の下で期待ディスク読み取り総数を最小化すると、偏った配分が得られます:

```
With fanout 10, the last level holds ~90% of all keys.
Uniform 10 bits/key spends ~90% of filter memory on that one level.

Monkey's optimal allocation: give SMALLER (upper) levels exponentially
LOWER FPRs — they're cheap to filter aggressively because they hold
few keys — and let the largest level run at a higher FPR.

Result: same total memory, sum-of-FPRs (expected wasted reads for a
missing key) shrinks; Monkey reports the same lookup latency with
~2× less filter memory, or up to ~2× lower lookup cost at equal memory.
```

一般原則はLSMツリーを超えて転用できます: 1つのフィルタ群がサイズもプローブ頻度も異なる複数の階層を守っているとき、偽陽性予算は1ビットのメモリが最も多くの回避作業を買える場所に配分すべきであり、決して一律にではありません。

---

## ブルームを超えて: モダンなフィルタの系譜

### Counting Bloomフィルタ — 4倍の価格での削除

各ビットを小さなカウンタ（通常4ビット）に置き換えます: 挿入はインクリメント、削除はデクリメント、クエリは全カウンタ > 0 を確認します。動作はしますが: メモリは4倍、カウンタは15で飽和しうる（飽和後のデクリメントは偽陰性のリスクがあるため、飽和カウンタは固着させるしかない — 緩やかな偽陽性率のリーク）、そして実際にはcuckooフィルタがあらゆる軸で優越します。Counting bloomは主に古いネットワーク機器と論文の中に生き残っています。

### Cuckooフィルタ — 正しくやる削除

Cuckooフィルタ（Fan, Andersen, Kaminsky & Mitzenmacher, 2014）は、partial-key cuckoo hashingを使い、4-wayバケット化ハッシュテーブルに小さな**フィンガープリント**（ハッシュ断片、例: 8〜12ビット）を格納します:

```
Each key has two candidate buckets:
  b₁ = hash(x)
  b₂ = b₁ XOR hash(fingerprint(x))     ← computable from b₁ + fingerprint
                                          alone, enabling eviction without x

Insert: place fingerprint in b₁ or b₂; if both full, evict a resident
        fingerprint to ITS alternate bucket, repeat (cuckoo hashing).
Query:  check both buckets for the fingerprint — exactly 2 cache misses.
Delete: remove one matching fingerprint copy. Sound, with one caveat:
        inserting the same key twice then deleting once leaves it present;
        deleting a never-inserted key can evict a colliding victim.
        Deletion is only safe if inserts/deletes are balanced by protocol.

Space: bits/key ≈ (log₂(1/p) + 3) / α,  α ≈ 0.95 at 4-way buckets
  p = 1%  → (6.64 + 3)/0.95 ≈ 10.1 bits/key   (≈ bloom)
  p = 0.1% → (9.97 + 3)/0.95 ≈ 13.6 bits/key  (beats bloom's 14.4)
```

経験則: 目標偽陽性率が約3%以下なら、cuckooはブルームと同等以上にコンパクトで、*かつ*削除と2キャッシュミスのクエリが手に入ります。弱点: 満杯に近づくと挿入が完全に失敗しうる（余裕を持ってサイジングし、失敗を処理する必要がある）、そして負荷が上限に近づくと挿入コストが跳ね上がります。

### Quotientフィルタ — マージとリサイズに優しい

`fingerprint = quotient‖remainder` を格納します: quotientがスロットを指定し、remainderがそこに格納され、衝突はスロットあたり3つのメタデータビットでランを再構築するlinear probingで解決します。すべてが連続メモリに収まり（キャッシュフレンドリー）、削除をサポートし、**元のキーを再ハッシュせずにリサイズでき**（フィンガープリントが十分な情報を運ぶ）、2つのフィルタは**ソート済みリストのマージのように合流します** — LSMコンパクションに魅力的です。空間はブルームより約10〜25%多く、実装は本当に厄介です。メインストリームのエンジンより研究システム（例: ゲノミクスのcounting quotient filter）で使われています。

### Ribbonフィルタ — RocksDBの静的な空間節約策

SSTableでは、キー集合は構築時点で凍結されます — *静的*フィルタはそれを活用できます。Ribbonフィルタ（RocksDB ≥ 6.15、`NewRibbonFilterPolicy`）は「キーiのプローブのXORが期待値に等しい」をGF(2)上の連立一次方程式として扱い、構築時に解きます:

```
  ~30% less space than blocked bloom at equal FPR (~1.05–1.1× the
   entropy floor), same query speed, but 3–4× slower to BUILD.

RocksDB's guidance: use ribbon for lower levels (built rarely, by
background compaction, hold most keys → memory savings dominate) and
bloom for L0/high levels (built on every flush → build speed dominates).
That per-level policy split is Monkey-style thinking applied to filter
*type*, not just filter *budget*.
```

### XorフィルタとBinary Fuseフィルタ — 静的フィルタの最先端

Ribbonと同じ静的な体制で、異なる構築法（ハイパーグラフのpeeling）: 各キーは3つの位置にマップされ、そこに格納された値のXORがキーのフィンガープリントに等しくなります。Xorフィルタ（Graf & Lemire, 2020）は `1.23 × log₂(1/p)` bits/keyを達成し、**Binary Fuseフィルタ**（2022）はこれを約1.13倍まで詰め、構築時間もほぼ線形です — 8ビットフィンガープリントで p ≈ 0.4% を約9 bits/key、同じ偽陽性率でブルームは約11.5ビット必要なので約20%小さくなります。挿入も削除もなし: 完全なキーリストから一度だけ構築します。出荷されるアーティファクトに理想的です — コンパイル済みブロックリスト、CDNの「このオブジェクトはオリジンに存在するか」マップ、マルウェアシグネチャ集合 — 集合が変異ではなくバージョン付きで丸ごと置き換えられる場所ならどこでも。

### 選び方

| フィルタ | 可変性 | 空間 @ ~1% | クエリコスト | 使うべき場面 |
|---|---|---|---|---|
| Blocked Bloom | 挿入のみ | ~10.5 bits/key | キャッシュミス1回 | 動的集合のデフォルト。L0/フラッシュパスのSSTableフィルタ |
| Standard Bloom | 挿入のみ | 9.6 bits/key | 最大kミス | 単純さがレイテンシに勝る場合のみ（キャッシュに収まる小さなフィルタ） |
| Cuckoo | 挿入 + 削除 | ~10.1 bits/key | キャッシュミス2回 | 削除が必要（キャッシュ、出入りのあるメンバーシップ）。FPR ≤ 3% |
| Quotient | 挿入 + 削除 + マージ + リサイズ | ~11–12 bits/key | 局所性の良いプローブラン1回 | マージ（コンパクションパイプライン）やin-place成長が必要な場合 |
| Ribbon | 静的 | ~7 bits/key | ~キャッシュミス1回 | 不変で構築時間に寛容: LSMの深いレベル |
| Binary fuse / xor | 静的 | ~7.5 bits/key | 3ミス（fuse: ~1–2） | 不変で構築速度に敏感: 出荷/バージョン付き集合 |

---

## ストレージエンジンを超えたブルームフィルタ

同じ「前払いの否定的な答え」パターンは、安価なチェックが高価な操作を拒否できる場所ならどこでも再登場します:

**分散結合とクエリエンジン。** ビルド側が小さくプローブ側が巨大なスキャンであるハッシュ結合では、エンジンはビルド側の結合キーに対してブルームフィルタを構築し、*プローブ側のスキャンに押し下げます*（Sparkのruntime filter、Snowflake/BigQueryのsemi-join reduction、Parquetのrow-groupスキップ）。フィルタを通らないプローブ行はスキャンオペレータから出ることすらありません — シャッフル/スキャンされるデータの90%超を排除することもしばしばです。ネットワーク越しでも同じ発想: コンパクトなセミ結合として、テーブルではなくフィルタを送るのです。

**CDNキャッシング: one-hit-wonder問題。** Akamaiの計測では、リクエストされたURLの約74%は数日間のウィンドウで正確に1回しかリクエストされません（Maggs & Sitaraman, 2015）。最初のリクエストでオブジェクトをキャッシュすると、ディスク書き込みの大半は二度と読まれないオブジェクトのためになります。解決策: 最近見たURLのブルームフィルタを持ち、*2回目の*リクエスト（フィルタヒット）でのみキャッシュします。この「アドミッションポリシーとしてのブルームフィルタ」はディスク書き込みをほぼ半減させ、ヒット率を改善しました — このフィルタは読み取りをまったく守っておらず、*書き込み*を守っているのです。

**Bitcoin SPV — 教訓的な物語（BIP-37 → BIP-158）。** ライトクライアントはかつて自分のアドレスのブルームフィルタをフルノードに送り、マッチするトランザクションを受け取っていました。偽陽性がもっともらしい否認可能性というプライバシーを提供する*はず*でした。これは失敗しました: 複数のフィルタとセッションを横断すると、「maybe」集合の積集合がウォレットをほぼ完全に非匿名化し、フィルタ付きスキャンの提供は安価なDoSを可能にしました。後継（BIP-158）は方向を反転させます — *サーバー*がブロックごとのコンパクトなフィルタ（Golomb-coded set。ブルームフィルタとほぼ同じ仕事を一回限りの転送用に最適圧縮したもの）を公開し、クライアントはローカルでマッチした完全なブロックをダウンロードするので、何も漏らしません。教訓: 確率的フィルタは繰り返しの観察の下でその内容を漏らします。フィルタはパフォーマンス構造であり、プライバシー機構ではありません。

**回転フィルタによるストリーム重複排除。** 「ほぼexactly-once」のイベントパイプライン（[配信保証](../05-messaging/04-delivery-guarantees.md)）はしばしば「このイベントIDを直近1時間で見たか？」を必要とします。単一のフィルタは過充填します（上述の過充填カーブ参照）。標準パターンはN個の時間バケット化フィルタ（例: 10分×6個）で、クエリは全部に、挿入は最新のものに行い、最古のものを丸ごと捨てます — 変異による削除ではなく、引退による削除です。

**同じ形の他の例:** ウェブクロールのフロンティア重複排除（数十億のURL。偽陽性 = 不必要にスキップされるページ — 許容可能）。Squidのcache digest（ピア同士が自分のキャッシュ内容のブルームフィルタを交換してリクエストをルーティング）。[セカンダリインデックスのscatter-gatherの刈り込み](../02-distributed-databases/06-secondary-indexes.md)（フィルタが「maybe」と言うパーティションだけにクエリ）。

---

<a id="障害モード"></a>

## 障害モード

**無音の過充填。** 上で扱いましたが、これが現実世界の障害の第1位です: エラーもログ行もなく、`n` が設計点を超えるにつれ偽陽性率が1%から80%へ漂うだけです。検出には実効偽陽性率の*計測*が必要です: `false positives / (false positives + true negatives)`。ここで偽陽性 = フィルタがmaybeと言い、ディスクがnoと言ったケース。RocksDBはまさにこれを `rocksdb.bloom.filter.full.positive` と `.full.true.positive` として、Cassandraは `BloomFilterFalseRatio` として公開しています。計測された偽陽性率が設定値の約2倍を超えたらアラートを出してください。

**メモリ圧下でのフィルタブロックの追い出し。** `cache_index_and_filter_blocks=true` のRocksDBでは、フィルタブロックはデータブロックとキャッシュを奪い合います。圧力の下でキャッシュがフィルタを追い出すと、次のルックアップはデータブロック読み取りを（もしかしたら）節約できるようになる前にフィルタを*ディスクから*読まなければなりません — 「存在しない」ルックアップが、フィルタなしより遅くなります。間欠的に、しかもまさに重要な過負荷の瞬間に。緩和策: `pin_l0_filter_and_index_blocks_in_cache`、`cache_index_and_filter_blocks_with_high_priority`、あるいはフィルタをブロックキャッシュの外で予算化し、`filter block read` 系のティッカーを監視します。

**マージまたはシリアライゼーションの不一致 → 偽陰性。** 2つのフィルタのビット配列のORによるマージは、`m`・`k`・ハッシュ関数・シードが*すべて*同一の場合にのみ可能です。不一致は、自信満々に誤った「確実に存在しない」を返す構造を生みます — ブルームフィルタが決してついてはならない唯一の嘘のクラスです。同じ障害はシリアライゼーション経由でもやってきます: ハッシュシードが異なるランタイム（プロセスごとにランダム化されるハッシュ、エンディアンネス、ビットインデックス導出を変えたライブラリのアップグレード）へのフィルタのロード。フィルタフォーマットをバージョン管理し、ハッシュIDとシードを埋め込み、不一致ならロードを拒否してください。

**「maybe」を「yes」として扱う。** フィルタだけを裏付けとするユーザー名可用性チェックは、*利用可能な*名前の約p%を拒否します（偽陽性 → 「使用済み」）— テストが衝突する名前に当たることは稀なので、静かに出荷されてしまう可用性バグです。フィルタは常に権威ある確認の前の高速パスであるべきで、決して権威そのものであってはなりません。双対の障害: counting/cuckooフィルタの削除を不均衡なプロトコルトラフィック（挿入なしの削除）に対して使うこと。これは偽陰性を製造し、こちらは*本当に*正しさを壊します。

**ホットな偽陽性。** たまたま偽陽性になった人気の存在しないキー1つが、100%の確率でディスク読み取りが発生するホットスポットに変わります（フィルタはすべてのクエリで同じ嘘を繰り返します — プローブは決定的です）。「存在しないキー」が問題になるほどホットなら、フィルタの背後に小さなネガティブキャッシュ（正確なもの。例: 不在確認済みキーの小さなLRU）を重ねてください。

**弱いまたは誤用されたハッシング。** 相関したプローブビット（悪いハッシュ、`h₂=0` の縮退、大きなフィルタ上の32ビットハッシュ）は、bits/keyに関わらず偽陽性率に見えない床を作ります。鍵なしハッシュへの敵対的入力はフィルタを増幅ベクトルに変えます。どちらもハッシングの節で扱いました — どちらも本番では「bits/keyを2倍にしたのに偽陽性率が動かなかった」として現れます。

---

## 意思決定フレームワーク

| 状況 | やること |
|---|---|
| LSM/SSTableのポイントルックアップフィルタ、デフォルト | Blocked bloom 約10 bits/key（RocksDB `format_version≥5`）。メモリ制約下なら最下層にribbon |
| 読み取りパスがほぼ*存在する*キーのルックアップ | フィルタを縮小または省略 — フィルタはミスでしか稼がない。まず `useful / queries` を計測 |
| レンジスキャンのみでアクセスされるテーブル | フィルタを無効化（`bloom_filter_fp_chance = 1.0`）。スキャンがプレフィックス境界内ならprefix bloomのみ検討 |
| 出入りのある削除が必要 | 5%以上のロード余裕を持つcuckooフィルタ。プロトコルが未挿入キーを削除しないことを保証 |
| 多数のノードに出荷される不変・バージョン付き集合 | Binary fuse（高速構築）またはribbon（最大圧縮）。バージョンごとに再構築 |
| 時間ウィンドウ上のストリーミング重複排除 | 回転バケット化ブルーム。フィルタ丸ごと引退させ、個別削除はしない |
| 集合が無制限に成長 / nが不明 | シール時にサイジングされるアーティファクトごとのフィルタ（LSM流）、またはscalable bloom（pを絞りながら連鎖するフィルタ群）。グローバルフィルタ1つは決して使わない |
| 攻撃者が影響しうるキー | デプロイメントごとの秘密鍵を持つ鍵付きハッシュ（SipHash） |
| 多階層にまたがるフィルタメモリ | Monkey流の傾斜配分: ランが小さく頻繁にプローブされる場所に最低の偽陽性率を。一律のbits/keyにしない |
| 読み取りではなく書き込みを守る（アドミッション） | 最近見たキーのブルームフィルタ。2回目の出現で行動（one-hit-wonderパターン） |

---

## 重要なポイント

1. **フィルタは「ディスクに行く価値はあるか？」の答えを前払いする** — その価値は*存在しない*キーのルックアップ率に比例する。存在するキー中心のワークロードでは純粋なオーバーヘッド。
2. **サイジングは 1.44 × log₂(1/p) bits/key** — 1%なら9.6ビット、偽陽性率1桁改善ごとに+4.8ビット。要素サイズには依存しない。
3. **偽陽性率は過充填で残酷に劣化する** — 設計nの2倍で設定値の約16倍。本番では実効偽陽性率を計測し、設定値を信用しない。
4. **ハッシュ演算ではなくキャッシュミスを数える** — blocked bloom（1ミス）が本番のデフォルト。標準ブルームのk回の散在プローブはインメモリレイテンシを支配しうる。
5. **ブルームはエントロピー下限より44%上** — 静的集合（SSTable、出荷アーティファクト）ならribbonとbinary fuseがそれを取り戻す。
6. **フィルタはレンジを見られない** — レンジスキャンには無力。prefix bloomはプレフィックス境界内のイテレーションのみ助ける。
7. **m/k/ハッシュ/シードが不一致のままORマージやデシリアライズをしない** — それはフィルタが偽陰性という、決してついてはならない唯一の嘘を覚える経路。
8. **偽陽性予算は不均等に配分する**（Monkey）: ランが小さく頻繁にプローブされる場所にビットを使い、最大の階層は緩く走らせる。
9. **確率的 ≠ プライベート** — フィルタの繰り返し観察はその内容を再構築する（BIP-37）。フィルタはパフォーマンス構造であり、情報障壁ではない。

---

## 参考文献

- Bloom, B. H. (1970). *Space/Time Trade-offs in Hash Coding with Allowable Errors*. CACM. — 偽陽性を許す確率的集合表現を定式化した原論文。
- Broder, A., & Mitzenmacher, M. (2004). *Network Applications of Bloom Filters: A Survey*. Internet Mathematics. — ネットワーク用途と設計上の意味論を整理したサーベイ。
- Kirsch, A., & Mitzenmacher, M. (2006). *Less Hashing, Same Performance: Building a Better Bloom Filter*. ESA. — 2つの基底ハッシュから複数プローブを生成する手法の根拠。
- Putze, F., Sanders, P., & Singler, J. (2007). *Cache-, Hash- and Space-Efficient Bloom Filters*. WEA. — キャッシュ局所性を考慮したblocked構成の基礎。
- Bose, P., et al. (2008). *On the False-Positive Rate of Bloom Filters*. Information Processing Letters. — 有限サイズでの偽陽性率近似を補正する解析。
- Fan, B., Andersen, D., Kaminsky, M., & Mitzenmacher, M. (2014). *Cuckoo Filter: Practically Better Than Bloom*. CoNEXT. — 削除可能なフィルタの負荷率、挿入失敗、性能特性を評価。
- Maggs, B., & Sitaraman, R. (2015). *Algorithmic Nuggets in Content Delivery*. ACM SIGCOMM CCR. — one-hit wonderを除外するキャッシュアドミッションの事例。
- Dayan, N., Athanassoulis, M., & Idreos, S. (2017). *Monkey: Optimal Navigable Key-Value Store*. SIGMOD. — LSM階層ごとに偽陽性予算を不均等配分する設計。
- Graf, T. M., & Lemire, D. (2020). *Xor Filters: Faster and Smaller Than Bloom and Cuckoo Filters*. ACM JEA; および (2022) *Binary Fuse Filters*. — 静的集合向け構築法と空間・検索性能の比較。
- RocksDB Wiki: *RocksDB Bloom Filter* と *Ribbon Filter*. — `format_version=5` の高速ローカルBloomとRibbonの実装上の選択肢。
- BIP-37 (Connection Bloom Filtering) と BIP-158 (Compact Block Filters). — フィルタ公開によるプライバシー問題と、異なる問い合わせモデルを示すBitcoin改善提案。
