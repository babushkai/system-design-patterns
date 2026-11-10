# 分散学習の内部構造

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../16-ml-systems/15-distributed-training-internals.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

大規模モデルの学習は、MLの衣装をまとったスーパーコンピューティングの問題です。分散を強制する制約は2つ: モデルの学習状態が1つのアクセラレータに収まらないこと（70Bパラメータのモデルは重み・勾配・オプティマイザ状態だけで約1.1TB必要）、そして計算が一生のうちに終わらないこと（同じモデルを15Tトークンで学習すると約500 GPU年 — 8,000基のGPUを3週間回すか、まったく回さないかです）。本章のすべては、この2つをどう分割するかから導かれます: **データ並列**はモデルを複製して勾配を平均し（all-reduceとバッチサイズの天井が噛みつくまでは安価）、**ZeRO/FSDP**は学習状態をレプリカ間でシャードし、**テンソル並列**は個々の行列積を分割し（NVLink級の帯域が必要なのでノード内に留まる）、**パイプライン並列**はレイヤーをステージに分割します（マイクロバッチ数とともに縮むバブルを支払う）。実際のジョブはこれらすべてを合成し、その合成はクラスタのメモリ階層によって決まります — 好みではなく。その上に載る運用の現実: 同期ジョブは最も遅いワーカーの速度で動き、10,000+ GPUでは数時間ごとに*何かが*故障し（Llama 3のチームは54日間で466回の中断を記録）、すべてを要約するメトリクス — MFU、理論FLOPsに対する実際の達成率 — はよく運用された大規模ジョブで40%前後です。本章では各要素の算術を組み立てます: メモリ台帳、集合通信のコスト、バブル率、チェックポイント間隔、そして失われた60%の行き先。

この上のオーケストレーション層 — パイプライン、再学習、再現性 — は[学習パイプライン](./05-training-pipelines.md)、単一アクセラレータから推論時にFLOPsを引き出す兄弟問題は[GPU推論の内部構造](../17-llm-systems/11-gpu-inference-internals.md)、クラスタレベルのサイジングとコストは[MLキャパシティとコストプランニング](./14-ml-capacity-cost-planning.md)です。

---

## なぜ分散するか: 2つの台帳

### メモリ台帳

学習状態はモデルよりはるかに大きい。標準的な混合精度Adamでは、パラメータ1つごとに:

```
bf16 weights                     2 bytes
bf16 gradients                   2 bytes
fp32 master weights              4 bytes
fp32 Adam momentum (m)           4 bytes
fp32 Adam variance (v)           4 bytes
────────────────────────────────────────
                                16 bytes per parameter

  7B model:   112 GB  of state  — doesn't fit an 80 GB H100
 70B model:   1.1 TB            — doesn't fit a NODE of 8 H100s (640 GB)
405B model:   6.5 TB            — needs ≥ 82 GPUs before ONE token is processed
```

しかもこれは**アクティベーション** — backward パスで使うために forward パス中に保存される中間テンソル — を含んでいません。アクティベーションは `batch × sequence_length × hidden × layers` でスケールし、長いシーケンス長ではパラメータ状態を日常的に超えます。（標準的な緩和策である*アクティベーションチェックポインティング*は、大半のアクティベーションを破棄してbackward中に再計算します — 約30%多いFLOPsを支払ってアクティベーションメモリを数分の一に削る。メモリと計算は代替可能であり、大規模モデルの学習は常に一方を他方と交換しています。）

### 計算台帳

学習FLOPsは `6 × parameters × tokens` でよく近似できます（forward ≈ 2·P·T、backward ≈ forwardの2倍）:

```
70B params × 15T tokens × 6 ≈ 6.3 × 10²⁴ FLOPs

One H100: ~990 TFLOP/s peak (bf16, dense). At a realistic 40% MFU:
  ~4 × 10¹⁴ FLOP/s achieved
  6.3e24 / 4e14 ≈ 1.6 × 10¹⁰ GPU-seconds  ≈  500 GPU-YEARS

  On 8,192 GPUs: ~22 days.  On 8 GPUs: ~63 years.
```

2つの台帳が設計空間を具体化します: 大規模モデルの学習状態を*保持する*だけで数十のGPUが必要で、カレンダー時間内に学習するには数千が必要です。本章の残りは、その数千のGPUが互いを待つのではなく助け合うように、状態と仕事を分割する話です。

---

## データ並列: モデルを複製し、勾配を平均する

ベースライン戦略: すべてのワーカーが完全なモデルレプリカを持ち、各バッチの異なるスライスを処理し、ステップ前に勾配を平均します — これによりすべてのレプリカはビット単位で同一に保たれ、N個のワーカーは数学的にN倍大きいバッチを持つ1ワーカーと等価になります。

この平均は**all-reduce**であり、毎ステップ繰り返されるためそのコストは正確に知る価値があります。帯域最適なringアルゴリズムでは、N個のワーカーそれぞれが送受信するのは

```
2 × (N-1)/N × D  ≈  2D bytes        (D = gradient bytes, N large)

70B model, bf16 gradients: D = 140 GB → each GPU moves ~280 GB per step.
On a 50 GB/s effective inter-node link: ~5.6 seconds — likely LONGER
than the compute step itself.
```

これが致命傷にならないのは3つの機構のおかげです:

- **オーバーラップ**: 後方のレイヤーの勾配は、前方のレイヤーがまだbackward中に確定します。フレームワークは勾配をバケット化し、各バケットが完成するたびにall-reduceを発行して、通信を計算の陰に隠します。よくチューニングされたジョブは2Dの大半を隠し、そうでないジョブは計算→通信を直列化してスループットの3分の1をネットワークに失います。
- **勾配蓄積**: ローカルでk個のマイクロバッチを回し、1回だけ同期する。通信頻度を1/kにする代わりに実効バッチがk倍になります — それはより大きいバッチが*欲しかった*場合にのみ無料です（下記参照）。
- **階層的トポロジ**: まずノード内でNVLink（~900 GB/s）越しにreduceし、その後はるかに遅いファブリック越しにノード間で行う — ringのコストは最も遅いリンクで決まるので、ハードウェアに合わせてringを組みます。

### 誰も逃れられない天井: バッチサイズ

データ並列のスケーリング限界は、たいていネットワークではなく*最適化*です。N個のワーカーはN倍のグローバルバッチを意味し、モデル/データセット依存の**クリティカルバッチサイズ**を超えると、バッチを大きくしても収束に必要なステップ数が減らなくなります: 同じ学習の進捗のためにより多くのFLOPsを燃やすだけです。gradient noise scaleの研究（McCandlish et al.）がこれを形式化しています。実務上の症状は、1kから4k GPUへのスケールで各*ステップ*は4倍大きくなるのに*学習*は有意に速くならないことです。学習率スケーリングとウォームアップ（Goyal et al.の線形スケーリング則）は天井を押し上げますが、取り除きはしません。これが、完璧なネットワークがあっても純粋なデータ並列がいずれ尽きる理由であり、1デバイスに技術的には収まるモデルにも他の並列化が存在する理由です。

---

## ZeRO / FSDP: 状態をレプリカ間でシャードする

素朴なデータ並列はメモリに鈍感です: N個のレプリカが16 bytes/paramの同一コピーをN個持ちます。ZeRO（およびPyTorchのFSDP、同じアイデア）は3つの段階でこの冗長性を取り除きます:

```
Per-GPU memory (P params, N data-parallel workers):

  Plain DP:        16P                    all state replicated
  ZeRO-1:          4P  + 12P/N            optimizer state sharded
  ZeRO-2:          2P  + 14P/N            + gradients sharded
  ZeRO-3 / FSDP:         16P/N            + parameters sharded

  70B on 64 GPUs with ZeRO-3: 1.12 TB / 64 ≈ 17.5 GB per GPU — fits,
  with room for activations.
```

ステージ3のメカニクス: 各レイヤーのパラメータはシャードされて存在し、レイヤーの実行直前（forwardでも、backwardでも再度）にワーカーがそのレイヤーの重みを**all-gather**し、使い、即座に解放します。勾配は**reduce-scatter**され、各ワーカーは自分のシャードだけを保持します。通信は素のDPの2Dから約3Dに増えます — N倍のメモリ削減のために支払う1.5倍の税で、DPのall-reduceと同様にオーバーラップします: 現在のレイヤーを計算しながら次のレイヤーのall-gatherをプリフェッチします。

吸収すべき設計感覚: **ZeROはクラスタの合計メモリを1つのプールとして扱い、それがローカルであるかのように振る舞うために帯域を支払う。** データ並列と自明に合成でき（それ自体がメモリ最適化されたデータ並列*です*）、7B〜70Bレンジのモデルのデフォルトの答えです。限界は、モデルの*レイヤー*が依然としてすべてのGPUで実行されることです — 単一レイヤーのワーキングセットやアクティベーションのトラフィックがノードを超えたら、計算そのものを分割する並列化が必要になります。

---

## テンソル並列とパイプライン並列: 計算を分割する

### テンソル並列: 行列を分割する

テンソル並列（Megatron流）は個々の重み行列をGPU間で分割します — 各GPUが列または行のスライスを持ち、部分行列積を計算し、all-reduceが結果を組み立てます。残酷な性質: そのall-reduceは**レイヤーごと、マイクロバッチごと、クリティカルパス内で**発生します — DPの勾配同期のように他の計算の陰に隠すことができません。

```
Consequence: TP lives or dies on interconnect latency+bandwidth.
  NVLink within a node:   ~900 GB/s  → TP works
  InfiniBand across nodes: ~50 GB/s  → TP dies

Rule that follows: TP degree ≤ GPUs per node (8, typically).
TP is not a scaling strategy — it's a "make the layer fit and keep
per-GPU matmuls large" strategy, confined to one node.
```

### パイプライン並列: レイヤーを分割する

パイプライン並列は連続するレイヤーブロックを異なるノード上の*ステージ*に割り当てます。アクティベーションはステージからステージへ前方に流れ、勾配は後方に流れます。ステージ間トラフィックはマイクロバッチあたりアクティベーションテンソル1つだけ — TPのおしゃべりに比べれば微小 — なので、PPは遅いリンクを喜んで越える並列化です。

その税は**バブル**: パイプラインの充填と排出の間、ステージはアイドルになります。pステージ、バッチあたりmマイクロバッチのとき:

```
bubble fraction = (p − 1) / (m + p − 1)

  p=8,  m=8:    47% of the schedule is idle  — catastrophic
  p=8,  m=64:   ~10%                          — acceptable
  p=8,  m=256:  ~3%                           — good

So PP demands many microbatches — which is the same resource the
batch-size ceiling limits. Deep pipelines + modest global batch =
bubbles you cannot schedule away. (1F1B scheduling and interleaved
stages reduce peak activation memory and shave the bubble, but the
(p−1)/(m+p−1) shape is the invariant to reason with.)
```

### 合成: 3D並列

実際の大規模モデルのジョブは3つすべてを使い、その合成は好みではなくハードウェア階層に従います:

```
  TP  innermost — needs NVLink        → within the node, degree ≤ 8
  PP  next      — tolerates slow links → across nodes, until it fits
  DP  outermost — embarrassingly parallel across the rest (often with
                  ZeRO-1 sharding the optimizer inside each replica)

Example, 70B on 512 H100s (64 nodes):
  TP=8 (one node) × PP=4 (4 nodes = one model replica of 32 GPUs)
  × DP=16 replicas
  Per-GPU: 1/32 of the model's layers × 1/8 of each matrix. Fits with
  activation headroom; global batch = 16 × microbatches × micro size.
```

ノブは相互作用します: PPを上げるとメモリは楽になるがより多くのマイクロバッチ（バブル）を要求し、それはグローバルバッチを天井に向けて押し上げます。TPを上げるとGPUあたりの行列積が縮み、テンソルコアを飽和させられなくなります。最適点を見つけるのは午後いっぱいの算術と1日のプロファイリングであり、本当に価値があります — 公開された構成（Megatron-LM、Llama 3）は、あなたのクラスタのファブリックにとって正しい出発点であって、正しい答えではありません。

*(知っておくべき2つの兄弟: **シーケンス/コンテキスト並列**は長コンテキスト学習のためにシーケンス次元を分割し、**エキスパート並列**はMoEのエキスパートを異なるGPUに配置します — 設計論理は同じで、通信パターンはall-to-allです。)*

---

## 行列計算ではない部分

### 入力パイプラインはGPUより速くなければならない

学習ステップは `global_batch × seq_len` トークンを消費します。ストレージと前処理のパスは、それを毎ステップ、永遠に供給し続けなければなりません:

```
8,192 GPUs × ~50K tokens/s/GPU ≈ 4 × 10⁸ tokens/s ≈ several GB/s of
decompressed, tokenized, shuffled data — sustained for weeks.
```

これはストレージシステムの問題です（フォーマットとシャーディングは[学習パイプライン](./05-training-pipelines.md)が扱います）。分散学習に固有の要件は**決定性と再開可能性**です — すべてのワーカーは互いに素なシャードを、正確に再現可能な順序で見なければならず、チェックポイントから再開されたジョブは（消費済みサンプルをスキップして）*同じ*データ列を継続しなければなりません。再サンプリングしてはいけません。データ順序のバグはこの分野で最も厄介な部類です: 再現不能な損失曲線、無音のサンプル重複、eval汚染として、数週間後に姿を現します。

### ストラグラー: ワーカーに対するmax()

同期ステップは*最も遅い*参加者が終わったときに完了します。スケールが大きくなると、これは稀な遅さを恒常的な遅さに変換します:

```
One GPU has a 1-in-1000-steps slow event (thermal throttle, ECC retry,
noisy neighbor on the NIC, background daemon):
  1 GPU:      0.1% of steps affected
  8,192 GPUs: essentially EVERY step waits for someone's slow event —
  the fleet moves at its collective p99.9.
```

防御は地味かつ必須です: 弱いハードウェアを排除する事前バーンイン、ランクごとのステップ時間の厳密な監視（ヒストグラムの外れ値が*そのまま*ジョブの速度です）、1つの過剰契約スイッチがring全体を遅くしないためのトポロジ意識配置、そしてその場でデバッグする代わりに差し替えるホットスペア。非同期SGD — 2012年頃の答え — はこの問題を、古い勾配とより悪い収束と引き換えました。現代の実務は、同期学習+容赦ないストラグラー排除です。

### 故障の数学とチェックポインティング

コンポーネントの故障はポアソン過程で、フリートは巨大で、掛け算になります:

```
If one GPU fails on average every ~5 years:
  16,384 GPUs → a hardware failure every ~2.7 hours of wall clock.

Llama 3 405B (Meta, 2024): 54 days on 16K H100s, 466 job
interruptions — one every ~2.8 hours — 78% attributed to hardware
(GPUs, HBM, NICs, cables). This is the NORMAL operating regime.

Synchronous training means one failure stops all 16K GPUs. Recovery =
restore from last checkpoint. Expected loss per failure:
  (checkpoint_interval / 2) × fleet — plus restart time × fleet.

Optimal checkpoint interval (Young/Daly):
  τ* ≈ sqrt(2 × δ × MTBF)      δ = time to write a checkpoint
  δ = 5 min, MTBF = 2.7 h  →  τ* ≈ 52 min.
  But drive δ down to 30 s (async, sharded) → τ* ≈ 16 min, and the
  expected goodput loss per failure drops ~3×.
```

最後の行が、現代のチェックポインティングがそれ自体エンジニアリングのテーマである理由です: **シャード化**（各ランクが自分の状態スライスを並列に書く — 6.5TBの状態を16Kランクで書けば管理可能、rank 0に集約すれば障害）、**非同期**（ホストメモリに数秒でスナップショットし、学習を続けながらバックグラウンドで[オブジェクトストレージ](../03-storage-engines/08-object-storage.md)に排出）、そして増えつつある**ピア冗長**（失われたランクの状態をストレージではなく隣のメモリから復元）。チェックポイント頻度は、恐る恐る設定するダイヤルであることをやめ、安価な保険になります。

エンドツーエンドの健全性メトリクスは**goodput**です: 壁時計のGPU時間のうち、最終モデルに寄与したステップに使われた割合 — 再起動、やり直し、ストール、初期化を差し引いた後の。成熟した大規模ジョブのチームは、上記の地味な機構だけで数百万ドルに相当するgoodput改善（85→95%+）を報告しています。

---

## MFU: すべてを要約するひとつの数字

**Model FLOPs Utilization** — 達成された有効FLOPs（`6·P·tokens/sec`）をフリートの理論ピークで割ったもの — は正直なスコアボードで、バッチサイズのごまかしにもハードウェア世代の混同にも影響されません:

```
MFU = (6 × P × tokens_per_second) / (N_gpus × peak_FLOPs_per_gpu)

Reference points:
  50-57%   exceptional (dense LLM, tuned Megatron-class stack, PaLM/
           Llama-3-scale engineering)
  35-45%   good, typical for well-run large jobs
  20-30%   common in practice — something is eating a third of the fleet
  <20%     the job has a bug wearing a performance costume

Where the missing fraction goes (the audit order):
  1. data stalls        — input pipeline can't feed (check first, it's
                          the cheapest fix and the most common)
  2. communication      — unoverlapped all-reduce/all-gather time
  3. pipeline bubbles   — (p−1)/(m+p−1), by construction
  4. kernel inefficiency— small matmuls (TP too high), missing fused
                          kernels (FlashAttention et al.)
  5. stragglers/restarts— the max() tax and the failure tax (goodput)
```

MFUは交渉の通貨でもあります: 「GPUが2倍必要です」と「MFUを25%から40%に上げられます」は予算にとって同じ文であり、たいてい後者のほうが安い（[MLキャパシティとコストプランニング](./14-ml-capacity-cost-planning.md)）。

---

## 障害モード

**ハングに見えるNCCLストール。** 1つのランクが死ぬか1つのNICがフラップすると、他のすべてのランクが集合通信の中でブロックし、GPUは100%使用率に張り付いたまま何もしません。ウォッチドッグがなければ、人間が気づくまでジョブは無音でハングします。防御: 集合通信のタイムアウト（NCCLウォッチドッグ）、学習ループの*外側*にあるランクごとの生存ハートビート、チェックポイントからの自動再起動 — 約3時間ごとの故障では、復旧は反射でなければならず、ページで起こされてからでは遅い。

**GPU問題と誤診されるデータローダー飢餓。** ステップ時間が長く、GPU使用率グラフは忙しそうに見え（表示されるのは占有率であって有効仕事ではない）、チームは1週間カーネルをチューニングします。見分け方: 1ステップをプロファイルする。GPUが入力キューを待っているなら、修正はCPU数、デコード/トークナイズのスループット、ストレージ帯域にあります。項目#4の前に必ず項目#1を監査すること。

**再起動後の無音の発散。** 再開がデータ順序カーソル、LRスケジュール、RNG状態のどれかを誤って扱うと、損失曲線は*もっともらしく*見えるのに、そのランはもはやあなたが思っているランではありません — 数週間後、evalの時点で発覚します。再開はテストされたコードパスとして扱うこと: 小さなジョブを途中で再起動し、無中断のコントロールと損失曲線をビット単位で突き合わせる。

**スケール時の損失スパイクと悪い数値。** 大きなバッチでのbf16学習は時折スパイクします。壊れた1ノード（NaNを出す、あるいはもっと悪いことに — NaNなしで*間違った数値*を出す不良HBM、silent data corruption）がグローバルall-reduce全体を汚染しえます。防御: 勾配ノルムのクリッピングと監視、ランクごとの勾配ノルム外れ値検出（腐ったランクは損失に現れる*前に*見える）、スパイク時のバッチスキップポリシー、疑わしいハードウェアへの定期的な既知解テスト。

**トポロジを無視した配置。** スケジューラがデータセンター中に散らばった512 GPUを与え、ringが過剰契約されたスパインリンクを横切り、all-reduceはノードローカル速度の何分の一かで走ります。トポロジを意識したギャングスケジューリング（ラック/ポッド丸ごと、[MLキャパシティとコストプランニング](./14-ml-capacity-cost-planning.md)）は、インフラの気配りではなく一次のスループット要因です。

**カーゴカルトされた並列構成。** あるファブリック（NVLink+NDR InfiniBand）向けにチューニングされた3D構成を、より遅いクラウドネットワークに移植するとトレードオフが反転します — ノード間TP、マイクロバッチが少なすぎるPP。公開された構成は他人のハードウェアをエンコードしています。自分の帯域の数字から再導出を。

---

## 意思決定フレームワーク

| 状況 | 選ぶもの |
|---|---|
| モデル+オプティマイザ状態が1 GPUに収まる | 素のDDP。機構を足す前に勾配蓄積を |
| 状態が1 GPUを超えるがクラスタ÷Nに収まる | ZeRO/FSDP（ステージ2、次に3）— 7B〜70Bの主力 |
| シャードしてもなお単一レイヤー/アクティベーションのワーキングセットがノードのメモリを超える | TPを追加（≤ノードサイズ）、その後ノード間にPP |
| 数百ノード、モデルが多数にまたがる | フル3D: 最内にTP≤8、収まるまでPP、最外にDP（+ZeRO-1） |
| 長コンテキスト学習がアクティベーションメモリを破壊 | まずアクティベーションチェックポインティング、次にシーケンス/コンテキスト並列 |
| MoEモデル | エキスパート並列（all-to-all）。そのためのファブリックを明示的に予算化 |
| DPをスケールしても収束までのステップ数が改善しなくなった | クリティカルバッチサイズに到達 — GPUを足しても無駄。並列の分割を変えるか、壁を受け入れる |
| 数千GPU以上のジョブ | 非同期シャード化チェックポインティング、NCCLウォッチドッグ、ホットスペア、ランクごとのステップ時間監視 — スケールの前に。最初の午前3時のハングの後ではなく |
| クラスタサイズ/予算の決定 | まず6PTの算術とMFU目標（[キャパシティプランニング](./14-ml-capacity-cost-planning.md)）。GPU数は出力であって入力ではない |

---

## 重要なポイント

1. **2つの台帳が分散を強制する**: 学習状態の16 bytes/param（メモリ）と6·P·TのFLOPs（計算）。アーキテクチャの議論の前にこの算術を — GPU数は出力である。
2. **データ並列はバッチサイズの天井までスケールする**。ネットワークだけの問題ではない — クリティカルバッチサイズを超えると、レプリカを足してもステップが大きくなるだけで学習は速くならない。
3. **ZeRO/FSDPは冗長性を取り除いたデータ並列** — 約1.5倍の通信で16P/Nのメモリ。「1レイヤーがノードに収まらない」閾値まではデフォルトの答え。
4. **TPはノード内に閉じ、PPはノードを越え、バブルは(p−1)/(m+p−1)** — 3Dの合成は相互接続の階層が決める。好みではない。
5. **同期学習はフリートのp99.9で動く** — ストラグラー排除、バーンイン、トポロジ意識配置はスループット機能である。
6. **10K+ GPUでは数時間ごとの故障が通常運転** — Young/Dalyがチェックポイント間隔を決め、チェックポイント*コスト*を下げること（シャード化・非同期）はどんな単一のカーネル最適化より価値がある。
7. **MFUはスコアボード、goodputは稼働率** — データストール → 通信 → バブル → カーネルの順に監査する。MFU約40%は良好であり、25%から40%への差はたいていGPU 60%増より安い。
8. **再開は正しさの機能** — データカーソル、RNG、LRスケジュールは再起動をビット単位で生き延びなければならず、それを知る唯一の方法は再開パスを本番コードのようにテストすることである。

---

## 参考文献

- Shoeybi, M., et al. (2019). *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*. (Tensor parallelism.)
- Rajbhandari, S., et al. (2019). *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*. SC.
- Huang, Y., et al. (2019). *GPipe*; Narayanan, D., et al. (2019/2021). *PipeDream* and *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (3D parallelism, 1F1B, MFU-style accounting).
- Goyal, P., et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. (Linear scaling rule.)
- McCandlish, S., et al. (2018). *An Empirical Model of Large-Batch Training*. (Gradient noise scale / critical batch size.)
- Chowdhery, A., et al. (2022). *PaLM: Scaling Language Modeling with Pathways*. (MFU definition and reference numbers.)
- Grattafiori, A., et al. (2024). *The Llama 3 Herd of Models*. (16K-GPU operations: 466 interruptions/54 days, failure taxonomy, MFU at scale.)
- Jiang, Z., et al. (2024). *MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs*. NSDI.
- Young, J. W. (1974) / Daly, J. T. (2006). Optimal checkpoint interval analyses.
- Zhao, Y., et al. (2023). *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*. VLDB.
- NCCL documentation: *Collective Operations* (ring/tree algorithms and their cost models).
