# ACIDトランザクション

> **翻訳の状態:** 現行の[英語版](../../01-foundations/01-acid-transactions.md)からの翻訳です。

## TL;DR

振込が正しいためには、両方の口座に指定額の増減が一度だけ反映され、残高の合計が保たれ、振込元が残高不足にならないことが必要です。データベースが一連の書き込みを原子的にコミットしても、渡された書き込みが誤っていれば、この条件は崩れます。設計レビューでは、不変条件とトランザクション境界を最初に確認します。次に、並行処理から見える状態、コミットが耐える障害、応答を失った呼び出し元が結果を確定する方法を確認します。

口座AからBへの100米ドルの振込を考えます。Bが存在せず入金の更新件数がゼロなら、引き落としだけがコミットされることがあります。二つの引き落としが競合する場合は、必要なロックを取得した後に残高を確認します。コミット応答を失った場合は、永続化された操作IDで結果を確認します。プライマリを失った場合は、どのスタンバイがコミットを保持し、どのスタンバイを昇格させるかで結果が決まります。

## ACIDが解決する問題

Aには500米ドル、Bには200米ドルがあります。100米ドルを振り込んだ後の残高は、Aが400米ドル、Bが300米ドルです。この参照設計では、両口座を一つのPostgreSQL 16データベースに置き、残高を整数のセントで表します。呼び出し元は論理的な振込ごとにUUIDを発行します。残高を書き換える処理はすべて同じ不変条件を守る必要があります。

| 性質 | 振込の設計で確認すること |
|---|---|
| 原子性 | 文の失敗やサーバーのクラッシュが起きたとき、どの書き込みが一緒にコミットされるか。 |
| 整合性 | どのデータベース制約とアプリケーションの検査が残高の不変条件を守るか。 |
| 分離性 | 別の引き落としや集計が振込と重なったとき、どの実行履歴が成立するか。 |
| 耐久性 | 成功応答の後、どの障害が起きても振込が残るか。 |

原子性と分離性は、データベースが受け取った処理に対する保証です。同じ処理を何度か試行したとき、それらを一つの論理操作に対応づけるためには、呼び出し側にもプロトコルが必要です。

## 原子性と不変条件

引き落とし、入金、操作記録を同じトランザクションに含めます。トランザクションに属する書き込みは、まとめてコミットされるか、まとめて破棄されます。振込先が存在しない場合に、その保証の境界が見えます。PostgreSQLでは、対象行がゼロ件でも `UPDATE` は成功します。アプリケーションが引き落としを実行し、入金の更新件数を無視してコミットすれば、データベースは渡された引き落としだけを原子的にコミットします。

残高を負にしないという行単位のルールは `CHECK` 制約で守ります。振込処理は、異なる二つの実在する口座に同額の増減を適用し、行をまたぐルールを守ります。残高を書き換える経路が複数あるときは、両方の仕組みが必要です。残高の制約は入金の欠落を検出できません。また、別の書き込み経路が振込処理を迂回すれば、その処理で不変条件を守ることはできません。

セーブポイントは、先行する変更を残したまま任意の作業を取り消すために使えます。入金の失敗時には振込全体を中止します。引き落とし後に設定したセーブポイントまで戻ってコミットすると、誤った書き込みが残ります。ロックの挙動はエンジンによって異なります。PostgreSQLはセーブポイントより後に取得したトランザクションロックを解放します。InnoDBはメモリに保持した行ロックを残すことがあります。一方、新規挿入行に対応するロックは、undoでその行を取り消すと解放されます。

遅延制約は不変条件の検査をコミット時まで遅らせます。二つの行が相互に必須の外部キーを持つ場合などに使えます。文が成功した後でコミットが失敗する可能性があるため、アプリケーションのエラー処理はコミットまで含めます。

## 引き落としが競合するときの分離性

二つの呼び出し元がそれぞれAから400米ドルを振り込むとします。どちらの書き込みよりも先に、両方が500米ドルを読み取ることは可能です。その値を根拠に引き落としを承認すると、両者が同じ残高で判断したことになります。PostgreSQLのRead Committedでは、残高確認と書き込みが同じ状態を根拠とするように設計します。

既知の一行の残高を守るには、`UPDATE accounts SET balance_cents = balance_cents - 40000 WHERE id = 1 AND balance_cents >= 40000` のような条件付き更新で十分です。並行更新が先行した場合、PostgreSQLはその更新を待ち、更新後の行で条件を再評価します。後続の引き落としの更新件数はゼロです。振込で両口座をロックする場合は、残高を確認する前にIDの昇順で取得します。これにより、逆方向の二つの振込が互いの振込先のロックを待つ循環を避けられます。他のリソースを介したデッドロックは引き続き起こりえます。

集計処理には別の問題があります。Read Committedでは、集計が振込前のAと振込後のBを読むことがあります。どちらのクエリもコミット済みデータを読みますが、集計結果は本来の700米ドルではなく800米ドルです。単一の通常の `SELECT` で集計できるなら、文単位のスナップショットで十分です。複数の文がスナップショットを共有する必要がある読み取り専用の集計には、Repeatable Readを使います。PostgreSQLがそのスナップショットを取得するのは、最初のトランザクション制御文以外の文を実行するときです。`BEGIN` と最初の `SELECT` の間にコミットされた振込は、集計に含まれます。

日次の引き落とし上限のように、不変条件が複数の振込行を対象とする場合、二つの口座行だけをロックしても、他の書き込み経路を制御できないことがあります。共通の管理行で条件を守るか、関連するトランザクションをSerializableで実行します。直列化失敗時にはトランザクション全体を再試行します。行の競合と述語の競合を分ける実行履歴は、[分離レベル](02-isolation-levels.md)で解説しています。

## 振込を検証する

使い捨てのPostgreSQL 16データベースで次を実行します。検証中は振込記録を保持してください。この関数はRead Committedを前提とし、トランザクションの開始と終了は呼び出し側が行います。認可と操作IDの保持期間はサービス境界で定めます。

```sql
CREATE TABLE accounts (
    id integer PRIMARY KEY,
    balance_cents bigint NOT NULL CHECK (balance_cents >= 0)
);

CREATE TABLE transfers (
    operation_id uuid PRIMARY KEY,
    source_id integer NOT NULL,
    destination_id integer NOT NULL,
    amount_cents bigint NOT NULL CHECK (amount_cents > 0),
    CHECK (source_id <> destination_id)
);

INSERT INTO accounts VALUES (1, 50000), (2, 20000);

CREATE FUNCTION post_transfer(
    p_operation_id uuid,
    p_source_id integer,
    p_destination_id integer,
    p_amount_cents bigint
) RETURNS void
LANGUAGE plpgsql VOLATILE AS $$
DECLARE
    locked_rows integer;
BEGIN
    INSERT INTO transfers VALUES (
        p_operation_id, p_source_id, p_destination_id, p_amount_cents
    ) ON CONFLICT (operation_id) DO NOTHING;

    IF NOT FOUND THEN
        IF NOT EXISTS (
            SELECT 1 FROM transfers
            WHERE operation_id = p_operation_id
              AND source_id = p_source_id
              AND destination_id = p_destination_id
              AND amount_cents = p_amount_cents
        ) THEN
            RAISE EXCEPTION 'operation ID reused with different inputs';
        END IF;
        RETURN;
    END IF;

    PERFORM id FROM accounts
    WHERE id IN (p_source_id, p_destination_id)
    ORDER BY id FOR UPDATE;
    GET DIAGNOSTICS locked_rows = ROW_COUNT;
    IF locked_rows <> 2 THEN
        RAISE EXCEPTION 'both accounts must exist';
    END IF;

    UPDATE accounts SET balance_cents = balance_cents - p_amount_cents
    WHERE id = p_source_id AND balance_cents >= p_amount_cents;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'debit rejected';
    END IF;

    UPDATE accounts SET balance_cents = balance_cents + p_amount_cents
    WHERE id = p_destination_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'credit rejected';
    END IF;
END;
$$;
```

同じ操作IDで二つの試行が競合したときは、一意インデックスが調停します。Read Committedでは、先に挿入したトランザクションを待った後、後続の挿入が何も変更せずに終わることがあります。関数内の次のSQL文は新しいスナップショットを取得するため、記録済みの入力を照合できます。先行処理が中止された場合は、後続処理が挿入を行い、そのまま進みます。操作IDが異なる場合は口座のロックを取得し、残高を取り合います。呼び出し元に成功を返すのは、外側のトランザクションがコミットしてからです。

```sql
BEGIN ISOLATION LEVEL READ COMMITTED;
SELECT post_transfer('00000000-0000-0000-0000-000000000001', 1, 2, 10000);
COMMIT;

SELECT id, balance_cents FROM accounts ORDER BY id;
SELECT operation_id, source_id, destination_id, amount_cents FROM transfers;
```

各試行の前に二つの口座を初期残高に戻し、振込テーブルを空にします。並行実行の試行には別々のセッションを使います。

| 試行 | 実行手順 | 確認する結果 |
|---|---|---|
| 振込先が存在しない | 振込先に `99` を指定します。エラー後にロールバックします。 | 残高は50,000セントと20,000セントのままです。振込記録は残りません。 |
| 引き落としの競合 | 異なるIDで40,000セントの振込を二回実行します。後続処理が待機するまで先行トランザクションを開いておき、その後コミットします。 | 後続処理は残高確認で失敗します。振込は一件だけ残り、残高は10,000セントと60,000セントです。 |
| 再試行とIDの衝突 | サンプルをコミットします。同じIDと入力で再試行し、その後、同じIDで別の正の金額を指定します。 | 再試行は成功し、入力を変更した要求は失敗します。振込は一件だけ残り、残高は40,000セントと30,000セントです。 |
| 読み取りスキュー | セッション1で読み取り専用トランザクションを開始し、Aを読みます。セッション2でサンプルの振込をコミットし、セッション1でBを読みます。分離レベルを変えて繰り返します。 | Read Committedでは50,000セントと30,000セント、Repeatable Readでは50,000セントと20,000セントが見えます。 |

## コミット応答を失ったとき

直列化失敗の場合、トランザクションは中止されたと判断できます。`COMMIT` の応答待ちで接続が切れた場合、結果は不明です。サーバーが振込を永続化した後に、ネットワークで応答が失われた可能性があります。ドライバーは接続を後処理できますが、サーバーで成立したコミットを取り消せません。

結果が不明なら、現在の書き込み権限を持つノードに、**同じ操作IDと入力**で再試行します。一意な操作記録は、その効果とともにコミットされます。最初の振込がコミット済みなら再試行はその結果を返し、中止されていれば再試行で振込を適用できます。遅延しているスタンバイに行がないことから、どちらの結果だったかは判断できません。フェイルオーバー直後は、どの履歴を基準にしてよいかを、書き込み側がまず確定する必要があります。操作記録の耐久性は、それが保護するトランザクションの耐久性と同じです。

再試行の検証では、呼び出し元が初回応答を無視した場合を模擬しています。ネットワーク障害の経路を確認するには、データベースでコミットが成立した後の応答を遮断し、同じIDで再試行して、振込件数と両口座の残高を確認します。本番では、結果不明の処理を論理操作IDで追跡し、サポートする再試行期間中は記録を保持します。データベースの外部で生じる効果には、[アウトボックス](../05-messaging/07-outbox-pattern.md)や、独自の冪等性を持つ処理プロトコルを使います。IDのスコープ、保持期間、突合は[冪等性と操作ID](08-idempotency.md)で解説しています。

## コミットから昇格までの耐久性

### WALの仕組み

PostgreSQLのログ対象テーブルでは、コミットレコードまで先行書き込みログ（WAL）をフラッシュすれば、トランザクションを永続化できます。変更されたテーブルのページは後から永続化できます。ダーティページを書き出す前に、エンジンはそのページの復旧に必要なWALをフラッシュします。再起動時、PostgreSQLはチェックポイントレコードを読み、そこに**記録された**redo位置から復旧を始めます。チェックポイントレコード自体の位置は別の座標です。

PostgreSQLは、コミットされなかったトランザクションの物理的な変更を再適用することもあります。可視性のルールによって、未コミットの新しい行バージョンは見えず、後からVACUUMが回収できます。InnoDBはロールバックと過去の読み取りのためにundoレコードを保持します。そのため、大規模なロールバックでは、アプリケーションが待つのをやめた後も処理が続きます。PostgreSQLでは後から行バージョンの回収コストが現れます。再適用、チェックポイント、部分書き込みからの復旧は[先行書き込みログ](../03-storage-engines/04-write-ahead-logging.md)を参照してください。

フラッシュの成功を信頼できるのは、ファイルシステムとストレージが永続化の要求に従う場合です。ベンチマークでフラッシュのコストは測れますが、電源断に対するプロバイダーの保証までは確認できません。PostgreSQLのfsync障害はその違いを示しています。Linuxでは書き戻しエラーの報告後にページがクリーンとされた例があり、後続のフラッシュが成功しても、失敗した書き込みがディスクに到達した証拠にはなりませんでした。Linux 4.13で書き戻しエラーの報告が改善され、PostgreSQLは復旧時の対応を変更して、当時サポートしていた旧版にも修正を適用しました。

### グループコミットとフラッシュのコスト

複数のコミットレコードを永続化する一回のWALフラッシュで、それらのコミットを保証できます。並行処理があれば、そのコストを複数のトランザクションで分担できます。意図的に待機を入れるとバッチが大きくなりますが、到着がまばらな場合はレイテンシが増えます。待機時間を変更する前に、実際の到着パターンでコミットレイテンシとWALフラッシュ待ちを測定します。

PostgreSQLで `synchronous_commit = off` にすると、ローカルのWALフラッシュより前に応答を返せます。クラッシュ復旧で直近の応答済みトランザクションが失われる一方、トランザクションの整合性は保たれます。`fsync` の無効化とは故障時の結果が異なります。非同期コミットを採用できるのは、応答済みデータの損失をアプリケーションが許容できるか、復元できる場合です。振込の成功応答には通常、それより強い耐久性が求められます。

### スタンバイの応答とフェイルオーバー

PostgreSQL 16では、スタンバイ側の進捗をwrite、flush、replayの三段階で捉えます。writeではスタンバイのOSまで届きます。flushでストレージに永続化されます。replayが済むと、そのスタンバイ上の新しいスナップショットからコミット結果を読めます。同期スタンバイが構成されていれば、`remote_write`、`on`、`remote_apply` はそれぞれの段階まで待ちます。`synchronous_standby_names` が空なら、`on` が待つのはローカルのWALフラッシュだけです。

`FIRST 2 (s1, s2, s3)` は、優先順位に従って選ばれた二つのスタンバイを待ちます。`ANY 2 (s1, s2, s3)` は、対象内の任意の二つのスタンバイの応答で待機を終えます。どちらの設定でも必要な応答が得られなければ、コミット待ちが長引きます。リーダー選出や昇格は別途実装します。

`s1` と `s2` がWALをフラッシュした後に、プライマリが振込の成功を返したとします。その後プライマリが故障し、自動化が遅れている `s3` を昇格させると、別の生存スタンバイに振込が残っていても、新しいプライマリには存在しないことがあります。想定する障害でデータ損失ゼロを約束するなら、保持するWALに基づいて昇格候補を決め、旧プライマリが古い履歴上で書き込みを受け付けないようにします。誤ったマイグレーションが全スタンバイに再適用されるような障害には、バックアップからの復元演習が必要です。

## デプロイされた契約を検証する

アプリケーションと同じ接続経路で有効な設定を確認します。プライマリからスタンバイのflush位置とreplay位置を比較し、どのスタンバイがコミット待ちの対象かを確認します。設定値だけでは、個々のトランザクションで何を待ったかや、障害時に生存ノードが約束を守れるかは分かりません。

```sql
SHOW transaction_isolation;
SHOW synchronous_commit;
SHOW synchronous_standby_names;
SHOW fsync;
SHOW full_page_writes;

SELECT application_name, state, sync_state, write_lsn, flush_lsn, replay_lsn
FROM pg_stat_replication;
```

| 演習 | 合格条件 |
|---|---|
| 正当な二つの引き落としを競合させる | 後続処理の失敗が確定する。残高の合計が保たれ、コミット済みの振込記録と一致する。 |
| コミット前後で接続を遮断する | 同じ操作IDで再試行しても効果は一度だけ生じる。中止と、結果を確定する証拠がない場合を区別できる。 |
| 成功応答の後にプライマリを失う | 定義した障害モデルで保護される応答済み振込を、新しい書き込みノードで確認できる。旧プライマリは書き込みを受け付けない。 |
| 誤削除から復旧する | 検証済みのバックアップとWALアーカイブから意図した復旧時点へ戻せる。復旧時間を測り、書き込み再開前に記録を突合する。 |

## 設計判断

- **一つの書き込み権限で不変条件を守れる場合は、ローカルトランザクションを使います。** 制約と条件付き更新から始め、既知の行をまたぐ判断には順序を統一したロックを使い、述語をまたぐ判断にはSerializableを検討します。不変条件を壊す実行履歴を試験します。
- **成功応答が耐える障害を明示します。** プロセス再起動、ホスト喪失、ゾーン喪失のどれを想定するかを記録します。スタンバイのフラッシュと昇格の手順を確認し、昇格後に応答済みの振込を照合します。
- **再試行には同じ操作IDを使います。** 直列化によって中止された場合はトランザクション全体を再実行します。コミット結果が不明な場合は元の操作IDを保持して解決します。
- **シャード間のコミットとシャード間の可視性をそれぞれ定義します。** 二相コミットは参加者に一つの永続的なcommitまたはabortの決定を与えます。準備済みの参加者が決定を待ち続ける場合も、一方のシャードが適用した変更を、他方の適用前に独立した読み取りから観測できる場合もあります。必要な読み取り保証を含む分散トランザクションプロトコルを使います。イベントの配信意図にはアウトボックスを用い、外部での効果を突合、補償する場合は永続的なワークフローを使います。

## 参考文献

- PostgreSQL 16の[UPDATE](https://www.postgresql.org/docs/16/sql-update.html#SQL-UPDATE-OUTPUTS)、[Transaction Isolation](https://www.postgresql.org/docs/16/transaction-iso.html)、[Function Volatility Categories](https://www.postgresql.org/docs/16/xfunc-volatility.html)は、更新件数ゼロの場合とスナップショットの取得時点を説明しています。
- PostgreSQL 16の[Explicit Locking](https://www.postgresql.org/docs/16/explicit-locking.html)とMySQL 8.4の[Savepoint Statements](https://dev.mysql.com/doc/refman/8.4/en/savepoint.html)は、部分ロールバック後のロックの挙動を説明しています。
- PostgreSQL 16の[WAL Internals](https://www.postgresql.org/docs/16/wal-internals.html)、[WAL Configuration](https://www.postgresql.org/docs/16/runtime-config-wal.html)、[Synchronous Replication](https://www.postgresql.org/docs/16/warm-standby.html#SYNCHRONOUS-REPLICATION)は、redo位置とコミット応答の条件を定義しています。
- PostgreSQLプロジェクトの[Fsync Errors](https://wiki.postgresql.org/wiki/Fsync_Errors)は、LinuxとPostgreSQLで行われた変更をまとめています。
- Amazon Builders' Libraryの[Making Retries Safe with Idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)は、リクエストの識別と安全な再試行を説明しています。
- Jim GrayとLeslie Lamportの[Consensus on Transaction Commit](https://www.microsoft.com/en-us/research/publication/consensus-on-transaction-commit/)は、原子的コミットの問題と二相コミットのブロッキングを形式化しています。

---

## 関連する章

- [分離レベル](02-isolation-levels.md)は、制約、ロック、直列化で防ぐべき実行履歴を説明します。
- [冪等性と操作ID](08-idempotency.md)は、コミット応答を失ったときの解決方法と、遅れて届いた再試行への対処を説明します。
- [先行書き込みログ](../03-storage-engines/04-write-ahead-logging.md)は、コミットレコードとページの復旧、チェックポイント、レプリカの再適用の関係を説明します。
- [分散トランザクション](../02-distributed-databases/07-distributed-transactions.md)は、不変条件が複数の書き込み権限にまたがったときに決定と可視性を調整する方法を説明します。
