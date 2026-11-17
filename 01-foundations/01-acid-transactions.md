# ACID Transactions

## TL;DR

A transfer is correct when both accounts change by the intended amount exactly once, the sum is conserved, and no debit overdraws the source. The database can commit its writes atomically and still violate those rules if the application submits the wrong writes. Start your review with the invariant and the transaction boundary. Then ask what concurrent transactions may observe, what a successful commit survives, and how a caller resolves a lost reply.

Follow a USD 100 transfer from account A to B. If B is missing, a zero-row credit can leave the debit committed. If two debits race, their code must check the balance after acquiring the relevant lock. If the client loses the commit reply, it needs a durable operation ID. If the primary is lost, the answer depends on which standby has the commit and which standby is promoted.

## The Problem ACID Solves

A holds USD 500 and B holds USD 200. Moving USD 100 should leave A with USD 400 and B with USD 300. In this reference design, both accounts live in one PostgreSQL 16 database, balances are integer cents, and each logical transfer has a caller-generated UUID. Every writer that can change an account balance must preserve the same rules.

| Property | Question to ask about the transfer |
|---|---|
| Atomicity | If a statement fails or the server crashes, which writes commit together? |
| Consistency | Which database constraints and application checks preserve the balance rules? |
| Isolation | Which histories are possible if a second debit or a report overlaps the transfer? |
| Durability | Which failures can occur after success without losing the transfer? |

Atomicity and isolation describe what the database does with the operations it receives. The caller also needs a protocol for mapping repeated attempts to one logical operation.

## Atomicity and the Business Invariant

Put the debit, credit, and operation record in the same transaction. A transaction provides all-or-nothing commit for its transactional writes. A missing destination exposes the boundary of that promise. In PostgreSQL, an `UPDATE` that matches zero rows succeeds. If the application issues the debit, ignores the credit's row count, and commits, the database atomically commits exactly the write set it was given.

Use a `CHECK` constraint for the local rule that a balance is nonnegative. The transfer procedure enforces the cross-row rule by applying equal and opposite changes to two distinct, existing accounts. Both mechanisms matter when there are multiple write paths. A constraint on each balance cannot detect that the credit was skipped. A transfer procedure cannot protect the invariant from a second writer that bypasses it.

Savepoints let a transaction abandon optional work while preserving earlier changes. A failed credit must abort the transfer. Returning to a savepoint that was set **after** the debit and then committing would preserve the wrong write set. Lock behavior also depends on the engine. PostgreSQL releases transaction locks acquired after the savepoint when rolling back to it. InnoDB can retain row locks stored in memory, although it releases the lock associated with an inserted row when undo removes the row.

Deferred constraints move an invariant check to commit. They are appropriate for relationships such as two rows with mandatory foreign keys to each other. A statement can succeed and the transaction can still fail at commit. Commit is part of the application error path.

## Isolation Under Concurrent Debits

Suppose two callers each want to move USD 400 from A. Both can read USD 500 before either writes. If both use that value to approve the debit, they made their decision on the same balance. At PostgreSQL Read Committed, an application must make the check and the write agree.

A conditional statement such as `UPDATE accounts SET balance_cents = balance_cents - 40000 WHERE id = 1 AND balance_cents >= 40000` is enough for the balance condition on one known row. When a concurrent update wins, PostgreSQL waits and re-evaluates the predicate on the updated row. The losing debit updates zero rows. For a transfer that locks both accounts, lock the rows in ascending ID order before checking the balance. This avoids the cycle where each of two opposite transfers holds the other's destination. Other resources can still deadlock.

A reporting transaction illustrates a different property. It can read A before the transfer and B after it under Read Committed. Each query sees committed data, but the report totals USD 800 instead of USD 700. For a report that fits in one ordinary `SELECT`, use one statement snapshot. For a read-only report that needs several statements to share a snapshot, use Repeatable Read. PostgreSQL takes that snapshot at the first non-transaction-control statement. A transfer committed between `BEGIN` and the first `SELECT` can therefore appear in the report.

When the invariant spans a predicate such as a daily debit limit across transfer rows, locking just the two account rows may leave other writers outside the protocol. Enforce the predicate at a shared guard or run the relevant transaction classes at Serializable. Plan to retry the whole transaction on a serialization failure. [Isolation Levels](02-isolation-levels.md) explains the histories that distinguish row conflicts from predicate conflicts.

## A Transfer You Can Verify

Run the following in a disposable PostgreSQL 16 database. Keep the transfer records for the exercise. The function is written for Read Committed, and its caller owns the transaction. Authorization and the retention policy for operation IDs belong at the service boundary.

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

The unique index arbitrates two concurrent attempts with the same operation ID. Under Read Committed, the insert can wait for a concurrent winner and then return without inserting. The function's next SQL command takes a fresh snapshot and can compare the stored inputs. If the first attempt aborts, the second attempt can insert and proceed. Distinct operation IDs proceed to the account locks and compete for the available balance. No success should be sent to the caller until the outer transaction commits.

```sql
BEGIN ISOLATION LEVEL READ COMMITTED;
SELECT post_transfer('00000000-0000-0000-0000-000000000001', 1, 2, 10000);
COMMIT;

SELECT id, balance_cents FROM accounts ORDER BY id;
SELECT operation_id, source_id, destination_id, amount_cents FROM transfers;
```

Reset to the initial two balances and empty transfer table before each trial. Use separate sessions for concurrent trials.

| Trial | What to run | Evidence |
|---|---|---|
| Missing B | Call with destination `99`. Roll back after the error. | A and B still hold 50,000 and 20,000 cents. No transfer row survives. |
| Competing debits | Call twice for 40,000 cents with distinct IDs. Hold the first transaction open until the second waits, then commit the first. | The second fails the balance check. One transfer survives. A and B hold 10,000 and 60,000 cents. |
| Replay or ID conflict | Commit the sample call. Repeat it with the same ID and inputs, then try that ID with a different positive amount. | The replay succeeds and the changed request fails. One transfer survives. A and B hold 40,000 and 30,000 cents. |
| Read skew | In session 1, begin a read-only transaction and read A. Commit the sample transfer in session 2, then read B in session 1. Repeat the trial with each isolation level. | Read Committed yields 50,000 and 30,000 cents. Repeatable Read yields 50,000 and 20,000 cents. |

## When the Commit Reply Is Lost

A serialization failure is a definite abort. A disconnect while waiting for `COMMIT` can leave the result unknown. The server may have persisted the transfer and lost the network response. A driver can clean up a connection. It cannot undo a commit the server already made.

Retry an unknown outcome with the **same** operation ID and inputs against the current write authority. The unique record and its effects commit together. If the first transaction committed, the replay returns its outcome. If it aborted, the replay can apply the transfer. A missing row on a lagging standby proves neither case. After failover, the authority must first establish which history it can safely serve. An idempotency record is only as durable as the transaction it protects.

The replay trial simulates the caller disregarding the first reply. To verify the network failure path, interrupt the reply after the database commits, retry the same ID, and assert both the transfer count and the two balances. In production, track unknown outcomes by logical operation ID and retain records for the supported retry horizon. For side effects outside the database, use an [outbox](../05-messaging/07-outbox-pattern.md) or an effect protocol with its own idempotency. [Idempotency and Operation Identity](08-idempotency.md) covers identity scope, retention, and reconciliation.

## Durability From Commit to Promotion

### WAL Mechanics

For a logged PostgreSQL table, the server can make the transaction durable by flushing write-ahead log (WAL) through the commit record. The modified table pages can reach durable storage later. Before flushing a dirty data page, the engine must flush the WAL needed to recover that page. On restart, PostgreSQL reads the checkpoint record and starts recovery at the redo position **stored in** that record. The position of the checkpoint record itself is a separate coordinate.

PostgreSQL may redo physical changes from a transaction that never committed. Its visibility rules keep that transaction's new row versions invisible, and vacuum can reclaim them later. InnoDB instead maintains undo records for rollback and older reads. A large InnoDB rollback can therefore keep doing substantial work after the application gives up. In PostgreSQL, the version cleanup can show up later. [Write-Ahead Logging](../03-storage-engines/04-write-ahead-logging.md) covers replay, checkpoints, and torn-page recovery.

A successful flush relies on the filesystem and storage honoring the durability request. A benchmark measures the cost of the flush. It cannot establish the provider's power-loss guarantee. The PostgreSQL fsync incident demonstrates the distinction. Some Linux writeback failures left pages clean after reporting an error, so a later successful flush did not prove the missing writes reached disk. Linux 4.13 improved writeback error reporting. PostgreSQL changed its recovery response and backpatched the change to supported earlier versions.

### Group Commit and the Cost of a Flush

One WAL flush can cover several commits if it persists their commit records. Under concurrency, multiple transactions can therefore share the flush cost. A deliberate delay can increase the batch size at the expense of latency when arrivals are sparse. Measure commit latency and WAL flush waits under the actual arrival pattern before changing the delay.

PostgreSQL `synchronous_commit = off` permits a reply before local WAL flush. Crash recovery can lose recently acknowledged transactions while preserving transactional consistency. That is a different failure from disabling `fsync`. Accept asynchronous commit only when the application can absorb or reconstruct the acknowledged data it may lose. A transfer acknowledgement normally carries a stronger promise.

### Standby Acknowledgement and Failover

In PostgreSQL 16, the three milestones on a standby are write, flush, and replay. Write reaches the standby's operating system. Flush establishes the storage boundary. Replay makes the committed change visible to a new snapshot on that standby. With synchronous standbys configured, `remote_write`, `on`, and `remote_apply` wait for those respective milestones. With an empty `synchronous_standby_names`, `on` waits for local WAL flush only.

With `FIRST 2 (s1, s2, s3)`, the primary waits for the two currently selected priority standbys. With `ANY 2 (s1, s2, s3)`, any two eligible standbys can satisfy the wait. Both settings can delay commits when the required acknowledgements cannot arrive. Neither setting is a leader election or promotion protocol.

Consider a successful transfer acknowledged by the primary after `s1` and `s2` flush its WAL. If automation promotes a lagging `s3` after the primary fails, the promoted database can omit the transfer even if a surviving standby holds it. To claim zero loss for the failure in scope, define which survivor may be promoted based on the WAL it has, and prevent the previous primary from accepting writes on its old history. A backup and restore exercise covers a different case, such as a bad migration that every standby faithfully replays.

## Verify the Deployed Contract

Inspect effective settings through the same application connection path. On the primary, compare standby flush and replay positions and identify which standbys count toward commit. A configuration value alone does not establish the effective transaction policy or show that a survivor can honor it.

```sql
SHOW transaction_isolation;
SHOW synchronous_commit;
SHOW synchronous_standby_names;
SHOW fsync;
SHOW full_page_writes;

SELECT application_name, state, sync_state, write_lsn, flush_lsn, replay_lsn
FROM pg_stat_replication;
```

| Exercise | Pass condition |
|---|---|
| Race two valid debits | The losing operation produces a definite failure. Balances conserve the total and match committed transfer records. |
| Disconnect around commit | A retry with the same operation ID produces one effect. The test distinguishes an abort from missing evidence. |
| Lose the primary after an acknowledgement | The elected writer exposes every acknowledged transfer covered by the stated failure model. The old primary cannot serve writes. |
| Recover after an accidental delete | Restore from a tested backup and WAL archive to the intended recovery point. Measure recovery time and reconcile records before reopening writes. |

## Decision Framework

- **Use a local transaction when one write authority owns the invariant.** Start with constraints and conditional updates. Add ordered locks when the decision spans known rows, or Serializable when it spans a predicate. Test the history that would otherwise break the invariant.
- **State the acknowledgement boundary in failure terms.** Write down whether success survives a process restart, host loss, or loss of a zone. Verify standby flush, promotion, and the exposure of acknowledged transfers after failover.
- **Apply one operation ID across retries.** Retrying a known serialization abort reruns the whole transaction. Resolving an unknown commit preserves the original operation identity.
- **Separate cross-shard commit from cross-shard visibility.** Two-phase commit gives participants one durable commit or abort decision. A participant that has prepared can wait indefinitely for that decision, and uninvolved readers can still observe one shard before the other applies it. Use a distributed transaction protocol covering the required read semantics. Use an outbox for publication intent or a durable workflow for external effects that can be reconciled or compensated.

## References

- PostgreSQL 16, [UPDATE](https://www.postgresql.org/docs/16/sql-update.html#SQL-UPDATE-OUTPUTS), [Transaction Isolation](https://www.postgresql.org/docs/16/transaction-iso.html), and [Function Volatility Categories](https://www.postgresql.org/docs/16/xfunc-volatility.html) document zero-row writes, snapshot boundaries, and snapshots within functions.
- PostgreSQL 16, [Explicit Locking](https://www.postgresql.org/docs/16/explicit-locking.html) and the MySQL 8.4 [Savepoint Statements](https://dev.mysql.com/doc/refman/8.4/en/savepoint.html) describe lock behavior after partial rollback.
- PostgreSQL 16, [WAL Internals](https://www.postgresql.org/docs/16/wal-internals.html), [WAL Configuration](https://www.postgresql.org/docs/16/runtime-config-wal.html), and [Synchronous Replication](https://www.postgresql.org/docs/16/warm-standby.html#SYNCHRONOUS-REPLICATION) define the redo pointer and commit acknowledgement modes.
- PostgreSQL project, [Fsync Errors](https://wiki.postgresql.org/wiki/Fsync_Errors) traces the Linux and PostgreSQL changes.
- Amazon Builders' Library, [Making Retries Safe with Idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/) documents request identity and safe retries.
- Jim Gray and Leslie Lamport, [Consensus on Transaction Commit](https://www.microsoft.com/en-us/research/publication/consensus-on-transaction-commit/) formalizes the atomic commit problem and the blocking behavior of two-phase commit.

---

## Related Chapters

- [Isolation Levels](02-isolation-levels.md): identify the transaction histories that a constraint, a lock, or serializable execution must prevent
- [Idempotency and Operation Identity](08-idempotency.md): resolve a lost commit reply and keep a delayed retry from applying a second effect
- [Write-Ahead Logging](../03-storage-engines/04-write-ahead-logging.md): follow the commit record through page recovery, checkpoints, and replica replay
- [Distributed Transactions](../02-distributed-databases/07-distributed-transactions.md): coordinate the decision and read visibility after an invariant spans write authorities
