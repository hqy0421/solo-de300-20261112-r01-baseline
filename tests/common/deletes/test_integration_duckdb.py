"""End-to-end integration of the delete policy with a DuckDB merge pipeline.

The load layer integration uses the classification engine plus a simple merge resource:
marker-flagged rows are soft-deleted (flag column set), hard-delete verdicts remove rows by
primary key, and the tracker is persisted in resource state across pipeline runs.
"""

import pytest

from dlt.common.deletes import (
    DeletePolicy,
    DeleteTracker,
    process_package,
)

duckdb = pytest.importorskip("duckdb")


@pytest.fixture()
def conn():
    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE customers (
            id BIGINT PRIMARY KEY,
            name VARCHAR,
            version BIGINT,
            _dlt_soft_deleted BOOLEAN
        )
        """
    )
    yield connection
    connection.close()


def _upsert(conn, row):
    conn.execute(
        """
        INSERT OR REPLACE INTO customers (id, name, version, _dlt_soft_deleted)
        VALUES (?, ?, ?, ?)
        """,
        [
            row["id"],
            row["name"],
            row.get("version"),
            row.get("_dlt_soft_deleted"),
        ],
    )


def _apply(conn, processed, policy=None):
    """Mimics the destination load.

    - hard-deleted keys are removed,
    - absent soft-deleted keys (full sync) are flagged with an UPDATE,
    - present soft-deleted rows already carry the flag and upsert,
    - undetermined present rows are skipped so stale data cannot revive a confirmed delete.
    """
    for key in processed.hard_deleted_keys:
        conn.execute("DELETE FROM customers WHERE id = ?", [key])
    if policy is not None:
        present_ids = {row["id"] for row in processed.rows_to_yield}
        for key in processed.soft_deleted_keys:
            if key not in present_ids:
                conn.execute(
                    f"UPDATE customers SET {policy.soft_delete_column_name} = TRUE"
                    " WHERE id = ?",
                    [key],
                )
    undetermined = set(processed.undetermined_keys)
    for row in processed.rows_to_yield:
        if row["id"] in undetermined:
            continue
        _upsert(conn, row)


def _load(conn, policy, tracker, rows, key_columns=("id",), full_sync=None):
    processed = process_package(
        policy, rows, list(key_columns), tracker, full_sync=full_sync
    )
    _apply(conn, processed, policy=policy)
    for verdict in processed.verdicts:
        tracker.confirm_verdict(
            verdict.record_key,
            verdict.verdict,
            version=verdict.version,
            marker_value=verdict.marker_value,
        )
    return processed


class TestEndToEndLifecycle:
    def _policy(self, **overrides):
        config = {
            "delete_marker_column": "is_deleted",
            "marker_action": "soft_delete",
            "soft_delete_column_name": "_dlt_soft_deleted",
            "row_version_column_name": "version",
        }
        config.update(overrides)
        return DeletePolicy.from_config(config, "customers")

    def test_marker_soft_delete_preserves_history(self, conn):
        policy = self._policy()
        tracker = DeleteTracker()

        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 1, "is_deleted": False},
        ])
        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 2, "is_deleted": True},
        ])

        row = conn.execute(
            "SELECT id, _dlt_soft_deleted FROM customers WHERE id = 1"
        ).fetchone()
        # soft delete keeps the row and flags it
        assert row == (1, True)
        assert tracker.state_of(1) == "soft_deleted"

    def test_late_update_does_not_revive(self, conn):
        policy = self._policy()
        tracker = DeleteTracker()

        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 1, "is_deleted": False},
            {"id": 2, "name": "grace", "version": 1, "is_deleted": False},
        ])
        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 2, "is_deleted": True},
        ])
        # a stale package arrives late with the same version and the flag cleared
        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 2, "is_deleted": False},
        ])

        row = conn.execute(
            "SELECT _dlt_soft_deleted FROM customers WHERE id = 1"
        ).fetchone()
        assert row == (True,)
        # the still-active second customer is untouched
        assert conn.execute(
            "SELECT _dlt_soft_deleted FROM customers WHERE id = 2"
        ).fetchone() == (None,)

    def test_newer_version_revives_soft_deleted_record(self, conn):
        policy = self._policy()
        tracker = DeleteTracker()

        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 1, "is_deleted": False},
        ])
        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 2, "is_deleted": True},
        ])
        # customer restored in the source with a strictly newer version
        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 3, "is_deleted": False},
        ])

        row = conn.execute(
            "SELECT version, _dlt_soft_deleted FROM customers WHERE id = 1"
        ).fetchone()
        # revived: new version loaded and the soft-delete flag cleared
        assert row == (3, None)
        assert tracker.state_of(1) == "active"

    def test_marker_hard_delete_removes_row_idempotently(self, conn):
        policy = self._policy(marker_action="hard_delete")
        tracker = DeleteTracker()

        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 1, "is_deleted": False},
        ])
        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 2, "is_deleted": True},
        ])
        assert conn.execute("SELECT count(*) FROM customers").fetchone() == (0,)
        assert tracker.is_hard_deleted(1)

        # the same tombstone is delivered again (retry / duplicate package)
        processed = _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 2, "is_deleted": True},
        ])
        assert all(v.verdict == "hard_delete" for v in processed.verdicts)
        assert conn.execute("SELECT count(*) FROM customers").fetchone() == (0,)

    def test_present_only_never_deletes_absent_rows(self, conn):
        policy = self._policy()
        tracker = DeleteTracker()

        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 1, "is_deleted": False},
            {"id": 2, "name": "grace", "version": 1, "is_deleted": False},
        ])
        # incremental package omits customer 2, but no full sync is declared
        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 2, "is_deleted": False},
        ])

        assert conn.execute("SELECT count(*) FROM customers").fetchone() == (2,)
        assert tracker.state_of(2) == "active"

    def test_full_sync_physically_deletes_absent_rows(self, conn):
        policy = DeletePolicy.from_config(
            {
                "delete_marker_column": "is_deleted",
                "sync_scope": "full_sync",
                "absent_action": "hard_delete",
                "soft_delete_column_name": "_dlt_soft_deleted",
                "row_version_column_name": "version",
            },
            "customers",
        )
        tracker = DeleteTracker()

        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 1, "is_deleted": False},
            {"id": 2, "name": "grace", "version": 1, "is_deleted": False},
            {"id": 3, "name": "linus", "version": 1, "is_deleted": True},
        ])
        # complete snapshot no longer contains customer 2
        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 2, "is_deleted": False},
        ])

        ids = {r[0] for r in conn.execute("SELECT id FROM customers").fetchall()}
        assert ids == {1}
        assert tracker.is_hard_deleted(2)

    def test_full_sync_absent_soft_delete_flags_instead_of_removing(self, conn):
        policy = DeletePolicy.from_config(
            {
                "delete_marker_column": "is_deleted",
                "sync_scope": "full_sync",
                "absent_action": "soft_delete",
                "soft_delete_column_name": "_dlt_soft_deleted",
            },
            "customers",
        )
        tracker = DeleteTracker()

        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "is_deleted": False},
            {"id": 2, "name": "grace", "is_deleted": False},
        ])
        processed = _load(
            conn, policy, tracker, [{"id": 1, "name": "ada", "is_deleted": False}]
        )

        assert processed.soft_deleted_keys == [2]
        row = conn.execute(
            "SELECT _dlt_soft_deleted FROM customers WHERE id = 2"
        ).fetchone()
        assert row == (True,)

    def test_tracker_state_survives_serialization_round_trip(self, conn):
        policy = self._policy()
        tracker = DeleteTracker()
        _load(conn, policy, tracker, [
            {"id": 1, "name": "ada", "version": 1, "is_deleted": True},
        ])

        # pipeline state would serialize the tracker between runs
        import json

        restored = DeleteTracker.from_state_dict(json.loads(json.dumps(tracker.to_state_dict())))
        assert restored.state_of(1) == "soft_deleted"
        assert restored.version_of(1) == 1
