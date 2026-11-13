"""Tests for applying delete policy verdicts to a data package."""

from dlt.common.deletes import (
    DeletePolicy,
    DeleteTracker,
    process_package,
)


def _marker_policy(**overrides):
    config = {
        "delete_marker_column": "is_deleted",
        "marker_action": "soft_delete",
        "soft_delete_column_name": "_dlt_soft_deleted",
        "row_version_column_name": "version",
    }
    config.update(overrides)
    return DeletePolicy.from_config(config, "customers")


class TestProcessPackage:
    def test_retained_rows_pass_through(self):
        policy = _marker_policy()
        tracker = DeleteTracker()
        processed = process_package(
            policy,
            [{"id": 1, "is_deleted": False, "version": 1}],
            ["id"],
            tracker,
        )
        assert processed.rows_to_yield == [
            {"id": 1, "is_deleted": False, "version": 1}
        ]
        assert not processed.has_deletes

    def test_soft_delete_row_is_flagged_and_yielded(self):
        policy = _marker_policy()
        tracker = DeleteTracker()
        tracker.confirm_active(1, version=1)
        processed = process_package(
            policy,
            [{"id": 1, "is_deleted": True, "version": 2}],
            ["id"],
            tracker,
        )
        assert processed.rows_to_yield == [
            {"id": 1, "is_deleted": True, "version": 2, "_dlt_soft_deleted": True}
        ]
        assert processed.soft_deleted_keys == [1]
        assert processed.hard_deleted_keys == []

    def test_hard_delete_row_is_not_yielded(self):
        policy = _marker_policy(marker_action="hard_delete")
        tracker = DeleteTracker()
        tracker.confirm_active(1, version=1)
        processed = process_package(
            policy,
            [
                {"id": 1, "is_deleted": 1, "version": 2},
                {"id": 2, "is_deleted": 0, "version": 1},
            ],
            ["id"],
            tracker,
        )
        assert processed.rows_to_yield == [
            {"id": 2, "is_deleted": 0, "version": 1}
        ]
        assert processed.hard_deleted_keys == [1]

    def test_tracker_is_not_mutated(self):
        policy = _marker_policy()
        tracker = DeleteTracker()
        tracker.confirm_active(1, version=1)
        process_package(
            policy,
            [{"id": 1, "is_deleted": True, "version": 2}],
            ["id"],
            tracker,
        )
        # confirmation happens only after a successful load
        assert tracker.state_of(1) == "active"

    def test_full_sync_absent_keys_are_reported_for_hard_delete(self):
        policy = DeletePolicy.from_config(
            {
                "delete_marker_column": "is_deleted",
                "sync_scope": "full_sync",
                "soft_delete_column_name": "_d",
            },
            "customers",
        )
        tracker = DeleteTracker()
        tracker.confirm_active(1)
        tracker.confirm_active(2)
        processed = process_package(
            policy,
            [{"id": 1, "is_deleted": False}],
            ["id"],
            tracker,
        )
        assert processed.hard_deleted_keys == [2]
        assert [r["id"] for r in processed.rows_to_yield] == [1]

    def test_present_only_absent_keys_are_undetermined(self):
        policy = _marker_policy()
        tracker = DeleteTracker()
        tracker.confirm_active(1)
        tracker.confirm_active(2)
        processed = process_package(
            policy,
            [{"id": 1, "is_deleted": False, "version": 1}],
            ["id"],
            tracker,
        )
        assert processed.hard_deleted_keys == []
        assert processed.undetermined_keys == [2]
        assert all(r["id"] != 2 for r in processed.rows_to_yield)

    def test_stale_revive_attempt_is_undetermined_and_yielded_unchanged(self):
        policy = _marker_policy()
        tracker = DeleteTracker()
        tracker.confirm_soft_delete(1, version=5)
        processed = process_package(
            policy,
            [{"id": 1, "is_deleted": False, "version": 5}],
            ["id"],
            tracker,
        )
        # undetermined rows still flow through but carry no soft-delete flag change
        assert processed.rows_to_yield == [
            {"id": 1, "is_deleted": False, "version": 5}
        ]
        assert processed.undetermined_keys == [1]
        assert processed.soft_deleted_keys == []

    def test_composite_keys(self):
        policy = DeletePolicy.from_config(
            {"sync_scope": "full_sync"}, "customers"
        )
        tracker = DeleteTracker()
        tracker.confirm_active(("t1", 9))
        processed = process_package(policy, [], ["tenant", "id"], tracker)
        assert processed.hard_deleted_keys == [("t1", 9)]
        assert processed.rows_to_yield == []

    def test_confirm_after_load_round_trip(self):
        policy = _marker_policy()
        tracker = DeleteTracker()
        tracker.confirm_active(1, version=1)
        processed = process_package(
            policy,
            [{"id": 1, "is_deleted": True, "version": 2}],
            ["id"],
            tracker,
        )
        # simulate the load committing
        for verdict in processed.verdicts:
            tracker.confirm_verdict(
                verdict.record_key,
                verdict.verdict,
                version=verdict.version,
                marker_value=verdict.marker_value,
            )
        assert tracker.state_of(1) == "soft_deleted"
        assert tracker.version_of(1) == 2
