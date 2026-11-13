"""Tests for the delete policy classification engine.

Covers the contract from the feature request:
- disabled by default,
- marker-based soft/hard delete,
- absent rows are never deleted without a full sync declaration,
- idempotent repeated deletes,
- late updates cannot revive confirmed deletes.
"""

import pytest
from dlt.common.deletes import (
    DEFAULT_ABSENT_ACTION,
    DEFAULT_MARKER_ACTION,
    DEFAULT_SYNC_SCOPE,
    DeleteMarkerColumnMissing,
    DeletePolicy,
    DeletePolicyConfigurationException,
    DeleteTracker,
    FullSyncDeclarationMissing,
    is_delete_marker_truthy,
)


def make_tracker(*states):
    tracker = DeleteTracker()
    for key, state, *rest in states:
        version = rest[0] if rest else None
        if state == "active":
            tracker.confirm_active(key, version=version)
        elif state == "soft_deleted":
            tracker.confirm_soft_delete(key, version=version)
        elif state == "hard_deleted":
            tracker.confirm_hard_delete(key, version=version)
    return tracker


class TestDisabledByDefault:
    def test_none_config_is_disabled(self):
        policy = DeletePolicy.from_config(None, "customers")
        assert policy.enabled is False

    def test_empty_config_is_disabled(self):
        policy = DeletePolicy.from_config({}, "customers")
        assert policy.enabled is False

    def test_disabled_policy_never_deletes_present_rows(self):
        policy = DeletePolicy.from_config(None, "customers")
        tracker = make_tracker((1, "active"))
        verdict = policy.classify_row({"id": 1, "is_deleted": True}, 1, tracker)
        assert verdict.verdict == "undetermined"

    def test_disabled_policy_never_deletes_absent_rows(self):
        policy = DeletePolicy.from_config(None, "customers")
        tracker = make_tracker((1, "active"))
        verdict = policy.classify_absent(1, tracker, full_sync=True)
        assert verdict.verdict == "undetermined"

    def test_defaults(self):
        assert DEFAULT_MARKER_ACTION == "soft_delete"
        assert DEFAULT_ABSENT_ACTION == "hard_delete"
        assert DEFAULT_SYNC_SCOPE == "present_only"


class TestMarkerTruthiness:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, False),
            (False, False),
            (0, False),
            (0.0, False),
            ("", False),
            ("  ", False),
            ("false", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("active", False),
            (True, True),
            (1, True),
            (-1, True),
            ("true", True),
            ("1", True),
            ("deleted", True),
            ("D", True),
            (["x"], True),
            ([], False),
        ],
    )
    def test_marker_values(self, value, expected):
        assert is_delete_marker_truthy(value) is expected


class TestMarkerBasedDeletes:
    def _policy(self, **overrides):
        config = {
            "delete_marker_column": "is_deleted",
            "marker_action": "soft_delete",
            "soft_delete_column_name": "_dlt_soft_deleted",
        }
        config.update(overrides)
        return DeletePolicy.from_config(config, "customers")

    def test_active_row_is_retained(self):
        policy = self._policy()
        tracker = make_tracker((1, "active"))
        verdict = policy.classify_row({"id": 1, "is_deleted": False}, 1, tracker)
        assert verdict.verdict == "retain"

    def test_truthy_marker_soft_deletes(self):
        policy = self._policy()
        tracker = make_tracker((1, "active"))
        verdict = policy.classify_row({"id": 1, "is_deleted": True}, 1, tracker)
        assert verdict.verdict == "soft_delete"
        assert verdict.marker_value is True
        assert policy.soft_delete_update(verdict) == {"_dlt_soft_deleted": True}

    def test_marker_can_hard_delete(self):
        policy = self._policy(marker_action="hard_delete")
        tracker = make_tracker((1, "active"))
        verdict = policy.classify_row({"id": 1, "is_deleted": "yes"}, 1, tracker)
        assert verdict.verdict == "hard_delete"

    def test_marker_value_is_propagated_to_soft_delete_column(self):
        policy = self._policy()
        tracker = make_tracker((1, "active"))
        verdict = policy.classify_row(
            {"id": 1, "is_deleted": "2026-01-01T00:00:00"}, 1, tracker
        )
        assert verdict.verdict == "soft_delete"
        assert policy.soft_delete_update(verdict) == {
            "_dlt_soft_deleted": "2026-01-01T00:00:00"
        }

    def test_missing_marker_column_raises_instead_of_guessing(self):
        policy = self._policy()
        tracker = make_tracker((1, "active"))
        with pytest.raises(DeleteMarkerColumnMissing):
            policy.classify_row({"id": 1}, 1, tracker)

    def test_repeated_tombstone_is_idempotent_without_version_column(self):
        policy = self._policy(marker_action="hard_delete")
        tracker = make_tracker((1, "hard_deleted"))
        first = policy.classify_row({"id": 1, "is_deleted": 1}, 1, tracker)
        second = policy.classify_row({"id": 1, "is_deleted": 1}, 1, tracker)
        assert first.verdict == "hard_delete"
        assert second.verdict == "hard_delete"
        assert "idempotent" in second.reason

    def test_unversioned_active_row_cannot_revive_hard_delete(self):
        # without a version column nothing can prove the row is newer than the confirmed delete
        policy = self._policy(marker_action="hard_delete")
        tracker = make_tracker((1, "hard_deleted"))
        verdict = policy.classify_row({"id": 1, "is_deleted": 0}, 1, tracker)
        assert verdict.verdict == "undetermined"

    def test_repeated_soft_tombstone_is_idempotent(self):
        policy = self._policy(row_version_column_name="version")
        tracker = make_tracker(("k", "soft_deleted", 5))
        verdict = policy.classify_row(
            {"id": "k", "is_deleted": True, "version": 5}, "k", tracker
        )
        assert verdict.verdict == "soft_delete"
        assert "idempotent" in verdict.reason


class TestNoResurrection:
    def _policy(self):
        return DeletePolicy.from_config(
            {
                "delete_marker_column": "is_deleted",
                "marker_action": "soft_delete",
                "soft_delete_column_name": "_dlt_soft_deleted",
                "row_version_column_name": "version",
            },
            "customers",
        )

    def test_same_version_update_cannot_revive_soft_delete(self):
        policy = self._policy()
        tracker = make_tracker((1, "soft_deleted", 5))
        verdict = policy.classify_row(
            {"id": 1, "is_deleted": False, "version": 5}, 1, tracker
        )
        assert verdict.verdict == "undetermined"
        assert "cannot revive" in verdict.reason

    def test_older_version_update_cannot_revive_hard_delete(self):
        policy = self._policy()
        tracker = make_tracker((1, "hard_deleted", 5))
        verdict = policy.classify_row(
            {"id": 1, "is_deleted": False, "version": 4}, 1, tracker
        )
        assert verdict.verdict == "undetermined"

    def test_unversioned_row_cannot_revive_versioned_delete(self):
        policy = self._policy()
        tracker = make_tracker((1, "soft_deleted", 5))
        verdict = policy.classify_row(
            {"id": 1, "is_deleted": False, "version": None}, 1, tracker
        )
        assert verdict.verdict == "undetermined"

    def test_newer_version_revives_record(self):
        policy = self._policy()
        tracker = make_tracker((1, "soft_deleted", 5))
        verdict = policy.classify_row(
            {"id": 1, "is_deleted": False, "version": 6}, 1, tracker
        )
        assert verdict.verdict == "retain"

    def test_incomparable_versions_block_resurrection(self):
        policy = self._policy()
        tracker = make_tracker((1, "soft_deleted", 5))
        verdict = policy.classify_row(
            {"id": 1, "is_deleted": False, "version": "abc"}, 1, tracker
        )
        assert verdict.verdict == "undetermined"


class TestAbsentRowsAndSyncScope:
    def test_present_only_absent_row_is_undetermined(self):
        policy = DeletePolicy.from_config(
            {
                "delete_marker_column": "is_deleted",
                "soft_delete_column_name": "_d",
            },
            "customers",
        )
        tracker = make_tracker((1, "active"))
        verdict = policy.classify_absent(1, tracker, full_sync=False)
        assert verdict.verdict == "undetermined"
        assert "full sync declaration" in verdict.reason

    def test_full_sync_absent_active_row_is_hard_deleted_by_default(self):
        policy = DeletePolicy.from_config(
            {
                "delete_marker_column": "is_deleted",
                "sync_scope": "full_sync",
                "soft_delete_column_name": "_d",
            },
            "customers",
        )
        tracker = make_tracker((1, "active"))
        verdict = policy.classify_absent(1, tracker, full_sync=True)
        assert verdict.verdict == "hard_delete"

    def test_full_sync_absent_can_soft_delete(self):
        policy = DeletePolicy.from_config(
            {
                "delete_marker_column": "is_deleted",
                "sync_scope": "full_sync",
                "absent_action": "soft_delete",
                "soft_delete_column_name": "_d",
            },
            "customers",
        )
        tracker = make_tracker((1, "active"))
        verdict = policy.classify_absent(1, tracker, full_sync=True)
        assert verdict.verdict == "soft_delete"

    def test_absent_unknown_record_is_undetermined(self):
        policy = DeletePolicy.from_config(
            {"sync_scope": "full_sync"}, "customers"
        )
        tracker = DeleteTracker()
        verdict = policy.classify_absent(42, tracker, full_sync=True)
        assert verdict.verdict == "undetermined"

    def test_repeated_full_sync_keeps_hard_delete_idempotent(self):
        policy = DeletePolicy.from_config(
            {"sync_scope": "full_sync"}, "customers"
        )
        tracker = make_tracker((1, "hard_deleted", 3))
        first = policy.classify_absent(1, tracker, full_sync=True)
        second = policy.classify_absent(1, tracker, full_sync=True)
        assert first.verdict == "hard_delete"
        assert second.verdict == "hard_delete"
        assert "idempotent" in second.reason

    def test_package_present_only_never_deletes_missing_rows(self):
        policy = DeletePolicy.from_config(
            {
                "delete_marker_column": "is_deleted",
                "soft_delete_column_name": "_d",
            },
            "customers",
        )
        tracker = make_tracker(("a", "active"), ("b", "active"))
        verdicts = policy.classify_package(
            [{"id": "a", "is_deleted": False}], ["id"], tracker
        )
        by_key = {v.record_key: v.verdict for v in verdicts}
        assert by_key == {"a": "retain", "b": "undetermined"}

    def test_full_sync_package_deletes_missing_rows(self):
        policy = DeletePolicy.from_config(
            {
                "delete_marker_column": "is_deleted",
                "sync_scope": "full_sync",
                "soft_delete_column_name": "_d",
            },
            "customers",
        )
        tracker = make_tracker(
            ("a", "active"), ("b", "active"), ("c", "hard_deleted")
        )
        verdicts = policy.classify_package(
            [{"id": "a", "is_deleted": False}], ["id"], tracker
        )
        by_key = {v.record_key: v.verdict for v in verdicts}
        assert by_key == {
            "a": "retain",
            "b": "hard_delete",
            "c": "hard_delete",
        }

    def test_full_sync_flag_on_present_only_policy_raises(self):
        policy = DeletePolicy.from_config(
            {
                "delete_marker_column": "is_deleted",
                "soft_delete_column_name": "_d",
            },
            "customers",
        )
        tracker = make_tracker(("b", "active"))
        with pytest.raises(FullSyncDeclarationMissing):
            policy.classify_package(
                [{"id": "a", "is_deleted": False}],
                ["id"],
                tracker,
                full_sync=True,
            )

    def test_composite_keys(self):
        policy = DeletePolicy.from_config(
            {"sync_scope": "full_sync"}, "customers"
        )
        tracker = DeleteTracker()
        tracker.confirm_active(("x", 1))
        verdicts = policy.classify_package([], ["tenant", "id"], tracker)
        assert len(verdicts) == 1
        assert verdicts[0].record_key == ("x", 1)
        assert verdicts[0].verdict == "hard_delete"


class TestConfigurationValidation:
    def test_unknown_action_rejected(self):
        with pytest.raises(DeletePolicyConfigurationException) as exc:
            DeletePolicy.from_config(
                {"delete_marker_column": "d", "marker_action": "explode"},
                "customers",
            )
        assert exc.value.field == "marker_action"

    def test_unknown_scope_rejected(self):
        with pytest.raises(DeletePolicyConfigurationException):
            DeletePolicy.from_config({"sync_scope": "whenever"}, "customers")

    def test_soft_delete_requires_destination_column(self):
        with pytest.raises(DeletePolicyConfigurationException) as exc:
            DeletePolicy.from_config(
                {"delete_marker_column": "d", "marker_action": "soft_delete"},
                "customers",
            )
        assert "soft_delete_column_name" in str(exc.value)

    def test_hard_delete_only_does_not_need_soft_column(self):
        policy = DeletePolicy.from_config(
            {
                "delete_marker_column": "d",
                "marker_action": "hard_delete",
                "sync_scope": "full_sync",
                "absent_action": "hard_delete",
            },
            "customers",
        )
        assert policy.enabled

    def test_full_sync_scope_enables_policy_without_marker(self):
        policy = DeletePolicy.from_config(
            {"sync_scope": "full_sync"}, "customers"
        )
        assert policy.enabled

    def test_round_trip_config(self):
        config = {
            "delete_marker_column": "is_deleted",
            "marker_action": "hard_delete",
            "sync_scope": "full_sync",
            "absent_action": "soft_delete",
            "soft_delete_column_name": "_d",
            "row_version_column_name": "version",
        }
        policy = DeletePolicy.from_config(config, "customers")
        assert policy.to_config() == config

    def test_disabled_round_trip_is_none(self):
        assert DeletePolicy.from_config(None).to_config() is None

    def test_soft_delete_update_without_column_raises(self):
        policy = DeletePolicy.from_config(
            {"sync_scope": "full_sync", "absent_action": "hard_delete"},
            "customers",
        )
        from dlt.common.deletes import DeletePolicyVerdict

        verdict = DeletePolicyVerdict(record_key=1, verdict="soft_delete")
        with pytest.raises(DeletePolicyConfigurationException):
            policy.soft_delete_update(verdict)
