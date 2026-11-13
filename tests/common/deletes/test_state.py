"""Tests for `DeleteTracker` lifecycle state and its serialization."""

import pytest
from dlt.common.deletes import DeleteTracker, RecordLifecycle


class TestRecordLifecycle:
    def test_states(self):
        tracker = DeleteTracker()
        tracker.confirm_active(1, version=1)
        assert tracker.state_of(1) == "active"
        assert not tracker.is_deleted(1)

        tracker.confirm_soft_delete(1, version=2, marker_value=True)
        assert tracker.state_of(1) == "soft_deleted"
        assert tracker.is_deleted(1)
        assert tracker.version_of(1) == 2

        tracker.confirm_hard_delete(1, version=3)
        assert tracker.is_hard_deleted(1)

    def test_unknown_key(self):
        tracker = DeleteTracker()
        assert tracker.get(1) is None
        assert tracker.state_of(1) is None
        assert not tracker.is_deleted(1)

    def test_composite_key_normalization(self):
        tracker = DeleteTracker()
        tracker.confirm_active(["a", 1])
        assert ("a", 1) in tracker
        assert tracker.state_of(("a", 1)) == "active"


class TestIdempotency:
    def test_repeated_soft_delete_keeps_first_confirmation(self):
        tracker = DeleteTracker()
        tracker.confirm_soft_delete(1, version=2, marker_value=True)
        first = tracker.get(1)
        # a duplicate package retries the same delete
        second = tracker.confirm_soft_delete(1, version=2, marker_value=True)
        assert first is second
        assert tracker.get(1).state == "soft_deleted"

    def test_repeated_hard_delete_is_noop(self):
        tracker = DeleteTracker()
        tracker.confirm_hard_delete(1, version=2)
        first = tracker.get(1)
        tracker.confirm_hard_delete(1, version=2)
        assert tracker.get(1) is first

    def test_repeated_active_confirmation_is_noop(self):
        tracker = DeleteTracker()
        tracker.confirm_active(1, version=7)
        first = tracker.get(1)
        assert tracker.confirm_active(1, version=7) is first

    def test_confirm_verdict_mapping(self):
        tracker = DeleteTracker()
        assert tracker.confirm_verdict(1, "retain", version=1).state == "active"
        assert (
            tracker.confirm_verdict(1, "soft_delete", version=2, marker_value="x").state
            == "soft_deleted"
        )
        assert tracker.confirm_verdict(1, "hard_delete", version=3).state == "hard_deleted"
        # undetermined must not mutate state
        record = tracker.get(1)
        assert tracker.confirm_verdict(1, "undetermined") is None
        assert tracker.get(1) is record

    def test_remove(self):
        tracker = DeleteTracker()
        tracker.confirm_active(1)
        tracker.remove(1)
        assert 1 not in tracker
        # removing an unknown key is safe
        tracker.remove(1)


class TestSerialization:
    def test_round_trip_with_scalar_keys(self):
        tracker = DeleteTracker()
        tracker.confirm_active(1, version=1)
        tracker.confirm_soft_delete("a", version=2, marker_value=True)
        tracker.confirm_hard_delete(7, version=3)

        restored = DeleteTracker.from_state_dict(tracker.to_state_dict())
        assert restored.state_of(1) == "active"
        record_a = restored.get("a")
        assert record_a.state == "soft_deleted"
        assert record_a.marker_value is True
        assert restored.state_of(7) == "hard_deleted"
        assert restored.version_of(7) == 3

    def test_round_trip_with_composite_keys(self):
        tracker = DeleteTracker()
        tracker.confirm_hard_delete(("tenant", 9), version=1)
        restored = DeleteTracker.from_state_dict(tracker.to_state_dict())
        assert restored.is_hard_deleted(("tenant", 9))

    def test_empty_state(self):
        assert DeleteTracker.from_state_dict(None) == DeleteTracker()
        assert DeleteTracker.from_state_dict({}).records == {}

    def test_state_is_json_compatible(self):
        import json

        tracker = DeleteTracker()
        tracker.confirm_soft_delete(1, version=2, marker_value="yes")
        encoded = json.dumps(tracker.to_state_dict())
        restored = DeleteTracker.from_state_dict(json.loads(encoded))
        assert restored.get(1).marker_value == "yes"

    def test_record_lifecycle_dict_round_trip(self):
        record = RecordLifecycle(state="soft_deleted", version=2, marker_value=1)
        assert RecordLifecycle.from_dict(record.to_dict()) == record
