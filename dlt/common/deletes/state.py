"""Lifecycle state of records under an optional delete policy.

The tracker stores, per record key, the *confirmed* lifecycle state:

- `active`: the latest confirmed version is a live record,
- `soft_deleted`: the record was soft-deleted (still present in the destination),
- `hard_deleted`: the record was hard-deleted (absent from the destination).

Only confirmed deletes are stored. A record that merely went missing from an incremental package
is never recorded as deleted, which is what keeps incremental loads safe.

The state is plain data (JSON-serializable) so it can live in pipeline resource state or in a
destination-side state table.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from dlt.common.deletes.typing import TDeleteVerdict, TRecordLifecycleState

TKey = TypeVar("TKey")

# sentinel version used when no row version column is configured
_NO_VERSION = None


def _normalize_key(record_key: Any) -> Any:
    """Makes composite keys hashable and keeps scalar keys unchanged."""
    if isinstance(record_key, (list, tuple)):
        return tuple(record_key)
    return record_key


@dataclass
class RecordLifecycle:
    """Confirmed lifecycle state of one record."""

    state: TRecordLifecycleState
    version: Any = None
    """Version of the record that confirmed this state (`None` without a version column)."""
    marker_value: Any = None
    """Original marker value that confirmed a soft delete, kept for idempotent re-application."""

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "version": self.version, "marker_value": self.marker_value}

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "RecordLifecycle":
        return cls(
            state=doc["state"],
            version=doc.get("version"),
            marker_value=doc.get("marker_value"),
        )


@dataclass
class DeleteTracker(Generic[TKey]):
    """Tracks confirmed lifecycle states keyed by the record's primary/merge key.

    The tracker never invents deletes: callers confirm a delete through `confirm_*` after the
    delete policy classified the record and the load layer applied it. Re-confirming the same
    state is a no-op, so deletes stay idempotent across retries and duplicate packages.
    """

    records: dict[Any, RecordLifecycle] = field(default_factory=dict)

    # ------------------------------------------------------------------ read
    def get(self, record_key: TKey) -> RecordLifecycle | None:
        return self.records.get(_normalize_key(record_key))

    def state_of(self, record_key: TKey) -> TRecordLifecycleState | None:
        record = self.get(record_key)
        return record.state if record is not None else None

    def version_of(self, record_key: TKey) -> Any:
        record = self.get(record_key)
        return record.version if record is not None else _NO_VERSION

    def is_deleted(self, record_key: TKey) -> bool:
        return self.state_of(record_key) in ("soft_deleted", "hard_deleted")

    def is_hard_deleted(self, record_key: TKey) -> bool:
        return self.state_of(record_key) == "hard_deleted"

    def known_keys(self) -> Iterable[Any]:
        return self.records.keys()

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, record_key: object) -> bool:
        return _normalize_key(record_key) in self.records

    # --------------------------------------------------------------- mutate
    def confirm_active(self, record_key: TKey, version: Any = _NO_VERSION) -> RecordLifecycle:
        """Confirms a live version of the record."""
        key = _normalize_key(record_key)
        existing = self.records.get(key)
        if existing is not None and existing.state == "active" and existing.version == version:
            return existing
        record = RecordLifecycle(state="active", version=version)
        self.records[key] = record
        return record

    def confirm_soft_delete(
        self, record_key: TKey, version: Any = _NO_VERSION, marker_value: Any = None
    ) -> RecordLifecycle:
        """Confirms a soft delete. Idempotent: repeating it keeps the first confirmation."""
        key = _normalize_key(record_key)
        existing = self.records.get(key)
        if existing is not None and existing.state == "soft_deleted":
            return existing
        record = RecordLifecycle(
            state="soft_deleted", version=version, marker_value=marker_value
        )
        self.records[key] = record
        return record

    def confirm_hard_delete(
        self, record_key: TKey, version: Any = _NO_VERSION
    ) -> RecordLifecycle:
        """Confirms a hard delete. Idempotent: repeating it keeps the first confirmation."""
        key = _normalize_key(record_key)
        existing = self.records.get(key)
        if existing is not None and existing.state == "hard_deleted":
            return existing
        record = RecordLifecycle(state="hard_deleted", version=version)
        self.records[key] = record
        return record

    def confirm_verdict(
        self,
        record_key: TKey,
        verdict: TDeleteVerdict,
        version: Any = _NO_VERSION,
        marker_value: Any = None,
    ) -> RecordLifecycle | None:
        """Applies a `TDeleteVerdict` to the tracker.

        `retain` confirms an active record, `soft_delete`/`hard_delete` confirm the matching
        state and `undetermined` changes nothing (the previously confirmed state is preserved).
        Returns the resulting lifecycle record, or `None` for an undetermined verdict.
        """
        if verdict == "retain":
            return self.confirm_active(record_key, version)
        if verdict == "soft_delete":
            return self.confirm_soft_delete(record_key, version, marker_value)
        if verdict == "hard_delete":
            return self.confirm_hard_delete(record_key, version)
        return None

    def remove(self, record_key: TKey) -> None:
        self.records.pop(_normalize_key(record_key), None)

    # ----------------------------------------------------------- serialization
    def to_state_dict(self) -> dict[str, Any]:
        """Serializes the tracker to JSON-compatible data."""
        return {
            "records": [
                {"key": key, **record.to_dict()} for key, record in self.records.items()
            ]
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any] | None) -> "DeleteTracker[TKey]":
        tracker = cls()
        if not state:
            return tracker
        for doc in state.get("records", []):
            key = tuple(doc["key"]) if isinstance(doc["key"], list) else doc["key"]
            tracker.records[key] = RecordLifecycle.from_dict(doc)
        return tracker

    def items(self) -> Iterable[tuple[Any, RecordLifecycle]]:
        return self.records.items()
