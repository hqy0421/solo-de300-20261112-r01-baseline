"""The delete policy classification engine.

`DeletePolicy` turns source rows plus the previously confirmed lifecycle state into one of four
verdicts per record:

- `retain`: an active (live) version,
- `soft_delete`: keep the row but flag it as deleted,
- `hard_delete`: physically remove the row,
- `undetermined`: leave the destination row untouched.

Rules enforced here:

1. **Opt-in.** A policy without configuration does nothing; everything is `undetermined`.
2. **Marker first.** Rows carrying a truthy delete marker are classified with `marker_action`.
3. **No full sync, no absent deletes.** With the default `present_only` scope, records missing
   from a package are always `undetermined`. Absent records are deleted only when the package
   explicitly declares `full_sync`, and then with `absent_action`.
4. **Idempotency.** Re-delivering a deleted marker or re-running a full sync yields the same
   verdict; an already deleted record stays deleted.
5. **No resurrection by late data.** A package row whose version is not newer than the version
   of an already confirmed delete is `undetermined`, even if the marker is gone.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dlt.common.deletes.exceptions import (
    DeleteMarkerColumnMissing,
    DeletePolicyConfigurationException,
    FullSyncDeclarationMissing,
)
from dlt.common.deletes.state import DeleteTracker, _normalize_key
from dlt.common.deletes.typing import (
    TDeleteAction,
    TDeletePolicyDict,
    TDeleteVerdict,
    TDeletionScope,
)
from dlt.common.schema.typing import get_args

DEFAULT_MARKER_ACTION: TDeleteAction = "soft_delete"
DEFAULT_ABSENT_ACTION: TDeleteAction = "hard_delete"
DEFAULT_SYNC_SCOPE: TDeletionScope = "present_only"

_DELETE_ACTIONS: tuple[TDeleteAction, ...] = get_args(TDeleteAction)
_DELETION_SCOPES: tuple[TDeletionScope, ...] = get_args(TDeletionScope)


# textual values that explicitly mean "record is active"
_FALSE_MARKER_TOKENS = frozenset(("false", "0", "no", "n", "f", "active", ""))


def is_delete_marker_truthy(value: Any) -> bool:
    """Interprets a source delete marker value.

    Truthy markers are booleans `True`, non-zero numbers, non-empty collections and any other
    non-`None` value. Strings are matched case-insensitively: explicit active tokens
    (`"false"`, `"0"`, `"no"`, `"active"`, ...) and empty/whitespace strings mean "not deleted",
    any other non-empty string means "deleted".
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_MARKER_TOKENS
    return bool(value)


@dataclass
class DeletePolicyVerdict:
    """The classification of one record in a load package."""

    record_key: Any
    verdict: TDeleteVerdict
    table_name: str | None = None
    version: Any = None
    """Version of the package row that produced the verdict, if any."""
    marker_value: Any = None
    """Raw marker value seen on the package row, if any."""
    previous_version: Any = None
    """Version of the previously confirmed lifecycle state, if any."""
    reason: str = ""
    """Human-readable explanation, useful in traces and logs."""

    @property
    def is_delete(self) -> bool:
        return self.verdict in ("soft_delete", "hard_delete")

    @property
    def undetermined(self) -> bool:
        return self.verdict == "undetermined"

    def asdict(self) -> dict[str, Any]:
        return {
            "record_key": self.record_key,
            "verdict": self.verdict,
            "table_name": self.table_name,
            "version": self.version,
            "marker_value": self.marker_value,
            "previous_version": self.previous_version,
            "reason": self.reason,
        }


@dataclass
class DeletePolicy:
    """Validated, immutable delete policy for one table/resource.

    Build one with `DeletePolicy.from_config(...)`. An empty/`None` config produces a disabled
    policy whose verdicts are always `undetermined`, matching the "disabled by default" rule.
    """

    table_name: str | None = None
    delete_marker_column: str | None = None
    marker_action: TDeleteAction = DEFAULT_MARKER_ACTION
    absent_action: TDeleteAction = DEFAULT_ABSENT_ACTION
    sync_scope: TDeletionScope = DEFAULT_SYNC_SCOPE
    soft_delete_column_name: str | None = None
    row_version_column_name: str | None = None
    enabled: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.enabled = self.delete_marker_column is not None or self.sync_scope == "full_sync"

        if self.marker_action not in _DELETE_ACTIONS:
            raise DeletePolicyConfigurationException(
                self.table_name,
                "unknown marker action",
                field="marker_action",
                value=self.marker_action,
                known_values=_DELETE_ACTIONS,
            )
        if self.absent_action not in _DELETE_ACTIONS:
            raise DeletePolicyConfigurationException(
                self.table_name,
                "unknown absent action",
                field="absent_action",
                value=self.absent_action,
                known_values=_DELETE_ACTIONS,
            )
        if self.sync_scope not in _DELETION_SCOPES:
            raise DeletePolicyConfigurationException(
                self.table_name,
                "unknown sync scope",
                field="sync_scope",
                value=self.sync_scope,
                known_values=_DELETION_SCOPES,
            )
        if not self.enabled:
            return
        # a soft delete is reachable only when the action producing it has a configured path:
        # the marker action needs a marker column, the absent action needs full sync scope
        soft_via_marker = (
            self.delete_marker_column is not None and self.marker_action == "soft_delete"
        )
        soft_via_absence = (
            self.sync_scope == "full_sync" and self.absent_action == "soft_delete"
        )
        if (soft_via_marker or soft_via_absence) and not self.soft_delete_column_name:
            raise DeletePolicyConfigurationException(
                self.table_name,
                "a soft delete is reachable (marker or absent action) but no destination column"
                " was provided to store the deletion flag",
                field="soft_delete_column_name",
            )
        if self.marker_action == "hard_delete" and self.delete_marker_column is None:
            raise DeletePolicyConfigurationException(
                self.table_name,
                "`marker_action` requires `delete_marker_column`",
                field="delete_marker_column",
            )

    # ------------------------------------------------------------- construction
    @classmethod
    def from_config(
        cls,
        config: TDeletePolicyDict | None,
        table_name: str | None = None,
    ) -> "DeletePolicy":
        """Creates a policy from the resource hint configuration.

        `None` or an empty dict means "disabled": no row is ever classified as deleted.
        """
        if not config:
            return cls(table_name=table_name)
        return cls(
            table_name=table_name,
            delete_marker_column=config.get("delete_marker_column"),
            marker_action=config.get("marker_action", DEFAULT_MARKER_ACTION),
            absent_action=config.get("absent_action", DEFAULT_ABSENT_ACTION),
            sync_scope=config.get("sync_scope", DEFAULT_SYNC_SCOPE),
            soft_delete_column_name=config.get("soft_delete_column_name"),
            row_version_column_name=config.get("row_version_column_name"),
        )

    def to_config(self) -> TDeletePolicyDict | None:
        """Serializes back to the stored config shape, or `None` when disabled."""
        if not self.enabled:
            return None
        config: TDeletePolicyDict = {"sync_scope": self.sync_scope}
        if self.delete_marker_column:
            config["delete_marker_column"] = self.delete_marker_column
            config["marker_action"] = self.marker_action
        if self.sync_scope == "full_sync":
            config["absent_action"] = self.absent_action
        if self.soft_delete_column_name:
            config["soft_delete_column_name"] = self.soft_delete_column_name
        if self.row_version_column_name:
            config["row_version_column_name"] = self.row_version_column_name
        return config

    # ------------------------------------------------------------- row helpers
    def _row_version(self, row: Mapping[str, Any]) -> Any:
        if not self.row_version_column_name:
            return None
        return row.get(self.row_version_column_name)

    def _is_stale_against(self, version: Any, tracker: DeleteTracker[Any], key: Any) -> bool:
        """Whether a package row cannot override a confirmed delete.

        Without a row version column, the policy cannot prove that an incoming row is newer, so
        any row targeting an already deleted record is treated as stale. With a version column,
        the row is stale only when its version is not strictly greater than the confirmed one.
        """
        record = tracker.get(key)
        if record is None or record.state == "active":
            return False
        if not self.row_version_column_name:
            return record.state in ("soft_deleted", "hard_deleted")
        if record.version is None or version is None:
            # a deleted record exists but one side carries no version -> cannot prove the row
            # is newer, treat as stale
            return True
        try:
            return not (version > record.version)
        except TypeError:
            # incomparable versions: never let an unprovable update resurrect a delete
            return True

    # ------------------------------------------------------------- classification
    def classify_row(
        self,
        row: Mapping[str, Any],
        record_key: Any,
        tracker: DeleteTracker[Any],
    ) -> DeletePolicyVerdict:
        """Classifies a single package row against the confirmed lifecycle state."""
        key = _normalize_key(record_key)
        if not self.enabled:
            return DeletePolicyVerdict(
                record_key=key,
                verdict="undetermined",
                table_name=self.table_name,
                reason="delete policy disabled",
            )

        version = self._row_version(row)
        previous = tracker.get(key)
        previous_version = previous.version if previous is not None else None

        marker_value: Any = None
        has_marker = self.delete_marker_column is not None
        if has_marker:
            if self.delete_marker_column not in row:
                # the policy was promised a marker; silently treating such rows as active would
                # make deletes disappear, so fail loudly instead of guessing
                raise DeleteMarkerColumnMissing(
                    self.table_name or "<unknown>", self.delete_marker_column
                )
            marker_value = row[self.delete_marker_column]
        is_tombstone = has_marker and is_delete_marker_truthy(marker_value)

        # late/stale data can never revive an already confirmed deleted version. Tombstones get
        # their own idempotent handling below instead of being swallowed here.
        if (
            previous is not None
            and previous.state in ("soft_deleted", "hard_deleted")
            and not is_tombstone
            and self._is_stale_against(version, tracker, key)
        ):
            return DeletePolicyVerdict(
                record_key=key,
                verdict="undetermined",
                table_name=self.table_name,
                version=version,
                previous_version=previous_version,
                reason=(
                    "stale or unversioned update cannot revive confirmed"
                    f" {previous.state} record"
                ),
            )

        if is_tombstone:
            # idempotency: re-delivering the same (or older) tombstone of an already confirmed
            # delete re-reports the matching delete verdict (state -> verdict), independent of
            # `marker_action`, because the confirmed state is what the destination already holds
            if (
                previous is not None
                and previous.state in ("soft_deleted", "hard_deleted")
                and self._is_stale_against(version, tracker, key)
            ):
                return DeletePolicyVerdict(
                    record_key=key,
                    verdict=(
                        "soft_delete"
                        if previous.state == "soft_deleted"
                        else "hard_delete"
                    ),
                    table_name=self.table_name,
                    version=version,
                    marker_value=marker_value,
                    previous_version=previous_version,
                    reason=(
                        "re-delivered tombstone for an already confirmed"
                        f" {previous.state} record; idempotent re-confirmation"
                    ),
                )
            return DeletePolicyVerdict(
                record_key=key,
                verdict=self.marker_action,
                table_name=self.table_name,
                version=version,
                marker_value=marker_value,
                previous_version=previous_version,
                reason=f"truthy delete marker `{self.delete_marker_column}`",
            )

        return DeletePolicyVerdict(
            record_key=key,
            verdict="retain",
            table_name=self.table_name,
            version=version,
            previous_version=previous_version,
            reason="active record present in package",
        )

    def classify_absent(
        self,
        record_key: Any,
        tracker: DeleteTracker[Any],
        *,
        full_sync: bool,
    ) -> DeletePolicyVerdict:
        """Classifies a record that was previously seen but is absent from the current package.

        Without an explicit full sync declaration the verdict is always `undetermined`: absence
        from an incremental package proves nothing.
        """
        key = _normalize_key(record_key)
        record = tracker.get(key)
        previous_version = record.version if record is not None else None

        if not self.enabled:
            return DeletePolicyVerdict(
                record_key=key,
                verdict="undetermined",
                table_name=self.table_name,
                previous_version=previous_version,
                reason="delete policy disabled",
            )

        scope = "full_sync" if full_sync else self.sync_scope
        if scope != "full_sync":
            return DeletePolicyVerdict(
                record_key=key,
                verdict="undetermined",
                table_name=self.table_name,
                previous_version=previous_version,
                reason=(
                    "row absent from a present_only package; missing rows are not deletes"
                    " without a full sync declaration"
                ),
            )

        if record is None:
            # never confirmed before -> nothing to delete
            return DeletePolicyVerdict(
                record_key=key,
                verdict="undetermined",
                table_name=self.table_name,
                reason="absent record has no confirmed lifecycle state",
            )
        if record.state == "hard_deleted":
            # idempotent: re-running a full sync keeps the hard delete
            return DeletePolicyVerdict(
                record_key=key,
                verdict="hard_delete",
                table_name=self.table_name,
                previous_version=previous_version,
                reason="already hard-deleted; idempotent re-confirmation",
            )
        if record.state == "soft_deleted" and self.absent_action == "soft_delete":
            return DeletePolicyVerdict(
                record_key=key,
                verdict="soft_delete",
                table_name=self.table_name,
                previous_version=previous_version,
                marker_value=record.marker_value,
                reason="already soft-deleted; idempotent re-confirmation",
            )
        return DeletePolicyVerdict(
            record_key=key,
            verdict=self.absent_action,
            table_name=self.table_name,
            previous_version=previous_version,
            reason=f"absent from full sync package; absent_action={self.absent_action}",
        )

    def classify_package(
        self,
        rows: Sequence[Mapping[str, Any]],
        key_columns: Sequence[str],
        tracker: DeleteTracker[Any],
        *,
        full_sync: bool | None = None,
    ) -> list[DeletePolicyVerdict]:
        """Classifies every row of a package and all previously seen absent records.

        `full_sync` lets the load package override the configured scope per run (e.g. a source
        that normally syncs incrementally performs one full refresh). It defaults to the
        policy's configured `sync_scope == "full_sync"`.

        Verdicts for present rows come first, followed by verdicts for absent records known to
        the tracker. The tracker is **not** modified here: the caller applies verdicts via
        `tracker.confirm_verdict(...)` only after the load succeeds, so failed loads leave the
        confirmed state untouched.
        """
        if full_sync is None:
            full_sync = self.sync_scope == "full_sync"

        verdicts: list[DeletePolicyVerdict] = []
        present_keys = set()
        for row in rows:
            key = tuple(row[c] for c in key_columns) if len(key_columns) > 1 else row[key_columns[0]]
            present_keys.add(_normalize_key(key))
            verdicts.append(self.classify_row(row, key, tracker))

        if full_sync:
            if self.sync_scope != "full_sync":
                # a package claims completeness that the policy never opted into
                raise FullSyncDeclarationMissing(self.table_name or "<unknown>")
            for key in tracker.known_keys():
                if key not in present_keys:
                    # hard-deleted rows are re-reported too, which keeps repeated full syncs
                    # idempotent (their verdict resolves to the same hard delete)
                    verdicts.append(self.classify_absent(key, tracker, full_sync=True))
        else:
            # present_only: explicitly judge absent rows as undetermined rather than deleting
            for key in tracker.known_keys():
                if key not in present_keys:
                    verdicts.append(self.classify_absent(key, tracker, full_sync=False))

        return verdicts

    # ------------------------------------------------------------- load actions
    def soft_delete_update(self, verdict: DeletePolicyVerdict) -> dict[str, Any]:
        """Column values to set on the destination row for a soft delete verdict."""
        if not self.soft_delete_column_name:
            raise DeletePolicyConfigurationException(
                self.table_name,
                "cannot build soft delete update without `soft_delete_column_name`",
                field="soft_delete_column_name",
            )
        value = verdict.marker_value
        if value is None or not is_delete_marker_truthy(value):
            value = True
        return {self.soft_delete_column_name: value}
