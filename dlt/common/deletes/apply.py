"""Applying delete policy verdicts to data packages.

`classify_package` only classifies rows. This module turns the verdicts into concrete row
operations that a resource can yield:

- `retain` / `undetermined` rows pass through untouched,
- `soft_delete` rows get the configured destination flag column set,
- `hard_delete` rows are removed from the yielded data and returned separately, so the caller
  can issue physical deletes by key.

The caller remains in charge of persisting `DeleteTracker` state *after* the load succeeds.
"""

from typing import Any, Mapping, Sequence

from dlt.common.deletes.policy import DeletePolicy, DeletePolicyVerdict
from dlt.common.deletes.state import DeleteTracker, _normalize_key


class ProcessedPackage:
    """Result of processing one data package through a delete policy."""

    def __init__(self, verdicts: Sequence[DeletePolicyVerdict]) -> None:
        self.verdicts = list(verdicts)
        self.rows_to_yield: list[dict[str, Any]] = []
        """Rows to load: retained, undetermined and soft-deleted (flagged) rows. Undetermined
        rows are passed through unchanged, since a disabled policy must not alter data flow;
        callers that want strict semantics skip the keys in `undetermined_keys` themselves."""
        self.hard_deleted_keys: list[Any] = []
        """Keys that must be physically deleted from the destination."""
        self.soft_deleted_keys: list[Any] = []
        """Keys whose destination row must carry the soft delete flag."""
        self.undetermined_keys: list[Any] = []
        """All keys the policy could not judge (present rows and absent records)."""

    @property
    def has_deletes(self) -> bool:
        return bool(self.hard_deleted_keys or self.soft_deleted_keys)


def process_package(
    policy: DeletePolicy,
    rows: Sequence[Mapping[str, Any]],
    key_columns: Sequence[str],
    tracker: DeleteTracker[Any],
    *,
    full_sync: bool | None = None,
) -> ProcessedPackage:
    """Classifies `rows` and applies the verdicts, without mutating the tracker.

    Returns a `ProcessedPackage` with the rows the resource should yield (soft-deleted rows
    included, with their flag column populated) and the keys requiring physical deletion.
    Pass the verdicts to `tracker.confirm_verdict(...)` only after the load commits.
    """
    verdicts = policy.classify_package(rows, key_columns, tracker, full_sync=full_sync)
    by_key = {_normalize_key(v.record_key): v for v in verdicts}
    processed = ProcessedPackage(verdicts)
    seen_hard: set[Any] = set()
    seen_soft: set[Any] = set()
    seen_undetermined: set[Any] = set()

    for row in rows:
        key = (
            tuple(row[c] for c in key_columns)
            if len(key_columns) > 1
            else row[key_columns[0]]
        )
        norm_key = _normalize_key(key)
        verdict = by_key[norm_key]
        out_row = dict(row)
        if verdict.verdict == "hard_delete":
            # do not yield a row that must be physically removed
            processed.hard_deleted_keys.append(key)
            seen_hard.add(norm_key)
            continue
        if verdict.verdict == "soft_delete":
            out_row.update(policy.soft_delete_update(verdict))
            processed.soft_deleted_keys.append(key)
            seen_soft.add(norm_key)
        elif verdict.verdict == "undetermined":
            processed.undetermined_keys.append(key)
            seen_undetermined.add(norm_key)
        processed.rows_to_yield.append(out_row)

    # absent-record verdicts only exist for keys not present in `rows`
    for verdict in verdicts:
        norm_key = _normalize_key(verdict.record_key)
        if verdict.verdict == "hard_delete" and norm_key not in seen_hard:
            processed.hard_deleted_keys.append(verdict.record_key)
        elif verdict.verdict == "soft_delete" and norm_key not in seen_soft:
            processed.soft_deleted_keys.append(verdict.record_key)
        elif verdict.verdict == "undetermined" and norm_key not in seen_undetermined:
            processed.undetermined_keys.append(verdict.record_key)

    return processed
