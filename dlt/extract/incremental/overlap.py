"""Event-time overlap incremental mode.

Unlike the regular forward-only cursor, this mode keeps a *re-readable* lower bound
(`read_boundary`) that trails the high-water mark (`last_value`) by an `overlap_window`.
Every run re-reads the whole overlap window so that records whose event-time cursor is
updated late (the correction may carry an older cursor than the saved one) still reach
the destination.

Deduplication works on business primary key plus a version field (the cursor itself by
default). For every batch, each primary key is buffered and only the row carrying the
highest version of that key is compared against the version already committed in state:
- a version at or below the committed one is a duplicate re-read and gets dropped;
- a strictly newer version (including a late correction whose cursor is older than the
  high-water mark) is forwarded, and the destination upserts it via `merge`.

Buffering per key also makes the result independent of row ordering inside a batch.
The boundary only moves forward together with the committed pipeline state, so a failed
or repeated run re-reads the same window and cannot lose data.
"""

from datetime import datetime  # noqa: I251
from typing import Any, Dict, List, Optional, Tuple

from dlt.common.jsonpath import compile_path, extract_simple_field_name, find_values
from dlt.common.typing import TDataItem

from dlt.extract.incremental.transform import JsonIncremental
from dlt.extract.incremental.lag import _apply_lag_to_value, _cursor_date_type


def shift_by_window(
    overlap_window: float,
    value: Any,
    last_value_func: Any,
    cursor_type: Optional[type],
    initial_value: Any,
) -> Any:
    """Move `value` backwards (max) or forwards (min) by `overlap_window` and clamp it so
    the boundary never crosses `initial_value`.

    Window units follow the lag convention: seconds for datetime cursors, days for date
    cursors, plain subtraction for numeric cursors.
    """
    date_type = _cursor_date_type(cursor_type, initial_value)
    shifted = _apply_lag_to_value(overlap_window, value, last_value_func, date_type)
    if (
        initial_value is not None
        and last_value_func((initial_value, shifted)) == initial_value
        and shifted != initial_value
    ):
        # do not cross initial_value (same rule as lag.apply_lag)
        return initial_value
    return shifted


class OverlapIncremental(JsonIncremental):
    """Filters and deduplicates rows in event-time overlap mode. JSON/dict items only.

    Rows are buffered per primary key within a batch and flushed by the owning
    `Incremental` through `flush_batch()`, which emits at most one row per key.
    """

    def __init__(
        self,
        resource_name: str,
        cursor_path: str,
        initial_value: Optional[Any],
        read_boundary: Optional[Any],
        last_value_func: Any,
        primary_key: Any,
        overlap_window: float,
        version_path: Optional[str],
        on_cursor_value_missing: str,
        cursor_type: Optional[type] = None,
        high_water_mark: Optional[Any] = None,
        seen_versions: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        # base class expects the regular constructor signature; dedup hashes are not
        # used in overlap mode, so pass an empty set and neutral range settings
        super().__init__(
            resource_name=resource_name,
            cursor_path=cursor_path,
            initial_value=initial_value,
            start_value=read_boundary,
            end_value=None,
            last_value_func=last_value_func,
            primary_key=primary_key,
            unique_hashes=set(),
            on_cursor_value_missing=on_cursor_value_missing,
            lag=None,
            range_start="closed",
            range_end="open",
        )
        self.overlap_window = overlap_window
        self.version_path = version_path
        self.cursor_type = cursor_type
        # fixed re-read lower bound for this run (as committed in state at bind time)
        self.read_boundary = read_boundary
        self.read_boundary_is_datetime = isinstance(read_boundary, datetime)
        # the high-water mark resumes from the committed last_value (which sits ahead
        # of the re-read boundary), so a late, lower-cursor correction never rolls it back
        self.last_value = high_water_mark
        self.high_water_mark_is_datetime = isinstance(high_water_mark, datetime)
        # pk digest -> [version_value, cursor_value], persisted and bounded by eviction
        self.seen_versions: Dict[str, List[Any]] = dict(seen_versions or {})
        # boundary recomputed from the current high-water mark, used for write-back
        self.new_boundary = read_boundary
        # pk digest -> best (highest-version) row observed in the current batch
        self._batch_best: Dict[str, TDataItem] = {}
        # rows replaced by a higher-version row of the same key within this batch
        self._batch_suppressed = 0

        self._compiled_version_path = None
        if version_path:
            compiled = compile_path(version_path)
            if simple_field_name := extract_simple_field_name(compiled):
                self.version_path = simple_field_name
            else:
                self._compiled_version_path = compiled

    def find_version_value(self, row: TDataItem) -> Any:
        if self._compiled_version_path is not None:
            return find_values(self._compiled_version_path, row)[0]
        try:
            try:
                return row[self.version_path]
            except TypeError:
                return getattr(row, self.version_path)
        except (KeyError, AttributeError):
            return None

    def _is_beyond_boundary(self, row_value: Any) -> bool:
        """True when `row_value` lies on the already-committed side of the read boundary."""
        if self.read_boundary is None:
            return False
        return (
            self.last_value_func((row_value, self.read_boundary)) == self.read_boundary
            and row_value != self.read_boundary
        )

    def _compute_boundary(self) -> Optional[Any]:
        if self.last_value is None:
            return self.initial_value
        return shift_by_window(
            self.overlap_window,
            self.last_value,
            self.last_value_func,
            self.cursor_type,
            self.initial_value,
        )

    def _is_newer_version(self, candidate_version: Any, known_version: Any) -> bool:
        """True when `candidate_version` is strictly newer than `known_version`."""
        winner = self.last_value_func((candidate_version, known_version))
        return winner == candidate_version and candidate_version != known_version

    def evict(self, boundary: Optional[Any]) -> int:
        """Bound the version index to the re-read window.

        Entries with a cursor below `boundary` can no longer arrive through the re-read
        window (such rows are hard-filtered in `__call__`), so they are dropped. Keeping
        them would make `seen_versions` grow with every distinct key ever seen instead of
        staying proportional to the overlap window. A key that reappears later at an
        in-window cursor is simply upserted again by the destination's `merge`.

        Returns the number of index entries removed by this call.
        """
        if boundary is None:
            return 0
        in_window: Dict[str, List[Any]] = {}
        for pk_hash, entry in self.seen_versions.items():
            if self.last_value_func((entry[1], boundary)) == boundary and entry[1] != boundary:
                continue
            in_window[pk_hash] = entry
        evicted = len(self.seen_versions) - len(in_window)
        self.seen_versions = in_window
        return evicted

    def __call__(
        self, row: TDataItem
    ) -> Tuple[Optional[TDataItem], bool, bool]:
        """Buffer rows per primary key; nothing is emitted until `flush_batch()`."""
        if row is None:
            return row, False, False

        row_value = self.find_cursor_value(row)
        if row_value is None:
            if self.on_cursor_value_missing == "exclude":
                return None, False, False
            # rows without a cursor cannot participate in window tracking: pass them
            # straight through (the destination still upserts by primary key)
            return row, False, False

        # align tz-awareness of the persisted bounds with the incoming data
        if not self.seen_data and isinstance(row_value, datetime):
            if self.read_boundary_is_datetime:
                self.read_boundary = self.start_value = self._adapt_timezone(
                    row_value, self.read_boundary, "read_boundary", self.resource_name
                )
            if self.high_water_mark_is_datetime:
                self.last_value = self._adapt_timezone(
                    row_value, self.last_value, "last_value", self.resource_name
                )

        # 1. hard lower bound: rows older than the committed boundary are never re-read
        if self._is_beyond_boundary(row_value):
            return None, True, False

        self.seen_data = True

        # 2. advance the high-water mark over every in-window row, including ones that
        #    will be deduplicated below, so an all-duplicate batch never rolls it back
        if self.last_value is None:
            self.last_value = row_value
        else:
            self.last_value = self.last_value_func((self.last_value, row_value))
        # trail the boundary behind the high-water mark
        self.new_boundary = self._compute_boundary()

        # 3. keep only the highest-version row of each primary key in this batch
        pk_hash = self.compute_unique_value(row, self._primary_key)
        version_value = self.find_version_value(row) if self.version_path else row_value
        incumbent = self._batch_best.get(pk_hash)
        if incumbent is None:
            self._batch_best[pk_hash] = row
        else:
            incumbent_version = (
                self.find_version_value(incumbent)
                if self.version_path
                else self.find_cursor_value(incumbent)
            )
            if self._is_newer_version(version_value, incumbent_version):
                self._batch_best[pk_hash] = row
                # the displaced row of the same key never leaves this batch
                self._batch_suppressed += 1
        # buffering only: the actual emit decision happens at flush time
        return None, False, False

    def flush_batch(self) -> Tuple[List[TDataItem], int, int]:
        """Resolve the buffered rows against committed versions and return `(rows,
        duplicate_count, update_count)` for this batch.

        Rows returned must reach the destination (new keys and newer versions / late
        corrections). A duplicate is any row that does not make it past the committed
        version, including rows displaced by a higher version of the same key inside
        this batch. Counters are reset for the next batch afterwards.
        """
        emitted: List[TDataItem] = []
        duplicates = self._batch_suppressed
        updates = 0
        for pk_hash, row in self._batch_best.items():
            row_value = self.find_cursor_value(row)
            version_value = (
                self.find_version_value(row) if self.version_path else row_value
            )
            known = self.seen_versions.get(pk_hash)
            if known is not None and not self._is_newer_version(version_value, known[0]):
                # best version in the batch is still at or below what was committed
                duplicates += 1
                continue
            if known is not None:
                updates += 1
            self.seen_versions[pk_hash] = [version_value, row_value]
            emitted.append(row)
        self._batch_best.clear()
        self._batch_suppressed = 0
        return emitted, duplicates, updates
