from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeVar, Union

from dlt.common.typing import TSortOrder, TTableHintTemplate, TColumnNames, TypedDict

TCursorValue = TypeVar("TCursorValue", bound=Any)
LastValueFunc = Callable[[Sequence[TCursorValue]], Any]
OnCursorValueMissing = Literal["raise", "include", "exclude"]

TIncrementalRange = Literal["open", "closed"]


class IncrementalColumnState(TypedDict, total=False):
    initial_value: Optional[Any]
    last_value: Optional[Any]
    unique_hashes: List[str]
    start_value: Optional[Any]
    # event-time overlap mode (see `Incremental(overlap_mode=True)`)
    read_boundary: Optional[Any]
    """Lowest event-time cursor value still re-read on the next run. Rows below it are
    guaranteed to have been committed, so the boundary can move forward safely."""
    seen_versions: Dict[str, List[Any]]
    """Bounded index of rows inside the overlap window, keyed by the primary key digest.
    Each value is `[version_value, cursor_value]` (a list so it stays JSON serializable).
    Entries whose cursor falls below `read_boundary` are evicted on every write-back."""


class IncrementalArgs(TypedDict, total=False):
    cursor_path: str
    initial_value: Optional[Any]
    last_value_func: Optional[Union[LastValueFunc[str], Literal["min", "max"]]]
    """Last value callable or name of built in function"""
    primary_key: Optional[TTableHintTemplate[TColumnNames]]
    end_value: Optional[Any]
    row_order: Optional[TSortOrder]
    allow_external_schedulers: Optional[bool]
    lag: Optional[Union[float, int]]
    on_cursor_value_missing: Optional[OnCursorValueMissing]
    range_start: Optional[TIncrementalRange]
    range_end: Optional[TIncrementalRange]
    overlap_mode: Optional[bool]
    overlap_window: Optional[Union[float, int]]
    version_path: Optional[str]
