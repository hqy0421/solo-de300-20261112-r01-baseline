"""Tests for the opt-in event-time overlap incremental mode (`overlap_mode=True`)."""

from datetime import datetime, timezone  # noqa: I251
from typing import Any, Dict, List, Optional

import pytest

import dlt
from dlt.common.configuration.container import Container
from dlt.common.pipeline import StateInjectableContext
from dlt.common.utils import uniq_id

from dlt.extract.incremental import Incremental, IncrementalResourceWrapper
from dlt.extract.incremental.exceptions import (
    IncrementalOverlapConfigurationError,
    IncrementalOverlapUnsupportedItemType,
    IncrementalPrimaryKeyMissing,
)
from dlt.extract.incremental.overlap import shift_by_window


CURSOR = "updated_at"


def _run_resource(
    state: Dict[str, Any],
    rows: List[Dict[str, Any]],
    *,
    cursor_path: str = CURSOR,
    initial_value: Any = 0,
    overlap_window: float = 10,
    version_path: Optional[str] = None,
    last_value_func: Any = max,
    primary_key: Any = "id",
    resource_name: str = None,
) -> List[Dict[str, Any]]:
    """Bind an overlap incremental to a fresh resource over shared in-memory state."""
    # a unique name keeps each call's state isolated inside the shared test section
    resource_name = resource_name or f"orders_{uniq_id()}"

    @dlt.resource(name=resource_name, primary_key=primary_key)
    def resource(
        cursor=dlt.sources.incremental(
            cursor_path,
            initial_value=initial_value,
            overlap_mode=True,
            overlap_window=overlap_window,
            primary_key=primary_key,
            version_path=version_path,
            last_value_func=last_value_func,
        )
    ):
        yield from rows

    with Container().injectable_context(StateInjectableContext(state=state)):
        bound = resource()
        return list(bound)


def _incr_state(state: Dict[str, Any], resource_name: str = None) -> Dict[str, Any]:
    source_name = next(iter(state["sources"]))
    resources = state["sources"][source_name]["resources"]
    if resource_name is None:
        # single-resource state: take the only incremental cursor state present
        incr = next(iter(resources.values()))["incremental"][CURSOR]
        return incr
    return resources[resource_name]["incremental"][CURSOR]


def test_overlap_mode_is_opt_in() -> None:
    incr = Incremental(CURSOR)
    assert incr.overlap_mode is False


def test_overlap_requires_window() -> None:
    incr = Incremental(CURSOR, overlap_mode=True, primary_key="id")
    with pytest.raises(IncrementalOverlapConfigurationError):
        incr.resolve()


def test_overlap_rejects_incompatible_options() -> None:
    with pytest.raises(IncrementalOverlapConfigurationError):
        Incremental(
            CURSOR,
            initial_value=0,
            overlap_mode=True,
            overlap_window=10,
            end_value=100,
            primary_key="id",
        ).resolve()
    with pytest.raises(IncrementalOverlapConfigurationError):
        Incremental(
            CURSOR, overlap_mode=True, overlap_window=10, lag=5, primary_key="id"
        ).resolve()
    with pytest.raises(IncrementalOverlapConfigurationError):
        Incremental(
            CURSOR,
            overlap_mode=True,
            overlap_window=10,
            range_start="open",
            primary_key="id",
        ).resolve()
    with pytest.raises(IncrementalOverlapConfigurationError):
        Incremental(
            CURSOR,
            overlap_mode=True,
            overlap_window=10,
            last_value_func=lambda vals: vals[0],
            primary_key="id",
        ).resolve()
    with pytest.raises(IncrementalOverlapConfigurationError):
        Incremental(
            CURSOR, overlap_mode=True, overlap_window=10, primary_key=()
        ).resolve()


def test_shift_by_window_numeric_and_datetime() -> None:
    assert shift_by_window(10, 50, max, int, 0) == 40
    # boundary never crosses the initial value
    assert shift_by_window(100, 50, max, int, 10) == 10
    assert shift_by_window(10, 50, min, int, 100) == 60
    assert shift_by_window(100, 50, min, int, 100) == 100

    t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    boundary = shift_by_window(3600, t, max, datetime, None)
    assert boundary == datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc)


def test_initial_run_loads_everything_and_persists_boundary() -> None:
    state: Dict[str, Any] = {}
    rows = [
        {"id": "a", CURSOR: 5},
        {"id": "b", CURSOR: 12},
        {"id": "c", CURSOR: 20},
    ]
    out = _run_resource(state, rows)
    assert [r["id"] for r in out] == ["a", "b", "c"]

    s = _incr_state(state)
    assert s["last_value"] == 20
    # boundary trails the high-water mark by the window
    assert s["read_boundary"] == 10
    # every key keeps at least its latest committed version, keyed by pk digest
    assert set(s["seen_versions"])
    assert all(len(entry) == 2 for entry in s["seen_versions"].values())


def test_overlapping_reread_deduplicates_and_applies_late_correction() -> None:
    state: Dict[str, Any] = {}
    name = f"r_{uniq_id()}"
    _run_resource(
        state,
        [
            {"id": "a", CURSOR: 5, "v": 1},
            {"id": "b", CURSOR: 12, "v": 1},
            {"id": "c", CURSOR: 20, "v": 1},
        ],
        resource_name=name,
    )

    # the API re-returns the overlapping window; b is corrected at an older event time
    # than the saved high-water mark (20), and d is brand new
    out = _run_resource(
        state,
        [
            {"id": "b", CURSOR: 12, "v": 1},  # unchanged duplicate
            {"id": "b", CURSOR: 15, "v": 2},  # late correction, newer cursor version
            {"id": "c", CURSOR: 20, "v": 1},  # duplicate at the boundary
            {"id": "d", CURSOR: 25, "v": 1},  # new row
        ],
        resource_name=name,
    )
    assert [(r["id"], r["v"]) for r in out] == [("b", 2), ("d", 1)]

    s = _incr_state(state, name)
    assert s["last_value"] == 25
    assert s["read_boundary"] == 15
    # every surviving entry still holds its latest committed [version, cursor] pair;
    # keys that left the re-read window (a) were evicted to keep the index bounded
    assert all(len(entry) == 2 for entry in s["seen_versions"].values())


def test_repeated_run_over_same_window_emits_nothing() -> None:
    """Restart / repeated run with the committed state must not duplicate inserts."""
    state: Dict[str, Any] = {}
    name = f"r_{uniq_id()}"
    rows = [
        {"id": "b", CURSOR: 15, "v": 2},
        {"id": "c", CURSOR: 20, "v": 1},
        {"id": "d", CURSOR: 25, "v": 1},
    ]
    _run_resource(state, rows, resource_name=name)
    # rerun with the exact same committed state and identical API response
    out = _run_resource(state, rows, resource_name=name)
    assert out == []
    s = _incr_state(state, name)
    assert s["last_value"] == 25


def test_rows_below_read_boundary_are_hard_filtered() -> None:
    state: Dict[str, Any] = {}
    name = f"r_{uniq_id()}"
    _run_resource(state, [{"id": "a", CURSOR: 20}], resource_name=name)
    assert _incr_state(state, name)["read_boundary"] == 10

    out = _run_resource(
        state,
        [
            {"id": "a", CURSOR: 6, "v": 9},  # outside the re-read window
            {"id": "e", CURSOR: 30, "v": 1},
        ],
        resource_name=name,
    )
    assert [r["id"] for r in out] == ["e"]


def test_explicit_version_path_drives_deduplication() -> None:
    """With an independent version field, a late correction wins even if its cursor
    value is unchanged."""
    state: Dict[str, Any] = {}
    name = f"r_{uniq_id()}"
    out = _run_resource(
        state,
        [{"id": "a", CURSOR: 12, "version": 1, "v": "old"}],
        version_path="version",
        resource_name=name,
    )
    assert len(out) == 1

    # same cursor, bumped version -> forwarded as an update
    out = _run_resource(
        state,
        [{"id": "a", CURSOR: 12, "version": 2, "v": "new"}],
        version_path="version",
        resource_name=name,
    )
    assert len(out) == 1
    assert out[0]["v"] == "new"

    # re-reading the old version is dropped even though its cursor is in the window
    out = _run_resource(
        state,
        [{"id": "a", CURSOR: 12, "version": 1, "v": "old"}],
        version_path="version",
        resource_name=name,
    )
    assert out == []


def test_late_correction_older_than_high_water_mark_passes() -> None:
    """The headline case: a correction whose cursor (18) is older than the saved
    high-water mark (25) still reaches the destination, recognized via a monotonic
    independent version field."""
    state: Dict[str, Any] = {}
    name = f"r_{uniq_id()}"
    _run_resource(
        state,
        [
            {"id": "a", CURSOR: 20, "version": 1},
            {"id": "b", CURSOR: 25, "version": 1},
        ],
        version_path="version",
        resource_name=name,
    )
    assert _incr_state(state, name)["read_boundary"] == 15

    out = _run_resource(
        state,
        [
            # correction inside the window even though far below the high-water mark
            {"id": "a", CURSOR: 18, "version": 2, "v": 2},
        ],
        version_path="version",
        resource_name=name,
    )
    assert len(out) == 1
    assert out[0]["v"] == 2
    # high-water mark is unchanged but the correction is still forwarded
    assert _incr_state(state, name)["last_value"] == 25

    # the same correction re-read on a restart/rerun is not forwarded twice
    out = _run_resource(
        state,
        [{"id": "a", CURSOR: 18, "version": 2, "v": 2}],
        version_path="version",
        resource_name=name,
    )
    assert out == []


def test_unordered_batch_keeps_highest_version_per_key() -> None:
    """Within a batch the older row arrives after the newer one; only the highest
    version of the key is emitted, regardless of ordering."""
    state: Dict[str, Any] = {}
    name = f"r_{uniq_id()}"
    _run_resource(
        state, [{"id": "a", CURSOR: 12, "version": 1}],
        version_path="version", resource_name=name,
    )
    # note the descending order: old version first would wrongly win under naive dedup
    out = _run_resource(
        state,
        [
            {"id": "a", CURSOR: 15, "version": 3, "v": "third"},
            {"id": "a", CURSOR: 14, "version": 2, "v": "second"},
        ],
        version_path="version",
        resource_name=name,
    )
    assert len(out) == 1
    assert out[0]["version"] == 3 and out[0]["v"] == "third"


def test_min_last_value_func_mirrors_window() -> None:
    state: Dict[str, Any] = {}
    name = f"r_{uniq_id()}"
    _run_resource(
        state,
        [{"id": "a", CURSOR: 100}, {"id": "b", CURSOR: 80}],
        initial_value=200,
        last_value_func=min,
        resource_name=name,
    )
    s = _incr_state(state, name)
    assert s["last_value"] == 80
    # min cursor: the boundary trails *upwards*, i.e. last_value + window
    assert s["read_boundary"] == 90

    out = _run_resource(
        state,
        [
            {"id": "a", CURSOR: 95},  # inside window, duplicate
            {"id": "c", CURSOR: 70},  # new minimum
        ],
        initial_value=200,
        last_value_func=min,
        resource_name=name,
    )
    assert [r["id"] for r in out] == ["c"]
    assert _incr_state(state, name)["read_boundary"] == 80


def test_metrics_count_updates_and_duplicates() -> None:
    state: Dict[str, Any] = {}
    name = f"r_{uniq_id()}"
    # first run commits version 1 of key 'a'
    _run_resource(
        state, [{"id": "a", CURSOR: 12, "version": 1}],
        version_path="version", resource_name=name,
    )

    # second extraction re-reads the committed version and a correction, plus a new key
    @dlt.resource(name=name, primary_key="id")
    def resource(
        updated_at=dlt.sources.incremental(
            CURSOR,
            initial_value=0,
            overlap_mode=True,
            overlap_window=10,
            primary_key="id",
            version_path="version",
        )
    ):
        yield [
            {"id": "a", CURSOR: 12, "version": 1},  # committed re-read, displaced by v2 below
            {"id": "a", CURSOR: 15, "version": 2},  # late correction: update
            {"id": "b", CURSOR: 16, "version": 1},  # new key
        ]

    with Container().injectable_context(StateInjectableContext(state=state)):
        bound = resource()
        list(bound)
        wrapper = bound._pipe.get_by_type(IncrementalResourceWrapper)
        metrics = wrapper._incremental._custom_metrics["incremental"][0]
    # the v1 row is suppressed in-batch by v2; one key is corrected against state
    assert metrics["overlap_duplicates_count"] == 1
    assert metrics["overlap_late_updates_count"] == 1


def test_eviction_metric_is_counted_per_batch_without_double_counting() -> None:
    """The eviction counter reports entries dropped from the bounded index by each batch,
    and never re-adds an earlier batch's total."""
    state: Dict[str, Any] = {}
    name = f"r_{uniq_id()}"
    # first run fills the index with keys spread across the whole window; its own
    # extraction already evicts the keys below the committed boundary (a, b)
    _run_resource(
        state,
        [
            {"id": "a", CURSOR: 1},
            {"id": "b", CURSOR: 5},
            {"id": "c", CURSOR: 20},
        ],
        resource_name=name,
    )
    first_state = _incr_state(state, name)
    assert first_state["read_boundary"] == 10
    assert sorted(e[1] for e in first_state["seen_versions"].values()) == [20]

    @dlt.resource(name=name, primary_key="id")
    def resource(
        updated_at=dlt.sources.incremental(
            CURSOR,
            initial_value=0,
            overlap_mode=True,
            overlap_window=10,
            primary_key="id",
        )
    ):
        # three batches inside one extraction, each advancing the window past old keys:
        # boundary 10 -> 20 (c leaves) -> 30 -> 40 (d leaves)
        yield [{"id": "d", CURSOR: 30}]
        yield [{"id": "e", CURSOR: 40}]
        yield [{"id": "f", CURSOR: 50}]

    with Container().injectable_context(StateInjectableContext(state=state)):
        bound = resource()
        list(bound)
        wrapper = bound._pipe.get_by_type(IncrementalResourceWrapper)
        metric = wrapper._incremental._custom_metrics["incremental"][0][
            "overlap_evicted_versions_count"
        ]
        incr_state = _incr_state(state, name)
    # c leaves in batch 1 and d in batch 3; every key is counted exactly once
    assert metric == 2
    # only keys inside the current re-read window remain in the bounded index
    remaining_cursors = sorted(entry[1] for entry in incr_state["seen_versions"].values())
    assert remaining_cursors == [40, 50]


def test_rows_with_null_cursor_follow_on_cursor_value_missing_policy() -> None:
    """Rows with an explicit `None` cursor cannot participate in window tracking: they are
    passed through under `include` and dropped under `exclude`."""
    state: Dict[str, Any] = {}
    name = f"r_{uniq_id()}"

    @dlt.resource(name=name, primary_key="id")
    def included(
        updated_at=dlt.sources.incremental(
            CURSOR,
            initial_value=0,
            overlap_mode=True,
            overlap_window=10,
            primary_key="id",
            on_cursor_value_missing="include",
        )
    ):
        yield [
            {"id": "a", CURSOR: 12},
            {"id": "z", CURSOR: None},  # null cursor, still forwarded for a merge upsert
        ]

    with Container().injectable_context(StateInjectableContext(state=state)):
        out = list(included())
    # passthrough rows are emitted before the buffered/flushed rows; compare as a set
    assert {r["id"] for r in out} == {"a", "z"}

    state_ex: Dict[str, Any] = {}

    @dlt.resource(name=f"exc_{uniq_id()}", primary_key="id")
    def excluded(
        updated_at=dlt.sources.incremental(
            CURSOR,
            initial_value=0,
            overlap_mode=True,
            overlap_window=10,
            primary_key="id",
            on_cursor_value_missing="exclude",
        )
    ):
        yield [{"id": "a", CURSOR: 12}, {"id": "z", CURSOR: None}]

    with Container().injectable_context(StateInjectableContext(state=state_ex)):
        out = list(excluded())
    assert [r["id"] for r in out] == ["a"]


def test_arrow_items_are_rejected_with_clear_error() -> None:
    import pyarrow as pa

    state: Dict[str, Any] = {}

    @dlt.resource(primary_key="id")
    def orders(
        updated_at=dlt.sources.incremental(
            CURSOR, initial_value=0, overlap_mode=True, overlap_window=10, primary_key="id"
        )
    ):
        yield pa.table({"id": ["a"], CURSOR: [12]})

    with Container().injectable_context(StateInjectableContext(state=state)):
        with pytest.raises(IncrementalOverlapUnsupportedItemType):
            list(orders())


def test_missing_primary_key_raises() -> None:
    state: Dict[str, Any] = {}

    @dlt.resource
    def orders(
        updated_at=dlt.sources.incremental(
            CURSOR, initial_value=0, overlap_mode=True, overlap_window=10
        )
    ):
        yield [{"id": "a", CURSOR: 12}]

    with Container().injectable_context(StateInjectableContext(state=state)):
        with pytest.raises(IncrementalPrimaryKeyMissing):
            list(orders())


def test_merge_keeps_overlap_mode_enabled() -> None:
    on = Incremental(CURSOR, overlap_mode=True, overlap_window=10, version_path="version")
    off = Incremental(initial_value=100)
    assert on.merge(off).overlap_mode is True
    assert off.merge(on).overlap_mode is True
    assert off.merge(Incremental(end_value=5, initial_value=1)).overlap_mode is False


def test_end_to_end_merge_upserts_late_correction_in_duckdb() -> None:
    """Full pipeline with `write_disposition='merge'`: a late correction updates the
    target row in place instead of inserting a duplicate."""
    import duckdb

    p = dlt.pipeline(
        pipeline_name="p" + uniq_id(),
        destination=dlt.destinations.duckdb(credentials=duckdb.connect(":memory:")),
    )

    @dlt.resource(primary_key="id", write_disposition="merge")
    def orders(
        updated_at=dlt.sources.incremental(
            CURSOR,
            initial_value=0,
            overlap_mode=True,
            overlap_window=10,
            primary_key="id",
            version_path="version",
        )
    ):
        # the API always re-returns everything at or behind the re-read boundary;
        # run 1 returns the initial backfill, later runs add the correction and new rows
        if updated_at.last_value is None:
            yield [
                {"id": "a", CURSOR: 12, "version": 1, "v": 1},
                {"id": "b", CURSOR: 20, "version": 1, "v": 1},
            ]
        else:
            # both backfill rows (cursor >= boundary 10) are re-read; 'b' also arrives
            # as a late correction: newer version but an event-time cursor (15) older
            # than the saved high-water mark (20); 'c' is brand new
            yield [
                {"id": "a", CURSOR: 12, "version": 1, "v": 1},
                {"id": "b", CURSOR: 20, "version": 1, "v": 1},
                {"id": "b", CURSOR: 15, "version": 2, "v": 2},
                {"id": "c", CURSOR: 25, "version": 1, "v": 1},
            ]

    p.run(orders())
    p.run(orders())

    with p.sql_client() as c:
        with c.execute_query(f"SELECT id, {CURSOR}, version, v FROM orders ORDER BY id") as cur:
            rows = cur.fetchall()
    # exactly one row per business key, b holds the corrected value
    assert rows == [("a", 12, 1, 1), ("b", 15, 2, 2), ("c", 25, 1, 1)]

    # a third, identical run changes nothing in the destination
    p.run(orders())
    with p.sql_client() as c:
        with c.execute_query("SELECT count(*) FROM orders") as cur:
            assert cur.fetchone()[0] == 3


def test_default_mode_semantics_unchanged() -> None:
    """Without overlap_mode, rows below the high-water mark are filtered as before, and
    no overlap-specific state is written."""
    state: Dict[str, Any] = {}

    @dlt.resource(name="plain_orders", primary_key="id")
    def plain_orders(rows, updated_at=dlt.sources.incremental(CURSOR, initial_value=0)):
        yield from rows

    with Container().injectable_context(StateInjectableContext(state=state)):
        # first run advances the high-water mark to 20
        list(plain_orders([{"id": "a", CURSOR: 20}]))
        # regular mode: a late correction at cursor 15 (below 20) is excluded
        out = list(
            plain_orders(
                [
                    {"id": "a", CURSOR: 15, "v": 2},
                    {"id": "b", CURSOR: 25, "v": 1},
                ]
            )
        )
    assert [r["id"] for r in out] == ["b"]
    s = _incr_state(state, "plain_orders")
    assert "read_boundary" not in s
    assert "seen_versions" not in s
