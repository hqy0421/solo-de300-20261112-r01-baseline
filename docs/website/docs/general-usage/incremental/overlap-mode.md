---
title: Event-time overlap incremental mode
description: Re-read an overlapping window and apply late corrections without duplicate inserts
keywords: [incremental loading, overlap, late data, late correction, deduplication, merge, upsert]
---
# Event-time overlap incremental mode

Many order/CDC-style APIs do not return a clean, non-overlapping slice of data on every call:

- they re-return records from an **overlapping window** around the last pull, and
- a **late correction** may arrive whose event-time cursor (e.g. `updated_at`) is *older*
  than the high-water mark already saved from a previous run.

The regular forward-only incremental only reads rows with a cursor greater than the saved
`last_value`, so such corrections are filtered out. The opt-in **event-time overlap mode**
(`overlap_mode=True`) handles both problems while guaranteeing exactly-once upserts.

## How it works

Instead of reading from the high-water mark, the incremental keeps a second, **re-readable
lower boundary** called `read_boundary`, which trails the high-water mark by
`overlap_window`:

- every run re-reads the whole window `[read_boundary, last_value]`, so late corrections
  whose cursor falls inside that window reach the destination;
- rows are deduplicated by **business primary key plus a version** (the cursor value by
  default, or an explicit `version_path`), so overlapping re-reads never produce duplicate
  inserts — only a strictly newer version of a key is forwarded;
- forwarded rows are meant to be upserted, so use the resource with
  `write_disposition="merge"` and a `primary_key`;
- `read_boundary` only advances together with the successfully committed pipeline state.
  A failed, restarted, or repeated run re-reads the same window and cannot lose data.

The high-water mark (`last_value`) keeps its usual monotonic, forward-only semantics: a
late, lower-cursor correction never rolls it back.

## Configuration

`overlap_mode` is disabled by default and, when left off, the incremental behaves exactly
as before. Enable it with:

- `overlap_mode=True`
- `overlap_window`: size of the re-read window. Seconds for datetime cursors, days for date
  cursors, plain cursor units for numeric cursors.
- a business `primary_key` (on the incremental or the resource) used for deduplication.
- optionally `version_path`: name/JSON path of a monotonic record version. When omitted,
  the cursor itself is the version — use an explicit version field if a correction can keep
  the same cursor or arrive with an older cursor than the previously seen version.

It cannot be combined with `end_value`, `lag`, an `open` start range, empty-tuple
primary keys, or custom (non `min`/`max`) `last_value_func`; configuring these raises an
`IncrementalOverlapConfigurationError`. Arrow/DataFrame and `Relation` items are not
supported yet — yield plain JSON items (dicts / Pydantic models).

## Example

```py
import dlt

@dlt.resource(primary_key="order_id", write_disposition="merge")
def orders(
    updated_at=dlt.sources.incremental(
        "updated_at",
        initial_value="2026-01-01T00:00:00Z",
        overlap_mode=True,
        overlap_window=3600,          # re-read the last hour of event time
        primary_key="order_id",
        version_path="record_version",  # monotonic source version (optional)
    )
):
    # the API may re-return the overlapping window and send late corrections
    yield from fetch_orders(since=updated_at.start_value)

pipeline.run(orders())
```

On each run:

- unchanged rows re-read from the overlap window are recognized by primary key + version
  and dropped;
- a newer version of an already-seen key (including one with an older `updated_at`) is
  forwarded and upserted by the `merge` disposition;
- rows below `read_boundary` are hard-filtered, and the in-memory version index is bounded
  to the window so state stays small.

## Persisted state

In addition to the regular `last_value` / `start_value`, overlap mode stores two fields in
the resource's incremental state:

- `read_boundary`: the committed re-read lower bound;
- `seen_versions`: a bounded map of primary-key digest to the latest seen
  `[version, cursor]` pair, used to recognize duplicate re-reads and out-of-order
  corrections.

Both are plain JSON-serializable values and evolve only via the normal pipeline state
commit, which is what makes restarts and repeated runs safe.

## Observability

The extract metrics report three overlap counters per incremental:

- `overlap_duplicates_count` — rows suppressed as already-committed re-reads;
- `overlap_late_updates_count` — keys forwarded because a strictly newer version arrived;
- `overlap_evicted_versions_count` — index entries dropped once they fell behind the re-read window.
