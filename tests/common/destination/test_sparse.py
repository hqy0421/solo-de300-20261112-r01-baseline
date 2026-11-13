"""Unit tests for sparse update normalization and the shared sparse marker helpers."""
from dlt.common.destination.sparse import (
    SPARSE_LIST_PATHS_KEY,
    SPARSE_PRESENCE_KEY,
    SPARSE_SEQ_KEY,
    get_sparse_context,
    is_mask_column,
    is_sparse_system_column,
    list_path_marker_name,
    make_mask_column,
    mask_column_name,
    new_sparse_context,
    sparse_data_columns,
    strip_sparse_context,
    unmask_column_name,
)
from dlt.common.normalizers.json.relational import (
    DataItemNormalizer as RelationalNormalizer,
)
from dlt.common.schema import Schema
from dlt.common.schema.typing import TColumnSchema, TTableSchema


def _sparse_schema() -> Schema:
    schema = Schema("sparse_unit")
    table: TTableSchema = {
        "name": "customers",
        "write_disposition": "merge",
        "x-merge-strategy": "upsert",
        "x-merge-sparse-update": True,
        "columns": {
            "id": TColumnSchema(
                name="id", data_type="bigint", nullable=False, primary_key=True
            ),
            "name": TColumnSchema(name="name", data_type="text", nullable=True),
            "phone": TColumnSchema(name="phone", data_type="text", nullable=True),
        },
    }
    schema.update_table(table)
    return schema


def test_sparse_context_lifecycle() -> None:
    ctx = new_sparse_context(7)
    assert ctx[SPARSE_SEQ_KEY] == 7
    assert ctx[SPARSE_PRESENCE_KEY] == set()
    assert ctx[SPARSE_LIST_PATHS_KEY] == set()

    row = {**ctx, "id": 1}
    assert get_sparse_context(row) is not None
    # a nested row carrying only the list-path marker is still a sparse row
    nested = {SPARSE_LIST_PATHS_KEY: {("contacts",)}}
    assert get_sparse_context(nested) is not None
    assert get_sparse_context({"id": 1}) is None

    strip_sparse_context(row)
    assert SPARSE_SEQ_KEY not in row
    assert SPARSE_PRESENCE_KEY not in row
    assert SPARSE_LIST_PATHS_KEY not in row


def test_mask_column_name_helpers() -> None:
    mask = mask_column_name("phone")
    assert is_mask_column(mask)
    assert unmask_column_name(mask) == "phone"
    assert make_mask_column("phone")["x-sparse-mask"] is True
    assert is_sparse_system_column(mask)
    assert is_sparse_system_column("_dlt_sparse_seq")
    assert is_sparse_system_column(list_path_marker_name("customers__contacts"))
    assert not is_sparse_system_column("phone")


def test_sparse_data_columns_exclude_markers() -> None:
    schema = _sparse_schema()
    table = schema.tables["customers"]
    table["columns"][mask_column_name("name")] = make_mask_column("name")
    assert "name" in sparse_data_columns(table)
    assert mask_column_name("name") not in sparse_data_columns(table)


def test_flatten_sparse_keeps_explicit_null_and_records_presence() -> None:
    norm: RelationalNormalizer = _sparse_schema().data_item_normalizer  # type: ignore[assignment]
    presence: set = set()
    flat, lists = norm._flatten_sparse(
        "customers", {"id": 1, "name": "Alice", "phone": None}, 1000, presence
    )
    # explicit null survives flattening (a regular flatten drops it)
    assert flat["phone"] is None
    assert flat["name"] == "Alice"
    # every provided field, including the null one, is in the presence set
    assert {"id", "name", "phone"} <= presence
    assert lists == {}


def test_normalize_data_item_threads_context_and_list_paths() -> None:
    schema = _sparse_schema()
    norm: RelationalNormalizer = schema.data_item_normalizer  # type: ignore[assignment]

    item = {
        "id": 1,
        "phone": None,
        "contacts": [{"kind": "email", "value": "a@b.c"}],
    }
    rows = {}
    for (table_name, _, _), row in norm.normalize_data_item(item, "load_1", "customers"):
        rows.setdefault(table_name, []).append(row)

    root = rows["customers"][0]
    ctx = get_sparse_context(root)
    assert ctx is not None
    assert ctx[SPARSE_SEQ_KEY] == 1
    # phone provided as null is still considered present
    assert "phone" in ctx[SPARSE_PRESENCE_KEY]
    # the list path was recorded and nested rows carry it
    assert ("contacts",) in ctx[SPARSE_LIST_PATHS_KEY]
    nested = rows["customers__contacts"][0]
    assert ("contacts",) in get_sparse_context(nested)[SPARSE_LIST_PATHS_KEY]

    # a following regular table item must not inherit the sparse context
    state_rows = list(
        norm.normalize_data_item({"version": 1}, "load_1", "_dlt_pipeline_state")
    )
    for _, state_row in state_rows:
        assert get_sparse_context(state_row) is None


def test_normalize_same_key_patches_get_monotonic_sequence() -> None:
    schema = _sparse_schema()
    norm: RelationalNormalizer = schema.data_item_normalizer  # type: ignore[assignment]
    seqs = []
    for item in ({"id": 1, "name": "A"}, {"id": 1, "name": "B"}, {"id": 1, "name": "C"}):
        for _, row in norm.normalize_data_item(item, "load_1", "customers"):
            seqs.append(row[SPARSE_SEQ_KEY])
    assert seqs == [1, 2, 3]
