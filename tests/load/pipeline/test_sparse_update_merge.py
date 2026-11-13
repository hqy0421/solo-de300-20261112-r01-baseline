"""Tests for the optional sparse update (partial patch) merge loading.

A sparse update treats staged rows as patches: a field missing from a row keeps its destination
value while an explicit `None` clears it. Explicitly provided nested lists replace the
destination list items, omitted lists are preserved, and several patches of the same key in one
batch take effect in their business version order.
"""
from typing import Any, Dict, List

import pytest

import dlt
from dlt.common.destination.sparse import is_sparse_system_column
from dlt.common.schema.exceptions import SchemaCorruptedException
from dlt.pipeline.exceptions import PipelineStepFailed
from tests.load.utils import destinations_configs, DestinationTestConfiguration


SPARSE_UPSERT = {"disposition": "merge", "strategy": "upsert", "sparse_update": True}


def _table_dicts(pipeline: dlt.Pipeline, table: str) -> List[Dict[str, Any]]:
    with pipeline.sql_client() as client:
        with client.execute_query(
            f"SELECT * FROM {client.make_qualified_table_name(table)}"
        ) as cur:
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def _table_columns(pipeline: dlt.Pipeline, table: str) -> List[str]:
    with pipeline.sql_client() as client:
        with client.execute_query(
            f"SELECT * FROM {client.make_qualified_table_name(table)}"
        ) as cur:
            return [d[0] for d in cur.description]


@pytest.mark.parametrize(
    "destination_config",
    destinations_configs(default_sql_configs=True, subset=["duckdb"]),
    ids=lambda x: x.name,
)
def test_sparse_root_patch_keeps_omitted_and_clears_null(
    destination_config: DestinationTestConfiguration,
) -> None:
    """Missing fields keep destination values; explicit null clears; same-batch patches fold."""
    pipeline = destination_config.setup_pipeline("sparse_root_patch", dev_mode=True)
    pipeline.run(
        [
            {"id": 1, "name": "Alice", "phone": "123", "address": {"city": "Rome"}},
            {"id": 2, "name": "Bob", "phone": "456", "address": {"city": "Paris"}},
        ],
        table_name="customers",
        primary_key="id",
        write_disposition=SPARSE_UPSERT,
        **destination_config.run_kwargs,
    )

    pipeline.run(
        [
            # patch A: name changed, phone explicitly cleared, address omitted
            {"id": 1, "name": "Alice2", "phone": None},
            # patch B (newer, same key): only address, must not undo A
            {"id": 1, "address": {"city": "Berlin"}},
            # brand new key
            {"id": 3, "name": "Carol", "phone": "789"},
        ],
        table_name="customers",
        primary_key="id",
        write_disposition=SPARSE_UPSERT,
        **destination_config.run_kwargs,
    )

    rows = {r["id"]: r for r in _table_dicts(pipeline, "customers")}
    # A and B composed: name/phone from A, address from B
    assert rows[1]["name"] == "Alice2"
    assert rows[1]["phone"] is None
    assert rows[1]["address__city"] == "Berlin"
    # untouched key
    assert rows[2]["name"] == "Bob" and rows[2]["phone"] == "456"
    # inserted key, omitted field lands as null
    assert rows[3]["name"] == "Carol" and rows[3]["address__city"] is None

    # markers never leak into the destination table
    cols = _table_columns(pipeline, "customers")
    assert not any(is_sparse_system_column(c) for c in cols)


@pytest.mark.parametrize(
    "destination_config",
    destinations_configs(default_sql_configs=True, subset=["duckdb"]),
    ids=lambda x: x.name,
)
def test_sparse_nested_list_replacement_and_preservation(
    destination_config: DestinationTestConfiguration,
) -> None:
    """A provided nested list replaces its items; an omitted list keeps them; empty list clears."""
    pipeline = destination_config.setup_pipeline("sparse_nested_list", dev_mode=True)
    pipeline.run(
        [
            {
                "id": 1,
                "name": "Alice",
                "contacts": [
                    {"kind": "email", "value": "a@old.com"},
                    {"kind": "phone", "value": "123"},
                ],
            },
            {"id": 2, "name": "Bob", "contacts": [{"kind": "email", "value": "b@old.com"}]},
        ],
        table_name="customers",
        primary_key="id",
        write_disposition=SPARSE_UPSERT,
        **destination_config.run_kwargs,
    )

    pipeline.run(
        [
            # explicit replacement list for id 1
            {"id": 1, "contacts": [{"kind": "email", "value": "a@new.com"}]},
            # newer patch of id 1 omits contacts: replacement must survive
            {"id": 1, "name": "Alice2"},
            # id 2 omits contacts entirely
            {"id": 2, "name": "Bob2"},
        ],
        table_name="customers",
        primary_key="id",
        write_disposition=SPARSE_UPSERT,
        **destination_config.run_kwargs,
    )

    nested = _table_dicts(pipeline, "customers__contacts")
    root_ids = {r["id"]: r["_dlt_id"] for r in _table_dicts(pipeline, "customers")}
    values_1 = {r["value"] for r in nested if r["_dlt_root_id"] == root_ids[1]}
    values_2 = {r["value"] for r in nested if r["_dlt_root_id"] == root_ids[2]}
    assert values_1 == {"a@new.com"}
    assert values_2 == {"b@old.com"}

    # explicit empty list clears id 2's children only
    pipeline.run(
        [{"id": 2, "contacts": []}],
        table_name="customers",
        primary_key="id",
        write_disposition=SPARSE_UPSERT,
        **destination_config.run_kwargs,
    )
    nested = _table_dicts(pipeline, "customers__contacts")
    root_ids = {r["id"]: r["_dlt_id"] for r in _table_dicts(pipeline, "customers")}
    assert not [r for r in nested if r["_dlt_root_id"] == root_ids[2]]
    # id 1 children remain untouched by id 2's patch
    assert {r["value"] for r in nested if r["_dlt_root_id"] == root_ids[1]} == {"a@new.com"}

    assert not any(is_sparse_system_column(c) for c in _table_columns(pipeline, "customers__contacts"))


@pytest.mark.parametrize(
    "destination_config",
    destinations_configs(default_sql_configs=True, subset=["duckdb"]),
    ids=lambda x: x.name,
)
def test_sparse_new_nested_field_does_not_wipe_old_rows(
    destination_config: DestinationTestConfiguration,
) -> None:
    """A column appearing first inside a nested row must not clear the other nested records."""
    pipeline = destination_config.setup_pipeline("sparse_new_nested_field", dev_mode=True)
    pipeline.run(
        [{"id": 1, "contacts": [{"kind": "email", "value": "a@x.com"}]}],
        table_name="customers",
        primary_key="id",
        write_disposition=SPARSE_UPSERT,
        **destination_config.run_kwargs,
    )
    pipeline.run(
        [{"id": 1, "contacts": [{"kind": "email", "value": "a@y.com", "label": "work"}]}],
        table_name="customers",
        primary_key="id",
        write_disposition=SPARSE_UPSERT,
        **destination_config.run_kwargs,
    )
    nested = _table_dicts(pipeline, "customers__contacts")
    assert len(nested) == 1
    assert nested[0]["value"] == "a@y.com"
    assert nested[0]["label"] == "work"


@pytest.mark.parametrize(
    "destination_config",
    destinations_configs(default_sql_configs=True, subset=["duckdb"]),
    ids=lambda x: x.name,
)
def test_sparse_replay_and_stale_patch_blocked(
    destination_config: DestinationTestConfiguration,
) -> None:
    """With a business version column, old/replayed patches cannot undo newer modifications."""
    pipeline = destination_config.setup_pipeline("sparse_replay", dev_mode=True)
    version_col = {"version": {"x-row-version": True, "data_type": "bigint", "nullable": False}}

    pipeline.run(
        [{"id": 1, "version": 5, "name": "Alice", "phone": "123",
          "contacts": [{"kind": "email", "value": "v5"}]}],
        table_name="customers",
        primary_key="id",
        write_disposition=SPARSE_UPSERT,
        columns=version_col,
        **destination_config.run_kwargs,
    )
    pipeline.run(
        [{"id": 1, "version": 10, "name": "Alice10", "phone": None,
          "contacts": [{"kind": "email", "value": "v10"}]}],
        table_name="customers",
        primary_key="id",
        write_disposition=SPARSE_UPSERT,
        **destination_config.run_kwargs,
    )

    # replay an older patch: root and nested rows must stay at v10
    pipeline.run(
        [{"id": 1, "version": 7, "name": "OLD", "phone": "stale",
          "contacts": [{"kind": "email", "value": "stale"}]}],
        table_name="customers",
        primary_key="id",
        write_disposition=SPARSE_UPSERT,
        **destination_config.run_kwargs,
    )
    root = _table_dicts(pipeline, "customers")[0]
    assert root["version"] == 10 and root["name"] == "Alice10" and root["phone"] is None
    nested = _table_dicts(pipeline, "customers__contacts")
    assert [r["value"] for r in nested] == ["v10"]

    # a newer partial patch applies and keeps the already-cleared phone / omitted list
    pipeline.run(
        [{"id": 1, "version": 12, "name": "Alice12"}],
        table_name="customers",
        primary_key="id",
        write_disposition=SPARSE_UPSERT,
        **destination_config.run_kwargs,
    )
    root = _table_dicts(pipeline, "customers")[0]
    assert root["version"] == 12 and root["name"] == "Alice12" and root["phone"] is None
    assert [r["value"] for r in _table_dicts(pipeline, "customers__contacts")] == ["v10"]

    # same batch: stale v8 and fresh v15 of one key -> only v15 effects survive
    pipeline.run(
        [
            {"id": 1, "version": 8, "name": "BATCH_OLD"},
            {"id": 1, "version": 15, "name": "BATCH_NEW", "phone": "999"},
        ],
        table_name="customers",
        primary_key="id",
        write_disposition=SPARSE_UPSERT,
        **destination_config.run_kwargs,
    )
    root = _table_dicts(pipeline, "customers")[0]
    assert root["version"] == 15 and root["name"] == "BATCH_NEW" and root["phone"] == "999"


@pytest.mark.parametrize("whole_row_strategy", ["delete-insert", "scd2"])
@pytest.mark.parametrize(
    "destination_config",
    destinations_configs(default_sql_configs=True, subset=["duckdb"]),
    ids=lambda x: x.name,
)
def test_sparse_update_rejects_whole_row_strategy(
    destination_config: DestinationTestConfiguration, whole_row_strategy: str
) -> None:
    """A whole-row merge strategy with sparse_update must fail loudly instead of nulling fields."""
    pipeline = destination_config.setup_pipeline("sparse_rejected", dev_mode=True)
    with pytest.raises(PipelineStepFailed) as exc:
        pipeline.run(
            [{"id": 1, "name": "Alice"}],
            table_name="customers",
            primary_key="id",
            write_disposition={
                "disposition": "merge",
                "strategy": whole_row_strategy,
                "sparse_update": True,
            },
            **destination_config.run_kwargs,
        )
    # rejected at extract time for an explicit whole-row strategy...
    assert isinstance(exc.value.__context__, (SchemaCorruptedException, ValueError))
    assert "sparse_update" in str(exc.value)


@pytest.mark.parametrize(
    "destination_config",
    destinations_configs(default_sql_configs=True, subset=["duckdb"]),
    ids=lambda x: x.name,
)
def test_sparse_cdc_deletes_absent_while_keeping_omitted_fields(
    destination_config: DestinationTestConfiguration,
) -> None:
    """A sparse `cdc` snapshot still deletes keys absent from it, while present keys keep the
    destination fields the sparse snapshot omits."""
    sparse_cdc = {"disposition": "merge", "strategy": "cdc", "sparse_update": True}
    pipeline = destination_config.setup_pipeline("sparse_cdc", dev_mode=True)
    pipeline.run(
        [
            {"id": 1, "name": "Alice", "phone": "111"},
            {"id": 2, "name": "Bob", "phone": "222"},
        ],
        table_name="customers",
        primary_key="id",
        write_disposition=sparse_cdc,
        **destination_config.run_kwargs,
    )

    # sparse cdc snapshot: id 2 omitted from the snapshot entirely -> must be deleted; id 1 only
    # carries name, so its phone stays
    pipeline.run(
        [{"id": 1, "name": "Alice2"}],
        table_name="customers",
        primary_key="id",
        write_disposition=sparse_cdc,
        **destination_config.run_kwargs,
    )
    rows = {r["id"]: r for r in _table_dicts(pipeline, "customers")}
    assert set(rows) == {1}
    assert rows[1]["name"] == "Alice2"
    assert rows[1]["phone"] == "111"
