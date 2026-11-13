"""Tests integrating the delete policy with resource hints and table schemas."""

import pytest
from dlt.common.deletes.exceptions import DeletePolicyConfigurationException
from dlt.common.deletes.validation import (
    get_delete_policy,
    resolve_delete_policy,
    validate_delete_policy_config,
)
from dlt.extract import resource


def _rows():
    yield [{"id": 1, "is_deleted": False, "version": 1}]


class TestResourceHints:
    def test_delete_policy_stored_as_x_hint(self):
        r = resource(
            _rows,
            name="customers",
            write_disposition={
                "disposition": "merge",
                "strategy": "upsert",
                "delete_policy": {
                    "delete_marker_column": "is_deleted",
                    "soft_delete_column_name": "_dlt_soft_deleted",
                    "row_version_column_name": "version",
                },
            },
            primary_key="id",
        )
        table = r.compute_table_schema()
        assert table["x-delete-policy"]["delete_marker_column"] == "is_deleted"
        policy = resolve_delete_policy(table)
        assert policy.enabled
        assert policy.marker_action == "soft_delete"
        assert policy.sync_scope == "present_only"

    def test_policy_round_trips_through_hints(self):
        config = {
            "delete_marker_column": "is_deleted",
            "marker_action": "hard_delete",
            "sync_scope": "full_sync",
            "absent_action": "soft_delete",
            "soft_delete_column_name": "_d",
        }
        r = resource(
            _rows,
            name="customers",
            write_disposition={
                "disposition": "merge",
                "strategy": "delete-insert",
                "delete_policy": config,
            },
            primary_key="id",
        )
        table = r.compute_table_schema()
        assert get_delete_policy(table)["marker_action"] == "hard_delete"

    def test_invalid_policy_rejected_at_hint_validation(self):
        with pytest.raises(DeletePolicyConfigurationException):
            resource(
                _rows,
                name="customers",
                write_disposition={
                    "disposition": "merge",
                    "strategy": "upsert",
                    # soft delete reachable but no destination column
                    "delete_policy": {"delete_marker_column": "is_deleted"},
                },
                primary_key="id",
            )

    def test_scd2_rejects_delete_policy(self):
        # the generic merge-option validation rejects scd2 at hint time
        with pytest.raises(ValueError) as exc:
            resource(
                _rows,
                name="customers",
                write_disposition={
                    "disposition": "merge",
                    "strategy": "scd2",
                    "delete_policy": {
                        "delete_marker_column": "is_deleted",
                        "soft_delete_column_name": "_d",
                    },
                },
                primary_key="id",
            )
        assert "scd2" in str(exc.value)

    def test_policy_without_strategy_default_accepted(self):
        r = resource(
            _rows,
            name="customers",
            write_disposition={
                "disposition": "merge",
                "delete_policy": {
                    "delete_marker_column": "is_deleted",
                    "soft_delete_column_name": "_d",
                },
            },
            primary_key="id",
        )
        table = r.compute_table_schema()
        assert resolve_delete_policy(table).enabled


class TestSchemaValidation:
    def test_missing_column_referenced_by_policy_rejected(self):
        with pytest.raises(DeletePolicyConfigurationException):
            validate_delete_policy_config(
                {
                    "delete_marker_column": "ghost",
                    "soft_delete_column_name": "_d",
                },
                table_name="customers",
                columns={"id": {"name": "id"}},
                strategy="upsert",
                check_columns=True,
            )

    def test_missing_columns_not_checked_before_inference(self):
        # at resource hint time columns may not be inferred yet; that must not fail
        policy = validate_delete_policy_config(
            {
                "delete_marker_column": "ghost",
                "soft_delete_column_name": "_d",
            },
            table_name="customers",
            columns={},
            strategy="upsert",
        )
        assert policy.enabled

    def test_existing_columns_pass(self):
        policy = validate_delete_policy_config(
            {
                "delete_marker_column": "is_deleted",
                "soft_delete_column_name": "_d",
                "row_version_column_name": "version",
            },
            table_name="customers",
            columns={
                "id": {"name": "id"},
                "is_deleted": {"name": "is_deleted"},
                "_d": {"name": "_d"},
                "version": {"name": "version"},
            },
            strategy="cdc",
            check_columns=True,
        )
        assert policy.enabled

    def test_strategy_mismatch_rejected(self):
        with pytest.raises(DeletePolicyConfigurationException):
            validate_delete_policy_config(
                {"sync_scope": "full_sync"},
                table_name="customers",
                columns={},
                strategy="scd2",
            )

    def test_disabled_policy_passes_any_strategy(self):
        policy = validate_delete_policy_config(
            {}, table_name="t", columns={}, strategy="scd2"
        )
        assert not policy.enabled
