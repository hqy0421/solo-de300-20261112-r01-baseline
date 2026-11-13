"""Validation of delete policy configuration and schema integration.

The configuration arrives as the `delete_policy` key of a merge write disposition dict and is
stored on the table as the `x-delete-policy` hint. This module validates the config against the
table schema (referenced columns must exist) and the chosen merge strategy (scd2 keeps retired
rows, so it cannot hard-delete them).
"""

from typing import Any, cast

from dlt.common.deletes.exceptions import DeletePolicyConfigurationException
from dlt.common.deletes.policy import DeletePolicy
from dlt.common.deletes.typing import TDeletePolicyDict
from dlt.common.schema.typing import MERGE_STRATEGY_OPTIONS, TTableSchema


def validate_delete_policy_config(
    # the config comes out of dynamic resource hints, so only its runtime shape is guaranteed
    config: dict[str, Any],
    *,
    table_name: Any = None,
    columns: Any = None,
    strategy: str | None = None,
    check_columns: bool = False,
) -> DeletePolicy:
    """Validates a raw delete policy config and returns the parsed `DeletePolicy`.

    Validates:
    - enum values and soft-delete column presence (done in `DeletePolicy.from_config`),
    - referenced columns exist in `columns` when `check_columns` is `True` and a schema is
      provided. Resource hint validation runs before data-driven column inference, so column
      existence is only checked later against the prepared destination schema,
    - the merge strategy supports delete policies (scd2 does not).
    """
    policy = DeletePolicy.from_config(
        cast(TDeletePolicyDict, config), table_name=table_name
    )
    if not policy.enabled:
        return policy

    if strategy is not None and strategy not in MERGE_STRATEGY_OPTIONS["delete_policy"]:
        supported = ", ".join(f"`{s}`" for s in MERGE_STRATEGY_OPTIONS["delete_policy"])
        raise DeletePolicyConfigurationException(
            table_name,
            f"the `{strategy}` merge strategy does not support a delete policy"
            f" (supported strategies: {supported}). scd2 preserves retired versions by design.",
            field="strategy",
            value=strategy,
        )

    if check_columns and columns is not None:
        referenced = [
            c
            for c in (
                policy.delete_marker_column,
                policy.soft_delete_column_name,
                policy.row_version_column_name,
            )
            if c is not None
        ]
        missing = [c for c in referenced if c not in columns]
        if missing:
            raise DeletePolicyConfigurationException(
                table_name,
                "the delete policy references columns missing from the table schema: "
                + ", ".join(f"`{c}`" for c in missing),
                field="delete_policy",
            )

    return policy


def get_delete_policy(table: TTableSchema) -> TDeletePolicyDict | None:
    """Reads the stored `x-delete-policy` hint from a table schema."""
    hint = table.get("x-delete-policy")
    return cast(TDeletePolicyDict | None, hint)


def resolve_delete_policy(table: TTableSchema) -> DeletePolicy:
    """Builds the validated `DeletePolicy` for a table. Disabled when no hint is stored."""
    return DeletePolicy.from_config(get_delete_policy(table), table_name=table.get("name"))
