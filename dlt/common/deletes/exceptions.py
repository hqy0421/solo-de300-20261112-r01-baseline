from collections.abc import Sequence
from typing import Any, Optional

from dlt.common.exceptions import DltException


class DeletePolicyException(DltException):
    """Base class for delete policy errors."""


class DeletePolicyConfigurationException(DeletePolicyException):
    """Raised when a `delete_policy` configuration is invalid."""

    def __init__(
        self,
        table_name: str | None,
        reason: str,
        *,
        field: str | None = None,
        value: Any = None,
        known_values: Sequence[Any] | None = None,
    ) -> None:
        self.table_name = table_name
        self.field = field
        self.value = value
        self.known_values = list(known_values) if known_values else None
        location = f"table `{table_name}`" if table_name else "the delete policy"
        msg = f"Invalid delete policy on {location}: {reason}"
        if field is not None:
            msg = f"{msg} (field `{field}`" + (
                f", value `{value!r}`" if value is not None else ""
            ) + ")"
        if known_values:
            msg += f". Expected one of: {', '.join(repr(v) for v in known_values)}"
        super().__init__(msg)


class DeleteMarkerColumnMissing(DeletePolicyException):
    """Raised when a package row does not contain the configured marker column."""

    def __init__(self, table_name: str, column_name: str) -> None:
        self.table_name = table_name
        self.column_name = column_name
        super().__init__(
            f"Delete marker column `{column_name}` configured for table `{table_name}` is missing"
            " from a data row. Make sure the source emits the marker, or adjust the delete policy."
        )


class FullSyncDeclarationMissing(DeletePolicyException):
    """Raised when hard-deleting absent rows is attempted without a full sync declaration."""

    def __init__(self, table_name: str) -> None:
        self.table_name = table_name
        super().__init__(
            f"Cannot treat absent rows of table `{table_name}` as deleted: the load package does"
            " not declare a full sync (`sync_scope='present_only'`). Rows missing from an"
            " incremental package stay undetermined. Declare `sync_scope='full_sync'` on the"
            " package or configure `absent_action` only for full sync packages."
        )
