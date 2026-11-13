"""Sparse update (partial patch) support for the merge write disposition.

A regular merge treats every staged row as a full record: the destination row is deleted and
recreated (or fully overwritten) from it. A *sparse update* treats the staged rows as patches
that only describe the fields they contain:

* a field missing from the patch keeps its destination value,
* a field explicitly set to `None` clears the destination value,
* a nested list present in the patch replaces the corresponding list items of the destination
  record, while a missing list leaves them untouched,
* several patches of the same key inside one batch take effect in their business version order,
  and replaying an old package can never undo a newer modification.

The normalizer records the field presence with boolean *mask columns*
(`C_SPARSE_MASK_PREFIX` + column name), a per-patch sequence (`C_SPARSE_SEQ`) and a marker on
nested rows (`C_SPARSE_LIST`). The loader merge jobs read those markers to compose the new row
from the destination row and the patch. The marker columns exist only in the staging tables and
are never created in the destination tables or the user-facing payload.
"""
from typing import Any, Dict, List, Optional

from dlt.common.schema.typing import (
    C_SPARSE_LIST,
    C_SPARSE_LIST_PREFIX,
    C_SPARSE_MASK_PREFIX,
    C_SPARSE_SEQ,
    TColumnSchema,
    TTableSchema,
)

SPARSE_UPDATE_HINT = "x-merge-sparse-update"

# private row-dict keys used to carry the sparse metadata through the relational normalizer.
# They are removed before the rows reach the data writers. The names start with the reserved
# dlt prefix and cannot collide with normalized user columns.
SPARSE_PRESENCE_KEY = "__dlt_sparse_presence"
"""Per-row set of flattened column names explicitly present in the patch (including nulls)."""
SPARSE_SEQ_KEY = "__dlt_sparse_seq"
"""Per-root-row monotonic sequence number of the patch inside the load package."""
SPARSE_LIST_PATHS_KEY = "__dlt_sparse_list_paths"
"""Per-root-row set of nested-table ident paths whose list the patch explicitly provides."""


def new_sparse_context(seq: int) -> Dict[str, Any]:
    """Creates the sparse metadata attached to the root row of one patch item."""
    return {
        SPARSE_SEQ_KEY: seq,
        SPARSE_PRESENCE_KEY: set(),
        SPARSE_LIST_PATHS_KEY: set(),
    }


def get_sparse_context(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # root rows carry the presence set; nested rows only carry the list-path marker, so either
    # private key identifies a row belonging to a sparse patch
    if SPARSE_PRESENCE_KEY in row or SPARSE_LIST_PATHS_KEY in row:
        return {
            SPARSE_SEQ_KEY: row.get(SPARSE_SEQ_KEY),
            SPARSE_PRESENCE_KEY: row.get(SPARSE_PRESENCE_KEY, set()),
            SPARSE_LIST_PATHS_KEY: row.get(SPARSE_LIST_PATHS_KEY, set()),
        }
    return None


def strip_sparse_context(row: Dict[str, Any]) -> None:
    """Removes the private sparse metadata keys from a row before it is written."""
    row.pop(SPARSE_PRESENCE_KEY, None)
    row.pop(SPARSE_SEQ_KEY, None)
    row.pop(SPARSE_LIST_PATHS_KEY, None)


def mask_column_name(column_name: str) -> str:
    """Name of the boolean presence-mask column backing `column_name`."""
    return f"{C_SPARSE_MASK_PREFIX}{column_name}"


def is_mask_column(column_name: str) -> bool:
    return column_name.startswith(C_SPARSE_MASK_PREFIX)


def unmask_column_name(mask_name: str) -> str:
    return mask_name[len(C_SPARSE_MASK_PREFIX):]


def make_mask_column(column_name: str, nullable: bool = True) -> TColumnSchema:
    """Creates a boolean presence-mask column for `column_name`."""
    return TColumnSchema(
        name=mask_column_name(column_name),
        data_type="bool",
        nullable=nullable,
        **{"x-sparse-mask": True},  # type: ignore[arg-type]
    )


def make_seq_column() -> TColumnSchema:
    """Creates the monotonic patch-sequence column (bigint)."""
    return TColumnSchema(name=C_SPARSE_SEQ, data_type="bigint", nullable=False)


def make_list_marker_column() -> TColumnSchema:
    """Creates the boolean marker that flags explicitly provided nested rows."""
    return TColumnSchema(
        name=C_SPARSE_LIST,
        data_type="bool",
        nullable=True,
        **{"x-sparse-list": True},  # type: ignore[arg-type]
    )


def list_path_marker_name(nested_table_name: str) -> str:
    """Name of the root-table boolean marker that records a patch explicitly provides the nested
    list materialized in `nested_table_name`. It is needed to clear a list provided as empty,
    which emits no nested rows and therefore cannot carry the per-row list marker."""
    return f"{C_SPARSE_LIST_PREFIX}{nested_table_name}"


def make_list_path_marker_column(nested_table_name: str) -> TColumnSchema:
    """Creates the root-table marker column for an explicitly provided (possibly empty) list."""
    return TColumnSchema(
        name=list_path_marker_name(nested_table_name),
        data_type="bool",
        nullable=True,
        **{"x-sparse-list": True},  # type: ignore[arg-type]
    )


def sparse_data_columns(table: TTableSchema) -> List[str]:
    """User/dlt data columns of a sparse table, i.e. all columns except the sparse marker columns."""
    return [
        name
        for name, column in table["columns"].items()
        if not column.get("x-sparse-mask")
        and not column.get("x-sparse-list")
        and name != C_SPARSE_SEQ
        and not name.startswith(C_SPARSE_LIST_PREFIX)
    ]


def is_sparse_system_column(column_name: str) -> bool:
    """`True` for the marker columns that only the sparse merge reads (never user data)."""
    return (
        is_mask_column(column_name)
        or column_name == C_SPARSE_SEQ
        or column_name == C_SPARSE_LIST
        or column_name.startswith(C_SPARSE_LIST_PREFIX)
    )
