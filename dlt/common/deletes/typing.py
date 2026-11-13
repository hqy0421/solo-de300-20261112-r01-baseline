"""Type aliases for the delete policy.

The authoritative definitions live in `dlt.common.schema.typing`; they are re-exported here so
callers can import the delete types without pulling in the whole schema module.
"""

from dlt.common.schema.typing import (
    TDeleteAction,
    TDeletePolicyDict,
    TDeleteVerdict,
    TDeletionScope,
    TRecordLifecycleState,
)

__all__ = [
    "TDeleteAction",
    "TDeletePolicyDict",
    "TDeleteVerdict",
    "TDeletionScope",
    "TRecordLifecycleState",
]
