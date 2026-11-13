"""Optional delete policy for merge resources.

Source systems represent customer invalidation in two ways:

1. an explicit deletion marker (tombstone flag) on a row that is still delivered, and
2. the physical absence of a row from a data package.

The destination must distinguish between:

- retained history (row stays active),
- soft deletes (the row is kept and flagged, so history is preserved), and
- hard deletes (the row is physically removed).

This package is destination-agnostic. It only *classifies* rows; the load layer turns the
verdicts into SQL or equivalent operations. Design rules:

- the policy is opt-in and disabled by default,
- absent rows are never treated as deleted without an explicit ``full_sync`` declaration,
- repeated deletes are idempotent,
- a late (stale) update can never revive an already soft- or hard-deleted record.
"""

from dlt.common.deletes.apply import ProcessedPackage, process_package
from dlt.common.deletes.exceptions import (
    DeleteMarkerColumnMissing,
    DeletePolicyConfigurationException,
    DeletePolicyException,
    FullSyncDeclarationMissing,
)
from dlt.common.deletes.policy import (
    DEFAULT_ABSENT_ACTION,
    DEFAULT_MARKER_ACTION,
    DEFAULT_SYNC_SCOPE,
    DeletePolicy,
    DeletePolicyVerdict,
    is_delete_marker_truthy,
)
from dlt.common.deletes.state import (
    DeleteTracker,
    RecordLifecycle,
)
from dlt.common.deletes.typing import (
    TDeleteAction,
    TDeletePolicyDict,
    TDeleteVerdict,
    TDeletionScope,
    TRecordLifecycleState,
)

__all__ = [
    "DEFAULT_ABSENT_ACTION",
    "DEFAULT_MARKER_ACTION",
    "DEFAULT_SYNC_SCOPE",
    "DeleteMarkerColumnMissing",
    "DeletePolicy",
    "DeletePolicyConfigurationException",
    "DeletePolicyException",
    "DeletePolicyVerdict",
    "DeleteTracker",
    "FullSyncDeclarationMissing",
    "ProcessedPackage",
    "RecordLifecycle",
    "TDeleteAction",
    "TDeletePolicyDict",
    "TDeleteVerdict",
    "TDeletionScope",
    "TRecordLifecycleState",
    "is_delete_marker_truthy",
    "process_package",
]
