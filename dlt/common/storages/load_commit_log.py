"""Optional per-table commit log for load packages.

The commit log records, for each table of a load package, the transitions
`prepared` -> `executed` -> `confirmed` (or `failed`) together with a fingerprint
of the input job files:

* `prepared` - the loader started jobs for the table, input files were fingerprinted
* `executed` - all jobs of the table reached a terminal state at the destination
* `confirmed` - all jobs of the table are durably in the `completed_jobs` folder
* `failed` - a job of the table failed terminally

When a load is interrupted (eg. the process exits while writing to the destination,
leaving some tables written and others not) the log is verified on the next run:

* `confirmed` tables are checked for completeness and are not re-executed
* `prepared`/`executed` (unconfirmed) tables have their uncommitted jobs re-queued,
  so only uncertain or failed tables are redone
* input fingerprints of pending jobs are re-checked so a package restored from a
  different input is detected instead of silently loaded

Package level events bind the final state of the whole package to the pipeline state
(source position) that is committed to the destination under the same load id, so the
committed load id always corresponds to a fully recorded set of table outcomes.

Logs of completed packages are pruned after a retention period; a traceable summary
is archived in `commit_log_archive` at the load storage root before the detailed log
is deleted.
"""

import hashlib
import os
import threading
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, TYPE_CHECKING, cast

from pendulum.datetime import DateTime

from dlt.common import logger, pendulum
from dlt.common.json import json
from dlt.common.storages.file_storage import FileStorage
from dlt.common.storages.load_package import PackageStorage, ParsedLoadJobFileName
from dlt.common.typing import DictStrAny, TypedDict

if TYPE_CHECKING:
    from dlt.common.storages.load_storage import LoadStorage


COMMIT_LOG_VERSION = 1
COMMIT_LOG_ARCHIVE_FOLDER = "commit_log_archive"
"""Folder at the load storage root where summaries of pruned commit logs are kept"""

# per-table events
COMMIT_EVENT_PREPARED = "prepared"
COMMIT_EVENT_EXECUTED = "executed"
COMMIT_EVENT_CONFIRMED = "confirmed"
COMMIT_EVENT_FAILED = "failed"
# package level events (recorded under the reserved `_package` table)
COMMIT_EVENT_PACKAGE_READY = "package_commit_ready"
COMMIT_EVENT_PACKAGE_COMMITTED = "package_committed"


class TTableCommitEvent(TypedDict, total=False):
    v: int
    """Event format version"""
    ts: str
    """ISO timestamp of the event"""
    load_id: str
    table: str
    """Table name or `_package` for package level events"""
    event: str
    """One of prepared/executed/confirmed/failed or a package level event"""
    jobs: List[str]
    """Job ids (table.file_id.format, no retry count) the event covers"""
    job_fingerprints: Dict[str, str]
    """`prepared` events: content fingerprint per job id"""
    fingerprint: str
    """`prepared` events: digest of `job_fingerprints` - the table input fingerprint"""
    message: str
    """`failed` events: terminal failure message"""
    state_hash: str
    """`package_commit_ready`: hash of the pipeline state (source position)"""
    schema_name: str
    schema_hash: str
    tables: Dict[str, str]
    """`package_commit_ready`: final per-table status committed with the load id"""


def fingerprint_file(full_path: str) -> str:
    """Content fingerprint of a single job file."""
    h = hashlib.sha256()
    with open(full_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def digest_job_fingerprints(job_fingerprints: Dict[str, str]) -> str:
    """Order-independent digest over per-job fingerprints - the table input fingerprint."""
    h = hashlib.sha256()
    for job_id in sorted(job_fingerprints):
        h.update(job_id.encode("utf-8"))
        h.update(b"\0")
        h.update(job_fingerprints[job_id].encode("utf-8"))
        h.update(b"\n")
    return "sha256:" + h.hexdigest()


def fingerprint_job_files(full_paths: Sequence[str]) -> Dict[str, str]:
    """Fingerprints job files at `full_paths`, keyed by job id.

    The job id does not contain the retry count, so the fingerprint is stable when a
    job file is renamed on retry (the content does not change).
    """
    fingerprints: Dict[str, str] = {}
    for path in sorted(full_paths):
        file_name = FileStorage.get_file_name_from_file_path(path)
        job_id = ParsedLoadJobFileName.parse(file_name).job_id()
        fingerprints[job_id] = fingerprint_file(path)
    return fingerprints


class TableCommitLog:
    """Append-only commit log of a single load package.

    Events are small JSON files in the `.commit_log` folder of the package, written
    atomically and ordered by a sequence number. The folder moves with the package
    when it is archived to the loaded packages storage. Writes are thread safe and
    idempotent: appending an event identical to the latest one of a table is a no-op.
    """

    COMMIT_LOG_FOLDER: ClassVar[str] = ".commit_log"
    PACKAGE_TABLE: ClassVar[str] = "_package"
    """Reserved table name for package level events"""

    def __init__(self, packages: PackageStorage, load_id: str) -> None:
        self.packages = packages
        self.load_id = load_id
        self._lock = threading.Lock()
        self._seq: Optional[int] = None

    #
    # write events
    #

    def record_prepared(self, table: str, job_fingerprints: Dict[str, str]) -> None:
        """Records that jobs of `table` were started. `job_fingerprints` maps job id to
        the content fingerprint of its input file."""
        self._append(
            table,
            COMMIT_EVENT_PREPARED,
            jobs=sorted(job_fingerprints),
            job_fingerprints=job_fingerprints,
            fingerprint=digest_job_fingerprints(job_fingerprints),
        )

    def record_executed(self, table: str, jobs: Sequence[str]) -> None:
        """Records that all jobs of `table` reached a terminal state at the destination."""
        self._append(table, COMMIT_EVENT_EXECUTED, jobs=sorted(jobs))

    def record_confirmed(self, table: str, jobs: Sequence[str]) -> None:
        """Records that all jobs of `table` are durably completed."""
        self._append(table, COMMIT_EVENT_CONFIRMED, jobs=sorted(jobs))

    def record_failed(self, table: str, jobs: Sequence[str], message: Optional[str]) -> None:
        """Records that a job of `table` failed terminally."""
        self._append(table, COMMIT_EVENT_FAILED, jobs=sorted(jobs), message=message)

    def record_package_commit_ready(
        self,
        state_hash: Optional[str],
        schema_name: str,
        schema_hash: str,
        tables: Dict[str, str],
    ) -> None:
        """Records that the package is about to be committed at the destination. Binds the
        final per-table statuses to the pipeline state (source position) and schema that
        will be committed under the same load id."""
        self._append(
            self.PACKAGE_TABLE,
            COMMIT_EVENT_PACKAGE_READY,
            state_hash=state_hash,
            schema_name=schema_name,
            schema_hash=schema_hash,
            tables=tables,
        )

    def record_package_committed(self) -> None:
        """Records that the destination committed the load id."""
        self._append(self.PACKAGE_TABLE, COMMIT_EVENT_PACKAGE_COMMITTED)

    #
    # read events
    #

    def has_events(self) -> bool:
        return bool(self._list_event_files())

    def events(self) -> List[TTableCommitEvent]:
        """All events in the order they were recorded. Unreadable events are skipped."""
        events: List[TTableCommitEvent] = []
        folder = self._folder_path()
        for file_name in self._list_event_files():
            rel_path = os.path.join(folder, file_name)
            try:
                events.append(json.loads(self.packages.storage.load(rel_path)))
            except Exception as ex:
                logger.warning(f"commit log: skipping unreadable event {rel_path}: {ex}")
        return events

    def table_statuses(self) -> Dict[str, TTableCommitEvent]:
        """Latest event per table, excluding package level events."""
        statuses: Dict[str, TTableCommitEvent] = {}
        for event in self.events():
            if event["table"] == self.PACKAGE_TABLE:
                continue
            statuses[event["table"]] = event
        return statuses

    def package_status(self) -> Optional[str]:
        """Latest package level event name or None."""
        status: Optional[str] = None
        for event in self.events():
            if event["table"] == self.PACKAGE_TABLE:
                status = event["event"]
        return status

    def prepared_batches(self, table: str) -> List[TTableCommitEvent]:
        """All `prepared` events of `table`, in recording order."""
        return [
            event
            for event in self.events()
            if event["table"] == table and event["event"] == COMMIT_EVENT_PREPARED
        ]

    def prepared_job_ids(self) -> Set[str]:
        """Job ids covered by any `prepared` event."""
        ids: Set[str] = set()
        for event in self.events():
            if event["event"] == COMMIT_EVENT_PREPARED:
                ids.update(event.get("job_fingerprints", {}))
        return ids

    #
    # summary and deletion
    #

    def build_summary(self) -> DictStrAny:
        """Builds a traceable summary of the whole log: final per-table statuses with
        input fingerprints plus the package level commit record. Archived before the
        detailed log is pruned."""
        events = self.events()
        ready: Optional[TTableCommitEvent] = None
        for event in events:
            if event["table"] != self.PACKAGE_TABLE:
                continue
            if event["event"] == COMMIT_EVENT_PACKAGE_READY:
                ready = event
        tables: Dict[str, DictStrAny] = {}
        for table, event in self.table_statuses().items():
            job_fingerprints: Dict[str, str] = {}
            for batch in self.prepared_batches(table):
                job_fingerprints.update(batch.get("job_fingerprints", {}))
            table_summary: DictStrAny = {
                "status": event["event"],
                "jobs": event.get("jobs", []),
                "fingerprint": (
                    digest_job_fingerprints(job_fingerprints) if job_fingerprints else None
                ),
            }
            if event["event"] == COMMIT_EVENT_FAILED:
                table_summary["message"] = event.get("message")
            tables[table] = table_summary
        return {
            "v": COMMIT_LOG_VERSION,
            "load_id": self.load_id,
            "archived_at": pendulum.now().to_iso8601_string(),
            "events_count": len(events),
            "first_event_ts": events[0]["ts"] if events else None,
            "last_event_ts": events[-1]["ts"] if events else None,
            "package_status": self.package_status(),
            "schema_name": ready.get("schema_name") if ready else None,
            "schema_hash": ready.get("schema_hash") if ready else None,
            "state_hash": ready.get("state_hash") if ready else None,
            "tables": tables,
        }

    def delete(self) -> None:
        """Deletes the commit log folder. Call only after a summary was archived."""
        folder = self._folder_path()
        if self.packages.storage.has_folder(folder):
            self.packages.storage.delete_folder(folder, recursively=True)

    #
    # internals
    #

    def _folder_path(self) -> str:
        return os.path.join(
            self.packages.get_package_path(self.load_id), self.COMMIT_LOG_FOLDER
        )

    def _list_event_files(self) -> List[str]:
        folder = self._folder_path()
        if not self.packages.storage.has_folder(folder):
            return []
        return sorted(
            file_name
            for file_name in self.packages.storage.list_folder_files(folder, to_root=False)
            if file_name.endswith(".json")
        )

    def _next_seq(self) -> int:
        with self._lock:
            if self._seq is None:
                self._seq = len(self._list_event_files())
            self._seq += 1
            return self._seq

    def _append(self, table: str, event: str, **fields: Any) -> None:
        # dedup: appending an event identical to the latest one of the table is a no-op
        # (eg. `confirmed` re-recorded for a restored table after a restart)
        latest: Optional[TTableCommitEvent] = None
        for existing in self.events():
            if existing["table"] == table:
                latest = existing
        if (
            latest is not None
            and latest["event"] == event
            and all(latest.get(key) == value for key, value in fields.items())
        ):
            return
        seq = self._next_seq()
        commit_event = cast(
            TTableCommitEvent,
            {
                "v": COMMIT_LOG_VERSION,
                "ts": pendulum.now().to_iso8601_string(),
                "load_id": self.load_id,
                "table": table,
                "event": event,
                **fields,
            },
        )
        file_name = f"{seq:08d}__{table}__{event}.json"
        folder = self._folder_path()
        # created lazily on the first event so packages without the feature stay untouched
        self.packages.storage.create_folder(folder, exists_ok=True)
        self.packages.storage.save(os.path.join(folder, file_name), json.dumps(commit_event))


def get_commit_log(packages: PackageStorage, load_id: str) -> TableCommitLog:
    return TableCommitLog(packages, load_id)


def prune_expired_commit_logs(
    load_storage: "LoadStorage", retention_days: float, now: Optional[DateTime] = None
) -> List[str]:
    """Archives a traceable summary and deletes the commit log of completed packages
    whose completion is older than `retention_days`. Returns the archived load ids."""
    now = now or pendulum.now()
    archived: List[str] = []
    packages = load_storage.loaded_packages
    for load_id in load_storage.list_loaded_packages():
        commit_log = TableCommitLog(packages, load_id)
        if not commit_log.has_events():
            continue
        marker_path = os.path.join(
            packages.get_package_path(load_id), PackageStorage.PACKAGE_COMPLETED_FILE_NAME
        )
        if not packages.storage.has_file(marker_path):
            continue
        completed_at = pendulum.from_timestamp(
            os.path.getmtime(packages.storage.make_full_path(marker_path))
        )
        if (now - completed_at).total_seconds() < retention_days * 86400:
            continue
        summary = commit_log.build_summary()
        summary["package_completed_at"] = completed_at.to_iso8601_string()
        # created lazily on the first archived summary
        load_storage.storage.create_folder(COMMIT_LOG_ARCHIVE_FOLDER, exists_ok=True)
        load_storage.storage.save(
            os.path.join(COMMIT_LOG_ARCHIVE_FOLDER, f"{load_id}.json"),
            json.dumps(summary, pretty=True),
        )
        commit_log.delete()
        archived.append(load_id)
    return archived
