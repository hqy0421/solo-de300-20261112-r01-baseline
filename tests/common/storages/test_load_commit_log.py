import os
from typing import Dict, List, Tuple

import pytest

from dlt.common import pendulum
from dlt.common.data_writers import DataWriter
from dlt.common.schema import Schema
from dlt.common.storages import LoadStorage, PackageStorage, ParsedLoadJobFileName
from dlt.common.storages.load_commit_log import (
    COMMIT_LOG_ARCHIVE_FOLDER,
    TableCommitLog,
    digest_job_fingerprints,
    fingerprint_file,
    fingerprint_job_files,
    prune_expired_commit_logs,
)
from dlt.common.json import json
from dlt.common.utils import uniq_id

from tests.common.storages.utils import load_storage, write_temp_job_file
from tests.utils import autouse_test_storage


def _make_package(
    s: LoadStorage, tables: Dict[str, list]
) -> Tuple[str, Dict[str, List[str]]]:
    """Creates a normalized package with one jsonl job per table. Returns the load id
    and job file names by table."""
    load_id = uniq_id()
    s.new_packages.create_package(load_id)
    item_storage = s.create_item_storage(
        DataWriter.writer_spec_from_file_format("jsonl", "object")
    )
    job_files: Dict[str, List[str]] = {}
    for table_name, rows in tables.items():
        file_name = write_temp_job_file(
            item_storage, s.storage, load_id, table_name, None, uniq_id(), rows
        )
        job_files.setdefault(table_name, []).append(file_name)
    s.new_packages.save_schema(load_id, Schema("mock"))
    s.new_packages.save_schema_updates(load_id, {})
    s.commit_new_load_package(load_id)
    return load_id, job_files


def _job_id(file_name: str) -> str:
    return ParsedLoadJobFileName.parse(file_name).job_id()


def test_table_status_transitions(load_storage: LoadStorage) -> None:
    load_id, job_files = _make_package(
        load_storage, {"mock_table": [{"a": 1}], "other_table": [{"b": 2}]}
    )
    commit_log = TableCommitLog(load_storage.normalized_packages, load_id)
    assert not commit_log.has_events()

    # mock_table goes prepared -> executed -> confirmed
    mt_id = _job_id(job_files["mock_table"][0])
    commit_log.record_prepared("mock_table", {mt_id: "sha256:mt"})
    assert commit_log.table_statuses()["mock_table"]["event"] == "prepared"
    commit_log.record_executed("mock_table", [mt_id])
    assert commit_log.table_statuses()["mock_table"]["event"] == "executed"
    commit_log.record_confirmed("mock_table", [mt_id])
    assert commit_log.table_statuses()["mock_table"]["event"] == "confirmed"

    # other_table fails
    ot_id = _job_id(job_files["other_table"][0])
    commit_log.record_prepared("other_table", {ot_id: "sha256:ot"})
    commit_log.record_failed("other_table", [ot_id], "terminal boom")

    statuses = commit_log.table_statuses()
    assert set(statuses) == {"mock_table", "other_table"}
    assert statuses["other_table"]["event"] == "failed"
    assert statuses["other_table"]["message"] == "terminal boom"

    # events are ordered and prepared events keep the fingerprints
    events = commit_log.events()
    assert [e["event"] for e in events] == [
        "prepared",
        "executed",
        "confirmed",
        "prepared",
        "failed",
    ]
    assert commit_log.prepared_batches("mock_table")[0]["job_fingerprints"] == {
        mt_id: "sha256:mt"
    }
    assert commit_log.prepared_batches("mock_table")[0]["fingerprint"] == (
        digest_job_fingerprints({mt_id: "sha256:mt"})
    )
    assert commit_log.prepared_job_ids() == {mt_id, ot_id}


def test_identical_event_is_not_duplicated(load_storage: LoadStorage) -> None:
    load_id, job_files = _make_package(load_storage, {"mock_table": [{"a": 1}]})
    commit_log = TableCommitLog(load_storage.normalized_packages, load_id)
    mt_id = _job_id(job_files["mock_table"][0])
    commit_log.record_confirmed("mock_table", [mt_id])
    commit_log.record_confirmed("mock_table", [mt_id])
    assert len(commit_log.events()) == 1
    # a changed event is appended
    commit_log.record_confirmed("mock_table", [mt_id, "mock_table.other.0.jsonl"])
    assert len(commit_log.events()) == 2


def test_fingerprint_stable_across_retry_rename(load_storage: LoadStorage) -> None:
    load_id, job_files = _make_package(load_storage, {"mock_table": [{"a": 1}]})
    packages = load_storage.normalized_packages
    file_name = job_files["mock_table"][0]

    def current_rel_path() -> str:
        (state, parsed), *rest = [
            (state, parsed)
            for state, parsed in packages.list_all_jobs_with_states(load_id)
        ]
        assert not rest
        return packages.get_job_file_path(load_id, state, parsed.file_name())

    rel_path = current_rel_path()
    fingerprints = fingerprint_job_files([packages.storage.make_full_path(rel_path)])
    job_id = _job_id(file_name)
    assert list(fingerprints) == [job_id]

    # start and retry the job: the file is renamed with a bumped retry count
    packages.start_job(load_id, file_name)
    retried_name = ParsedLoadJobFileName.parse(file_name).with_retry().file_name()
    packages.retry_job(load_id, file_name, "transient boom", "transient")
    rel_path = current_rel_path()
    assert ParsedLoadJobFileName.parse(rel_path).file_name() == retried_name
    # same job id and same content -> same fingerprint
    assert fingerprint_job_files([packages.storage.make_full_path(rel_path)]) == fingerprints

    # changed content -> different fingerprint
    full_path = packages.storage.make_full_path(rel_path)
    with open(full_path, "ab") as f:
        f.write(b'{"a": 2}\n')
    assert fingerprint_job_files([full_path]) != fingerprints


def test_package_commit_events_and_summary(load_storage: LoadStorage) -> None:
    load_id, job_files = _make_package(load_storage, {"mock_table": [{"a": 1}]})
    commit_log = TableCommitLog(load_storage.normalized_packages, load_id)
    mt_id = _job_id(job_files["mock_table"][0])
    commit_log.record_prepared("mock_table", {mt_id: "sha256:mt"})
    commit_log.record_confirmed("mock_table", [mt_id])
    assert commit_log.package_status() is None

    commit_log.record_package_commit_ready(
        "sha256:state", "mock", "schema_hash", {"mock_table": "confirmed"}
    )
    assert commit_log.package_status() == "package_commit_ready"
    commit_log.record_package_committed()
    assert commit_log.package_status() == "package_committed"

    summary = commit_log.build_summary()
    assert summary["load_id"] == load_id
    assert summary["package_status"] == "package_committed"
    assert summary["state_hash"] == "sha256:state"
    assert summary["schema_name"] == "mock"
    assert summary["schema_hash"] == "schema_hash"
    assert summary["events_count"] == len(commit_log.events())
    assert summary["tables"]["mock_table"]["status"] == "confirmed"
    assert summary["tables"]["mock_table"]["fingerprint"] == digest_job_fingerprints(
        {mt_id: "sha256:mt"}
    )
    # the summary is json serializable for archival
    json.dumps(summary)


def test_prune_expired_commit_logs(load_storage: LoadStorage) -> None:
    # an old completed package with a commit log
    load_id, job_files = _make_package(load_storage, {"mock_table": [{"a": 1}]})
    commit_log = TableCommitLog(load_storage.normalized_packages, load_id)
    mt_id = _job_id(job_files["mock_table"][0])
    commit_log.record_prepared("mock_table", {mt_id: "sha256:mt"})
    commit_log.record_confirmed("mock_table", [mt_id])
    commit_log.record_package_commit_ready(
        "sha256:state", "mock", "schema_hash", {"mock_table": "confirmed"}
    )
    commit_log.record_package_committed()
    load_storage.complete_load_package(load_id, aborted=False)

    # a fresh completed package without events and a fresh one with events
    fresh_no_log_id, _ = _make_package(load_storage, {"mock_table": [{"a": 1}]})
    load_storage.complete_load_package(fresh_no_log_id, aborted=False)
    fresh_log_id, fresh_files = _make_package(load_storage, {"mock_table": [{"a": 1}]})
    fresh_log = TableCommitLog(load_storage.normalized_packages, fresh_log_id)
    fresh_log.record_prepared("mock_table", {_job_id(fresh_files["mock_table"][0]): "sha256:x"})
    load_storage.complete_load_package(fresh_log_id, aborted=False)

    # age the first package beyond retention
    marker = os.path.join(
        load_storage.loaded_packages.get_package_path(load_id),
        PackageStorage.PACKAGE_COMPLETED_FILE_NAME,
    )
    full_marker = load_storage.loaded_packages.storage.make_full_path(marker)
    old_ts = pendulum.now().subtract(days=30).timestamp()
    os.utime(full_marker, (old_ts, old_ts))

    archived = prune_expired_commit_logs(load_storage, retention_days=14.0)
    assert archived == [load_id]

    # the detailed log is gone, a traceable summary was archived instead
    loaded_log = TableCommitLog(load_storage.loaded_packages, load_id)
    assert not loaded_log.has_events()
    summary_path = os.path.join(COMMIT_LOG_ARCHIVE_FOLDER, f"{load_id}.json")
    summary = json.loads(load_storage.storage.load(summary_path))
    assert summary["load_id"] == load_id
    assert summary["package_status"] == "package_committed"
    assert summary["state_hash"] == "sha256:state"
    assert summary["tables"]["mock_table"]["status"] == "confirmed"
    assert summary["package_completed_at"] is not None

    # fresh packages are untouched
    assert TableCommitLog(load_storage.loaded_packages, fresh_log_id).has_events()
    assert not load_storage.storage.has_file(
        os.path.join(COMMIT_LOG_ARCHIVE_FOLDER, f"{fresh_log_id}.json")
    )
    assert not load_storage.storage.has_file(
        os.path.join(COMMIT_LOG_ARCHIVE_FOLDER, f"{fresh_no_log_id}.json")
    )

    # pruning is idempotent
    assert prune_expired_commit_logs(load_storage, retention_days=14.0) == []
