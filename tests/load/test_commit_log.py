import os
from typing import Dict

import pytest

import dlt
from dlt.common import pendulum
from dlt.common.configuration.container import Container
from dlt.common.json import json
from dlt.common.storages import FileStorage, PackageStorage, ParsedLoadJobFileName
from dlt.common.storages.exceptions import LoadPackageInconsistent
from dlt.common.storages.load_commit_log import (
    COMMIT_LOG_ARCHIVE_FOLDER,
    TableCommitLog,
)
from dlt.common.storages.load_package import LoadPackageStateInjectableContext
from dlt.common.utils import uniq_id
from dlt.destinations.impl.dummy.configuration import DummyClientConfiguration

from dlt.load import Load
from dlt.load.configuration import LoaderConfiguration

from tests.utils import clean_test_storage, get_test_storage_root, init_test_logging
from tests.utils import skip_if_not_active
from tests.load.utils import prepare_load_package
from tests.load.test_dummy_client import NORMALIZED_FILES, run_all, setup_loader

skip_if_not_active("dummy")


@pytest.fixture(autouse=True)
def storage() -> FileStorage:
    return clean_test_storage(init_normalize=True, init_loader=True)


@pytest.fixture(scope="module", autouse=True)
def logger_autouse() -> None:
    init_test_logging()


def _commit_log_loader(**config_kwargs) -> Load:
    config_kwargs.setdefault("commit_log_enabled", True)
    return setup_loader(
        client_config=DummyClientConfiguration(loader_file_format="jsonl", completed_prob=1.0),
        loader_config=LoaderConfiguration(**config_kwargs),
    )


def _files_by_table(load: Load, load_id: str) -> Dict[str, str]:
    """Maps table name to job file name for all new jobs of a package."""
    return {
        ParsedLoadJobFileName.parse(f).table_name: FileStorage.get_file_name_from_file_path(f)
        for f in load.load_storage.list_new_jobs(load_id)
    }


def test_commit_log_disabled_by_default() -> None:
    load = setup_loader()
    load_id, schema = prepare_load_package(load.load_storage, NORMALIZED_FILES)
    run_all(load)
    assert load.load_storage.get_load_package_info(load_id).state == "loaded"
    # no commit log was created and the loader never held one
    assert load._commit_log is None
    package_path = load.load_storage.loaded_packages.get_package_path(load_id)
    assert not load.load_storage.loaded_packages.storage.has_folder(
        os.path.join(package_path, TableCommitLog.COMMIT_LOG_FOLDER)
    )


def test_commit_log_full_run() -> None:
    load = _commit_log_loader()
    load_id, schema = prepare_load_package(load.load_storage, NORMALIZED_FILES)
    run_all(load)
    assert load.load_storage.get_load_package_info(load_id).state == "loaded"

    # the commit log moved with the package into the loaded storage
    commit_log = TableCommitLog(load.load_storage.loaded_packages, load_id)
    statuses = commit_log.table_statuses()
    assert set(statuses) == {"event_user", "event_loop_interrupted"}
    for event in statuses.values():
        assert event["event"] == "confirmed"
        assert len(event["jobs"]) == 1

    # each table went through prepared -> executed -> confirmed with input fingerprints
    for table_name in statuses:
        table_events = [
            event["event"] for event in commit_log.events() if event["table"] == table_name
        ]
        assert table_events == ["prepared", "executed", "confirmed"]
        batch = commit_log.prepared_batches(table_name)[0]
        assert batch["fingerprint"].startswith("sha256:")
        assert list(batch["job_fingerprints"]) == statuses[table_name]["jobs"]

    # the package was committed with the final per-table state bound to the schema
    assert commit_log.package_status() == "package_committed"
    summary = commit_log.build_summary()
    assert summary["schema_name"] == "event"
    assert summary["package_status"] == "package_committed"
    assert all(t["status"] == "confirmed" for t in summary["tables"].values())


def test_commit_log_restart_recovers_uncertain_tables() -> None:
    """A process exits mid-load: one table is confirmed, the other was being written.
    On restart only the unconfirmed table is redone."""
    # --- run 1: confirm `event_user`, leave `event_loop_interrupted` mid-flight ---
    load = _commit_log_loader()
    load_id, schema = prepare_load_package(load.load_storage, NORMALIZED_FILES)
    packages = load.load_storage.normalized_packages
    load._commit_log = TableCommitLog(packages, load_id)

    files = _files_by_table(load, load_id)
    load._commit_log_record_prepared(list(load.load_storage.list_new_jobs(load_id)))

    # event_user: committed at the destination and moved to completed (confirmed)
    packages.start_job(load_id, files["event_user"])
    packages.save_pending_transition(load_id, files["event_user"], "completed")
    with Container().injectable_context(
        LoadPackageStateInjectableContext(storage=packages, load_id=load_id)
    ):
        jobs = load.resume_started_jobs(load_id, schema)
        _, finalized, pending_exc = load.complete_jobs(load_id, jobs, schema)
    assert pending_exc is None
    assert [job.job_file_info().table_name for job in finalized] == ["event_user"]

    # event_loop_interrupted: started but the destination never answered, then the
    # process "exits" (no pending transition, job left in started_jobs)
    packages.start_job(load_id, files["event_loop_interrupted"])

    commit_log = TableCommitLog(packages, load_id)
    assert commit_log.table_statuses()["event_user"]["event"] == "confirmed"
    assert commit_log.table_statuses()["event_loop_interrupted"]["event"] == "prepared"

    # --- run 2: a fresh loader verifies the log and recovers ---
    load2 = _commit_log_loader()
    run_all(load2)

    info = load2.load_storage.get_load_package_info(load_id)
    assert info.state == "loaded"
    completed = {
        job.job_file_info.table_name: job for job in info.jobs["completed_jobs"]
    }
    # the confirmed table was not re-executed: its job completed without retries
    assert completed["event_user"].job_file_info.retry_count == 0
    # the uncertain table was re-queued and re-executed exactly once
    assert completed["event_loop_interrupted"].job_file_info.retry_count == 1

    # the log closes with all tables confirmed and the package committed; the re-queued
    # job was already covered by the first prepared batch (no duplicate fingerprinting)
    commit_log = TableCommitLog(load2.load_storage.loaded_packages, load_id)
    assert all(e["event"] == "confirmed" for e in commit_log.table_statuses().values())
    assert commit_log.package_status() == "package_committed"
    assert len(commit_log.prepared_batches("event_loop_interrupted")) == 1


def test_commit_log_fingerprint_mismatch_raises() -> None:
    """The input of a pending job changed after a crash: the package is inconsistent."""
    load = _commit_log_loader()
    load_id, schema = prepare_load_package(load.load_storage, NORMALIZED_FILES)
    packages = load.load_storage.normalized_packages
    load._commit_log = TableCommitLog(packages, load_id)
    load._commit_log_record_prepared(list(load.load_storage.list_new_jobs(load_id)))

    # corrupt the input of a pending job
    victim = load.load_storage.list_new_jobs(load_id)[0]
    with open(packages.storage.make_full_path(victim), "ab") as f:
        f.write(b'{"corrupted": true}\n')

    with pytest.raises(LoadPackageInconsistent):
        load._verify_commit_log(load_id)


def test_commit_log_confirmed_table_missing_job_raises() -> None:
    """A table is confirmed but a completed job file is gone: the package is inconsistent."""
    load = _commit_log_loader()
    load_id, schema = prepare_load_package(load.load_storage, NORMALIZED_FILES)
    packages = load.load_storage.normalized_packages
    load._commit_log = TableCommitLog(packages, load_id)

    files = _files_by_table(load, load_id)
    job_id = ParsedLoadJobFileName.parse(files["event_user"]).job_id()
    packages.start_job(load_id, files["event_user"])
    packages.complete_job(load_id, files["event_user"])
    load._commit_log.record_confirmed("event_user", [job_id])

    # the completed job file disappears (eg. manual cleanup of the wrong folder)
    packages.storage.delete(
        packages.get_job_file_path(load_id, "completed_jobs", files["event_user"])
    )
    with pytest.raises(LoadPackageInconsistent):
        load._verify_commit_log(load_id)


def test_commit_log_failed_table_recorded() -> None:
    """A terminally failed table is recorded and visible in the final package state."""
    loader_config = LoaderConfiguration(
        commit_log_enabled=True,
        raise_on_failed_jobs=False,
        workers=1,
        pool_type="none",
    )
    load = setup_loader(
        client_config=DummyClientConfiguration(fail_prob=1.0), loader_config=loader_config
    )
    load_id, schema = prepare_load_package(load.load_storage, NORMALIZED_FILES)
    run_all(load)

    # the package completes with the failed jobs recorded
    assert load.load_storage.get_load_package_info(load_id).state == "loaded"
    commit_log = TableCommitLog(load.load_storage.loaded_packages, load_id)
    statuses = commit_log.table_statuses()
    assert set(statuses) == {"event_user", "event_loop_interrupted"}
    assert all(event["event"] == "failed" for event in statuses.values())
    assert all(event["message"] for event in statuses.values())

    # the final package record binds the failed tables to the committed load id
    ready = [e for e in commit_log.events() if e["event"] == "package_commit_ready"]
    assert len(ready) == 1
    assert ready[0]["tables"] == {
        "event_user": "failed",
        "event_loop_interrupted": "failed",
    }
    assert commit_log.package_status() == "package_committed"


def test_commit_log_prunes_and_archives() -> None:
    load = _commit_log_loader()
    load_id, schema = prepare_load_package(load.load_storage, NORMALIZED_FILES)
    run_all(load)
    commit_log = TableCommitLog(load.load_storage.loaded_packages, load_id)
    assert commit_log.has_events()

    # age the package beyond the retention and run the loader again: pruning happens
    # at the start of the run even when there is nothing to load
    marker = os.path.join(
        load.load_storage.loaded_packages.get_package_path(load_id),
        PackageStorage.PACKAGE_COMPLETED_FILE_NAME,
    )
    full_marker = load.load_storage.loaded_packages.storage.make_full_path(marker)
    old_ts = pendulum.now().subtract(days=30).timestamp()
    os.utime(full_marker, (old_ts, old_ts))

    load2 = _commit_log_loader(commit_log_retention_days=14.0)
    run_all(load2)

    # the detailed log is deleted, a traceable summary was archived first
    assert not commit_log.has_events()
    summary = json.loads(
        load2.load_storage.storage.load(
            os.path.join(COMMIT_LOG_ARCHIVE_FOLDER, f"{load_id}.json")
        )
    )
    assert summary["load_id"] == load_id
    assert summary["package_status"] == "package_committed"
    assert summary["schema_name"] == "event"
    assert summary["events_count"] > 0
    assert set(summary["tables"]) == {"event_user", "event_loop_interrupted"}
    for table_summary in summary["tables"].values():
        assert table_summary["status"] == "confirmed"
        assert table_summary["fingerprint"].startswith("sha256:")


def test_commit_log_pipeline_end_to_end() -> None:
    """A full pipeline run records the final package state bound to the source position."""
    import hashlib

    os.environ["LOAD__COMMIT_LOG_ENABLED"] = "true"
    pipeline = dlt.pipeline(
        pipeline_name="commit_log_e2e",
        destination="duckdb",
        dataset_name="commit_log_e2e_data",
        pipelines_dir=os.path.join(get_test_storage_root(), uniq_id()),
    )
    info = pipeline.run([{"id": 1}, {"id": 2}], table_name="items")
    load_id = info.loads_ids[0]

    # the commit log of the loaded package closed with all tables confirmed
    loaded_storage = FileStorage(os.path.join(pipeline.working_dir, "load", "loaded"))
    commit_log = TableCommitLog(PackageStorage(loaded_storage, "loaded"), load_id)
    statuses = commit_log.table_statuses()
    assert "items" in statuses
    assert all(event["event"] == "confirmed" for event in statuses.values())
    assert commit_log.package_status() == "package_committed"

    # the recorded state hash matches the pipeline state committed to the destination
    # under the same load id: the package and the source position are consistent
    summary = commit_log.build_summary()
    assert summary["state_hash"].startswith("sha256:")
    with pipeline.sql_client() as client:
        state_table = client.make_qualified_table_name(
            pipeline.default_schema.state_table_name
        )
        rows = client.execute_sql(f'SELECT "state", "_dlt_load_id" FROM {state_table}')
    matching = [row for row in rows if row[1] == load_id]
    assert len(matching) == 1
    state_hash = "sha256:" + hashlib.sha256(matching[0][0].encode("utf-8")).hexdigest()
    assert summary["state_hash"] == state_hash
