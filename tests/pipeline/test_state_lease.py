import json
import os
import time
from typing import Any, Iterator, List

import pytest

import dlt
from dlt.common.json import json as dlt_json
from dlt.common.pendulum import pendulum
from dlt.common.utils import uniq_id

from dlt.pipeline.exceptions import (
    PipelineStateGenerationFenced,
    PipelineStateLeaseLost,
    PipelineStepFailed,
)
from dlt.pipeline.state_lease import (
    LEASE_FILE_NAME,
    MIGRATIONS_FILE_NAME,
    STATE_DIR_NAME,
    StateLeaseStore,
)

from tests.common.configuration.utils import environment
from tests.utils import test_storage


@dlt.resource
def some_data():
    last_value = dlt.current.source_state().get("last_value", 0)
    yield [1, 2, 3]
    dlt.current.source_state()["last_value"] = last_value + 1


@dlt.source
def two_resources_source():
    @dlt.resource
    def resource_a():
        yield 1
        dlt.current.resource_state()["a_cursor"] = "a"

    @dlt.resource
    def resource_b():
        yield 2
        dlt.current.resource_state()["b_cursor"] = "b"

    return resource_a, resource_b


def _enable_lease(environment: Any, ttl_seconds: float = 300.0) -> None:
    environment["PIPELINES__STATE_LEASE__ENABLED"] = "true"
    environment["PIPELINES__STATE_LEASE__LEASE_TTL_SECONDS"] = str(ttl_seconds)


def _state_dir(pipeline_name: str) -> str:
    return os.path.join(dlt.current.pipeline().working_dir, STATE_DIR_NAME)


def _generation_dirs(pipeline_name: str) -> List[str]:
    state_dir = _state_dir(pipeline_name)
    if not os.path.isdir(state_dir):
        return []
    return sorted(d for d in os.listdir(state_dir) if d.startswith("gen_"))


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_audit_log(pipeline_name: str) -> List[Any]:
    path = os.path.join(_state_dir(pipeline_name), MIGRATIONS_FILE_NAME)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [dlt_json.typed_loads(line) for line in f]


def test_lease_disabled_by_default() -> None:
    """Single-instance default behavior is unchanged: only state.json is used"""
    p = dlt.pipeline(pipeline_name="lease_default", destination="dummy")
    assert p._state_lease_store is None
    p.extract(some_data())

    assert os.path.exists(os.path.join(p.working_dir, "state.json"))
    assert not os.path.exists(os.path.join(p.working_dir, STATE_DIR_NAME))
    assert p.state["sources"]["lease_default"]["last_value"] == 1

    # a second instance continues from state.json exactly as before
    p2 = dlt.pipeline(pipeline_name="lease_default", destination="dummy")
    assert p2._state_lease_store is None
    p2.extract(some_data())
    assert p2.state["sources"]["lease_default"]["last_value"] == 2


def test_partitioned_state_layout(environment) -> None:
    """State is partitioned by pipeline, source and run generation"""
    _enable_lease(environment)
    p = dlt.pipeline(pipeline_name="lease_layout", destination="dummy")
    assert p._state_lease_store is not None
    p.extract(some_data())

    state_dir = _state_dir("lease_layout")
    assert _generation_dirs("lease_layout") == ["gen_0000000001"]
    # lease file held by this instance at generation 1
    lease = _read_json(os.path.join(state_dir, LEASE_FILE_NAME))
    assert lease["pipeline_name"] == "lease_layout"
    assert lease["generation"] == 1
    assert lease["instance_id"] == p._state_lease_store.instance_id
    assert lease["derived_from_generation"] == 0

    gen_dir = os.path.join(state_dir, "gen_0000000001")
    # pipeline partition holds everything but the sources
    pipeline_doc = _read_json(os.path.join(gen_dir, "pipeline.json"))
    assert pipeline_doc["generation"] == 1
    assert pipeline_doc["instance_id"] == lease["instance_id"]
    assert "sources" not in pipeline_doc["state"]
    assert pipeline_doc["state"]["pipeline_name"] == "lease_layout"
    # source partition holds only the source state
    source_doc = _read_json(os.path.join(gen_dir, "sources", "lease_layout.json"))
    assert source_doc["source_name"] == "lease_layout"
    assert source_doc["state"] == {"last_value": 1}

    # the legacy state.json is kept as a compatibility view with the same content
    view = _read_json(os.path.join(p.working_dir, "state.json"))
    assert view["sources"]["lease_layout"]["last_value"] == 1
    assert view["_version_hash"] == pipeline_doc["state"]["_version_hash"]

    # the lease acquisition is audited
    records = _read_audit_log("lease_layout")
    assert [r["event"] for r in records] == ["lease_acquired"]
    assert records[0]["generation"] == 1


def test_generation_increments_and_state_continues(environment) -> None:
    """Each new instance runs in a new generation and inherits the recent state"""
    _enable_lease(environment)
    p1 = dlt.pipeline(pipeline_name="lease_gen", destination="dummy")
    p1.extract(some_data())
    gen1 = p1._state_lease_store.generation

    p2 = dlt.pipeline(pipeline_name="lease_gen", destination="dummy")
    assert p2._state_lease_store.generation == gen1 + 1
    p2.extract(some_data())

    assert _generation_dirs("lease_gen") == ["gen_0000000001", "gen_0000000002"]
    # cursor continues across generations, it does not roll back
    source_doc = _read_json(
        os.path.join(_state_dir("lease_gen"), "gen_0000000002", "sources", "lease_gen.json")
    )
    assert source_doc["state"]["last_value"] == 2
    # lineage is audited
    records = _read_audit_log("lease_gen")
    acquired = [r for r in records if r["event"] == "lease_acquired"]
    assert [r["generation"] for r in acquired] == [1, 2]
    assert acquired[1]["derived_from_generation"] == 1
    assert acquired[1]["previous_instance_id"] == p1._state_lease_store.instance_id


def test_stale_instance_cannot_write_back(environment) -> None:
    """An instance fenced by a newer generation cannot overwrite newer state"""
    _enable_lease(environment)
    # instance A extracts and keeps a stale in-memory state
    p_a = dlt.pipeline(pipeline_name="lease_fence", destination="dummy")
    p_a.extract(some_data())
    stale_state = p_a.state
    assert stale_state["sources"]["lease_fence"]["last_value"] == 1

    # instance B takes the lease over and advances the cursor twice
    p_b = dlt.pipeline(pipeline_name="lease_fence", destination="dummy")
    p_b.extract(some_data())
    p_b.extract(some_data())
    assert p_b.state["sources"]["lease_fence"]["last_value"] == 3

    # A cannot write its stale state back: neither directly nor via managed_state
    with pytest.raises(PipelineStateLeaseLost):
        p_a._save_state(stale_state)
    with pytest.raises(PipelineStateLeaseLost):
        with p_a.managed_state() as state:
            state["sources"]["lease_fence"]["last_value"] = -100

    # the newer state of B is intact in both the partitions and the compat view
    assert p_b.state["sources"]["lease_fence"]["last_value"] == 3
    view = _read_json(os.path.join(p_b.working_dir, "state.json"))
    assert view["sources"]["lease_fence"]["last_value"] == 3
    assert _generation_dirs("lease_fence") == ["gen_0000000001", "gen_0000000002"]


def test_lease_lost_stops_extraction(environment) -> None:
    """Extraction aborts when the lease is taken over by another instance mid-run"""
    _enable_lease(environment, ttl_seconds=1.0)
    pipeline_name = "lease_stop"
    took_over = {"done": False}

    @dlt.resource
    def slow_data():
        for i in range(50):
            if i == 2 and not took_over["done"]:
                # simulate another instance taking over the lease mid-extraction
                working_dir = dlt.current.pipeline().working_dir
                StateLeaseStore(working_dir, pipeline_name).acquire()
                took_over["done"] = True
            time.sleep(0.05)
            yield {"i": i}

    p = dlt.pipeline(pipeline_name=pipeline_name, destination="dummy")
    with pytest.raises(PipelineStepFailed) as pip_ex:
        p.extract(slow_data())
    assert isinstance(pip_ex.value.exception, PipelineStateLeaseLost)
    assert took_over["done"]


def test_legacy_state_migration_is_audited(environment) -> None:
    """Enabling the lease migrates the legacy state.json once and records it"""
    # run without the lease: legacy single-file state
    p = dlt.pipeline(pipeline_name="lease_migrate", destination="dummy")
    p.extract(some_data())
    p.extract(some_data())
    legacy_state = p.state
    assert legacy_state["sources"]["lease_migrate"]["last_value"] == 2
    assert not os.path.exists(os.path.join(p.working_dir, STATE_DIR_NAME))

    # enable the lease: the legacy state is migrated into the partitioned layout
    _enable_lease(environment)
    p2 = dlt.pipeline(pipeline_name="lease_migrate", destination="dummy")
    p2.extract(some_data())

    # the cursor continues from the migrated legacy state
    assert p2.state["sources"]["lease_migrate"]["last_value"] == 3
    assert _generation_dirs("lease_migrate") == ["gen_0000000001"]
    # the legacy file is kept in place
    assert os.path.exists(os.path.join(p2.working_dir, "state.json"))
    # the migration is recorded with the pre-migration state identity
    records = _read_audit_log("lease_migrate")
    migrated = [r for r in records if r["event"] == "legacy_state_migrated"]
    assert len(migrated) == 1
    assert migrated[0]["from"] == "state.json"
    assert migrated[0]["to_generation"] == 1
    assert migrated[0]["state_version"] == legacy_state["_state_version"]
    assert migrated[0]["version_hash"] == legacy_state["_version_hash"]
    assert migrated[0]["sources"] == ["lease_migrate"]


def test_source_isolation(environment) -> None:
    """Each source gets its own partition with only its own state"""
    _enable_lease(environment)
    p = dlt.pipeline(pipeline_name="lease_iso", destination="dummy")
    p.extract(two_resources_source().clone(with_name="source_one"))
    p.extract(two_resources_source().clone(with_name="source_two"))

    sources_dir = os.path.join(_state_dir("lease_iso"), "gen_0000000001", "sources")
    assert sorted(os.listdir(sources_dir)) == ["source_one.json", "source_two.json"]
    doc_one = _read_json(os.path.join(sources_dir, "source_one.json"))
    doc_two = _read_json(os.path.join(sources_dir, "source_two.json"))
    assert doc_one["state"]["resources"]["resource_a"] == {"a_cursor": "a"}
    assert "source_two" not in doc_one["state"]
    assert doc_two["state"]["resources"]["resource_b"] == {"b_cursor": "b"}
    # merged view exposes both sources again
    assert set(p.state["sources"].keys()) == {"source_one", "source_two"}


def test_run_end_to_end_with_lease(environment) -> None:
    """extract + normalize + load work with the lease enabled"""
    from dlt.destinations import dummy

    _enable_lease(environment)
    p = dlt.pipeline(
        pipeline_name="lease_run",
        destination=dummy(completed_prob=1),
        dataset_name="lease_run_data",
    )
    info = p.run(some_data())
    assert info.pipeline is p
    assert p.state["sources"]["lease_run"]["last_value"] == 1
    assert _generation_dirs("lease_run") == ["gen_0000000001"]


def test_store_lease_expiry_and_renewal(test_storage) -> None:
    """An expired lease may be taken over; the fenced instance fails afterwards"""
    working_dir = test_storage.make_full_path("pipeline_x")
    store_a = StateLeaseStore(working_dir, "pipeline_x", lease_ttl_seconds=0.3)
    lease_a = store_a.acquire()
    assert lease_a["generation"] == 1

    # renewal extends the expiry while the lease is ours
    time.sleep(0.2)
    store_a.ensure_valid(renew=True)
    renewed = store_a._read_lease()
    assert renewed["expires_at"] > pendulum.now().add(seconds=0.2)

    # let the lease expire: another instance takes it over with the next generation
    time.sleep(0.5)
    store_b = StateLeaseStore(working_dir, "pipeline_x", lease_ttl_seconds=300.0)
    lease_b = store_b.acquire()
    assert lease_b["generation"] == 2
    assert lease_b["previous_instance_id"] == store_a.instance_id

    # the fenced instance fails the lease check and cannot write
    with pytest.raises(PipelineStateLeaseLost):
        store_a.ensure_valid()
    with pytest.raises(PipelineStateLeaseLost):
        store_a.save_state({"_state_version": 1})  # type: ignore[arg-type]
    # the new holder writes fine
    store_b.save_state({"_state_version": 1})  # type: ignore[arg-type]


def test_store_explicit_generation_fencing(test_storage) -> None:
    """An explicitly requested generation must be newer than the stored one"""
    working_dir = test_storage.make_full_path("pipeline_y")
    StateLeaseStore(working_dir, "pipeline_y", generation=10).acquire()
    # same or older generation is fenced
    with pytest.raises(PipelineStateGenerationFenced):
        StateLeaseStore(working_dir, "pipeline_y", generation=10).acquire()
    with pytest.raises(PipelineStateGenerationFenced):
        StateLeaseStore(working_dir, "pipeline_y", generation=5).acquire()
    # a newer generation is accepted
    lease = StateLeaseStore(working_dir, "pipeline_y", generation=11).acquire()
    assert lease["generation"] == 11


def test_store_missing_lease_file(test_storage) -> None:
    """A wiped working folder invalidates the lease"""
    working_dir = test_storage.make_full_path("pipeline_z")
    store = StateLeaseStore(working_dir, "pipeline_z")
    store.acquire()
    os.remove(os.path.join(working_dir, STATE_DIR_NAME, LEASE_FILE_NAME))
    with pytest.raises(PipelineStateLeaseLost):
        store.ensure_valid()
