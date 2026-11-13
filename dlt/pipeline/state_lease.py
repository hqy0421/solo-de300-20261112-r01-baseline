"""Optional state lease and source isolation for pipelines sharing a working directory.

When enabled via `StateLeaseConfiguration`, the pipeline state is stored in a layout
partitioned by pipeline, source and run generation inside the pipeline working dir:

```
<working_dir>/state/
    lease.json                  # the current lease: owner instance, run generation, expiry
    migrations.jsonl            # append-only audit log (lease lineage, legacy state migration)
    gen_0000000001/             # state partitions of run generation 1
        pipeline.json           # pipeline-level state (everything except `sources`)
        sources/
            <source_name>.json  # state of a single source
```

A pipeline instance may persist state only while it holds the lease for its run
generation. A newer instance takes the lease over with a higher generation which fences
the previous instance: its state writes are rejected with `PipelineStateLeaseLost` and
its extraction loop stops, so a stale instance can never overwrite newer source state
(ie. move incremental cursors back) when several instances share one working directory.

The legacy `state.json` is still written by the lease holder as a compatibility view, so
tooling that reads it (CLI, restore snapshots) keeps working. When the lease is enabled
on a pipeline that already has a legacy `state.json`, the state is migrated into the
partitioned layout once and the migration is recorded in `migrations.jsonl`.

The feature is opt-in. With the lease disabled (default) the pipeline keeps using the
single `state.json` file exactly as before.
"""

import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, cast

from dlt.common import logger
from dlt.common.configuration import known_sections
from dlt.common.pendulum import pendulum
from dlt.common.pipeline import TPipelineState
from dlt.common.storages import FileStorage
from dlt.common.typing import DictStrAny, TypedDict
from dlt.common.utils import uniq_id
from dlt.common.versioned_state import json_decode_state, json_encode_state

from dlt.pipeline.exceptions import PipelineStateGenerationFenced, PipelineStateLeaseLost

STATE_DIR_NAME = "state"
"""Folder in the pipeline working directory that keeps the leased, partitioned state"""
LEASE_FILE_NAME = "lease.json"
"""File with the current state lease"""
MIGRATIONS_FILE_NAME = "migrations.jsonl"
"""Append-only audit log of lease generations and state migrations"""
PIPELINE_PARTITION_FILE_NAME = "pipeline.json"
"""Partition with the pipeline-level state (everything except `sources`)"""
SOURCES_DIR_NAME = "sources"
"""Folder with per-source state partitions"""

GENERATION_DIR_FORMAT = "gen_%010d"
"""Zero-padded run generation folder name so generations sort lexicographically"""

_UNSAFE_FILE_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


class TPipelineStateLease(TypedDict):
    """A lease granting a pipeline instance the right to persist state for one run generation"""

    pipeline_name: str
    instance_id: str
    generation: int
    acquired_at: pendulum.DateTime
    expires_at: pendulum.DateTime
    derived_from_generation: int
    previous_instance_id: Optional[str]


def _safe_file_name(name: str) -> str:
    """Makes a source name safe to use as a file name"""
    return _UNSAFE_FILE_CHARS.sub("_", name)


class StateLeaseStore:
    """Manages the state lease and the generation/source partitioned pipeline state.

    Only the instance named in `lease.json` may write state and only for the lease's run
    generation. A newer instance takes the lease over with a higher generation which
    fences the previous instance: its writes are rejected with `PipelineStateLeaseLost`
    so a stale instance can never overwrite newer state (cursor rollback).
    """

    def __init__(
        self,
        working_dir: str,
        pipeline_name: str,
        instance_id: str = None,
        lease_ttl_seconds: float = 300.0,
        generation: Optional[int] = None,
    ) -> None:
        self._pipeline_name = pipeline_name
        self._instance_id = instance_id or f"instance_{uniq_id(8)}"
        self._lease_ttl_seconds = lease_ttl_seconds
        self._requested_generation = generation
        self._lease: Optional[TPipelineStateLease] = None
        self._storage = FileStorage(os.path.join(working_dir, STATE_DIR_NAME), makedirs=False)

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def generation(self) -> Optional[int]:
        """Run generation of the acquired lease, None when the lease was not acquired yet"""
        return self._lease["generation"] if self._lease else None

    @property
    def lease(self) -> Optional[TPipelineStateLease]:
        return self._lease

    def acquire(self) -> TPipelineStateLease:
        """Takes the lease for a new run generation, fencing any previous instance.

        A live lease of another instance is taken over: the previous instance detects the
        fencing on its next lease check or state write and stops. When a run generation
        was explicitly requested it must be higher than the stored one.
        """
        now = pendulum.now()
        current = self._read_lease()
        previous_generation = current["generation"] if current else 0
        previous_instance_id = current["instance_id"] if current else None
        if self._requested_generation is not None:
            if self._requested_generation <= previous_generation:
                raise PipelineStateGenerationFenced(
                    self._pipeline_name, self._requested_generation, previous_generation
                )
            generation = self._requested_generation
        else:
            generation = previous_generation + 1
        if current and current["instance_id"] != self._instance_id and current["expires_at"] > now:
            logger.warning(
                f"Taking over a live state lease of pipeline {self._pipeline_name} from instance"
                f" {previous_instance_id} at generation {previous_generation}. The previous"
                " instance will be fenced and stop on its next lease check."
            )
        self._lease = TPipelineStateLease(
            pipeline_name=self._pipeline_name,
            instance_id=self._instance_id,
            generation=generation,
            acquired_at=now,
            expires_at=now.add(seconds=self._lease_ttl_seconds),
            derived_from_generation=previous_generation,
            previous_instance_id=previous_instance_id,
        )
        self._write_lease(self._lease)
        self._ensure_lease_held()
        self._append_audit(
            {
                "event": "lease_acquired",
                "at": now,
                "pipeline_name": self._pipeline_name,
                "instance_id": self._instance_id,
                "generation": generation,
                "derived_from_generation": previous_generation,
                "previous_instance_id": previous_instance_id,
            }
        )
        logger.info(
            f"Acquired the state lease of pipeline {self._pipeline_name} for instance"
            f" {self._instance_id} at generation {generation}"
        )
        return self._lease

    def ensure_valid(self, renew: bool = True) -> None:
        """Ensures that this instance still holds the lease, renewing it when due.

        Raises `PipelineStateLeaseLost` when the lease is missing or was taken over by
        another instance. An expired lease still owned by this instance is renewed in
        place: expiry only allows other instances to take the lease over.
        """
        if self._lease is None:
            raise PipelineStateLeaseLost(
                self._pipeline_name,
                self._instance_id,
                0,
                "the lease was never acquired by this instance",
            )
        self._ensure_lease_held()
        if renew:
            lease = self._read_lease()
            now = pendulum.now()
            # renew when half of the ttl passed or the lease already expired under us
            if lease["expires_at"] <= now.add(seconds=self._lease_ttl_seconds / 2):
                lease["expires_at"] = now.add(seconds=self._lease_ttl_seconds)
                self._write_lease(lease)
                # a concurrent acquire may have raced the renewal
                self._ensure_lease_held()

    def load_state(self) -> Optional[TPipelineState]:
        """Loads the merged state of the most recent generation, None if no partitions exist"""
        generation = self._latest_generation()
        if generation is None:
            return None
        gen_dir = self._generation_dir(generation)
        pipeline_doc = json_decode_state(
            self._storage.load(os.path.join(gen_dir, PIPELINE_PARTITION_FILE_NAME))
        )
        state = pipeline_doc["state"]
        sources_dir = os.path.join(gen_dir, SOURCES_DIR_NAME)
        if self._storage.has_folder(sources_dir):
            sources: DictStrAny = {}
            for file_name in self._storage.list_folder_files(sources_dir, to_root=False):
                if not file_name.endswith(".json"):
                    continue
                source_doc = json_decode_state(
                    self._storage.load(os.path.join(sources_dir, file_name))
                )
                # the original source name is kept in the envelope, the file name is sanitized
                sources[source_doc["source_name"]] = source_doc["state"]
            state[known_sections.SOURCES] = sources
        return state  # type: ignore[no-any-return]

    def save_state(self, state: TPipelineState) -> None:
        """Saves `state` into the partitions of the current run generation.

        The instance must hold a valid lease: a fenced instance (lease taken over by a
        newer instance or the working folder replaced) is rejected with
        `PipelineStateLeaseLost` and nothing is written.
        """
        self.ensure_valid(renew=True)
        self._write_partitions(self._lease["generation"], state)

    def migrate_legacy_state(self, state: TPipelineState) -> None:
        """Migrates the legacy single-file pipeline state into the partitioned layout.

        The legacy `state.json` is left in place and an audit record is appended to
        `migrations.jsonl` so the migration stays traceable.
        """
        if self._lease is None:
            raise PipelineStateLeaseLost(
                self._pipeline_name,
                self._instance_id,
                0,
                "the lease was never acquired by this instance",
            )
        generation = self._lease["generation"]
        self._write_partitions(generation, state)
        sources = cast(Dict[str, Dict[str, Any]], state.get(known_sections.SOURCES) or {})
        self._append_audit(
            {
                "event": "legacy_state_migrated",
                "at": pendulum.now(),
                "pipeline_name": self._pipeline_name,
                "instance_id": self._instance_id,
                "from": "state.json",
                "to_generation": generation,
                "state_version": state.get("_state_version"),
                "version_hash": state.get("_version_hash"),
                "sources": sorted(sources.keys()),
            }
        )
        logger.info(
            f"Migrated the legacy state of pipeline {self._pipeline_name} (state version"
            f" {state.get('_state_version')}) into the partitioned layout at generation"
            f" {generation}. The legacy state.json is kept in place."
        )

    def extract_guard(self, check_interval_seconds: float = None) -> Callable[[], None]:
        """Returns a throttled lease check that stops extraction when the lease is lost"""
        if check_interval_seconds is None:
            check_interval_seconds = max(0.05, min(5.0, self._lease_ttl_seconds / 4))
        last_check_at = [0.0]

        def _guard() -> None:
            now = time.monotonic()
            if now - last_check_at[0] < check_interval_seconds:
                return
            last_check_at[0] = now
            self.ensure_valid(renew=True)

        return _guard

    def _ensure_lease_held(self) -> None:
        """Raises `PipelineStateLeaseLost` unless the lease file is held by this instance"""
        lease = self._read_lease()
        if lease is None:
            raise PipelineStateLeaseLost(
                self._pipeline_name,
                self._instance_id,
                self.generation or 0,
                "the lease file is missing (the working folder was wiped or replaced)",
            )
        if lease["instance_id"] != self._instance_id or lease["generation"] != self.generation:
            raise PipelineStateLeaseLost(
                self._pipeline_name,
                self._instance_id,
                self.generation or 0,
                f"the lease is now held by instance `{lease['instance_id']}` at run generation"
                f" {lease['generation']}",
            )

    def _write_partitions(self, generation: int, state: TPipelineState) -> None:
        now = pendulum.now()
        gen_dir = self._generation_dir(generation)
        # write into a temp folder and publish atomically so a crashed save never
        # exposes a partially written generation: readers see either the complete new
        # generation or the complete previous one
        tmp_dir = f".{gen_dir}.tmp"
        bak_dir = f".{gen_dir}.bak"
        self._delete_folder_if_exists(tmp_dir)
        self._delete_folder_if_exists(bak_dir)
        self._storage.create_folder(tmp_dir)
        pipeline_state = {k: v for k, v in state.items() if k != known_sections.SOURCES}
        self._write_partition(
            tmp_dir, PIPELINE_PARTITION_FILE_NAME, generation, now, pipeline_state
        )
        sources = cast(Optional[Dict[str, Dict[str, Any]]], state.get(known_sections.SOURCES))
        if sources is not None:
            sources_dir = os.path.join(tmp_dir, SOURCES_DIR_NAME)
            self._storage.create_folder(sources_dir)
            for source_name, source_state in sources.items():
                self._write_partition(
                    sources_dir,
                    _safe_file_name(source_name) + ".json",
                    generation,
                    now,
                    source_state,
                    source_name=source_name,
                )
        if self._storage.has_folder(gen_dir):
            os.rename(self._storage.make_full_path(gen_dir), self._storage.make_full_path(bak_dir))
        os.rename(self._storage.make_full_path(tmp_dir), self._storage.make_full_path(gen_dir))
        self._delete_folder_if_exists(bak_dir)

    def _delete_folder_if_exists(self, relative_path: str) -> None:
        if self._storage.has_folder(relative_path):
            self._storage.delete_folder(relative_path, recursively=True)

    def _write_partition(
        self,
        folder: str,
        file_name: str,
        generation: int,
        now: pendulum.DateTime,
        state: DictStrAny,
        source_name: str = None,
    ) -> None:
        doc: DictStrAny = {
            "generation": generation,
            "instance_id": self._instance_id,
            "written_at": now,
            "state": state,
        }
        if source_name is not None:
            doc["source_name"] = source_name
        self._storage.save(os.path.join(folder, file_name), json_encode_state(doc))  # type: ignore[arg-type]

    def _generation_dir(self, generation: int) -> str:
        return GENERATION_DIR_FORMAT % generation

    def _latest_generation(self) -> Optional[int]:
        if not self._storage.has_folder(""):
            return None
        generations: List[int] = []
        for dir_name in self._storage.list_folder_dirs("", to_root=False):
            if dir_name.startswith("gen_") and dir_name[len("gen_") :].isdigit():
                generations.append(int(dir_name[len("gen_") :]))
        return max(generations) if generations else None

    def _read_lease(self) -> Optional[TPipelineStateLease]:
        try:
            return cast(TPipelineStateLease, json_decode_state(self._storage.load(LEASE_FILE_NAME)))
        except FileNotFoundError:
            return None

    def _write_lease(self, lease: TPipelineStateLease) -> None:
        self._storage.create_folder("", exists_ok=True)
        self._storage.save(LEASE_FILE_NAME, json_encode_state(lease))  # type: ignore[arg-type]

    def _append_audit(self, record: DictStrAny) -> None:
        self._storage.create_folder("", exists_ok=True)
        path = self._storage.make_full_path(MIGRATIONS_FILE_NAME)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json_encode_state(record) + "\n")  # type: ignore[arg-type]
