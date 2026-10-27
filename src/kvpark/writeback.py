"""Bounded RAM-first snapshots with independent foreground RAM restores.

Released is not durable. Dirty or failed entries cannot be evicted implicitly.
The disk implementation and format are shared with TieredStore, not duplicated.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .metrics import Metrics, ratio
from .store import ALIGNMENT, DTYPES, Metadata, TieredStore, _aligned_buffer, _bytes, _round_up

# Stable observability schema for the session layer. See store.COUNTER_SCHEMA.
COUNTER_SCHEMA = (
    "offload_calls", "offload_accepted", "offload_bytes",
    "offload_rejected_immutable", "offload_rejected_pending",
    "offload_rejected_oversize", "offload_rejected_occupied",
    "copy_ok", "copy_errors",
    "released_ok", "persist_ok", "persist_errors", "persist_bytes",
    "retry_calls", "retry_ok", "retry_errors",
    "retry_rejected_state", "retry_rejected_pending",
    "discard_calls", "discard_bytes",
    "read_calls", "read_ok", "read_errors",
    "read_hot", "read_hot_bytes", "read_disk_requests", "read_rejected_busy",
    "prefetch_calls", "prefetch_busy", "prefetch_ram_hits",
    "contains_calls", "contains_hits", "contains_misses",
    "evictions_clean", "evicted_clean_bytes",
    "close_calls", "close_persistence_failures",
)
LATENCY_SCHEMA = ("released_ms", "durable_ms", "retry_ms")


class CacheBusy(RuntimeError):
    """Bounded admission declined; wait for progress, recompute, or keep KV on GPU."""


class PersistenceError(RuntimeError):
    """close drained the workers but at least one snapshot is still RAM-only."""


class _Completion(Future):
    # Ownership has transferred to the store. Cancelling a public notification
    # must never cancel a copy, release a reservation, or drop dirty data.
    def cancel(self):
        return False


@dataclass(frozen=True)
class Offload:
    """Two milestones: input reusable after released; crash-safe after durable.

    Futures are notifications, not cancellation handles. Neither result contains
    a tensor, so retaining this handle does not itself retain GPU memory.
    """

    released: Future[None] = field(default_factory=_Completion)
    durable: Future[Metadata] = field(default_factory=_Completion)


@dataclass
class _Entry:
    key: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    cost: int
    handle: Offload = field(default_factory=Offload)
    host: torch.Tensor | None = None
    state: str = "copying"
    readers: int = 0
    started_ns: int = 0


@dataclass
class _Copy:
    entry: _Entry
    source: torch.Tensor | None
    ready: object


class RamFirstStore:
    """Exact snapshots, bounded dirty RAM, background disk, independent hot reads.

    Admission is nonblocking: CacheBusy means nothing was accepted. Full host
    storage is reserved before retaining the GPU input. Do not mutate an input
    until released completes or an output until get_async completes. RAM budget
    covers owned allocations/reservations, not allocator high-water or page cache.

    close() drains accepted work, then raises PersistenceError on unresolved
    persistence failures. Failed RAM-only entries remain readable until close,
    retry_persist, or an explicit discard. A process crash loses non-durable data.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        ram_bytes: int = 2 * 1024**3,
        max_pending: int = 2,
        max_reads: int = 8,
        **disk_options,
    ):
        if ram_bytes <= ALIGNMENT or max_pending < 1 or max_reads < 1:
            raise ValueError("positive RAM capacity and positive queue limits are required")
        if "foreground_first" in disk_options:
            raise ValueError("RamFirstStore always enables disk foreground priority")
        self._disk = TieredStore(root, ram_bytes=0, foreground_first=True, **disk_options)
        self.device = self._disk.device
        self.ram_budget = ram_bytes
        self.max_pending = max_pending
        self.max_reads = max_reads
        self._lock = threading.RLock()
        self.metrics = Metrics()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._disk_keys = set(self._disk._disk)
        self._used = self._pending = self._reads = self._peak = 0
        self._closed = False
        self._closing = threading.Event()
        self._copy = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-offload")
        self._hot = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-hot")
        self._offload_stream = self._hot_stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_args):
        try:
            self.close()
        except PersistenceError:
            if exc_type is None:
                raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError("store is closed")

    def _reserve(self, cost):
        if cost > self.ram_budget:
            self.metrics.add("offload_rejected_oversize")
            raise CacheBusy("snapshot exceeds RAM capacity; use streamed TieredStore.put")
        candidates = [
            key
            for key, entry in self._entries.items()
            if entry.state == "clean" and not entry.readers
        ]
        reclaimable = sum(self._entries[key].cost for key in candidates)
        if self._used + cost - reclaimable > self.ram_budget:
            self.metrics.add("offload_rejected_occupied")
            raise CacheBusy("RAM occupied by dirty, failed, or actively read snapshots")
        for key in candidates:
            if self._used + cost <= self.ram_budget:
                break
            old = self._entries.pop(key)
            self._used -= old.cost
            old.host = None
            self.metrics.add("evictions_clean")
            self.metrics.add("evicted_clean_bytes", old.cost)
        self._used += cost
        self._peak = max(self._peak, self._used)

    def offload_async(self, key: str, tensor: torch.Tensor) -> Offload:
        self.metrics.add("offload_calls")
        self._disk._check_key(key)
        if tensor.device != self.device:
            raise ValueError(f"input must be on {self.device}")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError("input must be contiguous and strided")
        if not 1 <= tensor.ndim <= 8 or not tensor.numel():
            raise ValueError("input must be nonempty with rank 1..8")
        if str(tensor.dtype).removeprefix("torch.") not in DTYPES:
            raise ValueError("unsupported tensor dtype")
        nbytes = tensor.numel() * tensor.element_size()
        if _round_up(nbytes) > self._disk.disk_budget:
            raise ValueError("snapshot exceeds disk budget")
        ready = None
        if self.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        with self._lock:
            self._check_open()
            if key in self._entries or key in self._disk_keys:
                self.metrics.add("offload_rejected_immutable")
                raise FileExistsError("snapshot keys are immutable")
            if self._pending >= self.max_pending:
                self.metrics.add("offload_rejected_pending")
                raise CacheBusy("pending archive count reached max_pending")
            entry = _Entry(
                key,
                tuple(tensor.shape),
                tensor.dtype,
                nbytes,
                _round_up(nbytes) + ALIGNMENT,
                started_ns=time.perf_counter_ns(),
            )
            self._reserve(entry.cost)
            self._pending += 1
            self._entries[key] = entry
            self.metrics.add("offload_accepted")
            self.metrics.add("offload_bytes", nbytes)
            self._copy.submit(self._copy_input, _Copy(entry, tensor, ready))
            return entry.handle

    def _copy_input(self, job):
        entry = job.entry
        error = host = None
        try:
            host = _aligned_buffer(_round_up(entry.nbytes), self.device.type == "cuda")
            if self.device.type == "cuda":
                if self._offload_stream is None:
                    self._offload_stream = torch.cuda.Stream(device=self.device)
                self._offload_stream.wait_event(job.ready)
                with torch.cuda.stream(self._offload_stream):
                    host[: entry.nbytes].copy_(_bytes(job.source), non_blocking=True)
                self._offload_stream.synchronize()
            else:
                host[: entry.nbytes].copy_(_bytes(job.source))
            with self._lock:
                entry.host = host[: entry.nbytes]
                entry.state = "dirty"
            self.metrics.add("copy_ok")
        except BaseException as exc:
            # Drain a partially enqueued copy before relinquishing source/host.
            if self._offload_stream is not None:
                try:
                    self._offload_stream.synchronize()
                except BaseException:
                    pass  # A poisoned CUDA context must still complete notifications.
            error = exc.with_traceback(None)
        finally:
            job.source = job.ready = host = None
        if error is not None:
            with self._lock:
                self._entries.pop(entry.key)
                self._used -= entry.cost
                self._pending -= 1
            self.metrics.add("copy_errors")
            entry.handle.released.set_exception(error)
            entry.handle.durable.set_exception(error)
            return
        # Submit before notifying clients: close waits the copy worker before
        # closing disk, and clients may synchronously wait on these milestones.
        self._disk._submit(self._persist, entry, priority=10)
        # Milestone 1: the GPU input may be reused. This is the number the
        # RAM-first design exists to protect, so it is measured separately from
        # the durability milestone below.
        self.metrics.observe("released_ms", time.perf_counter_ns() - entry.started_ns)
        self.metrics.add("released_ok")
        entry.handle.released.set_result(None)

    def _persist(self, entry):
        error = meta = None
        try:
            tensor = entry.host.view(entry.dtype).reshape(entry.shape)
            meta = self._disk._put(entry.key, tensor, None)
        except BaseException as exc:
            error = exc.with_traceback(None)
        tensor = None
        with self._lock:
            self._disk_keys = set(self._disk._disk)
            self._pending -= 1
            entry.state = "failed" if error else "clean"
        if error:
            self.metrics.add("persist_errors")
            entry.handle.durable.set_exception(error)
        else:
            # Milestone 2: crash-safe. Until this fires the data exists only in
            # host RAM, and a process crash loses it.
            self.metrics.observe("durable_ms", time.perf_counter_ns() - entry.started_ns)
            self.metrics.add("persist_ok")
            self.metrics.add("persist_bytes", entry.nbytes)
            entry.handle.durable.set_result(meta)

    def retry_persist(self, key: str) -> Future[Metadata]:
        """Retry a failed archive without copying or retaining the GPU input."""
        self.metrics.add("retry_calls")
        with self._lock:
            self._check_open()
            entry = self._entries[key]
            if entry.state != "failed":
                self.metrics.add("retry_rejected_state")
                raise ValueError("only failed entries can be retried")
            if self._pending >= self.max_pending:
                self.metrics.add("retry_rejected_pending")
                raise CacheBusy("pending archive count reached max_pending")
            # If directory fsync failed after rename, durability was not reported
            # but the disk entry may exist. Verify, sync, and register it below.
            entry.handle = Offload(released=entry.handle.released)
            entry.state = "dirty"
            self._pending += 1
            self._disk._submit(self._retry, entry, priority=10)
            return entry.handle.durable

    def _retry(self, entry):
        path = self._disk.root / entry.key
        if path.exists():
            try:
                meta = Metadata.read(path, self._disk.disk_budget)
                # A post-rename entry belongs to this failed job.
                # Reuse the normal verified read before declaring it durable.
                self._disk._read(meta, None)
                self._disk._fsync_directory(self._disk.root)
                if entry.key not in self._disk._disk:
                    self._disk._evict_disk(meta.stored_bytes)
                    self._disk._disk[entry.key] = meta
                    self._disk._disk_used += meta.stored_bytes
            except BaseException as exc:
                with self._lock:
                    entry.state = "failed"
                    self._pending -= 1
                self.metrics.add("retry_errors")
                entry.handle.durable.set_exception(exc.with_traceback(None))
                return
            with self._lock:
                self._disk_keys = set(self._disk._disk)
                entry.state = "clean"
                self._pending -= 1
            self.metrics.observe("retry_ms", time.perf_counter_ns() - entry.started_ns)
            self.metrics.add("retry_ok")
            entry.handle.durable.set_result(meta)
        else:
            self._persist(entry)
            # A retry that re-ran the archive counts as a retry success, not
            # only as a persist success, otherwise retry rate reads as 0%.
            self.metrics.observe("retry_ms", time.perf_counter_ns() - entry.started_ns)
            if entry.state == "clean":
                self.metrics.add("retry_ok")

    def discard(self, key: str):
        """Explicitly drop a clean/failed RAM copy; never drop an in-flight one."""
        with self._lock:
            self._check_open()
            entry = self._entries[key]
            if entry.state not in ("clean", "failed") or entry.readers:
                raise CacheBusy("entry is in flight")
            self._entries.pop(key)
            self._used -= entry.cost
            entry.host = None
            self.metrics.add("discard_calls")
            self.metrics.add("discard_bytes", entry.cost)

    def _finish_read(self, future, result=None, error=None, entry=None):
        with self._lock:
            self._reads -= 1
            if entry is not None:
                entry.readers -= 1
        if error is not None:
            self.metrics.add("read_errors")
            future.set_exception(error.with_traceback(None))
        else:
            self.metrics.add("read_ok")
            future.set_result(result)

    def get_async(self, key: str, *, out=None) -> Future[torch.Tensor]:
        self.metrics.add("read_calls")
        self._disk._check_key(key)
        ready = None
        if out is not None and out.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(out.device))
        future = _Completion()
        with self._lock:
            self._check_open()
            if self._reads >= self.max_reads:
                self.metrics.add("read_rejected_busy")
                raise CacheBusy("read queue reached max_reads")
            self._reads += 1
            entry = self._entries.get(key)
            if entry is None:
                self.metrics.add("read_disk_requests")
                # Foreground disk jobs jump queued writes, not an in-flight I/O.
                internal = self._disk._submit(self._disk._get, key, out, ready, priority=0)

                def done(value):
                    try:
                        result = value.result()
                    except BaseException as exc:
                        self._finish_read(future, error=exc)
                    else:
                        self._finish_read(future, result)

                internal.add_done_callback(done)
            else:
                self._entries.move_to_end(key)
                entry.readers += 1
                self.metrics.add("read_hot")
                self.metrics.add("read_hot_bytes", entry.nbytes)

                def copied(value):
                    try:
                        value.result()
                    except BaseException as exc:
                        self._finish_read(future, error=exc, entry=entry)
                    else:
                        self._hot.submit(self._restore_hot, entry, out, ready, future)

                entry.handle.released.add_done_callback(copied)
        return future

    def get(self, key: str, *, out=None):
        return self.get_async(key, out=out).result()

    def _restore_hot(self, entry, out, ready, future):
        error = None
        try:
            if out is None:
                out = torch.empty(entry.shape, dtype=entry.dtype, device=self.device)
            if (
                tuple(out.shape) != entry.shape
                or out.dtype != entry.dtype
                or out.device != self.device
                or not out.is_contiguous()
                or out.layout != torch.strided
            ):
                raise ValueError("destination shape, dtype, device, or layout does not match")
            with torch.inference_mode():
                if self.device.type == "cuda":
                    if self._hot_stream is None:
                        self._hot_stream = torch.cuda.Stream(device=self.device)
                    if ready is not None:
                        self._hot_stream.wait_event(ready)
                    with torch.cuda.stream(self._hot_stream):
                        _bytes(out).copy_(entry.host, non_blocking=True)
                    self._hot_stream.synchronize()
                else:
                    _bytes(out).copy_(entry.host)
        except BaseException as exc:
            if self._hot_stream is not None:
                try:
                    self._hot_stream.synchronize()
                except BaseException:
                    pass
            error = exc.with_traceback(None)
        if error is not None:
            out = None
        self._finish_read(future, out, error, entry)

    def prefetch(self, key: str) -> Future[bool]:
        """Explicit disk-to-RAM admission. False means RAM is unavailable.

        Like foreground reads, these requests count toward max_reads. They run
        before queued archives but after queued foreground cold restores.
        """
        self.metrics.add("prefetch_calls")
        self._disk._check_key(key)
        future = _Completion()
        with self._lock:
            self._check_open()
            if self._reads >= self.max_reads:
                self.metrics.add("prefetch_busy")
                raise CacheBusy("read queue reached max_reads")
            self._reads += 1
            self._disk._submit(self._prefetch, key, future, priority=5)
        return future

    def _prefetch(self, key, future):
        entry = None
        try:
            with self._lock:
                existing = self._entries.get(key)
            if existing is not None:
                self.metrics.add("prefetch_ram_hits")
                # A copy may be queued independently; do not block the disk
                # worker waiting on its callbacks or consume another allocation.
                def done(value):
                    try:
                        value.result()
                    except BaseException as exc:
                        self._finish_read(future, error=exc)
                    else:
                        self._finish_read(future, True)

                existing.handle.released.add_done_callback(done)
                return
            meta = self._disk._metadata(key)
            cost = _round_up(meta.nbytes) + ALIGNMENT
            with self._lock:
                try:
                    self._reserve(cost)
                except CacheBusy:
                    self._finish_read(future, False)
                    return
                entry = _Entry(
                    key, meta.shape, DTYPES[meta.dtype], meta.nbytes, cost, state="loading"
                )
                self._entries[key] = entry
            host = _aligned_buffer(_round_up(meta.nbytes), self.device.type == "cuda")
            self._disk._read(meta, host[: meta.nbytes], host_storage=host)
            with self._lock:
                entry.host = host[: meta.nbytes]
                host = None
                entry.state = "clean"
            entry.handle.released.set_result(None)
            entry.handle.durable.set_result(meta)
        except BaseException as exc:
            host = None
            error = exc.with_traceback(None)
            if entry is not None:
                with self._lock:
                    self._entries.pop(key)
                    self._used -= entry.cost
                entry.handle.released.set_exception(error)
                entry.handle.durable.set_exception(error)
            self._finish_read(future, error=error)
        else:
            self._finish_read(future, True)

    def contains(self, key):
        self._disk._check_key(key)
        with self._lock:
            self._check_open()
            hit = key in self._entries or key in self._disk_keys
        self.metrics.add("contains_calls")
        self.metrics.add("contains_hits" if hit else "contains_misses")
        return hit

    def stats(self):
        # No disk job: diagnostics and admission must not wait behind an archive.
        with self._lock:
            self._check_open()
            metrics = self.metrics.snapshot(COUNTER_SCHEMA, LATENCY_SCHEMA)
            counters = metrics["counters"]
            return {
                "ram_reserved_bytes": self._used,
                "ram_budget_bytes": self.ram_budget,
                "ram_peak_reserved_bytes": self._peak,
                "ram_entries": len(self._entries),
                "dirty_bytes": sum(
                    e.cost
                    for e in self._entries.values()
                    if e.state in ("copying", "dirty", "failed")
                ),
                "failed_entries": sum(e.state == "failed" for e in self._entries.values()),
                "pending_archives": self._pending,
                "max_pending": self.max_pending,
                "pending_reads": self._reads,
                "max_reads": self.max_reads,
                "disk_entries": len(self._disk_keys),
                "pinned": self.device.type == "cuda",
                "metrics": metrics,
                "hit_rate": {
                    "read": ratio(
                        counters.get("read_hot", 0),
                        counters.get("read_hot", 0) + counters.get("read_disk_requests", 0),
                    ),
                    "contains": ratio(
                        counters.get("contains_hits", 0), counters.get("contains_calls", 0)
                    ),
                    "persist": ratio(
                        counters.get("persist_ok", 0),
                        counters.get("persist_ok", 0) + counters.get("persist_errors", 0),
                    ),
                },
            }

    def close(self):
        self.metrics.add("close_calls")
        with self._lock:
            owner = not self._closed
            self._closed = True
        if not owner:
            self._closing.wait()
            return
        try:
            self._copy.shutdown(wait=True)
            # Disk callbacks may still complete reads. Drain before shutting hot.
            self._disk.close()
            self._hot.shutdown(wait=True)
            failed = [key for key, entry in self._entries.items() if entry.state == "failed"]
            for entry in self._entries.values():
                entry.host = None
            self._entries.clear()
            self._used = 0
            if failed:
                self.metrics.add("close_persistence_failures", len(failed))
                raise PersistenceError(f"{len(failed)} snapshot(s) were not durable at close")
        finally:
            self._closing.set()
