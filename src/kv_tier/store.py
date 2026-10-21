"""Write-through disk snapshots, pinned-RAM LRU, and bounded CUDA staging.

The worker serializes store operations. Within a disk restore, the CPU reads
chunk N+1 while CUDA copies chunk N. CUDA events protect every staging buffer
from reuse. Direct I/O, when explicitly requested, never silently falls back.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import uuid
import zlib
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ALIGNMENT = 4096
KEY_RE = re.compile(r"[0-9a-f]{64}\Z")
DTYPES = {
    name: getattr(torch, name)
    for name in ("float16", "bfloat16", "float32", "float64", "int8", "uint8", "int32", "int64")
}


class CacheCorruption(RuntimeError):
    """The persisted data or manifest failed validation."""


def _round_up(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().view(torch.uint8).reshape(-1)


def _view(tensor: torch.Tensor) -> memoryview:
    return memoryview(tensor.numpy()).cast("B")


def _aligned_buffer(size: int, pinned: bool) -> torch.Tensor:
    base = torch.empty(size + ALIGNMENT, dtype=torch.uint8, pin_memory=pinned)
    start = (-base.data_ptr()) % ALIGNMENT
    return base[start : start + size]


@dataclass(frozen=True)
class Metadata:
    version: int
    key: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    stored_bytes: int
    chunk_bytes: int
    crc32_chunks: tuple[str, ...]

    @classmethod
    def read(cls, directory: Path, max_bytes: int) -> Metadata:
        try:
            manifest = directory / "manifest.json"
            if manifest.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("oversized manifest")
            raw = json.loads(manifest.read_text())
            raw["shape"] = tuple(raw["shape"])
            raw["crc32_chunks"] = tuple(raw["crc32_chunks"])
            meta = cls(**raw)
            if any(
                type(value) is not int
                for value in (meta.version, meta.nbytes, meta.stored_bytes, meta.chunk_bytes)
            ):
                raise ValueError("manifest sizes and format version must be integers")
            if meta.version != 1 or meta.key != directory.name or not KEY_RE.fullmatch(meta.key):
                raise ValueError("invalid identity or format version")
            if meta.dtype not in DTYPES or not 1 <= len(meta.shape) <= 8:
                raise ValueError("unsupported dtype or tensor rank")
            if any(type(x) is not int or x <= 0 for x in meta.shape):
                raise ValueError("invalid tensor shape")
            expected = (
                math.prod(meta.shape) * torch.empty((), dtype=DTYPES[meta.dtype]).element_size()
            )
            if expected != meta.nbytes or not 0 < meta.nbytes <= max_bytes:
                raise ValueError("invalid or excessive tensor size")
            if meta.stored_bytes != _round_up(meta.nbytes):
                raise ValueError("invalid stored size")
            if type(meta.chunk_bytes) is not int or meta.chunk_bytes < ALIGNMENT:
                raise ValueError("invalid chunk size")
            if meta.chunk_bytes % ALIGNMENT or len(meta.crc32_chunks) != math.ceil(
                meta.nbytes / meta.chunk_bytes
            ):
                raise ValueError("invalid chunk table")
            if any(
                not isinstance(x, str) or not re.fullmatch(r"[0-9a-f]{8}", x)
                for x in meta.crc32_chunks
            ):
                raise ValueError("invalid checksum")
            if (directory / "data.bin").stat().st_size != meta.stored_bytes:
                raise ValueError("truncated or oversized data")
            return meta
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            raise CacheCorruption(f"invalid snapshot {directory.name}: {exc}") from exc


class TieredStore:
    """A single-owner, bounded, write-through tensor snapshot cache.

    This stores exact tensor bytes. It does not reduce the bytes required by
    dense attention or provide transparent active-decode offload. Use one root
    per process; an OS lock rejects concurrent owners. All returned futures
    complete only after their CUDA transfers finish. Do not mutate put inputs,
    or load destinations, until the corresponding future completes.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        ram_bytes: int = 256 * 1024**2,
        disk_bytes: int = 16 * 1024**3,
        chunk_bytes: int = 8 * 1024**2,
        staging_slots: int = 2,
        device: str = "cpu",
        direct_io: bool = False,
        verify: bool = True,
    ):
        if ram_bytes < 0 or disk_bytes <= 0:
            raise ValueError("RAM budget must be nonnegative and disk budget positive")
        if chunk_bytes < ALIGNMENT or chunk_bytes % ALIGNMENT or staging_slots < 1:
            raise ValueError(
                "chunk size must be 4096-aligned; at least one staging slot is required"
            )
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("only CPU and CUDA destinations are supported")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
        if direct_io and (not hasattr(os, "O_DIRECT") or not hasattr(os, "preadv")):
            raise RuntimeError("direct I/O requires Linux O_DIRECT and preadv")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.ram_budget = ram_bytes
        self.disk_budget = disk_bytes
        self.chunk_bytes = chunk_bytes
        self.staging_slots = staging_slots
        self.direct_io = direct_io
        self.verify = verify
        self._ram: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._disk: OrderedDict[str, Metadata] = OrderedDict()
        self._ram_used = 0
        self._disk_used = 0
        self._closed = False
        self._mutex = threading.Lock()
        self._lock = (self.root / ".owner.lock").open("a+b")
        try:
            import fcntl

            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, OSError) as exc:
            self._lock.close()
            raise RuntimeError("cache root requires an exclusive POSIX file lock") from exc
        try:
            for entry in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime_ns):
                if entry.name.startswith(".pending-") and entry.is_dir():
                    shutil.rmtree(entry)
                elif KEY_RE.fullmatch(entry.name) and entry.is_dir():
                    meta = Metadata.read(entry, self.disk_budget)
                    self._disk[entry.name] = meta
                    self._disk_used += meta.stored_bytes
            self._evict_disk(0)
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-tier")
            self._buffers: list[torch.Tensor] = []
            self._stream = None
        except BaseException:
            self._lock.close()
            raise

    def __enter__(self) -> TieredStore:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        with self._mutex:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=True)
        self._ram.clear()
        self._buffers.clear()
        self._lock.close()

    def _submit(self, function, *args) -> Future:
        with self._mutex:
            if self._closed:
                raise RuntimeError("store is closed")
            return self._pool.submit(function, *args)

    def _check_key(self, key: str) -> None:
        if not isinstance(key, str) or not KEY_RE.fullmatch(key):
            raise ValueError("key must be a lowercase SHA-256 hex digest")

    def _setup_transfer(self) -> None:
        if self._buffers:
            return
        pinned = self.device.type == "cuda"
        self._buffers = [
            _aligned_buffer(self.chunk_bytes, pinned) for _ in range(self.staging_slots)
        ]
        if pinned:
            self._stream = torch.cuda.Stream(device=self.device)

    def _evict_ram(self, incoming: int) -> None:
        while self._ram and self._ram_used + incoming > self.ram_budget:
            _, tensor = self._ram.popitem(last=False)
            self._ram_used -= tensor.untyped_storage().nbytes()

    def _admit(self, key: str, tensor: torch.Tensor) -> None:
        cost = tensor.untyped_storage().nbytes()
        if cost > self.ram_budget:
            return
        self._evict_ram(cost)
        self._ram[key] = tensor
        self._ram_used += cost

    def _evict_disk(self, incoming: int) -> None:
        while self._disk and self._disk_used + incoming > self.disk_budget:
            key, meta = self._disk.popitem(last=False)
            shutil.rmtree(self.root / key)
            self._disk_used -= meta.stored_bytes
            cached = self._ram.pop(key, None)
            if cached is not None:
                self._ram_used -= cached.untyped_storage().nbytes()

    def _metadata(self, key: str) -> Metadata:
        self._check_key(key)
        if key not in self._disk:
            raise KeyError(key)
        meta = Metadata.read(self.root / key, self.disk_budget)
        self._disk.move_to_end(key)
        return meta

    def contains(self, key: str) -> bool:
        self._check_key(key)
        return self._submit(lambda: key in self._disk).result()

    def stats(self) -> dict:
        return self._submit(
            lambda: {
                "ram_used_bytes": self._ram_used,
                "ram_budget_bytes": self.ram_budget,
                "ram_entries": len(self._ram),
                "disk_used_bytes": self._disk_used,
                "disk_budget_bytes": self.disk_budget,
                "disk_entries": len(self._disk),
                "staging_bytes": sum(x.untyped_storage().nbytes() for x in self._buffers),
                "pinned": self.device.type == "cuda",
                "direct_io": self.direct_io,
                "verify": self.verify,
            }
        ).result()

    def clear_ram(self) -> None:
        def clear():
            self._ram.clear()
            self._ram_used = 0

        self._submit(clear).result()

    def put_async(self, key: str, tensor: torch.Tensor) -> Future[Metadata]:
        self._check_key(key)
        if tensor.device != self.device:
            raise ValueError(f"input must be on {self.device}, not {tensor.device}")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError("input must be a contiguous, strided tensor; pack explicitly")
        if not 1 <= tensor.ndim <= 8 or tensor.numel() == 0:
            raise ValueError("input must be a nonempty tensor with rank 1..8")
        if str(tensor.dtype).removeprefix("torch.") not in DTYPES:
            raise ValueError("unsupported tensor dtype")
        if _round_up(tensor.numel() * tensor.element_size()) > self.disk_budget:
            raise ValueError("snapshot exceeds the disk budget")
        ready = None
        if self.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        return self._submit(self._put, key, tensor, ready)

    def put(self, key: str, tensor: torch.Tensor) -> Metadata:
        return self.put_async(key, tensor).result()

    def _put(self, key: str, tensor: torch.Tensor, ready) -> Metadata:
        if key in self._disk:
            raise FileExistsError("snapshot keys are immutable; choose a new semantic key")
        self._setup_transfer()
        if ready is not None:
            self._stream.wait_event(ready)
        source = _bytes(tensor)
        size = source.numel()
        stored = _round_up(size)
        self._evict_disk(stored)
        pending = self.root / f".pending-{uuid.uuid4().hex}"
        pending.mkdir(mode=0o700)
        checksums = []
        count = math.ceil(size / self.chunk_bytes)
        events = [None] * len(self._buffers)

        def schedule(index):
            slot = index % len(self._buffers)
            offset = index * self.chunk_bytes
            length = min(size - offset, self.chunk_bytes)
            buffer = self._buffers[slot]
            buffer[length : _round_up(length)].zero_()
            if self._stream is not None:
                with torch.cuda.stream(self._stream):
                    buffer[:length].copy_(source[offset : offset + length], non_blocking=True)
                    events[slot] = torch.cuda.Event()
                    events[slot].record(self._stream)
            else:
                buffer[:length].copy_(source[offset : offset + length])

        try:
            with (pending / "data.bin").open("wb", buffering=0) as output:
                os.chmod(pending / "data.bin", 0o600)
                for index in range(min(count, len(self._buffers))):
                    schedule(index)
                for index in range(count):
                    slot = index % len(self._buffers)
                    if events[slot] is not None:
                        events[slot].synchronize()
                    length = min(size - index * self.chunk_bytes, self.chunk_bytes)
                    view = _view(self._buffers[slot][: _round_up(length)])
                    checksums.append(f"{zlib.crc32(view[:length]):08x}")
                    while view:
                        written = output.write(view)
                        if not written:
                            raise OSError("short disk write")
                        view = view[written:]
                    if index + len(self._buffers) < count:
                        schedule(index + len(self._buffers))
                os.fsync(output.fileno())
            meta = Metadata(
                1,
                key,
                tuple(tensor.shape),
                str(tensor.dtype).removeprefix("torch."),
                size,
                stored,
                self.chunk_bytes,
                tuple(checksums),
            )
            manifest = pending / "manifest.json"
            with manifest.open("w") as output:
                os.chmod(manifest, 0o600)
                json.dump(asdict(meta), output, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            self._fsync_directory(pending)
            pending.rename(self.root / key)
            self._fsync_directory(self.root)
            self._disk[key] = meta
            self._disk_used += stored
            return meta
        finally:
            for event in events:
                if event is not None:
                    event.synchronize()
            if pending.exists():
                shutil.rmtree(pending)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def prefetch(self, key: str) -> Future[bool]:
        """Populate the pinned-RAM LRU; False means the item exceeds its budget."""
        self._check_key(key)

        def run():
            meta = self._metadata(key)
            if key in self._ram:
                self._ram.move_to_end(key)
                return True
            allocation = _round_up(meta.nbytes) + ALIGNMENT
            if allocation > self.ram_budget:
                return False
            self._evict_ram(allocation)
            host = _aligned_buffer(_round_up(meta.nbytes), self.device.type == "cuda")
            self._read(meta, host[: meta.nbytes], host_storage=host)
            self._admit(key, host[: meta.nbytes])
            return True

        return self._submit(run)

    def get_async(self, key: str, *, out: torch.Tensor | None = None) -> Future[torch.Tensor]:
        self._check_key(key)
        destination_ready = None
        if out is not None and out.device.type == "cuda":
            destination_ready = torch.cuda.Event()
            destination_ready.record(torch.cuda.current_stream(out.device))
        return self._submit(self._get, key, out, destination_ready)

    def get(self, key: str, *, out: torch.Tensor | None = None) -> torch.Tensor:
        return self.get_async(key, out=out).result()

    def _get(self, key: str, out: torch.Tensor | None, destination_ready) -> torch.Tensor:
        meta = self._metadata(key)
        if out is None:
            out = torch.empty(meta.shape, dtype=DTYPES[meta.dtype], device=self.device)
        if (
            tuple(out.shape) != meta.shape
            or out.dtype != DTYPES[meta.dtype]
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError("destination shape, dtype, device, or layout does not match")
        self._setup_transfer()
        if destination_ready is not None:
            self._stream.wait_event(destination_ready)
        destination = _bytes(out)
        if key in self._ram:
            self._ram.move_to_end(key)
            host = self._ram[key]
            if self._stream is not None:
                with torch.cuda.stream(self._stream):
                    destination.copy_(host, non_blocking=True)
                self._stream.synchronize()
            else:
                destination.copy_(host)
        else:
            self._read(meta, destination)
        return out

    def _read(self, meta: Metadata, destination: torch.Tensor, host_storage=None) -> None:
        self._setup_transfer()
        flags = os.O_RDONLY | (getattr(os, "O_DIRECT", 0) if self.direct_io else 0)
        descriptor = os.open(self.root / meta.key / "data.bin", flags)
        events = [None] * len(self._buffers)
        try:
            # Read chunks using the persisted chunk size, independent of the new
            # process's staging size. This also preserves checksum boundaries.
            for index, expected_hash in enumerate(meta.crc32_chunks):
                offset = index * meta.chunk_bytes
                length = min(meta.nbytes - offset, meta.chunk_bytes)
                padded = _round_up(length)
                slot = index % len(self._buffers)
                if events[slot] is not None:
                    events[slot].synchronize()
                if host_storage is not None:
                    buffer = host_storage[offset : offset + padded]
                elif padded <= self.chunk_bytes:
                    buffer = self._buffers[slot][:padded]
                else:
                    raise ValueError("reopen with chunk_bytes >= the stored chunk_bytes")
                view = _view(buffer)
                done = 0
                while done < padded:
                    if hasattr(os, "preadv"):
                        amount = os.preadv(descriptor, [view[done:]], offset + done)
                    else:
                        os.lseek(descriptor, offset + done, os.SEEK_SET)
                        amount = os.readv(descriptor, [view[done:]])
                    if amount == 0:
                        raise CacheCorruption("unexpected EOF in tensor payload")
                    done += amount
                if self.verify and f"{zlib.crc32(view[:length]):08x}" != expected_hash:
                    raise CacheCorruption(f"payload checksum mismatch at chunk {index}")
                if host_storage is None:
                    if self._stream is not None:
                        with torch.cuda.stream(self._stream):
                            destination[offset : offset + length].copy_(
                                buffer[:length], non_blocking=True
                            )
                            events[slot] = torch.cuda.Event()
                            events[slot].record(self._stream)
                    else:
                        destination[offset : offset + length].copy_(buffer[:length])
        finally:
            for event in events:
                if event is not None:
                    event.synchronize()
            os.close(descriptor)
