import gc
import hashlib
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from kv_tier import CacheBusy, PersistenceError, RamFirstStore, TieredStore
from kv_tier.scheduler import PriorityWorker


def key(i=0):
    return hashlib.sha256(f"writeback-{i}".encode()).hexdigest()


def barrier_writer(store, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    write = store._disk._put

    def blocked(*args):
        entered.set()
        assert release.wait(10)
        return write(*args)

    monkeypatch.setattr(store._disk, "_put", blocked)
    return entered, release


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int8])
def test_offload_milestones_and_reopen(tmp_path, dtype):
    source = torch.arange(5001, dtype=torch.float32).to(dtype)
    with RamFirstStore(tmp_path, ram_bytes=65536, chunk_bytes=4096) as store:
        handle = store.offload_async(key(), source)
        assert handle.released.result(timeout=5) is None
        handle.durable.result(timeout=5)
        assert torch.equal(source, store.get(key()))
        assert store.stats()["ram_peak_reserved_bytes"] <= store.ram_budget
        assert not handle.released.cancel() and not handle.durable.cancel()
        with pytest.raises(FileExistsError):
            store.offload_async(key(), source)
    with RamFirstStore(tmp_path, ram_bytes=65536, chunk_bytes=4096) as store:
        assert torch.equal(source, store.get(key()))
        assert store.prefetch(key()).result(timeout=5)
        assert torch.equal(source, store.get(key()))


def test_release_and_hot_reads_do_not_wait_for_disk(tmp_path, monkeypatch):
    with RamFirstStore(tmp_path, ram_bytes=65536, chunk_bytes=4096) as store:
        entered, release = barrier_writer(store, monkeypatch)
        try:
            source = torch.arange(5001)
            reference = source.clone()
            ref = weakref.ref(source)
            handle = store.offload_async(key(), source)
            handle.released.result(timeout=5)
            assert entered.wait(5)
            assert not handle.durable.done()
            source.zero_()  # released means it is safe to mutate/reuse input
            del source
            gc.collect()
            assert ref() is None, "store retained the engine's source after release"
            assert torch.equal(reference, store.get_async(key()).result(timeout=5))
            assert store.contains(key())
            assert store.stats()["pending_archives"] == 1
        finally:
            release.set()
        handle.durable.result(timeout=5)


def test_pending_count_and_dirty_ram_backpressure(tmp_path, monkeypatch):
    with RamFirstStore(tmp_path, ram_bytes=8192, max_pending=1, chunk_bytes=4096) as store:
        entered, release = barrier_writer(store, monkeypatch)
        try:
            handle = store.offload_async(key(), torch.ones(100))
            handle.released.result(timeout=5)
            assert entered.wait(5)
            with pytest.raises(CacheBusy, match="max_pending"):
                store.offload_async(key(1), torch.zeros(100))
            with pytest.raises(CacheBusy):
                store.discard(key())
            assert store.stats()["ram_reserved_bytes"] == 8192
        finally:
            release.set()
        handle.durable.result(timeout=5)
        store.offload_async(key(1), torch.zeros(100)).durable.result(timeout=5)
        assert store.stats()["ram_entries"] == 1
        assert torch.equal(store.get(key()), torch.ones(100))  # cold disk fallback


def test_budget_rejection_does_not_retain_or_evict_dirty_input(tmp_path, monkeypatch):
    with RamFirstStore(tmp_path, ram_bytes=8192, max_pending=3, chunk_bytes=4096) as store:
        entered, release = barrier_writer(store, monkeypatch)
        try:
            handle = store.offload_async(key(), torch.ones(100))
            handle.released.result(timeout=5)
            assert entered.wait(5)
            with pytest.raises(CacheBusy, match="dirty"):
                store.offload_async(key(1), torch.ones(100))
            assert store.stats()["pending_archives"] == 1
            assert torch.equal(store.get(key()), torch.ones(100))
        finally:
            release.set()


def test_failure_is_retained_and_retryable(tmp_path, monkeypatch):
    with RamFirstStore(tmp_path, ram_bytes=8192, chunk_bytes=4096) as store:
        original = store._disk._put

        def fail(*_args):
            raise OSError("injected disk full")

        monkeypatch.setattr(store._disk, "_put", fail)
        handle = store.offload_async(key(), torch.ones(100))
        handle.released.result(timeout=5)
        with pytest.raises(OSError, match="disk full"):
            handle.durable.result(timeout=5)
        assert store.stats()["failed_entries"] == 1
        assert torch.equal(store.get(key()), torch.ones(100))
        with pytest.raises(CacheBusy):
            store.offload_async(key(1), torch.ones(100))
        monkeypatch.setattr(store._disk, "_put", original)
        store.retry_persist(key()).result(timeout=5)
        assert store.stats()["failed_entries"] == 0
        store.discard(key())
        assert torch.equal(store.get(key()), torch.ones(100))


def test_post_rename_failure_retry_uses_bounded_verification(tmp_path, monkeypatch):
    with RamFirstStore(tmp_path, ram_bytes=65536, chunk_bytes=4096) as store:
        original = store._disk._fsync_directory

        def fail_parent(path):
            if path == store._disk.root:
                raise OSError("parent fsync failed")
            return original(path)

        monkeypatch.setattr(store._disk, "_fsync_directory", fail_parent)
        handle = store.offload_async(key(), torch.arange(5001))
        with pytest.raises(OSError):
            handle.durable.result(timeout=5)
        assert (tmp_path / key()).exists()
        monkeypatch.setattr(store._disk, "_fsync_directory", original)
        store.retry_persist(key()).result(timeout=5)
        store.discard(key())
        assert torch.equal(store.get(key()), torch.arange(5001))


def test_failed_close_reports_loss_and_releases_owner_lock(tmp_path, monkeypatch):
    store = RamFirstStore(tmp_path, ram_bytes=8192, chunk_bytes=4096)

    def fail(*_args):
        raise OSError("failed")

    monkeypatch.setattr(store._disk, "_put", fail)
    handle = store.offload_async(key(), torch.ones(100))
    with pytest.raises(OSError):
        handle.durable.result(timeout=5)
    with pytest.raises(PersistenceError):
        store.close()
    store.close()
    with TieredStore(tmp_path) as reopened:
        assert not reopened.contains(key())


def test_explicit_discard_acknowledges_failed_data_loss(tmp_path, monkeypatch):
    with RamFirstStore(tmp_path, ram_bytes=8192, chunk_bytes=4096) as store:

        def fail(*_args):
            raise OSError("failed")

        monkeypatch.setattr(store._disk, "_put", fail)
        handle = store.offload_async(key(), torch.ones(100))
        with pytest.raises(OSError):
            handle.durable.result(timeout=5)
        store.discard(key())
        assert not store.contains(key())


def test_copy_allocation_failure_releases_reservation(tmp_path, monkeypatch):
    import kv_tier.writeback as module

    with RamFirstStore(tmp_path, ram_bytes=8192, chunk_bytes=4096) as store:

        def fail(*_args):
            raise MemoryError("allocation failed")

        monkeypatch.setattr(module, "_aligned_buffer", fail)
        handle = store.offload_async(key(), torch.ones(100))
        for future in [handle.released, handle.durable]:
            with pytest.raises(MemoryError):
                future.result(timeout=5)
        assert store.stats()["ram_reserved_bytes"] == 0
        assert store.stats()["pending_archives"] == 0


def test_foreground_read_priority_is_nonpreemptive_and_fifo_on_ties():
    entered, release = threading.Event(), threading.Event()
    worker, order = PriorityWorker(), []

    def blocking():
        entered.set()
        assert release.wait(5)
        order.append("inflight")

    try:
        first = worker.submit(blocking, priority=10)
        assert entered.wait(5)
        write = worker.submit(order.append, "archive", priority=10)
        read1 = worker.submit(order.append, "read1", priority=0)
        read2 = worker.submit(order.append, "read2", priority=0)
        release.set()
        for future in [first, write, read1, read2]:
            future.result(timeout=5)
        assert order == ["inflight", "read1", "read2", "archive"]
    finally:
        release.set()
        worker.shutdown()


def test_read_queue_and_active_read_reservations_are_bounded(tmp_path, monkeypatch):
    with RamFirstStore(tmp_path, ram_bytes=8192, max_reads=1, chunk_bytes=4096) as store:
        store.offload_async(key(), torch.ones(100)).durable.result(timeout=5)
        entered, release = threading.Event(), threading.Event()
        restore = store._restore_hot

        def blocked(*args):
            entered.set()
            assert release.wait(5)
            restore(*args)

        monkeypatch.setattr(store, "_restore_hot", blocked)
        try:
            read = store.get_async(key())
            assert entered.wait(5)
            with pytest.raises(CacheBusy, match="max_reads"):
                store.get_async(key())
            with pytest.raises(CacheBusy):
                store.offload_async(key(1), torch.ones(100))
            assert not read.cancel()
        finally:
            release.set()
        assert torch.equal(read.result(timeout=5), torch.ones(100))
        assert store.stats()["pending_reads"] == 0


def test_concurrent_close_drains_accepted_archives(tmp_path):
    store = RamFirstStore(tmp_path, ram_bytes=65536, chunk_bytes=4096)
    handles = [store.offload_async(key(i), torch.full((100,), i)) for i in range(2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(store.close) for _ in range(2)]
        for future in futures:
            future.result(timeout=5)
    assert all(handle.durable.done() for handle in handles)
    with pytest.raises(RuntimeError, match="closed"):
        store.get(key())
    with TieredStore(tmp_path) as reopened:
        for i in range(2):
            assert torch.equal(reopened.get(key(i)), torch.full((100,), i))


def test_prefetch_budget_declines_without_allocation(tmp_path):
    with TieredStore(tmp_path, chunk_bytes=4096) as disk:
        disk.put(key(), torch.ones(3000))
    with RamFirstStore(tmp_path, ram_bytes=8192, chunk_bytes=4096) as store:
        assert not store.prefetch(key()).result(timeout=5)
        assert store.stats()["ram_reserved_bytes"] == 0
        assert torch.equal(store.get(key()), torch.ones(3000))


def test_invalid_input_and_output_do_not_poison_queues(tmp_path):
    with RamFirstStore(tmp_path, ram_bytes=65536, chunk_bytes=4096) as store:
        for tensor in [torch.empty(0), torch.zeros(3, 3).T, torch.ones(2, dtype=torch.bool)]:
            with pytest.raises(ValueError):
                store.offload_async(key(), tensor)
        store.offload_async(key(), torch.ones(100)).durable.result(timeout=5)
        with pytest.raises(ValueError, match="destination"):
            store.get(key(), out=torch.empty(1))
        assert store.stats()["pending_reads"] == 0
        assert torch.equal(store.get(key()), torch.ones(100))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_independent_lanes_and_caller_dependencies(tmp_path, monkeypatch):
    with RamFirstStore(tmp_path, ram_bytes=2**20, device="cuda", chunk_bytes=4096) as store:
        entered, release = barrier_writer(store, monkeypatch)
        try:
            producer = torch.cuda.Stream()
            with torch.cuda.stream(producer):
                source = torch.randn(12001, device="cuda", dtype=torch.float16)
                expected = source.clone()
                handle = store.offload_async(key(), source)
            handle.released.result(timeout=5)
            assert entered.wait(5)
            source.zero_()
            with torch.cuda.stream(producer), torch.inference_mode():
                out = torch.ones_like(expected)
                actual = store.get_async(key(), out=out)
            assert torch.equal(actual.result(timeout=5), expected)
            assert not handle.durable.done()
            assert store.stats()["pinned"]
        finally:
            release.set()
        handle.durable.result(timeout=5)
        store.discard(key())
        assert torch.equal(store.get(key()), expected)


def test_eviction_drops_host_even_when_handles_or_callbacks_survive(tmp_path, monkeypatch):
    with RamFirstStore(tmp_path, ram_bytes=8192, chunk_bytes=4096) as store:
        entered, release = barrier_writer(store, monkeypatch)
        callback_entered, callback_release = threading.Event(), threading.Event()

        def slow_callback(_future):
            callback_entered.set()
            assert callback_release.wait(5)

        try:
            first = store.offload_async(key(), torch.ones(100))
            first.released.result(timeout=5)
            assert entered.wait(5)
            host_ref = weakref.ref(store._entries[key()].host._base)
            first.durable.add_done_callback(slow_callback)
            release.set()
            assert callback_entered.wait(5)
            second = store.offload_async(key(1), torch.zeros(100))
            second.released.result(timeout=5)
            gc.collect()
            assert host_ref() is None, "evicted storage survived through a completed handle"
        finally:
            release.set()
            callback_release.set()
        second.durable.result(timeout=5)


def test_priority_worker_gives_old_archive_a_turn_under_read_flood():
    entered, release = threading.Event(), threading.Event()
    worker, order = PriorityWorker(), []

    def blocked():
        entered.set()
        assert release.wait(5)

    try:
        worker.submit(blocked)
        assert entered.wait(5)
        archive = worker.submit(order.append, "archive", priority=10)
        reads = [worker.submit(order.append, i, priority=0) for i in range(20)]
        release.set()
        archive.result(timeout=5)
        for future in reads:
            future.result(timeout=5)
        assert order.index("archive") <= 8
    finally:
        release.set()
        worker.shutdown()


def test_secondary_stream_error_does_not_strand_notifications(tmp_path, monkeypatch):
    import kv_tier.writeback as module

    class BrokenStream:
        def synchronize(self):
            raise RuntimeError("device context lost")

    with RamFirstStore(tmp_path, ram_bytes=8192, chunk_bytes=4096) as store:

        def fail(*_args):
            raise MemoryError("allocation failed")

        store._offload_stream = BrokenStream()
        monkeypatch.setattr(module, "_aligned_buffer", fail)
        handle = store.offload_async(key(), torch.ones(100))
        for future in [handle.released, handle.durable]:
            with pytest.raises(MemoryError):
                future.result(timeout=5)
        assert store.stats()["pending_archives"] == 0


def test_close_drops_host_with_surviving_completion_handles(tmp_path):
    store = RamFirstStore(tmp_path, ram_bytes=8192, chunk_bytes=4096)
    handle = store.offload_async(key(), torch.ones(100))
    handle.durable.result(timeout=5)
    # A read installs a callback on the released Future and returns a public
    # tensor Future. Neither handle should keep host storage alive after close.
    read = store.get_async(key())
    read.result(timeout=5)
    host_ref = weakref.ref(store._entries[key()].host._base)
    store.close()
    gc.collect()
    assert host_ref() is None
    assert handle.released.done() and read.done()


def test_cold_foreground_read_jumps_a_queued_archive(tmp_path, monkeypatch):
    with RamFirstStore(tmp_path, ram_bytes=65536, max_pending=3, chunk_bytes=4096) as store:
        store.offload_async(key(), torch.ones(100)).durable.result(timeout=5)
        store.discard(key())
        entered, release = barrier_writer(store, monkeypatch)
        try:
            a = store.offload_async(key(1), torch.ones(100))
            a.released.result(timeout=5)
            assert entered.wait(5)
            b = store.offload_async(key(2), torch.ones(100))
            b.released.result(timeout=5)
            order = []
            original_put, original_get = store._disk._put, store._disk._get

            def track_put(*args):
                order.append("archive")
                return original_put(*args)

            def track_get(*args):
                order.append("read")
                return original_get(*args)

            monkeypatch.setattr(store._disk, "_put", track_put)
            monkeypatch.setattr(store._disk, "_get", track_get)
            read = store.get_async(key())
        finally:
            release.set()
        assert torch.equal(read.result(timeout=5), torch.ones(100))
        b.durable.result(timeout=5)
        assert order == ["read", "archive"]


def test_prefetch_corruption_and_unknown_reads_release_reservations(tmp_path):
    from kv_tier import CacheCorruption

    with TieredStore(tmp_path, chunk_bytes=4096) as disk:
        disk.put(key(), torch.ones(100))
    with (tmp_path / key() / "data.bin").open("r+b") as output:
        output.write(b"bad!")
    with RamFirstStore(tmp_path, ram_bytes=8192, chunk_bytes=4096) as store:
        with pytest.raises(CacheCorruption):
            store.prefetch(key()).result(timeout=5)
        with pytest.raises(KeyError):
            store.get(key(99))
        stats = store.stats()
        assert stats["pending_reads"] == stats["ram_reserved_bytes"] == 0


def test_ram_copy_survives_disk_quota_eviction(tmp_path):
    with RamFirstStore(tmp_path, ram_bytes=16384, disk_bytes=4096, chunk_bytes=4096) as store:
        for i in range(2):
            store.offload_async(key(i), torch.full((100,), i)).durable.result(timeout=5)
        assert not (tmp_path / key()).exists()
        assert store.contains(key())
        assert torch.equal(store.get(key()), torch.zeros(100, dtype=torch.int64))
        with pytest.raises(FileExistsError):
            store.offload_async(key(), torch.ones(100))
        store.discard(key())
        assert not store.contains(key())


def test_failed_cold_read_handle_does_not_retain_allocated_output(tmp_path, monkeypatch):
    from kv_tier import CacheCorruption

    with TieredStore(tmp_path, chunk_bytes=4096) as disk:
        disk.put(key(), torch.ones(100))
    with (tmp_path / key() / "data.bin").open("r+b") as output:
        output.write(b"bad!")
    with RamFirstStore(tmp_path, ram_bytes=8192, chunk_bytes=4096) as store:
        empty, refs = torch.empty, []

        def tracked(shape, *args, **kwargs):
            result = empty(shape, *args, **kwargs)
            if shape == (100,):
                refs.append(weakref.ref(result))
            return result

        monkeypatch.setattr(torch, "empty", tracked)
        future = store.get_async(key())
        with pytest.raises(CacheCorruption):
            future.result(timeout=5)
        gc.collect()
        assert len(refs) == 1 and refs[0]() is None
