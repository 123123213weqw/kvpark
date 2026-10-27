import hashlib

import pytest
import torch

from kvpark import CacheBusy, CacheCorruption, RamFirstStore, TieredStore
from kvpark.metrics import EDGES_MS, Metrics, ratio


def key(i=0):
    return hashlib.sha256(f"metrics-{i}".encode()).hexdigest()


# ---------------------------------------------------------------- unit tests


def test_counters_accumulate_and_reset():
    metrics = Metrics()
    assert metrics.get("absent") == 0
    metrics.add("hits")
    metrics.add("hits", 4)
    assert metrics.get("hits") == 5
    metrics.reset()
    assert metrics.get("hits") == 0
    assert metrics.snapshot() == {"counters": {}, "latency_ms": {}}


def test_empty_latency_reports_none_not_zero():
    # An idle store must not look like a measured 0 ms operation.
    snap = Metrics().snapshot()
    assert snap["latency_ms"] == {}
    metrics = Metrics()
    metrics.observe("noop", 0)
    series = metrics.latencies()["noop"]
    assert series["count"] == 1 and series["min_ms"] == 0.0


@pytest.mark.parametrize("samples", [[1_000_000], [1_000_000] * 10 + [10_000_000] * 90])
def test_quantiles_are_upper_bounds_never_underestimates(samples):
    metrics = Metrics()
    for value in samples:
        metrics.observe("op", value)
    series = metrics.latencies()["op"]
    ordered = sorted(samples)
    for name, target in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
        exact = ordered[min(len(ordered) - 1, int(len(ordered) * target))]
        assert series[name] >= exact / 1e6
    assert series["count"] == len(samples)
    assert series["min_ms"] <= series["mean_ms"] <= series["max_ms"]


def test_latency_overflow_bucket_reports_exact_maximum():
    metrics = Metrics()
    metrics.observe("op", int(200_000e6))  # 200 s, beyond every edge
    series = metrics.latencies()["op"]
    assert series["max_ms"] == pytest.approx(200_000)
    assert series["p50"] == pytest.approx(200_000)
    assert EDGES_MS[-1] < 200_000


def test_snapshot_is_a_copy():
    metrics = Metrics()
    metrics.add("hits")
    snap = metrics.snapshot()
    snap["counters"]["hits"] = 999
    assert metrics.get("hits") == 1


def test_ratio_refuses_to_invent_a_number():
    assert ratio(0, 0) is None
    assert ratio(3, 4) == 0.75


# ------------------------------------------------------------- store metrics


def test_store_counts_ram_and_disk_hits(tmp_path):
    with TieredStore(tmp_path, chunk_bytes=4096, ram_bytes=65536) as store:
        store.put(key(), torch.arange(4096, dtype=torch.uint8))
        store.get(key())
        store.get(key())
        assert store.prefetch(key()).result()
        store.get(key())
        counters = store.stats()["metrics"]["counters"]
        assert counters["put_ok"] == 1
        assert counters["put_bytes"] == 4096
        assert counters["get_disk_hits"] == 2
        assert counters["get_ram_hits"] == 1
        assert counters["get_ok"] == 3
        assert counters["prefetch_ok"] == 1
        latency = store.stats()["metrics"]["latency_ms"]
        assert latency["get_ram_ms"]["count"] == 1
        assert latency["get_disk_ms"]["count"] == 2
        assert latency["get_ms"]["count"] == 3
        assert latency["put_ms"]["count"] == 1
        hit_rate = store.stats()["hit_rate"]
        assert hit_rate["get"] == 1.0
        assert hit_rate["ram_share_of_hits"] == pytest.approx(1 / 3)


def test_store_counts_misses_and_contains(tmp_path):
    with TieredStore(tmp_path, chunk_bytes=4096, ram_bytes=0) as store:
        store.put(key(), torch.zeros(4096, dtype=torch.uint8))
        assert store.contains(key()) is True
        assert store.contains(key(7)) is False
        with pytest.raises(KeyError):
            store.get(key(7))
        counters = store.stats()["metrics"]["counters"]
        assert counters["get_misses"] == 1
        assert counters["contains_calls"] == 2
        assert counters["contains_hits"] == 1
        assert counters["contains_misses"] == 1
        assert store.stats()["hit_rate"]["contains"] == 0.5


def test_store_counts_immutable_rejection(tmp_path):
    with TieredStore(tmp_path, chunk_bytes=4096, ram_bytes=0) as store:
        store.put(key(), torch.zeros(4096, dtype=torch.uint8))
        with pytest.raises(FileExistsError):
            store.put(key(), torch.zeros(4096, dtype=torch.uint8))
        counters = store.stats()["metrics"]["counters"]
        assert counters["put_calls"] == 2
        assert counters["put_ok"] == 1
        assert counters["put_rejected_immutable"] == 1
        assert counters["put_errors"] == 0


def test_store_counts_checksum_failure(tmp_path):
    payload = torch.zeros(8192, dtype=torch.uint8)
    with TieredStore(tmp_path, chunk_bytes=4096, ram_bytes=0) as store:
        store.put(key(), payload)
    blob = bytearray((tmp_path / key() / "data.bin").read_bytes())
    blob[0] ^= 0xFF
    (tmp_path / key() / "data.bin").write_bytes(bytes(blob))
    with TieredStore(tmp_path, chunk_bytes=4096, ram_bytes=0) as store:
        with pytest.raises(CacheCorruption):
            store.get(key())
        counters = store.stats()["metrics"]["counters"]
        assert counters["checksum_failures"] == 1
        assert counters["get_corrupt"] == 1
        assert counters["get_ok"] == 0


def test_store_counts_disk_eviction(tmp_path):
    with TieredStore(tmp_path, chunk_bytes=4096, ram_bytes=0, disk_bytes=4096) as store:
        store.put(key(1), torch.ones(4096, dtype=torch.uint8))
        store.put(key(2), torch.ones(4096, dtype=torch.uint8))
        counters = store.stats()["metrics"]["counters"]
        assert counters["evictions_disk"] == 1
        assert counters["evicted_disk_bytes"] == 4096


def test_store_counts_declined_prefetch(tmp_path):
    with TieredStore(tmp_path, chunk_bytes=4096, ram_bytes=4096) as store:
        store.put(key(), torch.zeros(7000))
        assert store.prefetch(key()).result() is False
        assert store.stats()["metrics"]["counters"]["prefetch_rejected_large"] == 1


def test_store_stats_keeps_pre_existing_keys(tmp_path):
    legacy = {
        "ram_used_bytes", "ram_budget_bytes", "ram_entries",
        "disk_used_bytes", "disk_budget_bytes", "disk_entries",
        "staging_bytes", "pinned", "direct_io", "verify", "pipeline_reads",
    }
    with TieredStore(tmp_path, chunk_bytes=4096) as store:
        assert legacy <= set(store.stats())


# --------------------------------------------------------- writeback metrics


def test_writeback_counts_milestones_and_hot_read(tmp_path):
    with RamFirstStore(tmp_path, ram_bytes=65536, chunk_bytes=4096) as store:
        handle = store.offload_async(key(), torch.arange(4096, dtype=torch.uint8))
        handle.released.result(timeout=5)
        handle.durable.result(timeout=5)
        assert torch.equal(store.get(key()), torch.arange(4096, dtype=torch.uint8))
        stats = store.stats()
        counters = stats["metrics"]["counters"]
        assert counters["offload_accepted"] == 1
        assert counters["released_ok"] == 1
        assert counters["persist_ok"] == 1
        assert counters["read_hot"] == 1
        assert counters["read_ok"] == 1
        latency = stats["metrics"]["latency_ms"]
        assert latency["released_ms"]["count"] == 1
        assert latency["durable_ms"]["count"] == 1
        # Durability cannot be reported before the input is reusable.
        assert latency["durable_ms"]["mean_ms"] >= latency["released_ms"]["mean_ms"]
        assert stats["hit_rate"]["read"] == 1.0
        assert stats["hit_rate"]["persist"] == 1.0


def test_writeback_counts_disk_read_after_discard(tmp_path):
    with RamFirstStore(tmp_path, ram_bytes=65536, chunk_bytes=4096) as store:
        handle = store.offload_async(key(), torch.ones(1024, dtype=torch.uint8))
        handle.durable.result(timeout=5)
        store.discard(key())
        assert torch.equal(store.get(key()), torch.ones(1024, dtype=torch.uint8))
        counters = store.stats()["metrics"]["counters"]
        assert counters["discard_calls"] == 1
        assert counters["read_disk_requests"] == 1
        assert counters["read_hot"] == 0
        assert store.stats()["hit_rate"]["read"] == 0.0


def test_writeback_counts_oversize_and_immutable_rejections(tmp_path):
    with RamFirstStore(tmp_path, ram_bytes=8192, chunk_bytes=4096) as store:
        with pytest.raises(CacheBusy):
            store.offload_async(key(), torch.zeros(1 << 20, dtype=torch.uint8))
        handle = store.offload_async(key(1), torch.ones(100))
        handle.durable.result(timeout=5)
        with pytest.raises(FileExistsError):
            store.offload_async(key(1), torch.ones(100))
        counters = store.stats()["metrics"]["counters"]
        assert counters["offload_rejected_oversize"] == 1
        assert counters["offload_rejected_immutable"] == 1
        assert counters["offload_accepted"] == 1
        assert counters["offload_calls"] == 3


def test_writeback_counts_persist_failure_and_retry(tmp_path, monkeypatch):
    with RamFirstStore(tmp_path, ram_bytes=8192, chunk_bytes=4096) as store:
        original = store._disk._put

        def fail(*_args):
            raise OSError("injected disk full")

        monkeypatch.setattr(store._disk, "_put", fail)
        handle = store.offload_async(key(), torch.ones(100))
        handle.released.result(timeout=5)
        with pytest.raises(OSError):
            handle.durable.result(timeout=5)
        counters = store.stats()["metrics"]["counters"]
        assert counters["persist_errors"] == 1
        assert counters["persist_ok"] == 0
        assert store.stats()["hit_rate"]["persist"] == 0.0

        monkeypatch.setattr(store._disk, "_put", original)
        store.retry_persist(key()).result(timeout=5)
        counters = store.stats()["metrics"]["counters"]
        assert counters["retry_calls"] == 1
        assert counters["retry_ok"] == 1
        assert counters["persist_ok"] == 1
        assert store.stats()["metrics"]["latency_ms"]["retry_ms"]["count"] == 1


def test_writeback_counts_contains(tmp_path):
    with RamFirstStore(tmp_path, ram_bytes=65536, chunk_bytes=4096) as store:
        handle = store.offload_async(key(), torch.ones(1024, dtype=torch.uint8))
        handle.durable.result(timeout=5)
        assert store.contains(key()) is True
        assert store.contains(key(3)) is False
        counters = store.stats()["metrics"]["counters"]
        assert counters["contains_calls"] == 2
        assert counters["contains_hits"] == 1
        assert counters["contains_misses"] == 1


def test_writeback_stats_keeps_pre_existing_keys(tmp_path):
    legacy = {
        "ram_reserved_bytes", "ram_budget_bytes", "ram_peak_reserved_bytes",
        "ram_entries", "dirty_bytes", "failed_entries", "pending_archives",
        "max_pending", "pending_reads", "max_reads", "disk_entries", "pinned",
    }
    with RamFirstStore(tmp_path, ram_bytes=65536, chunk_bytes=4096) as store:
        assert legacy <= set(store.stats())


def test_nested_store_metrics_are_separate_from_session_metrics(tmp_path):
    # The disk layer underneath RamFirstStore must keep its own counters so a
    # persisted archive is visible as disk work as well as session work.
    with RamFirstStore(tmp_path, ram_bytes=65536, chunk_bytes=4096) as store:
        handle = store.offload_async(key(), torch.ones(1024, dtype=torch.uint8))
        handle.durable.result(timeout=5)
        disk_counters = store._disk.metrics.counters()
        assert disk_counters["put_ok"] == 1
        assert store.stats()["metrics"]["counters"]["persist_ok"] == 1
