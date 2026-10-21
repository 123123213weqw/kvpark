import hashlib
import json
import sys

import pytest
import torch

from kv_tier import CacheCorruption, TieredStore, cache_key


def key(index=0):
    return hashlib.sha256(str(index).encode()).hexdigest()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int8])
@pytest.mark.parametrize("count", [1, 2047, 2048, 2049, 13333])
def test_exact_disk_roundtrip_and_reopen(tmp_path, dtype, count):
    source = torch.arange(count, dtype=torch.float32).to(dtype)
    with TieredStore(tmp_path, chunk_bytes=4096, ram_bytes=0) as store:
        store.put(key(), source)
        assert torch.equal(source.view(torch.uint8), store.get(key()).view(torch.uint8))
    with TieredStore(tmp_path, chunk_bytes=8192) as store:
        assert torch.equal(source.view(torch.uint8), store.get(key()).view(torch.uint8))
        assert store.contains(key())


def test_pinned_or_plain_ram_admission_is_bounded_and_lru(tmp_path):
    with TieredStore(tmp_path, chunk_bytes=4096, ram_bytes=3 * 4096) as store:
        for i in range(3):
            store.put(key(i), torch.full((4096,), i, dtype=torch.uint8))
            assert store.prefetch(key(i)).result()
            assert store.stats()["ram_used_bytes"] <= store.ram_budget
        assert store.stats()["ram_entries"] == 1
        assert torch.equal(store.get(key(2)), torch.full((4096,), 2, dtype=torch.uint8))
        store.clear_ram()
        assert store.stats()["ram_used_bytes"] == 0


def test_large_prefetch_declines_without_allocating(tmp_path):
    with TieredStore(tmp_path, chunk_bytes=4096, ram_bytes=4096) as store:
        store.put(key(), torch.zeros(7000))
        assert store.prefetch(key()).result() is False
        assert store.stats()["ram_entries"] == 0


def test_disk_quota_eviction(tmp_path):
    with TieredStore(tmp_path, disk_bytes=8192, chunk_bytes=4096) as store:
        for i in range(3):
            store.put(key(i), torch.full((4096,), i, dtype=torch.uint8))
        assert not store.contains(key(0))
        assert store.contains(key(1)) and store.contains(key(2))
        assert store.stats()["disk_used_bytes"] == 8192
        with pytest.raises(ValueError, match="budget"):
            store.put(key(9), torch.zeros(9000, dtype=torch.uint8))


def test_payload_corruption_detected(tmp_path):
    with TieredStore(tmp_path, chunk_bytes=4096) as store:
        store.put(key(), torch.arange(9000, dtype=torch.float32))
        path = tmp_path / key() / "data.bin"
        with path.open("r+b") as handle:
            handle.seek(5000)
            handle.write(b"\xff")
        with pytest.raises(CacheCorruption, match="checksum"):
            store.get(key())


def test_truncated_payload_and_hostile_metadata_rejected(tmp_path):
    with TieredStore(tmp_path, chunk_bytes=4096) as store:
        store.put(key(), torch.zeros(30))
    path = tmp_path / key() / "manifest.json"
    meta = json.loads(path.read_text())
    meta["shape"] = [2**60]
    path.write_text(json.dumps(meta))
    with pytest.raises(CacheCorruption):
        TieredStore(tmp_path)


def test_pending_files_are_cleaned_after_simulated_crash(tmp_path):
    pending = tmp_path / ".pending-crash"
    pending.mkdir()
    (pending / "data.bin").write_bytes(b"partial")
    with TieredStore(tmp_path) as store:
        assert store.stats()["disk_entries"] == 0
        assert not pending.exists()


def test_immutable_keys_and_invalid_destinations(tmp_path):
    with TieredStore(tmp_path, chunk_bytes=4096) as store:
        store.put(key(), torch.ones(10))
        with pytest.raises(FileExistsError):
            store.put(key(), torch.zeros(10))
        for out in [torch.empty(11), torch.empty(10, dtype=torch.float16)]:
            with pytest.raises(ValueError, match="destination"):
                store.get(key(), out=out)
        with pytest.raises(KeyError):
            store.get(key(99))
        with pytest.raises(ValueError):
            store.get("../../etc/passwd")


def test_noncontiguous_and_empty_rejected(tmp_path):
    with TieredStore(tmp_path) as store:
        with pytest.raises(ValueError, match="contiguous"):
            store.put(key(), torch.ones(4, 4).T)
        with pytest.raises(ValueError, match="nonempty"):
            store.put(key(), torch.empty(0))


def test_exclusive_root_lock_and_close(tmp_path):
    with TieredStore(tmp_path) as store:
        with pytest.raises(RuntimeError, match="lock"):
            TieredStore(tmp_path)
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.stats()


def test_semantic_key_is_sensitive_to_all_prefix_and_model_fields():
    signature = {"model": "example", "revision": "immutable-sha", "rope": {"theta": 10000}}
    a = cache_key(signature, [1, 2, 3])
    assert a == cache_key(dict(reversed(list(signature.items()))), [1, 2, 3])
    assert a != cache_key(signature, [9, 2, 3])
    assert a != cache_key(signature, [1, 2, 3], tensor_name="layer-1")
    assert a != cache_key({**signature, "adapter": "lora-2"}, [1, 2, 3])
    with pytest.raises(ValueError):
        cache_key({"model": "example"}, [1])
    with pytest.raises(ValueError):
        cache_key(signature, [-1])


@pytest.mark.linux
@pytest.mark.skipif(sys.platform != "linux", reason="O_DIRECT requires Linux")
def test_direct_io_unaligned_tail(tmp_path):
    source = torch.randn(12345, dtype=torch.float16)
    with TieredStore(tmp_path, chunk_bytes=8192, direct_io=True) as store:
        store.put(key(), source)
        assert torch.equal(store.get(key()).view(torch.uint8), source.view(torch.uint8))
        assert store.prefetch(key()).result()
        assert torch.equal(store.get(key()).view(torch.uint8), source.view(torch.uint8))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("slots", [1, 2, 3])
def test_cuda_ring_buffers_and_stream_dependencies(tmp_path, slots):
    with TieredStore(
        tmp_path, device="cuda", chunk_bytes=4096, staging_slots=slots, ram_bytes=2**20
    ) as store:
        producer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            source = torch.randn(2, 1, 4, 257, 32, device="cuda", dtype=torch.float16)
            put = store.put_async(key(), source)
        put.result()
        out = torch.empty_like(source)
        actual = store.get_async(key(), out=out).result()
        assert actual.data_ptr() == out.data_ptr()
        assert torch.equal(actual.view(torch.uint8), source.view(torch.uint8))
        assert store.prefetch(key()).result()
        assert store.stats()["pinned"]
        assert torch.equal(store.get(key()).view(torch.uint8), source.view(torch.uint8))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_error_path_does_not_recycle_inflight_buffers(tmp_path):
    with TieredStore(tmp_path, device="cuda", chunk_bytes=4096) as store:
        source = torch.randn(12345, device="cuda")
        store.put(key(), source)
        with (tmp_path / key() / "data.bin").open("r+b") as output:
            output.seek(8192)
            output.write(b"\x03\x04\x05\x06")
        with pytest.raises(CacheCorruption):
            store.get(key())
        store.put(key(1), source)
        assert torch.equal(store.get(key(1)), source)
