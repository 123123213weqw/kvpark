import hashlib
import os
import threading
import zlib

import pytest
import torch

from kvpark import CacheCorruption, TieredStore

KEY = hashlib.sha256(b"read-checksum-pipeline").hexdigest()


@pytest.mark.skipif(not hasattr(os, "preadv"), reason="preadv unavailable")
def test_next_disk_read_overlaps_current_checksum(tmp_path, monkeypatch):
    next_read = threading.Event()
    read = os.preadv
    crc = zlib.crc32
    first_crc = True

    def track_read(descriptor, buffers, offset):
        amount = read(descriptor, buffers, offset)
        if offset == 4096:
            next_read.set()
        return amount

    def verify_overlap(data):
        nonlocal first_crc
        if first_crc:
            first_crc = False
            assert next_read.wait(5), "next I/O did not start during the current checksum"
        return crc(data)

    with TieredStore(tmp_path, chunk_bytes=4096, pipeline_reads=True) as store:
        source = torch.randn(4096)
        store.put(KEY, source)
        monkeypatch.setattr(os, "preadv", track_read)
        monkeypatch.setattr(zlib, "crc32", verify_overlap)
        assert torch.equal(source, store.get(KEY))


@pytest.mark.skipif(not hasattr(os, "preadv"), reason="preadv unavailable")
def test_error_drains_a_pending_disk_read_before_recycling(tmp_path, monkeypatch):
    next_started = threading.Event()
    release = threading.Event()
    read = os.preadv
    crc = zlib.crc32

    def delayed_read(descriptor, buffers, offset):
        if offset == 4096:
            next_started.set()
            assert release.wait(5)
        return read(descriptor, buffers, offset)

    def corrupt(_data):
        assert next_started.wait(5)
        return 0

    with TieredStore(tmp_path, chunk_bytes=4096, pipeline_reads=True) as store:
        source = torch.randn(4096)
        store.put(KEY, source)
        monkeypatch.setattr(os, "preadv", delayed_read)
        monkeypatch.setattr(zlib, "crc32", corrupt)
        future = store.get_async(KEY)
        try:
            assert next_started.wait(5)
            assert not future.done()
        finally:
            release.set()
        with pytest.raises(CacheCorruption):
            future.result(timeout=5)
        monkeypatch.setattr(os, "preadv", read)
        monkeypatch.setattr(zlib, "crc32", crc)
        assert torch.equal(source, store.get(KEY))


@pytest.mark.parametrize("slots", [1, 2, 3])
def test_serial_and_pipelined_read_modes_are_identical(tmp_path, slots):
    source = torch.randn(5001)
    with TieredStore(tmp_path, chunk_bytes=4096, staging_slots=slots) as store:
        store.put(KEY, source)
    for enabled in (False, True):
        with TieredStore(
            tmp_path, chunk_bytes=4096, staging_slots=slots, pipeline_reads=enabled
        ) as store:
            assert torch.equal(source, store.get(KEY))
            assert store.prefetch(KEY).result()
            assert torch.equal(source, store.get(KEY))
