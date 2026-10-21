import hashlib
import json
import os

import pytest
import torch

from kv_tier import CacheCorruption, TieredStore

KEY = hashlib.sha256(b"fault-injection").hexdigest()


@pytest.mark.parametrize("field,value", [("nbytes", 120.0), ("version", True)])
def test_noninteger_manifest_fields_rejected(tmp_path, field, value):
    with TieredStore(tmp_path) as store:
        store.put(KEY, torch.ones(30))
    manifest = tmp_path / KEY / "manifest.json"
    data = json.loads(manifest.read_text())
    data[field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(CacheCorruption):
        TieredStore(tmp_path)


def test_truncation_after_open_is_detected(tmp_path):
    with TieredStore(tmp_path) as store:
        store.put(KEY, torch.ones(2000))
        with (tmp_path / KEY / "data.bin").open("r+b") as handle:
            handle.truncate(100)
        with pytest.raises(CacheCorruption):
            store.get(KEY)


@pytest.mark.skipif(not hasattr(os, "preadv"), reason="preadv unavailable")
def test_partial_buffered_reads_are_retried(tmp_path, monkeypatch):
    original = os.preadv

    def short_read(descriptor, buffers, offset):
        return original(descriptor, [buffers[0][:300]], offset)

    with TieredStore(tmp_path, chunk_bytes=4096) as store:
        source = torch.randn(8888, dtype=torch.float16)
        store.put(KEY, source)
        monkeypatch.setattr(os, "preadv", short_read)
        assert torch.equal(source.view(torch.uint8), store.get(KEY).view(torch.uint8))


def test_failed_commit_does_not_publish_incomplete_snapshot(tmp_path, monkeypatch):
    def broken_fsync(_descriptor):
        raise OSError("simulated full or failing device")

    with TieredStore(tmp_path, chunk_bytes=4096) as store:
        monkeypatch.setattr(os, "fsync", broken_fsync)
        with pytest.raises(OSError):
            store.put(KEY, torch.ones(100))
        assert not store.contains(KEY)
        assert not list(tmp_path.glob(".pending-*"))
