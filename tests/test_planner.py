import pytest

from kvpark.planner import choose_restore, kv_bytes


def test_one_gib_example():
    assert kv_bytes(layers=32, kv_heads=8, head_dim=128, tokens=8192) == 1024**3


def test_cold_hdd_should_recompute_short_prefix():
    decision = choose_restore(nbytes=1024**3, link_gbps=12, disk_gbps=0.18, recompute_ms=500)
    assert decision.action == "recompute"
    assert decision.estimated_restore_ms > 5000


def test_ram_prefetch_changes_admission():
    decision = choose_restore(
        nbytes=1024**3, link_gbps=12, disk_gbps=0.18, recompute_ms=500, ram_hit=True
    )
    assert decision.action == "restore"
    assert decision.estimated_restore_ms < 100


def test_invalid_input():
    with pytest.raises(ValueError):
        kv_bytes(layers=-1, kv_heads=8, head_dim=128, tokens=8192)
