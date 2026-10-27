from types import SimpleNamespace

import pytest
import torch

from kvpark import TieredStore, cache_key
from kvpark.hf import pack_dynamic_cache, resolve_model_revision, unpack_dynamic_cache


def test_real_attention_cache_preserves_logits(tmp_path):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(17)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config).eval()
    ids = torch.randint(0, 128, (1, 16))
    with torch.inference_mode():
        original = model(ids[:, :-1], use_cache=True)
        packed = pack_dynamic_cache(original.past_key_values)
        expected = model(ids[:, -1:], past_key_values=original.past_key_values).logits
        key = cache_key({"model": "tiny-llama", "revision": "seed17"}, ids[0, :-1].tolist())
        with TieredStore(tmp_path, chunk_bytes=4096) as store:
            store.put(key, packed)
            restored = store.get(key)
        actual = model(ids[:, -1:], past_key_values=unpack_dynamic_cache(restored)).logits
    assert torch.equal(expected, actual)


def test_static_or_unknown_cache_kinds_are_rejected():
    transformers = pytest.importorskip("transformers")

    class QuantizedLikeCache(transformers.DynamicCache):
        pass

    for cache in (object(), object.__new__(transformers.StaticCache), QuantizedLikeCache()):
        with pytest.raises(TypeError, match="DynamicCache"):
            pack_dynamic_cache(cache)


def test_sliding_layers_require_a_different_adapter():
    transformers = pytest.importorskip("transformers")
    cache = transformers.DynamicCache()
    cache.layers = [SimpleNamespace(is_sliding=True)]
    with pytest.raises(ValueError, match="sliding"):
        pack_dynamic_cache(cache)


def test_offline_commit_fallback_is_explicit_and_immutable():
    commit = "a" * 40
    assert resolve_model_revision(SimpleNamespace(_commit_hash=None), commit) == commit
    assert resolve_model_revision(SimpleNamespace(_commit_hash=commit), "main") == commit
    with pytest.raises(ValueError, match="immutable"):
        resolve_model_revision(SimpleNamespace(_commit_hash=None), "main")
    with pytest.raises(ValueError, match="differs"):
        resolve_model_revision(SimpleNamespace(_commit_hash=commit), "b" * 40)
