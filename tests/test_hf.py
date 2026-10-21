import pytest
import torch

from kv_tier import TieredStore, cache_key
from kv_tier.hf import pack_dynamic_cache, unpack_dynamic_cache


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
        expected = model(ids[:, -1:], past_key_values=unpack_dynamic_cache(packed)).logits
        key = cache_key({"model": "tiny-llama", "revision": "seed17"}, ids[0, :-1].tolist())
        with TieredStore(tmp_path, chunk_bytes=4096) as store:
            store.put(key, packed)
            restored = store.get(key)
        actual = model(ids[:, -1:], past_key_values=unpack_dynamic_cache(restored)).logits
    assert torch.equal(expected, actual)
