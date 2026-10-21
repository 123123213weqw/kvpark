"""Explicit adapter for uniform full-attention Hugging Face DynamicCache.

This is a snapshot/resume adapter, not a vLLM connector, hybrid-cache adapter,
or per-token disk paging implementation. Packing allocates an extra GPU tensor.
"""

import torch


def pack_dynamic_cache(cache):
    if hasattr(cache, "to_legacy_cache"):
        layers = cache.to_legacy_cache()
    elif hasattr(cache, "layers"):
        layers = tuple((layer.keys, layer.values) for layer in cache.layers)
    else:
        raise TypeError("expected a uniform DynamicCache with accessible K/V tensors")
    if not layers:
        raise ValueError("cache is empty")
    reference = layers[0][0]
    for pair in layers:
        if len(pair) != 2 or any(
            tensor.shape != reference.shape
            or tensor.dtype != reference.dtype
            or tensor.device != reference.device
            for tensor in pair
        ):
            raise ValueError("only uniform full-attention K/V shapes and dtypes are supported")
    return torch.stack([torch.stack(pair, dim=0) for pair in layers], dim=0).contiguous()


def unpack_dynamic_cache(packed):
    from transformers import DynamicCache

    if packed.ndim != 6 or packed.shape[1] != 2:
        raise ValueError("expected [layers, 2, batch, kv_heads, tokens, head_dim]")
    cache = DynamicCache()
    for index, layer in enumerate(packed):
        cache.update(layer[0], layer[1], index)
    return cache
