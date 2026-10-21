"""Explicit adapter for uniform, full-retained-history Hugging Face DynamicCache.

This is a snapshot/resume adapter, not a vLLM connector, hybrid-cache adapter,
or per-token disk paging implementation. Packing allocates an extra GPU tensor.
"""

import re

import torch


def resolve_model_revision(config, requested):
    """Require a resolved commit or an explicit full commit in offline mode.

    Some Transformers versions omit _commit_hash when loading offline. A full
    requested commit still pins the Hub lookup. This is not weight attestation;
    verify downloaded weight digests separately, as the experiment scripts do.
    """
    resolved = getattr(config, "_commit_hash", None)
    pinned = isinstance(requested, str) and re.fullmatch(r"[0-9a-f]{40}", requested)
    if resolved:
        if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
            raise ValueError("resolved model revision is not a full immutable commit")
        if pinned and resolved != requested:
            raise ValueError("resolved model revision differs from the requested pinned commit")
        return resolved
    if pinned:
        return requested
    raise ValueError("provide a full immutable --revision when offline resolution omits the commit")


def pack_dynamic_cache(cache):
    from transformers import DynamicCache

    if type(cache) is not DynamicCache:
        raise TypeError(
            "only standard DynamicCache is supported, not static/quantized/hybrid caches"
        )
    if any(getattr(layer, "is_sliding", False) for layer in getattr(cache, "layers", [])):
        raise ValueError(
            "sliding/rolling caches need absolute-position metadata and are unsupported"
        )
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
            raise ValueError("only uniform full-history K/V shapes and dtypes are supported")
    return torch.stack([torch.stack(pair, dim=0) for pair in layers], dim=0).contiguous()


def unpack_dynamic_cache(packed):
    from transformers import DynamicCache

    if packed.ndim != 6 or packed.shape[1] != 2:
        raise ValueError("expected [layers, 2, batch, kv_heads, tokens, head_dim]")
    cache = DynamicCache()
    for index, layer in enumerate(packed):
        cache.update(layer[0], layer[1], index)
    return cache
