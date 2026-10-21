"""Cache identity must bind the model, execution semantics, and complete prefix."""

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from typing import Any


def cache_key(
    model_signature: Mapping[str, Any],
    token_ids: Sequence[int],
    *,
    tensor_name: str = "kv",
) -> str:
    """Hash a caller-specified semantic identity and the *entire* causal prefix.

    Include immutable model/tokenizer revisions, adapters, RoPE configuration,
    cache dtype/layout, and tenant identity in model_signature. A model name
    alone is insufficient. No prompt text or token IDs are written to disk.
    """
    if not model_signature.get("model") or not model_signature.get("revision"):
        raise ValueError("model_signature requires nonempty model and revision")
    identity = json.dumps(
        {"format": 1, "model": dict(model_signature), "tensor": tensor_name},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    digest = hashlib.sha256(struct.pack("<Q", len(identity)) + identity)
    digest.update(struct.pack("<Q", len(token_ids)))
    for token in token_ids:
        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < 2**32:
            raise ValueError("token IDs must be uint32 integers")
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()
