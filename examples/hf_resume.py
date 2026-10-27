"""Verify model logits and greedy continuation after disk/RAM cache restoration."""

import argparse
import hashlib
import inspect
import json
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from kvpark import TieredStore, cache_key
from kvpark.benchmark import hardware, measure
from kvpark.hf import pack_dynamic_cache, resolve_model_revision, unpack_dynamic_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", help="Optional pretrained model ID or local model directory")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--model-cache-dir", help="Optional Hugging Face cache directory")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--continuation-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--direct-io", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.tokens < 2 or args.continuation_tokens < 1 or args.repeats < 1:
        parser.error("tokens must be >=2; continuation and repeats must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if args.model:
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                revision=args.revision,
                torch_dtype=dtype,
                attn_implementation=args.attn_implementation,
                trust_remote_code=False,
                weights_only=True,
                cache_dir=args.model_cache_dir,
                local_files_only=args.local_files_only,
            )
            .to(device)
            .eval()
        )
        model_name = (
            args.model if "/" not in args.model or not Path(args.model).exists() else "local"
        )
        revision = resolve_model_revision(model.config, args.revision)
    else:
        config = LlamaConfig(
            vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=max(4096, args.tokens + args.continuation_tokens),
        )
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
        model_name = "tiny-random-llama-no-download"
        revision = (
            "random-seed-1234-config-"
            + hashlib.sha256(config.to_json_string().encode()).hexdigest()
        )
    signature = {
        "model": model_name,
        "revision": revision,
        "config": model.config.to_dict(),
        "dtype": str(dtype),
        "layout": "L2BHTD-v1",
        "adapter": None,
        "tenant": "public-synthetic-benchmark",
    }
    ids = torch.randint(0, model.config.vocab_size, (1, args.tokens), device=device)
    prefix, last = ids[:, :-1], ids[:, -1:]
    prefill_kwargs = (
        {"logits_to_keep": 1}
        if "logits_to_keep" in inspect.signature(model.forward).parameters
        else {}
    )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        first = model(prefix, use_cache=True, **prefill_kwargs)
        original_cache = first.past_key_values
        synchronize()
        start = time.perf_counter()
        packed = pack_dynamic_cache(first.past_key_values)
        synchronize()
        pack_ms = (time.perf_counter() - start) * 1000
        del first
        nbytes = packed.numel() * packed.element_size()
        key = cache_key(signature, prefix[0].tolist(), tensor_name="all-retained-history-layers")

        def continuation(snapshot=None, *, cache=None):
            if cache is None:
                cache = unpack_dynamic_cache(snapshot)
            current = last
            tokens = []
            first_logits = None
            for _ in range(args.continuation_tokens):
                output = model(current, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                if first_logits is None:
                    first_logits = output.logits[:, -1].detach().clone()
                current = output.logits[:, -1].argmax(-1, keepdim=True)
                tokens.append(int(current.item()))
            return tokens, first_logits

        expected_tokens, expected_logits = continuation(cache=original_cache)
        del original_cache
        report = {
            "model": model_name,
            "model_revision": revision,
            "hardware": hardware(device),
            "prompt_kind": "seeded synthetic token IDs, not private prompts",
            "prefix_tokens": args.tokens - 1,
            "cache_bytes": nbytes,
            "cache_shape": list(packed.shape),
            "pack_ms": pack_ms,
            "direct_io": args.direct_io,
            "verification": True,
            "tiny_random_model": args.model is None,
            "reference": "original model-produced cache before packing or disk storage",
            "prefill_kwargs": prefill_kwargs,
        }
        parent = Path(args.root).expanduser()
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="kvpark-hf-", dir=parent) as temporary:
            with TieredStore(
                temporary,
                device=str(device),
                ram_bytes=nbytes + 8192,
                disk_bytes=max(2 * nbytes + 8192, 1024**2),
                direct_io=args.direct_io,
            ) as store:
                start = time.perf_counter()
                store.put(key, packed)
                report["durable_store_ms"] = (time.perf_counter() - start) * 1000
                restored = store.get(key)
                report["byte_exact"] = torch.equal(
                    restored.view(torch.uint8), packed.view(torch.uint8)
                )
                actual_tokens, actual_logits = continuation(restored)
                report["greedy_tokens_identical"] = actual_tokens == expected_tokens
                report["max_abs_logit_difference"] = float(
                    (actual_logits - expected_logits).abs().max().item()
                )
                assert report["byte_exact"] and report["greedy_tokens_identical"]
                assert torch.equal(actual_logits, expected_logits)
                report["prefix_recompute"] = measure(
                    lambda: model(prefix, use_cache=True, **prefill_kwargs),
                    repeats=args.repeats,
                    nbytes=nbytes,
                    device=device,
                )
                report["disk_restore"] = measure(
                    lambda: store.get(key, out=restored),
                    repeats=args.repeats,
                    nbytes=nbytes,
                    device=device,
                )
                start = time.perf_counter()
                store.prefetch(key).result()
                report["prefetch_to_ram_ms"] = (time.perf_counter() - start) * 1000
                report["ram_restore"] = measure(
                    lambda: store.get(key, out=restored),
                    repeats=args.repeats,
                    nbytes=nbytes,
                    device=device,
                )
                sample = packed.view(torch.uint8).flatten()[: 16 * 1024**2].cpu().numpy().tobytes()
                import zlib

                start = time.perf_counter()
                compressed = zlib.compress(sample, level=1)
                report["lossless_zlib_level1_sample"] = {
                    "input_bytes": len(sample),
                    "output_bytes": len(compressed),
                    "ratio": len(sample) / len(compressed),
                    "compress_ms": (time.perf_counter() - start) * 1000,
                    "scope": "at most 16 MiB; not the transport default",
                }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
