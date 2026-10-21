"""Actual model continuation while the next queued session is prefetched from disk.

The queue is known in advance. Service TTFT excludes time spent waiting in the
request queue; total wall time includes initial prefetch and all generation.
This is a reproducible sequential queue experiment, not a production scheduler.
"""

import argparse
import hashlib
import inspect
import json
import statistics
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from kv_tier import TieredStore, cache_key
from kv_tier.benchmark import hardware
from kv_tier.hf import pack_dynamic_cache, resolve_model_revision, unpack_dynamic_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--direct-io", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.tokens < 2 or args.sessions < 2 or args.decode_tokens < 1:
        parser.error("tokens/sessions must be >=2 and decode-tokens >=1")
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.set_num_threads(2)
    torch.manual_seed(195)
    if args.model:
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                revision=args.revision,
                torch_dtype=dtype,
                cache_dir=args.model_cache_dir,
                local_files_only=args.local_files_only,
                trust_remote_code=False,
                weights_only=True,
                attn_implementation=args.attn_implementation,
            )
            .to(device)
            .eval()
        )
        revision = resolve_model_revision(model.config, args.revision)
        model_name = args.model
    else:
        config = LlamaConfig(
            vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=max(4096, args.tokens + args.decode_tokens),
        )
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
        model_name, revision = "tiny-random-llama", "random-seed-195"
    signature = {
        "model": model_name,
        "revision": revision,
        "config": model.config.to_dict(),
        "dtype": str(dtype),
        "layout": "L2BHTD-v1",
        "tenant": "public-queue-experiment",
    }
    prefill_kwargs = (
        {"logits_to_keep": 1}
        if "logits_to_keep" in inspect.signature(model.forward).parameters
        else {}
    )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    session_ids = [
        torch.randint(0, model.config.vocab_size, (1, args.tokens), device=device)
        for _ in range(args.sessions)
    ]
    parent = Path(args.root).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": model_name,
        "model_revision": revision,
        "hardware": hardware(device),
        "tiny_random_model": args.model is None,
        "prefix_tokens": args.tokens - 1,
        "sessions": args.sessions,
        "decode_tokens_per_session": args.decode_tokens,
        "direct_io": args.direct_io,
        "verification": True,
        "gpu_exclusive": False,
        "scope": "known sequential queue; service TTFT excludes arrival queue wait",
        "arrival_model": "all requests are available at the start of each measured mode",
        "modes": {},
        "prefill_kwargs": prefill_kwargs,
    }
    with (
        torch.inference_mode(),
        tempfile.TemporaryDirectory(prefix="kv-tier-queue-", dir=parent) as temporary,
    ):
        initial = model(session_ids[0][:, :-1], use_cache=True, **prefill_kwargs)
        sample = pack_dynamic_cache(initial.past_key_values)
        nbytes = sample.numel() * sample.element_size()
        report["cache_bytes_per_session"] = nbytes
        report["ram_budget_bytes"] = 2 * (nbytes + 8192)
        del initial, sample
        with TieredStore(
            temporary,
            device=str(device),
            ram_bytes=report["ram_budget_bytes"],
            disk_bytes=max((args.sessions + 1) * (nbytes + 8192), 1024**2),
            direct_io=args.direct_io,
        ) as store:
            keys = []
            original_hashes = []
            for ids in session_ids:
                output = model(ids[:, :-1], use_cache=True, **prefill_kwargs)
                packed = pack_dynamic_cache(output.past_key_values)
                key = cache_key(signature, ids[0, :-1].tolist())
                store.put(key, packed)
                keys.append(key)
                original_cache = output.past_key_values
                current = ids[:, -1:]
                original_tokens = []
                for _ in range(args.decode_tokens):
                    prediction = model(current, past_key_values=original_cache, use_cache=True)
                    original_cache = prediction.past_key_values
                    current = prediction.logits[:, -1].argmax(-1, keepdim=True)
                    original_tokens.append(int(current.item()))
                original_hashes.append(
                    hashlib.sha256(json.dumps(original_tokens).encode()).hexdigest()
                )
                del original_cache, prediction
                del packed, output
            # A warmup decode avoids billing one-time CUDA initialization to one mode.
            restored = store.get(keys[0])
            model(session_ids[0][:, -1:], past_key_values=unpack_dynamic_cache(restored))
            synchronize()
            del restored
            for mode in ("on_demand_disk", "queued_lookahead"):
                store.clear_ram()
                measurements = []
                total_start = time.perf_counter()
                pending = store.prefetch(keys[0]) if mode == "queued_lookahead" else None
                for index, (ids, key) in enumerate(zip(session_ids, keys, strict=True)):
                    request_start = time.perf_counter()
                    wait_start = request_start
                    if pending is not None:
                        assert pending.result()
                    wait_ms = (time.perf_counter() - wait_start) * 1000
                    start = time.perf_counter()
                    restored = store.get(key)
                    restore_ms = (time.perf_counter() - start) * 1000
                    cache = unpack_dynamic_cache(restored)
                    if mode == "queued_lookahead" and index + 1 < len(keys):
                        pending = store.prefetch(keys[index + 1])
                    else:
                        pending = None
                    current = ids[:, -1:]
                    continuation = []
                    first_token_ms = None
                    for token_index in range(args.decode_tokens):
                        output = model(current, past_key_values=cache, use_cache=True)
                        cache = output.past_key_values
                        current = output.logits[:, -1].argmax(-1, keepdim=True)
                        continuation.append(int(current.item()))
                        if token_index == 0:
                            synchronize()
                            first_token_ms = (time.perf_counter() - request_start) * 1000
                    synchronize()
                    measurements.append(
                        {
                            "session": index,
                            "prefetch_wait_ms": wait_ms,
                            "restore_ms": restore_ms,
                            "service_ttft_ms": first_token_ms,
                            "arrival_ttft_ms": (request_start - total_start) * 1000
                            + first_token_ms,
                            "total_service_ms": (time.perf_counter() - request_start) * 1000,
                            "continuation_sha256": hashlib.sha256(
                                json.dumps(continuation).encode()
                            ).hexdigest(),
                        }
                    )
                    del restored, cache, output
                wall_ms = (time.perf_counter() - total_start) * 1000
                report["modes"][mode] = {
                    "wall_ms_including_initial_prefetch": wall_ms,
                    "median_service_ttft_ms": statistics.median(
                        row["service_ttft_ms"] for row in measurements
                    ),
                    "steady_state_median_service_ttft_ms": statistics.median(
                        row["service_ttft_ms"] for row in measurements[1:]
                    ),
                    "median_arrival_ttft_ms": statistics.median(
                        row["arrival_ttft_ms"] for row in measurements
                    ),
                    "requests": measurements,
                    "store_stats": store.stats(),
                }
        a = report["modes"]["on_demand_disk"]["requests"]
        b = report["modes"]["queued_lookahead"]["requests"]
        report["continuations_identical"] = all(
            x["continuation_sha256"] == y["continuation_sha256"] for x, y in zip(a, b, strict=True)
        )
        assert report["continuations_identical"]
        report["continuations_match_original_cache"] = all(
            row["continuation_sha256"] == expected
            for mode in report["modes"].values()
            for row, expected in zip(mode["requests"], original_hashes, strict=True)
        )
        assert report["continuations_match_original_cache"]
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
