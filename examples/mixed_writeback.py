"""A large synthetic archive competes with small, real model KV resumes.

Compare legacy FIFO/durable put with bounded RAM-first offload. All arrivals,
restore completion, and model first-token times include their actual queues.
The archive is synthetic; model correctness is tested on the separate real KV.
This is a shared-node experiment, not an HTTP service or latency SLA.
"""

import argparse
import hashlib
import inspect
import json
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from kv_tier import RamFirstStore, TieredStore, cache_key
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
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--archive-mib", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--arrival-interval-ms", type=float, default=50)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--direct-io", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if (
        min(args.tokens - 1, args.archive_mib, args.repeats, args.requests, args.decode_tokens) < 1
        or args.arrival_interval_ms < 0
    ):
        parser.error("positive sizes/counts and nonnegative arrival interval required")
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index or 0)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.set_num_threads(2)
    torch.manual_seed(20261022)
    if args.model:
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                revision=args.revision,
                cache_dir=args.model_cache_dir,
                local_files_only=args.local_files_only,
                trust_remote_code=False,
                weights_only=True,
                torch_dtype=dtype,
                attn_implementation="eager",
            )
            .to(device)
            .eval()
        )
        model_name = args.model
        revision = resolve_model_revision(model.config, args.revision)
    else:
        model = (
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=512,
                    hidden_size=128,
                    intermediate_size=256,
                    num_hidden_layers=4,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=max(4096, args.tokens + args.decode_tokens),
                )
            )
            .to(device=device, dtype=dtype)
            .eval()
        )
        model_name, revision = "tiny-random-llama", "seed-20261022"
    ids = torch.randint(0, model.config.vocab_size, (1, args.tokens), device=device)
    kwargs = (
        {"logits_to_keep": 1}
        if "logits_to_keep" in inspect.signature(model.forward).parameters
        else {}
    )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def decode(cache, on_first=None):
        with torch.inference_mode():
            current, generated, first_logits = ids[:, -1:], [], None
            for index in range(args.decode_tokens):
                prediction = model(current, past_key_values=cache, use_cache=True)
                cache = prediction.past_key_values
                current = prediction.logits[:, -1].argmax(-1, keepdim=True)
                generated.append(int(current.item()))
                if index == 0:
                    first_logits = prediction.logits[:, -1].clone()
                    if on_first is not None:
                        on_first()
            return generated, first_logits

    with torch.inference_mode():
        original = model(ids[:, :-1], use_cache=True, **kwargs)
        packed = pack_dynamic_cache(original.past_key_values)
        # Gold is the original model cache, before packing or unpacking it.
        golden, golden_logits = decode(original.past_key_values)
        del original
        decode(unpack_dynamic_cache(packed.clone()))  # warm model kernels
    kv_bytes = packed.numel() * packed.element_size()
    archive_bytes = args.archive_mib * 1024**2
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        if free < 2 * archive_bytes + 512 * 1024**2:
            raise RuntimeError("insufficient shared-GPU headroom; reduce --archive-mib")
    ram_bytes = archive_bytes + kv_bytes + 65536
    key = cache_key(
        {
            "model": model_name,
            "revision": revision,
            "dtype": str(dtype),
            "layout": "L2BHTD-v1",
            "tenant": "public-mixed-experiment",
        },
        ids[0, :-1].tolist(),
    )
    report = {
        "hardware": hardware(device),
        "model": model_name,
        "revision": revision,
        "prefix_tokens": args.tokens - 1,
        "kv_bytes": kv_bytes,
        "archive_bytes": archive_bytes,
        "archive_is_synthetic": True,
        "ram_budget_bytes": ram_bytes,
        "direct_io": args.direct_io,
        "gpu_exclusive": False,
        "repeats": args.repeats,
        "decode_tokens": args.decode_tokens,
        "arrival_interval_ms": args.arrival_interval_ms,
        "scope": "hot real-KV resumes during one large synthetic archive; no p99 claim",
        "trials": [],
    }
    parent = Path(args.root).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        modes = ("fifo_durable", "ram_first") if repeat % 2 == 0 else ("ram_first", "fifo_durable")
        for mode in modes:
            with tempfile.TemporaryDirectory(prefix="kv-tier-mixed-", dir=parent) as directory:
                cls = RamFirstStore if mode == "ram_first" else TieredStore
                options = {"max_reads": args.requests + 1} if mode == "ram_first" else {}
                with cls(
                    directory,
                    device=str(device),
                    ram_bytes=ram_bytes,
                    disk_bytes=2 * archive_bytes + kv_bytes + 65536,
                    direct_io=args.direct_io,
                    **options,
                ) as store:
                    if mode == "ram_first":
                        store.offload_async(key, packed).durable.result()
                    else:
                        store.put(key, packed)
                        assert store.prefetch(key).result()
                    store.get(key)  # warm transport setup outside the measurement
                    synchronize()
                    background = torch.randint(
                        0, 256, (archive_bytes,), dtype=torch.uint8, device=device
                    )
                    expected_archive = hashlib.sha256()
                    for offset in range(0, archive_bytes, 16 * 1024**2):
                        expected_archive.update(
                            background[offset : offset + 16 * 1024**2].cpu().numpy().tobytes()
                        )
                    synchronize()
                    allocated_before = (
                        torch.cuda.memory_allocated(device) if (device.type == "cuda") else None
                    )
                    archive_key = hashlib.sha256(f"archive-{repeat}-{mode}".encode()).hexdigest()
                    trial = {"repeat": repeat, "mode": mode, "requests": []}
                    start = time.perf_counter()
                    if mode == "ram_first":
                        handle = store.offload_async(archive_key, background)
                        released, durable = handle.released, handle.durable
                    else:
                        released = durable = store.put_async(archive_key, background)
                    del background  # only store-owned references now remain

                    def watch_release(
                        released=released,
                        trial=trial,
                        start=start,
                        allocated_before=allocated_before,
                    ):
                        released.result()
                        trial["input_reusable_ms"] = (time.perf_counter() - start) * 1000
                        if device.type == "cuda":
                            deadline = time.perf_counter() + 10
                            # Small foreground/model allocations may coexist. Require
                            # >=90% of the archive allocation to become unallocated.
                            threshold = allocated_before - int(archive_bytes * 0.9)
                            while time.perf_counter() < deadline:
                                allocated = torch.cuda.memory_allocated(device)
                                if allocated <= threshold:
                                    trial["gpu_allocation_released_ms"] = (
                                        time.perf_counter() - start
                                    ) * 1000
                                    trial["allocated_bytes_before_archive_release"] = (
                                        allocated_before
                                    )
                                    trial["allocated_bytes_after_archive_release"] = allocated
                                    break
                                time.sleep(0.001)
                            else:
                                raise AssertionError("GPU archive allocation remained live")

                    watcher = ThreadPoolExecutor(max_workers=1)
                    watched = watcher.submit(watch_release)
                    durable.add_done_callback(
                        lambda _f, trial=trial, start=start: trial.update(
                            durable_ms=(time.perf_counter() - start) * 1000
                        )
                    )

                    def serve(future, row, start=start, durable=durable):
                        with torch.inference_mode():
                            restored = future.result()
                            row["restore_ready_observed_by_decoder_ms"] = (
                                time.perf_counter() - row.pop("arrival_clock")
                            ) * 1000
                            assert torch.equal(restored.view(torch.uint8), packed.view(torch.uint8))

                            def first():
                                row["arrival_ttft_ms"] = (time.perf_counter() - start) * 1000 - row[
                                    "arrival_ms"
                                ]

                            tokens, logits = decode(unpack_dynamic_cache(restored), first)
                            row["max_next_logit_difference"] = float(
                                (logits - golden_logits).abs().max().item()
                            )
                            row["continuation_match"] = tokens == golden
                            assert row["continuation_match"] and (
                                row["max_next_logit_difference"] == 0
                            )
                            row["archive_still_pending_at_completion"] = not durable.done()

                    with ThreadPoolExecutor(max_workers=1) as decoder:
                        requests = []
                        for index in range(args.requests):
                            deadline = start + index * args.arrival_interval_ms / 1000
                            time.sleep(max(0, deadline - time.perf_counter()))
                            arrival = time.perf_counter()
                            row = {
                                "index": index,
                                "arrival_ms": (arrival - start) * 1000,
                                "arrival_clock": arrival,
                            }
                            future = store.get_async(key)
                            # Stamp transfer completion independently of decoder queue.
                            future.add_done_callback(
                                lambda _f, row=row, arrival=arrival: row.update(
                                    transport_completion_ms_including_queue=(
                                        time.perf_counter() - arrival
                                    )
                                    * 1000
                                )
                            )
                            requests.append(decoder.submit(serve, future, row))
                            trial["requests"].append(row)
                        for request in requests:
                            request.result()
                    durable.result()
                    watched.result()
                    watcher.shutdown()
                    trial["store_stats"] = store.stats()
                    # Validate persisted archive outside timings, forcing disk, not RAM.
                    if mode == "ram_first":
                        store.discard(archive_key)
                    else:
                        store.clear_ram()
                    restored = store.get(archive_key)
                    actual_archive = hashlib.sha256()
                    for offset in range(0, archive_bytes, 16 * 1024**2):
                        actual_archive.update(
                            restored[offset : offset + 16 * 1024**2].cpu().numpy().tobytes()
                        )
                    assert actual_archive.digest() == expected_archive.digest()
                    trial["archive_sha256"] = actual_archive.hexdigest()
                    del restored
                    trial["archive_disk_roundtrip_exact"] = True
                    report["trials"].append(trial)
    report["summary"] = {}
    for mode in ("fifo_durable", "ram_first"):
        trials = [trial for trial in report["trials"] if trial["mode"] == mode]
        report["summary"][mode] = {
            "median_input_reusable_ms": statistics.median(t["input_reusable_ms"] for t in trials),
            "median_durable_ms": statistics.median(t["durable_ms"] for t in trials),
            "median_arrival_ttft_ms": statistics.median(
                row["arrival_ttft_ms"] for t in trials for row in t["requests"]
            ),
            "median_transport_ms_including_queue": statistics.median(
                row["transport_completion_ms_including_queue"]
                for t in trials
                for row in t["requests"]
            ),
        }
        if device.type == "cuda":
            report["summary"][mode]["median_gpu_allocation_released_ms"] = statistics.median(
                t["gpu_allocation_released_ms"] for t in trials
            )
    encoded = json.dumps(report, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
