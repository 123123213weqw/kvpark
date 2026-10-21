"""Transport benchmarks. Synthetic bytes are NOT an end-to-end LLM benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .store import TieredStore


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(function, *, repeats, nbytes, device):
    samples = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        function()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples)
    median = statistics.median(samples)
    return {
        "samples_ms": samples,
        "median_ms": median,
        "min_ms": min(samples),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "GB_per_s": nbytes / (median * 1e6),
    }


def hardware(device):
    result = {
        "os": platform.system(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu=properties.name, gpu_total_bytes=properties.total_memory)
        try:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=pcie.link.gen.current,pcie.link.gen.max,"
                    "pcie.link.width.current,memory.used,utilization.gpu,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            result["nvidia_smi_start"] = query.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            result["nvidia_smi_start"] = "unavailable"
    return result


def run(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    if device.type == "cuda":
        device = torch.device("cuda", device.index or 0)
    report = {
        "schema": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware(device),
        "configuration": {
            "storage_label": args.label,
            "direct_io": args.direct_io,
            "verification": args.verify,
            "chunk_mib": args.chunk_mib,
            "staging_slots": args.slots,
            "repeats": args.repeats,
            "cpu_threads": args.cpu_threads,
            "synthetic_transport_only": True,
            "gpu_exclusive": False,
            "pipeline_reads": args.pipeline_reads,
        },
        "cases": [],
    }
    parent = Path(args.root).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    for mib in args.sizes_mib:
        nbytes = int(mib * 1024**2)
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            if 2 * nbytes + 512 * 1024**2 > free:
                raise RuntimeError("insufficient GPU headroom; reduce --sizes-mib")
        torch.manual_seed(20261022)
        source = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=device)
        destination = torch.empty_like(source)
        _sync(device)
        key = hashlib.sha256(f"synthetic-{nbytes}".encode()).hexdigest()
        case = {"size_mib": mib, "nbytes": nbytes}
        with tempfile.TemporaryDirectory(prefix="kv-tier-bench-", dir=parent) as temporary:
            with TieredStore(
                temporary,
                device=str(device),
                ram_bytes=nbytes + 8192,
                disk_bytes=max(2 * nbytes + 8192, 1024**2),
                chunk_bytes=args.chunk_mib * 1024**2,
                staging_slots=args.slots,
                direct_io=args.direct_io,
                verify=args.verify,
                pipeline_reads=args.pipeline_reads,
            ) as store:
                start = time.perf_counter()
                store.put(key, source)
                case["durable_store_ms"] = (time.perf_counter() - start) * 1000
                store.get(key, out=destination)
                if not torch.equal(source, destination):
                    raise AssertionError("initial disk roundtrip was not byte exact")
                label = "disk_direct" if args.direct_io else "buffered_cache_uncontrolled"
                case[label] = measure(
                    lambda key=key, out=destination: store.get(key, out=out),
                    repeats=args.repeats,
                    nbytes=nbytes,
                    device=device,
                )
                if not torch.equal(source, destination):
                    raise AssertionError("measured disk roundtrip was not byte exact")
                start = time.perf_counter()
                assert store.prefetch(key).result()
                case["prefetch_to_ram_ms"] = (time.perf_counter() - start) * 1000
                case["ram_hit"] = measure(
                    lambda key=key, out=destination: store.get(key, out=out),
                    repeats=args.repeats,
                    nbytes=nbytes,
                    device=device,
                )
                if not torch.equal(source, destination):
                    raise AssertionError("RAM roundtrip was not byte exact")
                case["store_stats"] = store.stats()
                store.clear_ram()
                pageable = source.cpu().clone() if device.type == "cuda" else source.clone()
                pinned = pageable.pin_memory() if device.type == "cuda" else pageable
                _sync(device)
                destination.copy_(pinned, non_blocking=True)
                _sync(device)
                case["pinned_copy_baseline"] = measure(
                    lambda out=destination, host=pinned: out.copy_(host, non_blocking=True),
                    repeats=args.repeats,
                    nbytes=nbytes,
                    device=device,
                )
                case["pageable_copy_baseline"] = measure(
                    lambda out=destination, host=pageable: out.copy_(host),
                    repeats=args.repeats,
                    nbytes=nbytes,
                    device=device,
                )
                case["byte_exact"] = torch.equal(source, destination)
                del pageable, pinned
        del source, destination
        report["cases"].append(case)
        print(json.dumps({"completed": case}, indent=2), flush=True)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Parent for disposable, owned benchmark data")
    parser.add_argument("--label", default="unspecified-storage")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sizes-mib", nargs="+", type=int, default=[16, 256, 1024])
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--direct-io", action="store_true")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pipeline-reads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.sizes_mib + [args.chunk_mib, args.slots, args.repeats, args.cpu_threads]) <= 0:
        parser.error("sizes, slots, repeats, and CPU threads must be positive")
    report = run(args)
    if not args.output:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
