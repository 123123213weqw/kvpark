"""Capacity and admission arithmetic; all speeds and prices are caller inputs."""

import argparse
import json
from dataclasses import asdict, dataclass


def kv_bytes(*, layers, kv_heads, head_dim, tokens, batch=1, bits=16):
    values = (layers, kv_heads, head_dim, tokens, batch, bits)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("all cache dimensions and bit width must be positive integers")
    return (2 * layers * kv_heads * head_dim * tokens * batch * bits + 7) // 8


@dataclass(frozen=True)
class RestoreDecision:
    action: str
    estimated_restore_ms: float
    measured_recompute_ms: float
    explanation: str


def choose_restore(*, nbytes, link_gbps, disk_gbps, recompute_ms, ram_hit=False, margin=0.9):
    """Conservative streaming estimate: pipeline service time plus one chunk tail.

    This is not a latency guarantee. Supply effective measured bandwidth, not
    PCIe specifications. Launch, packing, checksum, and queue delays must be
    included in an engine's measured curves before production admission.
    """
    if min(nbytes, link_gbps, disk_gbps, recompute_ms) <= 0 or not 0 < margin <= 1:
        raise ValueError("sizes, speeds, recompute time must be positive; margin must be in (0,1]")
    link_ms = nbytes / (link_gbps * 1e6)
    disk_ms = nbytes / (disk_gbps * 1e6)
    tail_ms = min(nbytes, 8 * 1024**2) / (link_gbps * 1e6)
    estimate = link_ms if ram_hit else max(link_ms, disk_ms) + tail_ms
    restore = estimate < recompute_ms * margin
    return RestoreDecision(
        "restore" if restore else "recompute",
        estimate,
        recompute_ms,
        "estimated restore fits the configured advantage margin"
        if restore
        else "restoring is not predictably cheaper than recomputing the prefix",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--link-gbps", type=float, required=True)
    parser.add_argument("--disk-gbps", type=float, required=True)
    parser.add_argument("--recompute-ms", type=float, required=True)
    parser.add_argument("--ram-hit", action="store_true")
    parser.add_argument("--disk-price-per-tb", type=float, help="User-supplied price, not a quote")
    args = parser.parse_args()
    nbytes = kv_bytes(
        layers=args.layers,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        tokens=args.tokens,
        bits=args.bits,
        batch=args.batch,
    )
    result = {
        "cache_bytes": nbytes,
        "cache_gib": nbytes / 1024**3,
        "decision": asdict(
            choose_restore(
                nbytes=nbytes,
                link_gbps=args.link_gbps,
                disk_gbps=args.disk_gbps,
                recompute_ms=args.recompute_ms,
                ram_hit=args.ram_hit,
            )
        ),
        "note": "Estimate only: measure packing, queues, validation, and engine TTFT separately.",
    }
    if args.disk_price_per_tb is not None:
        if args.disk_price_per_tb < 0:
            parser.error("price must be nonnegative")
        result["illustrative_storage_capex_per_snapshot"] = nbytes / 10**12 * args.disk_price_per_tb
        result["cost_scope"] = (
            "Capacity capex only; excludes power, redundancy, wear, RAM, and GPU."
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
