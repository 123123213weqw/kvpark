# v0.2.0 — validated consumer-GPU KV tiering

## What is delivered

- Exact, immutable tensor snapshots on HDD/SSD, a bounded retained pinned-RAM LRU,
  and CUDA-event-safe chunked restore buffers.
- Disk-read/checksum overlap: the measured verified 1-GiB SATA restore falls from
  3146.42 ms in the serial control to 2142.57 ms, about 31.9% lower latency.
- Prepared-RAM 1-GiB restore around 0.10 seconds on the target RTX 4090 D / PCIe Gen3 x16 node.
- Actual pretrained TinyStories cache restore, original-cache logit/continuation comparisons,
  and known-queue traces reporting both service and queue-inclusive latency.
- 54 passing tests on the actual CUDA node; source-hash-matched validation metadata.
- Reproducible transport/model scripts, capacity/admission arithmetic, and a low-incremental-cost plan.

## Honest boundaries

This is an experimental snapshot/resume lab, not a production vLLM connector or transparent
active-decode paging system. The GPU has GDDR6X, not HBM. A 1-GiB HDD cold restore still
takes about 9.1 seconds, and preparing a RAM hit has a real cost. Prefetch cannot remove a
sustained bandwidth deficit. The small trained-model queue trace is not a large-model or p99 benchmark.

Static, quantized, rolling/sliding-eviction, and hybrid cache objects are rejected by the HF adapter.
No cache quantization or default compression changes model state. There are no private prompts,
model weights, credentials, host addresses, or third-party service sources in this repository.

Existing GPU services were left running. No driver/kernel/route changes, global cache dropping,
root-only tuning, or Rust compilation were used. See the raw-backed results and production gate
before integrating the prototype into a serving path.
