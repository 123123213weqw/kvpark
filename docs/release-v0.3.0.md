# v0.3.0 — bounded RAM-first KV offload

## Delivered

- Opt-in `RamFirstStore` preserves the existing `TieredStore` durable API and disk format.
- Separate `released` (owned RAM copy / GPU input reusable) and `durable` (fsynced disk) milestones.
- Bounded full-host reservations, pending archives and reads; explicit `CacheBusy` backpressure.
- Dirty/failed entries are protected; explicit retry/discard and shutdown failure reporting.
- Independent foreground RAM H2D lane; cold-read/prefetch/archive priorities at disk-operation
  boundaries, with a bounded starvation guard. In-flight disk writes are not preempted.
- Post-rename failure accounting, bounded verification on retry, and cleanup of completed
  handles/callbacks so they do not retain evicted host storage or failed auto-allocated outputs.
- Reproducible mixed real-KV/synthetic-archive workload, source-matched validation and raw JSON.

## Actual 4090 D measurements

One 1 GiB random-byte archive competes with real pretrained TinyStories resumes
(about 2 MiB KV, 1023 prefix tokens, eight generated tokens). Three repeats per
mode/disk, three arrivals per trial, 50 ms apart. Medians include actual queues.

| Backing/mode | GPU input reusable | Queue-inclusive model TTFT | Durable archive |
| --- | ---: | ---: | ---: |
| HDD, legacy FIFO | 10197.59 ms | 10288.97 ms | 10197.41 ms |
| HDD, RAM-first | 98.08 ms | 156.69 ms | 10615.70 ms |
| SATA SSD, legacy FIFO | 4052.06 ms | 4147.14 ms | 4051.89 ms |
| SATA SSD, RAM-first | 97.45 ms | 150.85 ms | 4389.68 ms |

Observed allocated-GPU-memory release also completes at about 98 ms for RAM-first;
reserved allocator memory need not shrink. All 36 resumes match original-cache
bytes, next logits (difference zero), and greedy continuation. Every synthetic
archive passes a forced-disk SHA-256 check; all RAM-first resumes finish while
the background archive is still pending.

80 tests pass on the actual CUDA node (CPU and GPU cases, no skips/failures),
with a matching tested-source hash; local macOS has 74 passes and six hardware skips.
The pure-Python wheel passes isolated install/offload/reopen/prefetch/backpressure smoke testing.

## Boundaries

**GPU release is not disk durability.** Dirty RAM can be lost on a crash; caller
aliases still retain VRAM. Full-snapshot RAM capacity is necessary for fast offload.
Cold reads may still wait for an in-flight archive. Pinned allocator high-water,
page cache, and caller-owned GPU outputs are outside retained-entry accounting.

The large archive is synthetic, not a 1 GiB model cache. This is not a large-model,
p99, HTTP concurrency, active-decode paging, or sustained-storage-bandwidth benchmark.
The baseline is this lab's legacy FIFO implementation, not a competing production cache.
The 4090 D uses GDDR6X, not HBM; the node has HDD/SATA, not measured NVMe.

No existing GPU service, driver, route, SSH trust setting or global OS cache was changed.
No private prompts, KV payloads, model weights, host addresses or credentials are published.
Apache-2.0. See README, docs/writeback.md, docs/results.md and committed raw traces.
