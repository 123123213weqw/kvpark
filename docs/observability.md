# Observability

`stats()` is the only supported way to inspect a live store. Both classes keep a
fixed-size metric set, so memory does not grow with the number of cached keys.
Full per-metric tables follow; `docs/methodology.md` covers how measured numbers
are produced.

## What is measured, and what is not

Measured:

- operation counts, per outcome (including rejections and corruption)
- exact byte totals for every payload movement
- exactly one latency sample per **successful** operation, plus the exact
  `min`, `max`, and `mean` of those samples
- bucket upper bounds for `p50`/`p95`/`p99`

Not measured, by design:

- Per-key history. There is no per-key latency map; a million keys still cost
  one fixed set of counters.
- GPU utilization, clocks, or power. Those belong to `nvidia-smi` and to
  `kvpark-bench`, not to the cache.
- Allocator high-water. `ram_used_bytes` is owned payload plus reservations; it
  never claims to include page cache or allocator overhead.

## Reading the latency fields

Quantiles come from a fixed 1-2-5 geometric histogram, so **`p50`/`p95`/`p99`
are upper bounds, never underestimates**. The widest step in that progression is
`2.0 -> 5.0`, so a reported quantile is at most **2.5x** the true value. If a
number needs to be precise, use `min_ms`, `max_ms`, or `mean_ms`, which are
exact.

Latency samples cover successful operations only. A rejected or failed call is
counted (`put_rejected_immutable`, `get_corrupt`, ...) but not timed, so a fast
failure cannot drag a reported distribution downwards.

Empty series report `count: 0` and `null` for every statistic, rather than `0.0`,
so an idle store cannot be mistaken for a measured zero-latency operation.

## Schema stability

Every name listed below is always present, even when its code path never ran.
A dashboard or alert therefore cannot break because an operation type was not
exercised. Unknown names are never invented: only the schemas below are emitted.

## `TieredStore` counters

| Counter | Meaning |
| --- | --- |
| `put_calls` | `put_async` entries, including calls rejected by validation |
| `put_ok` | snapshots written and published to disk |
| `put_errors` | writes that raised (I/O, fsync, rename) |
| `put_rejected_immutable` | `FileExistsError`: the key already exists |
| `put_bytes` | payload bytes accepted |
| `put_stored_bytes` | bytes on disk after 4096-byte rounding |
| `get_calls` | `get_async` entries |
| `get_ok` | restores that returned a tensor |
| `get_errors` | restores that raised for a reason other than the two below |
| `get_misses` | `KeyError`: the key is not on disk |
| `get_corrupt` | `CacheCorruption` raised |
| `get_ram_hits` | served from the pinned-RAM LRU |
| `get_disk_hits` | served from disk |
| `get_ram_bytes` / `get_disk_bytes` | payload bytes delivered by each path |
| `prefetch_calls` | `prefetch` entries |
| `prefetch_ok` | payload admitted to RAM |
| `prefetch_ram_hits` | already resident; no read performed |
| `prefetch_errors` | read failed |
| `prefetch_rejected_large` | declined: allocation exceeds `ram_bytes` |
| `prefetch_bytes` | bytes admitted to RAM |
| `contains_calls` | `contains` entries |
| `contains_hits` / `contains_misses` | outcome of `contains` |
| `evictions_ram` / `evicted_ram_bytes` | RAM LRU evictions |
| `evictions_disk` / `evicted_disk_bytes` | disk LRU evictions |
| `admit_skipped` | payload larger than the RAM budget, so not admitted |
| `checksum_failures` | CRC32 mismatch per chunk |
| `corrupt_short_payload` | unexpected EOF in a payload file |
| `startup_orphan_pending` | `.pending-*` directories removed at open |
| `startup_disk_entries` / `startup_disk_bytes` | snapshots recovered by the startup scan |

## `TieredStore` latency

| Series | Covers |
| --- | --- |
| `put_ms` | D2H/host copy, chunk write, `fsync`, atomic rename, directory `fsync` |
| `get_ms` | whole restore, including `out` allocation and H2D |
| `get_ram_ms` | pinned-RAM to VRAM bulk copy only |
| `get_disk_ms` | open, chunked read, CRC32 verify, staged H2D |
| `prefetch_ms` | disk read plus RAM admission |

`put_ms` is measured inside the worker, so it is the disk-layer cost whether it
was reached through `put()` or through `RamFirstStore` persistence.

## `RamFirstStore` counters

| Counter | Meaning |
| --- | --- |
| `offload_calls` | `offload_async` entries, including rejected calls |
| `offload_accepted` | accepted for copy and archive |
| `offload_bytes` | payload bytes accepted |
| `offload_rejected_immutable` | key already exists |
| `offload_rejected_pending` | `max_pending` reached |
| `offload_rejected_oversize` | one snapshot exceeds the whole RAM budget |
| `offload_rejected_occupied` | RAM held by dirty, failed, or in-flight-read entries |
| `copy_ok` / `copy_errors` | GPU-to-host copy outcome |
| `released_ok` | inputs reported reusable |
| `persist_ok` / `persist_errors` | archive outcome |
| `persist_bytes` | bytes made durable |
| `retry_calls` | `retry_persist` entries |
| `retry_ok` / `retry_errors` | retry outcome |
| `retry_rejected_state` | retried an entry that was not failed |
| `retry_rejected_pending` | `max_pending` reached during retry |
| `discard_calls` / `discard_bytes` | explicit RAM copies dropped |
| `read_calls` | `get_async` entries |
| `read_ok` / `read_errors` | read outcome |
| `read_hot` / `read_hot_bytes` | served from the RAM copy |
| `read_disk_requests` | fell through to the disk layer |
| `prefetch_calls` / `prefetch_busy` / `prefetch_ram_hits` | session-level prefetch |
| `contains_calls` / `contains_hits` / `contains_misses` | session-level lookups |
| `evictions_clean` / `evicted_clean_bytes` | clean RAM entries reclaimed for room |
| `close_calls` / `close_persistence_failures` | close outcome |

## `RamFirstStore` latency

| Series | Covers |
| --- | --- |
| `released_ms` | submit until the GPU input is reusable (the v0.3 milestone) |
| `durable_ms` | submit until the archive is crash-safe |
| `retry_ms` | retry submission until durability is reported |

`released_ms` and `durable_ms` are deliberately separate: the design exists to
make the first small and to keep the second honest. `released` is not durable.

## Derived hit rates

`stats()["hit_rate"]` reports three ratios, or `null` when the denominator is
zero. A fabricated `0.0` on an idle store would read like a measured 0% hit rate.

| Key | `TieredStore` | `RamFirstStore` |
| --- | --- | --- |
| `get` / `read` | hits / (hits + misses) | RAM hits / (RAM hits + disk requests) |
| `contains` | hits / calls | hits / calls |
| `ram_share_of_hits` | RAM hits / all hits | not reported |
| `persist` | not reported | durable / (durable + failures) |

## Layering

`RamFirstStore` owns a `TieredStore(ram_bytes=0)` and both keep their own
counters. A persisted archive therefore appears as session work (`persist_ok`)
and as disk work (`put_ok`) at the same time. This is intentional: it makes the
two costs separable instead of blended.

## Example

```python
with RamFirstStore(root, device="cuda:0", ram_bytes=2 << 30) as store:
    handle = store.offload_async(key, kv)
    handle.released.result()          # GPU input reusable
    handle.durable.result()           # crash-safe
    stats = store.stats()

stats["metrics"]["latency_ms"]["released_ms"]["max_ms"]
stats["metrics"]["latency_ms"]["durable_ms"]["max_ms"]
stats["hit_rate"]["read"]             # None if nothing was read
```

## Cost

Recording a counter is one uncontended lock acquisition and one dict increment.
Recording a latency is one `perf_counter_ns()` call plus a bisect over 25 edges.
Neither allocates. `reset()` clears everything, which is useful between
benchmark cells.
