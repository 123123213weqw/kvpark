# RAM-first offload and foreground scheduling (v0.3)

`TieredStore` retains its original durable/write-through API. `RamFirstStore` is
an opt-in write-back wrapper over the same validated disk format. It trades a
bounded full-snapshot RAM reservation for earlier GPU input release.

```python
from kvpark import CacheBusy, RamFirstStore

# kv is already packed/contiguous on cuda:0; key binds the entire causal prefix.
with RamFirstStore(
    "/data/private-kv", device="cuda:0", ram_bytes=2 * 1024**3,
    disk_bytes=100 * 1024**3, max_pending=2, max_reads=8, direct_io=True,
) as store:
    try:
        handle = store.offload_async(key, kv)
    except CacheBusy:
        # Nothing accepted: retain GPU KV, wait for progress, or skip caching.
        pass
    else:
        handle.released.result()  # D2H complete; mutation/release is now safe.
        del kv                   # Drop ALL engine views/references to release VRAM.
        restored = store.get_async(key).result()  # RAM works before durable.
        handle.durable.result()  # Data+manifest+rename+directory fsync completed.
```

The two milestones are deliberately different:

| Milestone | Guaranteed | Not guaranteed |
| --- | --- | --- |
| `released` success | Owned exact host copy; producer dependency and D2H complete; no store-owned GPU input reference | Surviving a process/power failure; freeing caller-held views; returning allocator-reserved GPU memory to the driver |
| `durable` success | Atomic published snapshot, fsynced data/metadata/directory | Redundancy, future quota residency, or that RAM still retains it |

A failed `released` does **not** provide a valid saved copy. Retain/recompute the
engine state. A failed `durable` after a successful release leaves a readable,
non-evictable RAM copy. Inspect the Future and `failed_entries`, fix the cause,
and call `retry_persist(key)`; no GPU input is needed again. `discard(key)` is an
explicit acknowledgement that a clean/failed RAM copy may be dropped. It rejects
in-flight entries. `close()` drains accepted operations and raises
`PersistenceError` if failed RAM-only/uncertain-durability entries remain. After
close, RAM is gone, including failed entries; handle that exception before exit
if recovery is required. On any process crash, non-durable state may be lost.

## Ownership, bounds, and backpressure

Before retaining an input, admission reserves
`round_up(payload_bytes, 4096) + 4096` bytes of owned host allocation, including
alignment slack. Only clean, non-read-pinned entries can be LRU-evicted. Copying,
loading, dirty, failed, and actively read entries are protected. Clean eviction
clears its tensor reference even if a completed notification handle survives.

- `ram_bytes`: all full-host reservations/entries, including queued D2H jobs.
- `max_pending`: accepted D2H/queued/in-flight archives, not an unbounded GPU queue.
- `max_reads`: accepted foreground reads/prefetches, including waits for a D2H.
- Exceeding a bound raises `CacheBusy` immediately, before retaining the input or
  destination. Callers must not implement an unbounded retry queue around it.
- A snapshot larger than RAM is rejected; use streamed `TieredStore.put` and
  accept its slower GPU release, or skip caching. The disk budget still applies.
- Read/output Futures and milestone Futures cannot be cancelled. Ownership is
  not revoked by cancelling a notification. Context exit waits accepted work.

The disk staging ring adds `slots × (chunk_bytes + 4096)` bytes outside this RAM
budget (normally about 16 MiB). GPU outputs are engine/caller-owned. A contiguous
view can share a larger source allocation; drop all engine aliases after release.
Pinned allocator high-water and buffered filesystem page cache remain outside the
retained-allocation bound. This is not a hard RSS/cgroup memory limit.

## Lanes and priority

1. One offload worker owns a D2H stream and copies into a full reserved host buffer.
2. One hot-read worker owns an independent H2D stream. RAM hits, including dirty
   entries, do not queue behind disk persistence or a cold read.
3. One disk coordinator owns the existing staging buffers and read worker. Cold
   restores have priority 0, explicit prefetch priority 5, and archives priority 10.
   Lower numbers run first, FIFO on ties. After eight ordinary selections the
   oldest accepted job receives a turn, preventing indefinite write starvation.

Priority applies at **operation boundaries**, never by interrupting an in-flight
`write/fsync/read`. A cold read can still wait for one large archive already in
flight. Separate CUDA streams do not create additional PCIe bandwidth or ensure
hardware copy-engine preemption. Foreground RAM hits may contend with ongoing
D2H/H2D traffic. This is queue isolation, not a hard real-time guarantee.

`prefetch(key)` explicitly populates RAM after reopen/cold eviction. Its Future
returns False if a full snapshot cannot be admitted; `get` cold misses stream
through disk staging without implicit full-RAM admission. Disk quota may evict a
clean snapshot while its separate RAM copy remains usable; after both copies are
evicted it is a cache miss. Disk LRU tracks disk operations, not independent RAM
hits. Keys remain immutable while either copy exists.

Callbacks run on worker threads: keep them short and nonblocking. Do not call
`close()` or synchronously wait for work on the same lane from a done callback.
Always close explicitly/use a context manager; interpreter exit is not a flush.

## Reproduce the mixed workload

```bash
python examples/mixed_writeback.py --root /your/hdd/bench --device cuda:0 \
  --model roneneldan/TinyStories-1M \
  --revision 77f1b168e219585646439073245fe87e56b3023e \
  --model-cache-dir /your/model-cache --local-files-only --tokens 1024 \
  --archive-mib 1024 --repeats 3 --requests 3 --arrival-interval-ms 50 \
  --decode-tokens 8 --direct-io --output results/mixed-hdd.json
```

Each trial prepares a real model KV RAM hit, then archives a separate 1 GiB
random-byte synthetic tensor. Three actual requests arrive 50 ms apart, each
restoring the model KV and decoding eight tokens on a single decoder lane.
Modes alternate order across repeats. Release/durability timers begin at archive
submission; request latency begins at each **actual** arrival, includes all
store/decoder queue waits, and ends at transfer completion or first generated
token. Caller input references are dropped immediately after submission; a
watcher also measures when PyTorch's **allocated** GPU bytes fall by at least 90%
of the archive size. Reserved allocator memory need not fall. Warmup and archive
hashing are outside timings; RAM-first's full-buffer allocation is inside release
time. Pinned allocator reuse can therefore affect later trials.

The large archive is not a 1 GiB model KV. The real pretrained cache is about
2 MiB. Exact bytes, next logits against the original model cache, and greedy
continuation are checked for every resume; persisted archive SHA-256 is checked
on a forced disk read outside timings. The synthetic byte hash guards against
reordering as well as bit corruption. This experiment does not establish large
model speed, concurrent HTTP serving capacity, p99, quality, or sustained HDD
write/read equilibrium. RAM-first cannot make archival bandwidth infinite.
