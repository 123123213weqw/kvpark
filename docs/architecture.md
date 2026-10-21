# Architecture and operating envelope

## The economical target

The target is many **inactive** or reusable prefixes, with a much smaller active working set.
On a machine already equipped with abundant DRAM, a cheap HDD can hold cold snapshots and
DRAM can cover the predicted next working set. An SSD is a useful warm backing tier when
unpredictable cache misses need lower latency. Adding NVMe may improve cold restores, but
no such hardware is assumed or silently substituted into results.

The prototype has one persistent disk backing tier, a pinned-RAM LRU, and engine-owned GPU
tensors. The same API can target either HDD or SSD roots; automatic HDD↔SSD migration is
**not implemented**. An SSD write-back cache is a possible extension, not a shipped feature.

An exact, cost-effective policy is:

1. Keep active decoding state on the GPU if it fits.
2. Persist completed/paused prefixes to cheap disk outside the next request's critical path.
3. Predict the next session/prefix and prefetch its cache to pinned RAM before scheduling.
4. Restore a RAM hit with one asynchronous bulk H2D copy.
5. Stream an unpredicted disk miss through a bounded staging ring, or recompute if faster.
6. Evict the oldest in-memory entry when the bounded RAM budget is exhausted.

## Why not page active dense attention from HDD?

For full dense attention, every layer attends to the existing history for every generated token.
Moving a 1 GiB KV snapshot once per resumed request can be worthwhile. Reading it from HDD
for every generated token would require seconds per token on ordinary hardware. Copy/compute
overlap only hides work when sufficient slack exists; it does not create bandwidth.

Keeping recent tokens while omitting older ones changes attention semantics. Lossy KV
quantization can reduce storage and PCIe traffic but needs model/task-quality evaluation.
Neither is enabled by this exact-byte prototype. General-purpose lossless compression is
measured on real KV samples rather than assumed to provide a large saving.

## Data and persistence

Each immutable key owns a directory containing `data.bin` and `manifest.json`.
The manifest records dtype, shape, payload length, stored length, writer chunk size, and
per-chunk CRC32. Padding reaches 4096-byte alignment. The reader validates sizes before
allocating an output tensor and verifies chunks before copying them to the GPU.

Writes go into an owned `.pending-*` directory. Data and metadata are fsynced, then the
directory is renamed and its parent fsynced. On reopen, abandoned pending directories are
removed. A corrupt complete entry raises an explicit error; the lab does not silently serve
it. An engine should turn that error into a cache miss and recompute. Filesystem errors and
quota eviction can still lose disposable cache data; this is not a redundant database.

In `TieredStore`, RAM entries are populated only by explicit prefetch; foreground disk gets do not implicitly
admit the entire snapshot. This avoids polluting a small pinned pool with one-off restores.
Disk LRU order is process-local; after restart, initial ordering follows directory mtimes.
Snapshots larger than the configured disk budget are rejected. Reopen a store with a staging
chunk size at least as large as the largest chunk used to write its current entries.

## Transfer correctness

In `TieredStore`, one coordinator owns staging buffers and the copy stream; a single read worker can fill the next
free slot while the coordinator checks and transfers the current chunk. In a restore, a slot cannot be overwritten
until its previous H2D event completes. In a store, a D2H event must complete before CPU hashing
or disk writes read its buffer. Exceptional paths drain all outstanding events before buffers
are recycled. Caller stream dependencies are captured when submitting input/output tensors.
An asynchronous API future completes only after payload transfer and validation finish.
Pending disk reads are also drained on error before buffers or file descriptors are reused.
With one staging slot, read/checksum pipelining falls back to serial operation. Use
`pipeline_reads=False` or the benchmark's `--no-pipeline-reads` for the controlled serial path.

Direct I/O is explicit and Linux-specific. It bypasses the filesystem page cache for payload
reads, not drive-controller caches. Buffered mode may consume additional kernel page cache
outside the explicit retained-RAM budget. The pinned allocator may retain high-water memory.

## Limits and extension points

- A single cache-root owner, one coordinator, and one read worker: no cross-process deduplication or RPC server.
- Full snapshots, not paged prefix suffix-sharing or append-only token deltas.
- Shape-uniform retained-full-history Hugging Face adapter only; no static, rolling-eviction,
  hybrid, or RWKV state adapter. Local attention masks without KV eviction can be compatible.
- Source packing and destination allocation are explicit; transport-only timings exclude them.
- No driver changes, service replacement, global cache dropping, unsafe unpickling, or root requirement.
- No encryption/authentication or multi-tenant access-control server. Use an owner-private directory;
  include tenant identity in keys, and do not share a root across untrusted users.
- No compression/quantization by default. A transport-saving codec must justify its encode/decode
  costs and, if lossy, task-quality impact.
- The original `TieredStore` coordinator remains FIFO. Its durable put retains
  GPU input until the disk operation completes. This API is unchanged for compatibility.
- v0.3 adds opt-in [RAM-first write-back](writeback.md): two milestones, bounded
  dirty RAM/archive/read admission, independent foreground RAM copies, and
  non-preemptive disk priorities with a bounded starvation guard. A cold read
  may still wait behind an in-flight archive. Full RAM residency costs capacity;
  release does not mean durable, nor does priority create storage bandwidth.

The most useful next integration is an existing inference scheduler's upcoming-request signal,
plus empirical restore/recompute admission curves. That is more defensible than claiming a new
storage engine alone solves the RAM↔VRAM bandwidth gap.
