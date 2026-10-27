# kvpark

**Cheap, exact KV snapshot storage with a measured disk → pinned RAM → CUDA
restore path, plus bounded RAM-first offload.** This project targets consumer GPUs and reused/paused prefixes,
not the fiction that a hard drive can replace active GPU memory at no cost.

> Experimental research prototype. RTX 4090 D uses GDDR6X, **not HBM**.
> No direct-storage or end-to-end serving-speed claim is implied by a memcpy benchmark.

**Measured on the target 4090 D:** verified 1-GiB restore takes **9.11 s from HDD**,
**2.14 s from SATA SSD**, or **0.10 s from already prepared pinned RAM**. Preparing
that RAM hit itself takes about 10.23 s from HDD. See [raw-backed results](docs/results.md).

**New in v0.3:** while archiving 1 GiB to HDD, RAM-first makes the GPU input
reusable in **98 ms**, instead of waiting about **10.2 s** for the legacy FIFO put.
Concurrent real TinyStories resumes see median queue-inclusive TTFT **157 ms vs
10.29 s** in this small controlled workload. The actual model KV is about 2 MiB,
not 1 GiB; disk durability still takes **10.62 s**. [Mixed traces and limits](docs/results.md).

## Design

```text
Idle conversation/prefix
        │ immutable, lossless, crash-consistent snapshot
        ▼
Cheap HDD archive or SSD backing store
        │ explicit look-ahead prefetch
        ▼
Bounded pinned-RAM LRU                    cold miss
        │                                │ chunk N+1 read into staging
        │ one asynchronous H2D copy        │ chunk N copied to the GPU
        └─────────────────┬───────────────┘
                          ▼
                    GPU KV tensor
                          │ exact attention + continuation
                          ▼
                    Active generation
```

- Raw, contiguous KV payload bytes; no pickle/object deserialization in the cache format or default quantization.
- Preallocated, 4 KiB-aligned staging buffers protected by CUDA completion events.
- Serialized disk staging/read worker; opt-in independent D2H and hot-H2D lanes.
- Pipelined reads overlap chunk N+1 I/O with chunk N checksum/copy, rather than disabling validation.
- Explicit prefetch into a size-bounded pinned-RAM LRU; oversized entries decline admission.
- Durable write-through or RAM-first background persistence, padded payloads, per-chunk CRC32, JSON metadata, fsync,
  and atomic directory publication. CRC32 detects accidental corruption; it is **not authentication**.
- Linux `O_DIRECT` reads can measure actual backing-device I/O instead of reporting page-cache
  hits as SSD performance. Requested direct I/O never silently falls back.
- Payload-byte disk quota and eviction; complete entries only. Metadata/filesystem overhead is additional.
- Exclusive cache-root ownership prevents conflicting writers. Use one root per process.

**Where it helps:** restarting or switching conversations, prefix reuse, queued request
look-ahead, and freeing VRAM while a session is idle. Keep the actively decoding KV in VRAM
whenever possible. In full dense attention, old tokens remain necessary on every decode step.

## Install and test

Use an isolated environment and install the appropriate PyTorch CUDA wheel for your system first.
The package does not require a custom CUDA extension, kernel module, Rust compiler, root, or Redis.

```bash
python -m pip install -e '.[test,demo]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m ruff check src tests examples scripts
```

The CPU suite runs without CUDA; GPU ring-buffer and stream-dependency tests run when CUDA exists.
The Hugging Face test constructs a tiny random Llama locally; no model download is required.

## API

```python
import torch
from kvpark import TieredStore, cache_key

signature = {
    "model": "my-model",
    "revision": "immutable-model-commit",
    "tokenizer_revision": "immutable-tokenizer-commit",
    "rope": {"theta": 10000},
    "adapter": None,
    "dtype": "float16",
    "layout": "2BHTD-v1",
    "tenant": "my-private-tenant",
}
prefix_ids = [1, 42, 9]
key = cache_key(signature, prefix_ids, tensor_name="layer-0")
kv = torch.randn(2, 1, 8, 3, 128, device="cuda", dtype=torch.float16)

with TieredStore(
    "/data/my-kv-cache", device="cuda:0", ram_bytes=2 * 1024**3,
    disk_bytes=100 * 1024**3, chunk_bytes=8 * 1024**2,
    staging_slots=2, direct_io=True,
) as store:
    if not store.contains(key):
        store.put_async(key, kv).result()  # Do not mutate kv before completion.
    store.prefetch(key).result()           # Ideally scheduled before request admission.
    restored = store.get_async(key).result()
    assert torch.equal(kv.view(torch.uint8), restored.view(torch.uint8))
```

Futures complete **after** CUDA copies finish. Caller-supplied output tensors must not be used
or mutated until completion. Inputs must already be contiguous; packing is explicit and costs time
and potentially extra VRAM. Keys are write-once and must bind the full preceding causal prefix,
model revisions, adapters, position/RoPE semantics, dtype/layout, and tenant. A matching suffix
alone does not permit reuse. Token IDs and prompts are not persisted in manifests.

The constructor's `ram_bytes` bounds the physical storage of retained RAM entries, including
their alignment slack. Persistent staging adds roughly `slots × (chunk_bytes + 4096)` bytes.
PyTorch can retain freed pinned allocations in its allocator; process RSS/pinned high-water
is not guaranteed to shrink after eviction. GPU output tensors are caller/engine-owned.

## RAM-first fast GPU release (v0.3)

`TieredStore.put_async` is still durable/write-through. To stop waiting for disk
before releasing a paused session's VRAM, opt into **`RamFirstStore`**:

```python
from kvpark import CacheBusy, RamFirstStore

with RamFirstStore("/data/my-kv-cache", device="cuda:0", ram_bytes=2 * 1024**3,
                   disk_bytes=100 * 1024**3, max_pending=2, max_reads=8) as store:
    handle = store.offload_async(key, kv)  # May raise CacheBusy; nothing then accepted.
    handle.released.result()             # Exact owned RAM copy, D2H completed.
    del kv                               # Drop all caller aliases to release allocation.
    restored = store.get(key)             # Can read RAM while disk write is running.
    handle.durable.result()              # Only now crash-safe on backing storage.
```

Dirty/failed RAM copies are protected from eviction; admission is bounded by RAM,
pending archives, and outstanding reads. RAM hits use an independent foreground
lane. Cold reads jump queued archives, but cannot interrupt an in-flight disk
write. Failed persistence remains readable in RAM and can be retried explicitly.
**Released is not durable.** See [ownership, failure handling, backpressure, and
mixed-workload reproduction](docs/writeback.md). This needs RAM for a full snapshot;
when unavailable, retain the GPU state, recompute later, or use the slower streamed path.

## Observability

`stats()` reports a fixed-size metric set: per-outcome counters, exact byte
totals, and latency histograms whose quantiles are documented upper bounds. The
schema is stable, so a name is always present even when its code path never ran.
See [docs/observability.md](docs/observability.md) for every field and for the
2.5x quantile bound.

```python
stats = store.stats()
stats["hit_rate"]["read"]                              # None, not 0.0, when idle
stats["metrics"]["latency_ms"]["released_ms"]["max_ms"] # GPU input reusable
stats["metrics"]["latency_ms"]["durable_ms"]["max_ms"]  # crash-safe
```

## Reproduce transport measurements

```bash
kvpark-bench --root /your/hdd/bench --label hdd --device cuda:0 \
  --sizes-mib 16 256 1024 --repeats 5 --direct-io --output results/hdd.json
kvpark-bench --root /your/ssd/bench --label ssd --device cuda:0 \
  --sizes-mib 16 256 1024 --repeats 5 --direct-io --output results/ssd.json
```

These create and remove only a new, owned temporary directory. They do not drop global OS caches,
change driver settings, or stop services. Keep sufficient free GPU memory for source and destination.
Each result contains individual latency samples, effective decimal GB/s, byte-equality checks,
checksum setting, copy baselines, and hardware/software metadata. `ram_hit` is distinct from
`disk_direct`. Buffered reads are explicitly labeled as uncontrolled cache-state measurements.
Prefetch cost is reported, **not omitted and called free**. Allocations are outside timed restores.

To reproduce the longer matrix with an existing Python runtime:

```bash
PYTHON=/path/to/isolated/python HDD_ROOT=/your/hdd/bench SSD_ROOT=/your/ssd/bench \
  RESULT_DIR=results/my-system bash scripts/run_matrix.sh
```

## Model-level validation

```bash
# No downloaded weights: actual Llama attention, random tiny model.
python examples/hf_resume.py --root /your/ssd/bench --device cuda:0 \
  --tokens 512 --direct-io --output results/tiny-llama.json

# Optional public pretrained model; resolves and records its immutable commit.
python examples/hf_resume.py --root /your/ssd/bench --device cuda:0 \
  --model Qwen/Qwen2.5-0.5B-Instruct --tokens 8192 --direct-io \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --output results/qwen-half-b.json
```

The adapter handles **uniform full-retained-history DynamicCache only**. It packs all layers, restores
exact bytes, checks next-token logits, and compares a greedy continuation. It is not a vLLM plugin
or a hybrid/recurrent/sliding-window cache adapter. Packing overhead and durable store time are
reported separately. Pretrained runs use seeded synthetic token IDs, not private user prompts.
Static, sliding/rolling-eviction, and hybrid cache objects are rejected. A model may use local
attention masks while retaining the full K/V history, as in the tested TinyStories GPT-Neo checkpoint.

`examples/queued_resume.py` additionally generates from each restored session while prefetching
the next known queued session. Its service TTFT excludes arrival-queue waiting time; total wall
time includes initial prefetch. This is not a prediction oracle or a production serving benchmark.

The measured pretrained correctness/queue example uses
[TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M), pinned to commit
`77f1b168e219585646439073245fe87e56b3023e`, with `--attn-implementation eager --tokens 1024`.
The Qwen invocation above is an optional larger-model recipe, **not a completed pretrained benchmark**.

## Admission and cost

```bash
# Illustrative inputs, not measured defaults or a hardware price quote.
kvpark-plan --layers 32 --kv-heads 8 --head-dim 128 --tokens 8192 \
  --bits 16 --link-gbps 12 --disk-gbps 0.18 --recompute-ms 500
```

The example cache is exactly 1 GiB. Disk must win against **measured prefix recomputation**, not
just against reloading another way. When a cold restore is slower than recompute, skip it.
Look-ahead can move the disk cost off the critical path, but cannot remove device bandwidth,
capacity, queueing, energy, or storage-wear costs. A user-supplied `--disk-price-per-tb` computes
capacity capex only, excluding power, redundancy, RAM staging, and the GPU.

## Existing systems, scope, and references

This is not a claim that tiered KV storage is novel. For a production serving engine, start by
evaluating existing connectors rather than replacing them with this lab:

- [LMCache](https://docs.lmcache.ai/): production-oriented KV reuse and tiered storage.
- [py-kvcache](https://github.com/t348575/py-kvcache): direct-I/O/io_uring research backend;
  its documented custom vLLM requirements must be checked before deployment.
- [Mooncake](https://github.com/kvcache-ai/Mooncake): distributed KV storage and transport.
- [Hugging Face cache semantics](https://huggingface.co/docs/transformers/cache_explanation):
  per-layer K/V state and causal-prefix reuse.
- [PyTorch pinned-memory guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html):
  asynchronous-copy lifetimes and synchronization.
- [NVIDIA GDS support matrix](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html#support-matrix):
  do not assume a GeForce GPU has a supported direct disk-to-VRAM DMA path. This lab uses CPU staging.

See [architecture](docs/architecture.md), [benchmark methodology](docs/methodology.md),
the [RAM-first write-back design](docs/writeback.md), the [low-cost deployment plan](docs/deployment-plan.md), and [measured results](docs/results.md).
**Do not interpret synthetic transport throughput
as production tokens/s, TTFT, model-quality, or tail-latency evidence.**

## License

Apache-2.0. No private prompts, KV contents, model weights, credentials, SSH configuration,
or other services' source code are included.
