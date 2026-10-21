# Low-incremental-cost deployment plan

## Use the hardware already installed

The measured node has an RTX 4090 D (24 GB GDDR6X), about 125 GiB OS-visible system
RAM, a 4 TB-class spinning HDD, and a 480 GB-class SATA SSD. Its GPU link reports
PCIe Gen3 x16. There is no measured NVMe device. The existing GPU service was left running.

The cheapest first step is **zero new hardware purchase**, not a promise of zero running cost:

| Role | Initial policy | Reason |
| --- | --- | --- |
| GPU | Active decoding KV only; reserve service headroom | Repeated dense attention stays local |
| Pinned RAM | Start with 8–16 GiB retained entries, not the entire machine | Fast, predictable restores without locking all DRAM |
| Staging | Two 8 MiB buffers per store | Measured near the larger-buffer result at lower RAM cost |
| HDD | Cold immutable snapshots; bulk sequential files | Lowest incremental capacity cost on this installed machine |
| SATA SSD | Separate warm backing store for less predictable, latency-sensitive prefixes | Lower measured cold-read cost than HDD |

**Important implementation boundary:** the library supports a chosen HDD or SSD backing root.
It does not yet automatically migrate entries between both disks. A production integration should
use a tier-aware router or an existing serving cache such as LMCache for that orchestration.

Do not allocate 8–16 GiB merely to run the demos: their budgets are sized to the small test case.
Increase the retained pool only after observing reuse and pinned-memory high-water under load.

## When to choose each path

1. **Known queued next request:** begin HDD/SSD → RAM prefetch while the current request generates.
2. **User resumes an idle chat:** if the application supplies a reliable advance signal, prefetch then;
   otherwise compare cold restore with recompute and take the faster path.
3. **Repeated hot prefix:** keep it in pinned RAM, subject to measured benefit per retained byte.
4. **Short cold prefix:** usually recompute if storage cannot beat prefill on the actual model.
5. **No advance signal and strict sub-100-ms cold-restore target for 1 GiB:** the measured disks cannot
   satisfy it. More RAM residency, faster storage/interconnect, or a smaller cache representation is needed.
6. **KV larger than available VRAM during continuous dense decoding:** this snapshot design is not a
   transparent solution. Use explicit active-cache offload/CPU-attention/quantization and measure its
   token-generation penalty separately.

## Capacity arithmetic without invented prices

A 32-layer, 8-KV-head, 128-dimension FP16 cache costs 128 KiB per token. At 8192 tokens,
one snapshot is 1 GiB. One thousand such snapshots need about 1.074 TB of payload storage,
plus metadata, filesystem space, and any redundancy. They do **not** require a terabyte of RAM
if only a small predicted working set needs low-latency restoration.

Suppose a user-supplied disk purchase price is `P` currency units per decimal TB. The capacity
capex assigned to one 1-GiB snapshot is `0.001073741824 × P`. This is not a live market quote
and excludes power, wear, redundancy, RAM staging, GPU compute, and cache invalidation.

Each pause/store writes the snapshot again in the current full-snapshot design. Frequent incremental
chat snapshots can produce substantial write amplification. Append-only sealed token blocks and
prefix-sharing are future work; do not claim those savings for this release.

## Prefetch cannot cure a sustained bandwidth deficit

Let each cold snapshot contain `S` bytes, the resume arrival rate be `R` requests/s,
the effective disk read rate be `B` bytes/s, and the RAM/prefix-reuse hit fraction be `h`.
A necessary long-run storage condition is:

```text
(1 - h) × R × S < B
```

For the measured 1-GiB HDD path (about 9.1 seconds per cold restore), one resume per
second would require roughly **89% or better RAM/reuse hits** just to avoid a disk backlog,
before accounting for durable stores or other disk users. With all misses, its sustainable
rate is only about one such restore every 9.1 seconds. A longer look-ahead queue changes
when that cost is paid, not the underlying throughput limit.

Uniform random access across a huge catalog with a tiny RAM pool will not magically produce
that hit rate. Measure locality and queue-inclusive latency. If predicted reads cannot keep
up, prefer recompute, more warm capacity, a faster backing device, or a smaller validated
cache representation rather than hiding the backlog behind a service-only TTFT statistic.

## What changes were made to the experiment node

- A separate experiment directory and owned temporary cache directories were created.
- Existing Python/PyTorch libraries were reused read-only; test-only Python packages were installed
  into the experiment's own dependency directory, not the running service environment.
- No service was stopped or reconfigured, and no driver/kernel/route/SSH trust setting was changed.
- No global page-cache drop, root-only memory tuning, package-manager upgrade, or Rust compilation ran.
- Public model weights, if used, are kept outside the Git repository and verified against the
  official Hugging Face LFS SHA-256. No private prompts or service state were published.

## Production gate

Before placing this on a serving path: integrate the engine's native cache format and scheduler,
measure actual queue-inclusive TTFT and p95/p99 at representative concurrency, implement cache
miss/recompute fallback and tenant isolation, and validate power/write amplification. The queued
model experiment is a feasibility test, not an HTTP serving benchmark or multi-worker deployment.
