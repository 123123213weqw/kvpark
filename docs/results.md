# RTX 4090 D measurements

Measured on 2026-10-22 (Asia/Shanghai). All numbers below come from committed JSON in
[`results/4090d`](../results/4090d/). Medians are not guarantees or production p99 estimates.

## Current v0.3 result: release GPU state before disk finishes

A **1 GiB random-byte synthetic archive** runs while three requests, arriving
50 ms apart, restore the same real TinyStories KV (2095104 bytes, about 2 MiB)
and generate eight tokens each. There are three repeats per mode/disk, alternating
mode order. Archive columns are medians of three trials; TTFT is the median of
nine actual arrivals, including store and single-decoder queue waits.

| Backing/mode | Input reusable (ms) | Observed GPU allocation release (ms) | Durable archive (ms) | Queue-inclusive model TTFT (ms) |
| --- | ---: | ---: | ---: | ---: |
| HDD, FIFO durable | 10197.59 | 10198.07 | 10197.41 | 10288.97 |
| HDD, RAM-first | 98.08 | 98.37 | 10615.70 | 156.69 |
| SATA SSD, FIFO durable | 4052.06 | 4052.51 | 4051.89 | 4147.14 |
| SATA SSD, RAM-first | 97.45 | 97.75 | 4389.68 | 150.85 |

The allocated-memory watcher requires at least 90% of the archive allocation to
become unallocated; it does not claim the CUDA allocator returns reserved memory
to the driver. See raw before/after byte counts. RAM-first's full-buffer allocation
is inside its release timing. The retained/reserved host pool is bounded to about
1.002 GiB for this workload, plus the approximately 16 MiB disk staging ring;
allocator high-water and buffered-write page cache are additional.

The independent hot transfer lane's queue-inclusive transport median is
**0.88 ms (HDD archive)** / **0.91 ms (SATA archive)** for the small real KV, but
actual queue-inclusive model TTFT remains **156.69 / 150.85 ms**. Reporting only
that sub-millisecond memcpy would hide decoder waiting/execution. All 18 RAM-first
resumes finish before their background archive is durable. All 36 resumes across
both modes/disks match the original cache's bytes, next logits (max difference 0),
and greedy continuation. Each random archive passes SHA-256 on a forced cold read
outside the timed foreground phase.

**About 0.10 seconds means GPU input release, not disk durability.** HDD persistence
still takes about 10.62 seconds; SATA about 4.39 seconds with RAM-first. Background
write bandwidth does not improve, and cold reads cannot preempt an in-flight write.
The comparison is against this lab's legacy FIFO queue, not against LMCache or a
production engine. It is not a 1 GiB real-model KV, p99, HTTP concurrency, or sustained
load result. Existing services remain running; shared-node variability applies.

Raw traces: [`tinystories-mixed-hdd.json`](../results/4090d/tinystories-mixed-hdd.json),
[`tinystories-mixed-ssd.json`](../results/4090d/tinystories-mixed-ssd.json).
See [write-back ownership and reproduction](writeback.md).

## v0.2 restore result: validated read/checksum pipelining

| 1-GiB restore path | Median | Preparing the pinned-RAM hit |
| --- | ---: | ---: |
| HDD direct → GPU, CRC32 enabled | 9106.82 ms | 10234.42 ms |
| SATA SSD direct → GPU, CRC32 enabled | 2142.57 ms | 3153.60 ms |
| Prepared pinned RAM → GPU | 100.67–101.73 ms | Already paid above |

Raw sources: [`hdd-v02-pipelined.json`](../results/4090d/hdd-v02-pipelined.json),
[`ssd-v02-pipelined.json`](../results/4090d/ssd-v02-pipelined.json).
RAM-hit throughput is about 10.6 GB/s. Preparing a RAM hit is not free, and the HDD
cannot provide an unpredictable 1-GiB cold restore in 100 ms.

The v0.2 controlled SSD comparison keeps validation enabled in **both** modes:

| Payload | Serial disk read/checksum | Overlapped disk read/checksum |
| --- | ---: | ---: |
| 256 MiB | 787.50 ms | 543.45 ms |
| 1 GiB | 3146.42 ms | 2142.57 ms |

The 1-GiB case reduces restore latency by about **31.9%**. Rather than removing
checksums, one bounded read worker reads the next free slot while the coordinator
validates/transfers the current chunk. Outstanding reads and CUDA copies are drained
even on error. The HDD shows no material improvement in this small shared-node sample;
its backing-device service time still dominates.

The remaining transport tables below preserve the initial v0.1 baselines and tuning
sweep. They are historical observations, not claims that the current default is serial.

## Hardware and experiment conditions

- GPU: NVIDIA GeForce RTX 4090 D, 24 GB GDDR6X; node reports PCIe Gen3 x16.
- About 125 GiB OS-visible system RAM.
- HDD: Seagate ST4000NM0035-1V4107, 4 TB class, rotational.
- SSD: Intel SSDSC2KB480G7, SATA, 480 GB class. **Not NVMe.**
- PyTorch 2.14.1+cu126; CUDA runtime 12.6; NVIDIA driver 580.178.04; Python 3.12.3.
- A separate existing GPU workload remained running. Results are shared-node observations,
  not an exclusive-GPU hardware ceiling.
- Direct Linux O_DIRECT payload reads, CRC32 validation enabled unless explicitly noted.
- Two 8 MiB staging buffers; five measured repeats per main case; source/output allocation
  outside timed restores. Each transport case passed byte equality.

## v0.1 baseline: exact-byte transport

| Payload | HDD direct → GPU | SATA SSD direct → GPU | Pinned-RAM hit → GPU |
| --- | ---: | ---: | ---: |
| 16 MiB | 166.58 ms | 49.66 ms | 2.60–2.65 ms |
| 256 MiB | 2291.34 ms | 763.36 ms | 26.59–26.69 ms |
| 1 GiB | 9148.08 ms | 3056.91 ms | 100.52–101.16 ms |

For 1 GiB, the measured RAM-hit path reaches about 10.6–10.7 GB/s, close to this
node's pinned-copy baseline. That is roughly 91× less **restore-path** latency than its
HDD cold path. It is not a 91× LLM-throughput or user-visible end-to-end improvement.

Preparing that 1 GiB RAM hit took **10167.98 ms from HDD**, or **4118.01 ms from SATA SSD**,
including pinned allocation and checksum processing. Those costs must be supplied as
look-ahead slack or counted on the request's critical path. They are not free.

## v0.1 checksum-cost diagnosis

On this CPU/SATA combination, disabling read-side CRC32 reduced a 1 GiB SSD restore from
3056.91 ms to 2143.58 ms. That unchecked path is explicitly labeled in
[`ssd-no-read-checksum.json`](../results/4090d/ssd-no-read-checksum.json); it is not the
default verified path. Measured transfers still passed post-run byte equality. Do not silently
compare unchecked results with validated cache implementations.
The current v0.2 path achieves approximately this SSD I/O rate **with** CRC32 enabled
by overlapping reads and validation; unchecked results are not its headline performance.

## Staging sweep: 256 MiB SATA direct reads

| Chunk | Slots | Verified restore median |
| --- | ---: | ---: |
| 1 MiB | 1 | 1222.87 ms |
| 1 MiB | 2 | 1191.88 ms |
| 8 MiB | 1 | 791.38 ms |
| 8 MiB | 2 | 764.00 ms |
| 32 MiB | 1 | 763.48 ms |
| 32 MiB | 2 | 759.71 ms |

Eight MiB × two is a reasonable conservative default here: 32 MiB × two consumes
four times the staging capacity for less than 1% observed improvement. The small number
of repeats and shared workload do not establish statistical significance.

## Baseline attention correctness

The committed tiny-Llama CUDA test restores an actual full-attention KV cache, not just
an arbitrary transport buffer. The restored bytes and greedy continuation match exactly;
the maximum next-token logit difference is **0.0**. The random model is a correctness check,
not production model-quality evidence.

General-purpose zlib level 1 compressed that 523264-byte FP16 KV sample by only 1.13×,
taking about 32 ms to encode. This does not justify default compression for the fast path.
Pretrained-model and queued-session results are reported separately below; they must not
be conflated with the 1-GiB synthetic transport cases.

## Pretrained-model restore and a known-queue experiment

Model: [`roneneldan/TinyStories-1M`](https://huggingface.co/roneneldan/TinyStories-1M),
commit `77f1b168e219585646439073245fe87e56b3023e`, FP16 on the actual 4090 D.
Its 1023-token snapshot is **2095104 bytes**, about 2 MiB, not 1 GiB. This is trained-model
execution validation, **not a large-model serving benchmark**. The optional Qwen recipe was
not GPU-validated; its separate provenance record explicitly says so.

Compared directly with the **original model-produced cache before packing/storage**:

- Restored K/V bytes match exactly.
- Maximum next-token logit difference is **0.0**.
- The greedy continuation matches the original cache.
- Both queue modes match the original cache for all four sessions, each generating 64 tokens.

In the small single-prefix experiment, median prefix recomputation is **10.84 ms**, cold
SATA restore **11.69 ms**, and prepared-RAM restore **1.05 ms**. Thus cold restore does
**not** beat recomputation for this small prefix. The admission policy must be allowed to skip it.
See [`tinystories-cuda.json`](../results/4090d/tinystories-cuda.json).

The queue is known in advance; all four requests arrive at each mode's measured start.
The next cache is prefetched while the current session generates. Initial prefetch is counted.

| Backing/mode | Median service TTFT | Median arrival/queue-inclusive TTFT | Total wall time |
| --- | ---: | ---: | ---: |
| HDD, on-demand | 42.00 ms | 835.55 ms | 2156.99 ms |
| HDD, look-ahead | 11.05 ms | 800.22 ms | 2083.12 ms |
| SATA, on-demand | 19.12 ms | 977.55 ms | 2556.28 ms |
| SATA, look-ahead | 12.49 ms | 977.21 ms | 2456.32 ms |

The HDD service-only median improves substantially, but total wall time improves by only
about **3.4%**. In the SATA trace, median queue-inclusive TTFT is almost unchanged despite
better service-only latency. These four-request shared-node samples are feasibility evidence,
not p99 estimates. Cross-tier wall-time differences also include background GPU/CPU activity;
they do not establish that SATA is slower than HDD. Small payloads may fit device-controller caches.

Raw traces: [`tinystories-queued-hdd.json`](../results/4090d/tinystories-queued-hdd.json),
[`tinystories-queued-ssd.json`](../results/4090d/tinystories-queued-ssd.json).
The pretrained FP16 KV sample compresses only **1.086×** with zlib level 1 and takes
about **83.78 ms** to encode. Default compression would not help this measured fast path.

## Regression and reproducibility

**80 tests passed on the actual CUDA node**, with no skips, errors, or failures.
These include CPU and GPU cases, not 80 GPU-only tests. Local macOS CPU testing
passes 74 and skips 6 hardware/platform-specific cases.
[`validation-summary-v03.json`](../results/4090d/validation-summary-v03.json) records a matching
SHA-256 of the tested Python source/tests plus package configuration, runtime versions,
and sanitized test counts. No hostnames, private paths, or prompts are included.

Tests cover direct I/O tails, exact dtypes, quotas, immutable keys, corrupt/truncated data,
partial reads, failed commits, exclusive ownership, inference-mode destinations, GPU ring
reuse and stream dependencies, actual read/checksum overlap, error-path read draining,
unsupported cache rejection, offline pinned-commit identity, and original-cache model logits.
New cases cover early GPU-input reference release, hot reads during blocked writes,
RAM/pending/read admission, non-evictable dirty/failed entries, retries (including
post-rename directory-fsync failure), bounded verification, priority/fairness,
concurrent close, and storage cleanup even when Futures/callbacks survive.
The historical v0.2 [54-test summary](../results/4090d/validation-summary.json) is retained.

## Implication

The defensible low-cost design is **large cheap disk capacity + a small prepared pinned-RAM
working set + active GPU residency**. The measured HDD cannot support an unpredictable
sub-100-ms 1-GiB cold restore. Scheduler-driven prefetch, cache-size reduction, and
restore-versus-recompute admission matter more than a faster memcpy API alone.

See [deployment plan](deployment-plan.md) and [measurement rules](methodology.md).
