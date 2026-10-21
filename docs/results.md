# RTX 4090 D measurements

Measured on 2026-10-22 (Asia/Shanghai). All numbers below come from committed JSON in
[`results/4090d`](../results/4090d/). Medians are not guarantees or production p99 estimates.

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

## Exact-byte transport: cold disk versus prepared RAM

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

## Checksum cost

On this CPU/SATA combination, disabling read-side CRC32 reduced a 1 GiB SSD restore from
3056.91 ms to 2143.58 ms. That unchecked path is explicitly labeled in
[`ssd-no-read-checksum.json`](../results/4090d/ssd-no-read-checksum.json); it is not the
default verified path. Measured transfers still passed post-run byte equality. Do not silently
compare unchecked results with validated cache implementations.

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

## Attention correctness

The committed tiny-Llama CUDA test restores an actual full-attention KV cache, not just
an arbitrary transport buffer. The restored bytes and greedy continuation match exactly;
the maximum next-token logit difference is **0.0**. The random model is a correctness check,
not production model-quality evidence.

General-purpose zlib level 1 compressed that 523264-byte FP16 KV sample by only 1.13×,
taking about 32 ms to encode. This does not justify default compression for the fast path.
A pretrained-model sample and queued-session experiment are being evaluated separately.

## Implication

The defensible low-cost design is **large cheap disk capacity + a small prepared pinned-RAM
working set + active GPU residency**. The measured HDD cannot support an unpredictable
sub-100-ms 1-GiB cold restore. Scheduler-driven prefetch, cache-size reduction, and
restore-versus-recompute admission matter more than a faster memcpy API alone.

See [deployment plan](deployment-plan.md) and [measurement rules](methodology.md).
