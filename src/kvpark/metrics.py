"""Bounded, thread-safe observability for kvpark.

Design constraints, in priority order:

1. **Fixed memory.** No per-key maps, no unbounded label sets. A cache with a
   million keys must not grow a million counters.
2. **No behaviour change.** Recording never raises into a caller's path and
   never takes a lock that a caller could already hold in the reverse order.
   ``Metrics`` owns a private lock and calls no other code while holding it.
3. **Honest numbers.** Counts and byte totals are exact. Quantiles are *bucket
   upper bounds*: a fixed histogram cannot report an exact quantile, so they
   are reported as ``p50_ms``/``p95_ms``/``p99_ms`` and documented as upper
   bounds. Exact ``min``/``max``/``mean`` are reported beside them so a wrong
   histogram assumption is visible instead of hidden.

Hot-path cost is one uncontended lock acquisition plus integer arithmetic, or
one ``perf_counter_ns()`` call when a latency sample is taken.
"""

from __future__ import annotations

import math
import threading
import time
from bisect import bisect_left

# Fixed bucket edges in milliseconds: a 1-2-5 geometric progression, 25 edges
# plus one overflow slot. The progression is kept clean on purpose, because it
# makes the quantile error bound provable: any reported quantile is at most
# 2.5x the true value (the widest 1-2-5 step is 2.0 -> 5.0). Range spans
# microseconds to a hundred seconds: a pinned-RAM hit is about 0.005 ms and a
# cold 1 GiB HDD archive is about 10600 ms. Exact min/max/mean are reported
# beside the quantiles, so precision never depends on the histogram.
EDGES_MS = (
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
    1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
    1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0, 100000.0,
)

# (reported name, target quantile). The bucket upper bound is reported rather
# than an interpolated position, because the distribution inside a bucket is
# unknown. Values are therefore upper bounds, never underestimates.
QUANTILES = (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))


class Latency:
    """A fixed-size latency histogram in nanoseconds."""

    __slots__ = ("_count", "_total_ns", "_min_ns", "_max_ns", "_buckets")

    def __init__(self) -> None:
        self._count = 0
        self._total_ns = 0
        self._min_ns: int | None = None
        self._max_ns: int | None = None
        self._buckets = [0] * (len(EDGES_MS) + 1)

    def record(self, elapsed_ns: int) -> None:
        if elapsed_ns < 0:
            elapsed_ns = 0
        self._count += 1
        self._total_ns += elapsed_ns
        if self._min_ns is None or elapsed_ns < self._min_ns:
            self._min_ns = elapsed_ns
        if self._max_ns is None or elapsed_ns > self._max_ns:
            self._max_ns = elapsed_ns
        self._buckets[bisect_left(EDGES_MS, elapsed_ns / 1e6)] += 1

    def snapshot(self) -> dict:
        if not self._count:
            out = {"count": 0, "min_ms": None, "max_ms": None, "mean_ms": None}
            out |= {name: None for name, _ in QUANTILES}
            return out
        out = {
            "count": self._count,
            "min_ms": self._min_ns / 1e6,
            "max_ms": self._max_ns / 1e6,
            "mean_ms": self._total_ns / self._count / 1e6,
        }
        for name, target in QUANTILES:
            rank = max(1, math.ceil(self._count * target))
            seen = 0
            bound = self._max_ns / 1e6
            for index, hits in enumerate(self._buckets):
                seen += hits
                if seen >= rank and hits:
                    # Overflow slot reports the exact observed maximum, which is
                    # still an upper bound and cannot understate the truth.
                    bound = EDGES_MS[index] if index < len(EDGES_MS) else self._max_ns / 1e6
                    break
            out[name] = bound
        return out


class _Timer:
    """Context manager recording one latency sample, including on exception."""

    __slots__ = ("_metrics", "_name", "_start")

    def __init__(self, metrics: Metrics, name: str) -> None:
        self._metrics = metrics
        self._name = name
        self._start = 0

    def __enter__(self) -> _Timer:
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *_args) -> bool:
        self._metrics.observe(self._name, time.perf_counter_ns() - self._start)
        return False


class Metrics:
    """Counters plus latency histograms. Bounded regardless of cache size."""

    __slots__ = ("_lock", "_counters", "_latencies")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._latencies: dict[str, Latency] = {}

    # -- recording -----------------------------------------------------
    def add(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe(self, name: str, elapsed_ns: int) -> None:
        with self._lock:
            series = self._latencies.get(name)
            if series is None:
                series = self._latencies[name] = Latency()
            series.record(elapsed_ns)

    def time(self, name: str) -> _Timer:
        return _Timer(self, name)

    # -- inspection ----------------------------------------------------
    def get(self, name: str, default: int = 0) -> int:
        with self._lock:
            return self._counters.get(name, default)

    def counters(self) -> dict:
        with self._lock:
            return dict(self._counters)

    def latencies(self) -> dict:
        with self._lock:
            return {name: series.snapshot() for name, series in self._latencies.items()}

    def snapshot(self, counters=(), latencies=()) -> dict:
        """Return a schema-complete snapshot.

        Names passed in ``counters``/``latencies`` are emitted even when they
        were never incremented, so a dashboard or alert cannot break on a
        missing key just because a code path did not run.
        """
        with self._lock:
            report = {name: 0 for name in counters}
            report.update(self._counters)
            empty = {"count": 0, "min_ms": None, "max_ms": None, "mean_ms": None}
            empty |= {name: None for name, _ in QUANTILES}
            series = {name: dict(empty) for name in latencies}
            series.update(
                {name: value.snapshot() for name, value in self._latencies.items()}
            )
            return {"counters": report, "latency_ms": series}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latencies.clear()


def ratio(numerator: int, denominator: int) -> float | None:
    """Hit-rate helper that refuses to invent a number when nothing happened."""
    if not denominator:
        return None
    return numerator / denominator
