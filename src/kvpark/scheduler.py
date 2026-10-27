"""Single-owner disk scheduling; priorities apply only at operation boundaries."""

import heapq
import itertools
import threading
from concurrent.futures import Future


class PriorityWorker:
    """A draining, non-preemptive worker. Lower numbers run first, FIFO on ties.

    Explicit close is required for durability; the daemon must not keep a failed
    application alive forever. shutdown(wait=True) drains all accepted jobs.
    """

    def __init__(self, *, thread_name_prefix="kv-disk"):
        self._condition = threading.Condition()
        self._queue = []
        self._sequence = itertools.count()
        self._closed = False
        self._burst = 0
        self._thread = threading.Thread(target=self._run, name=thread_name_prefix, daemon=True)
        self._thread.start()

    def submit(self, function, *args, priority=0):
        future = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("worker is closed")
            heapq.heappush(self._queue, (priority, next(self._sequence), future, function, args))
            self._condition.notify()
        return future

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._closed)
                if not self._queue:
                    return
                # Bound priority starvation: after eight ordinary selections,
                # the oldest accepted operation gets one turn. Never preempt I/O.
                if self._burst == 8:
                    index = min(range(len(self._queue)), key=lambda i: self._queue[i][1])
                    _, _, future, function, args = self._queue.pop(index)
                    heapq.heapify(self._queue)
                    self._burst = 0
                else:
                    _, _, future, function, args = heapq.heappop(self._queue)
                    self._burst += 1
            if future.set_running_or_notify_cancel():
                try:
                    result = function(*args)
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(result)
            # Do not retain the last completed job's tensors while idle.
            future = function = args = result = None

    def shutdown(self, wait=True):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if wait:
            self._thread.join()
