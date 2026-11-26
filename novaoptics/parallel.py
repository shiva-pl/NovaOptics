"""Run independent optimizations concurrently.

JAX releases the GIL while it executes compiled code, and a single merit Jacobian keeps
only part of a multi-core CPU busy, so independent optimizations (architectures, snap
candidates, stop positions, bending alternatives, multi-starts) overlap well in
threads: on 4 cores, 2 threads ran 1.7x and 4 threads 2.1x faster than sequential.
(Vectorizing them with vmap instead gave no gain on the CPU.)

NOVAOPTICS_THREADS sets the worker count (default: number of CPUs; 1 disables).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor


def workers(n_items: int) -> int:
    env = os.environ.get("NOVAOPTICS_THREADS")
    n = int(env) if env else (os.cpu_count() or 1)
    return max(1, min(n, n_items))


def pmap(fn, items):
    """[fn(x) for x in items], concurrently; results keep the input order."""
    items = list(items)
    w = workers(len(items))
    if w == 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(w) as ex:
        return list(ex.map(fn, items))
