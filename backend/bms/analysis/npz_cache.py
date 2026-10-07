"""Process-wide LRU cache for parsed analysis ``.npz`` documents.

The annotation overlay polls an 8-second window around the playhead several times per second
while the user scrubs. Every request used to re-run ``np.load`` + decompress + parse the full
boxes/pose cache (300+ ms for a long match), so dragging the progress bar stalled the server
threadpool and made the UI freeze.

The parsed documents are immutable for callers (the overlay only reads them), so keeping the
last few decoded docs in memory turns repeated window requests into pure Python slicing. Cache
entries are keyed by absolute path and validated against ``(mtime_ns, size)``: a re-analysis
that rewrites the npz invalidates automatically. Failed loads are not cached.

Thread-safety: FastAPI runs sync routes in a worker threadpool and analysis jobs also touch
these loaders; an :class:`RLock` guards the OrderedDict.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from loguru import logger

T = dict  # parsed document type (plain dict; kept loose to avoid cross-module imports)
Loader = Callable[[Path], "dict | None"]


class NpzLruCache:
    """Bounded path -> parsed-document cache with stat-based invalidation."""

    def __init__(self, capacity: int = 4) -> None:
        self._capacity = max(1, int(capacity))
        self._items: "OrderedDict[str, tuple[tuple[int, int], dict]]" = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def _signature(self, p: Path) -> tuple[int, int] | None:
        try:
            st = p.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def get_or_load(self, path: str | Path, loader: Loader) -> "dict | None":
        """Return the parsed document for ``path``, decoding via ``loader`` on miss.

        ``None`` results (missing/corrupt file) are not cached, so a file appearing or being
        repaired later is picked up immediately.
        """
        p = Path(str(path))
        sig = self._signature(p)
        key = str(p)
        with self._lock:
            hit = self._items.get(key)
            if hit is not None and sig is not None and hit[0] == sig:
                self._items.move_to_end(key)
                self.hits += 1
                logger.debug("npz cache hit: {} (hits={} misses={})", p.name, self.hits, self.misses)
                return hit[1]
            if hit is not None:
                # File changed on disk; drop the stale copy before reloading.
                self._items.pop(key, None)
                logger.debug("npz cache invalidated by file change: {}", p.name)

        t0 = time.perf_counter()
        doc = loader(p)
        ms = (time.perf_counter() - t0) * 1000.0
        if doc is None:
            logger.debug("npz cache: loader returned None, not caching {} ({:.1f}ms)", p.name, ms)
            return None
        with self._lock:
            self.misses += 1
            self._items[key] = ((sig or (0, 0)), doc)
            self._items.move_to_end(key)
            while len(self._items) > self._capacity:
                old_key, _old = self._items.popitem(last=False)
                logger.debug("npz cache evict: {}", old_key)
        logger.debug("npz cache miss loaded: {} ({:.1f}ms, hits={} misses={})",
                     p.name, ms, self.hits, self.misses)
        return doc

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.hits = 0
            self.misses = 0


#: Shared overlay cache: boxes + pose docs for the 2 most recently inspected media pairs fit
#: comfortably (capacity 4 leaves headroom for alternating clips).
NPZ_CACHE = NpzLruCache(capacity=4)
