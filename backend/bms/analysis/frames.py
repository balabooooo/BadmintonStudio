"""One decode pass shared by the visual analysis stages.

Why
---
``motion`` and ``players`` used to open the proxy video separately, so the same frames were
decoded twice (``pose`` keeps a third, sparse pass -- see below). Decoding is not free: on a
960x540 proxy ``cap.grab()`` still runs the H.264 codec (~0.28 ms per source frame, measured)
and ``cap.retrieve()`` adds BGR conversion (~1.3 ms). :class:`VideoFrameStream` decodes each
source frame once and yields it to every consumer whose sampling grid contains that index,
which removes the duplicated pass without changing what either consumer sees.

Memory
------
Consumption is **synchronous**: the caller's ``for`` body processes the frame before the stream
decodes the next one, so at most one decoded frame plus the caller's own batch buffer is alive
(~1.5 MB for a 960x540 frame, 16 frames per YOLO batch). Nothing is retained for the whole
clip on purpose: a full-video frame cache would cost ``width*height*3*15* duration`` bytes
(roughly 3.7 GB for a one-hour clip).

Scope: what is *not* shared
---------------------------
``pose`` cannot be a consumer here. Its crop boxes are only known after the player timeline
(track merging, per-frame player selection, hole interpolation) has been assembled, i.e. after
this pass ends. Joining would require either buffering the whole clip or cropping from raw
pre-selection detections, which changes the pose input -- so pose keeps its own sparse pass
and is a candidate for the next step (threaded producer/consumer or crop caching).
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np


class VideoFrameStream:
    """Sequential decode over an already-open capture, yielding frames on a union grid.

    Each consumer is described by a **source-frame stride** (``steps``): a consumer with
    ``step == 2`` consumes frames 0, 2, 4, ... . The stream retrieves a frame only when at
    least one consumer needs it; all other frames are skipped with ``grab()`` only (no BGR
    copy), exactly like the previous per-module loops did for their own grids.

    The caller owns the :class:`cv2.VideoCapture` (opening, metadata probing and error
    messages stay in the stage modules, so their behavior is unchanged).
    """

    def __init__(
        self,
        cap,
        steps: list[int] | tuple[int, ...],
        limit: int = 0,
        cancel: Callable[[], bool] | None = None,
    ) -> None:
        self._cap = cap
        self._steps = [max(1, int(s)) for s in steps] or [1]
        # 0 = no explicit limit (whole capture).
        self._limit = int(limit) if limit and limit > 0 else 10 ** 9
        self._cancel = cancel

    @property
    def steps(self) -> list[int]:
        return list(self._steps)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(source_index, frame)`` for every index needed by any consumer."""
        idx = 0
        steps = self._steps
        cap = self._cap
        while True:
            if self._cancel is not None and self._cancel():
                return
            if not cap.grab():
                return
            # ``read`` matches the previous per-module loops: how many source frames have
            # been grabbed so far (1-based), which is also what ``limit`` counted.
            read = idx + 1
            if read > self._limit:
                return
            if any(idx % s == 0 for s in steps):
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield idx, frame
            idx += 1
