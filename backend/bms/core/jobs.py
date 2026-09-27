"""Background job management: progress reporting, cancellation, result caching.

A job runs in a worker thread; progress is pushed to the EventBus through callbacks and then
broadcast to the frontend over WebSocket.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable

from loguru import logger

from .models import JobInfo, now_ms
from ..i18n import get_lang, set_lang, tr, translate


def public_info(info: JobInfo) -> JobInfo:
    """Job snapshot suitable for broadcasting/listing: export results are kept (the UI needs the
    output path), every other kind has its (possibly multi-MB) analysis result stripped."""
    if info.kind != "export" and info.result is not None:
        return info.model_copy(update={"result": None})
    return info


class Job:
    def __init__(
        self,
        kind: str,
        title: str = "",
        fn: Callable[["Job"], Any] | None = None,
        media_id: str | None = None,
        lang: str | None = None,
    ) -> None:
        self.info = JobInfo(kind=kind, title=title, media_id=media_id)
        self._fn = fn
        #: Language of the submitting request; installed in the worker thread so
        #: deep analysis modules can localize their progress messages.
        self.lang = lang or get_lang()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.on_update: Callable[[JobInfo], None] | None = None

    # -------------------------------------------------- Public interface
    @property
    def id(self) -> str:
        return self.info.id

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()
        if self.info.status in ("queued", "running"):
            # Localize in the job's own language, not whatever thread happens to trigger the cancel.
            self.set(status="cancelled", stage="cancelled", message=translate("job.cancelled", self.lang))

    def progress(self, p: float, stage: str = "", message: str = "") -> None:
        patch: dict[str, Any] = {"progress": float(max(0.0, min(1.0, p)))}
        if stage:
            patch["stage"] = stage
        if message:
            patch["message"] = message
        self.set(**patch)

    def set(self, **kw: Any) -> None:
        with self._lock:
            for k, v in kw.items():
                if hasattr(self.info, k):
                    setattr(self.info, k, v)
            self.info.updated_at = now_ms()
            snapshot = self.info.model_copy(deep=False)
        if self.on_update:
            try:
                self.on_update(snapshot)
            except Exception:
                pass

    # -------------------------------------------------- Execution
    def start(self) -> "Job":
        if self._fn is None:
            return self
        self.info.status = "running"
        self.info.updated_at = now_ms()
        self._thread = threading.Thread(target=self._run, name=f"job-{self.id}", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        set_lang(self.lang)
        t0 = time.time()
        logger.info("job {} start: kind={} media={} title={!r}",
                    self.id, self.info.kind, self.info.media_id, self.info.title)
        try:
            result = self._fn(self)  # type: ignore[misc]
            if self.cancelled():
                self.set(status="cancelled", stage="cancelled", message=tr("job.cancelled"))
                logger.info("job {} cancelled", self.id)
            else:
                self.set(status="done", progress=1.0, stage="done",
                         message=tr("job.done", seconds=time.time() - t0), result=result)
                logger.info("job {} done in {:.1f}s", self.id, time.time() - t0)
        except Exception as e:  # noqa: BLE001
            if self.cancelled():
                # A cancelled job may still raise from deep inside a worker (e.g. ffmpeg killed by
                # the cancel flag in ensure_proxy). Cancellation is the intended outcome, so report
                # it as such instead of surfacing a spurious error to the UI.
                self.set(status="cancelled", stage="cancelled", message=tr("job.cancelled"))
                logger.info("job {} cancelled after error: {}", self.id, e)
                return
            self.set(status="error", stage="error",
                     message=str(e), error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}")
            logger.opt(exception=e).error("job {} failed: {}: {}", self.id, type(e).__name__, e)

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)


class JobManager:
    """Global job table."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[JobInfo], None]] = []

    def subscribe(self, cb: Callable[[JobInfo], None]) -> Callable[[], None]:
        self._subscribers.append(cb)

        def unsub() -> None:
            try:
                self._subscribers.remove(cb)
            except ValueError:
                pass

        return unsub

    def _broadcast(self, info: JobInfo) -> None:
        # Strip the heavy analysis result here (the broadcast path) so every connected client does
        # not get multi-MB payloads; export results are preserved so the UI can show the output path.
        info = public_info(info)
        for cb in list(self._subscribers):
            try:
                cb(info)
            except Exception:
                pass

    def submit(
        self,
        kind: str,
        title: str,
        fn: Callable[[Job], Any],
        media_id: str | None = None,
        lang: str | None = None,
    ) -> Job:
        job = Job(kind, title, fn, media_id=media_id, lang=lang)
        job.on_update = self._broadcast
        with self._lock:
            self._jobs[job.id] = job
        # In long sessions the job table grows without bound; each analysis job's result is the
        # full analysis result (possibly several MB), so completed jobs must be pruned regularly.
        self.prune()
        job.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[JobInfo]:
        with self._lock:
            items = sorted(self._jobs.values(), key=lambda j: j.info.created_at, reverse=True)
        out: list[JobInfo] = []
        for j in items[:limit]:
            out.append(public_info(j.info.model_copy(deep=False)))
        return out

    def active(self) -> list[JobInfo]:
        """All queued/running jobs, regardless of the list limit (for guards like cache clearing)."""
        with self._lock:
            return [
                j.info.model_copy(deep=False)
                for j in self._jobs.values()
                if j.info.status in ("queued", "running")
            ]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return False
        job.cancel()
        return True

    def cancel_for_media(self, media_ids: "set[str] | frozenset[str]") -> int:
        """Cancel all queued/running jobs serving the given media and return the number cancelled.

        Once media is removed from the project, continuing to run ffmpeg for it (generating a
        proxy, etc.) just wastes CPU/GPU.
        """
        if not media_ids:
            return 0
        with self._lock:
            jobs = list(self._jobs.values())
        n = 0
        for job in jobs:
            if job.info.media_id in media_ids and job.info.status in ("queued", "running"):
                job.cancel()
                n += 1
        return n

    def prune(self, keep: int = 100) -> None:
        with self._lock:
            if len(self._jobs) > keep:
                items = sorted(self._jobs.values(), key=lambda j: j.info.created_at)
                for j in items[: len(self._jobs) - keep]:
                    if j.info.status in ("done", "error", "cancelled"):
                        self._jobs.pop(j.id, None)
            # Drop the heavy result from retained finished non-export jobs: the broadcast already
            # strips it, so keeping it only bloats the table (and the default /api/jobs payload).
            for j in self._jobs.values():
                if j.info.kind != "export" and j.info.status in ("done", "error", "cancelled"):
                    j.info.result = None


manager = JobManager()
