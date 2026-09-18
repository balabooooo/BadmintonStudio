"""后台任务管理：进度上报、取消、结果缓存。

任务在一个工作线程里跑，进度通过回调推给 EventBus，再由 WebSocket 广播给前端。
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable

from .models import JobInfo, now_ms


class Job:
    def __init__(self, kind: str, title: str = "", fn: Callable[["Job"], Any] | None = None) -> None:
        self.info = JobInfo(kind=kind, title=title)
        self._fn = fn
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.on_update: Callable[[JobInfo], None] | None = None

    # -------------------------------------------------- 外部接口
    @property
    def id(self) -> str:
        return self.info.id

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()
        if self.info.status in ("queued", "running"):
            self.set(status="cancelled", stage="cancelled", message="已取消")

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

    # -------------------------------------------------- 运行
    def start(self) -> "Job":
        if self._fn is None:
            return self
        self.info.status = "running"
        self.info.updated_at = now_ms()
        self._thread = threading.Thread(target=self._run, name=f"job-{self.id}", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        t0 = time.time()
        try:
            result = self._fn(self)  # type: ignore[misc]
            if self.cancelled():
                self.set(status="cancelled", stage="cancelled", message="已取消")
            else:
                self.set(status="done", progress=1.0, stage="done",
                         message=f"完成（{time.time() - t0:.1f}s）", result=result)
        except Exception as e:  # noqa: BLE001
            self.set(status="error", stage="error",
                     message=str(e), error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}")

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)


class JobManager:
    """全局任务表。"""

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
        for cb in list(self._subscribers):
            try:
                cb(info)
            except Exception:
                pass

    def submit(self, kind: str, title: str, fn: Callable[[Job], Any]) -> Job:
        job = Job(kind, title, fn)
        job.on_update = self._broadcast
        with self._lock:
            self._jobs[job.id] = job
        # 长会话里任务表会无限增长，每个分析任务的 result 还是完整的分析结果
        # （可能几 MB），必须定期清理已完成的任务。
        self.prune()
        job.start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[JobInfo]:
        with self._lock:
            items = sorted(self._jobs.values(), key=lambda j: j.info.created_at, reverse=True)
        out: list[JobInfo] = []
        for j in items[:limit]:
            info = j.info.model_copy(deep=False)
            # 分析结果可达数 MB，列表里不需要；导出结果要留着好让界面显示成片路径。
            if info.kind != "export":
                info.result = None
            out.append(info)
        return out

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        job.cancel()
        return True

    def prune(self, keep: int = 100) -> None:
        with self._lock:
            if len(self._jobs) <= keep:
                return
            items = sorted(self._jobs.values(), key=lambda j: j.info.created_at)
            for j in items[: len(self._jobs) - keep]:
                if j.info.status in ("done", "error", "cancelled"):
                    self._jobs.pop(j.id, None)


manager = JobManager()
