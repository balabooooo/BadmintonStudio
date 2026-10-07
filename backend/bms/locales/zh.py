"""中文消息目录（默认语言）。"""

from .fragments import zh_messages

_BASE: dict[str, str] = {
    # ---------------------------------------------------------------- 应用
    "app.name": "羽毛球智能剪辑台",

    # ---------------------------------------------------------------- 任务
    "job.cancelled": "已取消",
    "job.done": "完成（{seconds:.1f}s）",
    "job.not_found": "任务不存在",
    "job.queued.prepare": "排队等待其他素材…",
    "job.queued.analyze": "排队等待其他分析…",
    "job.title.prepare": "准备素材 {name}",
    "job.title.analyze": "分析 {name}",
    "job.title.export": "导出 {name}",
    "job.title.export_n": "导出 {count} 段",
    "job.title.optimize": "优化切分参数 {name}",
    "job.title.cache_clear": "清理缓存",
}

MESSAGES: dict[str, str] = {**_BASE, **zh_messages()}
