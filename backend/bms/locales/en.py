"""English message catalog."""

from .fragments import en_messages

_BASE: dict[str, str] = {
    # ---------------------------------------------------------------- App
    "app.name": "Badminton Studio",

    # ---------------------------------------------------------------- Jobs
    "job.cancelled": "Cancelled",
    "job.done": "Done ({seconds:.1f}s)",
    "job.not_found": "Job not found",
    "job.queued.prepare": "Waiting for other media…",
    "job.queued.analyze": "Waiting for other analyses…",
    "job.title.prepare": "Prepare {name}",
    "job.title.analyze": "Analyze {name}",
    "job.title.export": "Export {name}",
    "job.title.export_n": "Export {count} clips",
    "job.title.optimize": "Optimize segmentation {name}",
    "job.title.cache_clear": "Clear cache",
}

MESSAGES: dict[str, str] = {**_BASE, **en_messages()}
