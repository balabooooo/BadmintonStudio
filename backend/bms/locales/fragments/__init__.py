"""Backend i18n catalog fragments.

Each module exports ``FRAG: list[tuple[str, str, str]]`` as
``(key, 中文, English)``. ``locales/zh.py`` and ``locales/en.py`` merge these
with their built-in base messages.
"""

from __future__ import annotations

from .main_msgs import FRAG as _main
from .core_msgs import FRAG as _core
from .render_msgs import FRAG as _render
from .analysis_msgs import FRAG as _analysis
from .track_msgs import FRAG as _track
from .court_msgs import FRAG as _court

FRAGMENTS: list[tuple[str, str, str]] = [
    *_main,
    *_core,
    *_render,
    *_analysis,
    *_track,
    *_court,
]


def zh_messages() -> dict[str, str]:
    return {k: z for k, z, _ in FRAGMENTS}


def en_messages() -> dict[str, str]:
    return {k: e for k, _, e in FRAGMENTS}
