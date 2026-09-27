"""Backend localization.

User-facing strings are addressed by stable message keys and rendered through
``tr()`` according to the request/job language. The language is carried in a
``ContextVar`` so that deep analysis modules can call ``tr()`` without threading
a language argument through every function.

Language resolution order (see ``parse_lang``):
- HTTP: middleware reads ``X-BMS-Lang`` first, then ``Accept-Language``.
- WebSocket: handler reads the ``lang`` query parameter.
- Jobs: ``JobManager.submit(..., lang=...)`` captures the submitting request's
  language and ``Job._run`` installs it in the worker thread.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Literal

from .locales import en as _en
from .locales import zh as _zh

Lang = Literal["zh", "en"]
DEFAULT_LANG: Lang = "zh"

_ZH = _zh.MESSAGES
_EN = _en.MESSAGES
_CATALOGS: dict[str, dict[str, str]] = {"zh": _ZH, "en": _EN}

_current: ContextVar[Lang] = ContextVar("bms_lang", default=DEFAULT_LANG)


def normalize_lang(value: str | None) -> Lang:
    if not value:
        return DEFAULT_LANG
    if value.strip().lower().startswith("en"):
        return "en"
    return "zh"


def _best_tag(value: str) -> str:
    """Pick the highest-q tag from an ``Accept-Language`` style header (``en;q=0.8,zh;q=0.9``)."""
    best = ""
    best_q = -1.0
    for piece in value.split(","):
        bits = piece.strip().split(";")
        tag = bits[0].strip()
        if not tag:
            continue
        q = 1.0
        for param in bits[1:]:
            param = param.strip()
            if param.startswith("q="):
                try:
                    q = float(param[2:])
                except ValueError:
                    q = 0.0
        if q > best_q:
            best_q, best = q, tag
    return best


def parse_lang(*sources: str | None) -> Lang:
    """First non-empty source wins; each may be an ``Accept-Language`` list (q-values respected)."""
    for src in sources:
        if not src:
            continue
        tag = _best_tag(src)
        if tag:
            return normalize_lang(tag)
    return DEFAULT_LANG


def get_lang() -> Lang:
    return _current.get()


def set_lang(lang: str | None) -> Lang:
    resolved = normalize_lang(lang)
    _current.set(resolved)
    return resolved


def translate(key: str, lang: str | None = None, **params: object) -> str:
    table = _CATALOGS.get(lang or get_lang(), _ZH)
    text = table.get(key) or _ZH.get(key) or key
    if params:
        try:
            text = text.format(**params)
        except (KeyError, IndexError, ValueError):
            pass
    return text


def tr(key: str, **params: object) -> str:
    """Translate ``key`` in the current language."""
    return translate(key, None, **params)


def app_name(lang: str | None = None) -> str:
    return translate("app.name", lang)
