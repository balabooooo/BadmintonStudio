"""HTTP Range 视频流。

浏览器 ``<video>`` 拖动进度条依赖 206 分段响应，Starlette 的 FileResponse
不支持 Range，所以这里自己实现。
"""

from __future__ import annotations

import mimetypes
import os
import re
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
CHUNK = 1024 * 512


def content_type(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


def range_response(request: Request, path: Path, cache_seconds: int = 0, extra_headers: dict | None = None):
    if not path.is_file():
        raise HTTPException(404, "文件不存在")
    size = path.stat().st_size
    ctype = content_type(path)
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": ctype,
        "Cache-Control": f"public, max-age={cache_seconds}" if cache_seconds else "no-store",
    }
    if extra_headers:
        headers.update(extra_headers)
    rng = request.headers.get("range") or request.headers.get("Range")
    if not rng:
        headers["Content-Length"] = str(size)
        return StreamingResponse(_iter(path, 0, size - 1), status_code=200, headers=headers)

    m = RANGE_RE.match(rng.strip())
    if not m:
        headers["Content-Length"] = str(size)
        return StreamingResponse(_iter(path, 0, size - 1), status_code=200, headers=headers)

    start_s, end_s = m.group(1), m.group(2)
    if start_s == "":
        # 后缀范围：最后 N 字节
        n = int(end_s or 0)
        start = max(0, size - n)
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        raise HTTPException(416, "范围无效", headers={"Content-Range": f"bytes */{size}"})

    length = end - start + 1
    headers["Content-Length"] = str(length)
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(_iter(path, start, end), status_code=206, headers=headers)


def _iter(path: Path, start: int, end: int, chunk: int = CHUNK):
    remaining = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            data = f.read(min(chunk, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


def file_size(path: Path) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0
