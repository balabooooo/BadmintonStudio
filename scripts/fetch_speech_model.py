"""Pre-download a faster-whisper model, used for "voice command scoring".

The model is not committed to version control (same as the YOLO weights); run this
manually when you want to pre-fetch it (otherwise faster-whisper downloads it on first use):

    python scripts/fetch_speech_model.py                 # medium (default)
    python scripts/fetch_speech_model.py --size small

It downloads ``Systran/faster-whisper-<size>`` into ``models/faster-whisper-<size>/``.
Analysis will find it automatically, or you can point the ``BMS_SPEECH_MODEL`` environment
variable at a model directory / size / HuggingFace repo id.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from bms.config import MODELS_DIR, ensure_dirs  # noqa: E402

SIZES = ("tiny", "base", "small", "medium", "large-v3")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="medium", choices=SIZES,
                    help="Whisper 模型规格（默认 medium）")
    ap.add_argument("--repo", default="", help="自定义 HuggingFace 仓库 id（默认 Systran/faster-whisper-<size>）")
    ap.add_argument("--force", action="store_true", help="目标目录已存在时也重新下载")
    args = ap.parse_args()

    ensure_dirs()
    repo_id = args.repo or f"Systran/faster-whisper-{args.size}"
    target = Path(MODELS_DIR) / f"faster-whisper-{args.size}"
    if target.exists() and not args.force:
        print(f"模型已存在：{target}（加 --force 可重新下载）")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # noqa: BLE001
        print(f"缺少 huggingface_hub：{type(e).__name__}: {e}")
        print("运行 pip install huggingface_hub 后重试；或直接 pip install faster-whisper（首次使用会自动下载模型）。")
        return 1

    print(f"下载 {repo_id} -> {target}")
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(target))
    except Exception as e:  # noqa: BLE001
        print(f"下载失败：{type(e).__name__}: {e}")
        print("可以手动下载后放到 models/ 目录，或用 BMS_SPEECH_MODEL 指向已有模型目录。")
        return 1
    print(f"完成：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
