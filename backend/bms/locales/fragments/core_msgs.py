"""i18n fragment: api + core modules. [(key, 中文, English), ...]"""

FRAG: list[tuple[str, str, str]] = [
    # ---------------------------------------------------------------- api / streaming
    ("api.project_not_found", "工程不存在", "Project not found"),
    ("api.media_not_found", "素材不存在", "Media not found"),
    ("api.file_not_found", "文件不存在", "File not found"),
    ("api.invalid_range", "范围无效", "Invalid range"),
    ("api.preset_missing_ids", "缺少 project_id / media_id", "Missing project_id / media_id"),
    ("api.preset_save_failed", "保存预设失败：{error}", "Failed to save preset: {error}"),
    ("api.preset_not_found", "预设不存在", "Preset not found"),
    # ---------------------------------------------------------------- annotations
    ("annotation.no_analysis", "还没有分析结果，请先运行一次 AI 分析", "No analysis result yet. Run an AI analysis first."),
    ("annotation.no_labels", "还没有人工标注，请先标注几个回合再优化", "No manual annotations yet. Annotate a few rallies before optimizing."),
    ("annotation.overlay_bad_window", "时间窗无效或过长（叠加层窗口最长 {max_window} 秒）", "Invalid or too-long time window (overlay window is at most {max_window} seconds)"),
    ("annotation.optimize_running", "该素材正在优化中，请等待当前优化完成", "This media is already being optimized; wait for the current run to finish."),
    ("annotation.optimize_guard", "优化进行中，暂时不能重新切分回合", "Rally segmentation is locked while optimization is running."),
    ("annotation.bad_params", "参数无效：{key}", "Invalid parameter: {key}"),
    # ---------------------------------------------------------------- optimize job stages
    ("job.optimize_prepare", "准备标注与信号", "Preparing labels and signals"),
    ("job.optimize_segment", "搜索回合切分参数", "Searching segmentation parameters"),
    ("job.optimize_gate", "搜索击球归因门限", "Searching hit-attribution gate"),
    ("job.optimize_sensitivity", "搜索击球灵敏度", "Searching hit sensitivity"),
    ("job.optimize_padding", "校准边界留白", "Calibrating boundary padding"),
    ("job.optimize_weights", "校准信号融合权重", "Calibrating fusion weights"),
    ("job.optimize_done", "优化完成", "Optimization done"),
    # ---------------------------------------------------------------- store
    ("store.invalid_id", "非法 id: {value!r}", "Invalid id: {value!r}"),
    ("project.untitled", "未命名工程", "Untitled project"),
    ("project.copy_name", "{name} 副本", "{name} (copy)"),
    ("timeline.track_default", "视频轨 {n}", "Video track {n}"),
    # ---------------------------------------------------------------- presets
    ("preset.invalid_id", "非法预设 id：{pid!r}", "Invalid preset id: {pid!r}"),
    ("preset.default_name", "预设 {date}", "Preset {date}"),
    # ---------------------------------------------------------------- ffmpeg
    ("ffmpeg.not_found", "找不到 ffmpeg。请将 ffmpeg.exe 放入 tools/ffmpeg/bin，或设置环境变量 BMS_FFMPEG。", "ffmpeg not found. Put ffmpeg.exe in tools/ffmpeg/bin, or set the BMS_FFMPEG environment variable."),
    ("ffmpeg.ffprobe_not_found", "找不到 ffprobe", "ffprobe not found"),
    ("ffmpeg.command_failed", "命令失败 ({code}): {cmd}\n{stderr}", "Command failed ({code}): {cmd}\n{stderr}"),
    ("ffmpeg.probe_failed", "ffprobe 失败: {stderr}", "ffprobe failed: {stderr}"),
]
