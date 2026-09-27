"""i18n fragment: main.py. [(key, 中文, English), ...]"""

FRAG: list[tuple[str, str, str]] = [
    # HTTP / id validation
    ("api.asset_not_found", "资源不存在", "Asset not found"),
    ("api.path_not_allowed", "该路径不在允许访问的范围内", "This path is outside the allowed access scope"),
    ("api.origin_denied", "已拒绝来自其它来源的请求", "Request from a non-local origin was rejected"),
    ("api.bad_request", "请求参数非法：{error}", "Invalid request parameters: {error}"),
    ("api.rally_patch_unknown", "不支持的回合字段：{field}", "Unsupported rally field: {field}"),
    ("api.internal_error", "服务内部错误", "Internal server error"),
    ("api.not_found", "未找到", "Not found"),
    ("api.id_invalid", "{what} 非法", "Invalid {what}"),
    ("api.project_id", "工程 id", "project id"),
    ("api.media_id", "素材 id", "media id"),
    ("api.project_not_found", "工程不存在", "Project not found"),
    ("api.file_not_found", "文件不存在", "File not found"),
    ("api.missing_paths", "缺少 path / paths", "Missing path / paths"),
    ("api.missing_ids", "缺少 ids", "Missing ids"),
    ("api.missing_media_id", "缺少 media_id", "Missing media_id"),
    ("api.missing_timeline", "缺少 timeline", "Missing timeline"),

    # Project
    ("project.untitled", "未命名工程", "Untitled project"),

    # Media
    ("media.no_video_in_dir", "目录中没有视频文件", "No video files in the directory"),
    ("media.unrecognized", "无法识别的媒体文件：{type}", "Unrecognized media file: {type}"),
    ("media.not_found", "素材不存在", "Media not found"),
    ("media.no_audio", "尚未提取音轨", "Audio track not extracted yet"),
    ("media.poster_failed", "无法生成封面", "Failed to generate poster"),
    ("media.sprite_failed", "缩略图生成失败", "Failed to generate thumbnails"),

    # Native file dialogs
    ("dialog.pick_videos", "选择视频素材（可多选）", "Select video files (multiple)"),
    ("dialog.filter_videos", "视频文件", "Video files"),
    ("dialog.filter_all", "所有文件", "All files"),
    ("dialog.pick_video_folder", "选择包含视频的文件夹", "Select a folder containing videos"),
    ("dialog.pick_export_folder", "选择导出文件夹", "Select export folder"),
    ("dialog.unsupported", "当前系统不支持原生选择框", "Native file dialogs are not supported on this system"),
    ("dialog.open_failed", "无法打开系统选择框：{type}: {error}", "Failed to open the system dialog: {type}: {error}"),

    # Prepare job messages
    ("prepare.proxy", "生成代理视频", "Generating proxy video"),
    ("prepare.audio", "提取音轨", "Extracting audio track"),
    ("prepare.poster", "生成封面", "Generating poster"),

    # Analysis
    ("analysis.failed", "分析失败", "Analysis failed"),
    ("analysis.no_media", "工程里没有可分析的素材", "No analyzable media in the project"),
    ("analysis.not_found", "没有找到可分析的素材", "No analyzable media found"),
    ("analysis.missing_result", "还没有分析结果，请先运行一次 AI 分析", "No analysis result yet; run an AI analysis first"),
    ("analysis.unknown_weights", "未知的评分口径：{value}", "Unknown scoring preset: {value}"),

    # Timeline
    ("timeline.clip_label", "#{index} {score:.0f}分", "#{index} {score:.0f}"),

    # Export presets
    ("export.preset.yt1080p.name", "横屏 1080p · 高画质", "Landscape 1080p · High quality"),
    ("export.preset.yt4k.name", "横屏 4K · 高画质", "Landscape 4K · High quality"),
    ("export.preset.yt720p.name", "横屏 720p · 体积小", "Landscape 720p · Small size"),
    ("export.preset.vertical.name", "竖屏 1080x1920 · 抖音/小红书", "Portrait 1080x1920 · TikTok/RED"),
    ("export.preset.square.name", "方形 1080x1080 · 朋友圈", "Square 1080x1080 · Social"),
    ("export.preset.hevc4k.name", "横屏 4K · HEVC 省空间", "Landscape 4K · HEVC space-saving"),
    ("export.preset.draft.name", "快速预览 720p", "Quick preview 720p"),

    # Export
    ("export.unknown_mode", "未知的导出方式：{mode}", "Unknown export mode: {mode}"),
    ("export.empty_timeline", "时间线为空，先执行自动剪辑", "Timeline is empty; run auto-cut first"),
    ("export.dir_absolute", "导出目录必须是绝对路径", "Export directory must be an absolute path"),
    ("export.dir_unusable", "导出目录不可用：{error}", "Export directory is not usable: {error}"),
    ("export.reveal_unsupported", "当前系统不支持打开资源管理器", "Opening the file explorer is not supported on this system"),
    ("export.reveal_failed", "无法打开资源管理器：{error}", "Failed to open the file explorer: {error}"),

    # Rally
    ("rally.not_found", "回合不存在", "Rally not found"),

    # Cache
    ("cache.busy", "还有任务在运行，请等任务结束后再清理缓存", "Jobs are still running; wait until they finish before clearing the cache"),
    ("cache.unknown_target", "未知的缓存清理目标：{target}", "Unknown cache clear target: {target}"),

    # Static frontend placeholder
    ("frontend.not_built", "前端尚未构建。开发时请运行 `npm run dev`（Vite，默认 http://127.0.0.1:5273）；构建请运行 `npm run build`（输出到 frontend/dist）。", "Frontend not built. For development run `npm run dev` (Vite, default http://127.0.0.1:5273); to build run `npm run build` (output to frontend/dist)."),
]
