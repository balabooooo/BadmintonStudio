/** Selective cache-clear dialog strings: `[key, 中文, English]` tuples. */

export const FRAG: [string, string, string][] = [
  // Dialog chrome
  ['clearcache.title', '清理缓存', 'Clear cache'],
  ['clearcache.subtitle', '勾选要清理的内容；安全项会在使用时自动重建', 'Pick what to remove; safe items are rebuilt automatically when needed'],
  ['clearcache.selectAll', '全选', 'Select all'],
  ['clearcache.deselectAll', '取消全选', 'Deselect all'],
  ['clearcache.selectedTotal', '已选 {size}', 'Selected: {size}'],
  ['clearcache.submit', '立即清理', 'Clear now'],
  ['clearcache.finish', '完成', 'Done'],
  ['clearcache.cancelJob', '取消清理', 'Cancel cleanup'],
  ['clearcache.empty', '没有可清理的内容', 'Nothing to clear'],
  ['clearcache.loadFailed', '加载缓存占用失败', 'Failed to load cache usage'],
  ['clearcache.dangerBadge', '删除后需重新下载', 'Re-download needed'],
  ['clearcache.files', '{count} 个文件', '{count} files'],

  // Confirmations
  ['clearcache.confirmTitle', '确认清理选中的缓存？', 'Clear the selected cache?'],
  ['clearcache.confirmDesc', '将释放约 {size} 磁盘空间。代理视频、缩略图等会在下次使用时自动重建，不影响原始素材和工程。', 'This frees about {size}. Proxies and thumbnails are rebuilt on next use; source videos and projects are untouched.'],
  ['clearcache.confirmModelsTitle', '确认删除模型文件？', 'Delete the model files?'],
  ['clearcache.confirmModelsDesc', '模型文件（约 {size}）将被删除。下次 AI 分析时会自动重新下载，下载完成前分析功能无法使用。', 'Model files ({size}) will be deleted. They are re-downloaded automatically on the next AI analysis; analysis stays unavailable until then.'],

  // Progress / results
  ['clearcache.running', '正在清理，请勿关闭应用…', 'Clearing, keep the app open…'],
  ['clearcache.cancelled', '清理已取消', 'Clearing cancelled'],
  ['clearcache.failed', '清理失败', 'Clearing failed'],
  ['clearcache.freedTotal', '已释放 {size}', 'Freed {size}'],
  ['clearcache.removedCount', '删除 {count} 项', '{count} removed'],
  ['clearcache.failedCount', '{count} 项被占用未删除', '{count} locked, skipped'],
  ['clearcache.failHint', '部分文件正在被其它程序占用，关闭相关程序后重试即可。', 'Some files are open in another program; close it and try again.'],
  ['clearcache.webviewLocked', '内嵌浏览器缓存被占用：请完全退出并重启应用后再次清理该项目。', 'The built-in browser cache is in use: quit and restart the app, then clear that item again.'],
  ['clearcache.timeoutHint', '部分内容未能在时间限制内完成清理，可稍后重试。', 'Some items could not be cleared within the time limit; try again later.'],

  // Target names
  ['clearcache.t.proxies', '代理视频', 'Proxy videos'],
  ['clearcache.t.thumbs', '缩略图与封面', 'Thumbnails & posters'],
  ['clearcache.t.audio', '提取的音轨', 'Extracted audio'],
  ['clearcache.t.frames', '临时抽帧', 'Temporary frames'],
  ['clearcache.t.ai', 'AI 中间结果', 'AI intermediate results'],
  ['clearcache.t.eval', '调参与评估产物', 'Tuning & evaluation artifacts'],
  ['clearcache.t.logs', '历史日志', 'Historical logs'],
  ['clearcache.t.webview', '内嵌浏览器缓存', 'Built-in browser cache'],
  ['clearcache.t.models', '模型文件', 'Model files'],

  // Target descriptions
  ['clearcache.hint.proxies', '分析用的低分辨率副本，体积较大，可安全重建', 'Low-resolution copies for analysis; large and safely rebuilt'],
  ['clearcache.hint.thumbs', '素材缩略图与视频封面，可自动再生成', 'Clip thumbnails and posters; regenerated automatically'],
  ['clearcache.hint.audio', '分析时从视频提取的音轨', 'Audio tracks extracted during analysis'],
  ['clearcache.hint.frames', '处理过程中产生的临时图片帧', 'Intermediate image frames produced during processing'],
  ['clearcache.hint.ai', '球员检测、姿态、场地标定的计算结果；清除后下次分析需重新计算', 'Detection, pose and court-calibration results; recomputed on the next analysis'],
  ['clearcache.hint.eval', '参数调优记录、评估报告与调试文件', 'Parameter-tuning runs, evaluation reports and debug files'],
  ['clearcache.hint.logs', '保留当天日志，仅清除历史日志文件', "Today's logs are kept; only rotated history is removed"],
  ['clearcache.hint.webview', 'HTTP/GPU/脚本缓存，不含登录状态与偏好设置', 'HTTP/GPU/script caches only; sign-in state and preferences are kept'],
  ['clearcache.hint.models', 'YOLO 与语音模型权重；清除后下次分析自动重新下载', 'YOLO and speech-model weights; re-downloaded automatically on next use'],
]
