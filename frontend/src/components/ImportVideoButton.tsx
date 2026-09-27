/** 导入视频按钮：优先调起 Windows 原生选择框（多选文件 / 选文件夹，均不复制文件）。
 *
 * - mode="files"：选多个视频文件；
 * - mode="folder"：选一个文件夹，后端递归展开其中的视频。
 *
 * 如果后端跑在非 Windows / 无桌面环境（接口返回 501），文件模式回退到浏览器
 * <input type="file" multiple> 上传；文件夹模式回退到 webkitdirectory 上传。
 */

import { useRef, useState } from 'react'
import { FileVideo, FolderOpen } from 'lucide-react'
import { ApiError, api } from '../lib/api'
import { Button } from './ui'
import { useStore } from '../store/useStore'
import { useT } from '../i18n/useT'

type Props = {
  mode?: 'files' | 'folder'
  label?: string
  variant?: 'primary' | 'ghost' | 'outline' | 'subtle'
  size?: 'sm' | 'md' | 'lg' | 'icon'
  className?: string
  /** 指定工程：会先打开它再导入（工程库卡片用得到） */
  projectId?: string
  /** 导入成功后的回调（例如关闭弹窗、跳转剪辑台） */
  onDone?: () => void
}

/** 与后端 `main.VIDEO_EXTS` 保持一致的视频后缀集合。 */
const VIDEO_EXTS = new Set([
  '.mp4', '.mov', '.mkv', '.avi', '.flv', '.wmv', '.m4v', '.webm', '.ts', '.mpg', '.mpeg', '.m2ts', '.3gp',
])

function isVideoFile(f: File): boolean {
  const i = f.name.lastIndexOf('.')
  return i >= 0 && VIDEO_EXTS.has(f.name.slice(i).toLowerCase())
}

export default function ImportVideoButton({
  mode = 'files',
  label,
  variant = 'primary',
  size = 'sm',
  className,
  projectId,
  onDone,
}: Props) {
  const importMedia = useStore((s) => s.importMedia)
  const importFiles = useStore((s) => s.importFiles)
  const openProject = useStore((s) => s.openProject)
  const toast = useStore((s) => s.toast)
  const t = useT()
  const fileRef = useRef<HTMLInputElement>(null)
  const dirRef = useRef<HTMLInputElement>(null)
  const [picking, setPicking] = useState(false)

  const text = label ?? (mode === 'folder' ? t('import.selectFolder') : t('import.selectVideo'))

  async function pick() {
    if (projectId) await openProject(projectId)
    if (!useStore.getState().project) {
      toast({ kind: 'warn', title: t('import.needProject') })
      return
    }
    setPicking(true)
    try {
      const res = mode === 'folder' ? await api.pickFolder() : await api.pickVideos()
      if (!res.cancelled && res.paths.length) {
        await importMedia(res.paths)
        onDone?.()
      }
    } catch (e) {
      // 原生选择框不可用（非 Windows / 无 tkinter）：退回浏览器上传
      if (e instanceof ApiError && (e.status === 501 || e.status === 404)) {
        ;(mode === 'folder' ? dirRef : fileRef).current?.click()
      } else {
        toast({ kind: 'error', title: t('import.openPickerFailed'), detail: String(e) })
      }
    } finally {
      setPicking(false)
    }
  }

  async function upload(files: File[]) {
    // 文件夹兜底上传时浏览器会把目录里的所有文件都塞进来（还有隐藏文件），
    // 先按后端认可的视频后缀过滤，别让一堆非视频白白报失败。
    const videos = files.filter(isVideoFile)
    const skipped = files.length - videos.length
    if (skipped) toast({ kind: 'warn', title: t('import.nonVideoSkipped', { n: skipped }) })
    if (!videos.length) {
      if (!skipped) toast({ kind: 'warn', title: t('import.noVideoFiles') })
      return
    }
    // 浏览器上传会丢掉目录，同名文件会变成重复素材；按文件名（含已在工程里的）去重。
    const project = useStore.getState().project
    const existing = new Set((project?.media ?? []).map((m) => m.name.toLowerCase()))
    const seen = new Set<string>()
    const kept: File[] = []
    let dup = 0
    for (const f of videos) {
      const key = f.name.toLowerCase()
      if (seen.has(key) || existing.has(key)) {
        dup += 1
        continue
      }
      seen.add(key)
      kept.push(f)
    }
    if (dup) toast({ kind: 'warn', title: t('import.duplicatesSkipped', { n: dup }) })
    if (!kept.length) return
    await importFiles(kept)
    onDone?.()
  }

  return (
    <>
      <Button variant={variant} size={size} className={className} loading={picking} onClick={pick} title={text || t('import.importVideo')}>
        {mode === 'folder' ? <FolderOpen size={13} /> : <FileVideo size={13} />}
        {text}
      </Button>
      <input
        ref={fileRef}
        type="file"
        accept="video/*,.mp4,.mov,.mkv,.avi,.flv,.wmv,.m4v,.webm,.ts,.mpg,.mpeg,.m2ts,.3gp"
        multiple
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? [])
          e.target.value = ''
          upload(files)
        }}
      />
      <input
        ref={dirRef}
        type="file"
        multiple
        // @ts-expect-error webkitdirectory 是非标准属性，浏览器支持但 React 类型里没有
        webkitdirectory=""
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? [])
          e.target.value = ''
          upload(files)
        }}
      />
    </>
  )
}
