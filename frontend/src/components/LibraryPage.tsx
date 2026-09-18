import { motion } from 'motion/react'
import { useEffect, useMemo, useState } from 'react'
import { FolderPlus, Film, Trash2, Clock, Sparkles, Search, Clapperboard, Copy } from 'lucide-react'
import { api } from '../lib/api'
import { humanDuration, relTime } from '../lib/format'
import { Button, Card, Empty, Modal, Skeleton, useConfirm } from './ui'
import { useStore } from '../store/useStore'

export default function LibraryPage() {
  const projects = useStore((s) => s.projects)
  const refresh = useStore((s) => s.refreshProjects)
  const createProject = useStore((s) => s.createProject)
  const openProject = useStore((s) => s.openProject)
  const deleteProject = useStore((s) => s.deleteProject)
  const booted = useStore((s) => s.booted)
  const toast = useStore((s) => s.toast)
  const confirm = useConfirm()

  const [q, setQ] = useState('')
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [importing, setImporting] = useState(false)
  const [importPath, setImportPath] = useState('')
  const [targetProject, setTargetProject] = useState<string | null>(null)

  useEffect(() => {
    refresh()
  }, [refresh])

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase()
    return s ? projects.filter((p) => p.name.toLowerCase().includes(s)) : projects
  }, [projects, q])

  const stats = useMemo(
    () => ({
      count: projects.length,
      hours: projects.reduce((a, p) => a + p.duration, 0) / 3600,
      rallies: projects.reduce((a, p) => a + p.rally_count, 0),
    }),
    [projects],
  )

  async function doImport() {
    if (!targetProject || !importPath.trim()) return
    const paths = importPath
      .split(/\r?\n|;/)
      .map((s) => s.trim())
      .filter(Boolean)
    setImporting(true)
    try {
      await openProject(targetProject)
      const s = useStore.getState()
      await s.importMedia(paths)
      setTargetProject(null)
      setImportPath('')
    } finally {
      setImporting(false)
    }
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1400px] px-8 py-8">
        {/* 头部 */}
        <div className="mb-8 flex flex-wrap items-end justify-between gap-5">
          <div>
            <motion.h1
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="text-[26px] font-semibold tracking-tight text-white"
            >
              工程库
            </motion.h1>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.08, duration: 0.4 }}
              className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12.5px] text-ink-400"
            >
              <span className="inline-flex items-center gap-1.5">
                <Clapperboard size={13} /> {stats.count} 个工程
              </span>
              <span className="inline-flex items-center gap-1.5">
                <Clock size={13} /> 素材总时长 {stats.hours.toFixed(1)} 小时
              </span>
              <span className="inline-flex items-center gap-1.5">
                <Sparkles size={13} /> 已识别 {stats.rallies} 个回合
              </span>
            </motion.p>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search size={14} className="absolute top-1/2 left-3 -translate-y-1/2 text-ink-500" />
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="搜索工程…"
                className="field w-[220px] pl-8"
              />
            </div>
            <Button
              variant="primary"
              size="lg"
              onClick={() => {
                setName(`羽毛球剪辑 ${new Date().toLocaleDateString('zh-CN')}`)
                setCreating(true)
              }}
            >
              <FolderPlus size={15} />
              新建工程
            </Button>
          </div>
        </div>

        {/* 列表 */}
        {!booted && !projects.length ? (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(300px,1fr))] gap-4">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-[220px]" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <Card className="py-6">
            <Empty
              icon={<Film size={34} />}
              title={q ? '没有匹配的工程' : '还没有工程'}
              desc={
                q
                  ? '换个关键词试试。'
                  : '新建一个工程，导入你的羽毛球录像，AI 会自动剔除无效片段、切分每个回合并给出质量评分。'
              }
              action={
                !q && (
                  <Button
                    variant="primary"
                    onClick={() => {
                      setName(`羽毛球剪辑 ${new Date().toLocaleDateString('zh-CN')}`)
                      setCreating(true)
                    }}
                  >
                    <FolderPlus size={15} />
                    新建第一个工程
                  </Button>
                )
              }
            />
          </Card>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(310px,1fr))] gap-4">
            {filtered.map((p, i) => (
              <motion.div
                key={p.id}
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(i * 0.045, 0.4), duration: 0.38, ease: [0.22, 1, 0.36, 1] }}
              >
                <Card hover className="group overflow-hidden">
                  <button
                    className="relative block h-[152px] w-full overflow-hidden bg-ink-850"
                    onClick={() => openProject(p.id)}
                  >
                    {p.poster ? (
                      <img
                        src={api.assetUrl(p.poster)}
                        alt=""
                        className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.05]"
                        loading="lazy"
                      />
                    ) : (
                      <div className="flex h-full items-center justify-center text-ink-700">
                        <Film size={34} />
                      </div>
                    )}
                    <div className="absolute inset-0 bg-gradient-to-t from-ink-950/92 via-ink-950/25 to-transparent" />
                    <div className="absolute bottom-2 left-3 flex items-center gap-2 text-[11px] text-ink-200">
                      <span className="mono">{humanDuration(p.duration)}</span>
                      <span className="text-ink-500">·</span>
                      <span>{p.media_count} 个素材</span>
                    </div>
                    {p.analyzed && (
                      <span className="absolute top-2.5 right-2.5 inline-flex items-center gap-1 rounded-md bg-court-500/85 px-1.5 py-[2px] text-[10.5px] font-semibold text-ink-950">
                        <Sparkles size={10} />
                        {p.rally_count} 回合
                      </span>
                    )}
                  </button>
                  <div
                    className="group/name flex cursor-pointer items-start justify-between gap-2 px-3.5 py-3"
                    onClick={() => openProject(p.id)}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[13px] font-medium text-white transition-colors group-hover/name:text-court-300">
                        {p.name}
                      </div>
                      <div className="mt-0.5 text-[11px] text-ink-500">
                        更新于 {relTime(p.updated_at)}
                        {!p.analyzed && <span className="ml-2 text-amber-glow">尚未分析</span>}
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                      <Button
                        variant="ghost"
                        size="icon"
                        title="复制工程"
                        onClick={async (e) => {
                          // 别让点击冒泡到卡片本身：那会顺手把工程打开，跳进剪辑台
                          e.stopPropagation()
                          await api.duplicateProject(p.id)
                          refresh()
                          toast({ kind: 'success', title: '已复制工程' })
                        }}
                      >
                        <Copy size={13} />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        title="删除工程"
                        onClick={async (e) => {
                          e.stopPropagation()
                          const ok = await confirm({
                            title: `删除工程「${p.name}」？`,
                            desc: '工程文件会被移到同目录的备份文件，素材本身不会被删除。',
                            danger: true,
                          })
                          if (ok) {
                            await deleteProject(p.id)
                            toast({ kind: 'info', title: '工程已删除' })
                          }
                        }}
                      >
                        <Trash2 size={13} className="text-rose-hot/80" />
                      </Button>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* 新建工程 */}
      <Modal
        open={creating}
        onClose={() => setCreating(false)}
        title="新建工程"
        subtitle="工程用于组织素材、分析结果与时间线"
        width={460}
        footer={
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setCreating(false)}>
              取消
            </Button>
            <Button
              variant="primary"
              disabled={!name.trim()}
              onClick={async () => {
                const id = await createProject(name.trim())
                setCreating(false)
                if (id) toast({ kind: 'success', title: '工程已创建', detail: '现在导入素材开始剪辑' })
              }}
            >
              创建并进入
            </Button>
          </div>
        }
      >
        <label className="mb-1.5 block text-[12px] text-ink-300">工程名称</label>
        <input
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && name.trim()) {
              createProject(name.trim()).then((id) => {
                setCreating(false)
                if (id) toast({ kind: 'success', title: '工程已创建' })
              })
            }
          }}
          className="field"
          placeholder="例如：2026-09-13 羽毛球训练"
        />
      </Modal>

      {/* 按路径导入 */}
      <Modal
        open={!!targetProject}
        onClose={() => setTargetProject(null)}
        title="导入素材"
        subtitle="每行一个文件路径，也可以直接把文件拖进剪辑台"
        width={620}
        footer={
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setTargetProject(null)}>
              取消
            </Button>
            <Button variant="primary" loading={importing} onClick={doImport} disabled={!importPath.trim()}>
              开始导入
            </Button>
          </div>
        }
      >
        <textarea
          value={importPath}
          onChange={(e) => setImportPath(e.target.value)}
          rows={7}
          className="field mono text-[11.5px]"
          placeholder={'C:\\Users\\me\\Videos\\match.mp4\nC:\\Users\\me\\Videos\\match2.mp4'}
        />
      </Modal>
    </div>
  )
}
