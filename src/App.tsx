import { useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'
import Editor from '@monaco-editor/react'
import type { OnMount } from '@monaco-editor/react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import {
  BookOpen,
  ArrowDownToLine,
  ArrowUpToLine,
  Bug,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronDown,
  Circle,
  Code2,
  CornerDownRight,
  ListFilter,
  History,
  Loader2,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  Plus,
  RotateCcw,
  Search,
  Square,
  Sun,
  Trophy,
  X,
} from 'lucide-react'
import { debugCode, fetchProblem, fetchProblems, runCode, submitCode } from './api'
import type { DebugResult, Difficulty, JudgeResult, ProblemDetail, ProblemMeta } from './api'

const difficultyText: Record<Difficulty, string> = {
  easy: '简单',
  medium: '中等',
  hard: '困难',
}

const difficultyClass: Record<Difficulty, string> = {
  easy: 'easy',
  medium: 'medium',
  hard: 'hard',
}

type View = 'list' | 'detail'
type Theme = 'light' | 'dark'
type Panel = 'description' | 'result' | 'debug' | 'solution' | 'history'
type Action = 'run' | 'submit' | 'debug' | null
type TestcaseView = 'cases' | 'result'

type SubmissionRecord = {
  id: string
  createdAt: string
  code: string
  passed: number
  total: number
  allPassed: boolean
  elapsedMs: number
  error?: string
}

type CustomCase = { input: string; output: string; explanation: string }

const DRAFT_VERSION = 2
const MIN_PROBLEM_PANE_WIDTH = 320
const MIN_EDITOR_PANE_WIDTH = 360
const RESIZER_WIDTH = 7
const MIN_TESTCASE_PANE_HEIGHT = 180
const MIN_CODE_PANE_HEIGHT = 240

function readStoredNumber(key: string, fallback: number) {
  const value = Number(localStorage.getItem(key))
  return Number.isFinite(value) && value > 0 ? value : fallback
}

function problemIdFromPath() {
  const match = window.location.pathname.match(/^\/problem\/([^/]+)\/?$/)
  return match ? decodeURIComponent(match[1]) : null
}

export default function App() {
  const initialProblemId = useMemo(problemIdFromPath, [])
  const [view, setView] = useState<View>(initialProblemId ? 'detail' : 'list')
  const [problems, setProblems] = useState<ProblemMeta[]>([])
  const [activeId, setActiveId] = useState<string | null>(initialProblemId)
  const [activeProblem, setActiveProblem] = useState<ProblemDetail | null>(null)
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('全部')
  const [code, setCode] = useState('')
  // Running public examples and submitting validation cases are separate workflows.
  // Keeping their results separate prevents a submission from rewriting the test dock.
  const [runResult, setRunResult] = useState<JudgeResult | null>(null)
  const [submissionResult, setSubmissionResult] = useState<JudgeResult | null>(null)
  const [action, setAction] = useState<Action>(null)
  const [panel, setPanel] = useState<Panel>('description')
  const [debugResult, setDebugResult] = useState<DebugResult | null>(null)
  const [debugIndex, setDebugIndex] = useState(0)
  const [breakpoints, setBreakpoints] = useState<Set<number>>(new Set())
  const [activeExampleIndex, setActiveExampleIndex] = useState(0)
  const [activeRunCaseIndex, setActiveRunCaseIndex] = useState(0)
  const [testcaseView, setTestcaseView] = useState<TestcaseView>('cases')
  const [submissionHistory, setSubmissionHistory] = useState<SubmissionRecord[]>([])
  const [selectedSubmissionId, setSelectedSubmissionId] = useState<string | null>(null)
  const [testCases, setTestCases] = useState<CustomCase[]>([])
  const [solvedIds, setSolvedIds] = useState<Set<string>>(new Set())
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => localStorage.getItem('ai-sidebar-collapsed-v3') !== 'false')
  const [problemPaneWidth, setProblemPaneWidth] = useState(() => readStoredNumber('ai-problem-pane-width', 560))
  const [resizing, setResizing] = useState(false)
  const [testcaseHeight, setTestcaseHeight] = useState(() => readStoredNumber('ai-testcase-pane-height', 280))
  const [testcaseResizing, setTestcaseResizing] = useState(false)
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = localStorage.getItem('ai-theme')
    if (stored === 'light' || stored === 'dark') return stored
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
  })
  const workPanesRef = useRef<HTMLDivElement>(null)
  const resizingRef = useRef(false)
  const editorPaneRef = useRef<HTMLElement>(null)
  const testcaseResizingRef = useRef(false)
  const editorRef = useRef<Parameters<OnMount>[0] | null>(null)
  const monacoRef = useRef<Parameters<OnMount>[1] | null>(null)
  const breakpointDecorationsRef = useRef<string[]>([])
  const debugDecorationsRef = useRef<string[]>([])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('ai-theme', theme)
  }, [theme])

  useEffect(() => {
    localStorage.setItem('ai-sidebar-collapsed-v3', String(sidebarCollapsed))
  }, [sidebarCollapsed])

  useEffect(() => {
    localStorage.setItem('ai-problem-pane-width', String(Math.round(problemPaneWidth)))
  }, [problemPaneWidth])

  useEffect(() => {
    localStorage.setItem('ai-testcase-pane-height', String(Math.round(testcaseHeight)))
  }, [testcaseHeight])

  useEffect(() => {
    fetchProblems().then((items) => {
      setProblems(items)
      setSolvedIds(new Set(items.filter((p) => localStorage.getItem(`ai-result:${p.id}`) === 'AC').map((p) => p.id)))
      if (initialProblemId && !items.some((item) => item.id === initialProblemId)) {
        window.history.replaceState({}, '', '/')
        setView('list')
        setActiveId(items[0]?.id ?? null)
      } else if (!activeId && items.length) setActiveId(items[0].id)
    })
  }, [activeId, initialProblemId])

  useEffect(() => {
    const handlePopState = () => {
      const problemId = problemIdFromPath()
      if (problemId) {
        setActiveId(problemId)
        setView('detail')
      } else {
        setView('list')
      }
    }
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  useEffect(() => {
    if (!activeId) return
    fetchProblem(activeId).then((problem) => {
      setActiveProblem(problem)
      const storedCode = localStorage.getItem(`ai-code:v${DRAFT_VERSION}:${problem.id}`)
      const isOldPrompt = storedCode?.includes('# 请从定义开始独立实现') && !/^\s*(def|class)\s/m.test(storedCode)
      setCode(!storedCode || isOldPrompt ? problem.starter : storedCode)
      setRunResult(null)
      setSubmissionResult(null)
      setDebugResult(null)
      setDebugIndex(0)
      setActiveExampleIndex(0)
      setActiveRunCaseIndex(0)
      setTestcaseView('cases')
      setSubmissionHistory(JSON.parse(localStorage.getItem(`ai-submissions:${problem.id}`) ?? '[]') as SubmissionRecord[])
      setSelectedSubmissionId(null)
      const savedCases = JSON.parse(localStorage.getItem(`ai-testcases:v2:${problem.id}`) ?? '[]') as CustomCase[]
      const legacyCases = JSON.parse(localStorage.getItem(`ai-custom-cases:${problem.id}`) ?? '[]') as CustomCase[]
      setTestCases(savedCases.length ? savedCases : [...problem.examples, ...legacyCases])
      const storedBreakpoints = JSON.parse(localStorage.getItem(`ai-breakpoints:${problem.id}`) ?? '[]') as number[]
      setBreakpoints(new Set(storedBreakpoints))
      setPanel('description')
    })
  }, [activeId])

  const categories = useMemo(() => ['全部', ...Array.from(new Set(problems.map((p) => p.category)))], [problems])

  const completed = solvedIds.size
  const allExamples = testCases
  const activeExample = allExamples[activeExampleIndex]
  const activeRunCaseResult = runResult?.cases[activeRunCaseIndex]
  const activePublicExample = activeProblem?.examples[activeRunCaseIndex]

  const filteredProblems = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return problems.filter((problem) => {
      const inCategory = category === '全部' || problem.category === category
      const inQuery =
        !needle ||
        problem.id.toLowerCase().includes(needle) ||
        problem.title.toLowerCase().includes(needle) ||
        problem.tags.some((tag) => tag.toLowerCase().includes(needle))
      return inCategory && inQuery
    })
  }, [category, problems, query])

  const openProblem = (id: string) => {
    setActiveId(id)
    setView('detail')
    window.history.pushState({}, '', `/problem/${encodeURIComponent(id)}`)
  }

  const openProblemFromSidebar = (id: string) => {
    if (id === activeId) return
    setActiveId(id)
    window.history.pushState({}, '', `/problem/${encodeURIComponent(id)}`)
  }

  const openProblemList = () => {
    setView('list')
    window.history.pushState({}, '', '/')
  }

  const handleCodeChange = (value: string | undefined) => {
    const next = value ?? ''
    setCode(next)
    if (activeProblem) localStorage.setItem(`ai-code:v${DRAFT_VERSION}:${activeProblem.id}`, next)
  }

  const saveTestCases = (next: CustomCase[]) => {
    setTestCases(next)
    if (activeProblem) localStorage.setItem(`ai-testcases:v2:${activeProblem.id}`, JSON.stringify(next))
  }

  const updateTestCase = (index: number, key: keyof CustomCase, value: string) => {
    saveTestCases(allExamples.map((item, currentIndex) => currentIndex === index ? { ...item, [key]: value } : item))
  }

  const handleRun = async () => {
    if (!activeProblem) return
    setAction('run')
    try {
      const nextResult = await runCode(activeProblem.id, code, allExamples.length)
      setRunResult(nextResult)
      setTestcaseView('result')
      const firstFailedCase = nextResult.cases.findIndex((item) => !item.passed)
      setActiveRunCaseIndex(firstFailedCase >= 0 ? firstFailedCase : 0)
    } catch (error) {
      setRunResult({ score: 0, all_passed: false, cases: [], mode: 'run', error: error instanceof Error ? error.message : '运行失败' })
      setActiveRunCaseIndex(0)
    } finally {
      setAction(null)
    }
  }

  const handleSubmit = async () => {
    if (!activeProblem) return
    setAction('submit')
    setPanel('result')
    try {
      const nextResult = await submitCode(activeProblem.id, code)
      setSubmissionResult(nextResult)
      const record: SubmissionRecord = {
        id: crypto.randomUUID(),
        createdAt: new Date().toISOString(),
        code,
        passed: nextResult.cases.filter((item) => item.passed).length,
        total: nextResult.cases.length,
        allPassed: nextResult.all_passed,
        elapsedMs: nextResult.cases.reduce((total, item) => total + item.elapsed_ms, 0),
        error: nextResult.error,
      }
      setSubmissionHistory((current) => {
        const next = [record, ...current]
        localStorage.setItem(`ai-submissions:${activeProblem.id}`, JSON.stringify(next))
        return next
      })
      localStorage.setItem(`ai-result:${activeProblem.id}`, nextResult.all_passed ? 'AC' : 'WA')
      setSolvedIds((current) => {
        const next = new Set(current)
        if (nextResult.all_passed) next.add(activeProblem.id)
        else next.delete(activeProblem.id)
        return next
      })
    } catch (error) {
      setSubmissionResult({
        score: 0,
        all_passed: false,
        cases: [],
        error: error instanceof Error ? error.message : '提交失败',
      })
    } finally {
      setAction(null)
    }
  }

  const handleDebug = async () => {
    if (!activeProblem) return
    setAction('debug')
    setPanel('debug')
    setDebugIndex(0)
    try {
      const next = await debugCode(activeProblem.id, code, [...breakpoints].sort((a, b) => a - b))
      setDebugResult(next)
    } catch (error) {
      setDebugResult({ events: [], error: error instanceof Error ? error.message : '调试失败' })
    } finally {
      setAction(null)
    }
  }

  const resetCode = () => {
    if (!activeProblem) return
    localStorage.removeItem(`ai-code:v${DRAFT_VERSION}:${activeProblem.id}`)
    setCode(activeProblem.starter)
  }

  const handleEditorMount: OnMount = (editor, monaco) => {
    editorRef.current = editor
    monacoRef.current = monaco
    editor.onMouseDown((event) => {
      if (event.target.type !== monaco.editor.MouseTargetType.GUTTER_GLYPH_MARGIN || !event.target.position) return
      const line = event.target.position.lineNumber
      setBreakpoints((current) => {
        const next = new Set(current)
        if (next.has(line)) next.delete(line)
        else next.add(line)
        if (activeId) localStorage.setItem(`ai-breakpoints:${activeId}`, JSON.stringify([...next]))
        return next
      })
    })
  }

  useEffect(() => {
    const editor = editorRef.current
    if (!editor) return
    breakpointDecorationsRef.current = editor.deltaDecorations(
      breakpointDecorationsRef.current,
      [...breakpoints].map((line) => ({
        range: { startLineNumber: line, startColumn: 1, endLineNumber: line, endColumn: 1 },
        options: { glyphMarginClassName: 'breakpoint-glyph', glyphMarginHoverMessage: { value: `断点：第 ${line} 行` } },
      })),
    )
  }, [breakpoints])

  const currentDebugEvent = debugResult?.events[debugIndex]

  useEffect(() => {
    const editor = editorRef.current
    if (!editor) return
    debugDecorationsRef.current = editor.deltaDecorations(
      debugDecorationsRef.current,
      currentDebugEvent
        ? [{
            range: {
              startLineNumber: currentDebugEvent.line,
              startColumn: 1,
              endLineNumber: currentDebugEvent.line,
              endColumn: 1,
            },
            options: { isWholeLine: true, className: 'debug-current-line', glyphMarginClassName: 'debug-current-glyph' },
          }]
        : [],
    )
    if (currentDebugEvent) {
      editor.revealLineInCenter(currentDebugEvent.line)
      editor.setPosition({ lineNumber: currentDebugEvent.line, column: 1 })
    }
  }, [currentDebugEvent])

  const clampProblemPaneWidth = (width: number) => {
    const containerWidth = workPanesRef.current?.getBoundingClientRect().width ?? window.innerWidth
    const maxWidth = Math.max(MIN_PROBLEM_PANE_WIDTH, containerWidth - MIN_EDITOR_PANE_WIDTH - RESIZER_WIDTH)
    return Math.min(Math.max(width, MIN_PROBLEM_PANE_WIDTH), maxWidth)
  }

  const resizeFromPointer = (clientX: number) => {
    const rect = workPanesRef.current?.getBoundingClientRect()
    if (!rect) return
    setProblemPaneWidth(clampProblemPaneWidth(clientX - rect.left))
  }

  const finishPaneResize = () => {
    resizingRef.current = false
    setResizing(false)
  }

  const finishTestcaseResize = () => {
    testcaseResizingRef.current = false
    setTestcaseResizing(false)
  }

  useEffect(() => {
    // Pointer capture can be lost when the pointer leaves the browser window.
    // Always clear global resize styles so the cursor cannot get stuck.
    const finishAllResizes = () => {
      if (resizingRef.current) finishPaneResize()
      if (testcaseResizingRef.current) finishTestcaseResize()
    }
    window.addEventListener('pointerup', finishAllResizes)
    window.addEventListener('pointercancel', finishAllResizes)
    window.addEventListener('blur', finishAllResizes)
    return () => {
      window.removeEventListener('pointerup', finishAllResizes)
      window.removeEventListener('pointercancel', finishAllResizes)
      window.removeEventListener('blur', finishAllResizes)
    }
  }, [])

  const handleResizeStart = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId)
    resizingRef.current = true
    setResizing(true)
    resizeFromPointer(event.clientX)
  }

  const handleResizeMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (resizingRef.current) resizeFromPointer(event.clientX)
  }

  const handleResizeEnd = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
    finishPaneResize()
  }

  const handleResizeKey = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
    event.preventDefault()
    const direction = event.key === 'ArrowLeft' ? -1 : 1
    setProblemPaneWidth((current) => clampProblemPaneWidth(current + direction * (event.shiftKey ? 48 : 16)))
  }

  const resetPaneWidth = () => {
    const containerWidth = workPanesRef.current?.getBoundingClientRect().width ?? window.innerWidth
    setProblemPaneWidth(clampProblemPaneWidth((containerWidth - RESIZER_WIDTH) / 2))
  }

  const clampTestcaseHeight = (height: number) => {
    const containerHeight = editorPaneRef.current?.getBoundingClientRect().height ?? window.innerHeight
    const maxHeight = Math.max(MIN_TESTCASE_PANE_HEIGHT, containerHeight - MIN_CODE_PANE_HEIGHT)
    return Math.min(Math.max(height, MIN_TESTCASE_PANE_HEIGHT), maxHeight)
  }

  const resizeTestcaseFromPointer = (clientY: number) => {
    const rect = editorPaneRef.current?.getBoundingClientRect()
    if (!rect) return
    setTestcaseHeight(clampTestcaseHeight(rect.bottom - clientY))
  }

  const handleTestcaseResizeStart = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId)
    testcaseResizingRef.current = true
    setTestcaseResizing(true)
    resizeTestcaseFromPointer(event.clientY)
  }

  const handleTestcaseResizeMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (testcaseResizingRef.current) resizeTestcaseFromPointer(event.clientY)
  }

  const handleTestcaseResizeEnd = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
    finishTestcaseResize()
  }

  const handleTestcaseResizeKey = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return
    event.preventDefault()
    const direction = event.key === 'ArrowUp' ? 1 : -1
    setTestcaseHeight((current) => clampTestcaseHeight(current + direction * (event.shiftKey ? 48 : 16)))
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-left">
          <button className="brand" onClick={openProblemList} title="返回题库">
            <Code2 size={22} />
            <span>AI First LeetCode</span>
          </button>
          {view === 'detail' && (
            <button
              className="topbar-drawer"
              onClick={() => setSidebarCollapsed((current) => !current)}
              title={sidebarCollapsed ? '展开题单' : '收起题单'}
              aria-label={sidebarCollapsed ? '展开题单' : '收起题单'}
            >
              {sidebarCollapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
            </button>
          )}
          {view === 'detail' && <span className="study-plan-name">AI 手撕题单</span>}
        </div>
        {view === 'detail' ? (
          panel === 'debug' && debugResult?.events.length ? (
            <DebugControls
              className="debug-command-bar"
              events={debugResult.events}
              index={debugIndex}
              onIndexChange={setDebugIndex}
              onStop={() => {
                setDebugResult(null)
                setPanel('description')
              }}
            />
          ) : (
            <div className="workspace-command-bar">
              <button onClick={handleDebug} disabled={action !== null || !activeProblem} title="调试示例 1" aria-label="调试示例 1">
                {action === 'debug' ? <Loader2 className="spin" size={17} /> : <Bug size={17} />}
              </button>
              <button onClick={handleRun} disabled={action !== null || !activeProblem} title="运行 3 个示例" aria-label="运行 3 个示例">
                {action === 'run' ? <Loader2 className="spin" size={17} /> : <Play size={17} fill="currentColor" />}
              </button>
              <button className="workspace-submit" onClick={handleSubmit} disabled={action !== null || !activeProblem} title="提交全部 20 个校验用例">
                {action === 'submit' ? <Loader2 className="spin" size={17} /> : <Check size={17} />}
                提交
              </button>
            </div>
          )
        ) : <div className="topbar-spacer" />}
        <div className="topbar-actions">
          <div className="progress">
            <Trophy size={16} />
            <span>
              {completed}/{problems.length}
            </span>
          </div>
          <button
            className="topbar-icon"
            onClick={() => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))}
            title={theme === 'dark' ? '切换到白天模式' : '切换到黑夜模式'}
            aria-label={theme === 'dark' ? '切换到白天模式' : '切换到黑夜模式'}
          >
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </header>

      {view === 'list' ? (
        <main className="problem-list">
          <section className="list-toolbar">
            <div className="searchbox">
              <Search size={18} />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索题目、编号或标签" />
            </div>
            <div className="selectbox">
              <ListFilter size={18} />
              <select value={category} onChange={(event) => setCategory(event.target.value)}>
                {categories.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>
            </div>
          </section>

          <section className="summary-band">
            <div>
              <span className="summary-number">{problems.length}</span>
              <span className="summary-label">道手撕 AI 题</span>
            </div>
            <div>
              <span className="summary-number">{categories.length - 1}</span>
              <span className="summary-label">个主题</span>
            </div>
            <div>
              <span className="summary-number">{completed}</span>
              <span className="summary-label">已通过</span>
            </div>
          </section>

          <section className="problem-table">
            {filteredProblems.map((problem) => {
              const accepted = solvedIds.has(problem.id)
              return (
                <button key={problem.id} className="problem-row" onClick={() => openProblem(problem.id)}>
                  <span className={accepted ? 'status-dot solved' : 'status-dot'}>
                    {accepted ? <CheckCircle2 size={20} strokeWidth={2.5} /> : <Circle size={20} strokeWidth={1.8} />}
                  </span>
                  <span className="problem-index">{String(problem.order).padStart(2, '0')}</span>
                  <span className="problem-main">
                    <strong>{problem.title}</strong>
                    <small>{problem.id}</small>
                  </span>
                  <span className={`difficulty ${difficultyClass[problem.difficulty]}`}>{difficultyText[problem.difficulty]}</span>
                  <span className="category-chip">{problem.category}</span>
                </button>
              )
            })}
          </section>
        </main>
      ) : (
        <main className={`workspace ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
          {!sidebarCollapsed && (
            <button
              className="sidebar-backdrop"
              onClick={() => setSidebarCollapsed(true)}
              aria-label="收起题单"
            />
          )}
          <aside className="sidebar">
            <div className="sidebar-header">
              {!sidebarCollapsed && (
                <button className="back-button" onClick={openProblemList}>
                  <ChevronLeft size={18} />
                  题库
                </button>
              )}
              <button
                className="collapse-button"
                onClick={() => setSidebarCollapsed((current) => !current)}
                title={sidebarCollapsed ? '展开题库栏' : '收起题库栏'}
                aria-label={sidebarCollapsed ? '展开题库栏' : '收起题库栏'}
              >
                {sidebarCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
              </button>
            </div>
            <div className="mini-list">
              {problems.map((problem) => (
                <button
                  key={problem.id}
                  className={problem.id === activeId ? 'mini-item active' : 'mini-item'}
                  onClick={() => openProblemFromSidebar(problem.id)}
                  title={problem.title}
                >
                  <span className={solvedIds.has(problem.id) ? 'mini-status solved' : 'mini-status'}>
                    {solvedIds.has(problem.id) ? <CheckCircle2 size={17} strokeWidth={2.5} /> : <Circle size={17} strokeWidth={1.8} />}
                  </span>
                  <span>{String(problem.order).padStart(2, '0')}</span>
                  <strong>{problem.title}</strong>
                </button>
              ))}
            </div>
          </aside>

          <div
            ref={workPanesRef}
            className={`work-panes ${resizing ? 'resizing' : ''}`}
            style={{ '--problem-pane-width': `${problemPaneWidth}px` } as CSSProperties}
          >
            <section className="content-pane">
            {!activeProblem ? (
              <div className="loading">载入中...</div>
            ) : (
              <>
                <div className="problem-tabs">
                  <button className={panel === 'description' ? 'active' : ''} onClick={() => setPanel('description')}>
                    <BookOpen size={16} />
                    题目描述
                  </button>
                  <button className={panel === 'result' ? 'active' : ''} onClick={() => setPanel('result')}>
                    <Play size={16} />
                    测试总览
                  </button>
                  <button className={panel === 'debug' ? 'active' : ''} onClick={() => setPanel('debug')}>
                    <Bug size={16} />
                    Debug
                  </button>
                  <button className={panel === 'solution' ? 'active' : ''} onClick={() => setPanel('solution')}>
                    <Code2 size={16} />
                    参考解析
                  </button>
                  <button className={panel === 'history' ? 'active' : ''} onClick={() => setPanel('history')}>
                    <History size={16} />
                    提交记录
                  </button>
                </div>
                <div className="markdown-panel">
                  {panel === 'description' && (
                    <>
                      <div className="problem-heading">
                        <div>
                          <span className="muted-id">{activeProblem.id}</span>
                          <h1>{activeProblem.title}</h1>
                        </div>
                        <span className={`difficulty ${difficultyClass[activeProblem.difficulty]}`}>
                          {difficultyText[activeProblem.difficulty]}
                        </span>
                      </div>
                      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                        {activeProblem.readme}
                      </ReactMarkdown>
                    </>
                  )}
                  {panel === 'solution' && (
                    <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                      {['## 参考实现', '```python', activeProblem.solution, '```'].join('\n')}
                    </ReactMarkdown>
                  )}
                  {panel === 'result' && (
                    <ResultView
                      result={submissionResult}
                      loading={action === 'run' || action === 'submit'}
                    />
                  )}
                  {panel === 'debug' && (
                    <DebugView
                      result={debugResult}
                      loading={action === 'debug'}
                      index={debugIndex}
                      onIndexChange={setDebugIndex}
                      example={activeExample}
                      onStop={() => {
                        setDebugResult(null)
                        setPanel('description')
                      }}
                    />
                  )}
                  {panel === 'history' && (
                    <SubmissionHistory records={submissionHistory} selectedId={selectedSubmissionId} onSelect={setSelectedSubmissionId} />
                  )}
                </div>
              </>
            )}
            </section>

            <div
              className="pane-resizer"
              role="separator"
              aria-label="调整题面和编辑器宽度"
              aria-orientation="vertical"
              aria-valuemin={MIN_PROBLEM_PANE_WIDTH}
              aria-valuenow={Math.round(problemPaneWidth)}
              tabIndex={0}
              onPointerDown={handleResizeStart}
              onPointerMove={handleResizeMove}
              onPointerUp={handleResizeEnd}
              onPointerCancel={handleResizeEnd}
              onLostPointerCapture={finishPaneResize}
              onKeyDown={handleResizeKey}
              onDoubleClick={resetPaneWidth}
              title="拖动调整宽度，双击恢复均分"
            />

            <section
              ref={editorPaneRef}
              className={`editor-pane ${testcaseResizing ? 'testcase-resizing' : ''}`}
              style={{ '--testcase-pane-height': `${testcaseHeight}px` } as CSSProperties}
            >
              <div className="editor-card">
                <div className="editor-titlebar">
                  <Code2 size={17} />
                  <span>代码</span>
                </div>
                <div className="editor-toolbar">
                  <div className="editor-heading">
                    <button className="language-button" title="当前语言">
                      Python
                      <ChevronDown size={14} />
                    </button>
                    <span className="editor-mode">智能模式</span>
                  </div>
                  <div className="editor-actions">
                    <button onClick={resetCode} title="重置代码">
                      <RotateCcw size={16} />
                    </button>
                  </div>
                </div>
                <div className="editor-body">
                  <Editor
                    height="100%"
                    language="python"
                    theme={theme === 'dark' ? 'vs-dark' : 'light'}
                    value={code}
                    onChange={handleCodeChange}
                    onMount={handleEditorMount}
                    options={{ glyphMargin: true, minimap: { enabled: false }, fontSize: 14, scrollBeyondLastLine: false, wordWrap: 'off' }}
                  />
                </div>
              </div>
              <div
                className="testcase-resizer"
                role="separator"
                aria-label="调整代码和测试区高度"
                aria-orientation="horizontal"
                tabIndex={0}
                onPointerDown={handleTestcaseResizeStart}
                onPointerMove={handleTestcaseResizeMove}
                onPointerUp={handleTestcaseResizeEnd}
                onPointerCancel={handleTestcaseResizeEnd}
                onLostPointerCapture={finishTestcaseResize}
                onKeyDown={handleTestcaseResizeKey}
                title="拖动调整代码与测试区高度"
              />
            <section className="testcase-dock" aria-label="测试用例与测试结果">
              <div className="testcase-tabs">
                <div>
                  <button className={testcaseView === 'cases' ? 'active' : ''} onClick={() => setTestcaseView('cases')}>
                    <CheckCircle2 size={15} />
                    测试用例
                  </button>
                  <button
                    className={testcaseView === 'result' ? 'active' : ''}
                    onClick={() => runResult?.cases.length && setTestcaseView('result')}
                    disabled={!runResult?.cases.length}
                  >
                    <Play size={15} />
                    测试结果
                  </button>
                </div>
                {testcaseView === 'cases' && <span><CheckCircle2 size={14} />公开示例</span>}
              </div>
              {testcaseView === 'cases' && activeExample && (
                <div className="testcase-content">
                  <div className="case-picker">
                    {allExamples.map((_, index) => (
                      <div key={index} className={activeExampleIndex === index ? 'case-tab active' : 'case-tab'}>
                        <button onClick={() => setActiveExampleIndex(index)}>Case {index + 1}</button>
                        {allExamples.length > 1 && <button className="remove-case-button" onClick={() => {
                          const next = allExamples.filter((_, currentIndex) => currentIndex !== index)
                          saveTestCases(next)
                          setActiveExampleIndex(Math.max(0, Math.min(activeExampleIndex, next.length - 1)))
                        }} title="删除测试用例" aria-label="删除测试用例"><X size={13} /></button>}
                      </div>
                    ))}
                    <button className="add-case-tab" onClick={() => {
                      const next = [...allExamples, { input: '', output: '', explanation: '' }]
                      saveTestCases(next)
                      setActiveExampleIndex(next.length - 1)
                    }} title="添加测试用例" aria-label="添加测试用例"><Plus size={18} /></button>
                    <button className="reset-case-button" onClick={() => {
                      if (!activeProblem) return
                      saveTestCases(activeProblem.examples)
                      setActiveExampleIndex(0)
                    }} title="重置测试用例" aria-label="重置测试用例"><RotateCcw size={15} /></button>
                  </div>
                  <div className="testcase-field">
                    <small>输入</small>
                    <textarea value={activeExample.input} onChange={(event) => updateTestCase(activeExampleIndex, 'input', event.target.value)} />
                  </div>
                  <div className="testcase-field">
                    <small>预期输出</small>
                    <textarea value={activeExample.output} onChange={(event) => updateTestCase(activeExampleIndex, 'output', event.target.value)} />
                  </div>
                </div>
              )}
              {testcaseView === 'result' && runResult?.cases.length && activeRunCaseResult && (
                <div className="case-result-panel">
                  <div className={activeRunCaseResult.passed ? 'case-result-status passed' : 'case-result-status failed'}>
                    <strong>{activeRunCaseResult.passed ? '通过' : '未通过'}</strong>
                    <span>公开示例 {activeRunCaseIndex + 1} · 执行用时：{activeRunCaseResult.elapsed_ms.toFixed(1)} ms</span>
                    {!activeRunCaseResult.passed && (
                      <button
                        className="add-case-button"
                        onClick={() => {
                          if (!activeProblem) return
                          const nextCase = {
                            input: activeRunCaseResult.input || activePublicExample?.input || '',
                            output: activeRunCaseResult.expected || activePublicExample?.output || '',
                            explanation: '从一次未通过的公开用例保存。',
                          }
                          saveTestCases([...allExamples, nextCase])
                          setActiveExampleIndex(allExamples.length)
                          setTestcaseView('cases')
                        }}
                      >
                        添加到测试用例
                      </button>
                    )}
                  </div>
                  <div className="result-case-picker">
                    {runResult.cases.map((item, index) => {
                      return (
                        <button key={index} className={activeRunCaseIndex === index ? 'active' : ''} onClick={() => setActiveRunCaseIndex(index)}>
                          {item?.passed ? <Check size={16} /> : <Circle size={16} />}
                          Case {index + 1}
                        </button>
                      )
                    })}
                  </div>
                  <div className="case-result-fields">
                    <div className="testcase-field">
                      <small>输入</small>
                      <pre>{activeRunCaseResult.input || activePublicExample?.input || '该示例未提供可展示输入。'}</pre>
                    </div>
                    {activeRunCaseResult.passed ? (
                      <>
                        <div className="testcase-field">
                          <small>输出</small>
                          <pre>{activeRunCaseResult.actual || activePublicExample?.output || '通过'}</pre>
                        </div>
                        <div className="testcase-field">
                          <small>预期结果</small>
                          <pre>{activeRunCaseResult.expected || activePublicExample?.output || '通过'}</pre>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="testcase-field">
                          <small>输出</small>
                          <pre className="case-output failed">{activeRunCaseResult.actual || '未返回结果'}</pre>
                        </div>
                        <div className="testcase-field">
                          <small>预期结果</small>
                          <pre className="case-output expected-output">{activeRunCaseResult.expected || activePublicExample?.output || '未提供预期结果'}</pre>
                        </div>
                        <details className="case-traceback">
                          <summary>完整错误信息</summary>
                          <pre>{activeRunCaseResult.error || activeRunCaseResult.reason || runResult.error}</pre>
                        </details>
                      </>
                    )}
                  </div>
                </div>
              )}
            </section>
            </section>
          </div>
        </main>
      )}
    </div>
  )
}

function ResultView({
  result,
  loading,
}: {
  result: JudgeResult | null
  loading: boolean
}) {
  if (loading) {
    return (
      <div className="result-empty">
        <Loader2 className="spin" size={22} />
        正在执行...
      </div>
    )
  }
  if (!result) return <div className="result-empty">提交后在这里查看全部测试用例的结果</div>
  if (result.error) {
    return (
      <div className="result-view">
        <div className="score rejected">
          <strong>运行错误</strong>
          <span>0/0</span>
        </div>
      </div>
    )
  }

  const passed = result.cases.filter((item) => item.passed).length
  const firstFailedIndex = result.cases.findIndex((item) => !item.passed)
  const firstFailedCase = firstFailedIndex >= 0 ? result.cases[firstFailedIndex] : undefined

  return (
    <div className="result-view">
      <div className={result.all_passed ? 'score accepted' : 'score rejected'}>
        <strong>{result.all_passed ? '通过' : '未通过'}</strong>
        <span>{passed} / {result.cases.length} 个通过的测试用例</span>
      </div>
      {!result.all_passed && firstFailedCase && (
        <article className="case failed">
          <span><Circle size={16} /></span>
          <strong>测试用例 {firstFailedIndex + 1}</strong>
          <small>{firstFailedCase.elapsed_ms.toFixed(1)} ms</small>
          <em>未通过</em>
          <div className="submission-failure-detail">
            <div>
              <small>输入</small>
              <pre>{firstFailedCase.input || '该用例未提供可展示输入。'}</pre>
            </div>
            <div>
              <small>输出</small>
              <pre className="case-output failed">{firstFailedCase.actual || '未返回结果'}</pre>
            </div>
            <div>
              <small>预期结果</small>
              <pre className="case-output expected-output">{firstFailedCase.expected || '未提供预期结果'}</pre>
            </div>
            <details className="case-traceback">
              <summary>完整错误信息</summary>
              <pre>{firstFailedCase.error || firstFailedCase.reason || '未提供错误详情。'}</pre>
            </details>
          </div>
        </article>
      )}
    </div>
  )
}

function SubmissionHistory({
  records,
  selectedId,
  onSelect,
}: {
  records: SubmissionRecord[]
  selectedId: string | null
  onSelect: (id: string | null) => void
}) {
  const selected = records.find((record) => record.id === selectedId)
  if (selected) {
    return (
      <div className="submission-detail">
        <button className="history-back" onClick={() => onSelect(null)}>返回提交记录</button>
        <div className={selected.allPassed ? 'submission-state passed' : 'submission-state failed'}>
          <strong>{selected.allPassed ? '通过' : '未通过'}</strong>
          <span>{selected.passed} / {selected.total} 个通过的测试用例</span>
        </div>
        <div className="submission-meta">提交于 {new Date(selected.createdAt).toLocaleString('zh-CN')} · {selected.elapsedMs.toFixed(1)} ms</div>
        {selected.error && <pre className="error-box">{selected.error}</pre>}
        <div className="submission-code"><small>代码 · Python</small><pre>{selected.code}</pre></div>
      </div>
    )
  }
  if (!records.length) return <div className="result-empty">暂无提交记录</div>
  return (
    <div className="submission-list">
      {records.map((record, index) => (
        <button key={record.id} onClick={() => onSelect(record.id)}>
          <span>{records.length - index}</span>
          <strong className={record.allPassed ? 'passed' : 'failed'}>{record.allPassed ? '通过' : '未通过'}</strong>
          <small>{record.passed} / {record.total} 个用例 · {record.elapsedMs.toFixed(1)} ms</small>
          <time>{new Date(record.createdAt).toLocaleString('zh-CN')}</time>
        </button>
      ))}
    </div>
  )
}

function DebugControls({
  className,
  events,
  index,
  onIndexChange,
  onStop,
}: {
  className: string
  events: DebugResult['events']
  index: number
  onIndexChange: (index: number) => void
  onStop?: () => void
}) {
  const event = events[index]
  const lastIndex = events.length - 1
  const stepInto = Math.min(lastIndex, index + 1)
  const stepOver = events.findIndex((item, eventIndex) => eventIndex > index && item.depth <= event.depth)
  const stepOut = events.findIndex((item, eventIndex) => eventIndex > index && item.depth < event.depth)
  const nextBreakpoint = events.findIndex((item, eventIndex) => eventIndex > index && item.breakpoint)
  const goTo = (target: number) => onIndexChange(target === -1 ? lastIndex : target)

  return (
    <div className={className}>
      {onStop && (
        <button className="debug-stop" onClick={onStop} title="结束调试" aria-label="结束调试">
          <Square size={15} fill="currentColor" />
        </button>
      )}
      <button onClick={() => onIndexChange(stepInto)} disabled={index === lastIndex} title="单步进入" aria-label="单步进入">
        <CornerDownRight size={17} />
      </button>
      <button onClick={() => goTo(stepOver)} disabled={index === lastIndex} title="单步跳过" aria-label="单步跳过">
        <ArrowDownToLine size={17} />
      </button>
      <button onClick={() => goTo(stepOut)} disabled={event.depth <= 1 || index === lastIndex} title="跳出当前函数" aria-label="跳出当前函数">
        <ArrowUpToLine size={17} />
      </button>
      <button onClick={() => goTo(nextBreakpoint)} disabled={index === lastIndex} title="继续到下一个断点" aria-label="继续到下一个断点">
        <Play size={16} fill="currentColor" />
      </button>
      <button onClick={() => onIndexChange(0)} disabled={index === 0} title="重新开始" aria-label="重新开始">
        <RotateCcw size={16} />
      </button>
      <span className="debug-position">{index + 1}/{events.length}</span>
    </div>
  )
}

function DebugView({
  result,
  loading,
  index,
  onIndexChange,
  example,
  onStop,
}: {
  result: DebugResult | null
  loading: boolean
  index: number
  onIndexChange: (index: number) => void
  example?: ProblemDetail['examples'][number]
  onStop: () => void
}) {
  if (loading) {
    return (
      <div className="result-empty">
        <Loader2 className="spin" size={22} />
        正在调试示例 1...
      </div>
    )
  }
  if (!result) return <div className="result-empty">在编辑器行号左侧设置断点，然后点击 Debug</div>
  if (!result.events.length) return <pre className="error-box">{result.error || '没有捕获到可调试的执行行'}</pre>

  const event = result.events[index]
  const variableEntries = Object.entries(event.variables)
  const globalEntries = Object.entries(event.globals)
  const completed = index === result.events.length - 1
  return (
    <div className="debug-view">
      <div className="debug-toolbar">
        <div>
          <strong>{result.case ?? '示例 1'}</strong>
          <span>第 {event.line} 行 · {event.function}() · 调用层级 {event.depth}</span>
        </div>
        <DebugControls className="debug-inline-controls" events={result.events} index={index} onIndexChange={onIndexChange} onStop={onStop} />
      </div>
      {completed && result.error && <pre className="error-box">{result.error}</pre>}
      {example && (
        <div className="debug-io-grid">
          <div>
            <small>输入</small>
            <pre>{example.input}</pre>
          </div>
          <div>
            <small>预期输出</small>
            <pre>{example.output}</pre>
          </div>
        </div>
      )}
      <div className="variables-table">
        <div className="variables-heading">局部变量</div>
        {variableEntries.length ? variableEntries.map(([name, value]) => (
          <div className="variable-row" key={name}>
            <code>{name}</code>
            <pre>{value}</pre>
          </div>
        )) : <div className="variables-empty">当前行没有局部变量</div>}
      </div>
      <div className="variables-table">
        <div className="variables-heading">全局变量</div>
        {globalEntries.length ? globalEntries.map(([name, value]) => (
          <div className="variable-row" key={name}>
            <code>{name}</code>
            <pre>{value}</pre>
          </div>
        )) : <div className="variables-empty">当前代码没有用户定义的全局变量</div>}
      </div>
      {completed && result.stdout && (
        <div className="stdout">
          <strong>输出</strong>
          <pre>{result.stdout}</pre>
        </div>
      )}
    </div>
  )
}
