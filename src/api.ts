export type Difficulty = 'easy' | 'medium' | 'hard'

export type ProblemMeta = {
  id: string
  title: string
  category: string
  difficulty: Difficulty
  tags: string[]
  order: number
}

export type ProblemDetail = ProblemMeta & {
  readme: string
  examples: Array<{
    input: string
    output: string
    explanation: string
  }>
  starter: string
  solution: string
  source: string
}

export type JudgeCase = {
  name: string
  passed: boolean
  reason?: string
  error?: string
  actual?: string
  expected?: string
  input?: string
  elapsed_ms: number
}

export type JudgeResult = {
  score: number
  all_passed: boolean
  cases: JudgeCase[]
  stdout?: string
  error?: string
  mode?: 'run' | 'submit'
}

export type DebugEvent = {
  line: number
  function: string
  variables: Record<string, string>
  globals: Record<string, string>
  breakpoint: boolean
  depth: number
}

export type DebugResult = {
  events: DebugEvent[]
  stdout?: string
  error?: string
  case?: string
  breakpoints?: number[]
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `请求失败：${response.status}`)
  }
  return response.json() as Promise<T>
}

export function fetchProblems() {
  return request<ProblemMeta[]>('/api/problems')
}

export function fetchProblem(id: string) {
  return request<ProblemDetail>(`/api/problems/${encodeURIComponent(id)}`)
}

export function submitCode(id: string, code: string) {
  return request<JudgeResult>(`/api/problems/${encodeURIComponent(id)}/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code }),
  })
}

export function runCode(id: string, code: string, caseCount: number) {
  return request<JudgeResult>(`/api/problems/${encodeURIComponent(id)}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, caseCount }),
  })
}

export function debugCode(id: string, code: string, breakpoints: number[]) {
  return request<DebugResult>(`/api/problems/${encodeURIComponent(id)}/debug`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, breakpoints }),
  })
}
