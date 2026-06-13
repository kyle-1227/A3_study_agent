"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import {
  ArrowDown as ArrowDownIcon,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  GripHorizontal,
  X,
} from "lucide-react"
import dagre from "@dagrejs/dagre"
import {
  Background,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  type Edge as RFEdge,
  type Node as RFNode,
  type NodeProps,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"

import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

export interface LogEntry {
  type: "info" | "error" | "warning" | "perf" | "usage"
  message: string
  ts: string
}

export interface NodeEvent {
  node: string
  status: "running" | "done" | "error"
  ts: string
  endTs?: string
  durationMs?: number
  error?: string
  synthetic?: boolean
}

interface RightPanelProps {
  logs: LogEntry[]
  nodeEvents: NodeEvent[]
  tokenUsage: { input: number; output: number; total: number }
  isInterrupted?: boolean
}

const NODE_LABELS: Record<string, string> = {
  supervisor: "意图识别",
  emotional_response: "学业支持",
  handle_unknown: "范围确认",
  search_query_rewriter: "查询改写",
  academic_router: "学术路由",
  rag_retrieve: "Local RAG",
  web_search: "Tavily 搜索",
  evidence_judge: "证据评审",
  evidence_summary_output: "证据摘要输出",
  generate_answer: "回答生成",
  evaluate_hallucination: "可信校验",
  rewrite_query: "查询重写",
  mindmap_planner: "导图规划",
  mindmap_agent: "JSON Tree",
  mindmap_reviewer: "导图审查",
  mindmap_rewrite: "导图重写",
  mindmap_output: "导图导出",
  exercise_planner: "练习规划",
  exercise_agent: "题目生成",
  exercise_reviewer: "题目审查",
  exercise_rewrite: "题目修订",
  exercise_output: "练习输出",
  review_doc_planner: "文档规划",
  review_doc_agent: "文档生成",
  review_doc_reviewer: "文档审查",
  review_doc_rewrite: "文档修订",
  review_doc_output: "文档输出",
  study_plan_emotional_intel: "情绪画像",
  study_plan_planner: "计划规划",
  study_plan_agent: "计划生成",
  study_plan_reviewer_academic: "学术审查",
  study_plan_reviewer_emotional: "负担审查",
  study_plan_consensus: "共识检查",
  study_plan_rewrite: "计划修订",
  study_plan_output: "计划输出",
}

const DAG_NODE_IDS = [
  "supervisor",
  "emotional_response",
  "handle_unknown",
  "search_query_rewriter",
  "academic_router",
  "rag_retrieve",
  "web_search",
  "evidence_judge",
  "evidence_summary_output",
  "generate_answer",
  "evaluate_hallucination",
  "rewrite_query",
  "mindmap_planner",
  "mindmap_agent",
  "mindmap_reviewer",
  "mindmap_rewrite",
  "mindmap_output",
  "exercise_planner",
  "exercise_agent",
  "exercise_reviewer",
  "exercise_rewrite",
  "exercise_output",
  "review_doc_planner",
  "review_doc_agent",
  "review_doc_reviewer",
  "review_doc_rewrite",
  "review_doc_output",
  "study_plan_emotional_intel",
  "study_plan_planner",
  "study_plan_agent",
  "study_plan_reviewer_academic",
  "study_plan_reviewer_emotional",
  "study_plan_consensus",
  "study_plan_rewrite",
  "study_plan_output",
] as const

interface DagEdgeDef {
  from: string
  to: string
  retry?: boolean
}

const DAG_EDGE_DEFS: DagEdgeDef[] = [
  { from: "supervisor", to: "emotional_response" },
  { from: "supervisor", to: "handle_unknown" },
  { from: "supervisor", to: "search_query_rewriter" },
  { from: "search_query_rewriter", to: "academic_router" },
  { from: "academic_router", to: "rag_retrieve" },
  { from: "academic_router", to: "web_search" },
  { from: "rag_retrieve", to: "evidence_judge" },
  { from: "web_search", to: "evidence_judge" },
  { from: "evidence_judge", to: "evidence_summary_output" },
  { from: "evidence_judge", to: "generate_answer" },
  { from: "evidence_judge", to: "mindmap_planner" },
  { from: "evidence_judge", to: "exercise_planner" },
  { from: "evidence_judge", to: "review_doc_planner" },
  { from: "evidence_judge", to: "study_plan_emotional_intel" },
  { from: "generate_answer", to: "evaluate_hallucination" },
  { from: "evaluate_hallucination", to: "rewrite_query", retry: true },
  { from: "rewrite_query", to: "academic_router", retry: true },
  { from: "mindmap_planner", to: "mindmap_agent" },
  { from: "mindmap_agent", to: "mindmap_reviewer" },
  { from: "mindmap_reviewer", to: "mindmap_rewrite", retry: true },
  { from: "mindmap_rewrite", to: "mindmap_agent", retry: true },
  { from: "mindmap_reviewer", to: "mindmap_output" },
  { from: "exercise_planner", to: "exercise_agent" },
  { from: "exercise_agent", to: "exercise_reviewer" },
  { from: "exercise_reviewer", to: "exercise_rewrite", retry: true },
  { from: "exercise_rewrite", to: "exercise_agent", retry: true },
  { from: "exercise_reviewer", to: "exercise_output" },
  { from: "review_doc_planner", to: "review_doc_agent" },
  { from: "review_doc_agent", to: "review_doc_reviewer" },
  { from: "review_doc_reviewer", to: "review_doc_rewrite", retry: true },
  { from: "review_doc_rewrite", to: "review_doc_agent", retry: true },
  { from: "review_doc_reviewer", to: "review_doc_output" },
  { from: "study_plan_emotional_intel", to: "study_plan_planner" },
  { from: "study_plan_planner", to: "study_plan_agent" },
  { from: "study_plan_agent", to: "study_plan_reviewer_academic" },
  { from: "study_plan_agent", to: "study_plan_reviewer_emotional" },
  { from: "study_plan_reviewer_academic", to: "study_plan_consensus" },
  { from: "study_plan_reviewer_emotional", to: "study_plan_consensus" },
  { from: "study_plan_consensus", to: "study_plan_rewrite", retry: true },
  { from: "study_plan_rewrite", to: "study_plan_agent", retry: true },
  { from: "study_plan_consensus", to: "study_plan_output" },
]

const NODE_WIDTH = 96
const NODE_HEIGHT = 38

type DagNodeState = "idle" | "running" | "done" | "error"

export function RightPanel({ logs, nodeEvents, tokenUsage, isInterrupted }: RightPanelProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [viewTab, setViewTab] = useState<"trail" | "graph">("trail")
  const [isLogsCollapsed, setIsLogsCollapsed] = useState(false)
  const [splitPct, setSplitPct] = useState(65)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef(false)
  const startYRef = useRef(0)
  const startSplitRef = useRef(65)

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  const handleDragStart = useCallback((event: React.MouseEvent) => {
    event.preventDefault()
    draggingRef.current = true
    startYRef.current = event.clientY
    startSplitRef.current = splitPct
    document.body.style.cursor = "row-resize"
    document.body.style.userSelect = "none"
  }, [splitPct])

  useEffect(() => {
    const handleDragMove = (event: MouseEvent) => {
      if (!draggingRef.current || !panelRef.current) return
      const rect = panelRef.current.getBoundingClientRect()
      const deltaY = event.clientY - startYRef.current
      const deltaPct = (deltaY / rect.height) * 100
      setSplitPct(Math.min(90, Math.max(20, startSplitRef.current + deltaPct)))
    }
    const handleDragEnd = () => {
      if (!draggingRef.current) return
      draggingRef.current = false
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }
    window.addEventListener("mousemove", handleDragMove)
    window.addEventListener("mouseup", handleDragEnd)
    return () => {
      window.removeEventListener("mousemove", handleDragMove)
      window.removeEventListener("mouseup", handleDragEnd)
    }
  }, [])

  return (
    <aside
      ref={panelRef}
      className={cn(
        "relative flex h-[100dvh] shrink-0 self-stretch flex-col overflow-hidden border-l border-sidebar-border bg-sidebar text-sidebar-foreground",
        "select-none transition-[width] duration-200 ease-out",
        isCollapsed ? "w-12" : "w-80",
      )}
      aria-label="运行状态面板"
    >
      {isCollapsed ? (
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(false)}
          className="absolute left-1 top-4 h-8 w-8 text-muted-foreground hover:text-foreground"
          title="展开运行状态面板"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
      ) : (
        <>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(true)}
            className="absolute left-2 top-4 z-10 h-8 w-8 text-muted-foreground hover:text-foreground"
            title="收起运行状态面板"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>

          <div
            className="flex min-h-0 flex-col"
            style={isLogsCollapsed ? { flex: "1 1 0%" } : { height: `${splitPct}%`, flexShrink: 0 }}
          >
            <div className="flex min-h-0 flex-1 flex-col p-4 pl-12">
              <div className="mb-3 flex items-center gap-2">
                <PanelTab active={viewTab === "trail"} onClick={() => setViewTab("trail")}>
                  Node Trail
                </PanelTab>
                <PanelTab active={viewTab === "graph"} onClick={() => setViewTab("graph")}>
                  Graph View
                </PanelTab>
              </div>

              <div className="flex-1 min-h-0">
                <ScrollArea className="h-full">
                  {viewTab === "trail" ? (
                    <div className="rounded-lg bg-[var(--surface-muted)] p-5">
                      {nodeEvents.length === 0 ? (
                        <div className="flex flex-col items-center gap-3">
                          <IdleNode label="等待请求..." />
                          <p className="mt-1 text-center text-xs leading-relaxed text-muted-foreground">
                            发送消息后，推理路径会在这里实时显示。
                          </p>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center gap-1">
                          {nodeEvents.map((event, index) => (
                            <div key={`${event.node}-${index}`} className="flex flex-col items-center">
                              {index > 0 && <ArrowDown />}
                              <TraversalNode event={event} />
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="rounded-lg bg-[var(--surface-muted)]" style={{ height: 430 }}>
                      <GraphDAGView nodeEvents={nodeEvents} />
                    </div>
                  )}
                </ScrollArea>
              </div>
            </div>

            {isInterrupted && (
              <div className="border-t border-[var(--warning)] bg-[var(--warning-soft)] px-4 py-2 pl-12">
                <p className="flex items-center gap-1.5 text-xs font-medium text-[var(--warning)]">
                  <span className="relative flex h-2 w-2">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[var(--warning)] opacity-60" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-[var(--warning)]" />
                  </span>
                  等待用户审核
                </p>
              </div>
            )}

            {tokenUsage.total > 0 && (
              <div className="border-t border-border bg-[var(--surface-muted)]/70 px-4 py-2 pl-12">
                <p className="font-mono text-xs text-primary">
                  Tokens: {tokenUsage.total}
                  <span className="ml-1 text-muted-foreground">
                    (in: {tokenUsage.input} / out: {tokenUsage.output})
                  </span>
                </p>
              </div>
            )}
          </div>

          {!isLogsCollapsed && (
            <div
              onMouseDown={handleDragStart}
              className="group z-10 flex h-6 shrink-0 cursor-row-resize items-center justify-center border-y border-border bg-border/40 transition-colors hover:bg-primary/10"
            >
              <GripHorizontal className="h-3 w-6 text-muted-foreground/60 transition-colors group-hover:text-primary" />
            </div>
          )}

          {!isLogsCollapsed ? (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
              <div className="flex items-center justify-between px-4 py-2">
                <h3 className="text-sm font-semibold text-primary">系统日志</h3>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsLogsCollapsed(true)}
                  className="h-6 w-6 text-muted-foreground hover:text-foreground"
                  title="收起系统日志"
                >
                  <ChevronDown className="h-3.5 w-3.5" />
                </Button>
              </div>
              <div className="flex-1 min-h-0">
                <ScrollArea className="h-full px-4">
                  <div className="flex flex-col gap-1 pb-4">
                    {logs.map((log, index) => (
                      <div
                        key={index}
                        className={cn(
                          "flex gap-2 rounded px-2 py-1 font-mono text-xs",
                          log.type === "error" && "bg-[var(--danger-soft)] text-[var(--danger)]",
                          log.type === "info" && "bg-[var(--surface-muted)] text-muted-foreground",
                          log.type === "warning" && "bg-[var(--warning-soft)] text-[var(--warning)]",
                          log.type === "perf" && "bg-[var(--info-soft)] text-[var(--info)]",
                          log.type === "usage" && "bg-primary/10 text-primary",
                        )}
                      >
                        <span className="shrink-0 opacity-55" suppressHydrationWarning>{log.ts}</span>
                        <span className="min-w-0 break-words">{log.message}</span>
                      </div>
                    ))}
                    <div ref={logsEndRef} />
                  </div>
                </ScrollArea>
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setIsLogsCollapsed(false)}
              className="flex shrink-0 items-center gap-1.5 border-t border-border px-4 py-2 text-xs text-muted-foreground transition-colors hover:bg-[var(--surface-muted)] hover:text-primary"
            >
              <ChevronUp className="h-3 w-3" />
              系统日志
              {logs.length > 0 && (
                <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">
                  {logs.length}
                </span>
              )}
            </button>
          )}
        </>
      )}
    </aside>
  )
}

function PanelTab({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-md px-2 py-1 text-xs transition-colors",
        active ? "bg-primary text-primary-foreground" : "text-primary hover:bg-primary/10",
      )}
    >
      {children}
    </button>
  )
}

function TraversalNode({ event }: { event: NodeEvent }) {
  const label = NODE_LABELS[event.node] || event.node
  const isRunning = event.status === "running"
  const isError = event.status === "error"

  return (
    <div
      className={cn(
        "w-[10.5rem] rounded-lg border-2 px-3 py-2 text-center text-xs font-medium transition-all duration-200",
        isRunning
          ? "animate-pulse border-[var(--warning)] bg-[var(--warning-soft)] text-[var(--warning)]"
          : isError
            ? "border-[var(--danger)] bg-[var(--danger-soft)] text-[var(--danger)]"
            : "border-primary bg-primary/10 text-primary",
      )}
    >
      <div className="flex items-center justify-center gap-1.5">
        {isRunning ? (
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[var(--warning)] opacity-60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-[var(--warning)]" />
          </span>
        ) : isError ? (
          <X className="h-3 w-3" strokeWidth={2.4} />
        ) : (
          <Check className="h-3 w-3" strokeWidth={2.4} />
        )}
        <span className="truncate">{label}</span>
        {event.synthetic ? <span className="text-[9px] opacity-75">synthetic</span> : null}
      </div>
      <div className="mt-0.5 text-[10px] opacity-65">
        {isRunning
          ? event.ts
          : `${event.ts} -> ${event.endTs ?? ""}${event.durationMs != null ? ` (${event.durationMs}ms)` : ""}`}
      </div>
      {isError && event.error ? (
        <div className="mt-1 line-clamp-2 text-[10px] leading-tight opacity-85" title={event.error}>
          {event.error}
        </div>
      ) : null}
    </div>
  )
}

function IdleNode({ label }: { label: string }) {
  return (
    <div className="rounded-lg border-2 border-dashed border-[var(--primary-line)] px-4 py-2 text-xs font-medium text-muted-foreground">
      {label}
    </div>
  )
}

function ArrowDown() {
  return (
    <div className="my-0.5 flex flex-col items-center text-[var(--primary-line)]">
      <div className="h-3 w-0.5 bg-[var(--primary-line)]" />
      <ArrowDownIcon className="h-2.5 w-2.5 opacity-80" strokeWidth={2.2} />
    </div>
  )
}

function traversedEdgeIds(nodeEvents: NodeEvent[]): Set<string> {
  const traversed = new Set<string>()
  const seenNodes = new Set<string>()
  for (const event of nodeEvents) {
    for (const edge of DAG_EDGE_DEFS) {
      if (edge.to !== event.node) continue
      if (seenNodes.has(edge.from)) {
        traversed.add(`${edge.from}-${edge.to}`)
      }
    }
    if (event.status !== "running") seenNodes.add(event.node)
  }
  return traversed
}

function buildLayoutedElements(
  nodeEvents: NodeEvent[],
  traversedEdges: Set<string>,
): { nodes: RFNode[]; edges: RFEdge[] } {
  const graph = new dagre.graphlib.Graph()
  graph.setDefaultEdgeLabel(() => ({}))
  graph.setGraph({ rankdir: "TB", nodesep: 32, ranksep: 44, marginx: 10, marginy: 10 })

  for (const id of DAG_NODE_IDS) {
    graph.setNode(id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  }
  for (const edge of DAG_EDGE_DEFS) {
    graph.setEdge(edge.from, edge.to)
  }
  dagre.layout(graph)

  const eventByNode = new Map<string, NodeEvent>()
  for (const event of nodeEvents) eventByNode.set(event.node, event)

  const nodes: RFNode[] = DAG_NODE_IDS.map((id) => {
    const event = eventByNode.get(id)
    const state: DagNodeState = event?.status ?? "idle"
    const position = graph.node(id)
    return {
      id,
      type: "dagNode",
      position: { x: position.x - NODE_WIDTH / 2, y: position.y - NODE_HEIGHT / 2 },
      data: {
        label: NODE_LABELS[id] || id,
        state,
        durationMs: event?.durationMs,
        error: event?.error,
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    }
  })

  const edges: RFEdge[] = DAG_EDGE_DEFS.map((edge) => {
    const active = traversedEdges.has(`${edge.from}-${edge.to}`)
    return {
      id: `${edge.from}-${edge.to}`,
      source: edge.from,
      target: edge.to,
      type: "smoothstep",
      style: {
        stroke: edge.retry
          ? active
            ? "var(--danger)"
            : "var(--primary-line)"
          : active
            ? "var(--primary)"
            : "var(--primary-line)",
        strokeWidth: active ? 1.6 : 1,
        strokeDasharray: edge.retry ? "5 3" : active ? "none" : "4 3",
        opacity: active ? 1 : 0.45,
      },
      animated: edge.retry && active,
      label: edge.retry ? "retry" : undefined,
      labelStyle: edge.retry ? { fontSize: 8, fill: "var(--danger)" } : undefined,
    }
  })

  return { nodes, edges }
}

function DagNodeComponent({ data }: NodeProps) {
  const label = String(data.label ?? "")
  const state = (data.state ?? "idle") as DagNodeState
  const durationMs = typeof data.durationMs === "number" ? data.durationMs : undefined
  const error = typeof data.error === "string" ? data.error : undefined

  return (
    <>
      <Handle type="target" position={Position.Top} className="!h-1 !min-h-0 !w-1 !min-w-0 !border-0 !bg-transparent" />
      <div
        className={cn(
          "flex flex-col items-center justify-center rounded-md border px-1 text-center transition-all duration-200",
          state === "idle" && "border-dashed border-[var(--primary-line)] bg-card/80 text-muted-foreground",
          state === "running" && "animate-pulse border-[var(--warning)] bg-[var(--warning-soft)] font-semibold text-[var(--warning)]",
          state === "done" && "border-primary bg-primary/10 text-primary",
          state === "error" && "border-[var(--danger)] bg-[var(--danger-soft)] font-semibold text-[var(--danger)]",
        )}
        style={{ width: NODE_WIDTH, height: NODE_HEIGHT }}
        title={error || label}
      >
        <span className="w-full truncate text-[9px] leading-tight">{label}</span>
        {state === "done" && durationMs != null && (
          <span className="text-[7px] leading-none opacity-65">{durationMs}ms</span>
        )}
        {state === "error" && (
          <span className="w-full truncate text-[7px] leading-none opacity-75">{error || "error"}</span>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} className="!h-1 !min-h-0 !w-1 !min-w-0 !border-0 !bg-transparent" />
    </>
  )
}

const rfNodeTypes = { dagNode: DagNodeComponent }

function GraphDAGView({ nodeEvents }: { nodeEvents: NodeEvent[] }) {
  const traversedEdges = useMemo(() => traversedEdgeIds(nodeEvents), [nodeEvents])
  const { nodes, edges } = useMemo(
    () => buildLayoutedElements(nodeEvents, traversedEdges),
    [nodeEvents, traversedEdges],
  )

  return (
    <div style={{ width: "100%", height: 430 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={rfNodeTypes}
        fitView
        fitViewOptions={{ padding: 0.16 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        proOptions={{ hideAttribution: true }}
        minZoom={0.28}
        maxZoom={2}
      >
        <MiniMap
          nodeStrokeWidth={1}
          nodeColor={(node) => {
            const state = (node.data as any)?.state
            if (state === "running") return "#fff4d8"
            if (state === "done") return "#35593f"
            if (state === "error") return "#c55447"
            return "#ddd9c8"
          }}
          style={{ height: 60, width: 82 }}
        />
        <Background gap={16} size={0.5} color="#b9c8b9" />
      </ReactFlow>
    </div>
  )
}
