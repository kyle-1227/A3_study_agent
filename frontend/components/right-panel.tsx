"use client"

import { useState, useEffect, useRef, useMemo, useCallback } from "react"
import { ChevronLeft, ChevronRight, ChevronUp, ChevronDown, GripHorizontal } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import {
  ReactFlow,
  MiniMap,
  Background,
  Handle,
  type Node as RFNode,
  type Edge as RFEdge,
  type NodeProps,
  Position,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import dagre from "@dagrejs/dagre"

// ── Exported types consumed by page.tsx ────────────────────────────

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

// ── Human-readable node labels ─────────────────────────────────────

const NODE_LABELS: Record<string, string> = {
  supervisor: "意图分类",
  academic_router: "学术路由",
  search_query_rewriter: "查询改写",
  rag_retrieve: "Local RAG",
  web_search: "Tavily Web Search",
  evidence_judge: "Evidence Judge",
  generate_answer: "回答生成",
  evaluate_hallucination: "幻觉评估",
  rewrite_query: "查询改写",
  gather_planning_context: "规划上下文检索",
  gather_intel: "情报收集",
  drafter: "计划起草",
  reviewer_academic: "学术审查",
  reviewer_emotional: "情绪审查",
  consensus_check: "共识检查",
  adv_rewrite: "计划修订",
  plan_output: "计划输出",
  feedback_router: "反馈分类",
  plan_tweak: "计划微调",
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
  review_doc_planner: "复习文档规划",
  review_doc_agent: "复习文档生成",
  review_doc_reviewer: "复习文档审查",
  review_doc_rewrite: "复习文档修订",
  review_doc_output: "复习文档输出",
  emotional_response: "情绪支持",
  handle_unknown: "未知意图",
}

// ── Main component ─────────────────────────────────────────────────

export function RightPanel({ logs, nodeEvents, tokenUsage, isInterrupted }: RightPanelProps) {
  const [isCollapsed, setIsCollapsed] = useState(true)
  const [viewTab, setViewTab] = useState<"trail" | "graph">("trail")
  const [isLogsCollapsed, setIsLogsCollapsed] = useState(false)
  const [splitPct, setSplitPct] = useState(65)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef(false)
  const startYRef = useRef(0)
  const startSplitRef = useRef(65)

  // Auto-scroll logs to bottom when new entries arrive
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  // ── Drag-to-resize divider ──────────────────────────────────────
  const handleDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    draggingRef.current = true
    startYRef.current = e.clientY
    startSplitRef.current = splitPct
    document.body.style.cursor = "row-resize"
    document.body.style.userSelect = "none"
  }, [splitPct])

  useEffect(() => {
    const handleDragMove = (e: MouseEvent) => {
      if (!draggingRef.current || !panelRef.current) return
      const rect = panelRef.current.getBoundingClientRect()
      const deltaY = e.clientY - startYRef.current
      const deltaPct = (deltaY / rect.height) * 100
      const newPct = Math.min(90, Math.max(20, startSplitRef.current + deltaPct))
      setSplitPct(newPct)
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
    <div
      ref={panelRef}
      className={cn(
        "relative h-full border-l border-border bg-sidebar flex flex-col overflow-hidden",
        "transition-all duration-300 ease-in-out select-none",
        isCollapsed ? "w-12" : "w-80"
      )}
    >
      {isCollapsed ? (
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(false)}
          className="absolute top-4 left-1 h-8 w-8 text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
      ) : (
        <>
          {/* Collapse Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(true)}
            className="absolute top-4 left-2 h-8 w-8 text-muted-foreground hover:text-foreground z-10"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>

          {/* ── Upper Section: Node Trail / Graph + status bars ───── */}
          <div
            className="flex flex-col min-h-0"
            style={
              isLogsCollapsed
                ? { flex: "1 1 0%" }
                : { height: `${splitPct}%`, flexShrink: 0 }
            }
          >
            {/* Tabs + Scroll Content */}
            <div className="p-4 pl-12 flex-1 flex flex-col min-h-0">
              <div className="flex items-center gap-2 mb-3">
                <button
                  onClick={() => setViewTab("trail")}
                  className={cn(
                    "text-xs px-2 py-1 rounded transition-colors",
                    viewTab === "trail"
                      ? "bg-[#3D5A40] text-white"
                      : "text-[#3D5A40] hover:bg-[#3D5A40]/10"
                  )}
                >
                  Node Trail
                </button>
                <button
                  onClick={() => setViewTab("graph")}
                  className={cn(
                    "text-xs px-2 py-1 rounded transition-colors",
                    viewTab === "graph"
                      ? "bg-[#3D5A40] text-white"
                      : "text-[#3D5A40] hover:bg-[#3D5A40]/10"
                  )}
                >
                  Graph View
                </button>
              </div>

              <ScrollArea className="flex-1">
                {viewTab === "trail" ? (
                  <div className="bg-[#F5F3E8] rounded-lg p-6">
                    {nodeEvents.length === 0 ? (
                      <div className="flex flex-col items-center gap-3">
                        <IdleNode label="等待请求..." />
                        <p className="text-xs text-muted-foreground mt-2">
                          发送消息后，推理路径将实时显示
                        </p>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-1">
                        {nodeEvents.map((event, idx) => (
                          <div key={`${event.node}-${idx}`} className="flex flex-col items-center">
                            {idx > 0 && <ArrowDown />}
                            <TraversalNode event={event} />
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="bg-[#F5F3E8] rounded-lg" style={{ height: 420 }}>
                    <GraphDAGView nodeEvents={nodeEvents} />
                  </div>
                )}
              </ScrollArea>
            </div>

            {/* HIL Interrupt Status */}
            {isInterrupted && (
              <div className="px-4 py-2 pl-12 border-t border-[#E8A87C] bg-[#FFF9E6]">
                <p className="text-xs font-medium text-[#5C3D2E] flex items-center gap-1.5">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#E8A87C] opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-[#E8A87C]" />
                  </span>
                  等待用户审批
                </p>
              </div>
            )}

            {/* Token Usage Counter */}
            {tokenUsage.total > 0 && (
              <div className="px-4 py-2 pl-12 border-t border-border bg-[#F5F3E8]/50">
                <p className="text-xs font-mono text-[#3D5A40]">
                  Tokens: {tokenUsage.total}
                  <span className="text-muted-foreground ml-1">
                    (in: {tokenUsage.input} / out: {tokenUsage.output})
                  </span>
                </p>
              </div>
            )}
          </div>

          {/* ── Draggable Divider ─────────────────────────────────── */}
          {!isLogsCollapsed && (
            <div
              onMouseDown={handleDragStart}
              className={cn(
                "flex-shrink-0 h-6 cursor-row-resize z-10",
                "flex items-center justify-center",
                "bg-[#E8E5D8]/60 hover:bg-[#3D5A40]/15 border-y border-[#E8E5D8]",
                "group transition-colors"
              )}
            >
              <GripHorizontal className="h-3 w-6 text-muted-foreground/50 group-hover:text-[#3D5A40]/70 transition-colors" />
            </div>
          )}

          {/* ── System Logs (collapsed/expanded) ──────────────────── */}
          {!isLogsCollapsed ? (
            <div className="flex-1 flex flex-col overflow-hidden min-h-0">
              <div className="px-4 py-2 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-[#3D5A40]">系统 Logs</h3>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsLogsCollapsed(true)}
                  className="h-6 w-6 text-muted-foreground hover:text-foreground"
                >
                  <ChevronDown className="h-3.5 w-3.5" />
                </Button>
              </div>
              <ScrollArea className="flex-1 px-4">
                <div className="flex flex-col gap-1 pb-4">
                  {logs.map((log, index) => (
                    <div
                      key={index}
                      className={cn(
                        "text-xs font-mono py-1 px-2 rounded flex gap-2",
                        log.type === "error" && "text-[#D97B6C] bg-[#D97B6C]/10",
                        log.type === "info" && "text-muted-foreground bg-[#F5F3E8]",
                        log.type === "warning" && "text-[#B8860B] bg-[#FFCC99]/20",
                        log.type === "perf" && "text-[#4A90D9] bg-[#4A90D9]/10",
                        log.type === "usage" && "text-[#8B5CF6] bg-[#8B5CF6]/10"
                      )}
                    >
                      <span className="opacity-50 shrink-0">{log.ts}</span>
                      <span>{log.message}</span>
                    </div>
                  ))}
                  <div ref={logsEndRef} />
                </div>
              </ScrollArea>
            </div>
          ) : (
            <button
              onClick={() => setIsLogsCollapsed(false)}
              className="flex-shrink-0 px-4 py-2 border-t border-border flex items-center gap-1.5 text-xs text-muted-foreground hover:text-[#3D5A40] hover:bg-[#F5F3E8]/50 transition-colors"
            >
              <ChevronUp className="h-3 w-3" />
              系统 Logs
              {logs.length > 0 && (
                <span className="bg-[#3D5A40]/10 text-[#3D5A40] text-[10px] px-1.5 py-0.5 rounded-full">
                  {logs.length}
                </span>
              )}
            </button>
          )}
        </>
      )}
    </div>
  )
}

// ── Graph DAG View (React Flow + dagre auto-layout) ──────────────

interface DagEdgeDef {
  from: string
  to: string
  retry?: boolean
}

const DAG_NODE_IDS = [
  "supervisor",
  "academic_router",
  "gather_planning_context",
  "emotional_response",
  "handle_unknown",
  "search_query_rewriter",
  "rag_retrieve",
  "web_search",
  "evidence_judge",
  "gather_intel",
  "generate_answer",
  "drafter",
  "evaluate_hallucination",
  "reviewer_academic",
  "reviewer_emotional",
  "rewrite_query",
  "consensus_check",
  "adv_rewrite",
  "plan_output",
  "feedback_router",
  "plan_tweak",
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
]

const DAG_EDGE_DEFS: DagEdgeDef[] = [
  // Supervisor routing
  { from: "supervisor", to: "search_query_rewriter" },
  { from: "supervisor", to: "emotional_response" },
  { from: "supervisor", to: "handle_unknown" },
  // Shared query rewrite routes to academic or planning
  { from: "search_query_rewriter", to: "academic_router" },
  { from: "search_query_rewriter", to: "gather_planning_context" },
  // Academic branch
  { from: "academic_router", to: "rag_retrieve" },
  { from: "academic_router", to: "web_search" },
  { from: "rag_retrieve", to: "evidence_judge" },
  { from: "web_search", to: "evidence_judge" },
  { from: "evidence_judge", to: "generate_answer" },
  { from: "evidence_judge", to: "mindmap_planner" },
  { from: "evidence_judge", to: "exercise_planner" },
  { from: "evidence_judge", to: "review_doc_planner" },
  { from: "mindmap_planner", to: "mindmap_agent" },
  { from: "mindmap_agent", to: "mindmap_reviewer" },
  { from: "mindmap_reviewer", to: "mindmap_output" },
  { from: "mindmap_reviewer", to: "mindmap_rewrite", retry: true },
  { from: "mindmap_rewrite", to: "mindmap_agent", retry: true },
  { from: "exercise_planner", to: "exercise_agent" },
  { from: "exercise_agent", to: "exercise_reviewer" },
  { from: "exercise_reviewer", to: "exercise_output" },
  { from: "exercise_reviewer", to: "exercise_rewrite", retry: true },
  { from: "exercise_rewrite", to: "exercise_agent", retry: true },
  { from: "review_doc_planner", to: "review_doc_agent" },
  { from: "review_doc_agent", to: "review_doc_reviewer" },
  { from: "review_doc_reviewer", to: "review_doc_output" },
  { from: "review_doc_reviewer", to: "review_doc_rewrite", retry: true },
  { from: "review_doc_rewrite", to: "review_doc_agent", retry: true },
  { from: "generate_answer", to: "evaluate_hallucination" },
  { from: "evaluate_hallucination", to: "rewrite_query" },
  { from: "rewrite_query", to: "academic_router", retry: true },
  // Planning branch
  { from: "gather_planning_context", to: "gather_intel" },
  { from: "gather_intel", to: "drafter" },
  { from: "drafter", to: "reviewer_academic" },
  { from: "drafter", to: "reviewer_emotional" },
  { from: "reviewer_academic", to: "consensus_check" },
  { from: "reviewer_emotional", to: "consensus_check" },
  { from: "consensus_check", to: "adv_rewrite" },
  { from: "consensus_check", to: "plan_output" },
  { from: "adv_rewrite", to: "drafter", retry: true },
  // Feedback loop
  { from: "plan_output", to: "feedback_router" },
  { from: "feedback_router", to: "plan_tweak" },
  { from: "feedback_router", to: "drafter", retry: true },
  { from: "plan_tweak", to: "plan_output" },
]

const NODE_WIDTH = 90
const NODE_HEIGHT = 36
type DagNodeState = "idle" | "running" | "done" | "error"

function traversedEdgeIds(nodeEvents: NodeEvent[]): Set<string> {
  const seenNodes = new Set<string>()
  const traversed = new Set<string>()
  for (const event of nodeEvents) {
    for (const edge of DAG_EDGE_DEFS) {
      if (edge.to !== event.node) continue
      if (edge.from === "academic_router" || seenNodes.has(edge.from)) {
        traversed.add(`${edge.from}-${edge.to}`)
      }
    }
    seenNodes.add(event.node)
  }
  return traversed
}

function buildLayoutedElements(
  nodeStates: Map<string, { state: DagNodeState; durationMs?: number; error?: string }>,
  traversedEdges: Set<string>,
): { nodes: RFNode[]; edges: RFEdge[] } {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: "TB", nodesep: 30, ranksep: 40, marginx: 10, marginy: 10 })

  for (const id of DAG_NODE_IDS) {
    g.setNode(id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  }
  for (const edge of DAG_EDGE_DEFS) {
    g.setEdge(edge.from, edge.to)
  }

  dagre.layout(g)

  const nodes: RFNode[] = DAG_NODE_IDS.map((id) => {
    const pos = g.node(id)
    const ns = nodeStates.get(id) ?? { state: "idle" as const }
    return {
      id,
      type: "dagNode",
      position: { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 },
      data: { label: NODE_LABELS[id] || id, state: ns.state, durationMs: ns.durationMs },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    }
  })

  const edges: RFEdge[] = DAG_EDGE_DEFS.map((edge) => {
    const edgeId = `${edge.from}-${edge.to}`
    const active = traversedEdges.has(edgeId)
    return {
      id: edgeId,
      source: edge.from,
      target: edge.to,
      type: "smoothstep",
      style: {
        stroke: edge.retry ? (active ? "#D97B6C" : "#7A9E7E") : (active ? "#3D5A40" : "#7A9E7E"),
        strokeWidth: active ? 1.5 : 1,
        strokeDasharray: edge.retry ? "5 3" : (active ? "none" : "4 2"),
        opacity: active ? 1 : 0.4,
      },
      animated: edge.retry && active,
      label: edge.retry ? "retry" : undefined,
      labelStyle: edge.retry ? { fontSize: 8, fill: "#D97B6C" } : undefined,
    }
  })

  return { nodes, edges }
}

function DagNodeComponent({ data }: NodeProps) {
  const { label, state, durationMs, error } = data as {
    label: string
    state: DagNodeState
    durationMs?: number
    error?: string
  }
  return (
    <>
      <Handle type="target" position={Position.Top} className="!w-1 !h-1 !min-w-0 !min-h-0 !bg-transparent !border-0" />
      <div
        className={cn(
          "rounded border text-center flex flex-col items-center justify-center px-1",
          "transition-all duration-300",
          state === "idle" &&
            "border-dashed border-[#7A9E7E]/50 bg-white/80 text-muted-foreground",
          state === "running" &&
            "border-[#E8A87C] bg-[#FFCC99] text-[#5C3D2E] font-semibold animate-pulse",
          state === "done" &&
            "border-[#3D5A40] bg-[#3D5A40]/10 text-[#3D5A40]",
          state === "error" &&
            "border-[#D97B6C] bg-[#D97B6C]/10 text-[#9F3A2F] font-semibold"
        )}
        style={{ width: NODE_WIDTH, height: NODE_HEIGHT }}
      >
        <span className="text-[9px] leading-tight truncate w-full">{label}</span>
        {state === "done" && durationMs != null && (
          <span className="text-[7px] opacity-60 leading-none">{durationMs}ms</span>
        )}
        {state === "error" && (
          <span className="text-[7px] opacity-70 leading-none truncate w-full">{error || "error"}</span>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} className="!w-1 !h-1 !min-w-0 !min-h-0 !bg-transparent !border-0" />
    </>
  )
}

const rfNodeTypes = { dagNode: DagNodeComponent }

function GraphDAGView({ nodeEvents }: { nodeEvents: NodeEvent[] }) {
  const nodeStates = useMemo(() => {
    const states = new Map<string, { state: DagNodeState; durationMs?: number; error?: string }>()
    for (const id of DAG_NODE_IDS) {
      let found: NodeEvent | undefined
      for (let i = nodeEvents.length - 1; i >= 0; i--) {
        if (nodeEvents[i].node === id) {
          found = nodeEvents[i]
          break
        }
      }
      if (!found) states.set(id, { state: "idle" })
      else if (found.status === "running") states.set(id, { state: "running" })
      else states.set(id, { state: found.status, durationMs: found.durationMs, error: found.error })
    }
    return states
  }, [nodeEvents])

  const traversedEdges = useMemo(() => traversedEdgeIds(nodeEvents), [nodeEvents])
  const { nodes, edges } = useMemo(
    () => buildLayoutedElements(nodeStates, traversedEdges),
    [nodeStates, traversedEdges],
  )

  return (
    <div style={{ width: "100%", height: 420 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={rfNodeTypes}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        proOptions={{ hideAttribution: true }}
        minZoom={0.3}
        maxZoom={2}
      >
        <MiniMap
          nodeStrokeWidth={1}
          nodeColor={(n) => {
            const s = (n.data as any)?.state
            if (s === "running") return "#FFCC99"
            if (s === "done") return "#3D5A40"
            if (s === "error") return "#D97B6C"
            return "#E8E5D8"
          }}
          style={{ height: 60, width: 80 }}
        />
        <Background gap={16} size={0.5} color="#7A9E7E" />
      </ReactFlow>
    </div>
  )
}

// ── Sub-components ─────────────────────────────────────────────────

function TraversalNode({ event }: { event: NodeEvent }) {
  const label = NODE_LABELS[event.node] || event.node
  const isRunning = event.status === "running"
  const isError = event.status === "error"

  return (
    <div
      className={cn(
        "px-4 py-2 rounded-lg border-2 text-xs font-medium w-40 text-center",
        "transition-all duration-300",
        isRunning
          ? "bg-[#FFCC99] border-[#E8A87C] text-[#5C3D2E] font-semibold animate-pulse"
          : isError
            ? "bg-[#D97B6C]/10 border-[#D97B6C] text-[#A5483D]"
            : "bg-[#3D5A40]/10 border-[#3D5A40] text-[#3D5A40]"
      )}
    >
      <div className="flex items-center justify-center gap-1.5">
        {isRunning ? (
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#E8A87C] opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-[#E8A87C]" />
          </span>
        ) : isError ? (
          <svg className="h-3 w-3 text-[#A5483D]" viewBox="0 0 12 12" fill="none">
            <path d="M3 3l6 6M9 3L3 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        ) : (
          <svg className="h-3 w-3 text-[#3D5A40]" viewBox="0 0 12 12" fill="none">
            <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
        {label}
        {event.synthetic ? <span className="text-[9px] uppercase opacity-70">synthetic</span> : null}
      </div>
      <div className="text-[10px] opacity-60 mt-0.5">
        {isRunning
          ? event.ts
          : `${event.ts} → ${event.endTs ?? ""}${event.durationMs != null ? ` (${event.durationMs}ms)` : ""}`}
      </div>
      {isError && event.error ? (
        <div className="mt-1 line-clamp-2 text-[10px] leading-tight opacity-80" title={event.error}>
          {event.error}
        </div>
      ) : null}
    </div>
  )
}

function IdleNode({ label }: { label: string }) {
  return (
    <div className="px-4 py-2 rounded-lg border-2 border-dashed border-[#7A9E7E]/50 text-xs font-medium text-muted-foreground">
      {label}
    </div>
  )
}

function ArrowDown() {
  return (
    <div className="flex flex-col items-center text-[#7A9E7E] my-0.5">
      <div className="w-0.5 h-3 bg-[#7A9E7E]/50" />
      <svg width="8" height="6" viewBox="0 0 8 6" fill="currentColor" className="opacity-70">
        <path d="M4 6L0 0h8L4 6z" />
      </svg>
    </div>
  )
}
