"use client"

import { useState, useEffect, useRef, useMemo } from "react"
import { ChevronLeft, ChevronRight } from "lucide-react"
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
  status: "running" | "done"
  ts: string
  endTs?: string
  durationMs?: number
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
  rag_retrieve: "RAG 检索",
  web_search: "网络搜索",
  generate_answer: "回答生成",
  evaluate_hallucination: "幻觉评估",
  rewrite_query: "查询改写",
  search_policy: "政策搜索",
  gather_intel: "情报收集",
  drafter: "计划起草",
  reviewer_academic: "学术审查",
  reviewer_emotional: "情绪审查",
  consensus_check: "共识检查",
  adv_rewrite: "计划修订",
  plan_output: "计划输出",
  feedback_router: "反馈分类",
  plan_tweak: "计划微调",
  emotional_response: "情绪支持",
  handle_unknown: "未知意图",
}

// ── Main component ─────────────────────────────────────────────────

export function RightPanel({ logs, nodeEvents, tokenUsage, isInterrupted }: RightPanelProps) {
  const [isCollapsed, setIsCollapsed] = useState(true)
  const [viewTab, setViewTab] = useState<"trail" | "graph">("trail")
  const logsEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll logs to bottom when new entries arrive
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  return (
    <div
      className={cn(
        "relative h-full border-l border-border bg-sidebar flex flex-col",
        "transition-all duration-300 ease-in-out",
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

          {/* Reasoning Path Visualization - 70% height */}
          <div className="p-4 pl-12 flex-[7] flex flex-col border-b border-border">
            {/* Tab toggle */}
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
            <div className="px-4 py-2 pl-12 border-b border-[#E8A87C] bg-[#FFF9E6]">
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
            <div className="px-4 py-2 pl-12 border-b border-border bg-[#F5F3E8]/50">
              <p className="text-xs font-mono text-[#3D5A40]">
                Tokens: {tokenUsage.total}
                <span className="text-muted-foreground ml-1">
                  (in: {tokenUsage.input} / out: {tokenUsage.output})
                </span>
              </p>
            </div>
          )}

          {/* System Logs - 30% height */}
          <div className="flex-[3] flex flex-col overflow-hidden min-h-0">
            <div className="px-4 py-3">
              <h3 className="text-sm font-semibold text-[#3D5A40]">系统 Logs</h3>
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
  "search_policy",
  "emotional_response",
  "handle_unknown",
  "rag_retrieve",
  "web_search",
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
]

const DAG_EDGE_DEFS: DagEdgeDef[] = [
  // Supervisor routing
  { from: "supervisor", to: "academic_router" },
  { from: "supervisor", to: "search_policy" },
  { from: "supervisor", to: "emotional_response" },
  { from: "supervisor", to: "handle_unknown" },
  // Academic branch
  { from: "academic_router", to: "rag_retrieve" },
  { from: "academic_router", to: "web_search" },
  { from: "rag_retrieve", to: "generate_answer" },
  { from: "web_search", to: "generate_answer" },
  { from: "generate_answer", to: "evaluate_hallucination" },
  { from: "evaluate_hallucination", to: "rewrite_query" },
  { from: "rewrite_query", to: "academic_router", retry: true },
  // Planning branch
  { from: "search_policy", to: "gather_intel" },
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

function buildLayoutedElements(
  nodeStates: Map<string, { state: "idle" | "running" | "done"; durationMs?: number }>,
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
    const targetState = nodeStates.get(edge.to)?.state
    const active = targetState === "running" || targetState === "done"
    return {
      id: `${edge.from}-${edge.to}`,
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
  const { label, state, durationMs } = data as {
    label: string
    state: "idle" | "running" | "done"
    durationMs?: number
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
            "border-[#3D5A40] bg-[#3D5A40]/10 text-[#3D5A40]"
        )}
        style={{ width: NODE_WIDTH, height: NODE_HEIGHT }}
      >
        <span className="text-[9px] leading-tight truncate w-full">{label}</span>
        {state === "done" && durationMs != null && (
          <span className="text-[7px] opacity-60 leading-none">{durationMs}ms</span>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} className="!w-1 !h-1 !min-w-0 !min-h-0 !bg-transparent !border-0" />
    </>
  )
}

const rfNodeTypes = { dagNode: DagNodeComponent }

function GraphDAGView({ nodeEvents }: { nodeEvents: NodeEvent[] }) {
  const nodeStates = useMemo(() => {
    const states = new Map<string, { state: "idle" | "running" | "done"; durationMs?: number }>()
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
      else states.set(id, { state: "done", durationMs: found.durationMs })
    }
    return states
  }, [nodeEvents])

  const { nodes, edges } = useMemo(
    () => buildLayoutedElements(nodeStates),
    [nodeStates],
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

  return (
    <div
      className={cn(
        "px-4 py-2 rounded-lg border-2 text-xs font-medium w-40 text-center",
        "transition-all duration-300",
        isRunning
          ? "bg-[#FFCC99] border-[#E8A87C] text-[#5C3D2E] font-semibold animate-pulse"
          : "bg-[#3D5A40]/10 border-[#3D5A40] text-[#3D5A40]"
      )}
    >
      <div className="flex items-center justify-center gap-1.5">
        {isRunning ? (
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#E8A87C] opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-[#E8A87C]" />
          </span>
        ) : (
          <svg className="h-3 w-3 text-[#3D5A40]" viewBox="0 0 12 12" fill="none">
            <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
        {label}
      </div>
      <div className="text-[10px] opacity-60 mt-0.5">
        {isRunning
          ? event.ts
          : `${event.ts} → ${event.endTs ?? ""}${event.durationMs != null ? ` (${event.durationMs}ms)` : ""}`}
      </div>
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
