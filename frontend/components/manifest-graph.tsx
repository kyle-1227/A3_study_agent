"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  type ReactFlowInstance,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"

import { latestActivityByNode } from "@/lib/activity-reducer"
import { layoutGraphManifest, type GraphViewMode } from "@/lib/graph-layout"
import type {
  ActivityEvent,
  ActivityStatus,
  GraphManifest,
  GraphManifestUnavailable,
} from "@/lib/observability-contracts"
import { cn } from "@/lib/utils"

type GraphNodeData = {
  label: string
  description: string
  kind: string
  logical: boolean
  status: ActivityStatus | "idle"
}

type ManifestNode = Node<GraphNodeData, "manifestNode">

type SurfaceSize = {
  width: number
  height: number
}

interface ManifestGraphProps {
  manifest: GraphManifest | null
  error: GraphManifestUnavailable | null
  loading: boolean
  activities: readonly ActivityEvent[]
  viewMode?: GraphViewMode
  fitViewSignal?: string
}

export function ManifestGraph({
  manifest,
  error,
  loading,
  activities,
  viewMode = "full_graph",
  fitViewSignal = "",
}: ManifestGraphProps) {
  const activeNodeIds = useMemo(
    () => activities.map((activity) => activity.node).filter(Boolean),
    [activities],
  )
  const layout = useMemo(
    () =>
      manifest
        ? layoutGraphManifest(manifest, { viewMode, activeNodeIds })
        : null,
    [activeNodeIds, manifest, viewMode],
  )
  const nodeActivity = useMemo(() => latestActivityByNode(activities), [activities])
  const nodes = useMemo<ManifestNode[]>(
    () =>
      layout?.nodes.map((node) => ({
        id: node.id,
        type: "manifestNode",
        position: { x: node.x, y: node.y },
        data: {
          label: node.label,
          description: node.description,
          kind: node.kind,
          logical: node.logical,
          status: nodeActivity.get(node.id)?.status ?? "idle",
        },
        draggable: false,
        selectable: true,
      })) ?? [],
    [layout, nodeActivity],
  )
  const edges = useMemo<Edge[]>(
    () =>
      layout?.edges.map((edge) => {
        const targetStatus = nodeActivity.get(edge.target)?.status
        const traversed = Boolean(targetStatus && targetStatus !== "queued")
        return {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.label || undefined,
          type: "smoothstep",
          style: {
            stroke: traversed ? "#55735D" : "#CFCDBF",
            strokeWidth: traversed ? 1.7 : 1,
            strokeDasharray: edge.conditional || edge.kind === "logical" ? "5 4" : undefined,
          },
        }
      }) ?? [],
    [layout, nodeActivity],
  )
  const surfaceRef = useRef<HTMLDivElement>(null)
  const flowRef = useRef<ReactFlowInstance<ManifestNode, Edge> | null>(null)
  const fitFrameRef = useRef<number | null>(null)
  const [surfaceSize, setSurfaceSize] = useState<SurfaceSize>({ width: 0, height: 0 })
  const layoutKey = `${layout?.graphVersion ?? "unavailable"}:${viewMode}`

  const updateSurfaceSize = useCallback((): SurfaceSize => {
    const element = surfaceRef.current
    if (!element) return { width: 0, height: 0 }
    const rect = element.getBoundingClientRect()
    const next = {
      width: Math.max(0, Math.round(rect.width || element.clientWidth)),
      height: Math.max(0, Math.round(rect.height || element.clientHeight)),
    }
    setSurfaceSize((current) =>
      current.width === next.width && current.height === next.height ? current : next,
    )
    return next
  }, [])

  const scheduleFitView = useCallback(() => {
    const instance = flowRef.current
    const element = surfaceRef.current
    if (!instance || !element || nodes.length === 0) return
    const rect = element.getBoundingClientRect()
    const width = rect.width || element.clientWidth
    const height = rect.height || element.clientHeight
    if (width <= 0 || height <= 0) return
    if (fitFrameRef.current !== null) window.cancelAnimationFrame(fitFrameRef.current)
    fitFrameRef.current = window.requestAnimationFrame(() => {
      fitFrameRef.current = window.requestAnimationFrame(() => {
        fitFrameRef.current = null
        flowRef.current?.fitView({ padding: 0.18, duration: 0, maxZoom: 1.2 })
      })
    })
  }, [layoutKey, nodes.length])

  const handleInit = useCallback(
    (instance: ReactFlowInstance<ManifestNode, Edge>) => {
      flowRef.current = instance
      updateSurfaceSize()
      scheduleFitView()
    },
    [scheduleFitView, updateSurfaceSize],
  )

  useEffect(() => {
    updateSurfaceSize()
    scheduleFitView()
  }, [
    edges.length,
    fitViewSignal,
    layoutKey,
    nodes.length,
    scheduleFitView,
    updateSurfaceSize,
  ])

  useEffect(() => {
    const element = surfaceRef.current
    if (!element) return undefined
    const observer = new ResizeObserver(() => {
      updateSurfaceSize()
      scheduleFitView()
    })
    observer.observe(element)
    updateSurfaceSize()
    return () => observer.disconnect()
  }, [scheduleFitView, updateSurfaceSize])

  useEffect(
    () => () => {
      if (fitFrameRef.current !== null) window.cancelAnimationFrame(fitFrameRef.current)
    },
    [],
  )

  if (loading) return <GraphMessage text="正在读取图清单..." />
  if (error) return <GraphMessage text={`图清单不可用：${error.reason}`} tone="error" />
  if (!layout) return <GraphMessage text="尚未收到图清单引用。" />

  const emptyMessage =
    viewMode === "current_path" ? "当前请求尚无可显示的执行路径。" : "图清单未包含可见节点。"

  return (
    <div
      ref={surfaceRef}
      data-testid="manifest-graph-surface"
      data-graph-version={layout.graphVersion}
      data-visible-node-count={nodes.length}
      data-surface-width={surfaceSize.width}
      data-surface-height={surfaceSize.height}
      className="h-[clamp(20rem,48dvh,38rem)] min-h-[20rem] w-full overflow-hidden rounded-md border border-border bg-[var(--surface-muted)]"
    >
      {nodes.length === 0 ? (
        <GraphMessage text={emptyMessage} />
      ) : (
        <ReactFlow
          key={layoutKey}
          nodes={nodes}
          edges={edges}
          nodeTypes={MANIFEST_NODE_TYPES}
          nodesConnectable={false}
          elementsSelectable
          fitView
          fitViewOptions={{ padding: 0.18 }}
          minZoom={0.2}
          maxZoom={1.6}
          onInit={handleInit}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={20} size={1} color="#DDD9C8" />
          <Controls showInteractive={false} position="bottom-right" />
          <MiniMap
            pannable
            zoomable
            nodeColor={(node) => statusColor((node.data as GraphNodeData).status)}
            maskColor="rgba(247, 246, 237, 0.76)"
          />
        </ReactFlow>
      )}
    </div>
  )
}

function ManifestNodeComponent({ data }: NodeProps<ManifestNode>) {
  return (
    <div
      className={cn(
        "flex h-12 w-[148px] items-center rounded-md border bg-card px-2.5 text-left",
        data.status === "idle" && "border-border text-muted-foreground",
        ["queued", "running", "retrying"].includes(data.status) && "border-[var(--warning)] bg-[var(--warning-soft)] text-foreground",
        ["waiting", "interrupted"].includes(data.status) && "border-[var(--info)] bg-[var(--info-soft)] text-foreground",
        data.status === "completed" && "border-[var(--success)] bg-[var(--success-soft)] text-foreground",
        data.status === "failed" && "border-destructive bg-destructive/5 text-foreground",
        data.status === "skipped" && "border-dashed border-muted-foreground/50 text-muted-foreground",
        data.logical && "border-dashed",
      )}
      title={data.description}
    >
      <Handle type="target" position={Position.Top} className="!h-1.5 !w-1.5 !border-0 !bg-muted-foreground" />
      <div className="min-w-0">
        <p className="truncate text-[11px] font-semibold leading-4">{data.label}</p>
        <p className="truncate text-[9px] leading-3 opacity-70">{data.kind}</p>
      </div>
      <Handle type="source" position={Position.Bottom} className="!h-1.5 !w-1.5 !border-0 !bg-muted-foreground" />
    </div>
  )
}

const MANIFEST_NODE_TYPES = { manifestNode: ManifestNodeComponent }

function GraphMessage({ text, tone = "muted" }: { text: string; tone?: "muted" | "error" }) {
  return (
    <div className={cn("flex h-full min-h-[20rem] items-center justify-center px-6 text-center text-xs", tone === "error" ? "text-destructive" : "text-muted-foreground")}>
      {text}
    </div>
  )
}

function statusColor(status: ActivityStatus | "idle"): string {
  if (status === "completed") return "#3E7A4A"
  if (["queued", "running", "retrying"].includes(status)) return "#B7791F"
  if (["waiting", "interrupted"].includes(status)) return "#3E6F99"
  if (status === "failed") return "#C55447"
  return "#B6B5AA"
}
