"use client"

import { useMemo } from "react"
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
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"

import { latestActivityByNode } from "@/lib/activity-reducer"
import { layoutGraphManifest } from "@/lib/graph-layout"
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

interface ManifestGraphProps {
  manifest: GraphManifest | null
  error: GraphManifestUnavailable | null
  loading: boolean
  activities: readonly ActivityEvent[]
}

export function ManifestGraph({ manifest, error, loading, activities }: ManifestGraphProps) {
  const layout = useMemo(() => (manifest ? layoutGraphManifest(manifest) : null), [manifest])
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

  if (loading) return <GraphMessage text="正在读取图清单..." />
  if (error) return <GraphMessage text={`图清单不可用：${error.reason}`} tone="error" />
  if (!layout) return <GraphMessage text="尚未收到图清单引用。" />

  return (
    <div className="h-full min-h-[260px] w-full bg-[var(--surface-muted)]">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={{ manifestNode: ManifestNodeComponent }}
        nodesConnectable={false}
        elementsSelectable
        fitView
        fitViewOptions={{ padding: 0.18 }}
        minZoom={0.2}
        maxZoom={1.6}
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

function GraphMessage({ text, tone = "muted" }: { text: string; tone?: "muted" | "error" }) {
  return (
    <div className={cn("flex h-full min-h-[260px] items-center justify-center px-6 text-center text-xs", tone === "error" ? "text-destructive" : "text-muted-foreground")}>
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
