import dagre from "@dagrejs/dagre"

import type { GraphManifest } from "@/lib/observability-contracts"

export const GRAPH_NODE_WIDTH = 148
export const GRAPH_NODE_HEIGHT = 48

export type GraphViewMode = "current_path" | "full_graph"

export interface GraphLayoutOptions {
  viewMode?: GraphViewMode
  activeNodeIds?: readonly string[]
}

export interface LayoutedGraphNode {
  id: string
  x: number
  y: number
  width: number
  height: number
  label: string
  description: string
  kind: string
  group: string
  logical: boolean
}

export interface LayoutedGraphEdge {
  id: string
  source: string
  target: string
  conditional: boolean
  kind: "graph" | "logical"
  label: string
}

export interface LayoutedGraph {
  graphVersion: string
  viewMode: GraphViewMode
  nodes: LayoutedGraphNode[]
  edges: LayoutedGraphEdge[]
}

export function layoutGraphManifest(
  manifest: GraphManifest,
  options: GraphLayoutOptions = {},
): LayoutedGraph {
  const graph = new dagre.graphlib.Graph()
  graph.setDefaultEdgeLabel(() => ({}))
  graph.setGraph({ rankdir: "TB", nodesep: 28, ranksep: 46, marginx: 18, marginy: 18 })

  const visibleNodes = manifest.nodes.filter((node) => node.visible)
  const visibleIds = new Set(visibleNodes.map((node) => node.nodeId))
  const allVisibleEdges = manifest.edges.filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
  )
  const viewMode = options.viewMode ?? "full_graph"
  const nodeIdsForView =
    viewMode === "current_path"
      ? resolveCurrentPathNodeIds(allVisibleEdges, visibleIds, options.activeNodeIds ?? [])
      : visibleIds
  const nodesForView = visibleNodes.filter((node) => nodeIdsForView.has(node.nodeId))
  const visibleEdges = allVisibleEdges.filter(
    (edge) => nodeIdsForView.has(edge.source) && nodeIdsForView.has(edge.target),
  )
  for (const node of nodesForView) {
    graph.setNode(node.nodeId, { width: GRAPH_NODE_WIDTH, height: GRAPH_NODE_HEIGHT })
  }
  for (const edge of visibleEdges) graph.setEdge(edge.source, edge.target)
  dagre.layout(graph)

  return {
    graphVersion: manifest.graphVersion,
    viewMode,
    nodes: nodesForView.map((node) => {
      const position = graph.node(node.nodeId) as { x: number; y: number }
      return {
        id: node.nodeId,
        x: position.x - GRAPH_NODE_WIDTH / 2,
        y: position.y - GRAPH_NODE_HEIGHT / 2,
        width: GRAPH_NODE_WIDTH,
        height: GRAPH_NODE_HEIGHT,
        label: node.label,
        description: node.description,
        kind: node.kind,
        group: node.group,
        logical: node.logical,
      }
    }),
    edges: visibleEdges.map((edge) => ({
      id: edge.edgeId,
      source: edge.source,
      target: edge.target,
      conditional: edge.conditional,
      kind: edge.kind,
      label: edge.label,
    })),
  }
}

export function resolveCurrentPathNodeIds(
  edges: readonly Pick<GraphManifest["edges"][number], "source" | "target" | "edgeId">[],
  visibleNodeIds: ReadonlySet<string>,
  activeNodeIds: readonly string[],
): Set<string> {
  const anchors: string[] = []
  const seen = new Set<string>()
  for (const nodeId of activeNodeIds) {
    if (!visibleNodeIds.has(nodeId) || seen.has(nodeId)) continue
    seen.add(nodeId)
    anchors.push(nodeId)
  }
  if (anchors.length === 0) return new Set()

  const selected = new Set(anchors)
  const adjacency = new Map<string, string[]>()
  for (const edge of [...edges].sort((left, right) => left.edgeId.localeCompare(right.edgeId))) {
    const targets = adjacency.get(edge.source) ?? []
    targets.push(edge.target)
    adjacency.set(edge.source, targets)
  }
  for (let index = 1; index < anchors.length; index += 1) {
    const path = shortestDirectedPath(adjacency, anchors[index - 1], anchors[index])
    for (const nodeId of path) selected.add(nodeId)
  }
  return selected
}

function shortestDirectedPath(
  adjacency: ReadonlyMap<string, readonly string[]>,
  source: string,
  target: string,
): string[] {
  if (source === target) return [source]
  const queue = [source]
  const previous = new Map<string, string>()
  const seen = new Set(queue)

  for (let index = 0; index < queue.length; index += 1) {
    const current = queue[index]
    for (const next of adjacency.get(current) ?? []) {
      if (seen.has(next)) continue
      previous.set(next, current)
      if (next === target) {
        const path = [target]
        let cursor = target
        while (previous.has(cursor)) {
          cursor = previous.get(cursor) as string
          path.push(cursor)
        }
        return path.reverse()
      }
      seen.add(next)
      queue.push(next)
    }
  }
  return []
}
