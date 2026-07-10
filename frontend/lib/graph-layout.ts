import dagre from "@dagrejs/dagre"

import type { GraphManifest } from "@/lib/observability-contracts"

export const GRAPH_NODE_WIDTH = 148
export const GRAPH_NODE_HEIGHT = 48

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
  nodes: LayoutedGraphNode[]
  edges: LayoutedGraphEdge[]
}

export function layoutGraphManifest(manifest: GraphManifest): LayoutedGraph {
  const graph = new dagre.graphlib.Graph()
  graph.setDefaultEdgeLabel(() => ({}))
  graph.setGraph({ rankdir: "TB", nodesep: 28, ranksep: 46, marginx: 18, marginy: 18 })

  const visibleNodes = manifest.nodes.filter((node) => node.visible)
  const visibleIds = new Set(visibleNodes.map((node) => node.nodeId))
  const visibleEdges = manifest.edges.filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
  )
  for (const node of visibleNodes) {
    graph.setNode(node.nodeId, { width: GRAPH_NODE_WIDTH, height: GRAPH_NODE_HEIGHT })
  }
  for (const edge of visibleEdges) graph.setEdge(edge.source, edge.target)
  dagre.layout(graph)

  return {
    graphVersion: manifest.graphVersion,
    nodes: visibleNodes.map((node) => {
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
