import { describe, expect, it } from "vitest"

import { layoutGraphManifest } from "@/lib/graph-layout"
import { parseGraphManifest } from "@/lib/observability-contracts"
import { graphManifestPayload } from "@/test/observability-fixtures"

describe("graph manifest layout", () => {
  it("uses backend topology and labels deterministically", () => {
    const manifest = parseGraphManifest(graphManifestPayload())
    const first = layoutGraphManifest(manifest)
    const second = layoutGraphManifest(manifest)
    expect(second).toEqual(first)
    expect(first.nodes.map((node) => [node.id, node.label])).toEqual([
      ["start_node", "后端起点"],
      ["output_node", "后端输出"],
    ])
    expect(first.edges).toEqual([
      expect.objectContaining({ source: "start_node", target: "output_node" }),
    ])
  })

  it("omits hidden nodes and their edges", () => {
    const payload = graphManifestPayload()
    const rawNodes = payload.nodes as Array<Record<string, unknown>>
    const manifest = parseGraphManifest(
      graphManifestPayload({
        nodes: rawNodes.map((node) =>
          node.node_id === "output_node" ? { ...node, visible: false } : node,
        ),
      }),
    )
    const layout = layoutGraphManifest(manifest)
    expect(layout.nodes.map((node) => node.id)).toEqual(["start_node"])
    expect(layout.edges).toEqual([])
  })

  it("renders a deterministic current path without silently substituting the full graph", () => {
    const manifest = parseGraphManifest(graphManifestPayload())
    const path = layoutGraphManifest(manifest, {
      viewMode: "current_path",
      activeNodeIds: ["start_node", "output_node"],
    })
    const empty = layoutGraphManifest(manifest, {
      viewMode: "current_path",
      activeNodeIds: [],
    })

    expect(path.viewMode).toBe("current_path")
    expect(path.nodes.map((node) => node.id)).toEqual(["start_node", "output_node"])
    expect(empty.nodes).toEqual([])
    expect(empty.edges).toEqual([])
  })
})
