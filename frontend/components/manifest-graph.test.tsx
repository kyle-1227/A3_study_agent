// @vitest-environment jsdom

import { render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const flowMocks = vi.hoisted(() => ({ fitView: vi.fn() }))

vi.mock("@xyflow/react", async () => {
  const React = await import("react")
  return {
    Background: () => null,
    Controls: () => null,
    Handle: () => null,
    MiniMap: () => null,
    Position: { Top: "top", Bottom: "bottom" },
    ReactFlow: ({ nodes, onInit }: { nodes: Array<{ id: string; data: { label: string } }>; onInit?: (instance: { fitView: typeof flowMocks.fitView }) => void }) => {
      React.useEffect(() => onInit?.({ fitView: flowMocks.fitView }), [onInit])
      return (
        <div className="react-flow">
          {nodes.map((node) => (
            <div key={node.id} className="react-flow__node">
              {node.data.label}
            </div>
          ))}
        </div>
      )
    },
  }
})

import { ManifestGraph } from "@/components/manifest-graph"
import { parseGraphManifest } from "@/lib/observability-contracts"
import { graphManifestPayload } from "@/test/observability-fixtures"

afterEach(() => {
  vi.unstubAllGlobals()
})

describe("ManifestGraph", () => {
  it("renders a React Flow node in a nonzero surface for a nonempty manifest", async () => {
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      callback(0)
      return 1
    })
    vi.stubGlobal("cancelAnimationFrame", () => undefined)
    vi.stubGlobal(
      "ResizeObserver",
      class {
        constructor(private readonly callback: ResizeObserverCallback) {}

        observe() {
          this.callback(
            [{ contentRect: { width: 320, height: 360 } } as ResizeObserverEntry],
            this as unknown as ResizeObserver,
          )
        }

        unobserve() {}

        disconnect() {}
      },
    )
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
      width: 320,
      height: 360,
      top: 0,
      right: 320,
      bottom: 360,
      left: 0,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    })

    const view = render(
      <ManifestGraph
        manifest={parseGraphManifest(graphManifestPayload())}
        error={null}
        loading={false}
        activities={[]}
        viewMode="full_graph"
      />,
    )

    const surface = screen.getByTestId("manifest-graph-surface")
    await waitFor(() => {
      expect(Number(surface.dataset.surfaceWidth)).toBeGreaterThan(0)
      expect(Number(surface.dataset.surfaceHeight)).toBeGreaterThan(0)
    })
    expect(Number(surface.dataset.visibleNodeCount)).toBeGreaterThan(0)
    expect(surface.querySelector(".react-flow__node")).not.toBeNull()
    expect(flowMocks.fitView).toHaveBeenCalled()

    flowMocks.fitView.mockClear()
    view.rerender(
      <ManifestGraph
        manifest={parseGraphManifest(graphManifestPayload())}
        error={null}
        loading={false}
        activities={[]}
        viewMode="full_graph"
        fitViewSignal="right-panel-resized"
      />,
    )
    await waitFor(() => expect(flowMocks.fitView).toHaveBeenCalled())
  })
})
