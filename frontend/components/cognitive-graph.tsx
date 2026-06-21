"use client";

import { useEffect, useState, useCallback } from "react";
import {
  ReactFlow, Node, Edge, Background, Controls,
  useNodesState, useEdgesState, MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface CogNode {
  id: string; label: string; type: string;
  size: number; level: number | null; confidence: number | null;
  details: string; color: string;
}
interface CogEdge {
  source: string; target: string; weight: number; label: string;
}
interface CogGraph {
  user_id: string; nodes: CogNode[]; edges: CogEdge[];
  summary: string; node_count: number; edge_count: number;
}

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function cogToFlowNode(n: CogNode): Node {
  const size = Math.max(40, n.size * 120);
  return {
    id: n.id,
    data: { label: n.label },
    position: { x: 0, y: 0 },
    style: {
      background: n.color,
      color: "#fff",
      border: "2px solid rgba(255,255,255,0.3)",
      borderRadius: "50%",
      width: size,
      height: size,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: Math.max(9, size * 0.22),
      fontWeight: 600,
      textAlign: "center" as const,
      padding: 4,
      wordBreak: "break-all" as const,
    },
  };
}

function cogToFlowEdge(e: CogEdge): Edge {
  return {
    id: `${e.source}-${e.target}-${e.label}`,
    source: e.source,
    target: e.target,
    label: e.label,
    animated: e.label === "requires",
    style: { stroke: e.label === "requires" ? "#ef4444" : "#94a3b8", strokeWidth: e.weight * 3 + 1 },
    markerEnd: { type: MarkerType.ArrowClosed, color: e.label === "requires" ? "#ef4444" : "#94a3b8" },
  };
}

export default function CognitiveGraph({ userId, subject = "" }: { userId: string; subject?: string }) {
  const [data, setData] = useState<CogGraph | null>(null);
  const [loading, setLoading] = useState(true);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    const params = new URLSearchParams({ subject });
    fetch(`${API}/analytics/cognitive-graph/${encodeURIComponent(userId)}?${params}`)
      .then(r => r.json())
      .then((d: CogGraph) => {
        setData(d);
        setNodes(d.nodes.map(cogToFlowNode));
        setEdges(d.edges.map(cogToFlowEdge));
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [userId, subject, setNodes, setEdges]);

  const typeCounts = data?.nodes.reduce((acc, n) => {
    acc[n.type] = (acc[n.type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>) || {};

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 justify-between">
          <span>认知模型图谱</span>
          <div className="flex gap-1">
            {Object.entries(typeCounts).map(([t, c]) => (
              <Badge key={t} variant="outline" className="text-xs">{t}:{c}</Badge>
            ))}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0" style={{ height: 420 }}>
        {loading ? (
          <div className="flex items-center justify-center h-full text-muted-foreground">Loading...</div>
        ) : !data || !data.nodes.length ? (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            No cognitive data yet. Complete onboarding and some exercises.
          </div>
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            fitView
            attributionPosition="bottom-left"
          >
            <Background />
            <Controls />
          </ReactFlow>
        )}
      </CardContent>
    </Card>
  );
}
