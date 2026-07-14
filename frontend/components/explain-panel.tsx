"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ChevronDown, ChevronRight, Brain, Search, FileText } from "lucide-react";

interface DecisionTrace {
  trace_id: string;
  node_name: string;
  timestamp: string;
  decision: string;
  evidence: string;
  reasoning_steps: string[];
  confidence: number;
  subject: string;
}

interface TraceList {
  user_id: string;
  traces: DecisionTrace[];
  total: number;
}

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const NODE_ICONS: Record<string, React.ReactNode> = {
  supervisor: <Brain className="w-4 h-4" />,
  memory_use_decider: <Search className="w-4 h-4" />,
  evidence_judge: <FileText className="w-4 h-4" />,
  generate_answer: <FileText className="w-4 h-4" />,
};

const NODE_LABELS: Record<string, string> = {
  supervisor: "意图分类",
  memory_use_decider: "记忆使用决策",
  evidence_judge: "证据判断",
  generate_answer: "答案生成",
};

export default function ExplainPanel({ userId, limit = 20 }: { userId: string; limit?: number }) {
  const [data, setData] = useState<TraceList | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  useEffect(() => {
    fetch(`${API}/analytics/decisions/${encodeURIComponent(userId)}?limit=${limit}`)
      .then(r => r.json())
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [userId, limit]);

  function toggle(id: string) {
    setExpanded(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  if (loading) return <div className="p-4 text-muted-foreground">Loading decision traces...</div>;
  if (!data || !data.traces.length) {
    return <div className="p-4 text-muted-foreground">No decision traces yet. Interact with the agent to see explainability data.</div>;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span>AI 决策可解释面板</span>
          <Badge variant="secondary">{data.total} traces</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 max-h-[500px] overflow-y-auto">
        {data.traces.map(trace => {
          const isExpanded = expanded.has(trace.trace_id);
          return (
            <div key={trace.trace_id} className="border rounded-lg p-3 text-sm">
              <button
                className="flex items-center gap-2 w-full text-left font-medium hover:text-primary"
                onClick={() => toggle(trace.trace_id)}
              >
                {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                <span className="flex items-center gap-1">
                  {NODE_ICONS[trace.node_name] || null}
                  {NODE_LABELS[trace.node_name] || trace.node_name}
                </span>
                <span className="text-muted-foreground font-normal truncate flex-1">
                  {trace.decision.slice(0, 80)}
                </span>
                <Badge variant={trace.confidence > 0.7 ? "default" : "secondary"} className="text-xs">
                  {(trace.confidence * 100).toFixed(0)}%
                </Badge>
              </button>
              {isExpanded && (
                <div className="mt-2 ml-6 space-y-1 text-muted-foreground">
                  <p><span className="font-medium text-foreground">决策:</span> {trace.decision}</p>
                  {trace.evidence && <p><span className="font-medium text-foreground">证据:</span> {trace.evidence}</p>}
                  {trace.reasoning_steps.length > 0 && (
                    <div>
                      <span className="font-medium text-foreground">推理步骤:</span>
                      <ol className="list-decimal list-inside mt-1 space-y-0.5">
                        {trace.reasoning_steps.map((step, i) => (
                          <li key={i}>{step}</li>
                        ))}
                      </ol>
                    </div>
                  )}
                  <p className="text-xs">
                    {trace.timestamp?.slice(0, 19).replace("T", " ")}
                    {trace.subject && ` · ${trace.subject}`}
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
