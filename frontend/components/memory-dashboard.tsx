"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { requirePublicApiBaseUrl } from "@/lib/public-config";
import GrowthChart from "./growth-chart";
import CognitiveGraph from "./cognitive-graph";
import ExplainPanel from "./explain-panel";

interface DashboardData {
  user_id: string;
  stats_summary: Record<string, unknown>;
}

const API = requirePublicApiBaseUrl();

export default function MemoryDashboard({ userId, subject = "", days = 30 }: {
  userId: string; subject?: string; days?: number;
}) {
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    const params = new URLSearchParams({ subject, days: String(days) });
    fetch(`${API}/analytics/dashboard/${encodeURIComponent(userId)}?${params}`)
      .then(r => r.json())
      .then(d => setStats(d.stats_summary || {}))
      .catch(console.error);
  }, [userId, subject, days]);

  return (
    <div className="space-y-6 p-4">
      {/* Stats Bar */}
      {stats && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(stats).map(([k, v]) => (
            <Badge key={k} variant="outline" className="text-xs">
              {k}: {String(v)}
            </Badge>
          ))}
        </div>
      )}

      {/* Growth Chart */}
      <GrowthChart userId={userId} subject={subject} days={days} />

      {/* Cognitive Graph + Explain Panel side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CognitiveGraph userId={userId} subject={subject} />
        <ExplainPanel userId={userId} limit={15} />
      </div>
    </div>
  );
}
