"use client";

import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { requirePublicApiBaseUrl } from "@/lib/public-config";

interface DataPoint {
  timestamp: string;
  skill_level: number;
  accuracy: number;
  topic: string;
  event_count: number;
}

interface GrowthSeries {
  topic: string;
  subject: string;
  data_points: DataPoint[];
  trend: string;
  current_level: number;
}

interface GrowthData {
  user_id: string;
  subject: string;
  days: number;
  series: GrowthSeries[];
  overall_accuracy: number;
  overall_trend: string;
  total_events: number;
}

const API = requirePublicApiBaseUrl();

function trendBadge(trend: string) {
  const map: Record<string, { label: string; variant: "default" | "secondary" | "destructive" }> = {
    improving: { label: "📈 上升", variant: "default" },
    declining: { label: "📉 下降", variant: "destructive" },
    stable: { label: "➡️ 稳定", variant: "secondary" },
  };
  const info = map[trend] || map.stable;
  return <Badge variant={info.variant}>{info.label}</Badge>;
}

function formatDate(iso: string) {
  return iso.slice(0, 10);
}

export default function GrowthChart({ userId, subject = "", days = 30 }: {
  userId: string; subject?: string; days?: number;
}) {
  const [data, setData] = useState<GrowthData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const params = new URLSearchParams({ subject, days: String(days) });
    fetch(`${API}/analytics/growth/${encodeURIComponent(userId)}?${params}`)
      .then(r => r.json())
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [userId, subject, days]);

  if (loading) return <div className="p-4 text-muted-foreground">Loading growth data...</div>;
  if (!data || !data.series.length) {
    return <div className="p-4 text-muted-foreground">No growth data yet. Complete some exercises to see progress.</div>;
  }

  // Flatten all data points across series, keyed by timestamp
  const chartData = new Map<string, Record<string, number | string>>();
  for (const series of data.series) {
    for (const pt of series.data_points) {
      const day = formatDate(pt.timestamp);
      if (!chartData.has(day)) chartData.set(day, { date: day });
      const row = chartData.get(day)!;
      row[`${series.topic}_level`] = pt.skill_level;
      row[`${series.topic}_acc`] = pt.accuracy;
    }
  }
  const merged = Array.from(chartData.values()).sort(
    (a, b) => String(a.date).localeCompare(String(b.date))
  );

  // Pick the first 3 series for display
  const displaySeries = data.series.slice(0, 3);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 justify-between">
          <span>技能成长曲线</span>
          <div className="flex gap-2 text-sm">
            {trendBadge(data.overall_trend)}
            <Badge variant="outline">准确率 {(data.overall_accuracy * 100).toFixed(0)}%</Badge>
            <Badge variant="outline">{data.total_events} 事件</Badge>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={merged}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.4} />
            <XAxis dataKey="date" fontSize={11} />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(v: number | string) => `${(Number(v) * 100).toFixed(0)}%`}
              fontSize={11}
            />
            <Tooltip formatter={(v) => `${(Number(v ?? 0) * 100).toFixed(0)}%`} />
            <Legend />
            {displaySeries.map((s, i) => (
              <Line
                key={s.topic}
                type="monotone"
                dataKey={`${s.topic}_level`}
                name={`${s.topic} (水平)`}
                stroke={["#4f46e5", "#f59e0b", "#10b981"][i % 3]}
                strokeWidth={2}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
        <div className="mt-4 grid grid-cols-3 gap-2 text-xs text-muted-foreground">
          {data.series.map(s => (
            <div key={s.topic} className="flex items-center gap-1">
              <span className="font-medium">{s.topic}</span>
              <span className="text-xs">{(s.current_level * 100).toFixed(0)}%</span>
              {trendBadge(s.trend)}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
