"use client";

import { useState } from "react";
import MemoryDashboard from "@/components/memory-dashboard";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function AnalyticsPage() {
  const [userId, setUserId] = useState("");
  const [subject, setSubject] = useState("");
  const [days, setDays] = useState(30);
  const [submitted, setSubmitted] = useState(false);
  const [submittedId, setSubmittedId] = useState("");
  const [submittedSubject, setSubmittedSubject] = useState("");
  const [submittedDays, setSubmittedDays] = useState(30);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmittedId(userId.trim() || "default");
    setSubmittedSubject(subject.trim());
    setSubmittedDays(days);
    setSubmitted(true);
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">学习分析仪表板</h1>
          <a href="/" className="text-sm text-muted-foreground hover:text-primary">← 返回聊天</a>
        </div>

        {/* Query Form */}
        <Card>
          <CardHeader>
            <CardTitle>查询参数</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="flex flex-wrap gap-3 items-end">
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">用户ID</label>
                <Input
                  placeholder="thread_id (默认: default)"
                  value={userId}
                  onChange={e => setUserId(e.target.value)}
                  className="w-48"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">学科</label>
                <Input
                  placeholder="python / math / ml"
                  value={subject}
                  onChange={e => setSubject(e.target.value)}
                  className="w-36"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">天数</label>
                <Input
                  type="number"
                  min={7}
                  max={365}
                  value={days}
                  onChange={e => setDays(Number(e.target.value))}
                  className="w-24"
                />
              </div>
              <Button type="submit">查询分析</Button>
            </form>
          </CardContent>
        </Card>

        {/* Dashboard */}
        {submitted ? (
          <MemoryDashboard
            userId={submittedId}
            subject={submittedSubject}
            days={submittedDays}
          />
        ) : (
          <div className="text-center py-20 text-muted-foreground">
            <p className="text-lg">输入用户ID和学科，查看学习分析数据</p>
            <p className="text-sm mt-2">
              包含: 技能成长曲线 · 认知模型图谱 · AI决策可解释面板
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
