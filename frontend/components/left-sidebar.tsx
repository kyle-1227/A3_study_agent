"use client"

import { useState, useEffect } from "react"
import { BrainCircuit, ChevronLeft, ChevronRight, MessageSquarePlus, MessageSquare, Settings, GraduationCap } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { useRouter } from "next/navigation"

interface ChatHistoryItem {
  id: string
  title: string
}

export interface VolunteerHistoryItem {
  id: string
  title: string
  targetRegion: string
  homeRegion: string
}

interface LeftSidebarProps {
  chatHistory: ChatHistoryItem[]
  onNewChat: () => void
  onSelectChat: (id: string) => void
  onClearChatHistory?: () => void
  selectedChatId?: string
}

const VOLUNTEER_STORAGE_KEY = "volunteer_chat_history"

export function getVolunteerHistory(): VolunteerHistoryItem[] {
  if (typeof window === "undefined") return []
  try {
    const raw = localStorage.getItem(VOLUNTEER_STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export function saveVolunteerHistory(items: VolunteerHistoryItem[]) {
  if (typeof window === "undefined") return
  localStorage.setItem(VOLUNTEER_STORAGE_KEY, JSON.stringify(items))
}

export function LeftSidebar({ chatHistory, onNewChat, onSelectChat, onClearChatHistory, selectedChatId }: LeftSidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [volunteerHistory, setVolunteerHistory] = useState<VolunteerHistoryItem[]>([])
  const router = useRouter()

  useEffect(() => {
    setVolunteerHistory(getVolunteerHistory())
    const onStorage = () => setVolunteerHistory(getVolunteerHistory())
    window.addEventListener("storage", onStorage)
    return () => window.removeEventListener("storage", onStorage)
  }, [])

  // Re-read from localStorage when sidebar gains focus / becomes visible
  useEffect(() => {
    const onFocus = () => setVolunteerHistory(getVolunteerHistory())
    window.addEventListener("focus", onFocus)
    return () => window.removeEventListener("focus", onFocus)
  }, [])

  const handleNewVolunteer = () => {
    router.push("/volunteer")
  }

  const handleSelectVolunteer = (id: string) => {
    router.push(`/volunteer?chatId=${encodeURIComponent(id)}`)
  }

  return (
    <div
      className={cn(
        "relative h-full border-r border-border bg-sidebar flex flex-col",
        "transition-all duration-300 ease-in-out",
        isCollapsed ? "w-12" : "w-72"
      )}
    >
      {isCollapsed ? (
        <>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(false)}
            className="absolute top-4 right-1 h-8 w-8 text-muted-foreground hover:text-foreground"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
          <div className="mt-12 flex flex-col items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={onNewChat}
              className="h-10 w-10 text-primary hover:bg-sidebar-accent"
            >
              <MessageSquarePlus className="h-5 w-5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleNewVolunteer}
              className="h-10 w-10 text-[#3D5A40] hover:bg-sidebar-accent"
              title="志愿填报"
            >
              <GraduationCap className="h-5 w-5" />
            </Button>
          </div>
        </>
      ) : (
        <>
          {/* Collapse Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(true)}
            className="absolute top-4 right-2 h-8 w-8 text-muted-foreground hover:text-foreground z-10"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>

          {/* Header */}
          <div className="p-4 pr-12">
            <div className="flex items-start gap-3">
              {/* Brand Icon */}
              <div className="relative flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-[#3D5A40] to-[#5A7A5E]">
                <BrainCircuit className="h-6 w-6 text-[#FFCC99]" strokeWidth={1.9} />
              </div>
              <div className="flex-1 min-w-0">
                <h1 className="text-base font-semibold text-[#3D5A40] leading-tight">高校学习 AI 助手</h1>
                <div className="flex flex-wrap gap-1 mt-1.5">
                  <Badge variant="secondary" className="text-xs px-1.5 py-0 bg-[#3D5A40]/10 text-[#3D5A40] border-0">
                    学科答疑
                  </Badge>
                  <Badge variant="secondary" className="text-xs px-1.5 py-0 bg-[#FFCC99]/40 text-[#8B5A3C] border-0">
                    情绪支持
                  </Badge>
                  <Badge variant="secondary" className="text-xs px-1.5 py-0 bg-[#7A9E7E]/20 text-[#3D5A40] border-0">
                    计划制定
                  </Badge>
                </div>
              </div>
            </div>
          </div>

          {/* 志愿填报历史纪录 */}
          <div className="px-4 pb-3">
            <div className="flex items-center justify-between pb-2">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">志愿填报</span>
              <button
                onClick={handleNewVolunteer}
                className="text-xs text-[#3D5A40] hover:text-[#4A6B4D] font-medium"
              >
                + 新建
              </button>
            </div>
            {volunteerHistory.length > 0 ? (
              <div className="flex flex-col gap-0.5 max-h-36 overflow-y-auto">
                {volunteerHistory.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => handleSelectVolunteer(item.id)}
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm text-left rounded-lg transition-colors text-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                  >
                    <GraduationCap className="h-4 w-4 flex-shrink-0 text-[#3D5A40]" />
                    <div className="min-w-0">
                      <span className="truncate block">{item.title}</span>
                      <span className="text-[10px] text-muted-foreground">
                        {item.homeRegion} → {item.targetRegion}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground px-3 py-1">暂无志愿填报记录</p>
            )}
          </div>

          {/* New Chat Button */}
          <div className="px-4 pb-4">
            <Button
              onClick={onNewChat}
              className="w-full justify-start gap-2 bg-[#3D5A40] hover:bg-[#4A6B4D] text-white"
            >
              <MessageSquarePlus className="h-4 w-4" />
              发起新对话
            </Button>
          </div>

          {/* Chat History */}
          <div className="flex-1 overflow-hidden flex flex-col">
            <div className="px-4 pb-2 flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">对话</span>
              {chatHistory.length > 0 && onClearChatHistory ? (
                <button
                  onClick={onClearChatHistory}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  清空
                </button>
              ) : null}
            </div>
            <ScrollArea className="flex-1 px-2">
              <div className="flex flex-col gap-1 pb-4">
                {chatHistory.map((chat) => (
                  <button
                    key={chat.id}
                    onClick={() => onSelectChat(chat.id)}
                    className={cn(
                      "w-full flex items-center gap-2 px-3 py-2 text-sm text-left rounded-lg transition-colors",
                      "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                      selectedChatId === chat.id
                        ? "bg-sidebar-accent text-sidebar-accent-foreground"
                        : "text-foreground"
                    )}
                  >
                    <MessageSquare className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
                    <span className="truncate">{chat.title}</span>
                  </button>
                ))}
              </div>
            </ScrollArea>
          </div>

          {/* Settings & Help */}
          <div className="p-4 border-t border-border">
            <Button
              variant="ghost"
              className="w-full justify-start gap-2 text-muted-foreground hover:text-foreground"
            >
              <Settings className="h-4 w-4" />
              设置与帮助
            </Button>
          </div>
        </>
      )}
    </div>
  )
}
