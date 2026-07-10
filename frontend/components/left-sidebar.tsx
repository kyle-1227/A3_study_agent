"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import {
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  GraduationCap,
  LogIn,
  LogOut,
  MessageSquare,
  MessageSquarePlus,
  Settings,
  Trash2,
  User2,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

interface ChatHistoryItem {
  id: string
  title: string
  threadId?: string
  updatedAt?: number
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
  onClearChat?: (id: string) => void
  onClearChatHistory?: () => void
  selectedChatId?: string
  userId?: string | null
  nickname?: string | null
  onStartOnboarding?: () => void
  onClearUser?: () => void
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

export function LeftSidebar({
  chatHistory,
  onNewChat,
  onSelectChat,
  onClearChat,
  onClearChatHistory,
  selectedChatId,
  userId,
  nickname,
  onStartOnboarding,
  onClearUser,
}: LeftSidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [volunteerHistory, setVolunteerHistory] = useState<VolunteerHistoryItem[]>([])
  const router = useRouter()

  useEffect(() => {
    if (window.matchMedia("(max-width: 767px)").matches) setIsCollapsed(true)
  }, [])

  useEffect(() => {
    setVolunteerHistory(getVolunteerHistory())
    const onStorage = () => setVolunteerHistory(getVolunteerHistory())
    window.addEventListener("storage", onStorage)
    return () => window.removeEventListener("storage", onStorage)
  }, [])

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
    <aside
      className={cn(
        "relative flex h-[100dvh] shrink-0 self-stretch flex-col overflow-hidden border-r border-sidebar-border bg-sidebar text-sidebar-foreground",
        "transition-[width] duration-200 ease-out",
        isCollapsed ? "w-12" : "w-72",
      )}
      aria-label="主导航"
    >
      {isCollapsed ? (
        <>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(false)}
            className="absolute right-1 top-4 h-8 w-8 text-muted-foreground hover:text-foreground"
            title="展开侧边栏"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
          <div className="mt-12 flex flex-col items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={onNewChat}
              className="h-10 w-10 text-primary hover:bg-sidebar-accent"
              title="发起新对话"
            >
              <MessageSquarePlus className="h-5 w-5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleNewVolunteer}
              className="h-10 w-10 text-primary hover:bg-sidebar-accent"
              title="志愿填报"
            >
              <GraduationCap className="h-5 w-5" />
            </Button>
          </div>
        </>
      ) : (
        <>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(true)}
            className="absolute right-2 top-4 z-10 h-8 w-8 text-muted-foreground hover:text-foreground"
            title="收起侧边栏"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>

          <div className="p-4 pr-12">
            <div className="flex items-start gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary text-primary-foreground">
                <BrainCircuit className="h-6 w-6 text-[#f4d6b8]" strokeWidth={1.9} />
              </div>
              <div className="min-w-0 flex-1">
                <h1 className="truncate text-base font-semibold leading-tight text-primary">高校学习 AI 助手</h1>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  <Badge variant="secondary" className="border-0 bg-primary/10 px-1.5 py-0 text-xs text-primary">
                    课程答疑
                  </Badge>
                  <Badge
                    variant="secondary"
                    className="border-0 bg-[var(--warning-soft)] px-1.5 py-0 text-xs text-[var(--warning)]"
                  >
                    学业支持
                  </Badge>
                  <Badge
                    variant="secondary"
                    className="border-0 bg-[var(--success-soft)] px-1.5 py-0 text-xs text-[var(--success)]"
                  >
                    计划生成
                  </Badge>
                </div>
              </div>
            </div>
          </div>

          <div className="px-4 pb-3">
            <div className="flex items-center justify-between pb-2">
              <span className="text-xs font-semibold text-muted-foreground">志愿填报</span>
              <button
                type="button"
                onClick={handleNewVolunteer}
                className="rounded px-1.5 py-0.5 text-xs font-medium text-primary hover:bg-sidebar-accent"
              >
                + 新建
              </button>
            </div>
            {volunteerHistory.length > 0 ? (
              <div className="flex max-h-36 flex-col gap-0.5 overflow-y-auto">
                {volunteerHistory.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => handleSelectVolunteer(item.id)}
                    className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                  >
                    <GraduationCap className="h-4 w-4 shrink-0 text-primary" />
                    <div className="min-w-0">
                      <span className="block truncate">{item.title}</span>
                      <span className="block truncate text-[10px] text-muted-foreground">
                        {item.homeRegion} -&gt; {item.targetRegion}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <p className="rounded-lg bg-sidebar-accent/50 px-3 py-2 text-xs text-muted-foreground">
                暂无志愿填报记录
              </p>
            )}
          </div>

          <div className="px-4 pb-4">
            <Button onClick={onNewChat} className="a3-button-primary w-full justify-start gap-2">
              <MessageSquarePlus className="h-4 w-4" />
              发起新对话
            </Button>
          </div>

          <div className="flex min-h-0 flex-1 flex-col">
            <div className="flex items-center justify-between px-4 pb-2">
              <span className="text-xs font-semibold text-muted-foreground">对话</span>
              {chatHistory.length > 0 && onClearChatHistory ? (
                <button
                  type="button"
                  onClick={onClearChatHistory}
                  className="rounded px-1.5 py-0.5 text-xs text-muted-foreground hover:bg-sidebar-accent hover:text-foreground"
                >
                  清空
                </button>
              ) : null}
            </div>
            <div className="flex-1 min-h-0">
              <ScrollArea className="h-full px-2">
                <div className="flex flex-col gap-1 pb-4">
                  {chatHistory.length === 0 ? (
                    <p className="px-3 py-2 text-xs text-muted-foreground">
                      开始一次课程学习对话后，历史会显示在这里。
                    </p>
                  ) : (
                    chatHistory.map((chat) => (
                      <div
                        key={chat.id}
                        className={cn(
                          "group flex items-center rounded-lg transition-colors",
                          "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                          selectedChatId === chat.id
                            ? "bg-sidebar-accent text-sidebar-accent-foreground"
                            : "text-foreground",
                        )}
                      >
                        <button
                          type="button"
                          onClick={() => onSelectChat(chat.id)}
                          className="flex min-w-0 flex-1 items-center gap-2 rounded-l-lg px-3 py-2 text-left text-sm"
                          title={chat.title}
                        >
                          <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground" />
                          <span className="truncate">{chat.title}</span>
                        </button>
                        {onClearChat ? (
                          <button
                            type="button"
                            onClick={() => onClearChat(chat.id)}
                            className="mr-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted-foreground opacity-60 transition hover:bg-[var(--danger-soft)] hover:text-[var(--danger)] hover:opacity-100 focus-visible:opacity-100 group-hover:opacity-100"
                            title="清除此对话"
                            aria-label={`清除对话：${chat.title}`}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        ) : null}
                      </div>
                    ))
                  )}
                </div>
              </ScrollArea>
            </div>
          </div>

          <div className="border-t border-sidebar-border p-4 space-y-2">
            {/* User identity section */}
            {userId ? (
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[var(--primary)] text-sm font-medium text-white">
                  {(nickname || "U").charAt(0).toUpperCase()}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">
                    {nickname || "用户"}
                  </p>
                  <p className="truncate text-xs text-muted-foreground">
                    {userId.slice(0, 10)}...
                  </p>
                </div>
                {onClearUser && (
                  <button
                    onClick={onClearUser}
                    className="rounded-md p-1.5 text-muted-foreground hover:bg-[var(--muted)] hover:text-foreground"
                    title="退出登录"
                  >
                    <LogOut className="h-4 w-4" />
                  </button>
                )}
              </div>
            ) : (
              <Button
                variant="ghost"
                className="w-full justify-start gap-2 text-muted-foreground hover:text-foreground"
                onClick={onStartOnboarding}
              >
                <LogIn className="h-4 w-4" />
                登录 / 注册
              </Button>
            )}

            {/* Settings button (always shown) */}
            <Button variant="ghost" className="w-full justify-start gap-2 text-muted-foreground hover:text-foreground">
              <Settings className="h-4 w-4" />
              设置与帮助
            </Button>
          </div>
        </>
      )}
    </aside>
  )
}
