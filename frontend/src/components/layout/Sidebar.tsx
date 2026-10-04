import { NavLink } from "react-router-dom"
import { History, LayoutDashboard, Loader2, MessageSquare, FileText, ClipboardList, Settings, Wrench } from "lucide-react"
import { useChatHistory } from "@/lib/ChatHistoryContext"
import { cn } from "@/lib/utils"

function historyPreview(conversation: { messages: { role: string; content: string }[] }) {
  const firstUserMessage = conversation.messages.find((m) => m.role === "user")
  return firstUserMessage?.content ?? "(empty conversation)"
}

function historyDate(iso: string) {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
}

const navItems = [
  { to: "/", label: "AI Dashboard", icon: LayoutDashboard, end: true },
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/documents", label: "Documents", icon: FileText },
  { to: "/sops", label: "SOPs", icon: ClipboardList },
]

interface SidebarProps {
  collapsed?: boolean
}

export function Sidebar({ collapsed = false }: SidebarProps) {
  const { conversations, loading, activeConversationId, selectConversation } = useChatHistory()

  return (
    <aside
      className={cn(
        "glass relative flex h-full shrink-0 flex-col overflow-hidden border-r border-sidebar-border bg-sidebar/90 text-sidebar-foreground transition-[width] duration-200 ease-out",
        collapsed ? "w-[76px]" : "w-64",
      )}
    >
      <div className={cn("flex items-center gap-3 px-5 py-5", collapsed && "justify-center px-0")}>
        <div className="brand-gradient flex size-9 shrink-0 items-center justify-center rounded-lg">
          <Wrench className="size-5 text-white" />
        </div>
        {!collapsed && (
          <span className="font-display text-[15px] font-semibold leading-tight tracking-tight whitespace-nowrap">
            Industrial
            <br />
            Maintenance AI
          </span>
        )}
      </div>

      <nav className="mt-2 flex flex-col gap-1 px-3">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            title={collapsed ? item.label : undefined}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium whitespace-nowrap text-sidebar-foreground/65 transition-all",
                collapsed && "justify-center px-0",
                isActive
                  ? "brand-gradient text-white"
                  : "hover:bg-sidebar-accent hover:text-sidebar-foreground",
              )
            }
          >
            <item.icon className="size-4.5 shrink-0" />
            {!collapsed && item.label}
          </NavLink>
        ))}
      </nav>

      <div className="my-3 border-t border-sidebar-border" />

      <nav className="flex flex-col gap-1 px-3">
        <NavLink
          to="/settings"
          title={collapsed ? "Settings" : undefined}
          className={({ isActive }) =>
            cn(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium whitespace-nowrap text-sidebar-foreground/65 transition-all",
              collapsed && "justify-center px-0",
              isActive
                ? "brand-gradient text-white"
                : "hover:bg-sidebar-accent hover:text-sidebar-foreground",
            )
          }
        >
          <Settings className="size-4.5 shrink-0" />
          {!collapsed && "Settings"}
        </NavLink>
      </nav>

      {!collapsed && (
        <>
          <div className="my-3 border-t border-sidebar-border" />

          <div className="flex min-h-0 flex-1 flex-col px-3">
            <div className="flex shrink-0 items-center gap-2 px-2 pb-2">
              <History className="size-3.5 text-sidebar-foreground/50" />
              <span className="font-mono text-xs font-medium tracking-wide text-sidebar-foreground/70 uppercase">
                History
              </span>
            </div>

            <div className="min-h-0 flex-1 space-y-1 overflow-y-auto pb-2">
              {loading ? (
                <div className="flex items-center gap-2 px-2 py-2 text-xs text-sidebar-foreground/50">
                  <Loader2 className="size-3.5 animate-spin" />
                  Loading...
                </div>
              ) : conversations.length === 0 ? (
                <p className="px-2 py-2 text-xs text-sidebar-foreground/50">No conversations yet.</p>
              ) : (
                conversations.map((conv) => (
                  <button
                    key={conv.id}
                    type="button"
                    onClick={() => selectConversation(conv)}
                    className={cn(
                      "flex w-full items-start gap-2 rounded-lg px-2 py-2 text-left transition-colors hover:bg-sidebar-accent",
                      activeConversationId === conv.id && "bg-sidebar-accent",
                    )}
                  >
                    <MessageSquare className="mt-0.5 size-3.5 shrink-0 text-sidebar-foreground/50" />
                    <div className="min-w-0">
                      <p className="truncate text-xs font-medium text-sidebar-foreground/90">
                        {historyPreview(conv)}
                      </p>
                      <p className="text-[10.5px] text-sidebar-foreground/50">{historyDate(conv.created_at)}</p>
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>
        </>
      )}

      <div className="mt-auto shrink-0 p-4">
        {collapsed ? (
          <div className="flex justify-center" title="AI System Status: Healthy">
            <span className="size-2 rounded-full bg-emerald-500 dark:bg-emerald-400" />
          </div>
        ) : (
          <div className="rounded-xl border border-sidebar-border bg-sidebar-accent/60 p-4">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-medium tracking-wide text-sidebar-foreground/90 uppercase">
                AI System Status
              </span>
              <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs font-medium text-emerald-600 dark:text-emerald-400">
                Healthy
              </span>
            </div>
            <div className="mt-2 flex items-center gap-1.5 font-mono text-xs text-sidebar-foreground/75">
              <span className="size-1.5 rounded-full bg-emerald-500 dark:bg-emerald-400" />
              Last updated: 10:15 AM
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}
