import { useEffect, useState } from "react"
import { Bot, FileText, MessageSquarePlus, PanelRightClose, PanelRightOpen } from "lucide-react"
import { ChatThread } from "@/components/chat/ChatThread"
import { CitationsCard } from "@/components/chat/CitationsCard"
import { PersonaTopicsCard } from "@/components/chat/PersonaTopicsCard"
import { SessionTopicsCard } from "@/components/chat/SessionTopicsCard"
import { ApiError, streamChatCompletion, submitFeedback, type Conversation } from "@/lib/api"
import { useAuth } from "@/lib/AuthContext"
import { useChatHistory } from "@/lib/ChatHistoryContext"
import { DEFAULT_PERSONA, getPersona, type PersonaId } from "@/lib/personas"
import type { ChatMessage } from "@/lib/types"

const SIDE_PANEL_COLLAPSED_KEY = "chat_side_panel_collapsed"
const LAST_PERSONA_KEY = "chat_last_persona"

function timeNow() {
  return new Date().toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })
}

export function ChatPage() {
  const { token, user, logout } = useAuth()
  const { refresh: refreshHistory, pendingConversation, clearPending, setActiveConversationId } = useChatHistory()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Chosen on the new-chat screen; the backend fixes it on the conversation at the first message.
  const [persona, setPersona] = useState<PersonaId>(() => getPersona(localStorage.getItem(LAST_PERSONA_KEY)).id)

  const [panelCollapsed, setPanelCollapsed] = useState(
    () => localStorage.getItem(SIDE_PANEL_COLLAPSED_KEY) === "1",
  )

  useEffect(() => {
    localStorage.setItem(SIDE_PANEL_COLLAPSED_KEY, panelCollapsed ? "1" : "0")
  }, [panelCollapsed])

  const activePersona = getPersona(persona)
  const lastCitations = [...messages].reverse().find((m) => m.role === "assistant")?.citations ?? []

  useEffect(() => {
    if (!pendingConversation) return
    applyConversation(pendingConversation)
    clearPending()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingConversation])

  function appendToMessage(localId: string, delta: string) {
    setMessages((prev) => prev.map((m) => (m.id === localId ? { ...m, content: m.content + delta } : m)))
  }

  async function runCompletion(
    history: { role: string; content: string }[],
    assistantLocalId: string,
    userLocalId?: string,
  ) {
    if (!token) return
    setError(null)
    setSending(true)
    try {
      const { conversationId: newConversationId, messageId, citations, topic } = await streamChatCompletion(
        token,
        {
          messages: history as { role: "user" | "assistant" | "system"; content: string }[],
          conversation_id: conversationId,
          persona,
        },
        (delta) => appendToMessage(assistantLocalId, delta),
      )
      setConversationId(newConversationId)
      setActiveConversationId(newConversationId)
      setMessages((prev) =>
        prev.map((m) => {
          if (m.id === assistantLocalId) return { ...m, citations, serverId: messageId }
          if (m.id === userLocalId) return { ...m, topic }
          return m
        }),
      )
      refreshHistory()
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        logout()
        return
      }
      setError(e instanceof ApiError ? e.message : "Failed to reach the chat API.")
    } finally {
      setSending(false)
    }
  }

  async function handleSend(text: string) {
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      author: user?.name ?? "You",
      time: timeNow(),
      content: text,
    }
    const assistantId = crypto.randomUUID()
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      author: "AI Assistant",
      time: timeNow(),
      content: "",
    }
    const history = [...messages, userMessage].map((m) => ({ role: m.role, content: m.content }))
    setMessages((prev) => [...prev, userMessage, assistantMessage])
    await runCompletion(history, assistantId, userMessage.id)
  }

  async function handleRegenerate(assistantLocalId: string) {
    const idx = messages.findIndex((m) => m.id === assistantLocalId)
    if (idx === -1) return
    let userIdx = idx - 1
    while (userIdx >= 0 && messages[userIdx].role !== "user") userIdx--
    if (userIdx < 0) return

    const history = messages.slice(0, userIdx + 1).map((m) => ({ role: m.role, content: m.content }))
    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantLocalId
          ? { ...m, content: "", citations: undefined, rating: null, comment: null, serverId: undefined }
          : m,
      ),
    )
    await runCompletion(history, assistantLocalId, messages[userIdx].id)
  }

  async function handleEditUser(userLocalId: string, newContent: string) {
    const idx = messages.findIndex((m) => m.id === userLocalId)
    if (idx === -1) return

    const truncated = messages.slice(0, idx)
    const editedUserMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      author: user?.name ?? "You",
      time: timeNow(),
      content: newContent,
    }
    const assistantId = crypto.randomUUID()
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      author: "AI Assistant",
      time: timeNow(),
      content: "",
    }
    const history = [...truncated, editedUserMessage].map((m) => ({ role: m.role, content: m.content }))
    setMessages([...truncated, editedUserMessage, assistantMessage])
    await runCompletion(history, assistantId, editedUserMessage.id)
  }

  async function handleFeedback(localId: string, rating: "up" | "down") {
    if (!token) return
    const msg = messages.find((m) => m.id === localId)
    if (!msg?.serverId) return

    const previousRating = msg.rating
    const newRating = previousRating === rating ? null : rating
    setMessages((prev) => prev.map((m) => (m.id === localId ? { ...m, rating: newRating } : m)))

    try {
      await submitFeedback(token, msg.serverId, { rating: newRating, comment: msg.comment })
    } catch {
      setMessages((prev) => prev.map((m) => (m.id === localId ? { ...m, rating: previousRating } : m)))
    }
  }

  async function handleComment(localId: string, comment: string) {
    if (!token) return
    const msg = messages.find((m) => m.id === localId)
    if (!msg?.serverId) return

    setMessages((prev) => prev.map((m) => (m.id === localId ? { ...m, comment } : m)))
    try {
      await submitFeedback(token, msg.serverId, { rating: msg.rating, comment })
    } catch {
      // best-effort; leave the optimistic value in place
    }
  }

  function handlePersonaChange(next: PersonaId) {
    setPersona(next)
    localStorage.setItem(LAST_PERSONA_KEY, next)
  }

  function handleNewChat() {
    setMessages([])
    setConversationId(null)
    setActiveConversationId(null)
    setError(null)
  }

  function applyConversation(conversation: Conversation) {
    setConversationId(conversation.id)
    setPersona(conversation.persona ? getPersona(conversation.persona).id : DEFAULT_PERSONA)
    setError(null)
    setMessages(
      conversation.messages.map((m) => ({
        id: m.id,
        serverId: m.id,
        role: m.role === "assistant" ? "assistant" : "user",
        author: m.role === "assistant" ? "AI Assistant" : (user?.name ?? "You"),
        time: new Date(conversation.created_at).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }),
        content: m.content,
        topic: m.topic,
        citations: m.citations,
        rating: m.rating,
        comment: m.feedback_comment,
      })),
    )
  }

  return (
    <div className="flex h-full min-h-0 gap-4 p-4">
      <div className="glass flex min-w-0 flex-1 flex-col rounded-xl border border-border">
        <div className="flex shrink-0 items-center justify-between border-b border-border px-5 py-4">
          <div className="flex items-center gap-3">
            <div className="brand-gradient flex size-9 items-center justify-center rounded-full">
              <Bot className="size-4.5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-display font-semibold tracking-tight text-foreground">AI Assistant</span>
                <span className="flex items-center gap-1 font-mono text-xs text-emerald-600 dark:text-emerald-400">
                  <span className="size-1.5 rounded-full bg-emerald-500 dark:bg-emerald-400" />
                  Online
                </span>
                {messages.length > 0 && (
                  <span
                    className="flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground"
                    title={`Answers for: ${activePersona.name} (${activePersona.detailLevel})`}
                  >
                    <activePersona.icon className="size-3" />
                    {activePersona.name} · {activePersona.detailLevel}
                  </span>
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                Ask questions about maintenance, SOPs, equipment and troubleshooting.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleNewChat}
              className="flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm font-medium text-foreground transition-colors hover:border-ring/50 hover:bg-accent"
            >
              <MessageSquarePlus className="size-4" />
              New Chat
            </button>
            <button
              type="button"
              onClick={() => setPanelCollapsed((v) => !v)}
              className="hidden rounded-lg border border-border p-2 text-muted-foreground transition-colors hover:border-ring/50 hover:bg-accent hover:text-foreground lg:flex"
              aria-label={panelCollapsed ? "Show sources" : "Hide sources"}
              title={panelCollapsed ? "Show sources" : "Hide sources"}
            >
              {panelCollapsed ? <PanelRightOpen className="size-4" /> : <PanelRightClose className="size-4" />}
            </button>
          </div>
        </div>

        {error && (
          <div className="shrink-0 border-b border-border bg-destructive/10 px-5 py-2 text-xs text-red-600 dark:text-red-300">
            {error}
          </div>
        )}

        <div className="min-h-0 flex-1">
          <ChatThread
            messages={messages}
            onSend={handleSend}
            sending={sending}
            persona={persona}
            onPersonaChange={handlePersonaChange}
            onFeedback={handleFeedback}
            onComment={handleComment}
            onRegenerate={handleRegenerate}
            onEditUser={handleEditUser}
          />
        </div>
      </div>

      {panelCollapsed ? (
        <div className="hidden shrink-0 lg:block">
          <button
            type="button"
            onClick={() => setPanelCollapsed(false)}
            className="glass flex w-12 flex-col items-center gap-3 rounded-xl border border-border py-4 text-muted-foreground transition-colors hover:text-foreground"
            aria-label="Show sources"
            title="Show sources"
          >
            <FileText className="size-4.5" />
          </button>
        </div>
      ) : (
        <div className="hidden w-80 shrink-0 space-y-4 overflow-y-auto lg:block">
          <PersonaTopicsCard persona={persona} onAsk={handleSend} disabled={sending} />
          <SessionTopicsCard messages={messages} />
          <CitationsCard citations={lastCitations} />
        </div>
      )}
    </div>
  )
}
