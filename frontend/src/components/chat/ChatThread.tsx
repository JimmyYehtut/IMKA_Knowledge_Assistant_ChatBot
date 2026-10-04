import { useEffect, useRef, useState } from "react"
import { Mic, Paperclip, Send } from "lucide-react"
import type { PersonaId } from "@/lib/personas"
import type { ChatMessage } from "@/lib/types"
import { ChatEmptyState } from "@/components/chat/ChatEmptyState"
import { MessageBubble } from "@/components/chat/MessageBubble"

interface ChatThreadProps {
  messages: ChatMessage[]
  onSend?: (text: string) => void
  sending?: boolean
  persona: PersonaId
  onPersonaChange?: (persona: PersonaId) => void
  onFeedback?: (localId: string, rating: "up" | "down") => void
  onComment?: (localId: string, comment: string) => void
  onRegenerate?: (localId: string) => void
  onEditUser?: (localId: string, newContent: string) => void
}

export function ChatThread({
  messages,
  onSend,
  sending = false,
  persona,
  onPersonaChange,
  onFeedback,
  onComment,
  onRegenerate,
  onEditUser,
}: ChatThreadProps) {
  const [draft, setDraft] = useState("")
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [messages])

  function submit() {
    const text = draft.trim()
    if (!text || sending) return
    onSend?.(text)
    setDraft("")
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-6 overflow-y-auto p-6">
        {messages.length === 0 && !sending && (
          <ChatEmptyState persona={persona} onPersonaChange={onPersonaChange} onAsk={onSend} />
        )}
        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            busy={sending}
            onFeedback={onFeedback}
            onComment={onComment}
            onRegenerate={onRegenerate}
            onEditUser={onEditUser}
          />
        ))}
      </div>

      <div className="shrink-0 border-t border-border p-4">
        <div className="glass focus-within:ring-2 focus-within:ring-ring flex items-center gap-2 rounded-full border border-border px-2 py-1.5 transition-shadow">
          <button type="button" className="rounded-full p-1.5 text-muted-foreground hover:bg-accent" aria-label="Attach file">
            <Paperclip className="size-4" />
          </button>
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit()
            }}
            type="text"
            placeholder="Ask anything about maintenance..."
            className="h-7 flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
          />
          <button type="button" className="rounded-full p-1.5 text-muted-foreground hover:bg-accent" aria-label="Voice input">
            <Mic className="size-4" />
          </button>
          <button
            type="button"
            onClick={submit}
            disabled={sending || !draft.trim()}
            className="brand-gradient flex size-8 shrink-0 items-center justify-center rounded-full text-white transition-transform hover:brightness-110 active:scale-95 disabled:opacity-40"
            aria-label="Send message"
          >
            <Send className="size-4" />
          </button>
        </div>
        <p className="mt-2 text-center text-xs text-muted-foreground">
          AI responses are generated from your organization's documents. Please verify critical actions.
        </p>
      </div>
    </div>
  )
}
