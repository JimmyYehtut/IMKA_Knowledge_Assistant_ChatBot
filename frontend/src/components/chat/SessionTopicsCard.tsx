import type { ChatMessage } from "@/lib/types"

interface SessionTopicsCardProps {
  messages: ChatMessage[]
}

/** Topics of the questions asked in the current chat session, oldest first. */
export function SessionTopicsCard({ messages }: SessionTopicsCardProps) {
  // topic === "" marks a greeting/meta message; undefined/null means no label was stored (older chats).
  const asked = messages.filter((m) => m.role === "user" && m.topic !== "")

  return (
    <div className="glass rounded-xl border border-border p-4">
      <h3 className="mb-3 text-sm font-semibold text-foreground">Session topics ({asked.length})</h3>

      {asked.length === 0 ? (
        <p className="text-xs text-muted-foreground">Topics you ask about in this chat will be listed here.</p>
      ) : (
        <ol className="space-y-1">
          {asked.map((m, i) => (
            <li key={m.id}>
              <button
                type="button"
                title={m.content}
                onClick={() =>
                  document.getElementById(`msg-${m.id}`)?.scrollIntoView({ behavior: "smooth", block: "start" })
                }
                className="flex w-full items-baseline gap-2 rounded-lg px-2 py-1.5 text-left text-sm text-foreground transition-colors hover:bg-accent"
              >
                <span className="font-mono text-xs text-muted-foreground">{i + 1}</span>
                <span className="min-w-0 truncate">{m.topic || m.content}</span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
