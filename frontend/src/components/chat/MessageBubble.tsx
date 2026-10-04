import { useState } from "react"
import {
  Bot,
  Check,
  Copy,
  FileText,
  MessageSquare,
  Pencil,
  RotateCcw,
  ThumbsDown,
  ThumbsUp,
  X,
} from "lucide-react"
import { MarkdownContent } from "@/components/chat/MarkdownContent"
import type { ChatMessage } from "@/lib/types"
import { citationLocation, cn, groupCitationsByDocument } from "@/lib/utils"

interface MessageBubbleProps {
  message: ChatMessage
  busy?: boolean
  onFeedback?: (localId: string, rating: "up" | "down") => void
  onComment?: (localId: string, comment: string) => void
  onRegenerate?: (localId: string) => void
  onEditUser?: (localId: string, newContent: string) => void
}

function IconButton({
  active,
  danger,
  label,
  onClick,
  children,
}: {
  active?: boolean
  danger?: boolean
  label: string
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={cn(
        "rounded-md p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground",
        active && !danger && "bg-primary/15 text-primary",
        active && danger && "bg-destructive/15 text-red-600 dark:text-red-400",
      )}
    >
      {children}
    </button>
  )
}

export function MessageBubble({ message, busy, onFeedback, onComment, onRegenerate, onEditUser }: MessageBubbleProps) {
  const [editing, setEditing] = useState(false)
  const [editDraft, setEditDraft] = useState(message.content)
  const [commenting, setCommenting] = useState(false)
  const [commentDraft, setCommentDraft] = useState(message.comment ?? "")
  const [copied, setCopied] = useState(false)

  function handleCopy() {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  function submitComment() {
    if (!commentDraft.trim()) return
    onComment?.(message.id, commentDraft.trim())
    setCommenting(false)
  }

  function submitEdit() {
    const text = editDraft.trim()
    if (!text || text === message.content) {
      setEditing(false)
      return
    }
    onEditUser?.(message.id, text)
    setEditing(false)
  }

  if (message.role === "user") {
    return (
      <div id={`msg-${message.id}`} className="group flex justify-end">
        <div className="max-w-[80%]">
          <div className="mb-1 flex items-baseline justify-end gap-2 text-xs text-muted-foreground">
            <span className="font-medium text-foreground">{message.author}</span>
            <span>{message.time}</span>
          </div>

          {editing ? (
            <div className="rounded-2xl rounded-tr-sm border border-border bg-background p-2">
              <textarea
                value={editDraft}
                onChange={(e) => setEditDraft(e.target.value)}
                rows={3}
                autoFocus
                className="w-full resize-none bg-transparent text-sm text-foreground focus:outline-none"
              />
              <div className="mt-1 flex justify-end gap-1.5">
                <button
                  type="button"
                  onClick={() => setEditing(false)}
                  className="rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-accent"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={submitEdit}
                  className="brand-gradient rounded-md px-2 py-1 text-xs font-medium text-white hover:brightness-110"
                >
                  Save & resend
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className="brand-gradient rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm whitespace-pre-wrap text-white">
                {message.content}
              </div>
              <div className="mt-1 flex justify-end opacity-0 transition-opacity group-hover:opacity-100">
                <IconButton label="Edit and resend" onClick={() => { setEditDraft(message.content); setEditing(true) }}>
                  <Pencil className="size-3.5" />
                </IconButton>
              </div>
            </>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-3">
      <div className="brand-gradient flex size-8 shrink-0 items-center justify-center rounded-full">
        <Bot className="size-4 text-white" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 flex items-baseline gap-2 text-xs text-muted-foreground">
          <span className="font-medium text-foreground">{message.author}</span>
          <span>{message.time}</span>
        </div>

        <div className="space-y-3 text-sm text-foreground">
          {message.content ? (
            <MarkdownContent
              content={message.content}
              citations={message.citations}
              citationAnchorPrefix={`cite-${message.id}`}
            />
          ) : (
            <div className="flex items-center gap-2 py-1 text-muted-foreground">
              <div className="flex gap-1">
                <span className="size-1.5 animate-bounce rounded-full bg-primary [animation-delay:-0.3s]" />
                <span className="size-1.5 animate-bounce rounded-full bg-primary [animation-delay:-0.15s]" />
                <span className="size-1.5 animate-bounce rounded-full bg-primary" />
              </div>
              <span className="font-mono text-xs">Thinking...</span>
            </div>
          )}

          {message.citations && message.citations.length > 0 && (
            <div className="space-y-1.5">
              {groupCitationsByDocument(message.citations).map((group) => (
                <div key={group.documentId} className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1.5 font-medium text-foreground">
                    <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                    {group.documentName}
                  </span>
                  {group.pages.map((page) => (
                    <span
                      key={page.page ?? "none"}
                      title={page.citations.map(citationLocation).join("\n")}
                      className="flex items-center gap-1 rounded-full bg-muted px-2 py-0.5"
                    >
                      {/* Each number keeps its own anchor so every [n] badge in the answer has a target. */}
                      {page.citations.map(
                        (c) =>
                          c.number != null && (
                            <span
                              key={c.number}
                              id={`cite-${message.id}-${c.number}`}
                              className="font-mono font-medium text-primary"
                            >
                              [{c.number}]
                            </span>
                          ),
                      )}
                      <span className="font-mono">{page.page !== null ? `p. ${page.page}` : "no page"}</span>
                    </span>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>

        {message.content && !busy && (
          <div className="mt-1.5 flex items-center gap-0.5">
            <IconButton label="Copy" onClick={handleCopy}>
              {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
            </IconButton>
            <IconButton
              label="Good response"
              active={message.rating === "up"}
              onClick={() => onFeedback?.(message.id, "up")}
            >
              <ThumbsUp className="size-3.5" />
            </IconButton>
            <IconButton
              label="Bad response"
              active={message.rating === "down"}
              danger
              onClick={() => onFeedback?.(message.id, "down")}
            >
              <ThumbsDown className="size-3.5" />
            </IconButton>
            <IconButton label="Add comment" active={commenting} onClick={() => setCommenting((v) => !v)}>
              <MessageSquare className="size-3.5" />
            </IconButton>
            <IconButton label="Regenerate" onClick={() => onRegenerate?.(message.id)}>
              <RotateCcw className="size-3.5" />
            </IconButton>
          </div>
        )}

        {message.comment && !commenting && (
          <p className="mt-1.5 rounded-lg bg-muted px-2.5 py-1.5 text-xs text-muted-foreground">
            Your feedback: {message.comment}
          </p>
        )}

        {commenting && (
          <div className="mt-1.5 flex items-start gap-1.5">
            <textarea
              value={commentDraft}
              onChange={(e) => setCommentDraft(e.target.value)}
              autoFocus
              rows={2}
              placeholder="What could be improved?"
              className="flex-1 resize-none rounded-lg border border-border bg-background p-2 text-xs text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <div className="flex flex-col gap-1">
              <button
                type="button"
                onClick={submitComment}
                className="brand-gradient rounded-md p-1.5 text-white hover:brightness-110"
                aria-label="Submit comment"
              >
                <Check className="size-3.5" />
              </button>
              <button
                type="button"
                onClick={() => setCommenting(false)}
                className="rounded-md p-1.5 text-muted-foreground hover:bg-accent"
                aria-label="Cancel comment"
              >
                <X className="size-3.5" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
