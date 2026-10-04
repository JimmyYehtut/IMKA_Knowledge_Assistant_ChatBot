import { Bot, MessageCircleQuestion } from "lucide-react"
import { PERSONAS, STARTER_QUESTIONS, type PersonaId } from "@/lib/personas"
import { cn } from "@/lib/utils"

interface ChatEmptyStateProps {
  persona: PersonaId
  onPersonaChange?: (persona: PersonaId) => void
  onAsk?: (question: string) => void
}

/** New-chat screen: pick the answer persona, then type or pick a starter question. */
export function ChatEmptyState({ persona, onPersonaChange, onAsk }: ChatEmptyStateProps) {
  return (
    <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col justify-center gap-8 py-4">
      <div className="flex flex-col items-center gap-2 text-center">
        <div className="brand-gradient flex size-11 items-center justify-center rounded-full">
          <Bot className="size-5 text-white" />
        </div>
        <h2 className="font-display text-lg font-semibold tracking-tight text-foreground">How can I help?</h2>
        <p className="text-sm text-muted-foreground">Choose who the answers are for, then ask a question.</p>
      </div>

      <section>
        <h3 className="mb-2 font-mono text-xs tracking-wide text-muted-foreground uppercase">Answer for</h3>
        <div role="radiogroup" aria-label="Answer persona" className="grid gap-3 sm:grid-cols-3">
          {PERSONAS.map((p) => {
            const selected = p.id === persona
            return (
              <button
                key={p.id}
                type="button"
                role="radio"
                aria-checked={selected}
                onClick={() => onPersonaChange?.(p.id)}
                className={cn(
                  "flex flex-col gap-2 rounded-xl border p-4 text-left transition-colors",
                  selected ? "border-primary bg-primary/10" : "border-border hover:border-ring/50 hover:bg-accent",
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <p.icon className={cn("size-5", selected ? "text-primary" : "text-muted-foreground")} />
                  <span
                    className={cn(
                      "rounded-full px-2 py-0.5 font-mono text-xs tracking-wide uppercase",
                      selected ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground",
                    )}
                  >
                    {p.detailLevel}
                  </span>
                </div>
                <p className="text-sm font-semibold text-foreground">{p.name}</p>
                <p className="text-xs leading-relaxed text-muted-foreground">{p.description}</p>
              </button>
            )
          })}
        </div>
      </section>

      <section>
        <h3 className="mb-2 font-mono text-xs tracking-wide text-muted-foreground uppercase">Try asking</h3>
        <div className="grid gap-2 sm:grid-cols-2">
          {STARTER_QUESTIONS.map((question) => (
            <button
              key={question}
              type="button"
              onClick={() => onAsk?.(question)}
              className="flex items-start gap-2.5 rounded-lg border border-border px-3 py-2.5 text-left text-sm text-foreground transition-colors hover:border-ring/50 hover:bg-accent"
            >
              <MessageCircleQuestion className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
              {question}
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}
