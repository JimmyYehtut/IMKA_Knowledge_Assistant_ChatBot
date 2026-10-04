import { useEffect, useState } from "react"
import { ChevronDown } from "lucide-react"
import { getTopics, type DocumentTopic } from "@/lib/api"
import { getPersona, type PersonaId } from "@/lib/personas"
import { cn } from "@/lib/utils"

interface PersonaTopicsCardProps {
  persona: PersonaId
  onAsk?: (question: string) => void
  disabled?: boolean
}

/** Topics the ingested documents cover for the current persona; clicking one asks about it. */
export function PersonaTopicsCard({ persona, onAsk, disabled = false }: PersonaTopicsCardProps) {
  // null = still loading; the list depends on the persona and on what has been ingested.
  const [topics, setTopics] = useState<DocumentTopic[] | null>(null)
  const [open, setOpen] = useState(true)

  useEffect(() => {
    let cancelled = false
    setTopics(null)
    getTopics(persona)
      .then((result) => !cancelled && setTopics(result))
      .catch(() => !cancelled && setTopics([]))
    return () => {
      cancelled = true
    }
  }, [persona])

  const chapters = [...new Set((topics ?? []).map((t) => t.chapter))]

  return (
    <div className="glass rounded-xl border border-border p-4">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-start justify-between gap-2 text-left"
      >
        <div>
          <h3 className="text-sm font-semibold text-foreground">
            Topics you can ask{topics ? ` (${topics.length})` : ""}
          </h3>
          <p className="text-xs text-muted-foreground">For the {getPersona(persona).name}</p>
        </div>
        <ChevronDown
          className={cn("mt-0.5 size-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")}
        />
      </button>

      {open && (
        <div className="mt-3">
          {topics === null ? (
            <p className="text-xs text-muted-foreground">Loading topics from the knowledge base...</p>
          ) : topics.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No topics found. Upload documents on the Documents page to build the knowledge base.
            </p>
          ) : (
            <div className="max-h-72 space-y-3 overflow-y-auto pr-1">
              {chapters.map((chapter) => (
                <div key={chapter}>
                  <p className="mb-1.5 text-xs font-medium text-foreground">{chapter}</p>
                  <div className="flex flex-wrap gap-1.5">
                    {topics
                      .filter((t) => t.chapter === chapter)
                      .map((t) => (
                        <button
                          key={`${t.document_name}-${t.title}`}
                          type="button"
                          disabled={disabled}
                          title={`${t.document_name}${t.page != null ? ` · p. ${t.page}` : ""}`}
                          onClick={() => onAsk?.(`What does the manual say about ${t.title.toLowerCase()}?`)}
                          className="rounded-full bg-muted px-2.5 py-1 text-left text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-50"
                        >
                          {t.title}
                        </button>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
