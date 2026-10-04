import { FileText } from "lucide-react"
import { API_URL } from "@/lib/api"
import type { ChatCitation } from "@/lib/types"
import { groupCitationsByDocument } from "@/lib/utils"

interface CitationsCardProps {
  citations: ChatCitation[]
}

export function CitationsCard({ citations }: CitationsCardProps) {
  const groups = groupCitationsByDocument(citations)
  const pageCount = groups.reduce((total, group) => total + group.pages.length, 0)

  return (
    <div className="glass rounded-xl border border-border p-4">
      <h3 className="mb-3 text-sm font-semibold text-foreground">Sources ({pageCount})</h3>

      {citations.length === 0 ? (
        <p className="text-xs text-muted-foreground">Ask a question to see the documents the answer cited.</p>
      ) : (
        <div className="space-y-2">
          {/* One card per file: the file name once, then each cited page under it. */}
          {groups.map((group) => (
            <div key={group.documentId} className="rounded-lg border border-border p-3">
              <p className="flex items-start gap-2 text-sm font-medium text-foreground">
                <FileText className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                <span className="min-w-0 break-words">{group.documentName}</span>
              </p>

              <ul className="mt-2 space-y-2">
                {group.pages.map((page) => {
                  // A page cited by several passages is one row: all their numbers, then the
                  // distinct sections, chunk types and diagrams those passages carry.
                  const sections = [...new Set(page.citations.map((c) => c.section_path).filter(Boolean))]
                  const types = [...new Set(page.citations.map((c) => c.chunk_type).filter(Boolean))]
                  const images = [...new Set(page.citations.map((c) => c.image_path).filter(Boolean))]
                  return (
                    <li key={page.page ?? "none"} className="flex flex-col gap-2">
                      <div className="text-xs text-muted-foreground">
                        <p className="flex flex-wrap items-center gap-1.5">
                          {page.citations.map(
                            (c) =>
                              c.number != null && (
                                <span
                                  key={c.number}
                                  className="flex h-5 min-w-5 items-center justify-center rounded bg-primary/15 px-1 font-mono font-medium text-primary"
                                >
                                  {c.number}
                                </span>
                              ),
                          )}
                          <span className="font-mono font-medium text-foreground">
                            {page.page !== null ? `p. ${page.page}` : "No page"}
                          </span>
                          {types.map((type) => (
                            <span key={type} className="rounded bg-muted px-1 py-0.5 font-mono tracking-wide uppercase">
                              {type}
                            </span>
                          ))}
                        </p>
                        {sections.map((section) => (
                          <p key={section} className="mt-0.5">
                            {section}
                          </p>
                        ))}
                      </div>
                      {images.map((image) => (
                        <img
                          key={image}
                          src={`${API_URL}/data/${image}`}
                          alt={`Diagram from ${group.documentName}${page.page !== null ? `, page ${page.page}` : ""}`}
                          className="max-h-64 w-full rounded-md border border-border bg-white object-contain"
                          loading="lazy"
                        />
                      ))}
                    </li>
                  )
                })}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
