import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import type { ChatCitation } from "@/lib/types"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export interface CitationPage {
  page: number | null
  /** Every cited passage on this page — each keeps its own [n] number. */
  citations: ChatCitation[]
}

export interface CitationGroup {
  documentId: string
  documentName: string
  pages: CitationPage[]
}

/**
 * Citations grouped by source file and then by page, in first-cited order, so
 * the file name is shown once and a page cited several times is shown once.
 */
export function groupCitationsByDocument(citations: ChatCitation[]): CitationGroup[] {
  const groups = new Map<string, CitationGroup>()
  for (const c of citations) {
    const key = c.document_id || c.document_name
    const group = groups.get(key) ?? { documentId: key, documentName: c.document_name, pages: [] }
    const page = group.pages.find((p) => p.page === c.page)
    if (page) page.citations.push(c)
    else group.pages.push({ page: c.page, citations: [c] })
    groups.set(key, group)
  }
  return [...groups.values()]
}

/** "Document · p. 12 · Section > Subsection" — whichever parts the citation has. */
export function citationLocation(c: ChatCitation): string {
  const parts = [c.document_name]
  if (c.page != null) parts.push(`p. ${c.page}`)
  if (c.section_path) parts.push(c.section_path)
  return parts.join(" · ")
}
