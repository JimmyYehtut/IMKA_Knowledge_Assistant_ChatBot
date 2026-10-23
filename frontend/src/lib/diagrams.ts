import { API_URL } from "@/lib/api"
import type { ChatCitation } from "@/lib/types"

/**
 * Diagrams are figure images cut from the source documents at ingestion
 * ("diagrams/<document_id>/figN_k.png", relative to backend/data) and served by
 * the API under /data. Answers embed them as Markdown images; citations of
 * figure passages carry one in `image_path`.
 */
export const DIAGRAM_PATH_PREFIX = "diagrams/"

/** A path the API serves from the diagrams folder — anything else is never loaded. */
export function isDiagramPath(path: string | null | undefined): path is string {
  return !!path && path.startsWith(DIAGRAM_PATH_PREFIX) && !path.includes("..") && !path.includes("\\")
}

export function diagramUrl(path: string): string {
  return `${API_URL}/data/${path}`
}

export interface Diagram {
  path: string
  caption: string
}

const IMAGE_MARKDOWN_RE = /!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)/g

/** Diagram paths the answer already shows inline, in order of appearance. */
export function embeddedDiagramPaths(content: string): Set<string> {
  const paths = new Set<string>()
  for (const match of content.matchAll(IMAGE_MARKDOWN_RE)) {
    if (isDiagramPath(match[2])) paths.add(match[2])
  }
  return paths
}

/**
 * Figures the answer cited (a `figure` passage, or any passage carrying an
 * image) but did not draw inline — shown under the answer so a cited diagram
 * is never only a source chip. One entry per image, in citation order.
 */
export function citedDiagramsNotEmbedded(content: string, citations: ChatCitation[] | undefined): Diagram[] {
  if (!citations?.length) return []
  const shown = embeddedDiagramPaths(content)
  const result: Diagram[] = []
  for (const c of citations) {
    const path = c.image_path
    if (!isDiagramPath(path) || shown.has(path)) continue
    shown.add(path)
    const location = c.section_path ? `${c.document_name} · ${c.section_path}` : c.document_name
    result.push({ path, caption: c.figure_caption || location })
  }
  return result
}

/**
 * While an answer streams in, a diagram arrives token by token and its
 * Markdown is briefly incomplete ("![Figure 9 Lifting of unpa" or
 * "...](diagrams/cdbd"), which would flash as raw text. Hide that unfinished
 * tail until the closing ")" arrives. Only the final line is inspected, and a
 * finished answer has no unclosed image there, so it renders unchanged.
 */
export function hideUnfinishedImage(content: string): string {
  const lineStart = content.lastIndexOf("\n") + 1
  const lastLine = content.slice(lineStart)
  const start = lastLine.lastIndexOf("![")
  if (start === -1) return content
  const tail = lastLine.slice(start)
  if (/^!\[[^\]]*\]\([^)]*\)/.test(tail)) return content
  return content.slice(0, lineStart + start)
}
