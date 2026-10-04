import { useMemo, type ReactNode } from "react"
import ReactMarkdown, { type Components } from "react-markdown"
import rehypeKatex from "rehype-katex"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"
import "katex/dist/katex.min.css"
import { API_URL } from "@/lib/api"
import type { ChatCitation } from "@/lib/types"
import { citationLocation } from "@/lib/utils"

/**
 * LLMs commonly write LaTeX math with \( \) / \[ \] delimiters (ChatGPT-style)
 * instead of the $ $ / $$ $$ that remark-math expects — normalize before parsing
 * so both conventions render instead of showing up as literal escaped brackets.
 */
function normalizeMathDelimiters(content: string): string {
  return content
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, expr) => `$$${expr}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, expr) => `$${expr}$`)
}

const CITE_HREF_PREFIX = "#cite-"
const DIAGRAM_PATH_PREFIX = "diagrams/"

/**
 * Turns the answer's inline citation markers ("[2]", "[1, 3]") into links the
 * `a` renderer shows as badges. Only numbers present in this message's
 * citations are linked, so unrelated bracketed numbers are left alone.
 */
function linkCitationMarkers(content: string, citations: ChatCitation[]): string {
  const known = new Set(citations.map((c) => c.number))
  return content.replace(/\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g, (match, group: string) => {
    const numbers = group.split(",").map((n) => Number(n.trim()))
    if (!numbers.every((n) => known.has(n))) return match
    return numbers.map((n) => `[${n}](${CITE_HREF_PREFIX}${n})`).join("")
  })
}

const components: Components = {
  p: ({ children }) => <p className="leading-relaxed whitespace-pre-wrap">{children}</p>,
  h1: ({ children }) => <h1 className="mt-3 mb-1.5 font-display text-base font-semibold tracking-tight first:mt-0">{children}</h1>,
  h2: ({ children }) => <h2 className="mt-3 mb-1.5 font-display text-[0.95rem] font-semibold tracking-tight first:mt-0">{children}</h2>,
  h3: ({ children }) => <h3 className="mt-2.5 mb-1 font-display text-sm font-semibold tracking-tight first:mt-0">{children}</h3>,
  ul: ({ children }) => <ul className="list-disc space-y-1 pl-5 marker:text-muted-foreground">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal space-y-1 pl-5 marker:text-muted-foreground">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-primary underline underline-offset-2 hover:no-underline">
      {children}
    </a>
  ),
  // Diagrams the answer embeds from the knowledge base ("diagrams/<document_id>/figN_k.png",
  // served by the API under /data). Any other image source is not loaded; only its caption shows.
  img: ({ src, alt }) => {
    const path = typeof src === "string" ? src : ""
    if (!path.startsWith(DIAGRAM_PATH_PREFIX)) {
      return <span className="text-muted-foreground italic">{alt}</span>
    }
    const url = `${API_URL}/data/${path}`
    return (
      <span className="my-2 block">
        <a href={url} target="_blank" rel="noopener noreferrer" title="Open diagram in a new tab">
          <img
            src={url}
            alt={alt ?? "Diagram"}
            loading="lazy"
            className="max-h-96 max-w-full rounded-lg border border-border bg-white object-contain p-2"
          />
        </a>
        {alt && <span className="mt-1.5 block text-xs font-medium text-muted-foreground">{alt}</span>}
      </span>
    )
  },
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-primary/40 pl-3 text-muted-foreground">{children}</blockquote>
  ),
  hr: () => <hr className="border-border" />,
  code: ({ className, children }) => {
    const isBlock = /language-/.test(className ?? "")
    if (isBlock) {
      return <code className={className}>{children}</code>
    }
    return (
      <code className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em] text-foreground">{children}</code>
    )
  },
  pre: ({ children }) => (
    <pre className="overflow-x-auto rounded-lg border border-border bg-muted/60 p-3 font-mono text-xs text-foreground">
      {children}
    </pre>
  ),
  // Answer tables: dark header row, full grid lines, zebra rows. `style` carries the
  // column alignment remark-gfm derives from the Markdown separator row (e.g. `---:`).
  table: ({ children }) => (
    <div className="overflow-x-auto rounded-lg border border-slate-400 dark:border-slate-500">
      <table className="w-full border-collapse text-left text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-slate-800 text-white dark:bg-slate-700">{children}</thead>,
  th: ({ children, style }) => (
    <th
      style={style}
      className="border-r border-slate-600 px-3 py-2 text-sm font-semibold last:border-r-0 dark:border-slate-500"
    >
      {children}
    </th>
  ),
  td: ({ children, style }) => (
    <td
      style={style}
      className="border-t border-r border-slate-300 px-3 py-2 align-top text-foreground last:border-r-0 dark:border-slate-600"
    >
      {children}
    </td>
  ),
  tr: ({ children }) => <tr className="even:bg-muted/60">{children}</tr>,
}

interface MarkdownContentProps {
  content: string
  /** With citations, [n] markers render as badges that jump to `${citationAnchorPrefix}-n`. */
  citations?: ChatCitation[]
  citationAnchorPrefix?: string
}

export function MarkdownContent({ content, citations, citationAnchorPrefix = "cite" }: MarkdownContentProps) {
  const numbered = useMemo(() => (citations ?? []).filter((c) => c.number != null), [citations])

  const markdownComponents = useMemo<Components>(() => {
    if (numbered.length === 0) return components
    const DefaultLink = components.a as (props: { children?: ReactNode; href?: string }) => ReactNode
    return {
      ...components,
      a: ({ children, href }) => {
        if (!href?.startsWith(CITE_HREF_PREFIX)) return DefaultLink({ children, href })
        const number = Number(href.slice(CITE_HREF_PREFIX.length))
        const citation = numbered.find((c) => c.number === number)
        const anchorId = `${citationAnchorPrefix}-${number}`
        return (
          <a
            href={`#${anchorId}`}
            title={citation ? citationLocation(citation) : undefined}
            onClick={(e) => {
              e.preventDefault()
              document.getElementById(anchorId)?.scrollIntoView({ behavior: "smooth", block: "nearest" })
            }}
            className="mx-0.5 inline-flex h-[1.125rem] min-w-[1.125rem] items-center justify-center rounded bg-primary/15 px-1 align-super font-mono text-[0.6875rem] font-medium text-primary no-underline hover:bg-primary/25"
          >
            {number}
          </a>
        )
      },
    }
  }, [numbered, citationAnchorPrefix])

  const text = normalizeMathDelimiters(content)

  return (
    <div className="space-y-2 text-sm text-foreground [&>*:first-child]:mt-0 [&_.katex]:text-foreground">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={markdownComponents}
      >
        {numbered.length > 0 ? linkCitationMarkers(text, numbered) : text}
      </ReactMarkdown>
    </div>
  )
}
