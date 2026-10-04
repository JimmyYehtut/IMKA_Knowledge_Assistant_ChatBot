import { useCallback, useEffect, useState } from "react"
import { AlertCircle, CheckCircle2, FileText, Loader2, Plus, Trash2, UploadCloud, X } from "lucide-react"
import {
  ApiError,
  deleteDocument,
  getHealth,
  listDocuments,
  uploadDocuments,
  type DocumentRecord,
  type HealthResponse,
  type IngestPipeline,
} from "@/lib/api"
import { cn } from "@/lib/utils"

// Keep in sync with SUPPORTED_EXTENSIONS in backend/ingest/load_documents.py
const ACCEPTED_EXTENSIONS = [
  ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv",
  ".html", ".htm", ".txt", ".md", ".json", ".xml", ".epub", ".msg",
]
const DOCLING_ACCEPTED_EXTENSIONS = [".pdf"]

const PIPELINE_OPTIONS: { value: IngestPipeline; label: string; hint: string }[] = [
  { value: "standard", label: "Standard", hint: "MarkItDown → Markdown, then Markdown-aware splitter. PDF, Office, HTML, CSV, text." },
  { value: "docling", label: "Docling (layout-aware)", hint: "Header/table-aware parsing via Docling. PDF only." },
]

interface MetadataRow {
  id: number
  key: string
  value: string
}

let nextRowId = 1

function formatBytes(bytes: number) {
  return `${(bytes / 1024).toFixed(1)} KB`
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  })
}

export function DocumentsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [loadingDocuments, setLoadingDocuments] = useState(true)

  const [files, setFiles] = useState<File[]>([])
  const [version, setVersion] = useState("1.0")
  const [pipeline, setPipeline] = useState<IngestPipeline>("standard")
  const [metadataRows, setMetadataRows] = useState<MetadataRow[]>([{ id: nextRowId++, key: "", value: "" }])

  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<{ type: "success" | "error"; message: string } | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const refreshDocuments = useCallback(() => {
    setLoadingDocuments(true)
    listDocuments()
      .then(setDocuments)
      .catch(() => setDocuments([]))
      .finally(() => setLoadingDocuments(false))
  }, [])

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
    refreshDocuments()
  }, [refreshDocuments])

  function addFiles(fileList: FileList | null) {
    if (!fileList) return
    setFiles((prev) => [...prev, ...Array.from(fileList)])
  }

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  function addMetadataRow() {
    setMetadataRows((prev) => [...prev, { id: nextRowId++, key: "", value: "" }])
  }

  function updateMetadataRow(id: number, field: "key" | "value", text: string) {
    setMetadataRows((prev) => prev.map((row) => (row.id === id ? { ...row, [field]: text } : row)))
  }

  function removeMetadataRow(id: number) {
    setMetadataRows((prev) => prev.filter((row) => row.id !== id))
  }

  async function handleSubmit() {
    if (files.length === 0) return
    setSubmitting(true)
    setResult(null)
    try {
      const metadata: Record<string, string> = {}
      for (const row of metadataRows) {
        if (row.key.trim()) metadata[row.key.trim()] = row.value
      }

      const response = await uploadDocuments({ files, version, metadata, pipeline })
      setResult({
        type: "success",
        message: `Ingested ${response.total_chunks} chunk${response.total_chunks === 1 ? "" : "s"} from ${response.documents.length} document${response.documents.length === 1 ? "" : "s"}.`,
      })
      setFiles([])
      refreshDocuments()
    } catch (e) {
      const message = e instanceof ApiError ? e.message : "Failed to reach the ingestion API."
      setResult({ type: "error", message })
    } finally {
      setSubmitting(false)
    }
  }

  async function handleDelete(documentId: string) {
    setDeletingId(documentId)
    try {
      await deleteDocument(documentId)
      setDocuments((prev) => prev.filter((d) => d.document_id !== documentId))
    } catch {
      // leave the row in place; user can retry
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-1 gap-4 overflow-y-auto p-4 lg:grid-cols-[420px_1fr] lg:overflow-hidden">
      <div className="flex min-h-0 flex-col gap-4 lg:overflow-y-auto lg:pr-1">
        <div className="glass rounded-xl border border-border p-4">
          <h2 className="font-display text-sm font-semibold tracking-tight text-foreground">Data Ingestion Pipeline</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Upload files, tag them with any metadata you like, then start the ingest pipeline.
          </p>

          <div className="mt-3 flex items-center gap-3 text-xs">
            <span className="flex items-center gap-1.5">
              <span className={cn("size-1.5 rounded-full", health?.openai_configured ? "bg-emerald-500" : "bg-red-500")} />
              OpenAI
            </span>
            <span className="flex items-center gap-1.5">
              <span className={cn("size-1.5 rounded-full", health?.qdrant_reachable ? "bg-emerald-500" : "bg-red-500")} />
              Qdrant
            </span>
          </div>

          <div className="mt-4">
            <label className="text-xs font-medium text-foreground">Ingestion Pipeline</label>
            <select
              value={pipeline}
              onChange={(e) => setPipeline(e.target.value as IngestPipeline)}
              className="mt-1 h-9 w-full rounded-lg border border-border bg-background px-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            >
              {PIPELINE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <p className="mt-1 text-xs text-muted-foreground">
              {PIPELINE_OPTIONS.find((opt) => opt.value === pipeline)?.hint}
            </p>
          </div>

          <label
            className="mt-3 flex cursor-pointer flex-col items-center gap-2 rounded-lg border border-dashed border-border px-4 py-6 text-center transition-all hover:border-primary/50 hover:bg-accent"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              addFiles(e.dataTransfer.files)
            }}
          >
            <UploadCloud className="size-6 text-muted-foreground" />
            <span className="text-sm font-medium text-foreground">Choose or drop files</span>
            <span className="text-xs text-muted-foreground">
              {pipeline === "docling" ? "PDF only" : "PDF, DOCX, PPTX, XLSX, HTML, CSV, TXT, MD, and more"}
            </span>
            <input
              type="file"
              multiple
              accept={(pipeline === "docling" ? DOCLING_ACCEPTED_EXTENSIONS : ACCEPTED_EXTENSIONS).join(",")}
              className="hidden"
              onChange={(e) => addFiles(e.target.files)}
            />
          </label>

          {files.length > 0 && (
            <ul className="mt-3 space-y-1.5">
              {files.map((file, i) => (
                <li key={`${file.name}-${i}`} className="flex items-center gap-2 rounded-lg bg-muted px-2.5 py-1.5 text-xs">
                  <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                  <span className="min-w-0 flex-1 truncate text-foreground">{file.name}</span>
                  <span className="shrink-0 text-muted-foreground">{formatBytes(file.size)}</span>
                  <button type="button" onClick={() => removeFile(i)} className="shrink-0 text-muted-foreground hover:text-foreground" aria-label="Remove file">
                    <X className="size-3.5" />
                  </button>
                </li>
              ))}
            </ul>
          )}

          <div className="mt-4 space-y-3">
            <div>
              <label className="text-xs font-medium text-foreground">Version</label>
              <input
                type="text"
                value={version}
                onChange={(e) => setVersion(e.target.value)}
                className="mt-1 h-9 w-full rounded-lg border border-border bg-background px-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>

            <div>
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-foreground">Custom Metadata</label>
                <button
                  type="button"
                  onClick={addMetadataRow}
                  className="flex items-center gap-1 text-xs font-medium text-blue-600 hover:underline"
                >
                  <Plus className="size-3.5" />
                  Add field
                </button>
              </div>

              <div className="mt-1.5 space-y-1.5">
                {metadataRows.map((row) => (
                  <div key={row.id} className="flex items-center gap-1.5">
                    <input
                      type="text"
                      placeholder="Key"
                      value={row.key}
                      onChange={(e) => updateMetadataRow(row.id, "key", e.target.value)}
                      className="h-8 w-2/5 rounded-lg border border-border bg-background px-2 text-xs text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                    />
                    <input
                      type="text"
                      placeholder="Value"
                      value={row.value}
                      onChange={(e) => updateMetadataRow(row.id, "value", e.target.value)}
                      className="h-8 flex-1 rounded-lg border border-border bg-background px-2 text-xs text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                    />
                    <button
                      type="button"
                      onClick={() => removeMetadataRow(row.id)}
                      className="shrink-0 rounded-md p-1 text-muted-foreground hover:bg-accent"
                      aria-label="Remove field"
                    >
                      <X className="size-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <button
            type="button"
            disabled={files.length === 0 || submitting}
            onClick={handleSubmit}
            className="brand-gradient mt-4 flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-white transition-transform hover:brightness-110 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {submitting && <Loader2 className="size-4 animate-spin" />}
            {submitting ? "Running pipeline..." : "Start Ingest Pipeline"}
          </button>

          {result && (
            <div
              className={cn(
                "mt-3 flex items-start gap-2 rounded-lg px-3 py-2 text-xs",
                result.type === "success"
                  ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                  : "bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300",
              )}
            >
              {result.type === "success" ? (
                <CheckCircle2 className="mt-0.5 size-3.5 shrink-0" />
              ) : (
                <AlertCircle className="mt-0.5 size-3.5 shrink-0" />
              )}
              {result.message}
            </div>
          )}
        </div>
      </div>

      <div className="glass min-h-0 rounded-xl border border-border p-4 lg:overflow-y-auto">
        <h2 className="font-display text-sm font-semibold tracking-tight text-foreground">
          Indexed Documents ({documents.length})
        </h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Documents chunked, embedded, and stored in the vector index.
        </p>

        <div className="mt-4 space-y-2">
          {loadingDocuments ? (
            <div className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              Loading documents...
            </div>
          ) : documents.length === 0 ? (
            <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-border py-10 text-center">
              <FileText className="size-6 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">No documents indexed yet.</p>
            </div>
          ) : (
            documents.map((doc) => (
              <div key={doc.document_id} className="flex items-start gap-3 rounded-lg border border-border p-3">
                <FileText className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-foreground">{doc.document_name}</p>
                  <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                    <span className="rounded-full bg-muted px-2 py-0.5">v{doc.version}</span>
                    <span className="rounded-full bg-muted px-2 py-0.5">{doc.chunk_count} chunks</span>
                    {Object.entries(doc.metadata).map(([key, value]) => (
                      <span key={key} className="rounded-full bg-muted px-2 py-0.5">
                        {key}: {value}
                      </span>
                    ))}
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">{formatDate(doc.uploaded_at)}</p>
                </div>
                <button
                  type="button"
                  onClick={() => handleDelete(doc.document_id)}
                  disabled={deletingId === doc.document_id}
                  className="shrink-0 rounded-md p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive disabled:opacity-50"
                  aria-label={`Delete ${doc.document_name}`}
                >
                  {deletingId === doc.document_id ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Trash2 className="size-4" />
                  )}
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
