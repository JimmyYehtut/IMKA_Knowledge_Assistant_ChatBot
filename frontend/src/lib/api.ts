export const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000"

export interface DocumentRecord {
  document_id: string
  document_name: string
  version: string
  chunk_count: number
  metadata: Record<string, string>
  status: string
  uploaded_at: string
}

export interface IngestResponse {
  total_chunks: number
  documents: DocumentRecord[]
}

export interface HealthResponse {
  status: string
  openai_configured: boolean
  qdrant_reachable: boolean
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new ApiError(res.status, body?.detail ?? res.statusText)
  }
  return res.json() as Promise<T>
}

export function getHealth(): Promise<HealthResponse> {
  return fetch(`${API_URL}/api/health`).then((res) => handle<HealthResponse>(res))
}

export function listDocuments(): Promise<DocumentRecord[]> {
  return fetch(`${API_URL}/api/documents`).then((res) => handle<DocumentRecord[]>(res))
}

export type IngestPipeline = "standard" | "docling"

export interface UploadDocumentsParams {
  files: File[]
  version: string
  metadata: Record<string, string>
  pipeline?: IngestPipeline
}

export function uploadDocuments(params: UploadDocumentsParams): Promise<IngestResponse> {
  const form = new FormData()
  for (const file of params.files) form.append("files", file)
  form.append("version", params.version)
  form.append("metadata", JSON.stringify(params.metadata))
  form.append("pipeline", params.pipeline ?? "standard")

  return fetch(`${API_URL}/api/documents`, { method: "POST", body: form }).then((res) =>
    handle<IngestResponse>(res),
  )
}

export function deleteDocument(documentId: string): Promise<{ deleted: boolean }> {
  return fetch(`${API_URL}/api/documents/${documentId}`, { method: "DELETE" }).then((res) =>
    handle<{ deleted: boolean }>(res),
  )
}

// ── Auth ────────────────────────────────────────────────────────────────────

export interface User {
  id: string
  email: string
  name: string
}

export interface AuthResponse {
  access_token: string
  token_type: string
  user: User
}

export function registerUser(params: { email: string; password: string; name: string }): Promise<User> {
  return fetch(`${API_URL}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  }).then((res) => handle<User>(res))
}

export function loginUser(params: { email: string; password: string }): Promise<AuthResponse> {
  return fetch(`${API_URL}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  }).then((res) => handle<AuthResponse>(res))
}

// ── Chat ────────────────────────────────────────────────────────────────────

export interface Citation {
  document_id: string
  document_name: string
  chunk_index: number
  bm25_score: number | null
  page: number | null
  image_path: string | null
  /** Absent on messages stored before answers carried inline [n] citations. */
  number?: number | null
  section_path?: string | null
  chunk_type?: string | null
}

export interface OpenAIMessage {
  role: "system" | "user" | "assistant"
  content: string
}

interface ChatCompletionChunk {
  choices: { delta: { role?: string; content?: string }; finish_reason: string | null }[]
  conversation_id?: string
  message_id?: string
  topic?: string
  citations?: Citation[]
}

/**
 * Streams an OpenAI-format chat completion from POST /v1/chat/completions
 * (Server-Sent Events). Calls onToken for each content delta as it arrives.
 */
export async function streamChatCompletion(
  token: string,
  params: { messages: OpenAIMessage[]; conversation_id?: string | null; persona?: string },
  onToken: (delta: string) => void,
): Promise<{ conversationId: string; messageId: string; citations: Citation[]; topic: string | null }> {
  const res = await fetch(`${API_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({
      messages: params.messages,
      conversation_id: params.conversation_id,
      persona: params.persona,
      stream: true,
    }),
  })

  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => null)
    throw new ApiError(res.status, body?.detail ?? res.statusText)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let conversationId = ""
  let messageId = ""
  let citations: Citation[] = []
  let topic: string | null = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const events = buffer.split("\n\n")
    buffer = events.pop() ?? ""

    for (const event of events) {
      const line = event.trim()
      if (!line.startsWith("data: ")) continue
      const data = line.slice(6)
      if (data === "[DONE]") continue

      const chunk: ChatCompletionChunk = JSON.parse(data)
      const delta = chunk.choices[0]?.delta
      if (delta?.content) onToken(delta.content)
      if (chunk.conversation_id) conversationId = chunk.conversation_id
      if (chunk.message_id) messageId = chunk.message_id
      if (chunk.citations) citations = chunk.citations
      if (chunk.topic !== undefined) topic = chunk.topic
    }
  }

  return { conversationId, messageId, citations, topic }
}

export interface FeedbackResult {
  message_id: string
  rating: "up" | "down" | null
  comment: string | null
}

export function submitFeedback(
  token: string,
  messageId: string,
  params: { rating?: "up" | "down" | null; comment?: string | null },
): Promise<FeedbackResult> {
  return fetch(`${API_URL}/api/messages/${messageId}/feedback`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify(params),
  }).then((res) => handle<FeedbackResult>(res))
}

export interface HistoryMessage {
  id: string
  role: string
  content: string
  topic: string | null
  citations: Citation[]
  rating: "up" | "down" | null
  feedback_comment: string | null
}

export interface Conversation {
  id: string
  created_at: string
  persona: string | null
  messages: HistoryMessage[]
}

export interface DocumentTopic {
  title: string
  chapter: string
  document_name: string
  page: number | null
}

/** Topics the ingested documents cover that suit the given answer persona. */
export function getTopics(persona: string): Promise<DocumentTopic[]> {
  return fetch(`${API_URL}/api/topics?persona=${encodeURIComponent(persona)}`).then((res) =>
    handle<DocumentTopic[]>(res),
  )
}

export function getHistory(token: string): Promise<Conversation[]> {
  return fetch(`${API_URL}/api/history`, {
    headers: { Authorization: `Bearer ${token}` },
  }).then((res) => handle<Conversation[]>(res))
}
