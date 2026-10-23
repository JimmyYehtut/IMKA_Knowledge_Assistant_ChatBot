End-to-end RAG workflow

Entry point: POST /v1/chat/completions (backend/api/openai_compat.py) — OpenAI-wire-compatible, used by both the frontend and any OpenAI-compatible client. Auth via JWT (get_current_user), conversation resolved/created in Postgres, then either streamed (SSE) or non-streamed depending on stream.

User message
   │
   ▼
1. Intent classification (rag/intent.py)
   │  greeting / capability → short LLM reply, STOP (no retrieval)
   ▼  knowledge
2. Query rewrite (rag/pipeline.py stage 1)
   ▼
3. Qdrant similarity search, top 15 (stage 2)
   ▼
4. BM25 rerank → dedupe → top 5 (stage 3)
   ▼
5. Context assembly, [Doc, p. N] labels (stage 4)
   ▼
6. Answer generation — streamed LLM call (stage 5)
   ▼
7. Citations built from same reranked chunks (stage 6)
   ▼
Persist user+assistant messages to Postgres, return answer + citations

Stage-by-stage with actual prompts

0. Intent classification (rag/intent.py) — runs on every message, before anything else. Structured output (Intent Pydantic model: intent, reply).

▎ System prompt: "You are the front-door intent classifier for a maintenance knowledge assistant chatbot. Classify the user's message as exactly one of: 'greeting'... 'capability'... 'knowledge'... If the intent is 'greeting' or 'capability', also write reply: a short, friendly, 1-3 sentence answer..."

If greeting/capability, the LLM's own reply is returned immediately — no retrieval happens. Only knowledge proceeds.

1. Query rewrite (_stage1_process_query) — cleans/disambiguates for embedding search:

▎ "You are a query optimizer for a knowledge base. Rewrite the user query to be clear, specific, and optimised for semantic search. Remove filler words, fix typos, and make implicit intent explicit. Return ONLY the rewritten query — no explanation."

Model: gpt-4o-mini (via OPENAI_CHAT_MODEL), temperature=0.

2. Retrieve — vectorstore.similarity_search(rewritten_query, k=15) against Qdrant.

3. BM25 rerank — dedupe candidates by first-200-chars fingerprint, tokenize (.lower().split()), score with BM25Okapi, keep top 5 (RERANK_TOP_N). This is a pure keyword-overlap pass layered on top of the semantic retrieval — no LLM call.

4. Context assembly — each chunk becomes:
[Document Name, p. 12]
<chunk text>
joined with \n\n---\n\n. Page number resolved by _page_number(): prefers Docling's 1-indexed page_no, falls back to PyPDFLoader's 0-indexed page (+1), else None (unpaginated sources).

5. Answer generation (build_answer_chain) — the actual user-facing generation, streamed token-by-token via .astream():

▎ System: "You are an enterprise knowledge assistant. Answer ONLY using the provided context — this is your sole grounding source. Do not hallucinate or add information not present in the context. If the context does not fully answer the question, say clearly what is missing rather than filling the gap with outside knowledge."

▎ Human template:
▎ Context:
▎ {context}
▎
▎ Question:
▎ {question}
▎
▎ Instructions:
▎ - Answer based solely on the context above — do not use outside knowledge.
▎ - Each context passage is labeled with its source document and, when known, its page
▎   number, e.g. "[Document Name, p. 12]". When you use a fact from a passage, mention
▎   the page number in natural language if it helps the reader locate it in the source
▎   (e.g. "...as shown on page 12"). Only state a page number that appears in the
▎   context labels — never guess or invent one. Do not reproduce the bracket label
▎   format itself in your answer.
▎ - Format the answer in Markdown when it improves clarity: use headings, bullet or
▎   numbered lists for steps/sequences, and a Markdown table when presenting tabular
▎   or comparative data...
▎ - Write any mathematical formula in LaTeX using $...$ for inline math and $$...$$
▎   for a standalone/display formula (not \( \) or \[ \]).
▎ - If the context does not contain enough information, say so clearly.

Note {question} here is the rewritten query from stage 1, not the raw user input.

6. Citations — built from the same 5 reranked chunks, deduped by document_id: {document_id, document_name, chunk_index, bm25_score, page}.

Wiring notes

- If retrieval returns nothing (context == ""), the pipeline short-circuits with a hardcoded fallback string — no LLM call for stage 5.
- Every chain invocation carries config={"metadata": {"conversation_id", "user_id"}} for LangSmith tracing (enabled via LANGCHAIN_TRACING_V2), so a run can be traced end-to-end per conversation.
- Non-streaming and streaming paths in openai_compat.py duplicate the same intent→pipeline→answer logic (_generate_answer vs _stream_sse) rather than sharing one function, because streaming needs to yield chunks incrementally.
- Both paths persist (user message, assistant message + citations JSON) to Postgres via _persist_exchange regardless of stream mode.

There are 3 total LLM calls in the knowledge path (intent classify → query rewrite → answer generation), or just 1 for greeting/capability.