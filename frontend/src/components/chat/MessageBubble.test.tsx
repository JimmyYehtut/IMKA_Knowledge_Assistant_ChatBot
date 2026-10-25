import { cleanup, render, screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import { MessageBubble } from "@/components/chat/MessageBubble"
import type { ChatCitation, ChatMessage } from "@/lib/types"

afterEach(cleanup)

const DOC = "cdbdf9f5-8397-485f-a095-563fa702be5f"
const fig = (n: number) => `diagrams/${DOC}/fig${n}_1.png`
const base = { document_id: DOC, document_name: "ABB Manual", chunk_index: 0, bm25_score: null }

function assistant(content: string, citations: ChatCitation[]): ChatMessage {
  return { id: "m1", role: "assistant", author: "AI Assistant", time: "10:00", content, citations }
}

describe("MessageBubble cited diagrams", () => {
  it("shows a cited figure the answer did not draw inline", () => {
    // Shape returned for "Show me the diagram of the sleeve bearing": figures cited, none embedded.
    const message = assistant("The referenced figures show a bearing plate. [2]", [
      {
        ...base,
        number: 2,
        page: 23,
        chunk_type: "figure",
        image_path: fig(5),
        figure_caption: "Figure 5 Bearing plate for sleeve bearings",
      },
    ])
    render(<MessageBubble message={message} />)
    const strip = screen.getByTestId("cited-diagrams")
    expect(within(strip).getByRole("img", { name: "Figure 5 Bearing plate for sleeve bearings" })).toBeTruthy()
  })

  it("does not repeat a figure the answer already shows", () => {
    const message = assistant(`Rating plate [2]\n\n![Figure 3 Rating plate](${fig(3)})`, [
      { ...base, number: 2, page: 20, chunk_type: "figure", image_path: fig(3), figure_caption: "Figure 3 Rating plate" },
    ])
    render(<MessageBubble message={message} />)
    expect(screen.queryByTestId("cited-diagrams")).toBeNull()
    expect(screen.getAllByRole("img")).toHaveLength(1)
  })
})
