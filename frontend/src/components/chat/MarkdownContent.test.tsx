import { cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import { MarkdownContent } from "@/components/chat/MarkdownContent"
import { API_URL } from "@/lib/api"
import type { ChatCitation } from "@/lib/types"

afterEach(cleanup)

const DOC = "cdbdf9f5-8397-485f-a095-563fa702be5f"
const fig = (n: number, k = 1) => `diagrams/${DOC}/fig${n}_${k}.png`
const citations: ChatCitation[] = [
  { number: 1, document_id: DOC, document_name: "ABB Manual", chunk_index: 0, bm25_score: null, page: 25, image_path: null },
]

describe("MarkdownContent diagrams", () => {
  it("draws a diagram placed after a list item, served from the API", () => {
    // Shape returned by the live pipeline for "How do I lift an unpacked machine?"
    const answer = [
      "1. Check the weight. [1]",
      "2. Lift the machine by crane from the lifting eyes. [1]  ",
      `![Figure 9 Lifting of unpacked machines](${fig(9)})`,
      "3. Do not use a forklift. [1]",
    ].join("\n")
    render(<MarkdownContent content={answer} citations={citations} />)

    const img = screen.getByRole("img", { name: "Figure 9 Lifting of unpacked machines" })
    expect(img.getAttribute("src")).toBe(`${API_URL}/data/${fig(9)}`)
    expect(img.closest("li")?.textContent).toContain("Lift the machine by crane")
    // The [1] markers still become citation badges around the image.
    expect(screen.getAllByRole("link", { name: "1" })).toHaveLength(3)
  })

  it("groups the panels of one figure under a single caption", () => {
    // Shape returned for "Which diagrams show how to lift the machine?"
    const caption = "Figure 9 Lifting of unpacked machines"
    const answer = [
      "Figures [1]",
      "",
      `  ![${caption}](${fig(9, 1)})  `,
      `  ![${caption}](${fig(9, 2)})  `,
      `  ![${caption}](${fig(9, 3)})`,
      "",
      `  ![Figure 6 Sling here](${fig(6)})`,
    ].join("\n")
    render(<MarkdownContent content={answer} citations={citations} />)

    const figures = screen.getAllByTestId("diagram-figure")
    expect(figures).toHaveLength(2)
    expect(within(figures[0]).getAllByRole("img")).toHaveLength(3)
    expect(figures[0].querySelector("figcaption")?.textContent).toBe(caption)
    expect(within(figures[1]).getByRole("img").getAttribute("src")).toContain("fig6_1.png")
    // The caption is shown once, not once per panel.
    expect(screen.getAllByText(caption)).toHaveLength(1)
  })

  it("does not load images from outside the diagrams folder", () => {
    render(<MarkdownContent content={"![tracking pixel](https://evil.example/p.png)\n\n![sneaky](diagrams/../../.env)"} />)
    expect(screen.queryByRole("img")).toBeNull()
    expect(screen.getByText("tracking pixel")).toBeTruthy()
  })

  it("shows the caption instead of a broken image when the file is missing", () => {
    render(<MarkdownContent content={`![Figure 99 Invented](${fig(99)})`} />)
    fireEvent.error(screen.getByRole("img"))
    expect(screen.queryByRole("img")).toBeNull()
    expect(screen.getByRole("note").textContent).toContain("Figure 99 Invented")
  })

  it("opens an enlarged view on click", async () => {
    render(<MarkdownContent content={`![Figure 38 Correlation](${fig(38)})`} />)
    fireEvent.click(screen.getByRole("button", { name: "Enlarge diagram: Figure 38 Correlation" }))
    const dialog = await screen.findByRole("dialog")
    const url = `${API_URL}/data/${fig(38)}`
    expect(within(dialog).getByRole("img").getAttribute("src")).toBe(url)
    expect(within(dialog).getByRole("link", { name: /open original/i }).getAttribute("href")).toBe(url)
  })

  it("hides a diagram that is still streaming in", () => {
    render(<MarkdownContent content={"Lift it.\n\n![Figure 9 Lifting of unp"} />)
    expect(screen.queryByText(/!\[Figure 9/)).toBeNull()
    expect(screen.getByText("Lift it.")).toBeTruthy()
  })
})
