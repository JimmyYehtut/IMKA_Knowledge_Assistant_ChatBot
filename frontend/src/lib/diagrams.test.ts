import { describe, expect, it } from "vitest"
import { citedDiagramsNotEmbedded, embeddedDiagramPaths, hideUnfinishedImage, isDiagramPath } from "@/lib/diagrams"
import type { ChatCitation } from "@/lib/types"

const DOC = "cdbdf9f5-8397-485f-a095-563fa702be5f"
const fig = (n: number, k = 1) => `diagrams/${DOC}/fig${n}_${k}.png`

function citation(over: Partial<ChatCitation>): ChatCitation {
  return { document_id: DOC, document_name: "ABB Manual", chunk_index: 0, bm25_score: null, page: 20, image_path: null, ...over }
}

describe("isDiagramPath", () => {
  it("accepts only files under diagrams/", () => {
    expect(isDiagramPath(fig(9))).toBe(true)
    expect(isDiagramPath("https://evil.example/x.png")).toBe(false)
    expect(isDiagramPath("diagrams/../.env")).toBe(false)
    expect(isDiagramPath("diagrams\\x.png")).toBe(false)
    expect(isDiagramPath(null)).toBe(false)
  })
})

describe("embeddedDiagramPaths", () => {
  it("finds diagram images and ignores other images and links", () => {
    const answer = `Step 1 [1]\n![Figure 9 Lifting](${fig(9)})\n![x](https://example.com/a.png)\n[link](${fig(3)})`
    expect([...embeddedDiagramPaths(answer)]).toEqual([fig(9)])
  })
})

describe("citedDiagramsNotEmbedded", () => {
  it("returns cited figures the answer did not draw, captioned by the figure caption", () => {
    const answer = `See below [1][2]\n\n![Figure 9 Lifting of unpacked machines](${fig(9)})`
    const citations = [
      citation({ number: 1, image_path: fig(9), chunk_type: "figure", figure_caption: "Figure 9 Lifting of unpacked machines" }),
      citation({ number: 2, image_path: fig(38), chunk_type: "figure", figure_caption: "Figure 38 Correlation" }),
      citation({ number: 3, image_path: fig(38) }),
      citation({ number: 4, image_path: null }),
    ]
    expect(citedDiagramsNotEmbedded(answer, citations)).toEqual([{ path: fig(38), caption: "Figure 38 Correlation" }])
  })

  it("falls back to the document location when the citation has no caption", () => {
    const [d] = citedDiagramsNotEmbedded("no images", [citation({ image_path: fig(5), section_path: "1.5.2 Rating plate" })])
    expect(d.caption).toBe("ABB Manual · 1.5.2 Rating plate")
  })

  it("never offers a non-diagram path", () => {
    expect(citedDiagramsNotEmbedded("", [citation({ image_path: "../secret.png" })])).toEqual([])
  })
})

describe("hideUnfinishedImage", () => {
  it("hides an image whose Markdown is still streaming in", () => {
    expect(hideUnfinishedImage("Lift it [1]\n![Figure 9 Lifting of unpa")).toBe("Lift it [1]\n")
    expect(hideUnfinishedImage(`Lift it [1]\n![Figure 9](diagrams/${DOC}/fi`)).toBe("Lift it [1]\n")
    expect(hideUnfinishedImage(`Done ![a](${fig(1)}) then ![b](diag`)).toBe(`Done ![a](${fig(1)}) then `)
  })

  it("leaves complete answers unchanged", () => {
    const answer = `Lift it [1]\n![Figure 9](${fig(9)})\n\nThen stop.`
    expect(hideUnfinishedImage(answer)).toBe(answer)
    expect(hideUnfinishedImage(`![Figure 9](${fig(9)})`)).toBe(`![Figure 9](${fig(9)})`)
  })
})
