import { useState } from "react"
import { Dialog } from "@base-ui/react/dialog"
import { ExternalLink, ImageOff, Maximize2, XIcon } from "lucide-react"
import { diagramUrl } from "@/lib/diagrams"
import { cn } from "@/lib/utils"

interface DiagramImageProps {
  /** A diagrams/... path (callers check it with isDiagramPath). */
  path: string
  /** Shown as alt text and in the enlarged view; the visible caption is the caller's. */
  caption: string
  className?: string
}

/**
 * One knowledge-base diagram: a thumbnail that opens an enlarged view in place
 * (labels on wiring and dimension drawings are unreadable at chat width). If
 * the file is missing — a path the model invented, or a document re-ingested
 * since — the caption is shown instead of a broken image.
 */
export function DiagramImage({ path, caption, className }: DiagramImageProps) {
  const [failed, setFailed] = useState(false)
  const url = diagramUrl(path)

  if (failed) {
    return (
      <span
        role="note"
        className="inline-flex items-center gap-1.5 rounded-md border border-dashed border-border px-2 py-1 text-xs text-muted-foreground italic"
      >
        <ImageOff className="size-3.5 shrink-0" aria-hidden />
        {caption || "Diagram unavailable"}
      </span>
    )
  }

  return (
    <Dialog.Root>
      <Dialog.Trigger
        className={cn(
          "group/diagram relative block cursor-zoom-in rounded-lg border border-border bg-white p-2 text-left focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
          className,
        )}
        aria-label={`Enlarge diagram: ${caption}`}
      >
        <img
          src={url}
          alt={caption}
          loading="lazy"
          onError={() => setFailed(true)}
          className="max-h-80 max-w-full object-contain"
        />
        <Maximize2
          aria-hidden
          className="absolute top-2 right-2 size-6 rounded bg-black/55 p-1 text-white opacity-0 transition-opacity group-hover/diagram:opacity-100 group-focus-visible/diagram:opacity-100"
        />
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-50 bg-black/70 transition-opacity duration-150 data-ending-style:opacity-0 data-starting-style:opacity-0" />
        <Dialog.Popup className="fixed inset-4 z-50 flex flex-col gap-3 rounded-xl border border-border bg-popover p-4 text-popover-foreground shadow-xl outline-none sm:inset-10">
          <div className="flex items-start justify-between gap-3">
            <Dialog.Title className="text-sm font-medium">{caption || "Diagram"}</Dialog.Title>
            <div className="flex shrink-0 items-center gap-1">
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
              >
                <ExternalLink className="size-3.5" aria-hidden />
                Open original
              </a>
              <Dialog.Close
                aria-label="Close"
                className="rounded-md p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground"
              >
                <XIcon className="size-4" />
              </Dialog.Close>
            </div>
          </div>
          <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto rounded-lg bg-white p-2">
            <img src={url} alt={caption} className="max-h-full max-w-full object-contain" />
          </div>
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
