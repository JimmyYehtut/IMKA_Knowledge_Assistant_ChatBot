import { Microscope, Settings2, Wrench, type LucideIcon } from "lucide-react"

// Persona ids must match PERSONAS in backend/rag/pipeline.py, which holds the
// prompt instructions each one applies to the answer.
export type PersonaId = "technician" | "engineer" | "specialist"

export interface Persona {
  id: PersonaId
  name: string
  detailLevel: string
  description: string
  icon: LucideIcon
}

export const PERSONAS: Persona[] = [
  {
    id: "technician",
    name: "Field Technician",
    detailLevel: "Brief",
    description: "Short, action-first steps with key values and safety warnings. No background theory.",
    icon: Wrench,
  },
  {
    id: "engineer",
    name: "Maintenance Engineer",
    detailLevel: "Standard",
    description: "A direct answer, then the likely causes, the procedure, and the relevant limits.",
    icon: Settings2,
  },
  {
    id: "specialist",
    name: "Reliability Specialist",
    detailLevel: "Detailed",
    description: "Every relevant cause, condition, and exception, organised with headings and tables.",
    icon: Microscope,
  },
]

export const DEFAULT_PERSONA: PersonaId = "engineer"

export function getPersona(id: string | null | undefined): Persona {
  return PERSONAS.find((p) => p.id === id) ?? PERSONAS.find((p) => p.id === DEFAULT_PERSONA)!
}

export const STARTER_QUESTIONS: string[] = [
  "What should I check when the machine arrives on site?",
  "What can cause a bearing to run too hot?",
  "Why is oil leaking from a sleeve bearing?",
  "Why are the brushes sparking, and how do I fix it?",
  "How do I check a Pt-100 temperature detector?",
  "What should I do when a protection trip stops the machine?",
]
