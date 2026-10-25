#!/usr/bin/env bash
# Ingests the test-A … test-G documents, one PDF at a time, with a log per PDF.
#
#   bash test/code/ingest.sh <stage> [pdf ...]      (from the repository root, Git Bash)
#
# stage: C1           standard pipeline (MarkItDown + 1,000-character splitter)
#        C2           Docling chunking
#        C3-mini      + figure descriptions by gpt-4o-mini   (test-C)
#        C3-luna      + figure descriptions by gpt-5.6-luna  (test-F, test-G)
#        C3-qwen      + figure descriptions and embeddings by Qwen on RunPod (deferred Qwen run),
#                     into the ragindex_qwen collection
# pdf:   file names under pdfs/; default all five, EMSA first (the one most likely to fail).
#
# Logs: test/output/ingest/<stage>-<pdf>.log; one line per PDF in test/output/ingest/status.log.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="${1:?stage: C1 C2 C3-mini C3-luna C3-qwen}"
shift
LOGS="$ROOT/test/output/ingest"
mkdir -p "$LOGS"

if [ $# -gt 0 ]; then
  PDFS=("$@")
else
  PDFS=("EMSA Diesel Generator manual-en.pdf"
        "Siemens_Manual_Vertical_Induction_Motors_EN.pdf"
        "ABB Manual for Induction Motors and Generators_EN.pdf"
        "Kohler Generator Manual.pdf"
        "Nidec Motor Manual.pdf")
fi

export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export EMBEDDING_PROVIDER=openai QDRANT_COLLECTION_NAME=ragindex
MODULE=ingest.embed_and_store_docling
case "$STAGE" in
  C1) MODULE=ingest.embed_and_store ;;
  C2) export CAPTION_FIGURES=0 ;;
  C3-mini) export CAPTION_FIGURES=1 DOCLING_CAPTION_MODEL=gpt-4o-mini ;;
  C3-luna) export CAPTION_FIGURES=1 DOCLING_CAPTION_MODEL=gpt-5.6-luna ;;
  C3-qwen) export CAPTION_FIGURES=1 DOCLING_CAPTION_MODEL=qwen \
                  EMBEDDING_PROVIDER=qwen QDRANT_COLLECTION_NAME=ragindex_qwen ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac

cd "$ROOT/backend"
for pdf in "${PDFS[@]}"; do
  log="$LOGS/$STAGE-${pdf%%.pdf}.log"
  start=$(date +%s)
  echo "$(date '+%F %T') $STAGE start  $pdf" >> "$LOGS/status.log"
  .venv/Scripts/python.exe -m "$MODULE" "$ROOT/pdfs/$pdf" > "$log" 2>&1
  code=$?
  ok=$(grep -c "Total chunks indexed" "$log")
  chunks=$(grep -o "Total chunks indexed: [0-9]*" "$log" | grep -o "[0-9]*$")
  status=$([ "$code" = 0 ] && [ "$ok" -gt 0 ] && echo "ok    " || echo "FAILED")
  echo "$(date '+%F %T') $STAGE $status $pdf ($(( $(date +%s) - start ))s, ${chunks:-0} chunks)" >> "$LOGS/status.log"
done
