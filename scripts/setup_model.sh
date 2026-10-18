#!/usr/bin/env bash
# Pull the pinned GLM-OCR model into local Ollama and verify its digest.
# Override with: MODEL=glm-ocr:q8_0 ./scripts/setup_model.sh
set -euo pipefail

MODEL="${MODEL:-glm-ocr:q8_0}"
# sha256 of the q8_0 LLM blob this project was benchmarked with
EXPECTED="6e5935a916fc8d6878a67c60c2b99648f54e10b4a4af4cbbfb2c308953000e5f"

echo "pulling $MODEL (about 1.8 GB)..."
ollama pull "$MODEL"

DIGEST_FILE=$(ollama show "$MODEL" --modelfile 2>/dev/null | awk '/^FROM/{print $2; exit}')
if [[ -n "$DIGEST_FILE" ]]; then
  ACTUAL=$(sha256sum "$DIGEST_FILE" | awk '{print $1}')
  if [[ "$ACTUAL" == "$EXPECTED" ]]; then
    echo "digest verified: $ACTUAL"
  else
    echo "WARNING: digest differs from the benchmarked build"
    echo "  expected: $EXPECTED"
    echo "  actual:   $ACTUAL"
    echo "(benchmarked performance claims apply to the expected build)"
  fi
fi
echo "done. next: scripts/extract_gguf_blobs.py (for llama.cpp) or use ollama directly."
