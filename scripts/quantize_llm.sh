#!/usr/bin/env bash
# Re-quantize the GLM-OCR LLM blob so the LLM + CLIP encoder fit the MX150's 2 GB together.
# Usage: scripts/quantize_llm.sh [q4_K_M|q4_0|q8_0]   (default q4_K_M)
set -euo pipefail
cd "$(dirname "$0")/.."

QUANT="${1:-q4_K_M}"
SRC="models/glm-ocr-f16.gguf"   # quantize from F16; requantizing q8_0 is disabled by llama.cpp
[[ -f "$SRC" ]] || SRC="models/glm-ocr-q8_0.gguf"
DST="models/glm-ocr-${QUANT}.gguf"

[[ -f "$SRC" ]] || { echo "missing $SRC — run scripts/extract_gguf_blobs.py first"; exit 1; }
[[ -x build/bin/llama-quantize ]] || { echo "missing build/bin/llama-quantize — build first (see README)"; exit 1; }

echo "quantizing $SRC -> $DST ($QUANT)..."
build/bin/llama-quantize $([[ "$SRC" == *q8_0* ]] && echo --allow-requantize) "$SRC" "$DST" "$QUANT"
ls -lh "$DST"
echo "reminder: the fp16 mmproj (~0.9 GB) is NOT requantized by this script;"
echo "q4 LLM (~0.6 GB) + fp16 encoder (~1.1 GB peak) ≈ 1.7 GB — fits 2 GB VRAM."
