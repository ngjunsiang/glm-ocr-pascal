#!/usr/bin/env python3
"""Extract GLM-OCR GGUF blobs from the local Ollama store into models/.

Identifies the LLM layer and the CLIP projector (mmproj) layer from the
ollama manifest, and copies them into models/ with readable names:
  models/glm-ocr-q8_0.gguf      (891M LLM, q8_0)
  models/glm-ocr-mmproj-f16.gguf (434M CLIP projector, fp16)
"""
import json
import shutil
import sys
from pathlib import Path

OLLAMA = Path.home() / ".ollama" / "models"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "glm-ocr:q8_0"
OUT = Path(__file__).resolve().parent.parent / "models"
OUT.mkdir(exist_ok=True)

name, tag = MODEL.split(":")
manifest = OLLAMA / "manifests" / "registry.ollama.ai" / "library" / name / tag
m = json.loads(manifest.read_text())

copied = {}
for layer in m["layers"]:
    mt = layer["mediaType"]
    if mt.endswith("model"):
        dst = OUT / f"{name}-{tag.replace(':', '_')}.gguf"
        kind = "llm"
    elif "projector" in mt:
        dst = OUT / f"{name}-mmproj-{m['config'].get('image', {}).get('format', 'f16')}.gguf"
        kind = "mmproj"
    else:
        continue
    src = OLLAMA / "blobs" / layer["digest"].replace(":", "-")
    print(f"{kind}: {src.name} ({layer['size'] / 1e9:.2f} GB) -> {dst.name}")
    if not dst.exists():
        shutil.copy(src, dst)
    copied[kind] = dst

if "llm" not in copied:
    sys.exit("no LLM layer found in manifest")
print("done:", ", ".join(str(p) for p in copied.values()))
