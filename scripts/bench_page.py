#!/usr/bin/env python3
"""Benchmark one OCR page against a multimodal llama-server (or Ollama).

Measures time-to-first-token, total time, output tokens and tok/s for the
canonical GLM-OCR trigger ("Text Recognition:") on one page image — directly
comparable with the CPU baseline recorded for this project:
  encode 35-57 s/page, generation 2-16 tok/s, ~65-470 s/page end-to-end.

Usage:
  python3 scripts/bench_page.py --page PATH [--backend llama|ollama]
      [--llama-url http://127.0.0.1:8080] [--runs N]
  (--page may be omitted if the BENCH_PAGE environment variable is set)

Start the server first, e.g.:
  build/bin/llama-server -m models/glm-ocr-q8_0.gguf -mm models/glm-ocr-mmproj-f16.gguf \
      -c 16384 --port 8080
(CUDA/Vulkan builds: same command — compare to the CPU baseline this script
records on first run; results are appended to results/bench.jsonl)
"""
import argparse
import base64
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PAGE = os.environ.get("BENCH_PAGE", "")
RESULTS = Path(__file__).resolve().parent.parent / "results" / "bench.jsonl"


def chat(url, model, image_b64, is_ollama, timeout=1800):
    if is_ollama:
        payload = {"model": model, "stream": True,
                   "messages": [{"role": "user", "content": "Text Recognition:",
                                 "images": [image_b64]}],
                   "options": {"num_ctx": 16384, "num_predict": 4096}}
    else:
        payload = {"model": model, "stream": True, "max_tokens": 4096,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": "Text Recognition:"},
                       {"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + image_b64}}]}]}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions" if not is_ollama
                                 else url.rstrip("/") + "/api/chat",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer ollama"})
    t0 = time.time()
    first = None
    out = ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode() if isinstance(raw, bytes) else raw
            line = line.strip()
            if not line.startswith("data:"):
                if is_ollama:
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    c = o.get("message", {}).get("content", "")
                    if c and first is None:
                        first = time.time() - t0
                    out += c
                    if o.get("done"):
                        return {"ttft_s": first, "total_s": time.time() - t0,
                                "out_tokens": o.get("eval_count", 0),
                                "tok_s": (o.get("eval_count", 0) / (o["eval_duration"] / 1e9)
                                          if o.get("eval_duration") else None),
                                "prompt_tokens": o.get("prompt_eval_count", 0),
                                "prompt_s": o.get("prompt_eval_duration", 0) / 1e9}
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            o = json.loads(body)
            delta = (o.get("choices") or [{}])[0].get("delta", {})
            c = delta.get("content") or ""
            if c and first is None:
                first = time.time() - t0
            out += c
    return {"ttft_s": first, "total_s": time.time() - t0, "out_tokens": None,
            "tok_s": None, "out_chars": len(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llama", "ollama"], default="llama")
    ap.add_argument("--page", default=DEFAULT_PAGE)

    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--label", default=None, help="e.g. 'cpu-q8_0', 'mx150-q8_0-split'")
    args = ap.parse_args()
    if not args.page:
        ap.error("--page is required (or set BENCH_PAGE)")

    image_b64 = base64.b64encode(Path(args.page).read_bytes()).decode()
    model = args.model or ("glm-ocr:q8_0" if args.backend == "ollama" else "glm-ocr")
    url = args.llama_url if args.backend == "llama" else args.ollama_url

    for run in range(1, args.runs + 1):
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "backend": args.backend,
               "label": args.label or args.backend, "page": Path(args.page).name,
               "model": model, "run": run,
               **chat(url, model, image_b64, args.backend == "ollama")}
        print(json.dumps(rec), flush=True)
        RESULTS.parent.mkdir(exist_ok=True)
        with RESULTS.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
