# glm-ocr-pascal

Run quantized [GLM-OCR](https://github.com/zai-org/GLM-OCR) on legacy NVIDIA Pascal GPUs (sm_61: GTX 10-series, MX150/2xx — 2 GB VRAM class) through Ollama, instead of CPU-only. The dominant cost in local OCR is the vision encoder; on a 2 GB Pascal card it runs on the GPU while the small 0.5 B decoder stays on the CPU, cutting end-to-end page time by about 4×.

Validated on a GeForce MX150 (GP108, 2 GB) under Arch Linux; everything here should apply to any sm_61 card.

## Measured results

One A4 text page at 150 DPI (2851-token prompt), cold encode, on the validation machine:

| Path | Encode | Generation | Total |
|---|---|---|---|
| Ollama, CPU-only | ~240 s (~12 tok/s) | 8–16 tok/s | 300–470 s |
| **Ollama, MX150 (encoder on GPU, decoder on CPU)** | **53.6 s (53 tok/s, GPU 100%)** | 13.1 tok/s | **76 s** |
| llama.cpp Vulkan (encoder on GPU, q4 decoder on CPU) | ~134 s (~21 tok/s) | ~4–6 tok/s | ~656 s |

Key facts behind the numbers:

- **No custom build is needed for the GPU path.** Ollama ships two CUDA runners: the `cuda_v13` one dropped sm_61 and self-skips, but the bundled `cuda_v12` runner still carries sm_61 kernels and is picked automatically once a Pascal driver is present.
- Ollama's automatic split puts the **vision encoder on the GPU and the 0.5 B decoder + KV cache on the CPU** — exactly what fits in 2 GB. (Full-model residency OOMs: the encode graph alone needs a ~677 MB activation buffer.)
- Generate-side speed is CPU-bound and unchanged; the win is the encoder, which dominates cold-page time.

## Requirements

- Arch Linux (or derivative) with a Pascal (sm_61) NVIDIA GPU, any VRAM ≥ ~1.9 GB free
- [Ollama](https://ollama.com) — official tarball (≥ 0.40) or repo package
- `nvidia-580xx-utils` + `nvidia-580xx-dkms` (AUR) and `linux-headers` — **the legacy 580 branch is the last driver line supporting Pascal**; mainline `nvidia` (≥ 590) dropped it, and CUDA ≥ 13 dropped sm_61 entirely

## Setup

### 1. Driver (one-time, needs sudo)

```bash
paru -S nvidia-580xx-utils nvidia-580xx-dkms linux-headers

# keep nouveau off, load compute modules at boot, then verify:
printf 'blacklist nouveau\noptions nouveau modeset=0\n' | sudo tee /etc/modprobe.d/nouveau-off.conf
printf 'nvidia\nnvidia_uvm\n' | sudo tee /etc/modules-load.d/nvidia-compute.conf
sudo modprobe nvidia nvidia_uvm && nvidia-smi
```

Graphics can stay on the iGPU: never load `nvidia_drm`/`nvidia_modeset`, no X/Wayland config changes.

### 2. Model

```bash
ollama pull glm-ocr:q8_0
```

The benchmarked build is pinned by digest — `scripts/setup_model.sh` pulls and verifies it:

| tag | sha256 of LLM blob | size |
|---|---|---|
| `glm-ocr:q8_0` | `6e5935a916fc8d6878a67c60c2b99648f54e10b4a4af4cbbfb2c308953000e5f` | 907 MB |

### 3. Ollama environment (load-bearing)

```bash
# systemd drop-in for the system ollama service:
sudo systemctl edit ollama
```

```ini
[Service]
Environment=OLLAMA_CONTEXT_LENGTH=16384
Environment=OLLAMA_NUM_PARALLEL=1
Environment=OLLAMA_FLASH_ATTENTION=1
Environment=OLLAMA_KV_CACHE_TYPE=q8_0
Environment=OLLAMA_KEEP_ALIVE=10m
```

`OLLAMA_CONTEXT_LENGTH=16384` is **required**: the default 4096 is smaller than a page image (~2851 tokens), and every request dies with an HTTP 500. `NUM_PARALLEL=1` keeps a single model resident; `KEEP_ALIVE=10m` unloads it 10 minutes after the last request, so idle cost drops back to ~nothing (while warm: ~2.3 GB RAM + ~1.8 GB VRAM).

## Usage

```bash
# OCR a page (text recognition) — first call encodes, repeats hit the KV cache:
ollama run glm-ocr:q8_0 "Text Recognition:" page.png     # (or any prompt of your choice)

# To free VRAM for other GPU work without stopping the server:
curl -s http://127.0.0.1:11434/api/generate -d '{"model": "glm-ocr:q8_0", "keep_alive": 0}'
```

### Benchmark harness

`scripts/bench_page.py` measures time-to-first-token, prompt/encode time, generation speed and total time against a running Ollama or `llama-server`, appending JSON lines to `results/bench.jsonl`:

```bash
python3 scripts/bench_page.py --backend ollama --label mx150-q8_0 \
  --page /path/to/page.png
```

Use a page that hasn't been sent before: unchanged images are KV-cache hits and will show `prompt_s ≈ 0`.

### Helper scripts (for llama.cpp, optional)

- `scripts/extract_gguf_blobs.py` — copy the LLM and mmproj GGUFs out of `~/.ollama/models` into `models/`
- `scripts/quantize_llm.sh` — re-quantize the F16 blob (e.g. `q4_K_M`, 524 MB) — needed only for full-GPU-residency experiments

## llama.cpp alternative (Vulkan)

A Vulkan build of llama.cpp works on Pascal without CUDA at all, but its encoder is ~2.6× slower than Ollama's CUDA path on this GPU class, so treat it as a fallback:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp
# llama.cpp's ggml-vulkan needs SPIRV-Headers; headers-only install is enough:
git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers third_party/SPIRV-Headers
cmake -S third_party/SPIRV-Headers -B build/spirv-headers -DCMAKE_INSTALL_PREFIX=$PWD/spirv-usr
cmake --install build/spirv-headers

cmake -S third_party/llama.cpp -B build-vk -DGGML_VULKAN=ON -DGGML_CUDA=OFF \
  -DCMAKE_PREFIX_PATH=$PWD/spirv-usr -DCMAKE_CXX_FLAGS="-I$PWD/spirv-usr/include"
cmake --build build-vk --target llama-server -j8

# encoder on GPU, decoder on CPU (-ngl 99 OOMs: 2 GB cannot hold weights + encode graph):
GGML_VK_VISIBLE_DEVICES=1 ./build-vk/bin/llama-server -m models/glm-ocr-q4_K_M.gguf \
  -mm models/glm-ocr-mmproj-q8_0.gguf -c 8192 -ngl 0 --port 8081
```

## Known gotchas

- Unchanged page images are KV-cache hits — cold-encode benchmarks need fresh pages (or restart the server).
- With the model warm, ~130 MB VRAM stays free; unload the model before other dGPU work.
- GLM-OCR through GGUF/Ollama is **text recognition only** — layout detection needs the upstream SDK with vLLM/SGLang on larger hardware.

## Repository layout

- `scripts/glmocr` — one-shot OCR CLI (image → text over a local Ollama server)
- `scripts/bench_page.py` — timing harness (appends to `results/`, created at runtime)
- `scripts/extract_gguf_blobs.py`, `scripts/quantize_llm.sh`, `scripts/setup_model.sh` — model and llama.cpp helpers

The full feasibility study (Pascal driver/CUDA support matrix, VRAM budget math, runtime comparison) lives in the project wiki.

## Related work

Generic GLM-OCR deployment is well covered elsewhere; this repo's niche is the legacy-Pascal/2 GB-VRAM story:

- Official: [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR) (incl. [examples/ollama-deploy](https://github.com/zai-org/GLM-OCR/blob/main/examples/ollama-deploy/README.md)), [Ollama library entry](https://ollama.com/library/glm-ocr), [vLLM recipe](https://docs.vllm.ai/projects/recipes/en/stable/GLM/GLM-OCR.html)
- Community: [Self-hosting GLM-OCR with vLLM (DebuggerCafe)](https://debuggercafe.com/self-hosting-glm-ocr-using-vllm-document-layout-and-ocr), [r/LocalLLaMA GGUF discussion](https://www.reddit.com/r/LocalLLaMA/comments/1rq1itg/do_i_need_to_use_ollama_to_get_the_full_feature/)

## License

[MIT](LICENSE)
