---
title: MediVision — Multimodal Medical Imaging AI Agent
emoji: 🩺
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: true
license: mit
---

# MediVision — Multimodal Medical Imaging AI Agent

> **AMD Developer Hackathon 2026 · Track 3: Vision & Multimodal AI**

[![Hugging Face Space](https://img.shields.io/badge/🤗%20HF%20Space-Live%20Demo-yellow)](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/medivision-ai-agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Powered by AMD](https://img.shields.io/badge/Powered%20by-AMD%20MI300X%20%2B%20ROCm-ED1C24)](https://www.amd.com/en/products/accelerators/instinct/mi300.html)

MediVision is a **multilingual** multimodal AI assistant that analyzes skin wound and disease images combined with patient symptom descriptions. It supports **English, Tiếng Việt, 中文, Español, Français, and 日本語** — every output (diagnosis, severity, actions, disclaimer) is delivered natively in the selected language. Inference is served by a **vLLM server running on AMD Developer Cloud** (AMD Instinct™ MI300X + ROCm), and the lightweight Gradio frontend on Hugging Face Spaces simply calls that API — no model weights are loaded in the Space itself.

---

## Live Demo

[https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/medivision-ai-agent](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/medivision-ai-agent)

---

## Architecture

```
User Browser
     │
     ▼
Hugging Face Space  (Gradio UI — CPU, no GPU needed)
     │  HTTP  (OpenAI-compatible chat/completions)
     ▼
AMD Developer Cloud VM
└── vLLM server  ←  Qwen/Qwen2.5-VL-7B-Instruct
        running on AMD Instinct™ MI300X + ROCm
```

The HF Space encodes any uploaded image as base64 and passes it to the vLLM
endpoint via the standard OpenAI vision message format.  This keeps the Space
footprint tiny (no PyTorch, no model download) while all heavy lifting happens
on the AMD GPU server.

---

## Features

- **Multimodal Analysis** — Combines skin image + freeform symptom text for richer diagnosis suggestions.
- **Multilingual** — Full support for 6 languages: English, Tiếng Việt, 中文 (Chinese), Español, Français, and 日本語 (Japanese). All output — diagnosis, severity, recommendations, and disclaimer — is rendered natively in the selected language.
- **Structured Output** — Every analysis returns:
  - Diagnosis suggestion
  - Severity badge: `Low` · `Medium` · `High` · `Urgent`
  - Actionable recommended steps (clinical-grade language)
  - Confidence score with visual progress bar
- **AMD MI300X Powered** — Inference via vLLM on AMD Instinct™ MI300X + ROCm.
- **HF Space Ready** — Minimal dependencies; no GPU required in the Space.

---

## Conditions Analyzed

| Condition | Typical Severity |
|---|---|
| Superficial Abrasions & Cuts | Low |
| Contact Dermatitis / Allergic Rash | Low – Medium |
| Atopic Eczema Flare | Medium |
| Tinea Corporis (Ringworm) | Low |
| Psoriasis Plaque | Medium |
| Partial-Thickness Burn (2nd degree) | High |
| Suspected Cellulitis / Skin Infection | Urgent |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vision Model | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Inference Runtime | **vLLM** (OpenAI-compatible API) |
| Inference Hardware | **AMD Instinct™ MI300X** (192 GB HBM3) + ROCm |
| Inference Host | AMD Developer Cloud |
| Frontend | Gradio 5.29 (Hugging Face Space — CPU) |
| Language Support | English · Tiếng Việt · 中文 · Español · Français · 日本語 |

---

## Project Structure

```
medivision-ai-agent/
├── app.py                   # HF Space entry point (Gradio UI)
├── requirements.txt         # Minimal dependencies (no PyTorch)
├── .env.example             # Environment variable reference
├── README.md                # This file
├── LICENSE                  # MIT
├── src/
│   ├── __init__.py
│   ├── config.py            # VLLM_API_URL, MODEL_NAME, MOCK_MODE, etc.
│   ├── model_loader.py      # OpenAI client → vLLM (image base64 + text)
│   ├── agent.py             # analyze_image_and_text(), mock data pools
│   └── inference.py         # MediVisionPipeline orchestrator
└── sample_test_images/
    └── ABOUT.md
```

---

## Running Locally

### Prerequisites

- Python 3.10+
- A running vLLM server with `Qwen/Qwen2.5-VL-7B-Instruct`  
  (AMD Developer Cloud or any machine with an AMD/NVIDIA GPU)

### 1. Clone

```bash
git clone https://github.com/your-org/medivision-ai-agent
cd medivision-ai-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the vLLM endpoint

```bash
cp .env.example .env
# Edit .env and set VLLM_API_URL to your AMD VM address
```

Or export directly:

```bash
export VLLM_API_URL=http://<AMD_VM_IP>:8000
export MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
```

### 4. Launch

```bash
python app.py
```

The app is available at `http://localhost:7860`.

---

## Starting the vLLM Server (AMD Developer Cloud)

On your AMD MI300X VM with ROCm installed:

```bash
pip install vllm

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 4096
```

The server exposes an OpenAI-compatible API at `http://<VM_IP>:8000/v1`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_API_URL` | `http://localhost:8000` | Base URL of the vLLM server |
| `MODEL_NAME` | `Qwen/Qwen2.5-VL-7B-Instruct` | Model ID served by vLLM |
| `VLLM_API_KEY` | `not-required` | API key (if vLLM auth is enabled) |

| `MAX_NEW_TOKENS` | `512` | Max tokens to generate |
| `TEMPERATURE` | `0.2` | Sampling temperature |

> **HF Space secret:** set `VLLM_API_URL` in the Space settings → *Repository secrets* so the Gradio app can reach your AMD VM.

---

## Disclaimer

MediVision is a **demonstration prototype** built for the AMD Developer Hackathon 2026. It is intended for educational and informational purposes only. It does **not** replace professional medical advice, diagnosis, or treatment. Always consult a licensed healthcare professional for any medical concerns.

---

## License

[MIT License](LICENSE) © 2026 MediVision Team

---

*Built with ❤️ on AMD ROCm · AMD Developer Cloud · AMD Developer Hackathon 2026*
