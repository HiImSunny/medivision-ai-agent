---
title: MediVision — Multimodal Medical Imaging AI Agent
emoji: 🩺
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
---

# MediVision — Multimodal Medical Imaging AI Agent

> **AMD Developer Hackathon 2026 · Track 3: Vision & Multimodal AI**

[![Hugging Face Space](https://img.shields.io/badge/🤗%20HF%20Space-Live%20Demo-yellow)](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/medivision-ai-agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Powered by AMD](https://img.shields.io/badge/Powered%20by-AMD%20MI300X%20%2B%20ROCm-ED1C24)](https://www.amd.com/en/products/accelerators/instinct/mi300.html)

MediVision is a bilingual (English / Vietnamese) multimodal AI assistant that analyzes skin wound and disease images combined with patient symptom descriptions. It runs on AMD Instinct™ MI300X GPUs with ROCm and delivers structured clinical insights via a sleek Gradio interface.

---

## Live Demo

[https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/medivision-ai-agent](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/medivision-ai-agent)

---

## Screenshots

> _Add screenshots of the live UI here_

| Upload & Analyze | Results Card |
|:---:|:---:|
| ![Upload screen](sample_test_images/screenshot_upload.png) | ![Results screen](sample_test_images/screenshot_results.png) |

---

## Features

- **Multimodal Analysis** — Combines skin image + freeform symptom text for richer diagnosis suggestions.
- **Bilingual** — Full English and Vietnamese (Tiếng Việt) support; auto-detects input language preference.
- **Structured Output** — Every analysis returns:
  - Diagnosis suggestion
  - Severity badge: `Low` · `Medium` · `High` · `Urgent`
  - Actionable recommended steps (clinical-grade language)
  - Confidence score with visual progress bar
- **AMD MI300X Optimized** — Inference runs on the world-class AMD Instinct™ MI300X via ROCm.
- **Graceful Mock Mode** — Falls back to realistic mock responses if the GPU model is unavailable, so the demo always runs.
- **HF Space Ready** — Single `app.py` entry point, compatible with Hugging Face Spaces.

---

## Conditions Analyzed

The model is trained to identify and advise on (but not limited to):

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
| Vision Model | [Qwen/Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) |
| Inference Runtime | `transformers` + `optimum[amd]` on **AMD ROCm** |
| Hardware | **AMD Instinct™ MI300X** (192 GB HBM3) |
| Agent Orchestration | LangChain |
| Frontend | Gradio 4 |
| Language Support | English / Tiếng Việt |

---

## Project Structure

```
medivision-ai-agent/
├── app.py                   # HF Space entry point (Gradio UI)
├── requirements.txt         # Pinned dependencies
├── README.md                # This file
├── LICENSE                  # MIT
├── src/
│   ├── __init__.py
│   ├── config.py            # MODEL_NAME, MOCK_MODE, device settings
│   ├── model_loader.py      # Qwen-VL-Chat loader (real + mock fallback)
│   ├── agent.py             # analyze_image_and_text(), mock data pools
│   └── inference.py         # MediVisionPipeline orchestrator
└── sample_test_images/
    └── ABOUT.md             # Instructions for adding test images
```

---

## Running Locally

### Prerequisites

- Python 3.10+
- AMD GPU with [ROCm 6.1+](https://rocm.docs.amd.com/) **or** any CUDA GPU **or** CPU (mock mode)

### 1. Clone

```bash
git clone https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/medivision-ai-agent
cd medivision-ai-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For AMD ROCm, replace the PyTorch install:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
```

### 3. (Optional) Set environment variables

```bash
export MODEL_NAME="Qwen/Qwen-VL-Chat"   # default
export MOCK_MODE=false                   # set true to skip model download
export HF_TOKEN="hf_..."                # if model is gated
```

### 4. Launch

```bash
python app.py
```

The app is available at `http://localhost:7860`.

### 5. Force mock mode (no GPU required)

```bash
MOCK_MODE=true python app.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen-VL-Chat` | HF model ID to load |
| `MOCK_MODE` | `false` | Force mock mode (skip real model) |
| `DEVICE` | `cuda` | Inference device (`cuda` / `cpu`) |
| `MAX_NEW_TOKENS` | `512` | Max tokens to generate |
| `TEMPERATURE` | `0.2` | Sampling temperature |
| `HF_TOKEN` | _(empty)_ | HF auth token for gated models |

---

## Disclaimer

MediVision is a **demonstration prototype** built for the AMD Developer Hackathon 2026. It is intended for educational and informational purposes only. It does **not** replace professional medical advice, diagnosis, or treatment. Always consult a licensed healthcare professional for any medical concerns.

---

## License

[MIT License](LICENSE) © 2026 MediVision Team

---

*Built with ❤️ on AMD ROCm · AMD Developer Hackathon 2026*
