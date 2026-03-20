# VLM-Orchestrated Hybrid OCR Pipeline

A sophisticated, multi-stage document intelligence system that combines the **spatial reasoning** of Vision-Language Models (VLM) with the **precision** of traditional OCR and the **contextual power** of Large Language Models (LLM).

## 🚀 Project Overview
Traditional OCR often fails with complex layouts, handwriting, and structural reconstruction. This project implements an **Agentic OCR Pipeline** that treats document extraction as a three-stage intelligent workflow:

1.  **Layout Interpretation (VLM):** `Florence-2` analyzes the full document page to identify logical text regions and their spatial coordinates (`<OCR_WITH_REGION>`).
2.  **Localized Production (Traditional OCR):** `EasyOCR` performs high-precision character recognition on the specific image crops identified by the VLM.
3.  **Contextual Correction (LLM):** `Qwen-2.5-3B` processes the raw text fragments alongside their bounding box coordinates to reconstruct a clean, spell-corrected Markdown document.

---

## 🏗️ Architecture & Stack
The pipeline is designed to be **100% Local**, optimized for a single NVIDIA T4 GPU (16GB VRAM) using **4-bit quantization (NF4)** to ensure all three models reside in memory simultaneously.

| Component | Model | Role |
| :--- | :--- | :--- |
| **VLM** | `Florence-2-Large` | Spatial reasoning & region detection |
| **OCR Engine** | `EasyOCR` | Raw character extraction from cropped regions |
| **LLM** | `Qwen-2.5-3B-Instruct` | Error correction, layout mapping, & Markdown formatting |

### Technical Highlights:
* **Prompt Compression:** Spatial data is converted from heavy JSON to a dense `[x,y,w,h]: text` format to optimize the LLM's context window and reduce VRAM usage.
* **Contextual Self-Correction:** The LLM uses surrounding vocabulary and document structure to "predict" correct spellings for ambiguous OCR characters (e.g., distinguishing '5' from 'S').
* **Stable Environment:** Specifically pinned to `transformers==4.49.0` to maintain compatibility with Microsoft’s custom Florence-2 architecture.

---

