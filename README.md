<div align="center">

# 🎨 Text-to-Image Generation Pipeline

### Production-Grade Generative AI System Using Stable Diffusion

[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-Hugging_Face-yellow?style=for-the-badge)](https://huggingface.co/spaces/Sairaj69/text-to-image-generator)
[![Open In Colab](https://img.shields.io/badge/Open_In-Colab-F9AB00?style=for-the-badge&logo=google-colab)](https://colab.research.google.com/drive/1QODiw5mcHV25hCY0Hn60QhwfAiUzswF7)
[![GitHub](https://img.shields.io/badge/Source-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/SairajNarayankar/text-to-image-generation-pipeline)

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Diffusers](https://img.shields.io/badge/🤗_Diffusers-0.25+-FFD21E?style=flat-square)](https://huggingface.co/docs/diffusers)
[![Stable Diffusion](https://img.shields.io/badge/Stable_Diffusion-v1.5-purple?style=flat-square)](https://stability.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<p align="center">
  <b>Engineered a complete generative AI pipeline with 14 style presets, advanced prompt engineering,<br>
  latent space operations, and production-ready deployment across CLI, Web UI, and REST API.</b>
</p>

</div>

---

## 🖼️ Sample Outputs

<div align="center">

| Photorealistic | Cyberpunk | Fantasy |
|:-:|:-:|:-:|
| ![Photorealistic](showcase/photorealistic_sample.png) | ![Cyberpunk](showcase/cyberpunk_sample.png) | ![Fantasy](showcase/fantasy_sample.png) |
| *Majestic lion at sunset* | *Neon-lit cyberpunk street* | *Enchanted forest* |

| Anime | Oil Painting | 3D Render |
|:-:|:-:|:-:|
| ![Anime](showcase/anime_sample.png) | ![Oil Painting](showcase/oil_painting_sample.png) | ![3D Render](showcase/3d_render_sample.png) |
| *Studio quality anime* | *Classical oil painting* | *Octane rendered scene* |

### Style Comparison (Same Prompt, Different Styles)
![Style Comparison](showcase/style_comparison.png)

### Marketing Portfolio
![Marketing](showcase/marketing_grid.png)

</div>

---

## ✨ Key Features

### 🎯 Model Implementation
- Engineered generative AI pipeline using **Stable Diffusion** and **Hugging Face Diffusers**
- Supports multiple models: **SD 1.5, SD 2.1, SDXL, DreamShaper 8**
- Full control over inference steps, guidance scale, resolution, and seed
- **DPM-Solver++ with Karras sigmas** for optimal quality/speed balance

### 📝 Prompt Engineering
- **14 built-in style presets** with optimized positive and negative keywords
- **4 quality levels** (Draft → Ultra) with cascading enhancements
- **60+ negative prompt terms** to prevent hallucinations and artifacts
- **Prompt analysis engine** with 5-category scoring and improvement suggestions
- **Emphasis weighting** syntax `(term:weight)` for fine-grained control
- **Token estimation** to prevent CLIP truncation (77 token limit)
- **Template system** with variable substitution for domain-specific generation

### ⚡ Pipeline Optimization
- **xformers** memory-efficient attention (~40% VRAM reduction)
- **Attention slicing** and **VAE slicing** for lower memory usage
- **14 noise schedulers** with runtime switching and recommendations
- **float16 precision** for faster inference on GPU
- **torch.compile** support for PyTorch 2.0+
- **Automatic device detection** (CUDA, MPS, CPU)

### 🧠 Latent Space Operations
- **SLERP interpolation** between latent representations (smooth morphing)
- **Latent arithmetic** for concept combination
- **Latent walk generation** for smooth animations
- **Noise strength injection** for controlled variation
- **Latent statistics analysis** for debugging

### 🖼️ Post-Processing Pipeline
- Enhancement: Sharpness, Contrast, Brightness, Saturation
- Grid/contact sheet creation with labels
- Side-by-side style and quality comparisons
- Text watermarking with configurable position/opacity
- Multi-size web export (thumbnail → original)
- Batch processing with chained operations

### 🖥️ Multiple Interfaces
- **CLI** (Click): Full command-line interface for automation
- **Gradio Web UI**: Interactive browser interface with 4 tabs
- **FastAPI REST API**: Production-ready API with OpenAPI docs

### 📦 Portfolio & Batch Tools
- Pre-built collections: Marketing, Creative Design, Rapid Prototyping
- Batch generation from JSON, CSV, YAML input files
- HTML gallery export with responsive design
- Generation history tracking and metadata logging

---

## 🚀 Quick Start

### Try Live Demo (No Setup Required)
👉 **[Launch on Hugging Face Spaces](https://huggingface.co/spaces/Sairaj69/text-to-image-generator)**

### Run on Google Colab (Free GPU)
👉 **[Open in Colab](https://colab.research.google.com/drive/1QODiw5mcHV25hCY0Hn60QhwfAiUzswF7)**

### Local Installation

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/text-to-image-generation-pipeline.git
cd text-to-image-generation-pipeline

# Setup
python -m venv venv && source venv/bin/activate  # Linux/Mac
# OR: python -m venv venv && venv\Scripts\activate  # Windows

# Install
pip install -r requirements.txt

# Generate
python -m app.main generate "A sunset over mountains" --style photorealistic --seed 42

╔════════════════════════════════════════════════════════════════════╗
║                       APPLICATION LAYER                            ║
║  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  ║
║  │   Gradio    │  │  FastAPI    │  │     CLI     │  │  Batch   │  ║
║  │   Web UI    │  │    API      │  │   (Click)   │  │  Engine  │  ║
║  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────┬─────┘  ║
╠═════════╧═════════════════╧═════════════════╧═════════════╧════════╣
║                        CORE PIPELINE                                ║
║               ┌─────────────────────────────────┐                   ║
║               │  TextToImagePipeline            │                   ║
║               │  (Orchestration & Execution)    │                   ║
║               └──────┬──────────┬──────────┬────┘                   ║
╠══════════════════════╧══════════╧══════════╧════════════════════════╣
║                      COMPONENT LAYER                                ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐   ║
║  │   Model Loader   │  │  Prompt Engineer │  │  Scheduler Mgr │   ║
║  ├──────────────────┤  ├──────────────────┤  ├────────────────┤   ║
║  │ • 6 models       │  │ • 14 styles      │  │ • 14 schedulers│   ║
║  │ • optimization   │  │ • analysis tools │  │ • recommend    │   ║
║  └──────────────────┘  └──────────────────┘  └────────────────┘   ║
║                                                                      ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐   ║
║  │ Latent Processor │  │ Image Processor  │  │   Utilities    │   ║
║  ├──────────────────┤  ├──────────────────┤  ├────────────────┤   ║
║  │ • SLERP          │  │ • enhance        │  │ • config       │   ║
║  │ • interpolation  │  │ • watermark      │  │ • logging      │   ║
║  │                  │  │ • grid/export    │  │ • device mgmt  │   ║
║  └──────────────────┘  └──────────────────┘  └────────────────┘   ║
╠═══════��══════════════════════════════════════════════════════════════╣
║                      FOUNDATION LAYER                               ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐   ║
║  │    Diffusers     │  │  Transformers    │  │    PyTorch     │   ║
║  │   (Stable Diff)  │  │  (CLIP / T5)     │  │ (CUDA / MPS)   │   ║
║  └──────────────────┘  └──────────────────┘  └────────────────┘   ║
╚════════════════════════════════════════════════════════════════════��═╝

text-to-image-generation-pipeline/
├── src/                        # Core library
│   ├── pipeline.py             # Main orchestrator
│   ├── model_loader.py         # Model management (6 models)
│   ├── prompt_engineer.py      # Prompt optimization (14 styles)
│   ├── scheduler_manager.py    # Scheduler control (14 schedulers)
│   ├── latent_manager.py       # Latent space operations
│   ├── image_processor.py      # Post-processing pipeline
│   └── utils.py                # Shared utilities
├── app/                        # Application interfaces
│   ├── main.py                 # CLI entry point
│   ├── gradio_app.py           # Web UI
│   ├── api.py                  # REST API
│   ├── portfolio_generator.py  # Portfolio collections
│   └── batch_generator.py      # Batch processing
├── configs/                    # YAML configurations
├── templates/                  # Generation templates (JSON)
├── scripts/                    # Utility scripts
├── tests/                      # Test suite
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Documentation
├── showcase/                   # Sample outputs
└── output/                     # Generated images

🔧 Usage
CLI Commands
python -m app.main generate "prompt" --style photorealistic --seed 42
python -m app.main variations "prompt" --num 4
python -m app.main compare-styles "prompt"
python -m app.main compare-quality "prompt" --steps-list "10,20,30,50"
python -m app.main portfolio --seed 42
python -m app.main batch templates/marketing_templates.json
python -m app.main analyze "your prompt"
python -m app.main webui --port 7860
python -m app.main api --port 8000


Python API
python
from src.pipeline import TextToImagePipeline

pipeline = TextToImagePipeline.from_config("config.yaml")
pipeline.setup()

result = pipeline.generate(
    prompt="A majestic castle at sunset",
    style="fantasy",
    quality="ultra",
    seed=42,
)
result["images"][0].show()


REST API
bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A sunset", "style": "photorealistic", "seed": 42}'


📊 Supported Models
Model	Resolution	VRAM	Quality
DreamShaper 8	512px	4 GB	⭐⭐⭐⭐⭐
Stable Diffusion 1.5	512px	4 GB	⭐⭐⭐⭐
Stable Diffusion 2.1	768px	5 GB	⭐⭐⭐⭐
SDXL	1024px	8 GB	⭐⭐⭐⭐⭐
SDXL Turbo	512px	6 GB	⭐⭐⭐⭐
Realistic Vision 5.1	512px	4 GB	⭐⭐⭐⭐⭐


🧪 Testing

pytest tests/ -v
pytest tests/ --cov=src --cov-report=html


🤝 Contributing
Fork the repository
Create feature branch: git checkout -b feature/new-feature
Commit: git commit -m "Add new feature"
Push: git push origin feature/new-feature
Open Pull Request


📄 License
MIT License — see LICENSE for details.


🙏 Acknowledgments
Hugging Face Diffusers
Stability AI — Stable Diffusion
Lykon — DreamShaper model
Gradio — Web UI framework
FastAPI — REST API framework


⭐ Star this repo if you found it helpful!
Built with ❤️ for the Generative AI community
