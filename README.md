---

```markdown
# 🎨 Text-to-Image Generation Pipeline

A production-grade Generative AI pipeline using Stable Diffusion and Hugging Face Diffusers to convert natural language prompts into high-resolution images.

## ✨ Features

- 14 style presets (photorealistic, digital art, anime, cyberpunk, etc.)
- 4 quality levels (draft, standard, high, ultra)
- 14 noise schedulers with runtime switching
- Prompt engineering with analysis, scoring, emphasis weighting
- Latent space operations (interpolation, arithmetic, walks)
- Post-processing (enhance, watermark, grid, web export)
- CLI, Gradio Web UI, FastAPI REST API
- Batch generation from JSON/CSV/YAML
- Portfolio generator for marketing, creative, prototyping assets

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/yourusername/text-to-image-pipeline.git
cd text-to-image-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Generate First Image

```python
from src.pipeline import TextToImagePipeline

pipeline = TextToImagePipeline.from_config("config.yaml")
pipeline.setup()
result = pipeline.generate("A cozy cabin in snowy mountains", style="photorealistic", seed=42)
result["images"][0].show()
pipeline.cleanup()
```

### CLI

```bash
python -m app.main generate "A sunset over mountains" --style photorealistic --seed 42
python -m app.main variations "A cyberpunk city" --num 4
python -m app.main compare-styles "A medieval castle"
python -m app.main portfolio --seed 42
python -m app.main batch templates/marketing_templates.json
python -m app.main webui --port 7860
python -m app.main api --port 8000
python -m app.main info
python -m app.main analyze "a cat sitting on a chair"
```

## 📁 Project Structure

```
text-to-image-pipeline/
├── README.md
├── requirements.txt
├── setup.py
├── config.yaml
├── .env
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # Main TextToImagePipeline
│   ├── model_loader.py        # Model download/cache/optimize
│   ├── prompt_engineer.py     # Prompt building/analysis
│   ├── scheduler_manager.py   # Noise scheduler management
│   ├── latent_manager.py      # Latent space operations
│   ├── image_processor.py     # Post-processing
│   └── utils.py               # Helpers
├── app/
│   ├── __init__.py
│   ├── main.py                # CLI (Click)
│   ├── gradio_app.py          # Gradio Web UI
│   ├── api.py                 # FastAPI REST API
│   ├── portfolio_generator.py # Portfolio collections
│   └── batch_generator.py     # Batch processing
├── configs/
│   ├── prompt_templates.yaml
│   ├── style_presets.yaml
│   └── generation_presets.yaml
├── templates/
│   ├── marketing_templates.json
│   ├── creative_templates.json
│   └── prototype_templates.json
├── scripts/
│   ├── download_models.py
│   ├── benchmark.py
│   └── export_portfolio.py
├── tests/
│   ├── test_pipeline.py
│   ├── test_prompt_engineer.py
│   ├── test_image_processor.py
│   └── test_latent_manager.py
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_prompt_engineering.ipynb
│   ├── 03_pipeline_optimization.ipynb
│   └── 04_latent_space_analysis.ipynb
├── docs/
│   ├── architecture.md
│   ├── prompt_guide.md
│   └── api_reference.md
└── output/
    ├── generated/
    ├── portfolio/
    ├── batch/
    ├── comparisons/
    ├── grids/
    ├── interpolations/
    ├── animations/
    ├── exports/
    └── metadata/
```

## 🔧 Python API Usage

### Basic Generation

```python
result = pipeline.generate(
    prompt="A majestic lion on a rocky cliff",
    style="photorealistic",
    quality="ultra",
    num_inference_steps=50,
    guidance_scale=8.5,
    width=768, height=512,
    seed=42,
    enhance_prompt=True,
    auto_enhance_image=True,
)
print(f"Time: {result['elapsed_time']:.2f}s")
print(f"Saved: {result['paths']}")
```

### Variations

```python
result = pipeline.generate_variations(
    prompt="A cyberpunk street scene",
    num_variations=6, seed_start=42, style="cyberpunk",
)
result["grid"].save("variations.png")
```

### Style Comparison

```python
result = pipeline.generate_style_comparison(
    prompt="A medieval castle",
    styles=["photorealistic","anime","oil_painting","cyberpunk","watercolor","fantasy"],
    seed=42,
)
result["grid"].save("styles.png")
```

### Quality Comparison

```python
result = pipeline.generate_quality_comparison(
    prompt="A steampunk mechanism",
    step_counts=[10, 15, 20, 30, 50, 75],
    seed=42,
)
print("Timing:", result["timing_data"])
```

### Latent Interpolation

```python
result = pipeline.generate_latent_interpolation(
    prompt="A serene landscape",
    num_frames=10, seed_start=42, seed_end=123, method="slerp",
)
pipeline.save_generation_gif(result["images"], "morph.gif")
```

### Prompt Engineer Standalone

```python
from src.prompt_engineer import PromptEngineer
engineer = PromptEngineer()

result = engineer.build_prompt(
    "a dragon over a castle",
    style="fantasy", quality="ultra",
    emphasis={"dragon": 1.5},
    negative_categories=["quality","watermark","anatomy"],
)
print("Positive:", result.positive)
print("Negative:", result.negative)

analysis = engineer.analyze_prompt("a cat")
print(f"Score: {analysis['score']}/100 ({analysis['rating']})")
for s in analysis["suggestions"]:
    print(f"  💡 {s}")
```

## 🎨 Style Presets

| Style | Best For |
|-------|----------|
| `photorealistic` | Product shots, portraits, landscapes |
| `digital_art` | Concept art, illustrations |
| `oil_painting` | Fine art, classical themes |
| `watercolor` | Soft artistic compositions |
| `anime` | Japanese-style illustrations |
| `minimalist` | UI mockups, clean designs |
| `cyberpunk` | Futuristic neon scenes |
| `vintage` | Retro, nostalgic imagery |
| `3d_render` | Product design, architecture |
| `sketch` | Concept sketches, storyboards |
| `fantasy` | Magical/mythical scenes |
| `pop_art` | Bold graphic designs |
| `cinematic` | Movie stills, dramatic scenes |
| `isometric` | Game assets, dioramas |

## ⏱️ Schedulers

| Scheduler | Speed | Quality | Best For |
|-----------|-------|---------|----------|
| `dpm_solver_multistep` | Fast | High | Default, balanced |
| `euler_ancestral` | Fast | High | Creative/varied results |
| `euler` | Fast | High | Deterministic results |
| `unipc` | Very Fast | High | Low step counts |
| `ddim` | Medium | Good | Supports inversion |
| `heun` | Slow | Very High | Maximum quality |
| `pndm` | Medium | Good | Default for many models |
| `lms` | Medium | Good | Linear multistep |
| `dpm_solver_singlestep` | Fast | High | Alternative DPM |
| `kdpm2` | Medium | High | Karras DPM2 |
| `kdpm2_ancestral` | Medium | High | Stochastic Karras |
| `deis` | Fast | High | Exponential integrator |
| `ddpm` | Very Slow | Baseline | Original (many steps) |

## ⚡ Generation Presets

| Preset | Steps | Size | Guidance | Use Case |
|--------|-------|------|----------|----------|
| `draft` | 10 | 384² | 6.0 | Prompt testing |
| `fast` | 15 | 512² | 7.0 | Quick previews |
| `balanced` | 30 | 512² | 7.5 | Default |
| `high_quality` | 50 | 768² | 9.0 | Portfolio |
| `ultra_quality` | 75 | 768² | 11.0 | Maximum quality |
| `portrait` | 30 | 512×768 | 7.5 | Portrait |
| `landscape` | 30 | 768×512 | 7.5 | Landscape |
| `widescreen` | 30 | 896×512 | 7.5 | Banners |

## 🤖 Supported Models

| Model | ID | Resolution | VRAM |
|-------|----|-----------|------|
| SD 1.5 | `runwayml/stable-diffusion-v1-5` | 512px | 4 GB |
| SD 2.1 | `stabilityai/stable-diffusion-2-1` | 768px | 5 GB |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | 1024px | 8 GB |
| SDXL Turbo | `stabilityai/sdxl-turbo` | 512px | 6 GB |
| DreamShaper 8 | `Lykon/dreamshaper-8` | 512px | 4 GB |
| Realistic Vision | `SG161222/Realistic_Vision_V5.1_noVAE` | 512px | 4 GB |

## ⚙️ Configuration

### config.yaml

```yaml
model:
  model_id: "runwayml/stable-diffusion-v1-5"
  torch_dtype: "float16"
  device: "cuda"
  cache_dir: "./model_cache"
  safety_checker: false

generation:
  default:
    num_inference_steps: 30
    guidance_scale: 7.5
    width: 512
    height: 512

optimization:
  enable_xformers: true
  enable_attention_slicing: true
  enable_vae_slicing: true
  enable_vae_tiling: false
  enable_model_cpu_offload: false
  torch_compile: false

scheduler:
  default: "dpm_solver_multistep"

output:
  base_dir: "./output"
  image_format: "png"
  save_metadata: true

logging:
  level: "INFO"
  log_dir: "./logs"
```

### .env

```env
MODEL_ID=runwayml/stable-diffusion-v1-5
MODEL_CACHE_DIR=./model_cache
DEVICE=cuda
HF_TOKEN=your_token_here
OUTPUT_DIR=./output
API_HOST=0.0.0.0
API_PORT=8000
ENABLE_XFORMERS=true
```

## 📡 REST API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/stats` | Generation stats |
| GET | `/styles` | List styles |
| GET | `/quality-levels` | List quality levels |
| GET | `/schedulers` | List schedulers |
| GET | `/models` | List models |
| GET | `/history?limit=10` | History |
| POST | `/generate` | Generate (base64) |
| POST | `/generate/stream` | Generate (bytes) |
| POST | `/analyze` | Analyze prompt |
| POST | `/enhance` | Enhance prompt |

### Generate Request

```json
{
    "prompt": "A beautiful sunset over mountains",
    "style": "photorealistic",
    "quality": "high",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "seed": 42,
    "num_images": 1,
    "enhance_prompt": true,
    "auto_enhance_image": false,
    "output_format": "png"
}
```

### Generate Response

```json
{
    "success": true,
    "images": ["base64_encoded_string"],
    "prompt_used": "enhanced prompt...",
    "negative_prompt_used": "negative prompt...",
    "settings": {"steps": 30, "guidance": 7.5},
    "elapsed_time": 3.45,
    "timestamp": "2024-01-01T12:00:00"
}
```

### Python Client

```python
import requests, base64, io
from PIL import Image

r = requests.post("http://localhost:8000/generate", json={
    "prompt": "A cyberpunk city", "style": "cyberpunk", "seed": 42
})
img = Image.open(io.BytesIO(base64.b64decode(r.json()["images"][0])))
img.save("result.png")
```

### cURL

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A cat in space","style":"digital_art","seed":42}'
```

## 📦 Portfolio Generation

```python
from app.portfolio_generator import PortfolioGenerator
generator = PortfolioGenerator(pipeline)
summary = generator.generate_full_portfolio(seed=42)
```

```bash
python -m app.main portfolio --seed 42
python scripts/export_portfolio.py --input ./output/portfolio --output ./output/exports --create-html
```

## 📋 Batch Generation

### JSON

```json
{
    "batch_name": "my_batch",
    "default_settings": {"style": "digital_art", "quality": "high"},
    "items": [
        {"prompt": "A sunset", "filename": "sunset", "seed": 42},
        {"prompt": "A forest", "filename": "forest", "seed": 43}
    ]
}
```

### CSV

```csv
prompt,filename,style,quality,seed,steps
A sunset,sunset,photorealistic,high,42,30
A forest,forest,digital_art,high,43,30
```

### Python

```python
from app.batch_generator import BatchGenerator
gen = BatchGenerator(pipeline)
result = gen.generate_from_prompts(["A sunset","A forest","A city"], style="photorealistic")
```

## 📝 Prompt Engineering Guide

### Structure
```
[Subject], [Details], [Style], [Lighting], [Quality], [Composition]
```

### Example
```
A majestic lion standing on a rocky cliff,
golden mane flowing in wind, intense gaze,
digital art, trending on artstation,
dramatic sunset lighting, volumetric rays,
highly detailed, sharp focus, 8k,
wide angle, epic composition
```

### Tips
- Be specific and descriptive
- Include lighting descriptions
- Specify art style
- Mention composition (close-up, wide angle)
- Use quality boosters (detailed, sharp, professional)
- Avoid contradictory terms
- Keep under 77 tokens for SD 1.5

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_prompt_engineer.py -v
pytest tests/test_image_processor.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_latent_manager.py -v
```

## 📊 Benchmarks

```bash
python scripts/benchmark.py --config config.yaml --runs 3
```

Tests step counts, resolutions, schedulers, and memory usage. Results saved as JSON.

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_exploration.ipynb` | Basic pipeline usage, styles, variations |
| `02_prompt_engineering.ipynb` | Prompt building, analysis, enhancement |
| `03_pipeline_optimization.ipynb` | Step/scheduler/guidance comparisons |
| `04_latent_space_analysis.ipynb` | Interpolation, arithmetic, statistics |

## 🔧 Scripts

| Script | Description | Command |
|--------|-------------|---------|
| `download_models.py` | Pre-download models | `python scripts/download_models.py --model sd-1.5` |
| `benchmark.py` | Performance testing | `python scripts/benchmark.py` |
| `export_portfolio.py` | Export + HTML gallery | `python scripts/export_portfolio.py --create-html` |

## ❓ Troubleshooting

### Out of Memory (OOM)

```yaml
# Enable in config.yaml
optimization:
  enable_attention_slicing: true
  enable_vae_slicing: true
  enable_model_cpu_offload: true
```

Or reduce resolution/steps:
```bash
python -m app.main generate "prompt" --width 384 --height 384 --steps 15
```

### xformers Not Available

```bash
pip install xformers
# If fails, pipeline works without it (uses default attention)
```

### CUDA Not Detected

```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Slow Generation on CPU

CPU generation is 10-50x slower than GPU. Options:
- Use Google Colab (free GPU)
- Use `draft` preset for faster results
- Reduce resolution to 384×384

### Model Download Fails

```bash
# Set HF token for gated models
export HF_TOKEN=your_token

# Or pre-download
python scripts/download_models.py --model sd-1.5 --cache-dir ./model_cache
```

### Black/Blank Images

- Increase `guidance_scale` (try 7.5-9.0)
- Increase `num_inference_steps` (try 30+)
- Check negative prompt isn't contradicting positive
- Try different seed

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request

### Development Setup

```bash
git clone https://github.com/yourusername/text-to-image-pipeline.git
cd text-to-image-pipeline
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/ -v

# Format code
black src/ app/ tests/

# Lint
flake8 src/ app/
```

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - Diffusion model library
- [Stability AI](https://stability.ai/) - Stable Diffusion models
- [CompVis](https://github.com/CompVis/stable-diffusion) - Original Stable Diffusion
- [RunwayML](https://runwayml.com/) - SD 1.5 model hosting
- [Gradio](https://gradio.app/) - Web UI framework
- [FastAPI](https://fastapi.tiangolo.com/) - REST API framework
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Loguru](https://github.com/Delgan/loguru) - Logging library
- [Rich](https://github.com/Textualize/rich) - Terminal formatting

---

<div align="center">

**Built with ❤️ for the Gen AI community**

⭐ Star this repo if you found it helpful!

</div>
```

---