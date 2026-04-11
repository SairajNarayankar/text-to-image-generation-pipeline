# 🏗️ Architecture Documentation

## System Architecture
┌─────────────────────────────────────────────────────┐
│ APPLICATION LAYER │
│ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌───────┐ │
│ │ Gradio │ │ FastAPI │ │ CLI │ │ Batch │ │
│ │ Web UI │ │ API │ │ (Click)│ │ Gen │ │
│ └────┬─────┘ └────┬─────┘ └───┬────┘ └──┬────┘ │
│ │ │ │ │ │
├───────┴──────────────┴────────────┴───────────┴──────┤
│ CORE PIPELINE │
│ ┌────────────────────────────────────────────────┐ │
│ │ TextToImagePipeline │ │
│ │ (Orchestrates all components below) │ │
│ └──────────┬─────────────────────────┬───────────┘ │
│ │ │ │
├─────────────┼─────────────────────────┼───────────────┤
│ ▼ COMPONENTS ▼ │
│ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ │
│ │ ModelLoader │ │ Prompt │ │ Scheduler │ │
│ │ │ │ Engineer │ │ Manager │ │
│ └──────────────┘ └──────────────┘ └────────────┘ │
│ ┌──────────────┐ ┌──────────────┐ │
│ │ LatentSpace │ │ Image │ │
│ │ Manager │ │ Processor │ │
│ └──────────────┘ └──────────────┘ │
│ │
├───────────────────────────────────────────────────────┤
│ FOUNDATION LAYER │
│ ┌───────────┐ ┌─────────────┐ ┌────────────────┐ │
│ │ Diffusers │ │Transformers │ │ PyTorch │ │
│ │ Library │ │ Library │ │ (CUDA/MPS) │ │
│ └───────────┘ └─────────────┘ └────────────────┘ │
└───────────────────────────────────────────────────────┘


## Component Responsibilities

### ModelLoader (`src/model_loader.py`)
- Downloads and caches SD models
- Supports multiple model variants (SD 1.5, 2.1, SDXL)
- Applies memory optimizations (xformers, attention slicing, VAE slicing)
- Custom VAE loading

### PromptEngineer (`src/prompt_engineer.py`)
- 14 built-in style presets
- 4 quality levels (draft → ultra)
- Prompt analysis and scoring
- Token estimation
- Emphasis weighting syntax
- Template system with variable substitution

### SchedulerManager (`src/scheduler_manager.py`)
- 14 noise schedulers
- Speed vs quality recommendations
- Runtime scheduler switching

### LatentSpaceManager (`src/latent_manager.py`)
- Latent noise creation
- Image encoding/decoding via VAE
- SLERP and linear interpolation
- Latent arithmetic
- Latent walk generation for animations

### ImageProcessor (`src/image_processor.py`)
- Enhancement (sharpness, contrast, brightness, saturation)
- Filters (blur, sharpen, detail)
- Resize, crop, upscale
- Grid and comparison creation
- Watermarking
- Multi-size web export
- Batch processing pipeline

### TextToImagePipeline (`src/pipeline.py`)
- Orchestrates all components
- Generation with full parameter control
- Variation generation
- Style comparison
- Quality/speed benchmarking
- Latent interpolation sequences
- GIF animation export
- Generation history tracking

## Data Flow

User Input (prompt + settings)
│
▼
PromptEngineer.build_prompt()
│ (enhanced positive + negative prompts)
▼
SchedulerManager.set_scheduler()
│ (configure noise scheduler)
▼
LatentSpaceManager.create_latent_noise()
│ (initial random latents)
▼
StableDiffusionPipeline.call()
│ (iterative denoising in latent space)
▼
VAE Decoder (latent → pixel space)
│
▼
ImageProcessor (post-processing)
│
▼
Output (save image + metadata)

undefined