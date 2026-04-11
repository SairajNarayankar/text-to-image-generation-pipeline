"""
Text-to-Image Generation Pipeline
==================================
A production-grade generative AI pipeline using Stable Diffusion
and Hugging Face Diffusers for converting natural language prompts
into high-resolution images.

Modules:
    - pipeline: Core Stable Diffusion pipeline
    - prompt_engineer: Prompt optimization and engineering
    - scheduler_manager: Scheduler configurations
    - image_processor: Post-processing utilities
    - latent_manager: Latent space operations
    - model_loader: Model loading and management
    - utils: Helper functions
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.pipeline import TextToImagePipeline
from src.prompt_engineer import PromptEngineer
from src.scheduler_manager import SchedulerManager
from src.image_processor import ImageProcessor
from src.latent_manager import LatentSpaceManager
from src.model_loader import ModelLoader
from src.utils import setup_logger, load_config, seed_everything

__all__ = [
    "TextToImagePipeline",
    "PromptEngineer",
    "SchedulerManager",
    "ImageProcessor",
    "LatentSpaceManager",
    "ModelLoader",
    "setup_logger",
    "load_config",
    "seed_everything",
]