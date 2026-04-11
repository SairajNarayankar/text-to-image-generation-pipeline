"""
Model Loader Module
Handles downloading, caching, and loading of Stable Diffusion models
"""

import os
import gc
import torch
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    AutoPipelineForText2Image,
)
from diffusers.models import AutoencoderKL


class ModelLoader:
    """
    Manages loading, caching, and optimization of Stable Diffusion models.

    Supports:
        - Multiple model variants (SD 1.5, SD 2.1, SDXL)
        - Custom VAE loading
        - Memory-efficient loading strategies
        - Model caching
    """

    # Supported model registry
    MODEL_REGISTRY = {
        "sd-1.5": {
            "model_id": "runwayml/stable-diffusion-v1-5",
            "description": "Stable Diffusion v1.5 - Most widely used",
            "resolution": 512,
            "vram_required_gb": 4
        },
        "sd-2.1": {
            "model_id": "stabilityai/stable-diffusion-2-1",
            "description": "Stable Diffusion v2.1 - Improved quality",
            "resolution": 768,
            "vram_required_gb": 5
        },
        "sdxl": {
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "description": "Stable Diffusion XL - Highest quality",
            "resolution": 1024,
            "vram_required_gb": 8
        },
        "sdxl-turbo": {
            "model_id": "stabilityai/sdxl-turbo",
            "description": "SDXL Turbo - Fast generation",
            "resolution": 512,
            "vram_required_gb": 6
        },
        "dreamshaper": {
            "model_id": "Lykon/dreamshaper-8",
            "description": "DreamShaper v8 - Creative/artistic",
            "resolution": 512,
            "vram_required_gb": 4
        },
        "realistic-vision": {
            "model_id": "SG161222/Realistic_Vision_V5.1_noVAE",
            "description": "Realistic Vision - Photorealistic focus",
            "resolution": 512,
            "vram_required_gb": 4
        }
    }

    # Custom VAE models for better quality
    VAE_REGISTRY = {
        "sd-vae-ft-mse": "stabilityai/sd-vae-ft-mse",
        "sd-vae-ft-ema": "stabilityai/sd-vae-ft-ema",
        "sdxl-vae": "madebyollin/sdxl-vae-fp16-fix"
    }

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_name: Optional[str] = None,
        cache_dir: str = "./model_cache",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = None,
    ):
        """
        Initialize ModelLoader

        Args:
            model_id: Direct Hugging Face model ID
            model_name: Shortcut name from MODEL_REGISTRY
            cache_dir: Directory for model caching
            device: Device to load model on
            torch_dtype: Precision for model weights
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or self._detect_device()
    
    # FIX: Auto-select dtype based on device
        if torch_dtype:
            self.torch_dtype = torch_dtype
        elif self.device == "cuda":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # Resolve model ID
        if model_name and model_name in self.MODEL_REGISTRY:
            self.model_info = self.MODEL_REGISTRY[model_name]
            self.model_id = self.model_info["model_id"]
        elif model_id:
            self.model_id = model_id
            self.model_info = {"model_id": model_id, "description": "Custom model"}
        else:
            self.model_id = self.MODEL_REGISTRY["sd-1.5"]["model_id"]
            self.model_info = self.MODEL_REGISTRY["sd-1.5"]

        logger.info(f"ModelLoader initialized | Model: {self.model_id} | Device: {self.device}")

    def _detect_device(self) -> str:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _check_vram(self) -> bool:
        """Check if sufficient VRAM is available"""
        if self.device != "cuda":
            return True

        available_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        required_gb = self.model_info.get("vram_required_gb", 4) if self.model_info else 4

        if available_gb < required_gb:
            logger.warning(
                f"Low VRAM: {available_gb:.1f} GB available, "
                f"{required_gb} GB recommended. "
                f"Enabling memory optimizations."
            )
            return False
        return True

    def load(
        self,
        pipeline_type: str = "text2img",
        custom_vae: Optional[str] = None,
        enable_safety_checker: bool = False,
        **kwargs
    ) -> "StableDiffusionPipeline":
        """
        Load the Stable Diffusion pipeline

        Args:
            pipeline_type: Type of pipeline (text2img, img2img, inpaint)
            custom_vae: Name of custom VAE from VAE_REGISTRY
            enable_safety_checker: Whether to enable NSFW filter
            **kwargs: Additional arguments for from_pretrained

        Returns:
            Loaded pipeline
        """
        logger.info(f"Loading model: {self.model_id} ({pipeline_type})")

        # Select pipeline class
        pipeline_classes = {
            "text2img": StableDiffusionPipeline,
            "img2img": StableDiffusionImg2ImgPipeline,
            "inpaint": StableDiffusionInpaintPipeline,
            "auto": AutoPipelineForText2Image,
        }

        PipelineClass = pipeline_classes.get(pipeline_type, StableDiffusionPipeline)

        # Build loading arguments
        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "cache_dir": str(self.cache_dir),
            "safety_checker": None if not enable_safety_checker else "auto",
            "requires_safety_checker": enable_safety_checker,
        }
        load_kwargs.update(kwargs)

        # Handle safety checker properly
        if not enable_safety_checker:
            load_kwargs["safety_checker"] = None
            load_kwargs["requires_safety_checker"] = False

        # Load custom VAE if specified
        vae = None
        if custom_vae and custom_vae in self.VAE_REGISTRY:
            logger.info(f"Loading custom VAE: {custom_vae}")
            vae = AutoencoderKL.from_pretrained(
                self.VAE_REGISTRY[custom_vae],
                torch_dtype=self.torch_dtype,
                cache_dir=str(self.cache_dir)
            )
            load_kwargs["vae"] = vae

        # Load the pipeline
        try:
            self.pipe = PipelineClass.from_pretrained(
                self.model_id,
                **load_kwargs
            )
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Move to device
        self.pipe = self.pipe.to(self.device)
        logger.info(f"Model moved to: {self.device}")

        return self.pipe

    def optimize(
        self,
        enable_xformers: bool = True,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        enable_sequential_cpu_offload: bool = False,
        enable_torch_compile: bool = False,
    ):
        """
        Apply various memory and speed optimizations

        Args:
            enable_xformers: Use xformers for memory-efficient attention
            enable_attention_slicing: Slice attention for lower VRAM
            enable_vae_slicing: Process VAE in slices
            enable_vae_tiling: Use tiling for very large images
            enable_model_cpu_offload: Offload model parts to CPU
            enable_sequential_cpu_offload: Aggressive CPU offloading
            enable_torch_compile: Use torch.compile for speed
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        logger.info("Applying optimizations...")

        # xformers - Most impactful optimization
        if enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("✅ xformers memory efficient attention enabled")
            except Exception as e:
                logger.warning(f"⚠️ xformers not available: {e}")

        # Attention slicing
        if enable_attention_slicing:
            self.pipe.enable_attention_slicing(slice_size="auto")
            logger.info("✅ Attention slicing enabled")

        # VAE slicing (helps with batch processing)
        if enable_vae_slicing:
            self.pipe.enable_vae_slicing()
            logger.info("✅ VAE slicing enabled")

        # VAE tiling (for very large images)
        if enable_vae_tiling:
            self.pipe.enable_vae_tiling()
            logger.info("✅ VAE tiling enabled")

        # CPU offloading strategies
        if enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
            logger.info("✅ Sequential CPU offload enabled (max memory savings)")
        elif enable_model_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            logger.info("✅ Model CPU offload enabled")

        # Torch compile (PyTorch 2.0+)
        if enable_torch_compile:
            try:
                self.pipe.unet = torch.compile(
                    self.pipe.unet,
                    mode="reduce-overhead",
                    fullgraph=True
                )
                logger.info("✅ torch.compile enabled for UNet")
            except Exception as e:
                logger.warning(f"⚠️ torch.compile failed: {e}")

        logger.info("All optimizations applied!")

    def unload(self):
        """Unload model and free memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Model unloaded and memory freed")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "loaded": self.pipe is not None,
        }

        if self.model_info:
            info.update(self.model_info)

        if self.pipe is not None and self.device == "cuda":
            info["memory_allocated_gb"] = round(
                torch.cuda.memory_allocated() / (1024 ** 3), 2
            )

        return info

    @classmethod
    def list_available_models(cls):
        """List all available models in the registry"""
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title="🤖 Available Models")
        table.add_column("Name", style="cyan")
        table.add_column("Model ID", style="green")
        table.add_column("Resolution", style="yellow")
        table.add_column("VRAM (GB)", style="red")
        table.add_column("Description", style="dim")

        for name, info in cls.MODEL_REGISTRY.items():
            table.add_row(
                name,
                info["model_id"],
                str(info.get("resolution", "N/A")),
                str(info.get("vram_required_gb", "N/A")),
                info.get("description", "")
            )

        console.print(table)