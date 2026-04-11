"""
Core Text-to-Image Pipeline Module
Main pipeline orchestrating all components
"""

import os
import time
import torch
import json
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
from datetime import datetime
from loguru import logger
from PIL import Image

from src.model_loader import ModelLoader
from src.scheduler_manager import SchedulerManager
from src.prompt_engineer import PromptEngineer, PromptResult
from src.image_processor import ImageProcessor
from src.latent_manager import LatentSpaceManager
from src.utils import (
    load_config,
    load_yaml,
    seed_everything,
    get_device,
    get_memory_usage,
    generate_image_hash,
    create_output_dirs,
    format_generation_info,
    setup_logger,
    save_json,
    console
)


class TextToImagePipeline:
    """
    Production-grade Text-to-Image Generation Pipeline.
    Orchestrates model loading, prompt engineering, scheduler management,
    latent space operations, image generation, and post-processing.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        output_dir: str = "./output",
        config: Optional[Dict] = None
    ):
        self.config = config or {}
        self.device = device or get_device()
        self.output_dir = Path(output_dir)
        self.is_setup = False
        self.generation_history = []
        self.generation_count = 0

        self.model_loader = ModelLoader(
            model_id=model_id or self.config.get("model", {}).get("model_id"),
            model_name=model_name,
            device=self.device,
            cache_dir=self.config.get("model", {}).get("cache_dir", "./model_cache")
        )

        self.prompt_engineer = PromptEngineer(
            default_style=self.config.get("prompt", {}).get("default_style", "digital_art"),
            default_quality=self.config.get("prompt", {}).get("default_quality", "high")
        )

        self.image_processor = ImageProcessor(output_dir=str(self.output_dir))
        self.scheduler_manager = None
        self.latent_manager = None
        self.pipe = None

        create_output_dirs(str(self.output_dir))
        logger.info(
            f"TextToImagePipeline initialized | Device: {self.device} | Output: {self.output_dir}"
        )

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "TextToImagePipeline":
        """Create pipeline from configuration file."""
        config = load_config(config_path)
        model_config = config.get("model", {})
        output_config = config.get("output", {})

        instance = cls(
            model_id=model_config.get("model_id"),
            device=model_config.get("device"),
            output_dir=output_config.get("base_dir", "./output"),
            config=config
        )
        logger.info(f"Pipeline created from config: {config_path}")
        return instance

    def setup(
        self,
        pipeline_type: str = "text2img",
        custom_vae: Optional[str] = None,
        scheduler: Optional[str] = None,
        **optimization_kwargs
    ):
        """Complete pipeline setup: load model, apply optimizations, initialize components."""
        logger.info("=" * 50)
        logger.info("Setting up Text-to-Image Pipeline...")
        logger.info("=" * 50)

        start_time = time.time()

        # Step 1: Load model
        self.pipe = self.model_loader.load(
            pipeline_type=pipeline_type,
            custom_vae=custom_vae
        )

        # Step 2: Apply optimizations
        opt_config = self.config.get("optimization", {})
        self.model_loader.optimize(
            enable_xformers=optimization_kwargs.get("enable_xformers", opt_config.get("enable_xformers", True)),
            enable_attention_slicing=optimization_kwargs.get("enable_attention_slicing", opt_config.get("enable_attention_slicing", True)),
            enable_vae_slicing=optimization_kwargs.get("enable_vae_slicing", opt_config.get("enable_vae_slicing", True)),
            enable_vae_tiling=optimization_kwargs.get("enable_vae_tiling", opt_config.get("enable_vae_tiling", False)),
            enable_model_cpu_offload=optimization_kwargs.get("enable_model_cpu_offload", opt_config.get("enable_model_cpu_offload", False)),
            enable_torch_compile=optimization_kwargs.get("enable_torch_compile", opt_config.get("torch_compile", False)),
        )

        # Step 3: Initialize scheduler manager
        self.scheduler_manager = SchedulerManager(self.pipe)
        if scheduler:
            self.scheduler_manager.set_scheduler(scheduler)
        else:
            default_scheduler = self.config.get("scheduler", {}).get("default", "dpm_solver_multistep")
            self.scheduler_manager.set_scheduler(default_scheduler)

        # Step 4: Initialize latent manager
        self.latent_manager = LatentSpaceManager(self.pipe, self.device)

        self.is_setup = True
        elapsed = time.time() - start_time
        logger.info(f"Pipeline setup complete! ({elapsed:.1f}s)")
        logger.info(f"Memory: {get_memory_usage()}")

    def _check_setup(self):
        """Verify pipeline is properly set up."""
        if not self.is_setup:
            raise RuntimeError("Pipeline not set up. Call pipeline.setup() first.")

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style: Optional[str] = None,
        quality: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        num_images: int = 1,
        preset: Optional[str] = None,
        enhance_prompt: bool = True,
        auto_enhance_image: bool = False,
        save: bool = True,
        save_metadata: bool = True,
        filename_prefix: Optional[str] = None,
        additional_positive: Optional[str] = None,
        additional_negative: Optional[str] = None,
        negative_categories: Optional[List[str]] = None,
        emphasis: Optional[Dict[str, float]] = None,
        callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate image(s) from text prompt.

        Returns:
            Dictionary with images, paths, metadata, and timing info.
        """
        self._check_setup()

        # --- RESOLVE SETTINGS ---
        gen_defaults = self.config.get("generation", {}).get("default", {})

        if preset:
            preset_config = self.config.get("generation", {}).get("presets", {}).get(preset, {})
            num_inference_steps = num_inference_steps or preset_config.get("num_inference_steps")
            guidance_scale = guidance_scale or preset_config.get("guidance_scale")
            width = width or preset_config.get("width")
            height = height or preset_config.get("height")
            preset_scheduler = preset_config.get("scheduler")
            if preset_scheduler:
                self.scheduler_manager.set_scheduler(preset_scheduler)

        num_inference_steps = num_inference_steps or gen_defaults.get("num_inference_steps", 30)
        guidance_scale = guidance_scale or gen_defaults.get("guidance_scale", 7.5)
        width = width or gen_defaults.get("width", 512)
        height = height or gen_defaults.get("height", 512)

        # --- PROMPT ENGINEERING ---
        if enhance_prompt:
            prompt_result = self.prompt_engineer.build_prompt(
                base_prompt=prompt,
                style=style,
                quality=quality,
                additional_positive=additional_positive,
                additional_negative=additional_negative,
                negative_categories=negative_categories or ["quality", "watermark", "anatomy"],
                emphasis=emphasis
            )
            final_positive = prompt_result.positive
            final_negative = negative_prompt or prompt_result.negative
        else:
            final_positive = prompt
            final_negative = negative_prompt or ""

        # --- SEED ---
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            seed_everything(seed)

        # --- SETTINGS DICT ---
        settings = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "num_images": num_images,
            "style": style,
            "quality": quality,
            "preset": preset,
            "scheduler": self.scheduler_manager.current_scheduler,
        }

        logger.info("=" * 40)
        logger.info("🎨 Generating Image(s)")
        logger.info(f"  Prompt: {prompt[:60]}...")
        logger.info(f"  Style: {style or 'default'} | Quality: {quality or 'default'}")
        logger.info(f"  Steps: {num_inference_steps} | Guidance: {guidance_scale}")
        logger.info(f"  Size: {width}x{height} | Seed: {seed}")
        logger.info("=" * 40)

        # --- GENERATE ---
        start_time = time.time()

        try:
            result = self.pipe(
                prompt=final_positive,
                negative_prompt=final_negative,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_images_per_prompt=num_images,
                generator=generator,
                callback=callback,
            )
            images = result.images
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

        elapsed = time.time() - start_time

        # --- POST-PROCESS ---
        if auto_enhance_image:
            images = [self.image_processor.auto_enhance(img) for img in images]

        # --- SAVE ---
        saved_paths = []
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = filename_prefix or "generated"

            for i, img in enumerate(images):
                img_hash = generate_image_hash(prompt, seed or 0, settings)
                filename = f"{prefix}_{timestamp}_{img_hash}_{i}.png"

                metadata_dict = {
                    "prompt": prompt,
                    "enhanced_prompt": final_positive,
                    "negative_prompt": final_negative,
                    "settings": settings,
                    "elapsed_time": round(elapsed, 2),
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_loader.model_id,
                    "device": self.device,
                }

                filepath = self.image_processor.save_image(
                    img,
                    filename,
                    subfolder="generated",
                    metadata=metadata_dict if save_metadata else None
                )
                saved_paths.append(filepath)

            # Save metadata JSON
            if save_metadata:
                meta_filename = f"{prefix}_{timestamp}_{img_hash}_metadata.json"
                meta_path = os.path.join(str(self.output_dir), "metadata", meta_filename)
                save_json(metadata_dict, meta_path)

        # --- TRACK HISTORY ---
        generation_record = {
            "id": self.generation_count,
            "prompt": prompt,
            "style": style,
            "quality": quality,
            "settings": settings,
            "elapsed_time": round(elapsed, 2),
            "num_images": len(images),
            "saved_paths": saved_paths,
            "timestamp": datetime.now().isoformat(),
        }
        self.generation_history.append(generation_record)
        self.generation_count += 1

        logger.info(f"✅ Generated {len(images)} image(s) in {elapsed:.2f}s")
        if saved_paths:
            logger.info(f"   Saved to: {saved_paths}")

        return {
            "images": images,
            "paths": saved_paths,
            "prompt": final_positive,
            "negative_prompt": final_negative,
            "settings": settings,
            "elapsed_time": round(elapsed, 2),
            "metadata": generation_record,
        }

    def generate_variations(
        self,
        prompt: str,
        num_variations: int = 4,
        seed_start: int = 42,
        style: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate multiple variations of the same prompt with different seeds.
        """
        self._check_setup()

        all_images = []
        all_paths = []
        all_metadata = []

        total_start = time.time()

        for i in range(num_variations):
            current_seed = seed_start + i
            logger.info(f"\n--- Variation {i + 1}/{num_variations} (seed: {current_seed}) ---")

            result = self.generate(
                prompt=prompt,
                seed=current_seed,
                style=style,
                filename_prefix=f"variation_{i}",
                **kwargs
            )

            all_images.extend(result["images"])
            all_paths.extend(result["paths"])
            all_metadata.append(result["metadata"])

        total_elapsed = time.time() - total_start

        # Create comparison grid
        if len(all_images) > 1:
            labels = [f"Seed: {seed_start + i}" for i in range(num_variations)]
            grid = self.image_processor.create_grid(
                all_images,
                cols=min(4, num_variations),
                labels=labels
            )
            grid_path = self.image_processor.save_image(
                grid,
                f"variations_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                subfolder="grids"
            )
            logger.info(f"Variation grid saved: {grid_path}")
        else:
            grid = None
            grid_path = None

        return {
            "images": all_images,
            "paths": all_paths,
            "grid": grid,
            "grid_path": grid_path,
            "total_time": round(total_elapsed, 2),
            "metadata": all_metadata,
        }

    def generate_style_comparison(
        self,
        prompt: str,
        styles: Optional[List[str]] = None,
        seed: int = 42,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate the same prompt in multiple styles for comparison.
        """
        self._check_setup()

        if styles is None:
            styles = ["photorealistic", "digital_art", "oil_painting", "anime",
                       "cyberpunk", "watercolor", "minimalist", "3d_render"]

        all_images = []
        all_paths = []
        style_results = {}

        total_start = time.time()

        for style in styles:
            logger.info(f"\n--- Style: {style} ---")
            result = self.generate(
                prompt=prompt,
                style=style,
                seed=seed,
                filename_prefix=f"style_{style}",
                **kwargs
            )
            all_images.extend(result["images"])
            all_paths.extend(result["paths"])
            style_results[style] = result

        total_elapsed = time.time() - total_start

        # Create comparison grid
        grid = self.image_processor.create_grid(
            all_images,
            cols=min(4, len(styles)),
            labels=styles
        )
        grid_path = self.image_processor.save_image(
            grid,
            f"style_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            subfolder="comparisons"
        )

        return {
            "images": all_images,
            "paths": all_paths,
            "grid": grid,
            "grid_path": grid_path,
            "style_results": style_results,
            "total_time": round(total_elapsed, 2),
        }

    def generate_quality_comparison(
        self,
        prompt: str,
        step_counts: Optional[List[int]] = None,
        seed: int = 42,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare different inference step counts for quality/speed tradeoff analysis.
        """
        self._check_setup()

        if step_counts is None:
            step_counts = [10, 15, 20, 30, 50, 75]

        all_images = []
        all_paths = []
        timing_data = {}

        total_start = time.time()

        for steps in step_counts:
            logger.info(f"\n--- Steps: {steps} ---")
            result = self.generate(
                prompt=prompt,
                num_inference_steps=steps,
                seed=seed,
                filename_prefix=f"steps_{steps}",
                **kwargs
            )
            all_images.extend(result["images"])
            all_paths.extend(result["paths"])
            timing_data[steps] = result["elapsed_time"]

        total_elapsed = time.time() - total_start

        # Create comparison grid
        labels = [f"{s} steps\n({timing_data[s]:.1f}s)" for s in step_counts]
        grid = self.image_processor.create_grid(
            all_images,
            cols=min(3, len(step_counts)),
            labels=labels
        )
        grid_path = self.image_processor.save_image(
            grid,
            f"quality_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            subfolder="comparisons"
        )

        return {
            "images": all_images,
            "paths": all_paths,
            "grid": grid,
            "grid_path": grid_path,
            "timing_data": timing_data,
            "total_time": round(total_elapsed, 2),
        }

    def generate_latent_interpolation(
        self,
        prompt: str,
        num_frames: int = 10,
        seed_start: int = 42,
        seed_end: int = 123,
        method: str = "slerp",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a latent space interpolation sequence.
        """
        self._check_setup()

        # Generate latent walk
        latent_frames = self.latent_manager.generate_latent_walk(
            num_frames=num_frames,
            height=kwargs.get("height", 512),
            width=kwargs.get("width", 512),
            seed_start=seed_start,
            seed_end=seed_end,
            method=method
        )

        # Build prompt
        prompt_result = self.prompt_engineer.build_prompt(
            base_prompt=prompt,
            style=kwargs.get("style"),
            quality=kwargs.get("quality"),
        )

        # Generate images for each latent frame
        images = []
        paths = []
        num_inference_steps = kwargs.get("num_inference_steps", 30)
        guidance_scale = kwargs.get("guidance_scale", 7.5)

        for i, latent in enumerate(latent_frames):
            logger.info(f"Generating frame {i + 1}/{num_frames}")

            # Scale initial latents
            latent = latent * self.pipe.scheduler.init_noise_sigma

            result = self.pipe(
                prompt=prompt_result.positive,
                negative_prompt=prompt_result.negative,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                latents=latent,
            )

            img = result.images[0]
            images.append(img)

            filepath = self.image_processor.save_image(
                img,
                f"interpolation_frame_{i:04d}.png",
                subfolder="interpolations"
            )
            paths.append(filepath)

        # Create animation grid
        grid = self.image_processor.create_grid(
            images,
            cols=min(5, num_frames),
            labels=[f"Frame {i}" for i in range(num_frames)]
        )
        grid_path = self.image_processor.save_image(
            grid,
            f"interpolation_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            subfolder="interpolations"
        )

        return {
            "images": images,
            "paths": paths,
            "grid": grid,
            "grid_path": grid_path,
            "num_frames": num_frames,
        }

    def save_generation_gif(
        self,
        images: List[Image.Image],
        filename: str = "animation.gif",
        duration: int = 200
    ) -> str:
        """Save a list of images as an animated GIF."""
        filepath = os.path.join(str(self.output_dir), "animations", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        images[0].save(
            filepath,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        logger.info(f"GIF saved: {filepath}")
        return filepath

    def get_history(self) -> List[Dict]:
        """Get generation history."""
        return self.generation_history

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        total_time = sum(r["elapsed_time"] for r in self.generation_history)
        total_images = sum(r["num_images"] for r in self.generation_history)

        return {
            "total_generations": self.generation_count,
            "total_images": total_images,
            "total_time_seconds": round(total_time, 2),
            "average_time_seconds": round(total_time / max(1, self.generation_count), 2),
            "model": self.model_loader.model_id,
            "device": self.device,
            "memory": get_memory_usage(),
        }

    def cleanup(self):
        """Clean up resources and free memory."""
        logger.info("Cleaning up pipeline resources...")
        self.model_loader.unload()
        self.pipe = None
        self.scheduler_manager = None
        self.latent_manager = None
        self.is_setup = False
        logger.info("Pipeline cleaned up successfully")