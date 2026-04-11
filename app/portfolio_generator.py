"""
Portfolio Generator Module
Creates themed collections of AI-generated assets for different business use cases
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from loguru import logger
from PIL import Image

from src.pipeline import TextToImagePipeline
from src.image_processor import ImageProcessor
from src.utils import save_json, console

from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn


class PortfolioGenerator:
    """
    Generates themed portfolios of AI-created images
    for marketing, creative design, and rapid prototyping.
    """

    def __init__(
        self,
        pipeline: TextToImagePipeline,
        output_dir: str = "./output/portfolio"
    ):
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_processor = ImageProcessor(output_dir=str(self.output_dir))
        self.collections = {}
        self.total_generated = 0

        logger.info(f"PortfolioGenerator initialized | Output: {self.output_dir}")

    def generate_collection(
        self,
        collection_name: str,
        items: List[Dict[str, Any]],
        create_grid: bool = True,
        create_web_exports: bool = False,
        add_watermark: bool = False,
        watermark_text: str = "AI Generated"
    ) -> Dict[str, Any]:
        """
        Generate a themed collection of images.

        Args:
            collection_name: Name for the collection
            items: List of generation configs, each containing:
                - prompt: str
                - filename: str
                - style: str (optional)
                - quality: str (optional)
                - seed: int (optional)
                - num_inference_steps: int (optional)
                - guidance_scale: float (optional)
                - width: int (optional)
                - height: int (optional)
            create_grid: Whether to create a grid overview
            create_web_exports: Whether to export web-optimized versions
            add_watermark: Whether to add watermark
            watermark_text: Watermark text

        Returns:
            Collection results dictionary
        """
        collection_dir = self.output_dir / collection_name
        collection_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"📁 Generating Collection: {collection_name}")
        logger.info(f"   Items: {len(items)}")
        logger.info("=" * 60)

        results = []
        images = []
        labels = []
        total_time = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Generating {collection_name}...",
                total=len(items)
            )

            for i, item in enumerate(items):
                prompt = item["prompt"]
                filename = item.get("filename", f"image_{i:03d}")

                logger.info(f"\n[{i + 1}/{len(items)}] {filename}")

                # Generate image
                gen_result = self.pipeline.generate(
                    prompt=prompt,
                    style=item.get("style"),
                    quality=item.get("quality", "high"),
                    seed=item.get("seed"),
                    num_inference_steps=item.get("num_inference_steps", 30),
                    guidance_scale=item.get("guidance_scale", 7.5),
                    width=item.get("width", 512),
                    height=item.get("height", 512),
                    save=False,  # We'll handle saving ourselves
                    enhance_prompt=item.get("enhance_prompt", True),
                )

                image = gen_result["images"][0]
                elapsed = gen_result["elapsed_time"]
                total_time += elapsed

                # Post-processing
                if item.get("auto_enhance", False):
                    image = self.image_processor.auto_enhance(image)

                if add_watermark:
                    image = self.image_processor.add_watermark(
                        image, text=watermark_text
                    )

                # Save image
                filepath = self.image_processor.save_image(
                    image,
                    f"{filename}.png",
                    subfolder=collection_name,
                    metadata={
                        "prompt": prompt,
                        "style": item.get("style", "default"),
                        "seed": item.get("seed"),
                        "collection": collection_name,
                    }
                )

                # Web exports
                web_paths = {}
                if create_web_exports:
                    web_paths = self.image_processor.export_for_web(
                        image, f"{collection_name}_{filename}"
                    )

                images.append(image)
                labels.append(filename)

                results.append({
                    "filename": filename,
                    "prompt": prompt,
                    "style": item.get("style"),
                    "filepath": filepath,
                    "web_exports": web_paths,
                    "elapsed_time": elapsed,
                    "seed": item.get("seed"),
                })

                progress.update(task, advance=1)
                self.total_generated += 1

        # Create grid
        grid_path = None
        if create_grid and len(images) > 1:
            grid = self.image_processor.create_grid(
                images,
                cols=min(4, len(images)),
                labels=labels
            )
            grid_path = self.image_processor.save_image(
                grid,
                f"{collection_name}_grid.png",
                subfolder=collection_name
            )
            logger.info(f"📊 Grid saved: {grid_path}")

        # Save collection metadata
        collection_metadata = {
            "collection_name": collection_name,
            "num_items": len(results),
            "total_time_seconds": round(total_time, 2),
            "average_time_seconds": round(total_time / max(1, len(results)), 2),
            "created_at": datetime.now().isoformat(),
            "grid_path": grid_path,
            "items": results,
        }

        meta_path = os.path.join(str(collection_dir), "collection_metadata.json")
        save_json(collection_metadata, meta_path)

        self.collections[collection_name] = collection_metadata

        logger.info(f"\n✅ Collection '{collection_name}' complete!")
        logger.info(f"   Images: {len(results)} | Time: {total_time:.1f}s")

        return collection_metadata

    def generate_marketing_portfolio(self, seed: int = 42) -> Dict[str, Any]:
        """Generate a complete marketing assets portfolio."""
        items = [
            {
                "prompt": "Modern wireless headphones product shot, studio lighting, white background, commercial photography",
                "filename": "headphones_product",
                "style": "photorealistic",
                "seed": seed,
                "quality": "ultra"
            },
            {
                "prompt": "Artisan coffee brand social media post, warm cozy atmosphere, lifestyle photography, morning light",
                "filename": "coffee_social_media",
                "style": "photorealistic",
                "seed": seed + 1,
                "quality": "high"
            },
            {
                "prompt": "Tech startup website hero image, abstract flowing data visualization, blue and purple gradient",
                "filename": "tech_hero_banner",
                "style": "digital_art",
                "seed": seed + 2,
                "width": 768,
                "height": 512
            },
            {
                "prompt": "Luxury watch advertisement, elegant product shot, dark moody background, dramatic side lighting",
                "filename": "watch_advertisement",
                "style": "cinematic",
                "seed": seed + 3,
                "quality": "ultra"
            },
            {
                "prompt": "Organic skincare products flatlay, natural ingredients, soft natural lighting, clean aesthetic",
                "filename": "skincare_flatlay",
                "style": "photorealistic",
                "seed": seed + 4,
            },
            {
                "prompt": "Food delivery app promotional image, colorful fresh meals, vibrant and appetizing, top view",
                "filename": "food_delivery_promo",
                "style": "photorealistic",
                "seed": seed + 5,
            },
        ]

        return self.generate_collection(
            "marketing_assets",
            items,
            create_grid=True,
            create_web_exports=True,
            add_watermark=True
        )

    def generate_creative_portfolio(self, seed: int = 100) -> Dict[str, Any]:
        """Generate a creative design portfolio."""
        items = [
            {
                "prompt": "Epic fantasy book cover, dragon soaring over ancient castle, storm clouds, lightning",
                "filename": "fantasy_book_cover",
                "style": "fantasy",
                "seed": seed,
                "quality": "ultra"
            },
            {
                "prompt": "Vintage travel poster for Tokyo Japan, Mount Fuji, cherry blossoms, retro illustration",
                "filename": "tokyo_travel_poster",
                "style": "vintage",
                "seed": seed + 1,
                "height": 768,
                "width": 512
            },
            {
                "prompt": "Abstract geometric wall art, vibrant complementary colors, modern contemporary, large format",
                "filename": "abstract_wall_art",
                "style": "minimalist",
                "seed": seed + 2,
            },
            {
                "prompt": "Cyberpunk city street scene, neon signs, rain reflections, futuristic vehicles, atmospheric",
                "filename": "cyberpunk_city",
                "style": "cyberpunk",
                "seed": seed + 3,
                "quality": "ultra"
            },
            {
                "prompt": "Watercolor landscape painting, serene mountain lake, autumn colors, misty morning",
                "filename": "watercolor_landscape",
                "style": "watercolor",
                "seed": seed + 4,
            },
            {
                "prompt": "Studio Ghibli inspired scene, magical forest with spirits, soft lighting, whimsical",
                "filename": "ghibli_forest",
                "style": "anime",
                "seed": seed + 5,
            },
        ]

        return self.generate_collection(
            "creative_design",
            items,
            create_grid=True,
            create_web_exports=True
        )

    def generate_prototyping_portfolio(self, seed: int = 200) -> Dict[str, Any]:
        """Generate a rapid prototyping portfolio."""
        items = [
            {
                "prompt": "Modern mobile fitness app UI design, clean interface, activity tracking dashboard, iOS style",
                "filename": "fitness_app_ui",
                "style": "minimalist",
                "seed": seed,
            },
            {
                "prompt": "Eco-friendly modern house exterior, solar panels, garden roof, wooden accents, golden hour",
                "filename": "eco_house_concept",
                "style": "photorealistic",
                "seed": seed + 1,
                "quality": "ultra"
            },
            {
                "prompt": "Futuristic electric car concept design, sleek aerodynamic body, metallic finish, studio shot",
                "filename": "ev_car_concept",
                "style": "3d_render",
                "seed": seed + 2,
            },
            {
                "prompt": "Modern restaurant interior design, industrial chic, exposed brick, warm ambient lighting",
                "filename": "restaurant_interior",
                "style": "photorealistic",
                "seed": seed + 3,
            },
            {
                "prompt": "Futuristic smartwatch design, holographic display, minimal bezel, titanium body",
                "filename": "smartwatch_concept",
                "style": "3d_render",
                "seed": seed + 4,
            },
            {
                "prompt": "Sustainable fashion collection, recycled materials, earth tones, editorial photography",
                "filename": "sustainable_fashion",
                "style": "cinematic",
                "seed": seed + 5,
            },
        ]

        return self.generate_collection(
            "rapid_prototyping",
            items,
            create_grid=True,
            create_web_exports=True
        )

    def generate_full_portfolio(self, seed: int = 42) -> Dict[str, Any]:
        """Generate all portfolio collections."""
        logger.info("\n" + "🎨" * 25)
        logger.info("GENERATING FULL PORTFOLIO")
        logger.info("🎨" * 25 + "\n")

        start_time = time.time()

        marketing = self.generate_marketing_portfolio(seed=seed)
        creative = self.generate_creative_portfolio(seed=seed + 100)
        prototyping = self.generate_prototyping_portfolio(seed=seed + 200)

        total_time = time.time() - start_time

        summary = {
            "total_images": self.total_generated,
            "total_time_seconds": round(total_time, 2),
            "collections": {
                "marketing_assets": marketing,
                "creative_design": creative,
                "rapid_prototyping": prototyping,
            },
            "created_at": datetime.now().isoformat(),
        }

        # Save portfolio summary
        summary_path = os.path.join(str(self.output_dir), "portfolio_summary.json")
        save_json(summary, summary_path)

        self._print_portfolio_summary(summary)

        return summary

    def _print_portfolio_summary(self, summary: Dict):
        """Print formatted portfolio summary."""
        table = Table(title="📊 Portfolio Generation Summary")
        table.add_column("Collection", style="cyan")
        table.add_column("Images", style="green")
        table.add_column("Time (s)", style="yellow")
        table.add_column("Avg Time", style="magenta")

        for name, data in summary["collections"].items():
            table.add_row(
                name,
                str(data["num_items"]),
                f"{data['total_time_seconds']:.1f}",
                f"{data['average_time_seconds']:.1f}"
            )

        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{summary['total_images']}[/bold]",
            f"[bold]{summary['total_time_seconds']:.1f}[/bold]",
            "",
            style="bold"
        )

        console.print(table)