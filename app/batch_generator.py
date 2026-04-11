"""
Batch Generator Module
Handles bulk image generation from configuration files
"""

import os
import json
import csv
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from loguru import logger

from src.pipeline import TextToImagePipeline
from src.utils import save_json, load_json, load_yaml, console

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn


class BatchGenerator:
    """
    Batch image generation from various input sources.

    Supports:
        - JSON batch configuration
        - CSV batch input
        - YAML batch configuration
        - Parallel generation queues
    """

    def __init__(
        self,
        pipeline: TextToImagePipeline,
        output_dir: str = "./output/batch"
    ):
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_history = []

        logger.info(f"BatchGenerator initialized | Output: {self.output_dir}")

    def generate_from_json(self, json_path: str) -> Dict[str, Any]:
        """
        Generate images from a JSON configuration file.

        Expected JSON format:
        {
            "batch_name": "my_batch",
            "default_settings": {
                "style": "digital_art",
                "quality": "high",
                "num_inference_steps": 30
            },
            "items": [
                {"prompt": "...", "filename": "...", "seed": 42},
                ...
            ]
        }
        """
        config = load_json(json_path)
        batch_name = config.get("batch_name", "json_batch")
        defaults = config.get("default_settings", {})
        items = config.get("items", [])

        return self._run_batch(batch_name, items, defaults)

    def generate_from_csv(
        self,
        csv_path: str,
        batch_name: str = "csv_batch"
    ) -> Dict[str, Any]:
        """
        Generate images from a CSV file.

        Expected CSV columns:
        prompt, filename, style, quality, seed, steps, guidance, width, height
        """
        items = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = {"prompt": row["prompt"]}

                if "filename" in row and row["filename"]:
                    item["filename"] = row["filename"]
                if "style" in row and row["style"]:
                    item["style"] = row["style"]
                if "quality" in row and row["quality"]:
                    item["quality"] = row["quality"]
                if "seed" in row and row["seed"]:
                    item["seed"] = int(row["seed"])
                if "steps" in row and row["steps"]:
                    item["num_inference_steps"] = int(row["steps"])
                if "guidance" in row and row["guidance"]:
                    item["guidance_scale"] = float(row["guidance"])
                if "width" in row and row["width"]:
                    item["width"] = int(row["width"])
                if "height" in row and row["height"]:
                    item["height"] = int(row["height"])

                items.append(item)

        return self._run_batch(batch_name, items, {})

    def generate_from_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """Generate images from a YAML configuration file."""
        config = load_yaml(yaml_path)
        batch_name = config.get("batch_name", "yaml_batch")
        defaults = config.get("default_settings", {})
        items = config.get("items", [])

        return self._run_batch(batch_name, items, defaults)

    def generate_from_prompts(
        self,
        prompts: List[str],
        batch_name: str = "quick_batch",
        style: Optional[str] = None,
        quality: Optional[str] = None,
        seed_start: int = 42,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Quick batch generation from a list of prompts.
        """
        items = []
        for i, prompt in enumerate(prompts):
            items.append({
                "prompt": prompt,
                "filename": f"image_{i:04d}",
                "seed": seed_start + i,
            })

        defaults = {}
        if style:
            defaults["style"] = style
        if quality:
            defaults["quality"] = quality
        defaults.update(kwargs)

        return self._run_batch(batch_name, items, defaults)

    def _run_batch(
        self,
        batch_name: str,
        items: List[Dict],
        defaults: Dict
    ) -> Dict[str, Any]:
        """
        Execute batch generation.

        Args:
            batch_name: Name for this batch
            items: List of generation configurations
            defaults: Default settings for all items
        """
        batch_dir = self.output_dir / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"🚀 Starting Batch: {batch_name}")
        logger.info(f"   Items: {len(items)} | Defaults: {defaults}")
        logger.info("=" * 60)

        results = []
        failed = []
        total_time = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Batch: {batch_name}", total=len(items))

            for i, item in enumerate(items):
                # Merge defaults with item-specific settings
                gen_config = {**defaults, **item}
                prompt = gen_config.pop("prompt")
                filename = gen_config.pop("filename", f"batch_{i:04d}")

                try:
                    gen_result = self.pipeline.generate(
                        prompt=prompt,
                        save=True,
                        filename_prefix=filename,
                        **gen_config
                    )

                    results.append({
                        "index": i,
                        "filename": filename,
                        "prompt": prompt,
                        "paths": gen_result["paths"],
                        "elapsed_time": gen_result["elapsed_time"],
                        "status": "success",
                    })
                    total_time += gen_result["elapsed_time"]

                except Exception as e:
                    logger.error(f"Failed to generate item {i} ({filename}): {e}")
                    failed.append({
                        "index": i,
                        "filename": filename,
                        "prompt": prompt,
                        "error": str(e),
                        "status": "failed",
                    })

                progress.update(task, advance=1)

        # Batch summary
        batch_summary = {
            "batch_name": batch_name,
            "total_items": len(items),
            "successful": len(results),
            "failed": len(failed),
            "total_time_seconds": round(total_time, 2),
            "average_time_seconds": round(total_time / max(1, len(results)), 2),
            "created_at": datetime.now().isoformat(),
            "results": results,
            "failures": failed,
        }

        # Save batch report
        report_path = os.path.join(str(batch_dir), "batch_report.json")
        save_json(batch_summary, report_path)

        self.batch_history.append(batch_summary)

        logger.info(f"\n✅ Batch '{batch_name}' complete!")
        logger.info(f"   Success: {len(results)}/{len(items)} | Time: {total_time:.1f}s")

        if failed:
            logger.warning(f"   Failed: {len(failed)} items")

        return batch_summary