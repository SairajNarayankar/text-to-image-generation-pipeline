"""
Benchmark Script
Test pipeline performance with various configurations

File: scripts/benchmark.py
Run: python scripts/benchmark.py
"""

import os
import sys
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
from rich.table import Table
from rich.console import Console

from src.pipeline import TextToImagePipeline
from src.utils import setup_logger, get_memory_usage, save_json

console = Console()


@click.command()
@click.option("--config", "-c", default="config.yaml", help="Config file path")
@click.option("--prompt", "-p", default="A beautiful mountain landscape at sunset, photorealistic",
              help="Test prompt")
@click.option("--runs", "-r", default=3, help="Number of runs per configuration")
@click.option("--output", "-o", default="./output/benchmark", help="Output directory")
def benchmark(config, prompt, runs, output):
    """Run performance benchmarks on the pipeline."""

    setup_logger()
    os.makedirs(output, exist_ok=True)

    console.print("\n[bold cyan]🏎️ Pipeline Benchmark[/bold cyan]\n")

    # Setup pipeline
    pipeline = TextToImagePipeline.from_config(config)
    pipeline.setup()

    # Benchmark configurations
    configs = {
        "Draft (10 steps, 384px)": {"num_inference_steps": 10, "width": 384, "height": 384},
        "Fast (15 steps, 512px)": {"num_inference_steps": 15, "width": 512, "height": 512},
        "Balanced (30 steps, 512px)": {"num_inference_steps": 30, "width": 512, "height": 512},
        "High Quality (50 steps, 512px)": {"num_inference_steps": 50, "width": 512, "height": 512},
        "High Res (30 steps, 768px)": {"num_inference_steps": 30, "width": 768, "height": 768},
    }

    scheduler_configs = {
        "DPM Solver Multistep": "dpm_solver_multistep",
        "Euler Ancestral": "euler_ancestral",
        "Euler": "euler",
        "UniPC": "unipc",
        "DDIM": "ddim",
    }

    results = []

    # --- STEP/RESOLUTION BENCHMARK ---
    console.print("[bold]Phase 1: Step Count & Resolution Benchmark[/bold]\n")

    table = Table(title="Step/Resolution Benchmark")
    table.add_column("Config", style="cyan")
    table.add_column("Avg Time (s)", style="green")
    table.add_column("Min Time (s)", style="yellow")
    table.add_column("Max Time (s)", style="red")
    table.add_column("Images/min", style="magenta")

    for config_name, config_params in configs.items():
        times = []

        for run in range(runs):
            result = pipeline.generate(
                prompt=prompt,
                seed=42,
                save=False,
                enhance_prompt=True,
                style="photorealistic",
                **config_params
            )
            times.append(result["elapsed_time"])

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        imgs_per_min = 60 / avg_time

        table.add_row(
            config_name,
            f"{avg_time:.2f}",
            f"{min_time:.2f}",
            f"{max_time:.2f}",
            f"{imgs_per_min:.1f}"
        )

        results.append({
            "config": config_name,
            "params": config_params,
            "avg_time": round(avg_time, 3),
            "min_time": round(min_time, 3),
            "max_time": round(max_time, 3),
            "images_per_minute": round(imgs_per_min, 1),
        })

    console.print(table)

    # --- SCHEDULER BENCHMARK ---
    console.print("\n[bold]Phase 2: Scheduler Benchmark (30 steps, 512px)[/bold]\n")

    scheduler_table = Table(title="Scheduler Benchmark")
    scheduler_table.add_column("Scheduler", style="cyan")
    scheduler_table.add_column("Avg Time (s)", style="green")
    scheduler_table.add_column("Images/min", style="magenta")

    for sched_name, sched_id in scheduler_configs.items():
        pipeline.scheduler_manager.set_scheduler(sched_id)
        times = []

        for run in range(runs):
            result = pipeline.generate(
                prompt=prompt,
                seed=42,
                save=False,
                num_inference_steps=30,
                width=512,
                height=512,
            )
            times.append(result["elapsed_time"])

        avg_time = sum(times) / len(times)
        imgs_per_min = 60 / avg_time

        scheduler_table.add_row(
            sched_name,
            f"{avg_time:.2f}",
            f"{imgs_per_min:.1f}"
        )

        results.append({
            "config": f"Scheduler: {sched_name}",
            "scheduler": sched_id,
            "avg_time": round(avg_time, 3),
            "images_per_minute": round(imgs_per_min, 1),
        })

    console.print(scheduler_table)

    # --- MEMORY BENCHMARK ---
    console.print("\n[bold]Phase 3: Memory Usage[/bold]\n")
    mem = get_memory_usage()
    mem_table = Table(title="GPU Memory Usage")
    mem_table.add_column("Metric", style="cyan")
    mem_table.add_column("Value", style="green")

    for key, value in mem.items():
        mem_table.add_row(key, f"{value:.2f} GB")

    console.print(mem_table)

    # --- SAVE REPORT ---
    report = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "runs_per_config": runs,
        "results": results,
        "memory": mem,
        "model": pipeline.model_loader.model_id,
        "device": pipeline.device,
    }

    report_path = os.path.join(output, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_json(report, report_path)
    console.print(f"\n📁 Report saved: {report_path}")

    pipeline.cleanup()


if __name__ == "__main__":
    benchmark()