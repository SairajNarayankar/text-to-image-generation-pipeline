"""
Main Application Entry Point
CLI interface for the Text-to-Image Pipeline

File: app/main.py
"""

import os
import sys
import click
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import TextToImagePipeline
from src.utils import setup_logger, print_system_info, load_config
from src.model_loader import ModelLoader
from src.scheduler_manager import SchedulerManager
from src.prompt_engineer import PromptEngineer


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, config, verbose):
    """🎨 Text-to-Image Generation Pipeline CLI"""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose

    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(log_level=log_level)


@cli.command()
def info():
    """Display system and environment information."""
    print_system_info()
    ModelLoader.list_available_models()
    SchedulerManager.list_schedulers()
    PromptEngineer.list_styles()


@cli.command()
@click.argument("prompt")
@click.option("--style", "-s", default=None, help="Style preset")
@click.option("--quality", "-q", default="high", help="Quality level")
@click.option("--steps", "-n", default=30, help="Inference steps")
@click.option("--guidance", "-g", default=7.5, help="Guidance scale")
@click.option("--width", "-W", default=512, help="Image width")
@click.option("--height", "-H", default=512, help="Image height")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--scheduler", default=None, help="Scheduler name")
@click.option("--model", "-m", default=None, help="Model name or ID")
@click.option("--output", "-o", default="./output", help="Output directory")
@click.option("--no-enhance", is_flag=True, help="Disable prompt enhancement")
@click.pass_context
def generate(ctx, prompt, style, quality, steps, guidance, width, height,
             seed, scheduler, model, output, no_enhance):
    """Generate an image from a text prompt."""
    config_path = ctx.obj["config_path"]

    pipeline = TextToImagePipeline.from_config(config_path)

    if model:
        pipeline.model_loader = ModelLoader(model_name=model, device=pipeline.device)

    pipeline.setup(scheduler=scheduler)

    result = pipeline.generate(
        prompt=prompt,
        style=style,
        quality=quality,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        seed=seed,
        enhance_prompt=not no_enhance,
        save=True,
    )

    click.echo(f"\n✅ Generated in {result['elapsed_time']:.2f}s")
    click.echo(f"📁 Saved to: {result['paths']}")

    pipeline.cleanup()


@cli.command()
@click.argument("prompt")
@click.option("--num", "-n", default=4, help="Number of variations")
@click.option("--style", "-s", default=None, help="Style preset")
@click.option("--seed-start", default=42, help="Starting seed")
@click.pass_context
def variations(ctx, prompt, num, style, seed_start):
    """Generate multiple variations of a prompt."""
    config_path = ctx.obj["config_path"]

    pipeline = TextToImagePipeline.from_config(config_path)
    pipeline.setup()

    result = pipeline.generate_variations(
        prompt=prompt,
        num_variations=num,
        seed_start=seed_start,
        style=style,
    )

    click.echo(f"\n✅ Generated {num} variations in {result['total_time']:.2f}s")
    click.echo(f"📊 Grid: {result['grid_path']}")

    pipeline.cleanup()


@cli.command()
@click.argument("prompt")
@click.option("--styles", "-s", multiple=True, help="Styles to compare")
@click.option("--seed", default=42, help="Seed for comparison")
@click.pass_context
def compare_styles(ctx, prompt, styles, seed):
    """Compare a prompt across different styles."""
    config_path = ctx.obj["config_path"]

    pipeline = TextToImagePipeline.from_config(config_path)
    pipeline.setup()

    style_list = list(styles) if styles else None

    result = pipeline.generate_style_comparison(
        prompt=prompt,
        styles=style_list,
        seed=seed,
    )

    click.echo(f"\n✅ Style comparison complete in {result['total_time']:.2f}s")
    click.echo(f"📊 Grid: {result['grid_path']}")

    pipeline.cleanup()


@cli.command()
@click.argument("prompt")
@click.option("--steps-list", default="10,15,20,30,50", help="Comma-separated step counts")
@click.option("--seed", default=42, help="Seed")
@click.pass_context
def compare_quality(ctx, prompt, steps_list, seed):
    """Compare quality at different step counts."""
    config_path = ctx.obj["config_path"]

    pipeline = TextToImagePipeline.from_config(config_path)
    pipeline.setup()

    step_counts = [int(s.strip()) for s in steps_list.split(",")]

    result = pipeline.generate_quality_comparison(
        prompt=prompt,
        step_counts=step_counts,
        seed=seed,
    )

    click.echo(f"\n✅ Quality comparison complete in {result['total_time']:.2f}s")
    click.echo(f"📊 Timing: {result['timing_data']}")
    click.echo(f"📊 Grid: {result['grid_path']}")

    pipeline.cleanup()


@cli.command()
@click.option("--seed", default=42, help="Base seed")
@click.pass_context
def portfolio(ctx, seed):
    """Generate a full portfolio of AI-generated assets."""
    from app.portfolio_generator import PortfolioGenerator

    config_path = ctx.obj["config_path"]

    pipeline = TextToImagePipeline.from_config(config_path)
    pipeline.setup()

    generator = PortfolioGenerator(pipeline)
    summary = generator.generate_full_portfolio(seed=seed)

    click.echo(f"\n✅ Portfolio complete! {summary['total_images']} images in {summary['total_time_seconds']:.1f}s")

    pipeline.cleanup()


@cli.command()
@click.argument("input_file")
@click.option("--format", "-f", "file_format", default=None, help="Input format: json, csv, yaml")
@click.pass_context
def batch(ctx, input_file, file_format):
    """Run batch generation from a file."""
    from app.batch_generator import BatchGenerator

    config_path = ctx.obj["config_path"]

    pipeline = TextToImagePipeline.from_config(config_path)
    pipeline.setup()

    generator = BatchGenerator(pipeline)

    if file_format is None:
        file_format = os.path.splitext(input_file)[1].lstrip(".")

    if file_format == "json":
        result = generator.generate_from_json(input_file)
    elif file_format == "csv":
        result = generator.generate_from_csv(input_file)
    elif file_format in ("yaml", "yml"):
        result = generator.generate_from_yaml(input_file)
    else:
        click.echo(f"Unsupported format: {file_format}")
        return

    click.echo(f"\n✅ Batch complete! {result['successful']}/{result['total_items']} succeeded")

    pipeline.cleanup()


@cli.command()
@click.option("--share", is_flag=True, help="Create public share link")
@click.option("--port", default=7860, help="Port number")
@click.pass_context
def webui(ctx, share, port):
    """Launch Gradio web interface."""
    from app.gradio_app import launch_gradio

    config_path = ctx.obj["config_path"]

    pipeline = TextToImagePipeline.from_config(config_path)
    pipeline.setup()

    click.echo(f"🌐 Launching Web UI on port {port}...")
    launch_gradio(pipeline, share=share, port=port)


@cli.command()
@click.option("--host", default="0.0.0.0", help="API host")
@click.option("--port", default=8000, help="API port")
@click.pass_context
def api(ctx, host, port):
    """Launch FastAPI REST API server."""
    from app.api import launch_api

    config_path = ctx.obj["config_path"]

    pipeline = TextToImagePipeline.from_config(config_path)
    pipeline.setup()

    click.echo(f"🚀 Launching API on {host}:{port}...")
    launch_api(pipeline, host=host, port=port)


@cli.command()
@click.argument("prompt")
def analyze(prompt):
    """Analyze a prompt and get improvement suggestions."""
    engineer = PromptEngineer()
    analysis = engineer.analyze_prompt(prompt)

    click.echo(f"\n📊 Prompt Analysis")
    click.echo(f"{'=' * 40}")
    click.echo(f"Score: {analysis['score']}/100 ({analysis['rating']})")
    click.echo(f"Words: {analysis['word_count']} | Tokens: ~{analysis['estimated_tokens']}")
    click.echo(f"\nSuggestions:")
    for s in analysis['suggestions']:
        click.echo(f"  💡 {s}")


if __name__ == "__main__":
    cli()