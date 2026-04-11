"""
Model Download Script
Pre-download models for offline use

File: scripts/download_models.py
Run: python scripts/download_models.py --model sd-1.5
"""

import os
import sys
import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import ModelLoader
from src.utils import setup_logger, print_system_info


@click.command()
@click.option("--model", "-m", default="sd-1.5",
              help="Model name from registry (sd-1.5, sd-2.1, sdxl, etc.)")
@click.option("--model-id", default=None,
              help="Custom Hugging Face model ID")
@click.option("--cache-dir", default="./model_cache",
              help="Cache directory for downloaded models")
@click.option("--all-models", is_flag=True,
              help="Download all models in registry")
@click.option("--list", "list_models", is_flag=True,
              help="List available models")
def download(model, model_id, cache_dir, all_models, list_models):
    """Download Stable Diffusion models for offline use."""

    setup_logger()

    if list_models:
        ModelLoader.list_available_models()
        return

    if all_models:
        models_to_download = list(ModelLoader.MODEL_REGISTRY.keys())
    elif model_id:
        models_to_download = [model_id]
    else:
        models_to_download = [model]

    click.echo(f"\n📥 Downloading {len(models_to_download)} model(s)...")
    click.echo(f"📁 Cache directory: {cache_dir}\n")

    for m in models_to_download:
        try:
            click.echo(f"⏳ Downloading: {m}")

            if m in ModelLoader.MODEL_REGISTRY:
                loader = ModelLoader(model_name=m, cache_dir=cache_dir, device="cpu")
            else:
                loader = ModelLoader(model_id=m, cache_dir=cache_dir, device="cpu")

            from diffusers import StableDiffusionPipeline
            import torch

            mid = loader.model_id
            StableDiffusionPipeline.from_pretrained(
                mid,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                safety_checker=None,
                requires_safety_checker=False,
            )

            click.echo(f"✅ Downloaded: {m}")

        except Exception as e:
            click.echo(f"❌ Failed to download {m}: {e}")

    click.echo(f"\n✅ Download complete!")
    click.echo(f"📁 Models cached at: {os.path.abspath(cache_dir)}")


if __name__ == "__main__":
    download()