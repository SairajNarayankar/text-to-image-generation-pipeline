"""
Portfolio Export Script
File: scripts/export_portfolio.py
"""

import os
import sys
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
from pathlib import Path
from PIL import Image
from src.image_processor import ImageProcessor
from src.utils import setup_logger, save_json


@click.command()
@click.option("--input", "-i", "input_dir", default="./output/portfolio")
@click.option("--output", "-o", "output_dir", default="./output/exports")
@click.option("--format", "-f", "export_format", default="webp")
@click.option("--quality", "-q", default=90, help="Export quality (1-100)")
@click.option("--sizes", "-s", default="thumb:150,small:300,medium:600,large:1200",
              help="Export sizes as name:pixels pairs")
@click.option("--watermark", "-w", default=None, help="Add watermark text")
@click.option("--create-html", is_flag=True, help="Create HTML gallery")
def export(input_dir, output_dir, export_format, quality, sizes, watermark, create_html):
    """Export portfolio images in web-optimized formats."""

    setup_logger()
    os.makedirs(output_dir, exist_ok=True)

    processor = ImageProcessor(output_dir=output_dir)

    size_map = {}
    for pair in sizes.split(","):
        name, px = pair.split(":")
        size_map[name] = (int(px), int(px))

    input_path = Path(input_dir)
    image_files = list(input_path.rglob("*.png")) + list(input_path.rglob("*.jpg"))
    image_files = [f for f in image_files if "grid" not in f.name]

    click.echo(f"\n📦 Exporting {len(image_files)} images")
    click.echo(f"   Format: {export_format} | Quality: {quality}")
    click.echo(f"   Sizes: {size_map}\n")

    export_manifest = []

    for img_path in image_files:
        click.echo(f"  Processing: {img_path.name}")
        image = Image.open(str(img_path))

        if watermark:
            image = processor.add_watermark(image, text=watermark)

        base_name = img_path.stem
        collection = img_path.parent.name

        collection_dir = os.path.join(output_dir, collection)
        os.makedirs(collection_dir, exist_ok=True)

        exports = {}
        for size_name, (w, h) in size_map.items():
            resized = image.resize((w, h), Image.LANCZOS)
            filename = f"{base_name}_{size_name}.{export_format}"
            filepath = os.path.join(collection_dir, filename)

            save_kwargs = {"format": export_format.upper()}
            if export_format in ("jpeg", "jpg", "webp"):
                save_kwargs["quality"] = quality

            resized.save(filepath, **save_kwargs)
            exports[size_name] = filepath

        original_path = os.path.join(collection_dir, f"{base_name}_original.{export_format}")
        save_kwargs = {"format": export_format.upper()}
        if export_format in ("jpeg", "jpg", "webp"):
            save_kwargs["quality"] = quality
        image.save(original_path, **save_kwargs)
        exports["original"] = original_path

        export_manifest.append({
            "name": base_name,
            "collection": collection,
            "exports": exports,
        })

    save_json({"exports": export_manifest}, os.path.join(output_dir, "manifest.json"))

    if create_html:
        _create_html_gallery(output_dir, export_manifest, export_format)

    click.echo(f"\n✅ Export complete! {len(export_manifest)} images exported to {output_dir}")


def _create_html_gallery(output_dir, manifest, fmt):
    """Create a simple HTML gallery."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Generated Portfolio</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #0a0a0a; color: #fff; padding: 40px; }
        h1 { text-align: center; margin-bottom: 10px; font-size: 2.5em; }
        .subtitle { text-align: center; color: #888; margin-bottom: 40px; }
        .collection { margin-bottom: 50px; }
        .collection h2 { color: #667eea; margin-bottom: 20px; border-bottom: 2px solid #333; padding-bottom: 10px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #1a1a2e; border-radius: 12px; overflow: hidden; transition: transform 0.3s; }
        .card:hover { transform: translateY(-5px); box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); }
        .card img { width: 100%; height: 300px; object-fit: cover; }
        .card .info { padding: 15px; }
        .card .info h3 { font-size: 1em; color: #ddd; }
        .footer { text-align: center; margin-top: 50px; color: #555; }
    </style>
</head>
<body>
    <h1>🎨 AI Generated Portfolio</h1>
    <p class="subtitle">Created with Text-to-Image Generation Pipeline</p>
"""

    collections = {}
    for item in manifest:
        col = item["collection"]
        if col not in collections:
            collections[col] = []
        collections[col].append(item)

    for col_name, items in collections.items():
        html += f'    <div class="collection">\n'
        html += f'        <h2>{col_name.replace("_", " ").title()}</h2>\n'
        html += f'        <div class="grid">\n'

        for item in items:
            medium = item["exports"].get("medium", item["exports"].get("original", ""))
            rel_path = os.path.relpath(medium, output_dir) if medium else ""

            html += f'            <div class="card">\n'
            html += f'                <img src="{rel_path}" alt="{item["name"]}" loading="lazy">\n'
            html += f'                <div class="info"><h3>{item["name"].replace("_", " ").title()}</h3></div>\n'
            html += f'            </div>\n'

        html += f'        </div>\n'
        html += f'    </div>\n'

    html += """    <div class="footer">
        <p>Generated using Stable Diffusion & Hugging Face Diffusers</p>
    </div>
</body>
</html>"""

    html_path = os.path.join(output_dir, "gallery.html")
    with open(html_path, "w") as f:
        f.write(html)

    click.echo(f"🌐 HTML gallery created: {html_path}")


if __name__ == "__main__":
    export()