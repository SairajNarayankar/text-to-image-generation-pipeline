"""
Image Processing Module
Post-processing utilities for generated images
"""

import os
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from loguru import logger
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont


class ImageProcessor:
    """
    Post-processing utilities for AI-generated images.

    Features:
        - Upscaling
        - Color adjustment
        - Filtering
        - Watermarking
        - Grid/contact sheet creation
        - Format conversion
        - Batch processing
    """

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ImageProcessor initialized | Output: {self.output_dir}")

    # ========================
    # ENHANCEMENT METHODS
    # ========================

    def enhance_sharpness(self, image: Image.Image, factor: float = 1.5) -> Image.Image:
        """Enhance image sharpness"""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    def enhance_contrast(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Enhance image contrast"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def enhance_brightness(self, image: Image.Image, factor: float = 1.1) -> Image.Image:
        """Enhance image brightness"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def enhance_saturation(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Enhance color saturation"""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def auto_enhance(
        self,
        image: Image.Image,
        sharpness: float = 1.3,
        contrast: float = 1.1,
        brightness: float = 1.05,
        saturation: float = 1.1
    ) -> Image.Image:
        """Apply automatic enhancement with balanced settings"""
        image = self.enhance_sharpness(image, sharpness)
        image = self.enhance_contrast(image, contrast)
        image = self.enhance_brightness(image, brightness)
        image = self.enhance_saturation(image, saturation)
        logger.debug("Auto-enhancement applied")
        return image

    # ========================
    # FILTER METHODS
    # ========================

    def apply_blur(self, image: Image.Image, radius: int = 2) -> Image.Image:
        """Apply Gaussian blur"""
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def apply_sharpen(self, image: Image.Image) -> Image.Image:
        """Apply sharpening filter"""
        return image.filter(ImageFilter.SHARPEN)

    def apply_edge_enhance(self, image: Image.Image) -> Image.Image:
        """Apply edge enhancement"""
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)

    def apply_smooth(self, image: Image.Image) -> Image.Image:
        """Apply smoothing filter"""
        return image.filter(ImageFilter.SMOOTH_MORE)

    def apply_detail(self, image: Image.Image) -> Image.Image:
        """Apply detail enhancement"""
        return image.filter(ImageFilter.DETAIL)

    # ========================
    # RESIZE & CROP METHODS
    # ========================

    def resize(
        self,
        image: Image.Image,
        width: int,
        height: int,
        method: str = "lanczos"
    ) -> Image.Image:
        """
        Resize image with specified method.

        Args:
            image: Input image
            width: Target width
            height: Target height
            method: Resampling method (lanczos, bilinear, bicubic, nearest)
        """
        methods = {
            "lanczos": Image.LANCZOS,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST
        }
        resample = methods.get(method, Image.LANCZOS)
        return image.resize((width, height), resample)

    def upscale(self, image: Image.Image, scale_factor: int = 2) -> Image.Image:
        """Simple upscaling using Lanczos interpolation"""
        new_width = image.width * scale_factor
        new_height = image.height * scale_factor
        upscaled = image.resize((new_width, new_height), Image.LANCZOS)
        logger.debug(f"Upscaled {scale_factor}x: {image.size} -> {upscaled.size}")
        return upscaled

    def center_crop(self, image: Image.Image, width: int, height: int) -> Image.Image:
        """Crop image from center to specified dimensions"""
        img_width, img_height = image.size
        left = (img_width - width) // 2
        top = (img_height - height) // 2
        right = left + width
        bottom = top + height
        return image.crop((left, top, right, bottom))

    def smart_crop(
        self,
        image: Image.Image,
        target_ratio: float = 1.0
    ) -> Image.Image:
        """
        Smart crop to target aspect ratio.

        Args:
            image: Input image
            target_ratio: Target width/height ratio (1.0 = square)
        """
        img_width, img_height = image.size
        current_ratio = img_width / img_height

        if current_ratio > target_ratio:
            # Too wide, crop width
            new_width = int(img_height * target_ratio)
            left = (img_width - new_width) // 2
            return image.crop((left, 0, left + new_width, img_height))
        else:
            # Too tall, crop height
            new_height = int(img_width / target_ratio)
            top = (img_height - new_height) // 2
            return image.crop((0, top, img_width, top + new_height))

    # ========================
    # GRID & LAYOUT METHODS
    # ========================

    def create_grid(
        self,
        images: List[Image.Image],
        cols: int = 3,
        padding: int = 10,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        labels: Optional[List[str]] = None
    ) -> Image.Image:
        """
        Create a grid/contact sheet from multiple images.

        Args:
            images: List of PIL images
            cols: Number of columns
            padding: Padding between images
            bg_color: Background color
            labels: Optional labels for each image
        """
        if not images:
            raise ValueError("No images provided")

        # Ensure all images are same size
        target_size = images[0].size
        resized_images = [img.resize(target_size, Image.LANCZOS) for img in images]

        rows = (len(resized_images) + cols - 1) // cols
        img_w, img_h = target_size

        # Calculate grid dimensions
        grid_w = cols * img_w + (cols + 1) * padding
        label_height = 30 if labels else 0
        grid_h = rows * (img_h + label_height) + (rows + 1) * padding

        grid = Image.new("RGB", (grid_w, grid_h), bg_color)
        draw = ImageDraw.Draw(grid)

        for idx, img in enumerate(resized_images):
            row = idx // cols
            col = idx % cols

            x = padding + col * (img_w + padding)
            y = padding + row * (img_h + label_height + padding)

            grid.paste(img, (x, y))

            # Add label
            if labels and idx < len(labels):
                label_y = y + img_h + 2
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                except OSError:
                    font = ImageFont.load_default()
                draw.text((x, label_y), labels[idx], fill=(0, 0, 0), font=font)

        logger.debug(f"Created grid: {cols}x{rows} ({len(resized_images)} images)")
        return grid

    def create_comparison(
        self,
        images: List[Image.Image],
        labels: List[str],
        title: Optional[str] = None,
        orientation: str = "horizontal"
    ) -> Image.Image:
        """
        Create a side-by-side comparison image.

        Args:
            images: List of images to compare
            labels: Labels for each image
            title: Optional title
            orientation: 'horizontal' or 'vertical'
        """
        padding = 15
        label_height = 35
        title_height = 45 if title else 0

        # Normalize sizes
        target_size = images[0].size
        resized = [img.resize(target_size, Image.LANCZOS) for img in images]
        img_w, img_h = target_size

        if orientation == "horizontal":
            total_w = len(resized) * img_w + (len(resized) + 1) * padding
            total_h = img_h + label_height + title_height + padding * 3
        else:
            total_w = img_w + padding * 2
            total_h = len(resized) * (img_h + label_height) + title_height + (len(resized) + 1) * padding

        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        # Draw title
        if title:
            draw.text((padding, padding), title, fill=(0, 0, 0), font=title_font)

        for i, (img, label) in enumerate(zip(resized, labels)):
            if orientation == "horizontal":
                x = padding + i * (img_w + padding)
                y = title_height + padding
            else:
                x = padding
                y = title_height + padding + i * (img_h + label_height + padding)

            canvas.paste(img, (x, y))
            draw.text(
                (x, y + img_h + 5),
                label,
                fill=(50, 50, 50),
                font=label_font
            )

        return canvas

    # ========================
    # WATERMARK METHODS
    # ========================

    def add_watermark(
        self,
        image: Image.Image,
        text: str = "AI Generated",
        position: str = "bottom_right",
        opacity: int = 128,
        font_size: int = 20
    ) -> Image.Image:
        """
        Add text watermark to image.

        Args:
            image: Input image
            text: Watermark text
            position: Position (top_left, top_right, bottom_left, bottom_right, center)
            opacity: Watermark opacity (0-255)
            font_size: Font size
        """
        watermarked = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", watermarked.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        # Calculate text size using textbbox
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Calculate position
        img_w, img_h = image.size
        positions = {
            "top_left": (10, 10),
            "top_right": (img_w - text_w - 10, 10),
            "bottom_left": (10, img_h - text_h - 10),
            "bottom_right": (img_w - text_w - 10, img_h - text_h - 10),
            "center": ((img_w - text_w) // 2, (img_h - text_h) // 2),
        }
        pos = positions.get(position, positions["bottom_right"])

        draw.text(pos, text, fill=(255, 255, 255, opacity), font=font)

        watermarked = Image.alpha_composite(watermarked, overlay)
        return watermarked.convert("RGB")

    # ========================
    # SAVE & EXPORT METHODS
    # ========================

    def save_image(
        self,
        image: Image.Image,
        filename: str,
        subfolder: str = "",
        format: str = "PNG",
        quality: int = 95,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save image with optional metadata.

        Args:
            image: Image to save
            filename: Output filename
            subfolder: Optional subfolder within output_dir
            format: Image format (PNG, JPEG, WEBP)
            quality: JPEG/WEBP quality
            metadata: Optional metadata to embed

        Returns:
            Full path to saved file
        """
        save_dir = self.output_dir / subfolder
        save_dir.mkdir(parents=True, exist_ok=True)

        filepath = save_dir / filename

        save_kwargs = {"format": format}
        if format.upper() in ("JPEG", "JPG", "WEBP"):
            save_kwargs["quality"] = quality

        if metadata and format.upper() == "PNG":
            from PIL.PngImagePlugin import PngInfo
            png_info = PngInfo()
            for key, value in metadata.items():
                png_info.add_text(key, str(value))
            save_kwargs["pnginfo"] = png_info

        image.save(str(filepath), **save_kwargs)
        logger.debug(f"Saved image: {filepath}")

        return str(filepath)

    def batch_process(
        self,
        images: List[Image.Image],
        operations: List[Dict[str, Any]]
    ) -> List[Image.Image]:
        """
        Apply a sequence of operations to multiple images.

        Args:
            images: List of input images
            operations: List of operation dicts, e.g.:
                [
                    {"method": "auto_enhance", "kwargs": {"sharpness": 1.5}},
                    {"method": "resize", "kwargs": {"width": 1024, "height": 1024}},
                    {"method": "add_watermark", "kwargs": {"text": "AI Art"}}
                ]

        Returns:
            List of processed images
        """
        processed = []

        for i, image in enumerate(images):
            current = image.copy()

            for op in operations:
                method_name = op.get("method")
                kwargs = op.get("kwargs", {})

                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    current = method(current, **kwargs)
                else:
                    logger.warning(f"Unknown method: {method_name}")

            processed.append(current)
            logger.debug(f"Processed image {i + 1}/{len(images)}")

        logger.info(f"Batch processed {len(processed)} images with {len(operations)} operations")
        return processed

    def export_for_web(
        self,
        image: Image.Image,
        filename: str,
        sizes: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> Dict[str, str]:
        """
        Export image in multiple sizes for web use.

        Args:
            image: Source image
            filename: Base filename
            sizes: Dict of {size_name: (width, height)}

        Returns:
            Dict of {size_name: filepath}
        """
        if sizes is None:
            sizes = {
                "thumbnail": (150, 150),
                "small": (300, 300),
                "medium": (600, 600),
                "large": (1200, 1200),
                "original": image.size
            }

        exports = {}
        base_name = Path(filename).stem

        for size_name, (w, h) in sizes.items():
            if size_name == "original":
                resized = image
            else:
                resized = image.resize((w, h), Image.LANCZOS)

            size_filename = f"{base_name}_{size_name}.webp"
            filepath = self.save_image(
                resized,
                size_filename,
                subfolder="exports/web",
                format="WEBP",
                quality=85
            )
            exports[size_name] = filepath

        logger.info(f"Exported {len(exports)} web sizes for {filename}")
        return exports