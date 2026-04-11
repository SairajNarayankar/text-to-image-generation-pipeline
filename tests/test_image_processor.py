"""
Tests for ImageProcessor module

File: tests/test_image_processor.py
Run: pytest tests/test_image_processor.py -v
"""

import os
import pytest
import tempfile
from PIL import Image
from src.image_processor import ImageProcessor


class TestImageProcessor:
    """Test suite for ImageProcessor."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = ImageProcessor(output_dir=self.temp_dir)
        # Create test image
        self.test_image = Image.new("RGB", (512, 512), color=(128, 64, 32))

    def teardown_method(self):
        """Cleanup temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        assert os.path.exists(self.temp_dir)

    # --- Enhancement Tests ---

    def test_enhance_sharpness(self):
        result = self.processor.enhance_sharpness(self.test_image, factor=1.5)
        assert result.size == self.test_image.size

    def test_enhance_contrast(self):
        result = self.processor.enhance_contrast(self.test_image, factor=1.2)
        assert result.size == self.test_image.size

    def test_enhance_brightness(self):
        result = self.processor.enhance_brightness(self.test_image, factor=1.1)
        assert result.size == self.test_image.size

    def test_enhance_saturation(self):
        result = self.processor.enhance_saturation(self.test_image, factor=1.2)
        assert result.size == self.test_image.size

    def test_auto_enhance(self):
        result = self.processor.auto_enhance(self.test_image)
        assert result.size == self.test_image.size

    # --- Resize/Crop Tests ---

    def test_resize(self):
        result = self.processor.resize(self.test_image, 256, 256)
        assert result.size == (256, 256)

    def test_upscale(self):
        small_img = Image.new("RGB", (128, 128), color="red")
        result = self.processor.upscale(small_img, scale_factor=2)
        assert result.size == (256, 256)

    def test_center_crop(self):
        result = self.processor.center_crop(self.test_image, 256, 256)
        assert result.size == (256, 256)

    def test_smart_crop_square(self):
        wide_img = Image.new("RGB", (800, 400), color="blue")
        result = self.processor.smart_crop(wide_img, target_ratio=1.0)
        assert result.size[0] == result.size[1]

    # --- Grid Tests ---

    def test_create_grid(self):
        images = [Image.new("RGB", (100, 100), color=c) for c in ["red", "green", "blue", "yellow"]]
        grid = self.processor.create_grid(images, cols=2)
        assert grid.size[0] > 100
        assert grid.size[1] > 100

    def test_create_grid_with_labels(self):
        images = [Image.new("RGB", (100, 100), color="red") for _ in range(4)]
        labels = ["A", "B", "C", "D"]
        grid = self.processor.create_grid(images, cols=2, labels=labels)
        assert grid is not None

    def test_create_comparison(self):
        images = [Image.new("RGB", (100, 100), color=c) for c in ["red", "blue"]]
        comp = self.processor.create_comparison(images, ["Red", "Blue"], title="Test")
        assert comp is not None

    def test_create_grid_empty_raises(self):
        with pytest.raises(ValueError):
            self.processor.create_grid([], cols=2)

    # --- Watermark Tests ---

    def test_add_watermark(self):
        result = self.processor.add_watermark(
            self.test_image, text="TEST", position="bottom_right"
        )
        assert result.size == self.test_image.size
        assert result.mode == "RGB"

    # --- Save Tests ---

    def test_save_image_png(self):
        path = self.processor.save_image(self.test_image, "test.png")
        assert os.path.exists(path)
        loaded = Image.open(path)
        assert loaded.size == self.test_image.size

    def test_save_image_jpeg(self):
        path = self.processor.save_image(
            self.test_image, "test.jpg", format="JPEG", quality=85
        )
        assert os.path.exists(path)

    def test_save_image_with_subfolder(self):
        path = self.processor.save_image(
            self.test_image, "test.png", subfolder="subdir"
        )
        assert os.path.exists(path)
        assert "subdir" in path

    def test_save_image_with_metadata(self):
        metadata = {"prompt": "test prompt", "seed": "42"}
        path = self.processor.save_image(
            self.test_image, "meta_test.png", metadata=metadata
        )
        assert os.path.exists(path)

    # --- Batch Processing Tests ---

    def test_batch_process(self):
        images = [Image.new("RGB", (100, 100), color="red") for _ in range(3)]
        operations = [
            {"method": "enhance_sharpness", "kwargs": {"factor": 1.5}},
            {"method": "resize", "kwargs": {"width": 50, "height": 50}},
        ]
        results = self.processor.batch_process(images, operations)
        assert len(results) == 3
        assert all(img.size == (50, 50) for img in results)

    # --- Export Tests ---

    def test_export_for_web(self):
        exports = self.processor.export_for_web(self.test_image, "web_test.png")
        assert "thumbnail" in exports
        assert "small" in exports
        assert "medium" in exports
        assert "large" in exports
        for path in exports.values():
            assert os.path.exists(path)

    # --- Filter Tests ---

    def test_apply_blur(self):
        result = self.processor.apply_blur(self.test_image, radius=2)
        assert result.size == self.test_image.size

    def test_apply_sharpen(self):
        result = self.processor.apply_sharpen(self.test_image)
        assert result.size == self.test_image.size

    def test_apply_detail(self):
        result = self.processor.apply_detail(self.test_image)
        assert result.size == self.test_image.size