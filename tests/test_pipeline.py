"""
Tests for TextToImagePipeline module (unit tests without GPU)

File: tests/test_pipeline.py
Run: pytest tests/test_pipeline.py -v
"""

import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch
from src.pipeline import TextToImagePipeline
from src.utils import load_config


class TestPipelineInit:
    """Test pipeline initialization (no GPU needed)."""

    def test_init_default(self):
        pipeline = TextToImagePipeline(output_dir=tempfile.mkdtemp())
        assert pipeline.is_setup is False
        assert pipeline.generation_count == 0
        assert pipeline.generation_history == []

    def test_init_with_config(self):
        config = {
            "model": {"model_id": "test/model"},
            "output": {"base_dir": tempfile.mkdtemp()},
        }
        pipeline = TextToImagePipeline(config=config)
        assert pipeline.config == config

    def test_check_setup_raises(self):
        pipeline = TextToImagePipeline(output_dir=tempfile.mkdtemp())
        with pytest.raises(RuntimeError, match="not set up"):
            pipeline._check_setup()

    def test_get_history_empty(self):
        pipeline = TextToImagePipeline(output_dir=tempfile.mkdtemp())
        assert pipeline.get_history() == []

    def test_get_stats_no_setup(self):
        pipeline = TextToImagePipeline(output_dir=tempfile.mkdtemp())
        # Should not crash even without setup
        assert pipeline.generation_count == 0


class TestPipelineConfig:
    """Test configuration loading."""

    def test_from_config_missing_file(self):
        """Should handle missing config gracefully."""
        pipeline = TextToImagePipeline.from_config("nonexistent.yaml")
        assert pipeline is not None

    def test_output_dirs_created(self):
        temp_dir = tempfile.mkdtemp()
        pipeline = TextToImagePipeline(output_dir=temp_dir)
        assert os.path.exists(temp_dir)