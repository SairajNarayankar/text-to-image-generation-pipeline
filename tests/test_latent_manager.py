"""
Tests for LatentSpaceManager module (requires mock or GPU)

File: tests/test_latent_manager.py
Run: pytest tests/test_latent_manager.py -v
"""

import pytest
import torch
from unittest.mock import MagicMock


class TestLatentManagerMock:
    """Test LatentSpaceManager with mocked pipeline (no GPU needed)."""

    def setup_method(self):
        """Create mock pipeline."""
        self.mock_pipe = MagicMock()
        self.mock_pipe.vae_scale_factor = 8
        self.mock_pipe.vae.config.scaling_factor = 0.18215

    def test_slerp_basic(self):
        """Test spherical interpolation math."""
        from src.latent_manager import LatentSpaceManager

        # Create manager with mock
        manager = LatentSpaceManager.__new__(LatentSpaceManager)
        manager.device = "cpu"

        a = torch.randn(1, 4, 64, 64)
        b = torch.randn(1, 4, 64, 64)

        result = manager._slerp(a, b, 0.5)
        assert result.shape == a.shape

    def test_slerp_endpoints(self):
        """Test slerp at alpha=0 and alpha=1."""
        from src.latent_manager import LatentSpaceManager

        manager = LatentSpaceManager.__new__(LatentSpaceManager)
        manager.device = "cpu"

        a = torch.randn(1, 4, 64, 64)
        b = torch.randn(1, 4, 64, 64)

        result_0 = manager._slerp(a, b, 0.0)
        result_1 = manager._slerp(a, b, 1.0)

        assert torch.allclose(result_0, a, atol=1e-5)
        assert torch.allclose(result_1, b, atol=1e-5)

    def test_latent_arithmetic(self):
        """Test latent arithmetic operations."""
        from src.latent_manager import LatentSpaceManager

        manager = LatentSpaceManager.__new__(LatentSpaceManager)
        manager.device = "cpu"

        a = torch.ones(1, 4, 8, 8)
        b = torch.ones(1, 4, 8, 8) * 2

        result = manager.latent_arithmetic([a, b], [1.0, -0.5])
        expected = a * 1.0 + b * (-0.5)
        assert torch.allclose(result, expected)

    def test_get_latent_statistics(self):
        """Test latent statistics computation."""
        from src.latent_manager import LatentSpaceManager

        manager = LatentSpaceManager.__new__(LatentSpaceManager)
        manager.device = "cpu"

        latent = torch.randn(1, 4, 64, 64)
        stats = manager.get_latent_statistics(latent)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "shape" in stats
        assert stats["shape"] == [1, 4, 64, 64]