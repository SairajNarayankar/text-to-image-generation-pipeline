"""
Tests for PromptEngineer module

File: tests/test_prompt_engineer.py
Run: pytest tests/test_prompt_engineer.py -v
"""

import pytest
from src.prompt_engineer import PromptEngineer, PromptResult


class TestPromptEngineer:
    """Test suite for PromptEngineer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.engineer = PromptEngineer(
            default_style="digital_art",
            default_quality="high"
        )

    def test_initialization(self):
        """Test PromptEngineer initializes correctly."""
        assert self.engineer.default_style == "digital_art"
        assert self.engineer.default_quality == "high"
        assert self.engineer.max_tokens == 77

    def test_build_prompt_basic(self):
        """Test basic prompt building."""
        result = self.engineer.build_prompt("a beautiful sunset")
        assert isinstance(result, PromptResult)
        assert "sunset" in result.positive
        assert len(result.positive) > len("a beautiful sunset")
        assert len(result.negative) > 0

    def test_build_prompt_with_style(self):
        """Test prompt building with specific style."""
        result = self.engineer.build_prompt(
            "a mountain landscape",
            style="photorealistic"
        )
        assert result.style == "photorealistic"
        assert "photorealistic" in result.positive.lower()

    def test_build_prompt_with_quality(self):
        """Test prompt building with quality level."""
        result = self.engineer.build_prompt(
            "a cat",
            quality="ultra"
        )
        assert result.quality_level == "ultra"
        assert "masterpiece" in result.positive.lower()

    def test_build_prompt_with_emphasis(self):
        """Test prompt building with emphasis weights."""
        result = self.engineer.build_prompt(
            "a red car on the street",
            emphasis={"red car": 1.5}
        )
        assert "(red car:1.5)" in result.positive

    def test_build_prompt_removes_duplicates(self):
        """Test that duplicate terms are removed."""
        result = self.engineer.build_prompt(
            "detailed art, detailed painting, detailed",
            optimize=True
        )
        parts = [p.strip().lower() for p in result.positive.split(",")]
        # Should have fewer duplicates
        assert len(parts) == len(set(parts))

    def test_all_styles_valid(self):
        """Test that all style presets produce valid results."""
        for style_name in PromptEngineer.STYLES:
            result = self.engineer.build_prompt("test image", style=style_name)
            assert isinstance(result, PromptResult)
            assert len(result.positive) > 0
            assert len(result.negative) > 0

    def test_all_quality_levels_valid(self):
        """Test that all quality levels produce valid results."""
        for quality_name in PromptEngineer.QUALITY_LEVELS:
            result = self.engineer.build_prompt("test image", quality=quality_name)
            assert isinstance(result, PromptResult)

    def test_unknown_style_warns(self):
        """Test that unknown style produces warning."""
        result = self.engineer.build_prompt(
            "test image",
            style="nonexistent_style"
        )
        assert len(result.warnings) > 0

    def test_analyze_prompt_basic(self):
        """Test prompt analysis."""
        analysis = self.engineer.analyze_prompt(
            "a beautiful sunset over mountains"
        )
        assert "original_prompt" in analysis
        assert "score" in analysis
        assert "suggestions" in analysis
        assert 0 <= analysis["score"] <= 100

    def test_analyze_prompt_detailed(self):
        """Test analysis of a detailed prompt."""
        analysis = self.engineer.analyze_prompt(
            "photorealistic image of a cozy coffee shop, warm lighting, "
            "detailed interior, sharp focus, professional photography"
        )
        assert analysis["score"] > 40
        assert analysis["has_quality_terms"] is True

    def test_analyze_prompt_minimal(self):
        """Test analysis of a minimal prompt."""
        analysis = self.engineer.analyze_prompt("cat")
        assert analysis["score"] < 50
        assert len(analysis["suggestions"]) > 0

    def test_enhance_prompt(self):
        """Test prompt enhancement."""
        enhanced = self.engineer.enhance_prompt(
            "a forest",
            enhancements=["lighting", "detail"]
        )
        assert "lighting" in enhanced.lower()
        assert len(enhanced) > len("a forest")

    def test_create_variation_prompts_style(self):
        """Test style variation creation."""
        variations = self.engineer.create_variation_prompts(
            "a mountain landscape",
            num_variations=3,
            variation_type="style"
        )
        assert len(variations) == 3
        styles = [v.style for v in variations]
        assert len(set(styles)) == 3  # All different styles

    def test_create_variation_prompts_quality(self):
        """Test quality variation creation."""
        variations = self.engineer.create_variation_prompts(
            "a mountain landscape",
            num_variations=3,
            variation_type="quality"
        )
        assert len(variations) == 3

    def test_negative_categories(self):
        """Test negative prompt categories."""
        result = self.engineer.build_prompt(
            "portrait of a person",
            negative_categories=["anatomy", "face", "hands"]
        )
        assert "bad anatomy" in result.negative.lower()

    def test_additional_terms(self):
        """Test additional positive/negative terms."""
        result = self.engineer.build_prompt(
            "a landscape",
            additional_positive="golden hour, dramatic sky",
            additional_negative="people, buildings"
        )
        assert "golden hour" in result.positive
        assert "people" in result.negative

    def test_token_estimation(self):
        """Test token estimation."""
        result = self.engineer.build_prompt("short")
        assert result.tokens_estimated > 0
        assert isinstance(result.tokens_estimated, int)


class TestPromptResult:
    """Test PromptResult dataclass."""

    def test_prompt_result_creation(self):
        result = PromptResult(
            positive="test positive",
            negative="test negative",
            style="digital_art",
            quality_level="high",
            tokens_estimated=10
        )
        assert result.positive == "test positive"
        assert result.warnings == []

    def test_prompt_result_with_warnings(self):
        result = PromptResult(
            positive="test",
            negative="test",
            style="test",
            quality_level="test",
            tokens_estimated=100,
            warnings=["Token limit exceeded"]
        )
        assert len(result.warnings) == 1