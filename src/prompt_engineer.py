"""
Prompt Engineering Module
Advanced prompt optimization for Stable Diffusion models
"""

import re
import yaml
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field


@dataclass
class PromptResult:
    """Structured result from prompt engineering"""
    positive: str
    negative: str
    style: str
    quality_level: str
    tokens_estimated: int
    warnings: List[str] = field(default_factory=list)


class PromptEngineer:
    """
    Advanced prompt engineering for Stable Diffusion.

    Features:
        - Style presets with positive/negative prompts
        - Quality level management
        - Domain-specific templates
        - Token estimation and optimization
        - Prompt weighting syntax
        - Negative prompt management
        - Template system with variable substitution
    """

    MAX_TOKENS_SD15 = 77
    MAX_TOKENS_SDXL = 77 * 2

    STYLES = {
        "photorealistic": {
            "positive": "photorealistic, ultra realistic, DSLR quality, 8k UHD, sharp focus, high detail, natural lighting, RAW photo",
            "negative": "cartoon, illustration, painting, drawing, anime, CGI, 3d render, sketch"
        },
        "digital_art": {
            "positive": "digital art, trending on artstation, highly detailed, vibrant colors, professional digital painting, concept art",
            "negative": "photo, photograph, realistic, blurry, low quality, amateur"
        },
        "oil_painting": {
            "positive": "oil painting on canvas, masterpiece, classical art style, rich colors, visible brush strokes, museum quality, fine art",
            "negative": "digital, photo, modern, minimalist, cartoon, flat colors"
        },
        "watercolor": {
            "positive": "watercolor painting, soft edges, flowing colors, paper texture, artistic, delicate, translucent washes",
            "negative": "digital, photo, sharp edges, bold lines, oil painting, acrylic"
        },
        "anime": {
            "positive": "anime style, vibrant colors, detailed, manga art, cel shading, Japanese animation, clean lines",
            "negative": "realistic, photo, western art, oil painting, 3d, uncanny valley"
        },
        "minimalist": {
            "positive": "minimalist design, clean lines, simple composition, modern, elegant, white space, flat design, geometric",
            "negative": "complex, cluttered, ornate, baroque, detailed texture, busy background"
        },
        "cyberpunk": {
            "positive": "cyberpunk style, neon lights, futuristic, dark atmosphere, rain, holographic, blade runner inspired, dystopian",
            "negative": "natural, bright, sunny, pastoral, vintage, rustic, peaceful"
        },
        "vintage": {
            "positive": "vintage style, retro, sepia tones, nostalgic, film grain, aged, classic aesthetic, old photograph",
            "negative": "modern, digital, neon, futuristic, clean, sharp, high contrast"
        },
        "3d_render": {
            "positive": "3D render, octane render, unreal engine 5, ray tracing, volumetric lighting, subsurface scattering, PBR materials, CGI",
            "negative": "2d, flat, painting, drawing, sketch, cartoon, hand drawn"
        },
        "sketch": {
            "positive": "pencil sketch, hand drawn, graphite, detailed linework, cross hatching, sketchbook, artistic",
            "negative": "color, painted, digital, photo, realistic, vibrant"
        },
        "fantasy": {
            "positive": "fantasy art, magical, ethereal glow, mythical, enchanted, epic fantasy illustration, detailed, otherworldly",
            "negative": "realistic, modern, urban, mundane, minimalist, boring"
        },
        "pop_art": {
            "positive": "pop art style, bold colors, halftone dots, Andy Warhol inspired, comic style, high contrast, graphic",
            "negative": "realistic, muted colors, subtle, minimalist, classical, traditional"
        },
        "cinematic": {
            "positive": "cinematic, movie still, dramatic lighting, depth of field, anamorphic lens, film grain, color grading, epic",
            "negative": "flat lighting, amateur, snapshot, low budget, cartoon"
        },
        "isometric": {
            "positive": "isometric view, 3D isometric, game asset, clean design, detailed, miniature, diorama style",
            "negative": "perspective, flat, 2d, blurry, realistic photo"
        }
    }

    QUALITY_LEVELS = {
        "draft": {
            "boosters": "decent quality",
            "negative": "worst quality, blurry"
        },
        "standard": {
            "boosters": "good quality, detailed, well-composed",
            "negative": "low quality, blurry, bad composition"
        },
        "high": {
            "boosters": "high quality, highly detailed, sharp focus, professional, well-composed",
            "negative": "low quality, blurry, artifacts, noise, distorted, poorly drawn, bad anatomy"
        },
        "ultra": {
            "boosters": (
                "masterpiece, best quality, ultra detailed, 8k resolution, "
                "sharp focus, intricate details, professional, award winning, "
                "perfect composition, stunning"
            ),
            "negative": (
                "worst quality, low quality, blurry, artifacts, noise, distorted, "
                "deformed, ugly, bad anatomy, bad proportions, watermark, text, "
                "logo, signature, jpeg artifacts, cropped, out of frame"
            )
        }
    }

    NEGATIVE_CATEGORIES = {
        "quality": "low quality, blurry, artifacts, noise, pixelated, jpeg artifacts",
        "anatomy": "bad anatomy, bad proportions, deformed, mutated, extra limbs, extra fingers, missing fingers",
        "face": "deformed face, ugly face, bad eyes, cross-eyed, asymmetric face",
        "hands": "bad hands, mutated hands, extra fingers, missing fingers, fused fingers",
        "composition": "cropped, out of frame, poorly composed, bad framing, cut off",
        "watermark": "watermark, text, logo, signature, caption, username, artist name",
        "nsfw": "nsfw, nude, explicit, sexual, inappropriate",
        "duplicates": "duplicate, clone, copy, repetitive, multiple subjects"
    }

    def __init__(
        self,
        default_style: str = "digital_art",
        default_quality: str = "high",
        max_tokens: int = 77,
        custom_styles_path: Optional[str] = None,
        custom_templates_path: Optional[str] = None
    ):
        self.default_style = default_style
        self.default_quality = default_quality
        self.max_tokens = max_tokens
        self.templates = {}

        if custom_styles_path:
            self._load_custom_styles(custom_styles_path)

        if custom_templates_path:
            self._load_templates(custom_templates_path)

        logger.info(
            f"PromptEngineer initialized | Style: {default_style} | "
            f"Quality: {default_quality} | Max tokens: {max_tokens}"
        )

    def _load_custom_styles(self, path: str):
        """Load custom styles from YAML file"""
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            if "styles" in data:
                self.STYLES.update(data["styles"])
                logger.info(f"Loaded {len(data['styles'])} custom styles from {path}")
        except Exception as e:
            logger.error(f"Failed to load custom styles: {e}")

    def _load_templates(self, path: str):
        """Load prompt templates from YAML file"""
        try:
            with open(path, "r") as f:
                self.templates = yaml.safe_load(f)
            logger.info(f"Loaded templates from {path}")
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimation"""
        words = len(text.split())
        special_chars = text.count(",") + text.count("(") + text.count(")")
        return int(words * 0.75 + special_chars * 0.5)

    def build_prompt(
        self,
        base_prompt: str,
        style: Optional[str] = None,
        quality: Optional[str] = None,
        additional_positive: Optional[str] = None,
        additional_negative: Optional[str] = None,
        negative_categories: Optional[List[str]] = None,
        emphasis: Optional[Dict[str, float]] = None,
        optimize: bool = True
    ) -> PromptResult:
        """
        Build an optimized prompt with style and quality modifiers.

        Args:
            base_prompt: Core description of desired image
            style: Style preset name
            quality: Quality level name
            additional_positive: Extra positive prompt terms
            additional_negative: Extra negative prompt terms
            negative_categories: List of negative categories to include
            emphasis: Dict of words/phrases to emphasize {phrase: weight}
            optimize: Whether to optimize prompt length

        Returns:
            PromptResult with positive/negative prompts
        """
        warnings = []
        style = style or self.default_style
        quality = quality or self.default_quality

        # --- BUILD POSITIVE PROMPT ---
        positive_parts = [base_prompt]

        # Add style modifiers
        if style in self.STYLES:
            positive_parts.append(self.STYLES[style]["positive"])
        else:
            warnings.append(f"Unknown style '{style}', using base prompt only")

        # Add quality boosters
        if quality in self.QUALITY_LEVELS:
            positive_parts.append(self.QUALITY_LEVELS[quality]["boosters"])

        # Add additional positive terms
        if additional_positive:
            positive_parts.append(additional_positive)

        positive_prompt = ", ".join(positive_parts)

        # Apply emphasis weighting
        if emphasis:
            positive_prompt = self._apply_emphasis(positive_prompt, emphasis)

        # --- BUILD NEGATIVE PROMPT ---
        negative_parts = []

        # Add style-specific negatives
        if style in self.STYLES:
            negative_parts.append(self.STYLES[style]["negative"])

        # Add quality-level negatives
        if quality in self.QUALITY_LEVELS:
            negative_parts.append(self.QUALITY_LEVELS[quality]["negative"])

        # Add category-specific negatives
        if negative_categories:
            for category in negative_categories:
                if category in self.NEGATIVE_CATEGORIES:
                    negative_parts.append(self.NEGATIVE_CATEGORIES[category])

        # Add additional negative terms
        if additional_negative:
            negative_parts.append(additional_negative)

        negative_prompt = ", ".join(negative_parts)

        # --- OPTIMIZE ---
        if optimize:
            positive_prompt = self._remove_duplicates(positive_prompt)
            negative_prompt = self._remove_duplicates(negative_prompt)

        # --- TOKEN CHECK ---
        token_count = self._estimate_tokens(positive_prompt)
        if token_count > self.max_tokens:
            warnings.append(
                f"Estimated {token_count} tokens exceeds limit of {self.max_tokens}. "
                f"Prompt may be truncated."
            )
            if optimize:
                positive_prompt = self._truncate_prompt(positive_prompt, self.max_tokens)

        result = PromptResult(
            positive=positive_prompt,
            negative=negative_prompt,
            style=style,
            quality_level=quality,
            tokens_estimated=self._estimate_tokens(positive_prompt),
            warnings=warnings
        )

        logger.debug(f"Prompt built | Tokens: ~{result.tokens_estimated} | Style: {style}")
        return result

    def _apply_emphasis(self, prompt: str, emphasis: Dict[str, float]) -> str:
        """
        Apply emphasis weighting to specific terms.
        Uses (term:weight) syntax compatible with Stable Diffusion.

        Args:
            prompt: Input prompt
            emphasis: {phrase: weight} mapping

        Returns:
            Prompt with emphasis applied
        """
        for phrase, weight in emphasis.items():
            if phrase in prompt:
                weighted = f"({phrase}:{weight:.1f})"
                prompt = prompt.replace(phrase, weighted)
        return prompt

    def _remove_duplicates(self, prompt: str) -> str:
        """Remove duplicate terms from a comma-separated prompt"""
        parts = [p.strip() for p in prompt.split(",")]
        seen = set()
        unique_parts = []
        for part in parts:
            part_lower = part.lower()
            if part_lower not in seen and part.strip():
                seen.add(part_lower)
                unique_parts.append(part)
        return ", ".join(unique_parts)

    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """Truncate prompt to approximate token limit"""
        parts = [p.strip() for p in prompt.split(",")]
        result = []
        current_tokens = 0

        for part in parts:
            part_tokens = self._estimate_tokens(part)
            if current_tokens + part_tokens <= max_tokens:
                result.append(part)
                current_tokens += part_tokens
            else:
                break

        return ", ".join(result)

    def from_template(
        self,
        domain: str,
        template_name: str,
        variables: Dict[str, str],
        style: Optional[str] = None,
        quality: Optional[str] = None,
        **kwargs
    ) -> PromptResult:
        """
        Generate prompt from a template with variable substitution.

        Args:
            domain: Template domain (marketing, creative_design, prototyping)
            template_name: Specific template name
            variables: Variables to substitute in template
            style: Style override
            quality: Quality override

        Returns:
            PromptResult
        """
        if not self.templates:
            logger.warning("No templates loaded. Using basic build_prompt.")
            return self.build_prompt(str(variables), style=style, quality=quality, **kwargs)

        try:
            template_config = self.templates[domain][template_name]
            template_str = template_config["template"]

            # Merge defaults with provided variables
            defaults = template_config.get("defaults", {})
            all_vars = {**defaults, **variables}

            # Substitute variables
            base_prompt = template_str.format(**all_vars)

            # Use template's style if not overridden
            template_style = style or template_config.get("style", self.default_style)

            return self.build_prompt(
                base_prompt=base_prompt,
                style=template_style,
                quality=quality,
                **kwargs
            )

        except KeyError as e:
            logger.error(f"Template not found: {domain}/{template_name} - {e}")
            raise ValueError(f"Template '{domain}/{template_name}' not found") from e
        except Exception as e:
            logger.error(f"Template error: {e}")
            raise

    def enhance_prompt(self, prompt: str, enhancements: List[str] = None) -> str:
        """
        Enhance a basic prompt with additional descriptors.

        Args:
            prompt: Base prompt
            enhancements: List of enhancement types

        Returns:
            Enhanced prompt string
        """
        enhancement_map = {
            "lighting": "dramatic lighting, volumetric rays, golden hour, rim lighting",
            "detail": "intricate details, fine textures, micro details, ultra sharp",
            "composition": "rule of thirds, balanced composition, leading lines, depth",
            "color": "vibrant colors, rich color palette, color harmony, vivid",
            "atmosphere": "atmospheric, moody, ambient, environmental storytelling",
            "depth": "depth of field, bokeh background, foreground interest, layered",
            "cinematic": "cinematic framing, wide angle, anamorphic, film look",
            "texture": "detailed textures, surface detail, material quality, tactile",
        }

        if enhancements is None:
            enhancements = ["lighting", "detail"]

        enhancement_parts = []
        for enh in enhancements:
            if enh in enhancement_map:
                enhancement_parts.append(enhancement_map[enh])

        enhanced = f"{prompt}, {', '.join(enhancement_parts)}"
        return enhanced

    def create_variation_prompts(
        self,
        base_prompt: str,
        num_variations: int = 4,
        variation_type: str = "style"
    ) -> List[PromptResult]:
        """
        Create multiple prompt variations for exploration.

        Args:
            base_prompt: Base prompt to create variations of
            num_variations: Number of variations
            variation_type: Type of variation (style, quality, enhancement)

        Returns:
            List of PromptResults
        """
        variations = []

        if variation_type == "style":
            styles = list(self.STYLES.keys())[:num_variations]
            for style in styles:
                result = self.build_prompt(base_prompt, style=style)
                variations.append(result)

        elif variation_type == "quality":
            qualities = list(self.QUALITY_LEVELS.keys())[:num_variations]
            for quality in qualities:
                result = self.build_prompt(base_prompt, quality=quality)
                variations.append(result)

        elif variation_type == "enhancement":
            enhancement_sets = [
                ["lighting", "detail"],
                ["cinematic", "atmosphere"],
                ["color", "texture"],
                ["depth", "composition"],
            ]
            for enhancements in enhancement_sets[:num_variations]:
                enhanced = self.enhance_prompt(base_prompt, enhancements)
                result = self.build_prompt(enhanced)
                variations.append(result)

        return variations

    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a prompt and provide suggestions.

        Args:
            prompt: Prompt to analyze

        Returns:
            Analysis dictionary with suggestions
        """
        analysis = {
            "original_prompt": prompt,
            "word_count": len(prompt.split()),
            "estimated_tokens": self._estimate_tokens(prompt),
            "has_quality_terms": False,
            "has_style_terms": False,
            "has_negative_indicators": False,
            "suggestions": [],
            "score": 0
        }

        prompt_lower = prompt.lower()

        # Check for quality terms
        quality_terms = ["detailed", "quality", "sharp", "professional", "masterpiece"]
        if any(term in prompt_lower for term in quality_terms):
            analysis["has_quality_terms"] = True
            analysis["score"] += 20
        else:
            analysis["suggestions"].append("Add quality terms like 'highly detailed, sharp focus'")

        # Check for style terms
        style_terms = ["style", "art", "painting", "photo", "render", "illustration"]
        if any(term in prompt_lower for term in style_terms):
            analysis["has_style_terms"] = True
            analysis["score"] += 20
        else:
            analysis["suggestions"].append("Specify an art style for better results")

        # Check for lighting terms
        lighting_terms = ["lighting", "light", "shadow", "bright", "dark", "glow"]
        if any(term in prompt_lower for term in lighting_terms):
            analysis["score"] += 15
        else:
            analysis["suggestions"].append("Consider adding lighting descriptions")

        # Check for composition terms
        composition_terms = ["view", "angle", "close-up", "wide", "portrait", "landscape"]
        if any(term in prompt_lower for term in composition_terms):
            analysis["score"] += 15
        else:
            analysis["suggestions"].append("Add composition details (close-up, wide angle, etc.)")

        # Check for color terms
        color_terms = ["color", "vibrant", "muted", "warm", "cool", "monochrome"]
        if any(term in prompt_lower for term in color_terms):
            analysis["score"] += 10
        else:
            analysis["suggestions"].append("Consider specifying color palette or mood")

        # Token check
        if analysis["estimated_tokens"] > self.max_tokens:
            analysis["suggestions"].append(f"Prompt may be too long (~{analysis['estimated_tokens']} tokens)")
        elif analysis["estimated_tokens"] < 10:
            analysis["suggestions"].append("Prompt is very short. Add more descriptive details.")
            analysis["score"] -= 10

        # Length bonus
        if 15 <= analysis["word_count"] <= 40:
            analysis["score"] += 20

        analysis["score"] = max(0, min(100, analysis["score"]))
        analysis["rating"] = (
            "Excellent" if analysis["score"] >= 80 else
            "Good" if analysis["score"] >= 60 else
            "Fair" if analysis["score"] >= 40 else
            "Needs Improvement"
        )

        return analysis

    @classmethod
    def list_styles(cls):
        """Print available styles"""
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title="🎨 Available Styles")
        table.add_column("Style", style="cyan")
        table.add_column("Positive Keywords", style="green", max_width=50)
        table.add_column("Negative Keywords", style="red", max_width=40)

        for name, config in cls.STYLES.items():
            table.add_row(
                name,
                config["positive"][:50] + "...",
                config["negative"][:40] + "..."
            )

        console.print(table)

    @classmethod
    def list_quality_levels(cls):
        """Print available quality levels"""
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title="⭐ Quality Levels")
        table.add_column("Level", style="cyan")
        table.add_column("Boosters", style="green", max_width=60)

        for name, config in cls.QUALITY_LEVELS.items():
            table.add_row(name, str(config["boosters"])[:60] + "...")

        console.print(table)