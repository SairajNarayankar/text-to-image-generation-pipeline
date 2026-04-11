# 📝 Prompt Engineering Guide

## Prompt Structure

A well-structured prompt follows this pattern:
[Subject], [Details], [Style], [Lighting], [Quality], [Composition]

### Example:
A majestic lion standing on a rocky cliff,
golden mane flowing in the wind,
digital art style, trending on artstation,
dramatic sunset lighting, volumetric rays,
highly detailed, sharp focus, 8k resolution,
wide angle shot, epic composition

## Available Styles

| Style | Best For |
|-------|----------|
| photorealistic | Product shots, portraits, landscapes |
| digital_art | Concept art, illustrations |
| oil_painting | Fine art, classical themes |
| watercolor | Soft, artistic compositions |
| anime | Japanese-style illustrations |
| minimalist | Clean designs, UI mockups |
| cyberpunk | Futuristic, neon-lit scenes |
| vintage | Retro, nostalgic imagery |
| 3d_render | Product design, architecture |
| sketch | Concept sketches, line art |
| fantasy | Magical, mythical scenes |
| pop_art | Bold, graphic designs |
| cinematic | Movie stills, dramatic scenes |
| isometric | Game assets, dioramas |

## Quality Levels

| Level | Use Case |
|-------|----------|
| draft | Quick testing |
| standard | General purpose |
| high | Portfolio quality |
| ultra | Maximum quality |

## Prompt Tips

### DO:
- Be specific and descriptive
- Include lighting descriptions
- Specify art style explicitly
- Mention composition (close-up, wide angle)
- Use quality boosters (detailed, sharp, professional)

### DON'T:
- Write long paragraphs (use comma-separated terms)
- Use contradictory terms
- Exceed 77 tokens (for SD 1.5)
- Include text you want rendered (SD struggles with text)

## Emphasis Weighting

Increase importance of terms using `(term:weight)` syntax:

```python
emphasis = {"golden light": 1.5, "sharp focus": 1.3}

Negative Prompts
Available categories:

quality: Removes low quality artifacts
anatomy: Fixes body proportion issues
face: Improves facial features
hands: Reduces hand deformities
composition: Improves framing
watermark: Removes text/logos


---

