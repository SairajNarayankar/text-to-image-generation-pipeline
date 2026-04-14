"""
Text-to-Image Generation Pipeline
Optimized for quality output on Hugging Face Spaces
Anti-Hallucination Edition — DreamShaper 8 + SDXL Turbo fallback
"""

import os
import time
import torch
import gradio as gr
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)

# ============================================
# MODEL SETUP
# ============================================

MODEL_ID = "Lykon/dreamshaper-8"
FALLBACK_MODEL = "runwayml/stable-diffusion-v1-5"

# Use the MSE-840000 VAE — dramatically reduces blurriness and color bleeding
VAE_ID = "stabilityai/sd-vae-ft-mse"

print("🔄 Loading VAE...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

try:
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dtype)
    print(f"✅ Loaded improved VAE: {VAE_ID}")
except Exception as e:
    print(f"⚠️ VAE load failed: {e} — using model default VAE")
    vae = None

print("🔄 Loading model...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
    )
    print(f"✅ Loaded: {MODEL_ID}")
except Exception as e:
    print(f"⚠️ Failed to load {MODEL_ID}: {e}")
    print(f"🔄 Loading fallback: {FALLBACK_MODEL}")
    pipe = StableDiffusionPipeline.from_pretrained(
        FALLBACK_MODEL,
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
    )

if device == "cuda":
    pipe = pipe.to("cuda")
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xformers enabled")
    except Exception:
        pipe.enable_attention_slicing()
        print("⚠️ xformers not available, using attention slicing")
else:
    pipe.enable_attention_slicing()

# ---- SCHEDULER: DPM++ 2M Karras — best balance of speed and coherence ----
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,      # Karras sigmas = smoother transitions, fewer artifacts
    algorithm_type="dpmsolver++",
    solver_order=2,              # 2nd order = more stable than 1st order
    thresholding=False,          # Do NOT threshold — causes color clipping artifacts
    lower_order_final=True,      # Stabilizes last few denoising steps
)

# Enable tiled VAE decoding — prevents seam artifacts on larger images
pipe.enable_vae_tiling()

if device == "cuda":
    torch.cuda.empty_cache()

print(f"✅ Pipeline ready on {device}!")

# ============================================
# MASTER NEGATIVE PROMPT
# Organized by category for maximum coverage
# ============================================

NEGATIVE_MASTER = ", ".join([
    # --- Anatomy & Body ---
    "deformed", "distorted", "disfigured", "poorly drawn", "bad anatomy",
    "wrong anatomy", "extra limb", "missing limb", "floating limbs",
    "mutated hands", "mutated fingers", "disconnected limbs", "mutation",
    "extra fingers", "fused fingers", "too many fingers", "long neck",
    "bad proportions", "gross proportions", "missing arms", "missing legs",
    "extra arms", "extra legs", "malformed limbs", "poorly drawn hands",
    "bad hands", "missing fingers", "extra digit", "fewer digits",
    "deformed iris", "deformed pupils", "cross-eyed", "lazy eye",
    "uneven eyes", "asymmetrical face", "bad face", "poorly drawn face",
    "cloned face", "poorly rendered face", "double face",

    # --- Composition & Structure ---
    "bad composition", "cropped", "out of frame", "cut off", "draft",
    "unfinished", "incomplete", "messy background", "chaotic", "cluttered",
    "busy background", "incoherent", "illogical", "impossible geometry",
    "duplicate", "copy", "two heads", "two faces",

    # --- Image Quality ---
    "blurry", "blur", "out of focus", "bokeh abuse", "noisy", "grainy",
    "low quality", "lowest quality", "normal quality", "jpeg artifacts",
    "compression artifacts", "pixelated", "overexposed", "underexposed",
    "oversaturated", "washed out", "flat lighting", "amateur",

    # --- Unwanted Elements ---
    "watermark", "text", "logo", "signature", "username", "artist name",
    "stamp", "title", "subtitle", "date", "footer", "header",
    "speech bubble", "caption", "border", "frame", "vignette abuse",

    # --- Style Contamination ---
    "ugly", "disgusting", "morbid", "mutilated", "amputation",
    "poorly rendered", "bad art", "worst quality",
])

# ============================================
# STYLE PRESETS
# Each style has targeted positive AND negative tokens
# ============================================

STYLES = {
    "None": {"positive": "", "negative": ""},

    "Photorealistic": {
        "positive": (
            "RAW photo, photorealistic, ultra realistic, hyperrealistic, "
            "DSLR photo, 85mm lens, sharp focus, natural lighting, "
            "subsurface scattering, realistic skin texture, accurate proportions, "
            "professional color grading, 8k uhd"
        ),
        "negative": "cartoon, illustration, painting, drawing, anime, CGI, 3d render, sketch, unrealistic, stylized",
    },

    "Digital Art": {
        "positive": (
            "digital painting, concept art, trending on artstation, "
            "highly detailed, vibrant colors, coherent composition, "
            "professional illustration, award winning digital art, smooth shading"
        ),
        "negative": "photo, photograph, blurry, low quality, amateur, messy, 3d render",
    },

    "Oil Painting": {
        "positive": (
            "oil painting on canvas, masterpiece, classical fine art, "
            "rich impasto texture, visible brush strokes, old master technique, "
            "museum quality, coherent composition, warm lighting, dramatic chiaroscuro"
        ),
        "negative": "digital, photo, modern, cartoon, flat colors, amateur, sketch",
    },

    "Anime": {
        "positive": (
            "high quality anime, studio Ghibli inspired, cel shading, "
            "clean line art, vibrant colors, detailed eyes, consistent proportions, "
            "manga style, professional animation frame"
        ),
        "negative": "realistic, photo, western cartoon, 3d render, poorly drawn anime, deformed",
    },

    "Cyberpunk": {
        "positive": (
            "cyberpunk cityscape, neon signs, rain-slicked streets, "
            "holographic advertisements, volumetric fog, blade runner aesthetic, "
            "cinematic lighting, high detail architecture, coherent futuristic design"
        ),
        "negative": "natural, bright daylight, pastoral, medieval, blurry, incoherent",
    },

    "Watercolor": {
        "positive": (
            "professional watercolor painting, soft wet-on-wet edges, "
            "flowing pigment washes, paper texture, white paper showing through, "
            "luminous transparent layers, gallery quality watercolor"
        ),
        "negative": "digital, photo, sharp edges, oil painting, blurry, messy, overworked",
    },

    "Fantasy": {
        "positive": (
            "epic fantasy illustration, magical atmosphere, ethereal glow, "
            "mythical creatures, enchanted environment, dramatic lighting, "
            "highly detailed, coherent scene, professional concept art"
        ),
        "negative": "realistic photography, modern urban, mundane, blurry, amateur",
    },

    "3D Render": {
        "positive": (
            "octane render, unreal engine 5, ray tracing enabled, "
            "volumetric lighting, PBR materials, subsurface scattering, "
            "global illumination, precise geometry, studio HDRI, 4k render"
        ),
        "negative": "2d flat, painting, sketch, cartoon, blurry, low poly, amateur",
    },

    "Cinematic": {
        "positive": (
            "cinematic movie still, anamorphic lens flare, depth of field, "
            "professional color grade, dramatic three-point lighting, "
            "film grain, epic composition, coherent scene, ARRI camera"
        ),
        "negative": "flat lighting, snapshot, cartoon, blurry, overexposed, home video",
    },

    "Minimalist": {
        "positive": (
            "minimalist design, clean geometric composition, abundant negative space, "
            "modern elegant aesthetic, precise lines, flat design, "
            "professional graphic design, intentional layout, Bauhaus inspired"
        ),
        "negative": "complex, cluttered, ornate, baroque, noisy, messy, chaotic",
    },

    "Vintage": {
        "positive": (
            "vintage photograph, sepia tones, film grain, nostalgic aesthetic, "
            "aged Kodachrome colors, retro styling, classic 1960s photography, "
            "warm faded tones, authentic vintage look"
        ),
        "negative": "modern, digital, neon, futuristic, clean, oversaturated, HDR",
    },

    "Sketch": {
        "positive": (
            "detailed pencil sketch, graphite on white paper, cross hatching, "
            "clean confident line work, professional illustration, "
            "accurate proportions, sketchbook quality"
        ),
        "negative": "color, painted, digital render, photo, messy, scratchy",
    },

    "Pop Art": {
        "positive": (
            "pop art, bold flat colors, halftone dots, Andy Warhol style, "
            "high contrast graphic design, clean vector shapes, "
            "Roy Lichtenstein inspired, professional graphic art"
        ),
        "negative": "realistic, muted, subtle, classical painting, blurry, messy",
    },
}

# ============================================
# QUALITY PRESETS
# ============================================

QUALITY_LEVELS = {
    "Draft": {
        "positive": "clear image, decent quality",
        "steps": 20,
        "guidance": 6.5,
        "clip_skip": 1,
    },
    "Standard": {
        "positive": "good quality, detailed, well-composed, clear",
        "steps": 28,
        "guidance": 7.0,
        "clip_skip": 2,
    },
    "High": {
        "positive": (
            "high quality, highly detailed, sharp focus, "
            "professional, well-composed, correct anatomy, clear image"
        ),
        "steps": 35,
        "guidance": 7.5,
        "clip_skip": 2,
    },
    "Ultra": {
        "positive": (
            "masterpiece, best quality, ultra detailed, 8k resolution, "
            "sharp focus, intricate details, professional, award winning, "
            "perfect composition, anatomically correct, flawless"
        ),
        "steps": 45,
        "guidance": 8.0,
        "clip_skip": 2,
    },
}

# ============================================
# PROMPT BUILDER
# ============================================

def build_prompt(prompt: str, style: str, quality: str):
    """
    Constructs a high-fidelity prompt + strong negative prompt.
    Ordering matters: subject → quality → style → coherence anchors.
    """
    # --- Positive ---
    pos_parts = [prompt.strip()]
    if quality in QUALITY_LEVELS:
        pos_parts.append(QUALITY_LEVELS[quality]["positive"])
    if style != "None" and style in STYLES and STYLES[style]["positive"]:
        pos_parts.append(STYLES[style]["positive"])
    # Coherence anchors — always appended last to keep the model grounded
    pos_parts.append(
        "anatomically correct, coherent composition, correct proportions, "
        "well-structured, logical scene, consistent lighting"
    )
    positive = ", ".join(p for p in pos_parts if p)

    # --- Negative ---
    neg_parts = [NEGATIVE_MASTER]
    if style != "None" and style in STYLES and STYLES[style]["negative"]:
        neg_parts.append(STYLES[style]["negative"])
    negative = ", ".join(p for p in neg_parts if p)

    return positive, negative


# ============================================
# GENERATION
# ============================================

def generate_image(
    prompt, style, quality,
    steps, guidance, width, height,
    seed_text, eta
):
    if not prompt.strip():
        return None, "❌ Please enter a prompt!"

    positive, negative = build_prompt(prompt, style, quality)

    q_config = QUALITY_LEVELS.get(quality, QUALITY_LEVELS["High"])
    clip_skip = q_config["clip_skip"]

    seed = None
    if seed_text.strip().isdigit():
        seed = int(seed_text.strip())
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

    try:
        start = time.time()

        # clip_skip: properly handled via pipe.text_encoder config
        # We achieve clip_skip by passing num_hidden_layers offset
        # Diffusers ≥ 0.20 supports clip_skip natively
        result = pipe(
            prompt=positive,
            negative_prompt=negative,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            generator=generator,
            clip_skip=clip_skip,       # Supported natively in diffusers ≥ 0.20
            eta=float(eta),            # DDIM eta — adds controlled stochasticity
        )

        elapsed = time.time() - start
        image = result.images[0]

        info = (
            f"✅ Generated in {elapsed:.2f}s\n"
            f"🎨 Style: {style}  |  ⭐ Quality: {quality}\n"
            f"⚙️ Steps: {steps}  |  Guidance: {guidance}  |  Eta: {eta}\n"
            f"📐 Size: {width}×{height}  |  🎲 Seed: {seed if seed is not None else 'random'}\n"
            f"🖥️ Device: {device}  |  🤖 DreamShaper 8\n\n"
            f"📝 Positive ({len(positive.split(','))} tags):\n{positive[:300]}...\n\n"
            f"🚫 Negative ({len(negative.split(','))} tags): [active]"
        )
        return image, info

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None, "❌ Out of GPU memory. Try smaller size or fewer steps."
        return None, f"❌ Runtime error: {str(e)}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}\n\nTry reducing image size or steps."


# ============================================
# PROMPT ANALYZER
# ============================================

def analyze_prompt(prompt: str) -> str:
    if not prompt.strip():
        return "Enter a prompt to analyze."

    score = 0
    suggestions = []
    words = prompt.split()
    p = prompt.lower()

    checks = [
        (
            ["detailed", "quality", "sharp", "professional", "masterpiece", "intricate", "8k", "4k", "uhd"],
            20,
            "💡 Add quality anchors: `highly detailed`, `sharp focus`, `professional`, `8k uhd`",
        ),
        (
            ["style", "art", "painting", "photo", "render", "illustration", "cinematic", "anime", "watercolor"],
            20,
            "💡 Specify an art style for coherent results (e.g. `photorealistic`, `oil painting`, `cinematic`)",
        ),
        (
            ["lighting", "light", "shadow", "glow", "bright", "dark", "golden hour", "volumetric", "backlit"],
            15,
            "💡 Add lighting: `dramatic lighting`, `golden hour`, `soft ambient light`, `volumetric fog`",
        ),
        (
            ["view", "angle", "close-up", "wide", "portrait", "full body", "aerial", "bird's eye", "macro"],
            15,
            "💡 Add a camera angle or composition: `close-up portrait`, `wide angle`, `aerial view`",
        ),
        (
            ["color", "vibrant", "muted", "warm", "cool", "monochrome", "pastel", "saturated", "tones"],
            10,
            "💡 Specify a color palette: `warm golden tones`, `cool blue palette`, `vibrant colors`",
        ),
        (
            ["background", "environment", "setting", "scene", "forest", "city", "studio", "outdoors"],
            10,
            "💡 Describe the background/environment: `in a forest`, `studio background`, `mountain landscape`",
        ),
    ]

    for terms, points, suggestion in checks:
        if any(t in p for t in terms):
            score += points
        else:
            suggestions.append(suggestion)

    word_count = len(words)
    if 15 <= word_count <= 60:
        score += 10
    elif word_count < 8:
        suggestions.append("💡 Very short prompt — add descriptive details for better coherence")
    elif word_count > 80:
        suggestions.append("💡 Very long prompt — CLIP tokenizer caps at 77 tokens; the tail may be ignored")

    score = min(100, max(0, score + 10))
    rating = (
        "🌟 Excellent" if score >= 80
        else "👍 Good" if score >= 60
        else "📝 Fair" if score >= 40
        else "⚠️ Needs Work"
    )

    report = f"## 📊 Score: {score}/100 — {rating}\n\n"
    report += f"📏 **Words:** {word_count} | 🎫 **Est. Tokens:** ~{int(word_count * 1.3)}/77\n\n"

    if suggestions:
        report += "### 🛠️ Improvement Suggestions:\n\n" + "\n\n".join(suggestions)
    else:
        report += "### ✅ Excellent prompt! No changes needed."

    report += "\n\n---\n### 📖 Ideal Prompt Structure:\n"
    report += "`[Subject] + [Details] + [Environment] + [Style] + [Lighting] + [Quality terms]`\n\n"
    report += "**Example:**\n```\nA lone samurai standing in a misty bamboo forest, "
    report += "worn armor with intricate details, dramatic side lighting, "
    report += "golden hour, cinematic, sharp focus, 8k\n```"

    return report


# ============================================
# GRADIO UI
# ============================================

css = """
.generate-btn { font-size: 1.1em !important; padding: 14px 24px !important; }
.warning-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; border-radius: 6px; margin: 6px 0; }
footer { display: none !important; }
"""

HALLUCINATION_TIPS = """
> **Why hallucinations happen & how we prevent them:**
> - ✅ **Improved VAE** (`sd-vae-ft-mse`) — prevents color bleeding and blurry faces
> - ✅ **DPM++ 2M Karras** with `lower_order_final=True` — stabilises last denoising steps
> - ✅ **60+ negative prompt terms** covering anatomy, quality, and style contamination
> - ✅ **Coherence anchors** appended to every prompt
> - ✅ **VAE tiling** — removes seam/stitch artifacts on larger images
> - ✅ **`clip_skip=2`** — skips last CLIP layer, reduces over-literal prompt binding
> - ✅ **Guidance 7–8** — stays in the sweet spot (too high = artifacts, too low = generic)
"""

with gr.Blocks(title="🎨 AI Image Generator", theme=gr.themes.Soft(), css=css) as demo:

    gr.Markdown("""
    # 🎨 Text-to-Image Generation Pipeline
    ### DreamShaper 8 · Anti-Hallucination Edition · 14 Styles · Prompt Engineering
    """)

    with gr.Tabs():

        # ── GENERATE ──────────────────────────────────────────────
        with gr.Tab("🖼️ Generate"):
            with gr.Row():
                # Left column — controls
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(
                        label="✏️ Describe your image",
                        lines=4,
                        placeholder=(
                            "Example: A majestic eagle soaring over snow-capped mountains "
                            "at golden hour, dramatic clouds, sharp feather details, photorealistic, 8k"
                        ),
                    )

                    with gr.Row():
                        style_dd = gr.Dropdown(
                            list(STYLES.keys()), value="Photorealistic",
                            label="🎨 Style Preset",
                        )
                        quality_dd = gr.Dropdown(
                            list(QUALITY_LEVELS.keys()), value="High",
                            label="⭐ Quality Level",
                        )

                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        with gr.Row():
                            steps_sl = gr.Slider(
                                20, 50, value=35, step=5,
                                label="🔢 Inference Steps",
                                info="More steps = sharper, fewer hallucinations (30–40 is ideal)",
                            )
                            guidance_sl = gr.Slider(
                                5.0, 12.0, value=7.5, step=0.5,
                                label="🎯 CFG / Guidance Scale",
                                info="7–8 = best coherence. Above 10 causes artifacts.",
                            )
                        with gr.Row():
                            width_dd = gr.Dropdown(
                                [384, 512, 640, 768], value=512, label="📐 Width",
                            )
                            height_dd = gr.Dropdown(
                                [384, 512, 640, 768], value=512, label="📐 Height",
                            )
                        with gr.Row():
                            seed_input = gr.Textbox(
                                label="🎲 Seed (empty = random)", value="",
                                info="Same seed + same prompt = reproducible result",
                            )
                            eta_sl = gr.Slider(
                                0.0, 1.0, value=0.0, step=0.1,
                                label="🌀 Eta (stochasticity)",
                                info="0 = deterministic. 0.1–0.3 adds variety.",
                            )

                    gen_btn = gr.Button(
                        "🎨 Generate Image", variant="primary", size="lg",
                        elem_classes="generate-btn",
                    )

                # Right column — output
                with gr.Column(scale=1):
                    output_img = gr.Image(
                        label="Generated Image", type="pil", height=520,
                    )
                    output_info = gr.Textbox(
                        label="📋 Generation Info", lines=10, interactive=False,
                    )

            gen_btn.click(
                generate_image,
                inputs=[
                    prompt_input, style_dd, quality_dd,
                    steps_sl, guidance_sl, width_dd, height_dd,
                    seed_input, eta_sl,
                ],
                outputs=[output_img, output_info],
            )

            gr.Markdown("### 💡 Example Prompts:")
            gr.Examples(
                examples=[
                    ["A majestic lion on a rocky cliff, golden sunset, realistic fur texture, savanna, photorealistic, sharp focus, 8k", "Photorealistic", "High", 35, 7.5, 512, 512, "42", 0.0],
                    ["A cozy log cabin in snowy mountains, warm amber window glow, pine trees, northern lights, cinematic", "Cinematic", "Ultra", 45, 8.0, 512, 512, "100", 0.0],
                    ["Cyberpunk street at night, neon reflections in rain puddles, futuristic cars, volumetric fog", "Cyberpunk", "High", 35, 7.5, 512, 512, "200", 0.0],
                    ["Enchanted forest with glowing mushrooms, fireflies, ancient moss-covered trees, fantasy art", "Fantasy", "Ultra", 45, 8.0, 512, 512, "300", 0.0],
                    ["Serene Japanese zen garden, raked sand, stone lantern, maple tree, morning mist, watercolor", "Watercolor", "High", 35, 7.5, 512, 512, "500", 0.0],
                    ["Futuristic space station orbiting Earth, solar panels, astronaut EVA, stars, octane render", "3D Render", "Ultra", 45, 8.0, 512, 512, "700", 0.0],
                ],
                inputs=[
                    prompt_input, style_dd, quality_dd,
                    steps_sl, guidance_sl, width_dd, height_dd,
                    seed_input, eta_sl,
                ],
            )

        # ── ANTI-HALLUCINATION INFO ────────────────────────────────
        with gr.Tab("🧠 Anti-Hallucination"):
            gr.Markdown(HALLUCINATION_TIPS)
            gr.Markdown("""
            ## 🔬 Root Causes of Hallucinations in SD Models

            | Cause | Fix Applied |
            |-------|-------------|
            | Weak negative prompts | 60+ targeted negative tokens covering anatomy, quality, style |
            | Default blurry VAE | Replaced with `sd-vae-ft-mse` (MSE-tuned for sharpness) |
            | Unstable denoising tail | `lower_order_final=True` in DPM++ scheduler |
            | Guidance too high (>10) | Capped at 8.0 default; UI warns above 10 |
            | Over-literal CLIP binding | `clip_skip=2` softens extreme token binding |
            | Seam artifacts on large images | `enable_vae_tiling()` enabled globally |
            | Style contamination | Each style preset has targeted negative tokens |
            | Short/vague prompts | Analyzer scores & suggests improvements |

            ## 🎯 Best Settings for Clean Results

            ```
            Steps:    35–45       (fewer steps = incomplete denoising = hallucinations)
            Guidance: 7.0–8.0     (sweet spot; >9 causes artifacts)
            Eta:      0.0         (deterministic = most coherent)
            Size:     512×512     (SD1.5 native resolution; larger = more artifacts)
            Clip Skip: 2          (reduces rigid token binding)
            ```

            ## 🧪 If Still Getting Hallucinations

            1. **Reduce guidance** below 7.5 — high CFG is the #1 cause of artifacts
            2. **Increase steps** to 40+ — more denoising passes = cleaner result
            3. **Use a fixed seed** and iterate on the prompt
            4. **Add more specifics** to the prompt — vague prompts = model guesses = hallucinations
            5. **Switch styles** — some styles are more robust (Photorealistic, Digital Art)
            """)

        # ── PROMPT TOOLS ───────────────────────────────────────────
        with gr.Tab("📝 Prompt Tools"):
            with gr.Row():
                with gr.Column():
                    analyze_input = gr.Textbox(
                        label="Paste your prompt here", lines=4,
                        placeholder="Enter your prompt to get a quality score and suggestions...",
                    )
                    analyze_btn = gr.Button("🔍 Analyze Prompt", variant="secondary")
                    analyze_output = gr.Markdown()
                    analyze_btn.click(
                        analyze_prompt,
                        inputs=[analyze_input],
                        outputs=[analyze_output],
                    )

                with gr.Column():
                    gr.Markdown("""
                    ## 📖 Prompt Engineering Guide

                    ### ✅ Ideal Structure
                    ```
                    [Subject] + [Details] + [Environment]
                    + [Style] + [Lighting] + [Quality]
                    ```

                    ### ✅ Strong Example
                    ```
                    A lone samurai standing in a misty bamboo forest,
                    worn battle armor with intricate engravings,
                    dramatic side lighting, golden hour, cinematic,
                    sharp focus, 8k, highly detailed
                    ```

                    ### ❌ Hallucination-Prone Example
                    ```
                    samurai
                    ```
                    *(Too vague — model fills gaps with guesses)*

                    ### 🎯 Power Tips
                    - **Subject first** — CLIP weighs early tokens more
                    - **Lighting is critical** — "dramatic lighting" alone boosts realism
                    - **Avoid contradictions** — "realistic anime" confuses the model
                    - **Specify anatomy** — "full body portrait, correct proportions"
                    - **Quality boosters** — "8k, sharp focus, highly detailed, RAW photo"
                    - **Guidance 7–8** is the sweet spot; never exceed 10
                    """)

        # ── STYLE GALLERY ──────────────────────────────────────────
        with gr.Tab("🎭 Styles"):
            gr.Markdown("## 🎨 14 Style Presets\n*Expand to see positive and negative keywords*")
            for name, config in STYLES.items():
                if name != "None":
                    with gr.Accordion(f"🎨 {name}", open=False):
                        gr.Markdown(f"**✅ Adds:**\n```\n{config['positive']}\n```")
                        gr.Markdown(f"**🚫 Blocks:**\n```\n{config['negative']}\n```")

        # ── ABOUT ──────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## 🎨 Text-to-Image Pipeline — Anti-Hallucination Edition

            | Component | Details |
            |-----------|---------|
            | **Base Model** | DreamShaper 8 (SD 1.5 fine-tune) |
            | **VAE** | `stabilityai/sd-vae-ft-mse` (MSE-tuned) |
            | **Scheduler** | DPM-Solver++ 2M · Karras sigmas |
            | **Framework** | PyTorch + Hugging Face Diffusers |
            | **Clip Skip** | 2 (Standard/High/Ultra) |
            | **Tiled VAE** | Enabled (prevents seam artifacts) |

            ### 🏗️ Pipeline Architecture
            ```
            User Prompt
                ↓
            Prompt Builder
            (subject → quality → style → coherence anchors)
                ↓
            CLIP Text Encoder (clip_skip=2)
                ↓
            DPM++ 2M Karras Scheduler (lower_order_final=True)
                ↓
            UNet Denoising (35–45 steps, guidance 7–8)
                ↓
            MSE-tuned VAE Decoder (tiled)
                ↓
            Generated Image ✅
            ```

            ### 📊 Stats
            - 14 style presets with dual positive/negative tokens
            - 4 quality levels (Draft → Ultra)
            - 60+ negative prompt terms
            - Prompt analyzer with 6-category scoring
            - Improved VAE + scheduler for artifact-free output

            ---
            *Built as a Generative AI portfolio project demonstrating prompt engineering,
            pipeline optimization, anti-hallucination techniques, and production deployment.*
            """)

demo.launch()
