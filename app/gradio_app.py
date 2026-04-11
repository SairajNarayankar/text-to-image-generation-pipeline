"""
Gradio Web UI Module
Interactive web interface for the Text-to-Image Pipeline
"""

import gradio as gr
from typing import Optional
from loguru import logger
from PIL import Image

from src.pipeline import TextToImagePipeline
from src.prompt_engineer import PromptEngineer
from src.scheduler_manager import SchedulerManager


def create_gradio_app(pipeline: Optional[TextToImagePipeline] = None) -> gr.Blocks:
    """
    Create a Gradio web application for the pipeline.

    Args:
        pipeline: Pre-configured TextToImagePipeline instance

    Returns:
        Gradio Blocks application
    """

    # Get available options
    styles = list(PromptEngineer.STYLES.keys())
    quality_levels = list(PromptEngineer.QUALITY_LEVELS.keys())
    schedulers = list(SchedulerManager.SCHEDULERS.keys())

    def generate_image(
        prompt, negative_prompt, style, quality, scheduler,
        steps, guidance, width, height, seed, enhance_prompt,
        auto_enhance
    ):
        """Generate image from UI inputs."""
        if pipeline is None or not pipeline.is_setup:
            return None, "Pipeline not initialized. Please set up the pipeline first."

        try:
            # Update scheduler if changed
            if scheduler:
                pipeline.scheduler_manager.set_scheduler(scheduler)

            seed_val = int(seed) if seed and seed.strip() else None

            result = pipeline.generate(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                style=style if style != "None" else None,
                quality=quality,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=int(width),
                height=int(height),
                seed=seed_val,
                enhance_prompt=enhance_prompt,
                auto_enhance_image=auto_enhance,
                save=True,
            )

            image = result["images"][0]
            info_text = (
                f"✅ Generated in {result['elapsed_time']:.2f}s\n"
                f"📝 Enhanced Prompt: {result['prompt'][:100]}...\n"
                f"⚙️ Steps: {steps} | Guidance: {guidance} | "
                f"Size: {width}x{height}\n"
                f"🎲 Seed: {seed_val or 'random'}\n"
                f"💾 Saved: {result['paths'][0] if result['paths'] else 'N/A'}"
            )
            return image, info_text

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return None, f"❌ Error: {str(e)}"

    def analyze_prompt(prompt):
        """Analyze prompt quality."""
        if not prompt.strip():
            return "Enter a prompt to analyze."

        engineer = PromptEngineer()
        analysis = engineer.analyze_prompt(prompt)

        report = f"""
📊 **Prompt Analysis Report**

📝 **Original:** {analysis['original_prompt']}
📏 **Words:** {analysis['word_count']}
🎫 **Estimated Tokens:** {analysis['estimated_tokens']}
⭐ **Score:** {analysis['score']}/100 ({analysis['rating']})

**Checklist:**
{'✅' if analysis['has_quality_terms'] else '❌'} Quality terms
{'✅' if analysis['has_style_terms'] else '❌'} Style terms

**Suggestions:**
"""
        for suggestion in analysis['suggestions']:
            report += f"💡 {suggestion}\n"

        return report

    def preview_enhanced_prompt(prompt, style, quality):
        """Preview what the enhanced prompt will look like."""
        engineer = PromptEngineer()
        result = engineer.build_prompt(
            base_prompt=prompt,
            style=style if style != "None" else None,
            quality=quality,
        )
        return f"**Positive:**\n{result.positive}\n\n**Negative:**\n{result.negative}"

    # --- BUILD GRADIO UI ---
    with gr.Blocks(
        title="🎨 AI Image Generator",
        theme=gr.themes.Soft(),
        css="""
        .generate-btn {background-color: #667eea !important; color: white !important;}
        .header {text-align: center; margin-bottom: 20px;}
        """
    ) as app:

        gr.Markdown("""
        # 🎨 Text-to-Image Generation Pipeline
        ### Powered by Stable Diffusion & Hugging Face Diffusers
        """)

        with gr.Tabs():
            # --- TAB 1: GENERATE ---
            with gr.Tab("🖼️ Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_input = gr.Textbox(
                            label="✏️ Prompt",
                            placeholder="Describe the image you want to create...",
                            lines=3,
                        )
                        negative_input = gr.Textbox(
                            label="❌ Negative Prompt (optional)",
                            placeholder="What to avoid...",
                            lines=2,
                        )

                        with gr.Row():
                            style_dropdown = gr.Dropdown(
                                choices=["None"] + styles,
                                value="digital_art",
                                label="🎨 Style"
                            )
                            quality_dropdown = gr.Dropdown(
                                choices=quality_levels,
                                value="high",
                                label="⭐ Quality"
                            )

                        with gr.Row():
                            scheduler_dropdown = gr.Dropdown(
                                choices=schedulers,
                                value="dpm_solver_multistep",
                                label="⏱️ Scheduler"
                            )

                        with gr.Row():
                            steps_slider = gr.Slider(
                                minimum=5, maximum=100, value=30, step=1,
                                label="🔢 Inference Steps"
                            )
                            guidance_slider = gr.Slider(
                                minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                                label="🎯 Guidance Scale"
                            )

                        with gr.Row():
                            width_dropdown = gr.Dropdown(
                                choices=[384, 512, 640, 768, 896, 1024],
                                value=512,
                                label="📐 Width"
                            )
                            height_dropdown = gr.Dropdown(
                                choices=[384, 512, 640, 768, 896, 1024],
                                value=512,
                                label="📐 Height"
                            )

                        seed_input = gr.Textbox(
                            label="🎲 Seed (leave empty for random)",
                            placeholder="42"
                        )

                        with gr.Row():
                            enhance_prompt_cb = gr.Checkbox(
                                value=True, label="🔧 Enhance Prompt"
                            )
                            auto_enhance_cb = gr.Checkbox(
                                value=False, label="✨ Auto Enhance Image"
                            )

                        generate_btn = gr.Button(
                            "🎨 Generate Image",
                            variant="primary",
                            elem_classes="generate-btn"
                        )

                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="Generated Image",
                            type="pil"
                        )
                        output_info = gr.Textbox(
                            label="📋 Generation Info",
                            lines=5,
                            interactive=False
                        )

                generate_btn.click(
                    fn=generate_image,
                    inputs=[
                        prompt_input, negative_input, style_dropdown,
                        quality_dropdown, scheduler_dropdown,
                        steps_slider, guidance_slider,
                        width_dropdown, height_dropdown,
                        seed_input, enhance_prompt_cb, auto_enhance_cb
                    ],
                    outputs=[output_image, output_info]
                )

            # --- TAB 2: PROMPT TOOLS ---
            with gr.Tab("📝 Prompt Tools"):
                with gr.Row():
                    with gr.Column():
                        analyze_input = gr.Textbox(
                            label="Enter prompt to analyze",
                            lines=3
                        )
                        analyze_btn = gr.Button("🔍 Analyze Prompt")
                        analyze_output = gr.Markdown(label="Analysis")

                        analyze_btn.click(
                            fn=analyze_prompt,
                            inputs=[analyze_input],
                            outputs=[analyze_output]
                        )

                    with gr.Column():
                        preview_input = gr.Textbox(
                            label="Enter prompt to enhance",
                            lines=3
                        )
                        preview_style = gr.Dropdown(
                            choices=["None"] + styles,
                            value="digital_art",
                            label="Style"
                        )
                        preview_quality = gr.Dropdown(
                            choices=quality_levels,
                            value="high",
                            label="Quality"
                        )
                        preview_btn = gr.Button("👁️ Preview Enhanced Prompt")
                        preview_output = gr.Markdown(label="Enhanced Prompt")

                        preview_btn.click(
                            fn=preview_enhanced_prompt,
                            inputs=[preview_input, preview_style, preview_quality],
                            outputs=[preview_output]
                        )

            # --- TAB 3: STYLE GALLERY ---
            with gr.Tab("🎭 Style Reference"):
                gr.Markdown("## Available Styles")
                for style_name, style_config in PromptEngineer.STYLES.items():
                    with gr.Accordion(f"🎨 {style_name}", open=False):
                        gr.Markdown(f"**Positive:** {style_config['positive']}")
                        gr.Markdown(f"**Negative:** {style_config['negative']}")

            # --- TAB 4: HELP ---
            with gr.Tab("❓ Help"):
                gr.Markdown("""
                ## 📖 User Guide

                ### Quick Start
                1. Enter a descriptive prompt
                2. Select a style and quality level
                3. Adjust generation settings
                4. Click "Generate Image"

                ### Tips for Better Results
                - **Be descriptive:** Include details about subject, style, lighting, colors
                - **Use style presets:** They add optimized keywords automatically
                - **Adjust guidance scale:** Higher = more faithful to prompt (7-9 recommended)
                - **Experiment with steps:** 20-30 for quick, 50+ for high quality
                - **Use seeds:** Same seed + same prompt = same image (reproducibility)

                ### Scheduler Guide
                - **DPM Solver Multistep:** Best overall (fast + high quality)
                - **Euler Ancestral:** More creative/varied results
                - **UniPC:** Fastest, good for drafts
                - **Heun:** Highest quality, slower
                """)

    logger.info("Gradio app created successfully")
    return app


def launch_gradio(pipeline: TextToImagePipeline, share: bool = False, port: int = 7860):
    """Launch the Gradio web application."""
    app = create_gradio_app(pipeline)
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share
    )