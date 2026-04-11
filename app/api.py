"""
FastAPI REST API Module
RESTful API for the Text-to-Image Pipeline

File: app/api.py
"""

import io
import os
import time
import base64
from typing import Optional, List
from datetime import datetime
from loguru import logger

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from src.pipeline import TextToImagePipeline
from src.prompt_engineer import PromptEngineer
from src.scheduler_manager import SchedulerManager


# ==========================================
# REQUEST / RESPONSE MODELS
# ==========================================

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    style: Optional[str] = Field(None, description="Style preset name")
    quality: Optional[str] = Field("high", description="Quality level")
    num_inference_steps: Optional[int] = Field(30, ge=5, le=100)
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=20.0)
    width: Optional[int] = Field(512, ge=256, le=1024)
    height: Optional[int] = Field(512, ge=256, le=1024)
    seed: Optional[int] = Field(None, description="Random seed")
    num_images: Optional[int] = Field(1, ge=1, le=4)
    enhance_prompt: Optional[bool] = Field(True)
    auto_enhance_image: Optional[bool] = Field(False)
    output_format: Optional[str] = Field("png", description="png, jpeg, webp")


class GenerateResponse(BaseModel):
    success: bool
    images: List[str]
    prompt_used: str
    negative_prompt_used: str
    settings: dict
    elapsed_time: float
    timestamp: str


class AnalyzeRequest(BaseModel):
    prompt: str


class AnalyzeResponse(BaseModel):
    original_prompt: str
    word_count: int
    estimated_tokens: int
    score: int
    rating: str
    suggestions: List[str]


class EnhanceRequest(BaseModel):
    prompt: str
    style: Optional[str] = "digital_art"
    quality: Optional[str] = "high"


class EnhanceResponse(BaseModel):
    original_prompt: str
    enhanced_positive: str
    enhanced_negative: str
    estimated_tokens: int


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    is_ready: bool
    uptime_seconds: float
    total_generations: int


class StatsResponse(BaseModel):
    total_generations: int
    total_images: int
    total_time_seconds: float
    average_time_seconds: float
    model: str
    device: str


# ==========================================
# API FACTORY
# ==========================================

def create_api(pipeline: TextToImagePipeline) -> FastAPI:
    """Create and configure FastAPI application."""

    api_start_time = time.time()

    app = FastAPI(
        title="Text-to-Image Generation API",
        description=(
            "RESTful API for AI image generation using Stable Diffusion.\n\n"
            "## Features\n"
            "- Text-to-image generation with customizable settings\n"
            "- Multiple style presets\n"
            "- Prompt analysis and enhancement\n"
            "- Batch generation support\n"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- HELPERS ---

    def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
        buffer = io.BytesIO()
        image.save(buffer, format=fmt.upper())
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format=fmt.upper())
        buffer.seek(0)
        return buffer.getvalue()

    # --- ENDPOINTS ---

    @app.get("/", tags=["Root"])
    async def root():
        return {"message": "Text-to-Image Generation API", "version": "1.0.0", "docs": "/docs"}

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        stats = pipeline.get_stats() if pipeline.is_setup else {}
        return HealthResponse(
            status="healthy" if pipeline.is_setup else "initializing",
            model=pipeline.model_loader.model_id,
            device=pipeline.device,
            is_ready=pipeline.is_setup,
            uptime_seconds=round(time.time() - api_start_time, 2),
            total_generations=stats.get("total_generations", 0),
        )

    @app.get("/stats", response_model=StatsResponse, tags=["System"])
    async def get_stats():
        if not pipeline.is_setup:
            raise HTTPException(status_code=503, detail="Pipeline not ready")
        stats = pipeline.get_stats()
        return StatsResponse(**stats)

    @app.get("/styles", tags=["Info"])
    async def list_styles():
        return {"styles": list(PromptEngineer.STYLES.keys())}

    @app.get("/quality-levels", tags=["Info"])
    async def list_quality_levels():
        return {"quality_levels": list(PromptEngineer.QUALITY_LEVELS.keys())}

    @app.get("/schedulers", tags=["Info"])
    async def list_schedulers():
        return {
            "schedulers": {
                name: info["description"]
                for name, info in SchedulerManager.SCHEDULERS.items()
            }
        }

    @app.get("/models", tags=["Info"])
    async def list_models():
        from src.model_loader import ModelLoader
        return {"models": ModelLoader.MODEL_REGISTRY}

    @app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
    async def generate_image(request: GenerateRequest):
        if not pipeline.is_setup:
            raise HTTPException(status_code=503, detail="Pipeline not ready")

        try:
            result = pipeline.generate(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                style=request.style,
                quality=request.quality,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                seed=request.seed,
                num_images=request.num_images,
                enhance_prompt=request.enhance_prompt,
                auto_enhance_image=request.auto_enhance_image,
                save=True,
            )

            b64_images = [
                image_to_base64(img, request.output_format)
                for img in result["images"]
            ]

            return GenerateResponse(
                success=True,
                images=b64_images,
                prompt_used=result["prompt"],
                negative_prompt_used=result["negative_prompt"],
                settings=result["settings"],
                elapsed_time=result["elapsed_time"],
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"API generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/generate/stream", tags=["Generation"])
    async def generate_image_stream(request: GenerateRequest):
        """Generate and return image directly as bytes (no base64)."""
        if not pipeline.is_setup:
            raise HTTPException(status_code=503, detail="Pipeline not ready")

        try:
            result = pipeline.generate(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                style=request.style,
                quality=request.quality,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                seed=request.seed,
                enhance_prompt=request.enhance_prompt,
                save=False,
            )

            img_bytes = image_to_bytes(result["images"][0], request.output_format)
            media_types = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}

            return StreamingResponse(
                io.BytesIO(img_bytes),
                media_type=media_types.get(request.output_format, "image/png"),
                headers={"X-Elapsed-Time": str(result["elapsed_time"])}
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/analyze", response_model=AnalyzeResponse, tags=["Prompt Tools"])
    async def analyze_prompt(request: AnalyzeRequest):
        engineer = PromptEngineer()
        analysis = engineer.analyze_prompt(request.prompt)
        return AnalyzeResponse(
            original_prompt=analysis["original_prompt"],
            word_count=analysis["word_count"],
            estimated_tokens=analysis["estimated_tokens"],
            score=analysis["score"],
            rating=analysis["rating"],
            suggestions=analysis["suggestions"],
        )

    @app.post("/enhance", response_model=EnhanceResponse, tags=["Prompt Tools"])
    async def enhance_prompt(request: EnhanceRequest):
        engineer = PromptEngineer()
        result = engineer.build_prompt(
            base_prompt=request.prompt,
            style=request.style,
            quality=request.quality,
        )
        return EnhanceResponse(
            original_prompt=request.prompt,
            enhanced_positive=result.positive,
            enhanced_negative=result.negative,
            estimated_tokens=result.tokens_estimated,
        )

    @app.get("/history", tags=["System"])
    async def get_history(limit: int = Query(default=10, ge=1, le=100)):
        history = pipeline.get_history()
        return {"history": history[-limit:], "total": len(history)}

    return app


def launch_api(pipeline: TextToImagePipeline, host: str = "0.0.0.0", port: int = 8000):
    """Launch the FastAPI server."""
    import uvicorn
    app = create_api(pipeline)
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)