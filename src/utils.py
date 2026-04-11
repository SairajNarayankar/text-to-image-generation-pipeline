"""
Utility functions for the Text-to-Image Pipeline
"""

import os
import yaml
import json
import random
import hashlib
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Rich console for pretty printing
console = Console()


def setup_logger(log_level: str = "INFO", log_dir: str = "./logs", log_file: str = "pipeline.log"):
    """
    Configure loguru logger with file and console outputs

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        log_file: Log filename
    """
    os.makedirs(log_dir, exist_ok=True)

    # Remove default logger
    logger.remove()

    # Console logging with color
    logger.add(
        lambda msg: console.print(msg, end=""),
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True
    )

    # File logging
    logger.add(
        os.path.join(log_dir, log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )

    logger.info(f"Logger initialized | Level: {log_level} | Log dir: {log_dir}")
    return logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from: {config_path}")
    return config


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load any YAML file"""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data: Dict, file_path: str, indent: int = 2):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def seed_everything(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.debug(f"Random seed set to: {seed}")


def get_device() -> str:
    """Detect and return the best available device"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS")
    else:
        device = "cpu"
        logger.warning("No GPU detected! Using CPU (this will be slow)")

    return device


def get_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
        "total_gb": torch.cuda.get_device_properties(0).total_mem / (1024 ** 3),
        "free_gb": (torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated()) / (1024 ** 3)
    }


def generate_image_hash(prompt: str, seed: int, settings: Dict) -> str:
    """Generate a unique hash for an image based on its generation parameters"""
    hash_input = f"{prompt}_{seed}_{json.dumps(settings, sort_keys=True)}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def create_output_dirs(base_dir: str = "./output"):
    """Create all necessary output directories"""
    dirs = [
        os.path.join(base_dir, "portfolio"),
        os.path.join(base_dir, "batch"),
        os.path.join(base_dir, "experiments"),
        os.path.join(base_dir, "exports"),
        os.path.join(base_dir, "metadata"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.debug(f"Output directories created at: {base_dir}")


def format_generation_info(
    prompt: str,
    negative_prompt: str,
    settings: Dict,
    elapsed_time: float,
    output_path: str
) -> Dict:
    """Format generation metadata for saving"""
    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "settings": settings,
        "elapsed_time_seconds": round(elapsed_time, 2),
        "output_path": output_path,
        "timestamp": datetime.now().isoformat(),
        "device": get_device(),
        "memory_usage": get_memory_usage()
    }


def print_system_info():
    """Print system information in a formatted table"""
    table = Table(title="🖥️ System Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Python", f"{__import__('sys').version.split()[0]}")
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))

    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda or "N/A")
        table.add_row("GPU", torch.cuda.get_device_name(0))
        mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        table.add_row("GPU Memory", f"{mem:.1f} GB")

    try:
        import diffusers
        table.add_row("Diffusers", diffusers.__version__)
    except ImportError:
        table.add_row("Diffusers", "Not installed")

    console.print(table)


def print_generation_summary(results: list):
    """Print a summary table of generated images"""
    table = Table(title="📊 Generation Summary")
    table.add_column("#", style="dim")
    table.add_column("Prompt", style="cyan", max_width=40)
    table.add_column("Style", style="magenta")
    table.add_column("Time (s)", style="green")
    table.add_column("Output", style="yellow")

    for i, result in enumerate(results, 1):
        table.add_row(
            str(i),
            result.get("prompt", "N/A")[:40] + "...",
            result.get("style", "N/A"),
            str(result.get("elapsed_time", "N/A")),
            result.get("output_path", "N/A")
        )

    console.print(table)