"""
Scheduler Manager Module
Manages different noise schedulers for the diffusion process
"""

from typing import Dict, Optional, Any
from loguru import logger

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    UniPCMultistepScheduler,
    DEISMultistepScheduler,
)


class SchedulerManager:
    """
    Manages noise schedulers for the Stable Diffusion pipeline.

    Different schedulers offer different tradeoffs:
    - Speed vs Quality
    - Deterministic vs Stochastic
    - Low-step vs High-step performance
    """

    # Complete scheduler registry
    SCHEDULERS = {
        # --- RECOMMENDED ---
        "dpm_solver_multistep": {
            "class": DPMSolverMultistepScheduler,
            "description": "DPM-Solver++ (Multistep) - Fast & high quality",
            "recommended_steps": "15-30",
            "speed": "fast",
            "quality": "high",
            "stochastic": False,
        },
        "euler_ancestral": {
            "class": EulerAncestralDiscreteScheduler,
            "description": "Euler Ancestral - Creative, stochastic",
            "recommended_steps": "20-40",
            "speed": "fast",
            "quality": "high",
            "stochastic": True,
        },
        "euler": {
            "class": EulerDiscreteScheduler,
            "description": "Euler - Fast, deterministic",
            "recommended_steps": "20-40",
            "speed": "fast",
            "quality": "high",
            "stochastic": False,
        },
        "unipc": {
            "class": UniPCMultistepScheduler,
            "description": "UniPC - Fast convergence, good for low steps",
            "recommended_steps": "10-25",
            "speed": "very fast",
            "quality": "high",
            "stochastic": False,
        },

        # --- STANDARD ---
        "ddim": {
            "class": DDIMScheduler,
            "description": "DDIM - Classic, deterministic, supports inversion",
            "recommended_steps": "30-50",
            "speed": "medium",
            "quality": "good",
            "stochastic": False,
        },
        "pndm": {
            "class": PNDMScheduler,
            "description": "PNDM - Default for many models",
            "recommended_steps": "30-50",
            "speed": "medium",
            "quality": "good",
            "stochastic": False,
        },
        "lms": {
            "class": LMSDiscreteScheduler,
            "description": "LMS Discrete - Linear multistep",
            "recommended_steps": "30-50",
            "speed": "medium",
            "quality": "good",
            "stochastic": False,
        },

        # --- ADVANCED ---
        "dpm_solver_singlestep": {
            "class": DPMSolverSinglestepScheduler,
            "description": "DPM-Solver++ (Singlestep) - Alternative DPM",
            "recommended_steps": "15-30",
            "speed": "fast",
            "quality": "high",
            "stochastic": False,
        },
        "heun": {
            "class": HeunDiscreteScheduler,
            "description": "Heun - Higher accuracy, 2x computation",
            "recommended_steps": "20-40",
            "speed": "slow",
            "quality": "very high",
            "stochastic": False,
        },
        "kdpm2": {
            "class": KDPM2DiscreteScheduler,
            "description": "KDPM2 - Karras DPM2",
            "recommended_steps": "20-40",
            "speed": "medium",
            "quality": "high",
            "stochastic": False,
        },
        "kdpm2_ancestral": {
            "class": KDPM2AncestralDiscreteScheduler,
            "description": "KDPM2 Ancestral - Stochastic variant",
            "recommended_steps": "20-40",
            "speed": "medium",
            "quality": "high",
            "stochastic": True,
        },
        "deis": {
            "class": DEISMultistepScheduler,
            "description": "DEIS - Diffusion Exponential Integrator",
            "recommended_steps": "15-30",
            "speed": "fast",
            "quality": "high",
            "stochastic": False,
        },
        "ddpm": {
            "class": DDPMScheduler,
            "description": "DDPM - Original scheduler, requires many steps",
            "recommended_steps": "100-1000",
            "speed": "very slow",
            "quality": "baseline",
            "stochastic": True,
        },
    }

    def __init__(self, pipe=None):
        """
        Initialize SchedulerManager

        Args:
            pipe: The Stable Diffusion pipeline
        """
        self.pipe = pipe
        self.current_scheduler = None
        self.original_config = None

        if pipe is not None:
            self.original_config = pipe.scheduler.config
            self.current_scheduler = type(pipe.scheduler).__name__

        logger.info(f"SchedulerManager initialized | Current: {self.current_scheduler}")

    def set_scheduler(
        self,
        scheduler_name: str,
        custom_config: Optional[Dict[str, Any]] = None
    ):
        """
        Switch to a different scheduler

        Args:
            scheduler_name: Name from SCHEDULERS registry
            custom_config: Optional custom configuration overrides
        """
        if self.pipe is None:
            raise RuntimeError("No pipeline set. Pass pipe in constructor or use set_pipe()")

        if scheduler_name not in self.SCHEDULERS:
            available = ", ".join(self.SCHEDULERS.keys())
            raise ValueError(
                f"Unknown scheduler: {scheduler_name}. "
                f"Available: {available}"
            )

        scheduler_info = self.SCHEDULERS[scheduler_name]
        SchedulerClass = scheduler_info["class"]

        # Build config
        config = dict(self.pipe.scheduler.config)
        if custom_config:
            config.update(custom_config)

        # Create and set new scheduler
        try:
            self.pipe.scheduler = SchedulerClass.from_config(config)
            self.current_scheduler = scheduler_name
            logger.info(
                f"✅ Scheduler set to: {scheduler_name} "
                f"({scheduler_info['description']})"
            )
        except Exception as e:
            logger.error(f"Failed to set scheduler {scheduler_name}: {e}")
            raise

    def set_pipe(self, pipe):
        """Set or update the pipeline reference"""
        self.pipe = pipe
        self.original_config = pipe.scheduler.config
        self.current_scheduler = type(pipe.scheduler).__name__

    def get_recommendation(self, priority: str = "balanced") -> str:
        """
        Get scheduler recommendation based on priority

        Args:
            priority: "speed", "quality", or "balanced"

        Returns:
            Recommended scheduler name
        """
        recommendations = {
            "speed": "unipc",
            "quality": "heun",
            "balanced": "dpm_solver_multistep",
            "creative": "euler_ancestral",
            "deterministic": "euler",
            "low_steps": "unipc",
        }

        scheduler = recommendations.get(priority, "dpm_solver_multistep")
        info = self.SCHEDULERS[scheduler]
        logger.info(
            f"Recommendation for '{priority}': {scheduler} "
            f"(Steps: {info['recommended_steps']})"
        )
        return scheduler

    def get_scheduler_info(self, scheduler_name: Optional[str] = None) -> Dict:
        """Get detailed info about a scheduler"""
        name = scheduler_name or self.current_scheduler
        if name in self.SCHEDULERS:
            return {
                "name": name,
                **self.SCHEDULERS[name],
                "class": self.SCHEDULERS[name]["class"].__name__
            }
        return {"name": name, "description": "Unknown scheduler"}

    def reset_to_default(self):
        """Reset scheduler to the original model default"""
        if self.original_config and self.pipe:
            self.pipe.scheduler = type(self.pipe.scheduler).from_config(self.original_config)
            self.current_scheduler = type(self.pipe.scheduler).__name__
            logger.info(f"Scheduler reset to default: {self.current_scheduler}")

    @classmethod
    def list_schedulers(cls):
        """Print a formatted table of all available schedulers"""
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title="⏱️ Available Schedulers")
        table.add_column("Name", style="cyan")
        table.add_column("Speed", style="green")
        table.add_column("Quality", style="yellow")
        table.add_column("Steps", style="magenta")
        table.add_column("Stochastic", style="red")
        table.add_column("Description", style="dim", max_width=40)

        for name, info in cls.SCHEDULERS.items():
            table.add_row(
                name,
                info["speed"],
                info["quality"],
                info["recommended_steps"],
                "✓" if info["stochastic"] else "✗",
                info["description"]
            )

        console.print(table)