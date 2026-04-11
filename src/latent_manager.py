"""
Latent Space Manager Module
Handles latent space operations for the diffusion pipeline
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from loguru import logger
from PIL import Image


class LatentSpaceManager:
    """
    Manages latent space operations for Stable Diffusion.

    Operations:
        - Latent noise generation and manipulation
        - Seed management for reproducibility
        - Latent interpolation (morphing between images)
        - Latent arithmetic (combining concepts)
        - Noise scheduling visualization
        - Latent space exploration
    """

    def __init__(self, pipe, device: str = "cuda"):
        """
        Initialize LatentSpaceManager.

        Args:
            pipe: Stable Diffusion pipeline
            device: Computation device
        """
        self.pipe = pipe
        self.device = device
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.vae_scale_factor = pipe.vae_scale_factor

        logger.info("LatentSpaceManager initialized")

    def create_latent_noise(
        self,
        batch_size: int = 1,
        num_channels: int = 4,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """
        Create initial latent noise tensor.

        Args:
            batch_size: Number of images in batch
            num_channels: Latent channels (4 for SD)
            height: Target image height
            width: Target image width
            seed: Random seed for reproducibility
            dtype: Tensor data type

        Returns:
            Latent noise tensor
        """
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        shape = (batch_size, num_channels, latent_height, latent_width)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        latents = torch.randn(
            shape,
            generator=generator,
            device=self.device,
            dtype=dtype
        )

        logger.debug(
            f"Created latent noise | Shape: {shape} | "
            f"Seed: {seed} | dtype: {dtype}"
        )

        return latents

    def encode_image_to_latent(self, image: Image.Image) -> torch.Tensor:
        """
        Encode a PIL image into latent space using VAE encoder.

        Args:
            image: PIL Image to encode

        Returns:
            Latent representation tensor
        """
        import torchvision.transforms as transforms

        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        image_tensor = transform(image).unsqueeze(0).to(
            device=self.device,
            dtype=self.vae.dtype
        )

        # Encode
        with torch.no_grad():
            latent_dist = self.vae.encode(image_tensor).latent_dist
            latents = latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        logger.debug(f"Image encoded to latent | Shape: {latents.shape}")
        return latents

    def decode_latent_to_image(self, latents: torch.Tensor) -> Image.Image:
        """
        Decode latent representation back to PIL image.

        Args:
            latents: Latent tensor to decode

        Returns:
            Decoded PIL Image
        """
        with torch.no_grad():
            # Scale latents
            scaled_latents = latents / self.vae.config.scaling_factor
            image_tensor = self.vae.decode(scaled_latents).sample

        # Convert to PIL
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
        image_np = (image_np[0] * 255).round().astype(np.uint8)

        return Image.fromarray(image_np)

    def interpolate_latents(
        self,
        latent_a: torch.Tensor,
        latent_b: torch.Tensor,
        num_steps: int = 10,
        method: str = "slerp"
    ) -> List[torch.Tensor]:
        """
        Interpolate between two latent representations.

        Args:
            latent_a: Starting latent
            latent_b: Ending latent
            num_steps: Number of interpolation steps
            method: Interpolation method ('linear' or 'slerp')

        Returns:
            List of interpolated latent tensors
        """
        alphas = torch.linspace(0, 1, num_steps)
        interpolated = []

        for alpha in alphas:
            if method == "slerp":
                interp = self._slerp(latent_a, latent_b, alpha.item())
            elif method == "linear":
                interp = (1 - alpha) * latent_a + alpha * latent_b
            else:
                raise ValueError(f"Unknown interpolation method: {method}")

            interpolated.append(interp)

        logger.info(f"Interpolated {num_steps} latents using {method}")
        return interpolated

    def _slerp(
        self,
        latent_a: torch.Tensor,
        latent_b: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Spherical linear interpolation between two latents.

        Args:
            latent_a: First latent
            latent_b: Second latent
            alpha: Interpolation factor [0, 1]

        Returns:
            Interpolated latent
        """
        # Flatten for computation
        flat_a = latent_a.flatten()
        flat_b = latent_b.flatten()

        # Compute angle
        dot = torch.dot(flat_a, flat_b)
        dot = dot / (torch.norm(flat_a) * torch.norm(flat_b))
        dot = torch.clamp(dot, -1.0, 1.0)

        theta = torch.acos(dot)

        if theta.abs() < 1e-5:
            # Very close, use linear interpolation
            return (1 - alpha) * latent_a + alpha * latent_b

        sin_theta = torch.sin(theta)
        weight_a = torch.sin((1 - alpha) * theta) / sin_theta
        weight_b = torch.sin(alpha * theta) / sin_theta

        return weight_a * latent_a + weight_b * latent_b

    def latent_arithmetic(
        self,
        latents: List[torch.Tensor],
        weights: List[float]
    ) -> torch.Tensor:
        """
        Perform arithmetic operations in latent space.
        E.g., "king - man + woman = queen" style operations.

        Args:
            latents: List of latent tensors
            weights: Corresponding weights (positive to add, negative to subtract)

        Returns:
            Combined latent tensor
        """
        if len(latents) != len(weights):
            raise ValueError("Number of latents must match number of weights")

        result = torch.zeros_like(latents[0])
        for latent, weight in zip(latents, weights):
            result += weight * latent

        logger.debug(f"Latent arithmetic: {len(latents)} latents with weights {weights}")
        return result

    def add_noise_at_strength(
        self,
        latents: torch.Tensor,
        strength: float = 0.5,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Add noise to existing latents at a specific strength level.

        Args:
            latents: Input latents
            strength: Noise strength [0, 1]
            seed: Random seed

        Returns:
            Noised latent tensor
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=self.device,
            dtype=latents.dtype
        )

        noised_latents = (1 - strength) * latents + strength * noise
        logger.debug(f"Added noise at strength {strength}")
        return noised_latents

    def generate_latent_walk(
        self,
        num_frames: int = 30,
        height: int = 512,
        width: int = 512,
        seed_start: int = 42,
        seed_end: int = 123,
        method: str = "slerp"
    ) -> List[torch.Tensor]:
        """
        Generate a smooth walk through latent space (for animations).

        Args:
            num_frames: Number of frames in the walk
            height: Image height
            width: Image width
            seed_start: Starting seed
            seed_end: Ending seed
            method: Interpolation method

        Returns:
            List of latent tensors for each frame
        """
        latent_a = self.create_latent_noise(
            height=height, width=width, seed=seed_start
        )
        latent_b = self.create_latent_noise(
            height=height, width=width, seed=seed_end
        )

        frames = self.interpolate_latents(
            latent_a, latent_b,
            num_steps=num_frames,
            method=method
        )

        logger.info(f"Generated latent walk: {num_frames} frames")
        return frames

    def get_latent_statistics(self, latents: torch.Tensor) -> Dict[str, float]:
        """
        Get statistical information about a latent tensor.

        Args:
            latents: Latent tensor to analyze

        Returns:
            Dictionary of statistics
        """
        return {
            "shape": list(latents.shape),
            "mean": latents.mean().item(),
            "std": latents.std().item(),
            "min": latents.min().item(),
            "max": latents.max().item(),
            "norm": latents.norm().item(),
            "dtype": str(latents.dtype),
            "device": str(latents.device),
            "memory_mb": latents.element_size() * latents.nelement() / (1024 ** 2)
        }