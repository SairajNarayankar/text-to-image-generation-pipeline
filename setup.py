from setuptools import setup, find_packages

setup(
    name="text-to-image-pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A production-grade Text-to-Image Generation Pipeline using Stable Diffusion",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/text-to-image-pipeline",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.25.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0.1",
        "loguru>=0.7.0",
        "rich>=13.7.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
        "web": ["gradio", "fastapi", "uvicorn"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
    ],
)