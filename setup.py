from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = ROOT / "README.md"

# NOTE:
# - JAX's current CUDA 12 extra is `jax[cuda12]`.
# - If you need the locally-installed CUDA variant instead, swap to `jax[cuda12-local]`.
# - The `simulations` workspace member from uv is represented here as a local package
#   assumption; adjust package discovery if your repo layout differs.

setup(
    name="decomposed-dynamics",
    version="0.1.0",
    description="Add your description here",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    author="Sai Koukuntla",
    author_email="sai.koukunt@gmail.com",
    python_requires=">=3.12",
    packages=find_packages(exclude=("tests", "tests.*")),
    install_requires=[
        "jax[cuda12]>=0.10.0",
        "jaxopt>=0.8.5",
        "optax>=0.2.8",
        "tqdm>=4.67.3",
    ],
    extras_require={
        "dev": [
            "matplotlib>=3.10.9",
            "simulations",
        ]
    },
    include_package_data=True,
)