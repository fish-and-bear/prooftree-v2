"""Setup script for Graph Neural Algebra Tutor."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="graph-neural-algebra-tutor",
    version="0.1.0",
    author="Graph Neural Algebra Tutor Team",
    author_email="contact@graphalgebratutor.ai",
    description="A Graph Neural Network-based step-by-step algebra tutor with symbolic verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/graph-neural-algebra-tutor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Education :: Computer Aided Instruction (CAI)",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gnat-train=scripts.train:main",
            "gnat-evaluate=scripts.evaluate:main",
            "gnat-web=web.app:main",
            "gnat-solve=solver.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "graph_neural_algebra_tutor": ["configs/*.yaml", "data/*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/graph-neural-algebra-tutor/issues",
        "Source": "https://github.com/yourusername/graph-neural-algebra-tutor",
        "Documentation": "https://graph-neural-algebra-tutor.readthedocs.io/",
    },
) 