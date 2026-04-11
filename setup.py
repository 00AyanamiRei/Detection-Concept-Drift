from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fca-drift-detector",
    version="0.1.0",
    author="Maksym Kozlov",
    author_email="your.email@example.com",
    description="FCA-based Concept Drift Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fca-drift-detector",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "river==0.23.0",
        "concepts>=0.9.2",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "graphviz>=0.20.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "sphinx>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fca-drift=fca_drift.__main__:main",
        ],
    },
)
