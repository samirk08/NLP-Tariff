"""
Setup configuration for tariff_search package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tariff-search",
    version="0.1.0",
    author="Samir Kadariya",
    description="Fast similarity search for US tariff descriptions (1789-2023)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samirk08/NLP-Tariff",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "torch>=1.7.0",
        "transformers>=4.0.0",
        "rapidfuzz>=2.0.0",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "faiss": ["faiss-cpu>=1.7.0"],
        "faiss-gpu": ["faiss-gpu>=1.7.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "tariff-search=tariff_search.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tariff_search": ["data/*"],
    },
)