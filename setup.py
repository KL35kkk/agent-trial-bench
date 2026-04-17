#!/usr/bin/env python3
"""
Setup script for agent-trial-bench package
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agent-trial-bench",
    version="2.1.0",
    author="Agent Trial Bench Team",
    author_email="dab@example.com",
    description="Agent Trial Bench – Domain-Agnostic Benchmark for AI Agents (with first-class Web3 support).",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/dab-team/agent-trial-bench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "atb=agent_trial_bench:main",
        ],
    },
    include_package_data=True,
    package_data={
        "agent_trial_bench": [
            "data/*.csv",
            "examples/*.py",
        ],
    },
    keywords="evaluation, agent, llm, benchmark, assessment, web3, tool-use, trajectory",
    project_urls={
        "Bug Reports": "https://github.com/dab-team/agent-trial-bench/issues",
        "Source": "https://github.com/dab-team/agent-trial-bench",
        "Documentation": "https://agent-trial-bench.readthedocs.io/",
    },
)
