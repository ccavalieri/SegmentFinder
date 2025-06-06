from setuptools import setup, find_packages

# Reads README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except:
    long_description = "SegmentFinder - Biblioteca para descoberta de segmentos com uplift positivo"

# Reads dependencies
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except:
    requirements = ["pandas>=1.3.0", "numpy>=1.20.0", "scipy>=1.7.0", "tqdm>=4.60.0"]

setup(
    name="segmentfinder",
    version="1.0.0",
    author="Seu Nome",
    author_email="seu.email@exemplo.com",
    description="Biblioteca para descoberta de segmentos com uplift positivo em testes A/B",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ccavalieri/SegmentFinder",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    include_package_data=True,
)
