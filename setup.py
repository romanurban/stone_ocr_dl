from setuptools import setup, find_packages

setup(
    name="stone_ocr",
    version="0.1.0",
    description="Stone OCR - ML project for stone inscription recognition",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],  # Use requirements.txt or pyproject.toml for dependencies
)
