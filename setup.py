from setuptools import setup, find_packages

setup(
    name="voice_processing",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "PyQt6>=6.4.0",
        "matplotlib>=3.7.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "torch>=2.0.0",
        "demucs>=4.0.0",
        "qt-material>=2.14.0",
        "scikit-learn>=1.3.0"
    ],
    python_requires=">=3.8",
) 