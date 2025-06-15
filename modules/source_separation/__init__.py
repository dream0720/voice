"""
Source Separation Module
========================

This module provides music source separation capabilities using:
- Demucs: State-of-the-art music source separation
- Support for multiple models and stem types
- Comprehensive analysis and reporting
"""

from .demucs_separator import DemucsSourceSeparator

__all__ = ['DemucsSourceSeparator']
