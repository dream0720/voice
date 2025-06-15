"""
Voice Matching Module
====================

This module provides voice matching capabilities to identify the best matching
separated voice based on reference audio using:
- MFCC (Mel-frequency cepstral coefficients)
- Mel-spectrogram features
- Spectral features (centroid, rolloff, bandwidth, etc.)
- Temporal features (tempo, rhythm)
- Comprehensive similarity analysis and reporting
"""

from .voice_matcher import VoiceMatcher

__all__ = ['VoiceMatcher']
