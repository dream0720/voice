"""
Speaker Separation Module
========================

This module provides speaker separation capabilities using:
- Pyannote: Speaker diarization and separation
- VoiceFilter-WavLM: Advanced voice separation
- Automatic Hugging Face authentication
- Comprehensive analysis and reporting
"""

from .speaker_separator import SpeakerSeparator

__all__ = ['SpeakerSeparator']
