#!/usr/bin/env python3
"""
Utility Functions for Voice Processing
=====================================

This module provides various utility functions for audio processing:
- Audio format conversion
- File path handling
- Logging utilities
- Configuration management
"""

import os
import sys
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import logging
from datetime import datetime


class AudioConverter:
    """Audio format conversion utilities"""
    
    @staticmethod
    def convert_to_16bit(input_path: str, output_path: str) -> bool:
        """Convert audio to 16-bit PCM format
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Read audio data
            data, samplerate = sf.read(input_path)
            
            # Check maximum absolute value
            max_val = np.max(np.abs(data))
            
            # Normalize if exceeding normal range
            if max_val > 1:
                data = data / max_val
            
            # Apply safe clipping with margin
            data = np.clip(data, -1.0, 1.0) * 0.98
            
            # Convert to int16 format
            data_int16 = (data * 32767).astype(np.int16)
            
            # Save converted audio
            sf.write(output_path, data_int16, samplerate, subtype='PCM_16')
            
            return True
            
        except Exception as e:
            print(f"❌ Audio conversion failed: {e}")
            return False
    
    @staticmethod
    def get_audio_info(file_path: str) -> Optional[Dict[str, Any]]:
        """Get detailed audio file information
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary containing audio information, or None if failed
        """
        try:
            info = sf.info(file_path)
            
            return {
                'frames': info.frames,
                'samplerate': info.samplerate,
                'channels': info.channels,
                'duration': info.frames / info.samplerate,
                'format': info.format,
                'subtype': info.subtype,
                'endian': info.endian,
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
            }
            
        except Exception as e:
            print(f"❌ Failed to get audio info: {e}")
            return None


class PathManager:
    """File path management utilities"""
    
    @staticmethod
    def ensure_dir(directory: str) -> bool:
        """Ensure directory exists, create if not
        
        Args:
            directory: Directory path
            
        Returns:
            True if directory exists or was created successfully
        """
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    @staticmethod
    def get_safe_filename(filename: str) -> str:
        """Get safe filename by removing invalid characters
        
        Args:
            filename: Original filename
            
        Returns:
            Safe filename
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        safe_name = filename
        
        for char in invalid_chars:
            safe_name = safe_name.replace(char, '_')
        
        return safe_name
    
    @staticmethod
    def get_unique_filename(directory: str, base_name: str, extension: str) -> str:
        """Generate unique filename to avoid conflicts
        
        Args:
            directory: Target directory
            base_name: Base filename without extension
            extension: File extension (with dot)
            
        Returns:
            Unique filename
        """
        counter = 1
        original_path = os.path.join(directory, f"{base_name}{extension}")
        
        if not os.path.exists(original_path):
            return original_path
        
        while True:
            new_name = f"{base_name}_{counter}{extension}"
            new_path = os.path.join(directory, new_name)
            
            if not os.path.exists(new_path):
                return new_path
            
            counter += 1


class ConfigManager:
    """Configuration management utilities"""
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize configuration manager
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file
        
        Returns:
            Configuration dictionary
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load config: {e}")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'audio': {
                'sample_rate': 16000,
                'hop_length': 256,
                'n_fft': 1024,
                'n_mfcc': 13,
                'n_mels': 80
            },
            'preprocessing': {
                'apply_bandpass': True,
                'apply_spectral_subtraction': True,
                'apply_wiener': False,
                'low_freq': 80,
                'high_freq': 8000
            },
            'demucs': {
                'model': 'htdemucs',
                'device': 'cpu'
            },
            'paths': {
                'input_dir': 'input',
                'output_dir': 'output',
                'reference_dir': 'reference',
                'temp_dir': 'temp'
            },
            'gui': {
                'theme': 'light',
                'language': 'en'
            }
        }
    
    def save_config(self) -> bool:
        """Save configuration to file
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ Failed to save config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path
        
        Args:
            key: Key path (e.g., 'audio.sample_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value by key path
        
        Args:
            key: Key path (e.g., 'audio.sample_rate')
            value: Value to set
            
        Returns:
            True if set successfully, False otherwise
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        return self.save_config()


class Logger:
    """Logging utilities"""
    
    def __init__(self, name: str = "voice_processing", level: int = logging.INFO):
        """Initialize logger
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_dir = "logs"
        PathManager.ensure_dir(log_dir)
        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)


class ProgressTracker:
    """Progress tracking utility"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """Initialize progress tracker
        
        Args:
            total_steps: Total number of steps
            description: Process description
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, step: int = 1, message: str = ""):
        """Update progress
        
        Args:
            step: Number of steps to advance
            message: Progress message
        """
        self.current_step += step
        progress = (self.current_step / self.total_steps) * 100
        
        elapsed = datetime.now() - self.start_time
        
        print(f"\r{self.description}: {progress:.1f}% ({self.current_step}/{self.total_steps}) "
              f"- {message} [Elapsed: {elapsed.total_seconds():.1f}s]", end="", flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # New line when complete
    
    def finish(self, message: str = "Completed"):
        """Finish progress tracking
        
        Args:
            message: Completion message
        """
        elapsed = datetime.now() - self.start_time
        print(f"\n✅ {self.description} {message} in {elapsed.total_seconds():.1f}s")


def validate_audio_file(file_path: str) -> bool:
    """Validate if file is a valid audio file
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if valid audio file, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        info = sf.info(file_path)
        return info.frames > 0 and info.samplerate > 0
    except:
        return False


def get_supported_audio_formats() -> List[str]:
    """Get list of supported audio formats
    
    Returns:
        List of supported audio file extensions
    """
    return ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma']


def find_audio_files(directory: str, recursive: bool = True) -> List[str]:
    """Find all audio files in directory
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of audio file paths
    """
    audio_files = []
    supported_formats = get_supported_audio_formats()
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_formats):
                    audio_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in supported_formats):
                audio_files.append(file_path)
    
    return sorted(audio_files)


# Initialize global instances
config_manager = ConfigManager()
logger = Logger()


def main():
    """Main function for standalone utility testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Voice Processing Utilities')
    parser.add_argument('--convert', nargs=2, metavar=('INPUT', 'OUTPUT'),
                       help='Convert audio to 16-bit format')
    parser.add_argument('--info', help='Get audio file information')
    parser.add_argument('--find-audio', help='Find audio files in directory')
    
    args = parser.parse_args()
    
    if args.convert:
        input_file, output_file = args.convert
        if AudioConverter.convert_to_16bit(input_file, output_file):
            print(f"✅ Converted {input_file} to {output_file}")
        else:
            print(f"❌ Failed to convert {input_file}")
    
    elif args.info:
        info = AudioConverter.get_audio_info(args.info)
        if info:
            print(f"Audio Information for {args.info}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"❌ Failed to get info for {args.info}")
    
    elif args.find_audio:
        files = find_audio_files(args.find_audio)
        print(f"Found {len(files)} audio files in {args.find_audio}:")
        for file in files:
            print(f"  {file}")


if __name__ == "__main__":
    main()
