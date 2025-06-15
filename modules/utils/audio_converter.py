#!/usr/bin/env python3
"""
Audio Format Converter
=====================

This module provides robust audio format conversion with proper normalization
to fix audio quality issues from speaker separation outputs.
"""

import numpy as np
import soundfile as sf
import os
from pathlib import Path
from typing import Optional, Tuple


class AudioConverter:
    """Audio format converter with advanced normalization"""
    
    def __init__(self):
        self.supported_formats = ['.wav', '.flac', '.ogg', '.mp3']
    
    def convert_and_fix_audio(self, input_path: str, output_path: Optional[str] = None, 
                             target_format: str = 'wav', target_subtype: str = 'PCM_16',
                             normalize: bool = True, safety_margin: float = 0.95) -> str:
        """Convert and fix audio with proper normalization
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output (None for auto-generation)
            target_format: Target audio format ('wav', 'flac', etc.)
            target_subtype: Audio subtype ('PCM_16', 'PCM_24', 'FLOAT', etc.)
            normalize: Whether to apply normalization
            safety_margin: Safety margin for normalization (0-1)
            
        Returns:
            Path to converted audio file
        """
        print(f"🔄 Converting and fixing audio: {Path(input_path).name}")
        
        # Load audio data
        try:
            data, samplerate = sf.read(input_path)
            print(f"  📊 Original format: {sf.info(input_path).subtype}")
            print(f"  📊 Sample rate: {samplerate} Hz")
            print(f"  📊 Duration: {len(data)/samplerate:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return None
        
        # Analyze audio quality
        max_val = np.max(np.abs(data))
        rms_val = np.sqrt(np.mean(data ** 2))
        dynamic_range = max_val / (rms_val + 1e-10)
        
        print(f"  📊 Max amplitude: {max_val:.4f}")
        print(f"  📊 RMS level: {rms_val:.4f}")
        print(f"  📊 Dynamic range: {dynamic_range:.2f}")
        
        # Apply normalization if needed
        if normalize:
            processed_data = self._smart_normalize(data, safety_margin)
            print(f"  ✅ Applied smart normalization (margin: {safety_margin})")
        else:
            processed_data = data
        
        # Ensure data is in valid range
        processed_data = np.clip(processed_data, -1.0, 1.0)
        
        # Generate output path if not provided
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_fixed.{target_format}")
        
        # Convert to target format
        try:
            if target_subtype == 'PCM_16':
                # Convert to 16-bit integer with proper scaling
                processed_data_int16 = (processed_data * 32767 * safety_margin).astype(np.int16)
                sf.write(output_path, processed_data_int16, samplerate, subtype=target_subtype)
            elif target_subtype == 'PCM_24':
                # Convert to 24-bit integer
                processed_data_int24 = (processed_data * 8388607 * safety_margin).astype(np.int32)
                sf.write(output_path, processed_data_int24, samplerate, subtype=target_subtype)
            else:
                # Keep as float
                sf.write(output_path, processed_data, samplerate, subtype=target_subtype)
                
            print(f"  💾 Saved to: {output_path}")
            print(f"  📊 Target format: {target_subtype}")
            
            # Verify the conversion
            self._verify_conversion(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
            return None
    
    def _smart_normalize(self, data: np.ndarray, safety_margin: float = 0.95) -> np.ndarray:
        """Apply smart normalization to audio data
        
        Args:
            data: Input audio data
            safety_margin: Safety margin for normalization
            
        Returns:
            Normalized audio data
        """
        # Calculate peak and RMS values
        peak_val = np.max(np.abs(data))
        rms_val = np.sqrt(np.mean(data ** 2))
        
        if peak_val == 0:
            return data
        
        # Use peak normalization with safety margin
        if peak_val > 1.0:
            # Audio is clipped, normalize by peak
            normalization_factor = safety_margin / peak_val
        elif peak_val < 0.1:
            # Audio is too quiet, boost it
            normalization_factor = safety_margin * 0.5 / peak_val
        else:
            # Audio is in reasonable range, apply gentle normalization
            normalization_factor = safety_margin / peak_val
        
        normalized_data = data * normalization_factor
        
        # Apply gentle limiting to prevent any remaining clipping
        normalized_data = np.tanh(normalized_data * 0.9) * 0.9
        
        return normalized_data
    
    def _verify_conversion(self, output_path: str) -> bool:
        """Verify the converted audio file
        
        Args:
            output_path: Path to converted file
            
        Returns:
            True if verification successful
        """
        try:
            info = sf.info(output_path)
            data, _ = sf.read(output_path)
            
            max_val = np.max(np.abs(data))
            has_clipping = np.any(np.abs(data) >= 0.99)
            
            print(f"  ✅ Verification - Max: {max_val:.4f}, Clipping: {'Yes' if has_clipping else 'No'}")
            
            return not has_clipping
            
        except Exception as e:
            print(f"  ⚠️ Verification failed: {e}")
            return False
    
    def batch_convert(self, input_dir: str, output_dir: str, 
                     pattern: str = "*.wav", **kwargs) -> list:
        """Batch convert audio files
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            pattern: File pattern to match
            **kwargs: Additional arguments for convert_and_fix_audio
            
        Returns:
            List of converted file paths
        """
        print(f"🔄 Batch converting audio files from {input_dir}")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        converted_files = []
        
        for file_path in input_path.glob(pattern):
            output_file = output_path / f"{file_path.stem}_fixed{file_path.suffix}"
            result = self.convert_and_fix_audio(str(file_path), str(output_file), **kwargs)
            if result:
                converted_files.append(result)
        
        print(f"✅ Batch conversion completed: {len(converted_files)} files")
        return converted_files


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Format Converter with Smart Normalization')
    parser.add_argument('input', help='Input audio file or directory')
    parser.add_argument('-o', '--output', help='Output file or directory')
    parser.add_argument('--format', default='wav', help='Target format (wav, flac, etc.)')
    parser.add_argument('--subtype', default='PCM_16', help='Target subtype (PCM_16, PCM_24, FLOAT)')
    parser.add_argument('--no-normalize', action='store_true', help='Skip normalization')
    parser.add_argument('--safety-margin', type=float, default=0.95, help='Safety margin for normalization')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    
    args = parser.parse_args()
    
    converter = AudioConverter()
    
    if args.batch or os.path.isdir(args.input):
        # Batch processing
        output_dir = args.output or f"{args.input}_converted"
        results = converter.batch_convert(
            args.input, output_dir,
            normalize=not args.no_normalize,
            target_format=args.format,
            target_subtype=args.subtype,
            safety_margin=args.safety_margin
        )
        print(f"\n🎉 Batch conversion completed: {len(results)} files")
    else:
        # Single file processing
        result = converter.convert_and_fix_audio(
            args.input, args.output,
            normalize=not args.no_normalize,
            target_format=args.format,
            target_subtype=args.subtype,
            safety_margin=args.safety_margin
        )
        if result:
            print(f"\n🎉 Conversion completed: {result}")


if __name__ == "__main__":
    main()
