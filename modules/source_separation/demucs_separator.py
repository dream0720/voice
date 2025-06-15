#!/usr/bin/env python3
"""
Source Separation Module using Demucs
=====================================

This module provides music source separation capabilities using the Demucs model.
It can separate mixed audio into different stems (vocals, drums, bass, other).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import soundfile as sf
import numpy as np


class DemucsSourceSeparator:
    """Demucs-based source separation processor"""
    
    def __init__(self, model_name: str = "htdemucs"):
        """Initialize the Demucs source separator
        
        Args:
            model_name: Demucs model to use (htdemucs, htdemucs_ft, etc.)
        """
        self.model_name = model_name
        self.available_models = [
            "htdemucs", "htdemucs_ft", "htdemucs_6s", 
            "hdemucs_mmi", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q"
        ]
        
        if model_name not in self.available_models:
            print(f"⚠️  Warning: Model '{model_name}' not in known models list")
            print(f"Available models: {', '.join(self.available_models)}")
    
    def check_demucs_installation(self) -> bool:
        """Check if Demucs is properly installed
        
        Returns:
            True if Demucs is installed, False otherwise
        """
        try:
            result = subprocess.run(['python', '-m', 'demucs', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def install_demucs(self) -> bool:
        """Install Demucs if not already installed
        
        Returns:
            True if installation successful, False otherwise
        """
        print("📦 Installing Demucs...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'demucs'], 
                         check=True)
            print("✅ Demucs installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Demucs: {e}")
            return False
    
    def separate_audio(self, input_path: str, output_dir: str, 
                      stems: Optional[List[str]] = None, 
                      device: str = "cpu") -> Dict[str, Any]:
        """Separate audio into different stems using Demucs
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save separated stems
            stems: List of stems to extract (None for all: vocals, drums, bass, other)
            device: Device to use for processing ('cpu' or 'cuda')
            
        Returns:
            Dictionary containing separation results and file paths
        """
        print("🎵 Starting Demucs Source Separation")
        print("=" * 50)
        
        # Check if Demucs is installed
        if not self.check_demucs_installation():
            print("⚠️  Demucs not found, attempting to install...")
            if not self.install_demucs():
                return None
        
        # Validate input file
        if not os.path.exists(input_path):
            print(f"❌ Input file not found: {input_path}")
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
          # Build Demucs command
        cmd = [
            'python', '-m', 'demucs',
            '--name', self.model_name,  # 使用 -n/--name 而不是 --model
            '--out', output_dir,
            '--device', device,
        ]
        
        # Add two-stems mode if only specific stems requested
        if stems and len(stems) == 1:
            cmd.extend(['--two-stems', stems[0]])
        
        cmd.append(input_path)
        
        print(f"📁 Input file: {Path(input_path).name}")
        print(f"📂 Output directory: {output_dir}")
        print(f"🤖 Model: {self.model_name}")
        print(f"💻 Device: {device}")
        if stems:
            print(f"🎼 Stems: {', '.join(stems)}")
        else:
            print(f"🎼 Stems: all (vocals, drums, bass, other)")
        
        print(f"\n🔄 Running Demucs separation...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run Demucs
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                print("✅ Demucs separation completed successfully!")
                
                # Find output files
                input_name = Path(input_path).stem
                model_output_dir = os.path.join(output_dir, self.model_name, input_name)
                
                separated_files = {}
                expected_stems = stems if stems else ['vocals', 'drums', 'bass', 'other']
                
                for stem in expected_stems:
                    stem_file = os.path.join(model_output_dir, f"{stem}.wav")
                    if os.path.exists(stem_file):
                        separated_files[stem] = stem_file
                        print(f"  📄 {stem.capitalize()}: {stem_file}")
                    else:
                        print(f"  ⚠️  {stem.capitalize()} file not found: {stem_file}")
                
                # Analyze separated files
                analysis = self.analyze_separated_audio(separated_files)
                
                # Generate report
                report_path = os.path.join(output_dir, f"{input_name}_separation_report.txt")
                self.generate_separation_report(input_path, separated_files, analysis, report_path)
                
                results = {
                    'input_path': input_path,
                    'output_dir': model_output_dir,
                    'separated_files': separated_files,
                    'analysis': analysis,
                    'report_path': report_path,
                    'model_used': self.model_name,
                    'success': True
                }
                
                return results
                
            else:
                print(f"❌ Demucs separation failed!")
                print(f"Error output: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("❌ Demucs separation timed out (>1 hour)")
            return None
        except Exception as e:
            print(f"❌ Unexpected error during separation: {e}")
            return None
    
    def analyze_separated_audio(self, separated_files: Dict[str, str]) -> Dict[str, Any]:
        """Analyze the separated audio files
        
        Args:
            separated_files: Dictionary mapping stem names to file paths
            
        Returns:
            Dictionary containing analysis results
        """
        print("\n🔍 Analyzing separated audio files...")
        
        analysis = {}
        
        for stem, file_path in separated_files.items():
            try:
                # Load audio
                audio, sr = sf.read(file_path)
                
                # Basic statistics
                duration = len(audio) / sr
                rms_energy = np.sqrt(np.mean(audio ** 2))
                peak_amplitude = np.max(np.abs(audio))
                dynamic_range = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
                
                analysis[stem] = {
                    'duration': duration,
                    'sample_rate': sr,
                    'samples': len(audio),
                    'rms_energy': rms_energy,
                    'peak_amplitude': peak_amplitude,
                    'dynamic_range_db': dynamic_range,
                    'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
                }
                
                print(f"  {stem.capitalize()}:")
                print(f"    Duration: {duration:.2f}s")
                print(f"    RMS Energy: {rms_energy:.4f}")
                print(f"    Peak: {peak_amplitude:.4f}")
                print(f"    Dynamic Range: {dynamic_range:.1f} dB")
                
            except Exception as e:
                print(f"  ❌ Failed to analyze {stem}: {e}")
                analysis[stem] = {'error': str(e)}
        
        return analysis
    
    def generate_separation_report(self, input_path: str, separated_files: Dict[str, str], 
                                 analysis: Dict[str, Any], report_path: str):
        """Generate detailed separation report
        
        Args:
            input_path: Original input file path
            separated_files: Dictionary of separated file paths
            analysis: Analysis results
            report_path: Path to save the report
        """
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Demucs Source Separation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Separation Information:\n")
            f.write(f"  Input file: {Path(input_path).name}\n")
            f.write(f"  Model used: {self.model_name}\n")
            f.write(f"  Stems separated: {len(separated_files)}\n\n")
            
            f.write("Separated Files:\n")
            for stem, file_path in separated_files.items():
                f.write(f"  {stem.capitalize()}: {Path(file_path).name}\n")
            f.write("\n")
            
            f.write("Audio Analysis:\n")
            for stem, data in analysis.items():
                if 'error' in data:
                    f.write(f"  {stem.capitalize()}: Analysis failed - {data['error']}\n")
                else:
                    f.write(f"  {stem.capitalize()}:\n")
                    f.write(f"    Duration: {data['duration']:.2f} seconds\n")
                    f.write(f"    Sample Rate: {data['sample_rate']} Hz\n")
                    f.write(f"    RMS Energy: {data['rms_energy']:.4f}\n")
                    f.write(f"    Peak Amplitude: {data['peak_amplitude']:.4f}\n")
                    f.write(f"    Dynamic Range: {data['dynamic_range_db']:.1f} dB\n")
                    f.write(f"    File Size: {data['file_size_mb']:.2f} MB\n\n")
            
            f.write("Model Information:\n")
            f.write(f"  Demucs Model: {self.model_name}\n")
            f.write("  Technology: Hybrid Transformer Demucs (HTDemucs)\n")
            f.write("  Architecture: U-Net with transformer layers\n")
            f.write("  Training: Large-scale music dataset\n")
        
        print(f"📄 Separation report saved to: {report_path}")
    
    def get_vocals_file(self, separated_files: Dict[str, str]) -> Optional[str]:
        """Get the vocals file path from separation results
        
        Args:
            separated_files: Dictionary of separated file paths
            
        Returns:
            Path to vocals file, or None if not found
        """
        return separated_files.get('vocals')
    
    def list_available_models(self) -> List[str]:
        """List all available Demucs models
        
        Returns:
            List of available model names
        """
        return self.available_models.copy()


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demucs Source Separation Tool')
    parser.add_argument('input', help='Input audio file path')
    parser.add_argument('-o', '--output', default='output/demucs_output', 
                       help='Output directory')
    parser.add_argument('-m', '--model', default='htdemucs', 
                       help='Demucs model to use')
    parser.add_argument('--stems', nargs='+', 
                       help='Specific stems to extract (vocals, drums, bass, other)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for processing')
    
    args = parser.parse_args()
    
    separator = DemucsSourceSeparator(model_name=args.model)
    results = separator.separate_audio(
        input_path=args.input,
        output_dir=args.output,
        stems=args.stems,
        device=args.device
    )
    
    if results and results['success']:
        print(f"\n🎉 Source separation completed successfully!")
        print(f"Output directory: {results['output_dir']}")
        print(f"Vocals file: {results['separated_files'].get('vocals', 'Not available')}")
        print(f"Report: {results['report_path']}")
    else:
        print("❌ Source separation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
