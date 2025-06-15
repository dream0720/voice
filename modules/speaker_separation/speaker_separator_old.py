#!/usr/bin/env python3
"""
Speaker Separation Module using VoiceFilter-WavLM and Pyannote
==============================================================

This module provides speaker separation capabilities using:
- Pyannote for speaker diarization and separation
- VoiceFilter-WavLM for advanced voice separation
- Automatic Hugging Face authentication
"""

import os
import sys
import subprocess
import torch
import torchaudio
import scipy.io.wavfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import soundfile as sf


class SpeakerSeparator:
    """Speaker separation processor using multiple methods"""
    
    def __init__(self, hf_token: Optional[str] = None, device: str = "auto"):
        """Initialize the speaker separator
        
        Args:
            hf_token: Hugging Face access token
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.hf_token = hf_token
        
        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.pipeline = None
        
        print(f"🔧 Speaker Separator initialized")
        print(f"   Device: {self.device}")
        print(f"   HF Token: {'✅ Provided' if hf_token else '❌ Not provided'}")
    
    def setup_huggingface_auth(self, token: str) -> bool:
        """Setup Hugging Face authentication
        
        Args:
            token: Hugging Face access token
            
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            # Install huggingface_hub if not available
            try:
                from huggingface_hub import login
            except ImportError:
                print("📦 Installing huggingface_hub...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub'], 
                             check=True)
                from huggingface_hub import login
            
            # Login with token
            login(token=token, add_to_git_credential=False)
            print("✅ Hugging Face authentication successful")
            return True
            
        except Exception as e:
            print(f"❌ Hugging Face authentication failed: {e}")
            return False
    
    def load_pyannote_pipeline(self) -> bool:
        """Load and initialize Pyannote pipeline
        
        Returns:
            True if pipeline loaded successfully, False otherwise
        """
        try:
            if not self.hf_token:
                print("❌ Hugging Face token required for Pyannote pipeline")
                return False
            
            # Setup authentication
            if not self.setup_huggingface_auth(self.hf_token):
                return False
            
            print("🤖 Loading Pyannote speech separation pipeline...")
            
            # Load pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speech-separation-ami-1.0",
                use_auth_token=self.hf_token
            )
            
            # Move to device
            self.pipeline.to(self.device)
            
            print(f"✅ Pyannote pipeline loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Pyannote pipeline: {e}")
            return False
    
    def separate_speakers_pyannote(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """Separate speakers using Pyannote pipeline
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save separated speaker files
            
        Returns:
            Dictionary containing separation results
        """
        print("🎵 Starting Pyannote Speaker Separation")
        print("=" * 50)
        
        # Load pipeline if not already loaded
        if self.pipeline is None:
            if not self.load_pyannote_pipeline():
                return None
        
        # Validate input file
        if not os.path.exists(input_path):
            print(f"❌ Input file not found: {input_path}")
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            print(f"📁 Input file: {Path(input_path).name}")
            print(f"📂 Output directory: {output_dir}")
            
            # Load audio
            print("🔄 Loading audio file...")
            waveform, sample_rate = torchaudio.load(input_path)
            
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Duration: {waveform.shape[1] / sample_rate:.2f} seconds")
            print(f"   Channels: {waveform.shape[0]}")
            
            # Run pipeline with progress monitoring
            print("🔄 Running speaker separation...")
            with ProgressHook() as hook:
                diarization, sources = self.pipeline({
                    "waveform": waveform, 
                    "sample_rate": sample_rate
                }, hook=hook)
            
            # Save diarization results
            input_name = Path(input_path).stem
            rttm_path = os.path.join(output_dir, f"{input_name}_diarization.rttm")
            
            with open(rttm_path, "w") as rttm:
                diarization.write_rttm(rttm)
            
            print(f"📄 Diarization saved to: {rttm_path}")
            
            # Save separated speaker audio files
            separated_files = {}
            speakers = list(diarization.labels())
            
            print(f"👥 Found {len(speakers)} speakers: {speakers}")
            
            for s, speaker in enumerate(speakers):
                speaker_file = os.path.join(output_dir, f"{input_name}_{speaker}.wav")
                
                # Extract speaker audio
                speaker_audio = sources.data[:, s]
                
                # Convert to 16-bit and save
                speaker_audio_16bit = (speaker_audio * 32767).astype(np.int16)
                scipy.io.wavfile.write(speaker_file, 16000, speaker_audio_16bit)
                
                separated_files[speaker] = speaker_file
                print(f"  💾 {speaker}: {speaker_file}")
            
            # Analyze separated files
            analysis = self.analyze_separated_speakers(separated_files)
            
            # Generate report
            report_path = os.path.join(output_dir, f"{input_name}_speaker_report.txt")
            self.generate_speaker_report(input_path, separated_files, analysis, 
                                       diarization, report_path)
            
            results = {
                'input_path': input_path,
                'output_dir': output_dir,
                'separated_files': separated_files,
                'diarization_file': rttm_path,
                'diarization': diarization,
                'analysis': analysis,
                'report_path': report_path,
                'speakers': speakers,
                'success': True
            }
            
            print("✅ Pyannote speaker separation completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Speaker separation failed: {e}")
            return None
    
    def separate_with_voicefilter_wavlm(self, input_path: str, reference_path: str, 
                                      output_dir: str) -> Dict[str, Any]:
        """Separate speakers using VoiceFilter-WavLM
        
        Args:
            input_path: Path to mixed audio file
            reference_path: Path to reference audio for target speaker
            output_dir: Directory to save output
            
        Returns:
            Dictionary containing separation results
        """
        print("🎵 Starting VoiceFilter-WavLM Speaker Separation")
        print("=" * 50)
        
        # Check if VoiceFilter-WavLM is available
        voicefilter_dir = Path("VoiceFilter-WavLM")
        if not voicefilter_dir.exists():
            print("❌ VoiceFilter-WavLM directory not found")
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            print(f"📁 Mixed audio: {Path(input_path).name}")
            print(f"📁 Reference audio: {Path(reference_path).name}")
            print(f"📂 Output directory: {output_dir}")
            
            # Prepare output path
            input_name = Path(input_path).stem
            output_path = os.path.join(output_dir, f"{input_name}_voicefilter_separated.wav")
            
            # Build VoiceFilter command (assuming the interface exists)
            cmd = [
                sys.executable, os.path.join(voicefilter_dir, "inference.py"),
                "--mixed", input_path,
                "--reference", reference_path,
                "--output", output_path
            ]
            
            print("🔄 Running VoiceFilter-WavLM separation...")
            print(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=voicefilter_dir, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0 and os.path.exists(output_path):
                print("✅ VoiceFilter-WavLM separation completed successfully!")
                
                # Analyze results
                analysis = self.analyze_voicefilter_output(input_path, reference_path, output_path)
                
                # Generate report
                report_path = os.path.join(output_dir, f"{input_name}_voicefilter_report.txt")
                self.generate_voicefilter_report(input_path, reference_path, output_path, 
                                               analysis, report_path)
                
                results = {
                    'input_path': input_path,
                    'reference_path': reference_path,
                    'output_path': output_path,
                    'output_dir': output_dir,
                    'analysis': analysis,
                    'report_path': report_path,
                    'success': True
                }
                
                return results
            else:
                print("❌ VoiceFilter-WavLM separation failed!")
                print(f"Error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("❌ VoiceFilter-WavLM separation timed out")
            return None
        except Exception as e:
            print(f"❌ VoiceFilter-WavLM separation error: {e}")
            return None
    
    def analyze_separated_speakers(self, separated_files: Dict[str, str]) -> Dict[str, Any]:
        """Analyze separated speaker audio files
        
        Args:
            separated_files: Dictionary mapping speaker IDs to file paths
            
        Returns:
            Dictionary containing analysis results
        """
        print("\n🔍 Analyzing separated speaker files...")
        
        analysis = {}
        
        for speaker, file_path in separated_files.items():
            try:
                # Load audio
                audio, sr = sf.read(file_path)
                
                # Basic statistics
                duration = len(audio) / sr
                rms_energy = np.sqrt(np.mean(audio ** 2))
                peak_amplitude = np.max(np.abs(audio))
                zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(audio))))
                
                # Spectral features
                stft = np.abs(np.fft.fft(audio))
                spectral_centroid = np.sum(np.arange(len(stft)) * stft) / np.sum(stft)
                
                analysis[speaker] = {
                    'duration': duration,
                    'sample_rate': sr,
                    'rms_energy': rms_energy,
                    'peak_amplitude': peak_amplitude,
                    'zero_crossing_rate': zero_crossing_rate,
                    'spectral_centroid': spectral_centroid,
                    'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
                }
                
                print(f"  {speaker}:")
                print(f"    Duration: {duration:.2f}s")
                print(f"    RMS Energy: {rms_energy:.4f}")
                print(f"    Peak: {peak_amplitude:.4f}")
                
            except Exception as e:
                print(f"  ❌ Failed to analyze {speaker}: {e}")
                analysis[speaker] = {'error': str(e)}
        
        return analysis
    
    def analyze_voicefilter_output(self, input_path: str, reference_path: str, 
                                 output_path: str) -> Dict[str, Any]:
        """Analyze VoiceFilter-WavLM output
        
        Args:
            input_path: Original mixed audio path
            reference_path: Reference audio path  
            output_path: Separated output audio path
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {}
        
        try:
            # Load all audio files
            mixed_audio, mixed_sr = sf.read(input_path)
            ref_audio, ref_sr = sf.read(reference_path)
            sep_audio, sep_sr = sf.read(output_path)
            
            # Basic statistics for each
            for name, audio, sr in [
                ('mixed', mixed_audio, mixed_sr),
                ('reference', ref_audio, ref_sr),
                ('separated', sep_audio, sep_sr)
            ]:
                duration = len(audio) / sr
                rms_energy = np.sqrt(np.mean(audio ** 2))
                peak_amplitude = np.max(np.abs(audio))
                
                analysis[name] = {
                    'duration': duration,
                    'sample_rate': sr,
                    'rms_energy': rms_energy,
                    'peak_amplitude': peak_amplitude
                }
            
            # Calculate improvement metrics
            if len(sep_audio) > 0 and len(mixed_audio) > 0:
                # Signal-to-noise ratio improvement (simplified)
                mixed_noise_floor = np.percentile(np.abs(mixed_audio), 10)
                sep_noise_floor = np.percentile(np.abs(sep_audio), 10)
                snr_improvement = 20 * np.log10(sep_noise_floor / (mixed_noise_floor + 1e-10))
                
                analysis['improvement'] = {
                    'snr_improvement_db': snr_improvement,
                    'energy_ratio': analysis['separated']['rms_energy'] / analysis['mixed']['rms_energy']
                }
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def generate_speaker_report(self, input_path: str, separated_files: Dict[str, str],
                              analysis: Dict[str, Any], diarization: Any, report_path: str):
        """Generate detailed speaker separation report
        
        Args:
            input_path: Original input file path
            separated_files: Dictionary of separated file paths
            analysis: Analysis results
            diarization: Diarization results
            report_path: Path to save the report
        """
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Pyannote Speaker Separation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Separation Information:\n")
            f.write(f"  Input file: {Path(input_path).name}\n")
            f.write(f"  Pipeline: pyannote/speech-separation-ami-1.0\n")
            f.write(f"  Speakers found: {len(separated_files)}\n")
            f.write(f"  Device used: {self.device}\n\n")
            
            f.write("Separated Files:\n")
            for speaker, file_path in separated_files.items():
                f.write(f"  {speaker}: {Path(file_path).name}\n")
            f.write("\n")
            
            f.write("Speaker Analysis:\n")
            for speaker, data in analysis.items():
                if 'error' in data:
                    f.write(f"  {speaker}: Analysis failed - {data['error']}\n")
                else:
                    f.write(f"  {speaker}:\n")
                    f.write(f"    Duration: {data['duration']:.2f} seconds\n")
                    f.write(f"    RMS Energy: {data['rms_energy']:.4f}\n")
                    f.write(f"    Peak Amplitude: {data['peak_amplitude']:.4f}\n")
                    f.write(f"    Zero Crossing Rate: {data['zero_crossing_rate']:.4f}\n")
                    f.write(f"    Spectral Centroid: {data['spectral_centroid']:.2f} Hz\n")
                    f.write(f"    File Size: {data['file_size_mb']:.2f} MB\n\n")
            
            # Add diarization timeline
            f.write("Speaker Timeline:\n")
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                f.write(f"  {segment.start:.2f}-{segment.end:.2f}s: {speaker}\n")
        
        print(f"📄 Speaker separation report saved to: {report_path}")
    
    def generate_voicefilter_report(self, input_path: str, reference_path: str, 
                                  output_path: str, analysis: Dict[str, Any], report_path: str):
        """Generate VoiceFilter-WavLM separation report
        
        Args:
            input_path: Mixed audio file path
            reference_path: Reference audio file path
            output_path: Separated output file path
            analysis: Analysis results
            report_path: Path to save the report
        """
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("VoiceFilter-WavLM Separation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Separation Information:\n")
            f.write(f"  Mixed audio: {Path(input_path).name}\n")
            f.write(f"  Reference audio: {Path(reference_path).name}\n")
            f.write(f"  Output audio: {Path(output_path).name}\n")
            f.write(f"  Model: VoiceFilter-WavLM\n\n")
            
            if 'error' not in analysis:
                f.write("Audio Analysis:\n")
                for audio_type in ['mixed', 'reference', 'separated']:
                    if audio_type in analysis:
                        data = analysis[audio_type]
                        f.write(f"  {audio_type.capitalize()} Audio:\n")
                        f.write(f"    Duration: {data['duration']:.2f} seconds\n")
                        f.write(f"    Sample Rate: {data['sample_rate']} Hz\n")
                        f.write(f"    RMS Energy: {data['rms_energy']:.4f}\n")
                        f.write(f"    Peak Amplitude: {data['peak_amplitude']:.4f}\n\n")
                
                if 'improvement' in analysis:
                    imp = analysis['improvement']
                    f.write("Separation Quality:\n")
                    f.write(f"  SNR Improvement: {imp['snr_improvement_db']:.2f} dB\n")
                    f.write(f"  Energy Ratio: {imp['energy_ratio']:.3f}\n")
            else:
                f.write(f"Analysis Error: {analysis['error']}\n")
        
        print(f"📄 VoiceFilter separation report saved to: {report_path}")


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Speaker Separation Tool')
    parser.add_argument('input', help='Input audio file path')
    parser.add_argument('-o', '--output', default='output/speaker_output', 
                       help='Output directory')
    parser.add_argument('--method', choices=['pyannote', 'voicefilter'], 
                       default='pyannote', help='Separation method to use')
    parser.add_argument('--reference', help='Reference audio file (for VoiceFilter)')
    parser.add_argument('--hf-token', help='Hugging Face access token')
    parser.add_argument('--device', default='auto', 
                       choices=['cpu', 'cuda', 'auto'], help='Device to use')
    
    args = parser.parse_args()
    
    # Get HF token from environment if not provided
    hf_token = args.hf_token or os.getenv('HF_ACCESS_TOKEN')
    
    separator = SpeakerSeparator(hf_token=hf_token, device=args.device)
    
    if args.method == 'pyannote':
        results = separator.separate_speakers_pyannote(args.input, args.output)
    elif args.method == 'voicefilter':
        if not args.reference:
            print("❌ Reference audio required for VoiceFilter method")
            sys.exit(1)
        results = separator.separate_with_voicefilter_wavlm(
            args.input, args.reference, args.output)
    
    if results and results['success']:
        print(f"\n🎉 Speaker separation completed successfully!")
        print(f"Output directory: {results['output_dir']}")
        print(f"Report: {results['report_path']}")
    else:
        print("❌ Speaker separation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
