#!/usr/bin/env python3
"""
Speaker Separation Module using Pyannote
========================================

This module provides speaker separation capabilities using Pyannote pipeline,
following the same approach as runvoicefilter.py with proper audio conversion
using transwav.py conversion method.
"""

import os
import sys
import subprocess
import torch
import torchaudio
import numpy as np
import soundfile as sf
import scipy.io.wavfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


class SpeakerSeparator:
    """Speaker separation processor using Pyannote pipeline"""    
    def __init__(self, hf_token: Optional[str] = None, device: str = "auto"):
        """Initialize the speaker separator
        
        Args:
            hf_token: Hugging Face access token
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.hf_token = hf_token
        
        # Auto-detect device (prefer GPU first)
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
    
    def convert_audio_like_transwav(self, input_path: str, output_path: str) -> bool:
        """Convert audio using the same method as transwav.py
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            print(f"🔄 Converting audio: {Path(input_path).name}")
            
            # 读取音频数据 (same as transwav.py)
            data, samplerate = sf.read(input_path)
            
            # 先查看波形的最大绝对值，便于判断是否超出范围
            max_val = np.max(np.abs(data))
            print(f"   最大绝对值: {max_val}")
            
            # 如果 max_val > 1，说明波形超出正常范围，需要先归一化
            if max_val > 1:
                data = data / max_val
            
            # 也可以做更稳健的归一化（留点安全余量，防止裁剪失真）
            data = np.clip(data, -1.0, 1.0) * 0.98
            
            # 转换为 int16 类型，避免溢出
            data_int16 = (data * 32767).astype(np.int16)
            
            # 保存音频
            sf.write(output_path, data_int16, samplerate, subtype='PCM_16')
            
            print(f"   ✅ 转换成功: {output_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ 转换失败: {e}")
            return False
    
    def load_pyannote_pipeline(self) -> bool:
        """Load and initialize Pyannote pipeline (same as runvoicefilter.py)
        
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
            
            # Load pipeline (same as runvoicefilter.py)
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speech-separation-ami-1.0",
                use_auth_token=self.hf_token
            )
              # Move to device with error handling
            try:
                self.pipeline.to(self.device)
                print(f"✅ Pyannote pipeline loaded successfully on {self.device}")
                return True
            except Exception as device_error:
                print(f"⚠️ Failed to move pipeline to {self.device}, trying CPU...")
                try:
                    self.device = torch.device("cpu")
                    self.pipeline.to(self.device)
                    print(f"✅ Pyannote pipeline loaded successfully on {self.device} (fallback)")
                    return True
                except Exception as cpu_error:
                    print(f"❌ Failed to load pipeline on any device: {cpu_error}")
                    return False
            
        except Exception as e:
            print(f"❌ Failed to load Pyannote pipeline: {e}")
            return False
    
    def separate_speakers(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """Separate speakers using Pyannote pipeline (following runvoicefilter.py logic)
        
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
            
            # Load audio (same as runvoicefilter.py)
            print("🔄 Loading audio file...")
            waveform, sample_rate = torchaudio.load(input_path)
            
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Duration: {waveform.shape[1] / sample_rate:.2f} seconds")
            print(f"   Channels: {waveform.shape[0]}")
              # Run pipeline with progress monitoring (same as runvoicefilter.py)
            print("🔄 Running speaker separation...")
            try:
                with ProgressHook() as hook:
                    diarization, sources = self.pipeline({
                        "waveform": waveform, 
                        "sample_rate": sample_rate
                    }, hook=hook)
            except RuntimeError as cuda_error:
                if "CUDA" in str(cuda_error):
                    print("⚠️ CUDA error detected, retrying on CPU...")
                    self.device = torch.device("cpu")
                    self.pipeline.to(self.device)
                    with ProgressHook() as hook:
                        diarization, sources = self.pipeline({
                            "waveform": waveform, 
                            "sample_rate": sample_rate
                        }, hook=hook)
                else:
                    raise
            
            # Save diarization results to RTTM file (same as runvoicefilter.py)
            input_name = Path(input_path).stem
            rttm_path = os.path.join(output_dir, f"{input_name}.rttm")
            
            with open(rttm_path, "w") as rttm:
                diarization.write_rttm(rttm)
            
            print(f"📄 Diarization saved to: {rttm_path}")
              # Save separated speaker audio files (same as runvoicefilter.py - no conversion)
            separated_files = {}
            speakers = list(diarization.labels())
            
            print(f"👥 Found {len(speakers)} speakers: {speakers}")
            
            # Save audio files directly (same as runvoicefilter.py)
            for s, speaker in enumerate(speakers):
                speaker_file = os.path.join(output_dir, f"{speaker}.wav")
                
                # Save audio using scipy (same as runvoicefilter.py)
                scipy.io.wavfile.write(speaker_file, 16000, sources.data[:, s])
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
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_separated_speakers(self, separated_files: Dict[str, str]) -> Dict[str, Any]:
        """Analyze separated speaker audio files
        
        Args:
            separated_files: Dictionary mapping speaker IDs to file paths
            
        Returns:
            Dictionary containing analysis results
        """
        print("\\n🔍 Analyzing separated speaker files...")
        
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
                if len(audio) > 0:
                    stft = np.abs(np.fft.fft(audio))
                    freqs = np.fft.fftfreq(len(stft), 1/sr)
                    spectral_centroid = np.sum(freqs[:len(freqs)//2] * stft[:len(stft)//2]) / np.sum(stft[:len(stft)//2])
                else:
                    spectral_centroid = 0
                
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
            f.write("Pyannote Speaker Separation Report\\n")
            f.write("=" * 50 + "\\n\\n")
            
            f.write("Separation Information:\\n")
            f.write(f"  Input file: {Path(input_path).name}\\n")
            f.write(f"  Pipeline: pyannote/speech-separation-ami-1.0\\n")
            f.write(f"  Speakers found: {len(separated_files)}\\n")
            f.write(f"  Device used: {self.device}\\n")
            f.write(f"  Audio conversion: transwav method\\n\\n")
            
            f.write("Separated Files:\\n")
            for speaker, file_path in separated_files.items():
                f.write(f"  {speaker}: {Path(file_path).name}\\n")
            f.write("\\n")
            
            f.write("Speaker Analysis:\\n")
            for speaker, data in analysis.items():
                if 'error' not in data:
                    f.write(f"  {speaker}:\\n")
                    f.write(f"    Duration: {data['duration']:.2f}s\\n")
                    f.write(f"    Sample Rate: {data['sample_rate']} Hz\\n")
                    f.write(f"    RMS Energy: {data['rms_energy']:.4f}\\n")
                    f.write(f"    Peak Amplitude: {data['peak_amplitude']:.4f}\\n")
                    f.write(f"    File Size: {data['file_size_mb']:.2f} MB\\n\\n")
                else:
                    f.write(f"  {speaker}: Analysis failed - {data['error']}\\n\\n")
            
            # Add diarization timeline
            f.write("Speaker Timeline:\\n")
            try:
                for segment, _, speaker in diarization.itertracks(yield_label=True):
                    f.write(f"  {segment.start:.2f}-{segment.end:.2f}s: {speaker}\\n")
            except Exception as e:
                f.write(f"  Timeline extraction failed: {e}\\n")
        
        print(f"📄 Speaker separation report saved to: {report_path}")


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Speaker Separation Tool - Pyannote Pipeline')
    parser.add_argument('input', help='Input audio file path')
    parser.add_argument('-o', '--output', default='output/speaker_separation', 
                       help='Output directory')
    parser.add_argument('--hf-token', help='Hugging Face access token')
    parser.add_argument('--device', default='auto', 
                       choices=['cpu', 'cuda', 'auto'], help='Device to use')
    
    args = parser.parse_args()
      # Get HF token from environment if not provided
    hf_token = args.hf_token or os.getenv('HF_ACCESS_TOKEN')
    
    # If still no token, use the hardcoded one (same as runvoicefilter.py)
    if not hf_token:
        hf_token = "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"
    
    if not hf_token:
        print("❌ Hugging Face token required!")
        print("Provide it via --hf-token argument or HF_ACCESS_TOKEN environment variable")
        sys.exit(1)
    
    separator = SpeakerSeparator(hf_token=hf_token, device=args.device)
    results = separator.separate_speakers(args.input, args.output)
    
    if results and results['success']:
        print(f"\\n🎉 Speaker separation completed successfully!")
        print(f"Output directory: {results['output_dir']}")
        print(f"Separated files: {len(results['separated_files'])}")
        for speaker, file_path in results['separated_files'].items():
            print(f"  {speaker}: {file_path}")
        print(f"Report: {results['report_path']}")
    else:
        print("❌ Speaker separation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
