#!/usr/bin/env python3
"""
Audio Preprocessing Module
=========================

This module provides comprehensive audio preprocessing capabilities including:
- FFT-based frequency domain analysis
- Band-pass filtering
- Spectral subtraction noise reduction
- Wiener filtering
- Time and frequency domain feature analysis

Based on signals and systems theory fundamentals.
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from pathlib import Path
import os
from typing import Tuple, Optional, Dict, Any

class AudioPreprocessor:
    """Audio preprocessing and denoising processor"""
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize the audio preprocessor
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        
    def load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load audio file with specified sample rate
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Loaded audio signal as numpy array, or None if failed
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            print(f"📁 Loaded audio: {Path(audio_path).name}")
            print(f"   Sample rate: {sr} Hz")
            print(f"   Duration: {len(audio)/sr:.2f} seconds")
            print(f"   Samples: {len(audio)}")
            return audio
        except Exception as e:
            print(f"❌ Audio loading failed: {e}")
            return None
    
    def analyze_frequency_spectrum(self, audio: np.ndarray, title: str = "Frequency Spectrum Analysis") -> Dict[str, Any]:
        """Analyze frequency spectrum using FFT
        
        Args:
            audio: Input audio signal
            title: Title for the analysis
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"\n🔍 {title}")
        print("-" * 50)
        
        # Fast Fourier Transform (FFT)
        N = len(audio)
        audio_fft = fft(audio)
        frequencies = fftfreq(N, 1/self.sample_rate)
        
        # Compute magnitude spectrum
        magnitude_spectrum = np.abs(audio_fft)
        power_spectrum = magnitude_spectrum ** 2
        
        # Find dominant frequencies
        pos_freqs = frequencies[:N//2]
        pos_magnitude = magnitude_spectrum[:N//2]
        
        # Statistical analysis
        mean_freq = np.average(pos_freqs, weights=pos_magnitude)
        dominant_freq_idx = np.argmax(pos_magnitude)
        dominant_freq = pos_freqs[dominant_freq_idx]
        
        # Energy distribution
        total_energy = np.sum(power_spectrum)
        low_freq_energy = np.sum(power_spectrum[np.abs(frequencies) < 1000])
        mid_freq_energy = np.sum(power_spectrum[(np.abs(frequencies) >= 1000) & (np.abs(frequencies) < 4000)])
        high_freq_energy = np.sum(power_spectrum[np.abs(frequencies) >= 4000])
        
        analysis_results = {
            'frequencies': frequencies,
            'magnitude_spectrum': magnitude_spectrum,
            'power_spectrum': power_spectrum,
            'mean_frequency': mean_freq,
            'dominant_frequency': dominant_freq,
            'total_energy': total_energy,
            'low_freq_energy_ratio': low_freq_energy / total_energy,
            'mid_freq_energy_ratio': mid_freq_energy / total_energy,
            'high_freq_energy_ratio': high_freq_energy / total_energy
        }
        
        print(f"   Mean frequency: {mean_freq:.2f} Hz")
        print(f"   Dominant frequency: {dominant_freq:.2f} Hz")
        print(f"   Energy distribution:")
        print(f"     Low freq (0-1kHz): {analysis_results['low_freq_energy_ratio']*100:.1f}%")
        print(f"     Mid freq (1-4kHz): {analysis_results['mid_freq_energy_ratio']*100:.1f}%")
        print(f"     High freq (4kHz+): {analysis_results['high_freq_energy_ratio']*100:.1f}%")
        
        return analysis_results
    def design_bandpass_filter(self, low_freq: float, high_freq: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Design Butterworth band-pass filter
        
        Args:
            low_freq: Lower cutoff frequency (Hz)
            high_freq: Higher cutoff frequency (Hz)
            order: Filter order
            
        Returns:
            Filter coefficients (b, a)
        """
        # Ensure frequencies are within valid range
        max_freq = self.nyquist * 0.95  # Use 95% of nyquist frequency as maximum
        low_freq = max(1.0, min(low_freq, max_freq - 100))  # Minimum 1 Hz
        high_freq = min(high_freq, max_freq)
        
        # Ensure high_freq > low_freq
        if high_freq <= low_freq:
            high_freq = low_freq + 100
        
        low_norm = low_freq / self.nyquist
        high_norm = high_freq / self.nyquist
        
        # Final validation
        if low_norm <= 0 or low_norm >= 1 or high_norm <= 0 or high_norm >= 1:
            print(f"⚠️ Invalid filter frequencies, using default range")
            low_norm = 80 / self.nyquist
            high_norm = min(4000, self.nyquist * 0.8) / self.nyquist
        
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        
        print(f"🔧 Designed bandpass filter:")
        print(f"   Frequency range: {low_freq:.1f}-{high_freq:.1f} Hz")
        print(f"   Normalized range: {low_norm:.3f}-{high_norm:.3f}")
        print(f"   Filter order: {order}")
        
        return b, a
    
    def apply_bandpass_filter(self, audio: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Apply band-pass filter to audio signal
        
        Args:
            audio: Input audio signal
            low_freq: Lower cutoff frequency (Hz)
            high_freq: Higher cutoff frequency (Hz)
            
        Returns:
            Filtered audio signal
        """
        b, a = self.design_bandpass_filter(low_freq, high_freq)
        filtered_audio = signal.filtfilt(b, a, audio)
        
        print(f"✅ Applied bandpass filter ({low_freq}-{high_freq} Hz)")
        
        return filtered_audio
    
    def spectral_subtraction_denoising(self, audio: np.ndarray, alpha: float = 1.5, beta: float = 0.1) -> np.ndarray:
        """Apply spectral subtraction noise reduction
        
        Args:
            audio: Input audio signal
            alpha: Over-subtraction factor
            beta: Spectral floor factor
            
        Returns:
            Denoised audio signal
        """
        print(f"\n🧹 Applying spectral subtraction denoising")
        print(f"   Alpha (over-subtraction): {alpha}")
        print(f"   Beta (spectral floor): {beta}")
        
        # Estimate noise from first 0.5 seconds
        noise_duration = int(0.5 * self.sample_rate)
        noise_segment = audio[:noise_duration]
        
        # STFT parameters
        hop_length = 512
        win_length = 1024
        
        # Short-Time Fourier Transform
        stft = librosa.stft(audio, hop_length=hop_length, win_length=win_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise spectrum
        noise_stft = librosa.stft(noise_segment, hop_length=hop_length, win_length=win_length)
        noise_magnitude = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
        
        # Spectral subtraction
        subtracted_magnitude = magnitude - alpha * noise_magnitude
        
        # Apply spectral floor
        spectral_floor = beta * magnitude
        enhanced_magnitude = np.maximum(subtracted_magnitude, spectral_floor)
        
        # Reconstruct signal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length, win_length=win_length)
        
        # Ensure same length as input
        if len(enhanced_audio) > len(audio):
            enhanced_audio = enhanced_audio[:len(audio)]
        elif len(enhanced_audio) < len(audio):
            enhanced_audio = np.pad(enhanced_audio, (0, len(audio) - len(enhanced_audio)))
        
        print(f"✅ Spectral subtraction completed")
        
        return enhanced_audio
    
    def wiener_filter(self, audio: np.ndarray, noise_power_ratio: float = 0.1) -> np.ndarray:
        """Apply Wiener filter for noise reduction
        
        Args:
            audio: Input audio signal
            noise_power_ratio: Estimated noise-to-signal power ratio
            
        Returns:
            Filtered audio signal
        """
        print(f"\n🔧 Applying Wiener filter")
        print(f"   Noise power ratio: {noise_power_ratio}")
        
        # STFT
        hop_length = 512
        win_length = 1024
        stft = librosa.stft(audio, hop_length=hop_length, win_length=win_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate signal power
        signal_power = magnitude ** 2
        
        # Estimate noise power
        noise_power = noise_power_ratio * np.mean(signal_power)
        
        # Wiener filter
        wiener_gain = signal_power / (signal_power + noise_power)
        filtered_magnitude = magnitude * wiener_gain
        
        # Reconstruct signal
        filtered_stft = filtered_magnitude * np.exp(1j * phase)
        filtered_audio = librosa.istft(filtered_stft, hop_length=hop_length, win_length=win_length)
        
        # Ensure same length as input
        if len(filtered_audio) > len(audio):
            filtered_audio = filtered_audio[:len(audio)]
        elif len(filtered_audio) < len(audio):
            filtered_audio = np.pad(filtered_audio, (0, len(audio) - len(filtered_audio)))
        
        print(f"✅ Wiener filtering completed")
        
        return filtered_audio
    def gentle_noise_reduction(self, audio: np.ndarray, reduction_factor: float = 0.3) -> np.ndarray:
        """Apply gentle noise reduction using simple smoothing
        
        Args:
            audio: Input audio signal
            reduction_factor: Noise reduction strength (0-1)
            
        Returns:
            Gently processed audio signal
        """
        print(f"\n🧹 Applying gentle noise reduction")
        print(f"   Reduction factor: {reduction_factor}")
        
        # Simple high-pass filter to remove low-frequency noise
        from scipy.signal import butter, filtfilt
        
        # High-pass filter at 60Hz to remove low-frequency noise
        nyquist = self.sample_rate / 2
        low_cutoff = 60 / nyquist
        
        if low_cutoff < 1.0:  # Only apply if cutoff is valid
            b, a = butter(2, low_cutoff, btype='high')
            filtered_audio = filtfilt(b, a, audio)
        else:
            filtered_audio = audio.copy()
        
        # Apply gentle smoothing to reduce noise
        if reduction_factor > 0:
            # Simple moving average for very gentle smoothing
            window_size = max(3, int(self.sample_rate * 0.001))  # 1ms window
            if window_size % 2 == 0:
                window_size += 1
            
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(filtered_audio, size=window_size, mode='nearest')
            
            # Blend original and smoothed audio
            result = (1 - reduction_factor) * filtered_audio + reduction_factor * smoothed
        else:
            result = filtered_audio
        
        print(f"✅ Gentle noise reduction completed")
        return result
    def create_visualization(self, original_audio: np.ndarray, processed_audio: np.ndarray, 
                           original_analysis: Dict, processed_analysis: Dict, output_path: str):
        """Create comprehensive visualization plots
        
        Args:
            original_audio: Original audio signal
            processed_audio: Processed audio signal
            original_analysis: Analysis results for original audio
            processed_analysis: Analysis results for processed audio
            output_path: Path to save the visualization
        """
        try:
            # Use Agg backend to avoid GUI issues in threads
            import matplotlib
            matplotlib.use('Agg')
            
            plt.style.use('default')
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle('Audio Preprocessing Analysis', fontsize=16, fontweight='bold')
            
            # Time domain comparison
            time_orig = np.arange(len(original_audio)) / self.sample_rate
            time_proc = np.arange(len(processed_audio)) / self.sample_rate
            
            axes[0, 0].plot(time_orig, original_audio, alpha=0.7, color='blue')
            axes[0, 0].set_title('Original Audio - Time Domain')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(time_proc, processed_audio, alpha=0.7, color='red')
            axes[0, 1].set_title('Processed Audio - Time Domain')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Amplitude')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Frequency domain comparison
            orig_freqs = original_analysis['frequencies'][:len(original_audio)//2]
            orig_mag = original_analysis['magnitude_spectrum'][:len(original_audio)//2]
            proc_freqs = processed_analysis['frequencies'][:len(processed_audio)//2]
            proc_mag = processed_analysis['magnitude_spectrum'][:len(processed_audio)//2]
            
            axes[1, 0].semilogy(orig_freqs, orig_mag, alpha=0.7, color='blue')
            axes[1, 0].set_title('Original Audio - Frequency Spectrum')
            axes[1, 0].set_xlabel('Frequency (Hz)')
            axes[1, 0].set_ylabel('Magnitude')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xlim(0, 8000)
            
            axes[1, 1].semilogy(proc_freqs, proc_mag, alpha=0.7, color='red')
            axes[1, 1].set_title('Processed Audio - Frequency Spectrum')
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_ylabel('Magnitude')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xlim(0, 8000)
            
            # Energy distribution comparison
            categories = ['Low\n(0-1kHz)', 'Mid\n(1-4kHz)', 'High\n(4kHz+)']
            orig_ratios = [original_analysis['low_freq_energy_ratio'], 
                          original_analysis['mid_freq_energy_ratio'],
                          original_analysis['high_freq_energy_ratio']]
            proc_ratios = [processed_analysis['low_freq_energy_ratio'],
                          processed_analysis['mid_freq_energy_ratio'], 
                          processed_analysis['high_freq_energy_ratio']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes[2, 0].bar(x - width/2, [r*100 for r in orig_ratios], width, 
                          label='Original', alpha=0.7, color='blue')
            axes[2, 0].bar(x + width/2, [r*100 for r in proc_ratios], width,
                          label='Processed', alpha=0.7, color='red')
            axes[2, 0].set_title('Energy Distribution Comparison')
            axes[2, 0].set_xlabel('Frequency Band')
            axes[2, 0].set_ylabel('Energy Percentage (%)')
            axes[2, 0].set_xticks(x)
            axes[2, 0].set_xticklabels(categories)
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # Spectrograms
            axes[2, 1].specgram(processed_audio, Fs=self.sample_rate, cmap='viridis')
            axes[2, 1].set_title('Processed Audio - Spectrogram')
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].set_ylabel('Frequency (Hz)')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # 关闭图形以释放内存
            print(f"📊 Visualization saved to: {output_path}")
            
            return fig
            
        except Exception as e:
            print(f"⚠️ Visualization creation failed: {e}")
            return None   

    def process_audio(self, input_path: str, output_dir: str, 
                     apply_light_processing: bool = True, low_freq: float = 100, high_freq: float = 8000) -> Dict[str, Any]:
        """Complete audio preprocessing pipeline - 轻量化版本
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save outputs
            apply_light_processing: 是否应用轻量化处理（仅基础带通滤波）
            low_freq: Lower cutoff frequency for bandpass filter
            high_freq: Higher cutoff frequency for bandpass filter
            
        Returns:
            Dictionary containing processing results and file paths
        """
        print("🎵 Starting Audio Preprocessing Pipeline (Light Version)")
        print("=" * 60)
        
        # Load audio
        original_audio = self.load_audio(input_path)
        if original_audio is None:
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze original audio
        original_analysis = self.analyze_frequency_spectrum(original_audio, "Original Audio Analysis")
        
        # Processing pipeline - 简化版本
        processed_audio = original_audio.copy()
        processing_steps = []
        
        if apply_light_processing:
            # 仅应用轻量的带通滤波，去除极低和极高频噪声
            processed_audio = self.apply_bandpass_filter(processed_audio, low_freq, high_freq)
            processing_steps.append(f"Light Bandpass Filter ({low_freq}-{high_freq} Hz)")
        else:
            processing_steps.append("No processing applied")
        
        # Analyze processed audio
        processed_analysis = self.analyze_frequency_spectrum(processed_audio, "Processed Audio Analysis")
        
        # Save processed audio
        input_name = Path(input_path).stem
        output_audio_path = os.path.join(output_dir, f"{input_name}_preprocessed.wav")
        sf.write(output_audio_path, processed_audio, self.sample_rate)
        print(f"💾 Saved processed audio to: {output_audio_path}")
        
        # Create visualization
        viz_path = os.path.join(output_dir, f"{input_name}_analysis.png")
        fig = self.create_visualization(original_audio, processed_audio, 
                                      original_analysis, processed_analysis, viz_path)
        
        # Generate report
        report_path = os.path.join(output_dir, f"{input_name}_report.txt")
        self.generate_report(input_path, output_audio_path, processing_steps,
                           original_analysis, processed_analysis, report_path)
        
        results = {
            'input_path': input_path,
            'output_audio_path': output_audio_path,
            'visualization_path': viz_path,
            'report_path': report_path,
            'original_audio': original_audio,
            'processed_audio': processed_audio,
            'original_analysis': original_analysis,
            'processed_analysis': processed_analysis,
            'processing_steps': processing_steps,
            'figure': fig
        }
        
        print("✅ Audio preprocessing pipeline completed successfully!")
        print("=" * 60)
        
        return results
    
    def generate_report(self, input_path: str, output_path: str, processing_steps: list,
                       original_analysis: Dict, processed_analysis: Dict, report_path: str):
        """Generate detailed processing report
        
        Args:
            input_path: Input file path
            output_path: Output file path
            processing_steps: List of applied processing steps
            original_analysis: Original audio analysis results
            processed_analysis: Processed audio analysis results  
            report_path: Path to save the report
        """
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Audio Preprocessing Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Input Information:\n")
            f.write(f"  Input file: {Path(input_path).name}\n")
            f.write(f"  Output file: {Path(output_path).name}\n")
            f.write(f"  Sample rate: {self.sample_rate} Hz\n\n")
            
            f.write("Processing Steps Applied:\n")
            for i, step in enumerate(processing_steps, 1):
                f.write(f"  {i}. {step}\n")
            f.write("\n")
            
            f.write("Original Audio Analysis:\n")
            f.write(f"  Mean frequency: {original_analysis['mean_frequency']:.2f} Hz\n")
            f.write(f"  Dominant frequency: {original_analysis['dominant_frequency']:.2f} Hz\n")
            f.write(f"  Energy distribution:\n")
            f.write(f"    Low freq (0-1kHz): {original_analysis['low_freq_energy_ratio']*100:.1f}%\n")
            f.write(f"    Mid freq (1-4kHz): {original_analysis['mid_freq_energy_ratio']*100:.1f}%\n")
            f.write(f"    High freq (4kHz+): {original_analysis['high_freq_energy_ratio']*100:.1f}%\n\n")
            
            f.write("Processed Audio Analysis:\n")
            f.write(f"  Mean frequency: {processed_analysis['mean_frequency']:.2f} Hz\n")
            f.write(f"  Dominant frequency: {processed_analysis['dominant_frequency']:.2f} Hz\n")
            f.write(f"  Energy distribution:\n")
            f.write(f"    Low freq (0-1kHz): {processed_analysis['low_freq_energy_ratio']*100:.1f}%\n")
            f.write(f"    Mid freq (1-4kHz): {processed_analysis['mid_freq_energy_ratio']*100:.1f}%\n")
            f.write(f"    High freq (4kHz+): {processed_analysis['high_freq_energy_ratio']*100:.1f}%\n\n")
            
            # Calculate improvements
            energy_change = {
                'low': (processed_analysis['low_freq_energy_ratio'] - original_analysis['low_freq_energy_ratio']) * 100,
                'mid': (processed_analysis['mid_freq_energy_ratio'] - original_analysis['mid_freq_energy_ratio']) * 100,
                'high': (processed_analysis['high_freq_energy_ratio'] - original_analysis['high_freq_energy_ratio']) * 100
            }
            
            f.write("Processing Effects:\n")
            f.write(f"  Low frequency energy change: {energy_change['low']:+.1f}%\n")
            f.write(f"  Mid frequency energy change: {energy_change['mid']:+.1f}%\n")
            f.write(f"  High frequency energy change: {energy_change['high']:+.1f}%\n\n")
            
            f.write("Signal Processing Theory Applied:\n")
            f.write("  1. Fast Fourier Transform (FFT) for frequency domain analysis\n")
            f.write("  2. Butterworth bandpass filter design\n")
            f.write("  3. Spectral subtraction for noise reduction\n")
            f.write("  4. Short-Time Fourier Transform (STFT) for time-frequency analysis\n")
            f.write("  5. Wiener filtering for optimal noise suppression\n")
        
        print(f"📄 Processing report saved to: {report_path}")


def main():
    """Main function for standalone execution"""
    import argparse
    parser = argparse.ArgumentParser(description='Audio Preprocessing Tool - Light Version')
    parser.add_argument('input', help='Input audio file path')
    parser.add_argument('-o', '--output', default='output/preprocessing', help='Output directory')
    parser.add_argument('--no-bandpass', action='store_true', help='Skip light processing')
    parser.add_argument('--low-freq', type=float, default=100, help='Bandpass low frequency')
    parser.add_argument('--high-freq', type=float, default=8000, help='Bandpass high frequency')
    
    args = parser.parse_args()
    preprocessor = AudioPreprocessor()
    results = preprocessor.process_audio(
        input_path=args.input,
        output_dir=args.output,
        apply_light_processing=not args.no_bandpass,  # 使用新的参数名
        low_freq=args.low_freq,
        high_freq=args.high_freq
    )
    
    if results:
        print(f"\n🎉 Processing completed successfully!")
        print(f"Output files:")
        print(f"  Audio: {results['output_audio_path']}")
        print(f"  Visualization: {results['visualization_path']}")
        print(f"  Report: {results['report_path']}")


if __name__ == "__main__":
    main()
