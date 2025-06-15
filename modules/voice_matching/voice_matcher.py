#!/usr/bin/env python3
"""
Voice Matching Module
====================

This module provides voice matching capabilities to identify the best matching
separated voice based on reference audio using multiple acoustic features:
- MFCC (Mel-frequency cepstral coefficients)
- Mel-spectrogram features
- Spectral features (centroid, rolloff, zero-crossing rate)
- Temporal features (tempo, rhythm)
- Comprehensive similarity analysis
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import os
from typing import Dict, List, Tuple, Optional, Any
import subprocess


class VoiceMatcher:
    """Voice matching processor for identifying similar voices"""
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize the voice matcher
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self.hop_length = 256
        self.n_fft = 1024
        self.n_mfcc = 13
        self.n_mels = 80
        
        print(f"🎵 Voice Matcher initialized")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   MFCC coefficients: {self.n_mfcc}")
        print(f"   Mel filters: {self.n_mels}")
    
    def load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load audio file with error handling
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Loaded audio signal as numpy array, or None if failed
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(audio) / self.sample_rate
            print(f"📁 Loaded: {Path(audio_path).name} ({duration:.2f}s)")
            return audio
        except Exception as e:
            print(f"❌ Failed to load {audio_path}: {e}")
            return None
    
    def convert_audio_format(self, input_path: str, output_path: str) -> bool:
        """Convert audio format (64-bit to 16-bit)
        
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
            print(f"  Max absolute value: {max_val}")
            
            # Normalize if exceeding normal range
            if max_val > 1:
                print(f"  ⚠️  Data exceeds range, normalizing...")
                data = data / max_val
                max_val = 1.0
            
            # Convert to int16 range
            if data.dtype != np.int16:
                print(f"  🔄 Converting to 16-bit format...")
                data_int16 = np.round(data * 32767).astype(np.int16)
            else:
                data_int16 = data
            
            # Save converted audio
            sf.write(output_path, data_int16, samplerate, subtype='PCM_16')
            print(f"  ✅ Format conversion completed: {Path(output_path).name}")
            return True
            
        except Exception as e:
            print(f"  ❌ Format conversion failed: {e}")
            return False
    
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio
        
        Args:
            audio: Input audio signal
            
        Returns:
            MFCC feature matrix
        """
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        return mfcc
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram features
        
        Args:
            audio: Input audio signal
            
        Returns:
            Mel-spectrogram feature matrix
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral features from audio
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary containing various spectral features
        """
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length)[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        
        # RMS energy
        rms = librosa.feature.rms(
            y=audio, hop_length=self.hop_length)[0]
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zcr,
            'spectral_bandwidth': spectral_bandwidth,
            'rms_energy': rms
        }
    
    def extract_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract temporal features from audio
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary containing temporal features
        """
        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        
        # Duration
        duration = len(audio) / self.sample_rate
        
        # Energy-based features
        short_time_energy = np.sum(audio ** 2)
        avg_energy = short_time_energy / len(audio)
        
        return {
            'tempo': tempo,
            'duration': duration,
            'total_energy': short_time_energy,
            'average_energy': avg_energy
        }
    
    def extract_comprehensive_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive feature set from audio
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary containing all extracted features
        """
        print("  🔍 Extracting comprehensive features...")
        
        features = {}
        
        # MFCC features
        features['mfcc'] = self.extract_mfcc_features(audio)
        features['mfcc_mean'] = np.mean(features['mfcc'], axis=1)
        features['mfcc_std'] = np.std(features['mfcc'], axis=1)
        
        # Mel-spectrogram features
        features['mel_spectrogram'] = self.extract_mel_spectrogram(audio)
        features['mel_mean'] = np.mean(features['mel_spectrogram'], axis=1)
        features['mel_std'] = np.std(features['mel_spectrogram'], axis=1)
        
        # Spectral features
        spectral_features = self.extract_spectral_features(audio)
        for key, value in spectral_features.items():
            features[f'{key}_mean'] = np.mean(value)
            features[f'{key}_std'] = np.std(value)
        
        # Temporal features
        temporal_features = self.extract_temporal_features(audio)
        features.update(temporal_features)
        
        # Fundamental frequency estimation
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0_clean = f0[voiced_flag]
            if len(f0_clean) > 0:
                features['f0_mean'] = np.mean(f0_clean)
                features['f0_std'] = np.std(f0_clean)
                features['voiced_ratio'] = np.sum(voiced_flag) / len(voiced_flag)
            else:
                features['f0_mean'] = 0
                features['f0_std'] = 0
                features['voiced_ratio'] = 0
        except:
            features['f0_mean'] = 0
            features['f0_std'] = 0
            features['voiced_ratio'] = 0
        
        print(f"    ✅ Extracted {len(features)} feature types")
        return features
    
    def calculate_feature_similarity(self, ref_features: Dict[str, Any], 
                                   target_features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate similarity between reference and target features
        
        Args:
            ref_features: Reference audio features
            target_features: Target audio features
            
        Returns:
            Dictionary containing similarity scores
        """
        similarities = {}
        
        # MFCC similarity (cosine similarity)
        if 'mfcc_mean' in ref_features and 'mfcc_mean' in target_features:
            mfcc_sim = 1 - cosine(ref_features['mfcc_mean'], target_features['mfcc_mean'])
            similarities['mfcc_similarity'] = max(0, mfcc_sim)  # Ensure non-negative
        
        # Mel-spectrogram similarity
        if 'mel_mean' in ref_features and 'mel_mean' in target_features:
            mel_sim = 1 - cosine(ref_features['mel_mean'], target_features['mel_mean'])
            similarities['mel_similarity'] = max(0, mel_sim)
        
        # Spectral feature similarities
        spectral_features = ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'rms_energy']
        spectral_similarities = []
        
        for feature in spectral_features:
            mean_key = f'{feature}_mean'
            if mean_key in ref_features and mean_key in target_features:
                # Normalized absolute difference (converted to similarity)
                ref_val = ref_features[mean_key]
                target_val = target_features[mean_key]
                
                if ref_val != 0:
                    diff_ratio = abs(target_val - ref_val) / abs(ref_val)
                    similarity = np.exp(-diff_ratio)  # Exponential decay similarity
                    spectral_similarities.append(similarity)
                    similarities[f'{feature}_similarity'] = similarity
        
        if spectral_similarities:
            similarities['spectral_avg_similarity'] = np.mean(spectral_similarities)
        
        # F0 (pitch) similarity
        if ('f0_mean' in ref_features and 'f0_mean' in target_features and 
            ref_features['f0_mean'] > 0 and target_features['f0_mean'] > 0):
            
            f0_ratio = min(ref_features['f0_mean'], target_features['f0_mean']) / \
                      max(ref_features['f0_mean'], target_features['f0_mean'])
            similarities['f0_similarity'] = f0_ratio
        
        # Tempo similarity
        if 'tempo' in ref_features and 'tempo' in target_features:
            tempo_ratio = min(ref_features['tempo'], target_features['tempo']) / \
                         max(ref_features['tempo'], target_features['tempo'])
            similarities['tempo_similarity'] = tempo_ratio
        
        # Voiced ratio similarity
        if 'voiced_ratio' in ref_features and 'voiced_ratio' in target_features:
            voiced_diff = abs(ref_features['voiced_ratio'] - target_features['voiced_ratio'])
            similarities['voiced_similarity'] = 1 - voiced_diff
        
        return similarities
    
    def calculate_composite_score(self, similarities: Dict[str, float]) -> float:
        """Calculate composite similarity score with weighted features
        
        Args:
            similarities: Dictionary of individual similarity scores
            
        Returns:
            Composite similarity score (0-1)
        """
        # Define feature weights based on importance for voice matching
        weights = {
            'mfcc_similarity': 0.25,        # Most important for voice characteristics
            'mel_similarity': 0.20,         # Important for spectral characteristics
            'spectral_avg_similarity': 0.15, # Important for voice quality
            'f0_similarity': 0.15,          # Important for pitch matching
            'tempo_similarity': 0.10,       # Moderately important
            'voiced_similarity': 0.10,      # Moderately important
        }
        
        total_weight = 0
        weighted_score = 0
        
        for feature, weight in weights.items():
            if feature in similarities:
                weighted_score += similarities[feature] * weight
                total_weight += weight
        
        # Normalize by actual total weight to handle missing features
        if total_weight > 0:
            composite_score = weighted_score / total_weight
        else:
            composite_score = 0
        
        return composite_score
    
    def match_voices(self, reference_path: str, candidate_paths: List[str], 
                    output_dir: str) -> Dict[str, Any]:
        """Match voices against reference audio and select the best match
        
        Args:
            reference_path: Path to reference audio file
            candidate_paths: List of paths to candidate audio files
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing matching results
        """
        print("🎵 Starting Voice Matching Analysis")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load reference audio
        print(f"📁 Loading reference audio...")
        ref_audio = self.load_audio(reference_path)
        if ref_audio is None:
            return None
        
        # Extract reference features
        print("🔍 Analyzing reference audio features...")
        ref_features = self.extract_comprehensive_features(ref_audio)
        
        # Analyze all candidates
        candidates_analysis = {}
        similarities_results = {}
        
        print(f"\n👥 Analyzing {len(candidate_paths)} candidate audio files...")
        
        for i, candidate_path in enumerate(candidate_paths, 1):
            candidate_name = Path(candidate_path).stem
            print(f"\n[{i}/{len(candidate_paths)}] Processing: {candidate_name}")
            
            # Load candidate audio
            candidate_audio = self.load_audio(candidate_path)
            if candidate_audio is None:
                continue
              # Extract candidate features
            candidate_features = self.extract_comprehensive_features(candidate_audio)
            candidates_analysis[candidate_name] = candidate_features
            
            # Calculate similarities
            similarities = self.calculate_feature_similarity(ref_features, candidate_features)
            composite_score = self.calculate_composite_score(similarities)
            
            # 确保 composite_score 是标量
            if isinstance(composite_score, np.ndarray):
                composite_score = float(composite_score.item())
            
            similarities['composite_score'] = composite_score
            similarities_results[candidate_name] = similarities
            
            print(f"    📊 Composite similarity score: {composite_score:.3f}")
            
            # Show top similarity metrics
            top_metrics = sorted([(k, v) for k, v in similarities.items() 
                                if k != 'composite_score'], key=lambda x: x[1], reverse=True)[:3]
            for metric, score in top_metrics:
                print(f"      {metric}: {score:.3f}")
        
        # Find best match
        if similarities_results:
            best_match = max(similarities_results.items(), key=lambda x: x[1]['composite_score'])
            best_candidate_name, best_similarities = best_match
            print(f"\n🏆 Best Match Found!")
            print(f"   Candidate: {best_candidate_name}")
            
            # 确保 best_score 是标量
            best_score = best_similarities['composite_score']
            if isinstance(best_score, np.ndarray):
                best_score = float(best_score.item())
            
            print(f"   Composite Score: {best_score:.3f}")
            
            # Copy best match to output
            best_candidate_path = None
            for path in candidate_paths:
                if Path(path).stem == best_candidate_name:
                    best_candidate_path = path
                    break
            
            if best_candidate_path:
                ref_name = Path(reference_path).stem
                output_path = os.path.join(output_dir, f"{ref_name}_best_match.wav")
                
                # Convert format if needed and copy
                if self.convert_audio_format(best_candidate_path, output_path):
                    print(f"💾 Best match saved to: {output_path}")
                else:
                    # Fallback: direct copy
                    import shutil
                    shutil.copy2(best_candidate_path, output_path)
                    print(f"💾 Best match copied to: {output_path}")
            
            # Create visualization
            viz_path = os.path.join(output_dir, f"{ref_name}_matching_analysis.png")
            self.create_matching_visualization(similarities_results, viz_path)
            
            # Generate detailed report
            report_path = os.path.join(output_dir, f"{ref_name}_matching_report.txt")
            self.generate_matching_report(reference_path, candidate_paths, 
                                        similarities_results, best_match, report_path)
            
            results = {
                'reference_path': reference_path,
                'candidate_paths': candidate_paths,
                'best_match_name': best_candidate_name,
                'best_match_path': best_candidate_path,
                'best_match_output': output_path,
                'best_score': best_similarities['composite_score'],
                'all_similarities': similarities_results,
                'reference_features': ref_features,
                'candidates_analysis': candidates_analysis,
                'visualization_path': viz_path,
                'report_path': report_path,
                'success': True
            }
            
            print("✅ Voice matching analysis completed successfully!")
            print("=" * 60)
            
            return results
        
        else:
            print("❌ No valid candidates found for analysis")
            return None
    
    def create_matching_visualization(self, similarities_results: Dict[str, Dict[str, float]], 
                                    output_path: str):
        """Create visualization of matching results
        
        Args:
            similarities_results: Dictionary of similarity results for all candidates
            output_path: Path to save the visualization
        """
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
          # Overall similarity scores
        names = list(similarities_results.keys())
        composite_scores = []
        
        # 确保所有 composite_scores 都是标量
        for name in names:
            score = similarities_results[name]['composite_score']
            if isinstance(score, np.ndarray):
                composite_scores.append(float(score.item()))
            else:
                composite_scores.append(float(score))
        
        bars1 = ax1.bar(range(len(names)), composite_scores, alpha=0.7, color='skyblue')
        ax1.set_title('Voice Matching - Composite Similarity Scores', fontweight='bold')
        ax1.set_xlabel('Candidate Audio Files')
        ax1.set_ylabel('Similarity Score')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
          # Add value labels on bars
        for bar, score in zip(bars1, composite_scores):
            height = bar.get_height()
            # 确保 score 是标量
            if isinstance(score, np.ndarray):
                score = float(score.item())
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
          # Detailed feature comparison for best match
        if names:
            best_name = max(names, key=lambda x: similarities_results[x]['composite_score'])
            best_similarities = similarities_results[best_name]
            
            feature_names = [k for k in best_similarities.keys() if k != 'composite_score']
            feature_scores = []
            
            # 确保所有 feature_scores 都是标量
            for k in feature_names:
                score = best_similarities[k]
                if isinstance(score, np.ndarray):
                    if score.size == 1:
                        feature_scores.append(float(score.item()))
                    else:
                        # 如果是多维数组，取平均值
                        feature_scores.append(float(np.mean(score)))
                elif isinstance(score, (list, tuple)):
                    # 如果是列表或元组，取平均值
                    feature_scores.append(float(np.mean(score)))
                else:
                    feature_scores.append(float(score))
            
            bars2 = ax2.barh(range(len(feature_names)), feature_scores, alpha=0.7, color='lightcoral')
            ax2.set_title(f'Feature Breakdown - Best Match: {best_name}', fontweight='bold')
            ax2.set_xlabel('Similarity Score')
            ax2.set_ylabel('Features')
            ax2.set_yticks(range(len(feature_names)))
            ax2.set_yticklabels([name.replace('_', ' ').title() for name in feature_names])
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 1)
              # Add value labels on bars
            for bar, score in zip(bars2, feature_scores):
                width = bar.get_width()
                # 确保 score 是标量
                if isinstance(score, np.ndarray):
                    score = float(score.item())
                ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                        f'{score:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Matching visualization saved to: {output_path}")
    
    def generate_matching_report(self, reference_path: str, candidate_paths: List[str],
                               similarities_results: Dict[str, Dict[str, float]], 
                               best_match: Tuple[str, Dict[str, float]], report_path: str):
        """Generate detailed matching report
        
        Args:
            reference_path: Reference audio file path
            candidate_paths: List of candidate file paths
            similarities_results: All similarity results
            best_match: Best match result tuple
            report_path: Path to save the report
        """
        best_name, best_similarities = best_match
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Voice Matching Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Analysis Information:\n")
            f.write(f"  Reference audio: {Path(reference_path).name}\n")
            f.write(f"  Candidates analyzed: {len(candidate_paths)}\n")
            f.write(f"  Feature extraction: MFCC, Mel-spectrogram, Spectral, Temporal\n\n")
            f.write("Best Match Results:\n")
            f.write(f"  Best candidate: {best_name}\n")
            
            # 确保 best composite_score 是标量
            best_composite_score = best_similarities['composite_score']
            if isinstance(best_composite_score, np.ndarray):
                best_composite_score = float(best_composite_score.item())
            else:
                best_composite_score = float(best_composite_score)
                
            f.write(f"  Composite score: {best_composite_score:.4f}\n\n")
            f.write("Detailed Feature Scores (Best Match):\n")
            for feature, score in sorted(best_similarities.items(), key=lambda x: x[1], reverse=True):
                if feature != 'composite_score':
                    # 确保 score 是标量
                    if isinstance(score, np.ndarray):
                        if score.size == 1:
                            score_value = float(score.item())
                        else:
                            score_value = float(np.mean(score))
                    elif isinstance(score, (list, tuple)):
                        score_value = float(np.mean(score))
                    else:
                        score_value = float(score)
                    
                    f.write(f"  {feature.replace('_', ' ').title()}: {score_value:.4f}\n")
            f.write("\n")
            
            f.write("All Candidates Ranking:\n")
            ranked_results = sorted(similarities_results.items(), 
                                  key=lambda x: x[1]['composite_score'], reverse=True)
            for rank, (name, similarities) in enumerate(ranked_results, 1):
                # 确保 composite_score 是标量
                composite_score = similarities['composite_score']
                if isinstance(composite_score, np.ndarray):
                    composite_score = float(composite_score.item())
                else:
                    composite_score = float(composite_score)
                    
                f.write(f"  {rank}. {name}: {composite_score:.4f}\n")
            f.write("\n")
            
            f.write("Feature Analysis Methodology:\n")
            f.write("  • MFCC (Mel-frequency Cepstral Coefficients): Voice timbre characteristics\n")
            f.write("  • Mel-spectrogram: Frequency distribution analysis\n")
            f.write("  • Spectral features: Voice quality metrics (centroid, rolloff, bandwidth)\n")
            f.write("  • F0 (Fundamental frequency): Pitch characteristics\n")
            f.write("  • Temporal features: Rhythm and timing patterns\n")
            f.write("  • Composite scoring: Weighted combination of all features\n\n")
            
            f.write("Similarity Calculation:\n")
            f.write("  • Cosine similarity for MFCC and Mel features\n")
            f.write("  • Normalized difference for spectral features\n")
            f.write("  • Ratio-based similarity for pitch and tempo\n")
            f.write("  • Weighted average for composite score\n")
        
        print(f"📄 Matching report saved to: {report_path}")


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Voice Matching Tool')
    parser.add_argument('reference', help='Reference audio file path')
    parser.add_argument('candidates', nargs='+', help='Candidate audio file paths')
    parser.add_argument('-o', '--output', default='output/final_output', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    matcher = VoiceMatcher()
    results = matcher.match_voices(
        reference_path=args.reference,
        candidate_paths=args.candidates,
        output_dir=args.output
    )

    if results and results['success']:
        print(f"\n🎉 Voice matching completed successfully!")
        
        # 确保 best_score 是标量
        best_score = results['best_score']
        if isinstance(best_score, np.ndarray):
            best_score = float(best_score.item())
            
        print(f"Best match: {results['best_match_name']} (score: {best_score:.3f})")
        print(f"Output: {results['best_match_output']}")
        print(f"Report: {results['report_path']}")
    else:
        print("❌ Voice matching failed!")
        sys.exit(1)  


if __name__ == "__main__":
    main()
