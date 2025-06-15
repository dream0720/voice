#!/usr/bin/env python3
"""
Complete Voice Processing Pipeline
=================================

This module integrates all processing steps into a complete pipeline:
1. Audio Preprocessing (filtering, noise reduction)
2. Source Separation (Demucs)
3. Speaker Separation (Pyannote/VoiceFilter-WavLM)
4. Voice Matching (reference-based selection)

The pipeline can be used programmatically or through command line interface.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import processing modules
from modules.preprocessing import AudioPreprocessor
from modules.source_separation import DemucsSourceSeparator
from modules.speaker_separation import SpeakerSeparator
from modules.voice_matching import VoiceMatcher
from modules.utils import PathManager, Logger, ProgressTracker, config_manager


class VoiceProcessingPipeline:
    """Complete voice processing pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None, hf_token: str = None):
        """Initialize the processing pipeline
        
        Args:
            config: Configuration dictionary (uses default if None)
            hf_token: Hugging Face access token for speaker separation
        """
        self.config = config or config_manager.config
        self.hf_token = hf_token or "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"
        
        # Initialize processing modules
        sample_rate = self.config['audio']['sample_rate']
        
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.source_separator = DemucsSourceSeparator(
            model_name=self.config['demucs']['model']
        )
        self.speaker_separator = SpeakerSeparator(
            hf_token=self.hf_token,
            device=self.config['demucs']['device']
        )
        self.voice_matcher = VoiceMatcher(sample_rate=sample_rate)
        
        # Setup output directories
        self.base_output_dir = self.config['paths']['output_dir']
        self.setup_output_directories()
        
        self.logger = Logger("pipeline")
        
        print("🚀 Voice Processing Pipeline Initialized")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Demucs Model: {self.config['demucs']['model']}")
        print(f"   Device: {self.config['demucs']['device']}")
        print(f"   HF Token: {'✅ Available' if self.hf_token else '❌ Missing'}")
    
    def setup_output_directories(self):
        """Setup output directory structure"""
        self.output_dirs = {
            'preprocessing': os.path.join(self.base_output_dir, 'preprocessing'),
            'demucs_output': os.path.join(self.base_output_dir, 'demucs_output'),
            'speaker_output': os.path.join(self.base_output_dir, 'speaker_output'),
            'final_output': os.path.join(self.base_output_dir, 'final_output')
        }
        
        for dir_path in self.output_dirs.values():
            PathManager.ensure_dir(dir_path)
    
    def run_preprocessing(self, input_path: str, enable_steps: Dict[str, bool] = None) -> Dict[str, Any]:
        """Run audio preprocessing step
        
        Args:
            input_path: Path to input audio file
            enable_steps: Dictionary specifying which preprocessing steps to enable
            
        Returns:
            Preprocessing results dictionary
        """
        print("\n" + "="*60)
        print("🎵 STEP 1: AUDIO PREPROCESSING")
        print("="*60)
        
        enable_steps = enable_steps or {
            'bandpass': self.config['preprocessing']['apply_bandpass'],
            'spectral_subtraction': self.config['preprocessing']['apply_spectral_subtraction'],
            'wiener': self.config['preprocessing']['apply_wiener']
        }
        
        try:
            results = self.preprocessor.process_audio(
                input_path=input_path,
                output_dir=self.output_dirs['preprocessing'],
                apply_bandpass=enable_steps['bandpass'],
                apply_spectral_subtraction=enable_steps['spectral_subtraction'],
                apply_wiener=enable_steps['wiener'],
                low_freq=self.config['preprocessing']['low_freq'],
                high_freq=self.config['preprocessing']['high_freq']
            )
            
            if results:
                self.logger.info(f"Preprocessing completed: {results['output_audio_path']}")
                return results
            else:
                raise Exception("Preprocessing failed")
                
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            return None
    
    def run_source_separation(self, input_path: str) -> Dict[str, Any]:
        """Run source separation step using Demucs
        
        Args:
            input_path: Path to preprocessed audio file
            
        Returns:
            Source separation results dictionary
        """
        print("\n" + "="*60)
        print("🎼 STEP 2: SOURCE SEPARATION (DEMUCS)")
        print("="*60)
        
        try:
            results = self.source_separator.separate_audio(
                input_path=input_path,
                output_dir=self.output_dirs['demucs_output'],
                device=self.config['demucs']['device']
            )
            
            if results and results['success']:
                self.logger.info(f"Source separation completed: {len(results['separated_files'])} stems")
                return results
            else:
                raise Exception("Source separation failed")
                
        except Exception as e:
            self.logger.error(f"Source separation error: {e}")
            return None
    
    def run_speaker_separation(self, vocals_path: str, method: str = "pyannote") -> Dict[str, Any]:
        """Run speaker separation step
        
        Args:
            vocals_path: Path to separated vocals file
            method: Separation method ("pyannote" or "voicefilter")
            
        Returns:
            Speaker separation results dictionary
        """
        print("\n" + "="*60)
        print(f"👥 STEP 3: SPEAKER SEPARATION ({method.upper()})")
        print("="*60)
        
        try:
            if method == "pyannote":
                results = self.speaker_separator.separate_speakers_pyannote(
                    input_path=vocals_path,
                    output_dir=self.output_dirs['speaker_output']
                )
            elif method == "voicefilter":
                # For VoiceFilter, we need a reference audio
                reference_path = os.path.join(self.config['paths']['reference_dir'], 'lttgd_ref.wav')
                results = self.speaker_separator.separate_with_voicefilter_wavlm(
                    input_path=vocals_path,
                    reference_path=reference_path,
                    output_dir=self.output_dirs['speaker_output']
                )
            else:
                raise ValueError(f"Unknown speaker separation method: {method}")
            
            if results and results['success']:
                self.logger.info(f"Speaker separation completed using {method}")
                return results
            else:
                raise Exception(f"Speaker separation failed with {method}")
                
        except Exception as e:
            self.logger.error(f"Speaker separation error: {e}")
            return None
    
    def run_voice_matching(self, candidate_paths: List[str], reference_path: str) -> Dict[str, Any]:
        """Run voice matching step
        
        Args:
            candidate_paths: List of paths to candidate audio files
            reference_path: Path to reference audio file
            
        Returns:
            Voice matching results dictionary
        """
        print("\n" + "="*60)
        print("🎯 STEP 4: VOICE MATCHING")
        print("="*60)
        
        try:
            results = self.voice_matcher.match_voices(
                reference_path=reference_path,
                candidate_paths=candidate_paths,
                output_dir=self.output_dirs['final_output']
            )
            
            if results and results['success']:
                self.logger.info(f"Voice matching completed: {results['best_match_name']}")
                return results
            else:
                raise Exception("Voice matching failed")
                
        except Exception as e:
            self.logger.error(f"Voice matching error: {e}")
            return None
    
    def run_complete_pipeline(self, input_path: str, reference_path: str,
                            enable_preprocessing: bool = True,
                            speaker_separation_method: str = "pyannote") -> Dict[str, Any]:
        """Run the complete processing pipeline
        
        Args:
            input_path: Path to mixed input audio file
            reference_path: Path to reference audio file for matching
            enable_preprocessing: Whether to run preprocessing step
            speaker_separation_method: Method for speaker separation
            
        Returns:
            Complete pipeline results dictionary
        """
        print("🚀 STARTING COMPLETE VOICE PROCESSING PIPELINE")
        print("="*80)
        
        start_time = datetime.now()
        pipeline_results = {
            'input_path': input_path,
            'reference_path': reference_path,
            'start_time': start_time,
            'steps_completed': [],
            'success': False
        }
        
        try:
            # Initialize progress tracker
            total_steps = 4 if enable_preprocessing else 3
            progress = ProgressTracker(total_steps, "Pipeline Progress")
            
            # Step 1: Preprocessing (optional)
            if enable_preprocessing:
                progress.update(message="Audio Preprocessing")
                preprocessing_results = self.run_preprocessing(input_path)
                
                if not preprocessing_results:
                    raise Exception("Preprocessing step failed")
                
                pipeline_results['preprocessing'] = preprocessing_results
                pipeline_results['steps_completed'].append('preprocessing')
                
                # Use preprocessed audio for next step
                current_audio_path = preprocessing_results['output_audio_path']
            else:
                current_audio_path = input_path
            
            # Step 2: Source Separation
            progress.update(message="Source Separation")
            source_sep_results = self.run_source_separation(current_audio_path)
            
            if not source_sep_results:
                raise Exception("Source separation step failed")
            
            pipeline_results['source_separation'] = source_sep_results
            pipeline_results['steps_completed'].append('source_separation')
            
            # Get vocals file for speaker separation
            vocals_path = source_sep_results['separated_files'].get('vocals')
            if not vocals_path or not os.path.exists(vocals_path):
                raise Exception("Vocals file not found from source separation")
            
            # Step 3: Speaker Separation
            progress.update(message=f"Speaker Separation ({speaker_separation_method})")
            speaker_sep_results = self.run_speaker_separation(vocals_path, speaker_separation_method)
            
            if not speaker_sep_results:
                raise Exception("Speaker separation step failed")
            
            pipeline_results['speaker_separation'] = speaker_sep_results
            pipeline_results['steps_completed'].append('speaker_separation')
            
            # Get candidate files for voice matching
            if speaker_separation_method == "pyannote":
                candidate_paths = list(speaker_sep_results['separated_files'].values())
            else:  # voicefilter
                candidate_paths = [speaker_sep_results['output_path']]
            
            # Step 4: Voice Matching
            progress.update(message="Voice Matching")
            voice_matching_results = self.run_voice_matching(candidate_paths, reference_path)
            
            if not voice_matching_results:
                raise Exception("Voice matching step failed")
            
            pipeline_results['voice_matching'] = voice_matching_results
            pipeline_results['steps_completed'].append('voice_matching')
            
            # Pipeline completed successfully
            progress.finish("Successfully")
            pipeline_results['success'] = True
            pipeline_results['end_time'] = datetime.now()
            pipeline_results['total_duration'] = pipeline_results['end_time'] - start_time
            
            # Generate final report
            self.generate_pipeline_report(pipeline_results)
            
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"💾 Final Output: {voice_matching_results['best_match_output']}")
            print(f"📊 Best Match Score: {voice_matching_results['best_score']:.3f}")
            print(f"⏱️  Total Duration: {pipeline_results['total_duration'].total_seconds():.1f}s")
            print("="*80)
            
            return pipeline_results
            
        except Exception as e:
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now()
            
            self.logger.error(f"Pipeline failed: {e}")
            print(f"\n❌ PIPELINE FAILED: {e}")
            print("="*80)
            
            return pipeline_results
    
    def generate_pipeline_report(self, results: Dict[str, Any]):
        """Generate comprehensive pipeline report
        
        Args:
            results: Complete pipeline results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dirs['final_output'], 
                                 f"pipeline_report_{timestamp}.txt")
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("Voice Processing Pipeline Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Pipeline Information:\n")
                f.write(f"  Input file: {Path(results['input_path']).name}\n")
                f.write(f"  Reference file: {Path(results['reference_path']).name}\n")
                f.write(f"  Start time: {results['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  End time: {results['end_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  Total duration: {results['total_duration'].total_seconds():.1f} seconds\n")
                f.write(f"  Steps completed: {', '.join(results['steps_completed'])}\n")
                f.write(f"  Success: {'Yes' if results['success'] else 'No'}\n\n")
                
                if results['success']:
                    best_match = results['voice_matching']
                    f.write("Final Results:\n")
                    f.write(f"  Best match: {best_match['best_match_name']}\n")
                    f.write(f"  Similarity score: {best_match['best_score']:.4f}\n")
                    f.write(f"  Output file: {Path(best_match['best_match_output']).name}\n\n")
                else:
                    f.write(f"Error: {results.get('error', 'Unknown error')}\n\n")
                
                f.write("Step Details:\n")
                for step in results['steps_completed']:
                    if step in results:
                        step_results = results[step]
                        f.write(f"  {step.replace('_', ' ').title()}:\n")
                        
                        if step == 'preprocessing':
                            f.write(f"    Input: {Path(step_results['input_path']).name}\n")
                            f.write(f"    Output: {Path(step_results['output_audio_path']).name}\n")
                            f.write(f"    Steps: {', '.join(step_results['processing_steps'])}\n")
                        
                        elif step == 'source_separation':
                            f.write(f"    Model: {step_results['model_used']}\n")
                            f.write(f"    Stems: {', '.join(step_results['separated_files'].keys())}\n")
                        
                        elif step == 'speaker_separation':
                            if 'speakers' in step_results:
                                f.write(f"    Speakers found: {len(step_results['speakers'])}\n")
                                f.write(f"    Speaker IDs: {', '.join(step_results['speakers'])}\n")
                        
                        elif step == 'voice_matching':
                            f.write(f"    Candidates analyzed: {len(step_results['candidate_paths'])}\n")
                            f.write(f"    Best match: {step_results['best_match_name']}\n")
                            f.write(f"    Score: {step_results['best_score']:.4f}\n")
                        
                        f.write("\n")
                
                f.write("Technical Details:\n")
                f.write(f"  Sample rate: {self.config['audio']['sample_rate']} Hz\n")
                f.write(f"  Demucs model: {self.config['demucs']['model']}\n")
                f.write(f"  Processing device: {self.config['demucs']['device']}\n")
                f.write(f"  MFCC coefficients: {self.config['audio']['n_mfcc']}\n")
                f.write(f"  Mel filters: {self.config['audio']['n_mels']}\n")
            
            print(f"📄 Pipeline report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate pipeline report: {e}")


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Voice Processing Pipeline')
    parser.add_argument('input', help='Input mixed audio file path')
    parser.add_argument('reference', help='Reference audio file path')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    parser.add_argument('--no-preprocessing', action='store_true', 
                       help='Skip preprocessing step')
    parser.add_argument('--speaker-method', choices=['pyannote', 'voicefilter'], 
                       default='pyannote', help='Speaker separation method')
    parser.add_argument('--hf-token', help='Hugging Face access token')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Processing device')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = config_manager.config.copy()
    config['paths']['output_dir'] = args.output
    config['demucs']['device'] = args.device
    
    # Initialize pipeline
    pipeline = VoiceProcessingPipeline(config=config, hf_token=args.hf_token)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        input_path=args.input,
        reference_path=args.reference,
        enable_preprocessing=not args.no_preprocessing,
        speaker_separation_method=args.speaker_method
    )
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()
