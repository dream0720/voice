#!/usr/bin/env python3
"""
Test Script for Demucs Source Separation Module
==============================================

This script allows you to test the Demucs source separation module independently.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.source_separation.demucs_separator import DemucsSourceSeparator
from modules.utils.audio_converter import AudioConverter


def test_demucs_separation():
    """Test Demucs source separation with audio conversion"""
    
    print("🎵 Testing Demucs Source Separation Module")
    print("=" * 60)
    
    # Configuration
    input_file = "input/lttgd.wav"  # Change this to your test file
    output_dir = "output/test_demucs"
    model_name = "htdemucs"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Please place a test audio file in the input directory")
        return False
    
    # Initialize separator
    separator = DemucsSourceSeparator(model_name=model_name)
    
    # Run separation
    print(f"\n🔄 Running Demucs separation...")
    results = separator.separate_audio(
        input_path=input_file,
        output_dir=output_dir,
        device="cpu"  # Change to "cuda" if you have GPU
    )
    
    if not results:
        print("❌ Separation failed!")
        return False
    
    print(f"\n✅ Separation completed successfully!")
    
    # Convert and fix separated audio files
    print(f"\n🔄 Converting separated audio files...")
    converter = AudioConverter()
    
    converted_files = []
    for stem, file_path in results['separated_files'].items():
        if os.path.exists(file_path):
            print(f"\n📁 Processing {stem}...")
            converted_path = converter.convert_and_fix_audio(
                input_path=file_path,
                target_subtype='PCM_16',
                safety_margin=0.95
            )
            if converted_path:
                converted_files.append((stem, converted_path))
                print(f"  ✅ Converted: {converted_path}")
            else:
                print(f"  ❌ Failed to convert: {file_path}")
    
    # Summary
    print(f"\n🎉 Test completed successfully!")
    print(f"📊 Results Summary:")
    print(f"  Input file: {input_file}")
    print(f"  Output directory: {results['output_dir']}")
    print(f"  Model used: {results['model_used']}")
    print(f"  Original separated files: {len(results['separated_files'])}")
    print(f"  Converted files: {len(converted_files)}")
    
    print(f"\n📁 Separated stems:")
    for stem, file_path in results['separated_files'].items():
        print(f"  🎵 {stem.capitalize()}: {file_path}")
    
    print(f"\n📁 Converted files:")
    for stem, converted_path in converted_files:
        print(f"  🎵 {stem.capitalize()} (fixed): {converted_path}")
    
    return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Demucs Source Separation Module')
    parser.add_argument('-i', '--input', default='input/lttgd.wav', help='Input audio file')
    parser.add_argument('-o', '--output', default='output/test_demucs', help='Output directory')
    parser.add_argument('-m', '--model', default='htdemucs', help='Demucs model name')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Update global variables
    global input_file, output_dir, model_name
    input_file = args.input
    output_dir = args.output
    model_name = args.model
    
    # Run test
    success = test_demucs_separation()
    
    if success:
        print(f"\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ Tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
