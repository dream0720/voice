#!/usr/bin/env python3
"""
分析参考音频和双人声音频的频谱特征
用于优化男女声分离算法
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf

def analyze_audio_features(audio_path, label):
    """分析音频的频谱特征"""
    print(f"\n🔍 分析{label}音频特征: {audio_path}")
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=22050)
    print(f"   时长: {len(audio)/sr:.2f}秒")
    print(f"   RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    
    # 基频分析
    try:
        f0 = librosa.yin(audio, fmin=60, fmax=500, sr=sr)
        f0_clean = f0[f0 > 0]
        if len(f0_clean) > 0:
            f0_mean = np.mean(f0_clean)
            f0_std = np.std(f0_clean)
            f0_median = np.median(f0_clean)
            print(f"   基频均值: {f0_mean:.1f} Hz")
            print(f"   基频中位数: {f0_median:.1f} Hz")
            print(f"   基频标准差: {f0_std:.1f} Hz")
        else:
            f0_mean = f0_median = f0_std = 0
            print("   无法检测到基频")
    except:
        f0_mean = f0_median = f0_std = 0
        print("   基频检测失败")
    
    # 频谱特征
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85))
    
    print(f"   频谱质心: {spectral_centroid:.1f} Hz")
    print(f"   频谱带宽: {spectral_bandwidth:.1f} Hz")
    print(f"   频谱滚降点: {spectral_rolloff:.1f} Hz")
    
    # MFCC特征
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    print(f"   MFCC均值: {mfcc_mean[:5]}")  # 显示前5个系数
    
    # 能量分布分析
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    
    # 频率带能量分析
    freqs = librosa.fft_frequencies(sr=sr)
    
    # 定义频率带
    low_freq_mask = (freqs >= 80) & (freqs <= 200)    # 低频带 (男声基频区)
    mid_freq_mask = (freqs >= 200) & (freqs <= 400)   # 中频带 (女声基频区)
    high_freq_mask = (freqs >= 400) & (freqs <= 2000) # 高频带 (共振峰区)
    
    if np.any(low_freq_mask):
        low_energy = np.mean(magnitude[low_freq_mask, :])
    else:
        low_energy = 0
        
    if np.any(mid_freq_mask):
        mid_energy = np.mean(magnitude[mid_freq_mask, :])
    else:
        mid_energy = 0
        
    if np.any(high_freq_mask):
        high_energy = np.mean(magnitude[high_freq_mask, :])
    else:
        high_energy = 0
    
    total_energy = low_energy + mid_energy + high_energy
    if total_energy > 0:
        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy
    else:
        low_ratio = mid_ratio = high_ratio = 0
    
    print(f"   低频能量比例 (80-200Hz): {low_ratio:.3f}")
    print(f"   中频能量比例 (200-400Hz): {mid_ratio:.3f}")
    print(f"   高频能量比例 (400-2000Hz): {high_ratio:.3f}")
    
    return {
        'f0_mean': f0_mean,
        'f0_median': f0_median, 
        'f0_std': f0_std,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_rolloff': spectral_rolloff,
        'mfcc_mean': mfcc_mean,
        'low_energy_ratio': low_ratio,
        'mid_energy_ratio': mid_ratio,
        'high_energy_ratio': high_ratio,
        'rms': np.sqrt(np.mean(audio**2))
    }

def main():
    print("🎯 音频特征分析 - 优化男女声分离")
    print("=" * 60)
    
    # 文件路径
    reference_path = "reference/refne2.wav"
    vocals_path = "temp/demucs_output/htdemucs/answer/vocals.wav"
    
    # 检查文件
    if not Path(reference_path).exists():
        print(f"❌ 参考文件不存在: {reference_path}")
        return
    
    if not Path(vocals_path).exists():
        print(f"❌ 人声文件不存在: {vocals_path}")
        return
    
    # 分析参考音频特征
    ref_features = analyze_audio_features(reference_path, "参考")
    
    # 分析双人声音频特征
    vocals_features = analyze_audio_features(vocals_path, "双人声")
    
    # 特征对比分析
    print(f"\n📊 特征对比分析:")
    print("=" * 40)
    
    # 基频对比
    if ref_features['f0_mean'] > 0:
        print(f"参考音频基频: {ref_features['f0_mean']:.1f} Hz")
        if ref_features['f0_mean'] < 150:
            voice_type = "男声 (低基频)"
            target_range = (80, 180)
            suppress_range = (180, 350)
        else:
            voice_type = "女声 (高基频)"
            target_range = (150, 350)
            suppress_range = (80, 180)
        print(f"识别为: {voice_type}")
        print(f"建议保留频率范围: {target_range[0]}-{target_range[1]} Hz")
        print(f"建议抑制频率范围: {suppress_range[0]}-{suppress_range[1]} Hz")
    
    # 频谱特征对比
    print(f"\n频谱质心对比:")
    print(f"   参考音频: {ref_features['spectral_centroid']:.1f} Hz")
    print(f"   双人声: {vocals_features['spectral_centroid']:.1f} Hz")
    
    # 能量分布对比
    print(f"\n能量分布对比:")
    print(f"   参考音频 - 低频: {ref_features['low_energy_ratio']:.3f}, 中频: {ref_features['mid_energy_ratio']:.3f}, 高频: {ref_features['high_energy_ratio']:.3f}")
    print(f"   双人声 - 低频: {vocals_features['low_energy_ratio']:.3f}, 中频: {vocals_features['mid_energy_ratio']:.3f}, 高频: {vocals_features['high_energy_ratio']:.3f}")
    
    # 生成优化建议
    print(f"\n🔧 算法优化建议:")
    print("=" * 40)
    
    if ref_features['f0_mean'] > 0:
        if ref_features['f0_mean'] < 150:  # 男声
            print("1. 目标是男声，需要抑制女声")
            print(f"2. 使用带通滤波器保留 80-180 Hz 基频区域")
            print(f"3. 抑制 200-400 Hz 女声基频区域")
            print(f"4. 保留低频共振峰，抑制高频共振峰")
        else:  # 女声
            print("1. 目标是女声，需要抑制男声")
            print(f"2. 使用带通滤波器保留 150-350 Hz 基频区域")
            print(f"3. 抑制 80-180 Hz 男声基频区域")
            print(f"4. 保留高频共振峰，抑制低频共振峰")
    
    print("5. 结合谱减法进一步消除非目标声音")
    print("6. 使用自适应滤波根据实时特征调整")

if __name__ == "__main__":
    main()
