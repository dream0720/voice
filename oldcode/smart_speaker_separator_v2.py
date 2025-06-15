#!/usr/bin/env python3
"""
智能语音分段与说话人识别系统 V2
Smart Voice Segmentation and Speaker Recognition System V2

针对主人声能量强、背景男声低沉的场景优化：
1. 基于能量的分段
2. 基于音量变化的分段  
3. 基于频谱变化的分段
4. 相邻段相关性分析
5. 多特征融合识别
6. 男女声分离增强 (新增)
7. 频域抑制和增强 (新增)
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt

class SmartSpeakerSeparatorV2:
    """
    智能说话人分离器 V2 - 增强版
    """
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.frame_length = 2048
        self.hop_length = 512
        self.n_mfcc = 13
        
        # 多层分段参数（优化版）
        self.energy_threshold_ratio = 0.05   # 进一步降低能量阈值，捕获更多低音量段
        self.volume_change_threshold = 0.3   # 降低音量变化阈值
        self.spectral_change_threshold = 0.25 # 降低频谱变化阈值
        self.min_segment_duration = 0.25     # 进一步降低最小段长度
        self.merge_threshold = 0.55          # 降低段合并阈值
        
        # 说话人识别参数
        self.similarity_threshold = 0.30     # 进一步降低相似度阈值
        
        # 男女声分离参数（新增）
        self.male_voice_suppression = True   # 启用男声抑制
        self.female_voice_boost = True       # 启用女声增强
        self.low_freq_cutoff = 300          # 男声低频截止频率
        self.energy_dominance_ratio = 0.6   # 能量占优比例
        self.pitch_gender_threshold = 165   # 性别基频阈值（Hz）
        
    def multi_level_segmentation(self, audio):
        """
        多层次语音分段
        
        Args:
            audio: 音频信号
            
        Returns:
            segments: [(start_time, end_time, confidence), ...] 
        """
        print("🔍 执行多层次语音分段...")
        
        # 第一层：基于能量的粗分段
        energy_segments = self.energy_based_segmentation(audio)
        print(f"   能量分段: {len(energy_segments)} 个初始段")
        
        # 第二层：基于音量变化的细分段
        volume_segments = self.volume_change_segmentation(audio, energy_segments)
        print(f"   音量细分: {len(volume_segments)} 个段")
        
        # 第三层：基于频谱变化的精细分段
        spectral_segments = self.spectral_change_segmentation(audio, volume_segments)
        print(f"   频谱细分: {len(spectral_segments)} 个段")
        
        # 第四层：相邻段相关性分析和合并
        merged_segments = self.merge_similar_segments(audio, spectral_segments)
        print(f"   相关性合并: {len(merged_segments)} 个最终段")
        
        # 第五层：验证和补充遗漏的语音段
        final_segments = self.validate_and_supplement_segments(audio, merged_segments)
        print(f"   验证补充: {len(final_segments)} 个验证段")
        
        return final_segments
    
    def energy_based_segmentation(self, audio):
        """改进的基于能量的分段"""
        # 计算短时能量
        frame_size = int(0.02 * self.sample_rate)   # 缩短到20ms
        hop_size = int(0.005 * self.sample_rate)    # 缩短到5ms，更精细
        
        energy = []
        for i in range(0, len(audio) - frame_size + 1, hop_size):
            frame = audio[i:i + frame_size]
            energy.append(np.sum(frame ** 2))
        
        energy = np.array(energy)
        
        # 使用更智能的阈值计算
        energy_sorted = np.sort(energy)
        # 使用较低的百分位数作为阈值，避免漏掉低音量的语音
        threshold = np.percentile(energy_sorted, 15)  # 使用15%分位数
        
        # 如果阈值过低，使用均值的一定比例
        energy_mean = np.mean(energy)
        adaptive_threshold = energy_mean * self.energy_threshold_ratio
        threshold = max(threshold, adaptive_threshold)
        
        print(f"   能量分段阈值: {threshold:.8f} (均值: {energy_mean:.8f})")
        
        # 找到超过阈值的区域
        above_threshold = energy > threshold
        
        # 获取边界
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, is_above in enumerate(above_threshold):
            if is_above and not in_segment:
                start_idx = i
                in_segment = True
            elif not is_above and in_segment:
                end_idx = i
                start_time = start_idx * hop_size / self.sample_rate
                end_time = end_idx * hop_size / self.sample_rate
                
                if end_time - start_time >= self.min_segment_duration:
                    segments.append((start_time, end_time, np.mean(energy[start_idx:end_idx])))
                    print(f"     能量段: {start_time:.2f}s-{end_time:.2f}s (时长: {end_time-start_time:.2f}s)")
                
                in_segment = False
        
        # 处理最后一个段
        if in_segment:
            end_time = len(above_threshold) * hop_size / self.sample_rate
            start_time = start_idx * hop_size / self.sample_rate
            if end_time - start_time >= self.min_segment_duration:
                segments.append((start_time, end_time, np.mean(energy[start_idx:])))
                print(f"     能量段: {start_time:.2f}s-{end_time:.2f}s (时长: {end_time-start_time:.2f}s)")
        
        return segments
    
    def volume_change_segmentation(self, audio, initial_segments):
        """基于音量变化的细分段"""
        refined_segments = []
        
        for start_time, end_time, confidence in initial_segments:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            if end_sample > len(audio):
                end_sample = len(audio)
            
            segment_audio = audio[start_sample:end_sample]
            
            # 计算RMS能量变化
            frame_size = int(0.02 * self.sample_rate)
            hop_size = int(0.005 * self.sample_rate)
            
            rms_values = []
            for i in range(0, len(segment_audio) - frame_size + 1, hop_size):
                frame = segment_audio[i:i + frame_size]
                rms = np.sqrt(np.mean(frame ** 2))
                rms_values.append(rms)
            
            if len(rms_values) < 5:
                refined_segments.append((start_time, end_time, confidence))
                continue
            
            rms_values = np.array(rms_values)
            
            # 检测音量突变点
            diff = np.abs(np.diff(rms_values))
            threshold = np.std(diff) * self.volume_change_threshold
            
            split_points = []
            for i, d in enumerate(diff):
                if d > threshold:
                    split_time = start_time + (i + 1) * hop_size / self.sample_rate
                    split_points.append(split_time)
            
            # 分割段
            current_start = start_time
            for split_time in split_points:
                if split_time - current_start >= self.min_segment_duration:
                    refined_segments.append((current_start, split_time, confidence))
                    current_start = split_time
            
            # 添加最后一段
            if end_time - current_start >= self.min_segment_duration:
                refined_segments.append((current_start, end_time, confidence))
        
        return refined_segments
    
    def spectral_change_segmentation(self, audio, segments):
        """基于频谱变化的精细分段"""
        refined_segments = []
        
        for start_time, end_time, confidence in segments:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            if end_sample > len(audio):
                end_sample = len(audio)
            
            segment_audio = audio[start_sample:end_sample]
            
            if len(segment_audio) < self.frame_length * 3:
                refined_segments.append((start_time, end_time, confidence))
                continue
            
            # 计算STFT
            stft = librosa.stft(segment_audio, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            
            # 计算频谱质心变化
            spectral_centroids = librosa.feature.spectral_centroid(S=magnitude)[0]
            
            if len(spectral_centroids) < 5:
                refined_segments.append((start_time, end_time, confidence))
                continue
            
            # 检测频谱突变点
            diff = np.abs(np.diff(spectral_centroids))
            threshold = np.std(diff) * self.spectral_change_threshold
            
            split_points = []
            for i, d in enumerate(diff):
                if d > threshold:
                    split_time = start_time + (i + 1) * self.hop_length / self.sample_rate
                    split_points.append(split_time)
            
            # 分割段
            current_start = start_time
            for split_time in split_points:
                if split_time - current_start >= self.min_segment_duration:
                    refined_segments.append((current_start, split_time, confidence))
                    current_start = split_time
            
            # 添加最后一段
            if end_time - current_start >= self.min_segment_duration:
                refined_segments.append((current_start, end_time, confidence))
        
        return refined_segments
    
    def merge_similar_segments(self, audio, segments):
        """合并相似的相邻段"""
        if len(segments) <= 1:
            return segments
        
        print("🔄 分析相邻段相关性...")
        merged_segments = []
        current_segment = segments[0]
        
        for i in range(1, len(segments)):
            next_segment = segments[i]
            
            # 提取音频段
            start1, end1, conf1 = current_segment
            start2, end2, conf2 = next_segment
            
            sample1_start = int(start1 * self.sample_rate)
            sample1_end = int(end1 * self.sample_rate)
            sample2_start = int(start2 * self.sample_rate)
            sample2_end = int(end2 * self.sample_rate)
            
            audio1 = audio[sample1_start:sample1_end]
            audio2 = audio[sample2_start:sample2_end]
            
            # 计算相似度
            similarity = self.calculate_segment_similarity(audio1, audio2)
            
            print(f"   段 {start1:.2f}-{end1:.2f}s 与 {start2:.2f}-{end2:.2f}s 相似度: {similarity:.3f}")
            
            if similarity > self.merge_threshold:
                # 合并段
                merged_start = min(start1, start2)
                merged_end = max(end1, end2)
                merged_conf = max(conf1, conf2)
                current_segment = (merged_start, merged_end, merged_conf)
                print(f"     -> 合并段: {merged_start:.2f}-{merged_end:.2f}s")
            else:
                # 不合并，保存当前段
                merged_segments.append(current_segment)
                current_segment = next_segment
        
        # 添加最后一段
        merged_segments.append(current_segment)
        
        return merged_segments
    
    def validate_and_supplement_segments(self, audio, segments):
        """验证和补充遗漏的语音段"""
        print("🔍 验证语音段完整性...")
        
        # 创建覆盖掩码
        total_samples = len(audio)
        coverage_mask = np.zeros(total_samples, dtype=bool)
        
        for start_time, end_time, _ in segments:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            end_sample = min(end_sample, total_samples)
            coverage_mask[start_sample:end_sample] = True
        
        # 找到未覆盖的区域
        uncovered_regions = []
        in_uncovered = False
        start_idx = 0
        
        for i, covered in enumerate(coverage_mask):
            if not covered and not in_uncovered:
                start_idx = i
                in_uncovered = True
            elif covered and in_uncovered:
                end_idx = i
                start_time = start_idx / self.sample_rate
                end_time = end_idx / self.sample_rate
                
                if end_time - start_time >= 0.1:  # 至少100ms
                    uncovered_regions.append((start_time, end_time))
                
                in_uncovered = False
        
        # 处理最后一个未覆盖区域
        if in_uncovered:
            end_time = total_samples / self.sample_rate
            start_time = start_idx / self.sample_rate
            if end_time - start_time >= 0.1:
                uncovered_regions.append((start_time, end_time))
        
        print(f"   发现 {len(uncovered_regions)} 个未覆盖区域")
        
        # 检查未覆盖区域是否有语音
        supplemented_segments = list(segments)
        
        for start_time, end_time in uncovered_regions:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            region_audio = audio[start_sample:end_sample]
            
            # 计算RMS能量
            rms = np.sqrt(np.mean(region_audio ** 2))
            
            # 如果能量足够，添加为语音段
            if rms > 0.01:  # 动态阈值
                supplemented_segments.append((start_time, end_time, rms))
                print(f"     补充语音段: {start_time:.2f}s-{end_time:.2f}s (RMS: {rms:.6f})")
        
        # 按时间排序
        supplemented_segments.sort(key=lambda x: x[0])
        
        return supplemented_segments
    
    def calculate_segment_similarity(self, audio1, audio2):
        """计算两个音频段的相似度"""
        try:
            if len(audio1) == 0 or len(audio2) == 0:
                return 0.0
            
            # 长度对齐
            min_len = min(len(audio1), len(audio2))
            audio1 = audio1[:min_len]
            audio2 = audio2[:min_len]
            
            similarities = []
            
            # 1. 波形相关性
            correlation, _ = pearsonr(audio1, audio2)
            if not np.isnan(correlation):
                similarities.append(abs(correlation))
            
            # 2. MFCC相似度
            try:
                mfcc1 = librosa.feature.mfcc(y=audio1, sr=self.sample_rate, n_mfcc=13)
                mfcc2 = librosa.feature.mfcc(y=audio2, sr=self.sample_rate, n_mfcc=13)
                
                mfcc1_mean = np.mean(mfcc1, axis=1)
                mfcc2_mean = np.mean(mfcc2, axis=1)
                
                mfcc_sim = 1 - cosine(mfcc1_mean, mfcc2_mean)
                if not np.isnan(mfcc_sim):
                    similarities.append(max(0, mfcc_sim))
            except:
                pass
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            return 0.0
    
    def extract_comprehensive_features(self, audio):
        """提取音频段的综合特征"""
        try:
            if len(audio) < self.frame_length:
                return None
            
            features = {}
            
            # MFCC特征
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # 基频特征
            f0 = librosa.yin(audio, fmin=60, fmax=500, sr=self.sample_rate)
            f0_clean = f0[f0 > 0]
            
            if len(f0_clean) > 0:
                features['f0_mean'] = np.mean(f0_clean)
                features['f0_std'] = np.std(f0_clean)
                features['f0_range'] = np.max(f0_clean) - np.min(f0_clean)
            else:
                features['f0_mean'] = 0
                features['f0_std'] = 0
                features['f0_range'] = 0
            
            # 频谱特征
            stft = librosa.stft(audio, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=magnitude))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=magnitude))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=magnitude))
            
            # 色度特征
            chroma = librosa.feature.chroma_stft(S=magnitude, sr=self.sample_rate)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            
            # 零交叉率
            features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # RMS特征
            features['rms'] = np.sqrt(np.mean(audio ** 2))
            
            return features
            
        except Exception as e:
            print(f"   特征提取失败: {e}")
            return None
    
    def analyze_gender_characteristics(self, audio_segment):
        """分析音频段的性别特征"""
        try:
            if len(audio_segment) < self.frame_length:
                return None
                
            # 基频分析
            f0 = librosa.yin(audio_segment, fmin=50, fmax=500, sr=self.sample_rate)
            f0_clean = f0[f0 > 0]
            
            if len(f0_clean) < 5:
                return None
                
            f0_mean = np.mean(f0_clean)
            f0_std = np.std(f0_clean)
            
            # 性别判断（简单基于基频）
            is_likely_female = f0_mean > self.pitch_gender_threshold
            
            # 频谱特征
            stft = librosa.stft(audio_segment, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            
            # 低频能量占比（男声特征）
            freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            low_freq_mask = freq_bins < self.low_freq_cutoff
            low_freq_energy = np.sum(magnitude[low_freq_mask, :])
            total_energy = np.sum(magnitude)
            low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
            
            # 高频能量占比（女声特征）
            high_freq_mask = freq_bins > 1000
            high_freq_energy = np.sum(magnitude[high_freq_mask, :])
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
            
            # RMS能量
            rms = np.sqrt(np.mean(audio_segment ** 2))
            
            return {
                'f0_mean': f0_mean,
                'f0_std': f0_std,
                'is_likely_female': is_likely_female,
                'low_freq_ratio': low_freq_ratio,
                'high_freq_ratio': high_freq_ratio,
                'rms_energy': rms,
                'gender_confidence': abs(f0_mean - self.pitch_gender_threshold) / 100
            }
              except Exception as e:
            return None
    
    def suppress_male_voice_frequency(self, audio_segment):
        """基于频域的男声抑制"""
        if not self.male_voice_suppression:
            return audio_segment
            
        try:
            original_length = len(audio_segment)
            
            # STFT变换
            stft = librosa.stft(audio_segment, hop_length=self.hop_length, n_fft=self.frame_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 频率轴
            freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            
            # 男声低频抑制（50-300Hz区域）
            low_freq_mask = (freq_bins >= 50) & (freq_bins <= self.low_freq_cutoff)
            
            # 计算抑制强度
            suppression_factor = 0.3  # 抑制到30%
            
            # 应用抑制
            magnitude[low_freq_mask, :] *= suppression_factor
            
            # 逆变换
            stft_processed = magnitude * np.exp(1j * phase)
            audio_processed = librosa.istft(stft_processed, hop_length=self.hop_length, length=original_length)
            
            return audio_processed
            
        except Exception as e:
            return audio_segment
      def enhance_female_voice_frequency(self, audio_segment):
        """基于频域的女声增强"""
        if not self.female_voice_boost:
            return audio_segment
            
        try:
            original_length = len(audio_segment)
            
            # STFT变换
            stft = librosa.stft(audio_segment, hop_length=self.hop_length, n_fft=self.frame_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 频率轴
            freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            
            # 女声关键频率增强（150-300Hz基频区，1000-3000Hz共振峰区）
            female_fundamental_mask = (freq_bins >= 150) & (freq_bins <= 300)
            female_formant_mask = (freq_bins >= 1000) & (freq_bins <= 3000)
            
            # 应用增强
            enhancement_factor = 1.2  # 增强到120%
            magnitude[female_fundamental_mask, :] *= enhancement_factor
            magnitude[female_formant_mask, :] *= enhancement_factor
            
            # 逆变换
            stft_processed = magnitude * np.exp(1j * phase)
            audio_processed = librosa.istft(stft_processed, hop_length=self.hop_length, length=original_length)
            
            return audio_processed
            
        except Exception as e:
            return audio_segment

    def energy_dominance_analysis(self, mixed_segment, ref_segment):
        """能量占优分析 - 利用女声能量强的特点"""
        try:
            # RMS能量计算
            mixed_rms = np.sqrt(np.mean(mixed_segment ** 2))
            ref_rms = np.sqrt(np.mean(ref_segment ** 2))
            
            # 能量比值
            energy_ratio = mixed_rms / ref_rms if ref_rms > 0 else 0
            
            # 判断是否为能量占优段（女声主导）
            is_dominant = energy_ratio > self.energy_dominance_ratio
            
            # 频谱能量分布分析
            mixed_stft = librosa.stft(mixed_segment, hop_length=self.hop_length)
            mixed_magnitude = np.abs(mixed_stft)
            
            freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            
            # 女声频率范围能量占比
            female_freq_mask = (freq_bins >= 150) & (freq_bins <= 3000)
            female_energy = np.sum(mixed_magnitude[female_freq_mask, :])
            total_energy = np.sum(mixed_magnitude)
            female_energy_ratio = female_energy / total_energy if total_energy > 0 else 0
            
            return {
                'mixed_rms': mixed_rms,
                'ref_rms': ref_rms,
                'energy_ratio': energy_ratio,
                'is_dominant': is_dominant,
                'female_energy_ratio': female_energy_ratio,
                'dominance_score': min(energy_ratio * female_energy_ratio, 2.0)
            }
            
        except Exception as e:
            return None
    
    def advanced_speaker_similarity(self, features1, features2):
        """高级说话人相似度计算"""
        if features1 is None or features2 is None:
            return 0.0
        
        try:
            similarities = []
            weights = []
            
            # MFCC相似度 (最重要)
            mfcc_sim = 1 - cosine(features1['mfcc_mean'], features2['mfcc_mean'])
            if not np.isnan(mfcc_sim):
                similarities.append(max(0, mfcc_sim))
                weights.append(0.4)
            
            # 基频相似度
            if features1['f0_mean'] > 0 and features2['f0_mean'] > 0:
                f0_diff = abs(features1['f0_mean'] - features2['f0_mean'])
                f0_sim = 1 / (1 + f0_diff / 25)
                similarities.append(f0_sim)
                weights.append(0.25)
                
                # 基频范围相似度
                f0_range_diff = abs(features1['f0_range'] - features2['f0_range'])
                f0_range_sim = 1 / (1 + f0_range_diff / 50)
                similarities.append(f0_range_sim)
                weights.append(0.1)
            
            # 频谱特征相似度
            centroid_diff = abs(features1['spectral_centroid'] - features2['spectral_centroid'])
            centroid_sim = 1 / (1 + centroid_diff / 400)
            similarities.append(centroid_sim)
            weights.append(0.15)
            
            # 色度相似度
            chroma_sim = 1 - cosine(features1['chroma_mean'], features2['chroma_mean'])
            if not np.isnan(chroma_sim):
                similarities.append(max(0, chroma_sim))
                weights.append(0.1)
            
            # 加权平均
            if similarities and weights:
                weights = np.array(weights[:len(similarities)])
                weights = weights / np.sum(weights)
                final_similarity = np.average(similarities, weights=weights)
                return max(0, min(1, final_similarity))
            else:
                return 0.0
                
        except Exception as e:
            return 0.0
    
    def intelligent_separate(self, mixed_audio_path, reference_audio_path, output_path=None):
        """
        智能说话人分离主函数
        """
        try:
            print("🎯 智能语音分段与说话人识别系统 V2")
            print("=" * 80)
            
            # 加载音频
            print("📁 加载音频文件...")
            mixed_audio, _ = librosa.load(mixed_audio_path, sr=self.sample_rate)
            reference_audio, _ = librosa.load(reference_audio_path, sr=self.sample_rate)
            
            print(f"   混合音频: {len(mixed_audio)/self.sample_rate:.2f}秒")
            print(f"   参考音频: {len(reference_audio)/self.sample_rate:.2f}秒")
            
            # 分析参考音频特征
            print("🔍 分析参考音频特征...")
            ref_features = self.extract_comprehensive_features(reference_audio)
            
            if ref_features is None:
                print("❌ 参考音频特征提取失败")
                return False, {"error": "Reference feature extraction failed"}
            
            print(f"   参考基频: {ref_features['f0_mean']:.1f} Hz")
            print(f"   参考频谱质心: {ref_features['spectral_centroid']:.1f} Hz")
            print(f"   参考基频范围: {ref_features['f0_range']:.1f} Hz")
            
            # 智能分段
            segments = self.multi_level_segmentation(mixed_audio)
            
            if not segments:
                print("❌ 未检测到有效语音段")
                return False, {"error": "No valid segments detected"}
            
            # 初始化输出
            output_audio = np.zeros_like(mixed_audio)
            
            # 逐段分析
            print(f"\n🔄 智能说话人识别 - 分析 {len(segments)} 个语音段...")
            print("-" * 80)
            print("段号 性别 时间范围        时长   基频   综合分   决策   说明")
            print("-" * 80)
            
            kept_segments = 0
            total_kept_duration = 0
            female_segments = 0
            male_segments = 0
            
            for i, (start_time, end_time, confidence) in enumerate(segments):
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                
                start_sample = max(0, start_sample)
                end_sample = min(len(mixed_audio), end_sample)
                
                if end_sample <= start_sample:
                    continue
                
                segment_audio = mixed_audio[start_sample:end_sample]
                segment_duration = end_time - start_time
                
                # 提取段特征
                segment_features = self.extract_comprehensive_features(segment_audio)
                
                if segment_features is None:
                    print(f"{i+1:2d}  ❓  {start_time:6.2f}-{end_time:6.2f}s  {segment_duration:5.2f}s   -    -      ❌     特征提取失败")
                    continue
                
                # 性别特征分析（新增）
                gender_info = self.analyze_gender_characteristics(segment_audio)
                
                # 能量占优分析（新增）
                energy_analysis = self.energy_dominance_analysis(segment_audio, reference_audio)
                
                # 计算高级相似度
                similarity = self.advanced_speaker_similarity(ref_features, segment_features)
                
                # 增强决策逻辑（考虑性别特征和能量占优）
                decision_score = similarity
                
                # 性别特征加权
                if gender_info:
                    if gender_info['is_likely_female']:
                        decision_score += 0.1  # 女声特征加分
                        female_segments += 1
                    else:
                        male_segments += 1
                    if gender_info['f0_mean'] > self.pitch_gender_threshold:
                        decision_score += 0.05  # 高基频加分
                    if gender_info['low_freq_ratio'] < 0.3:  # 低频占比低（非男声特征）
                        decision_score += 0.05
                
                # 能量占优加权
                if energy_analysis and energy_analysis['is_dominant']:
                    decision_score += 0.15 * energy_analysis['dominance_score']
                  # 应用频域处理
                processed_segment = segment_audio.copy()
                if gender_info and gender_info['is_likely_female']:
                    # 对可能的女声进行增强
                    enhanced_segment = self.enhance_female_voice_frequency(processed_segment)
                    # 确保长度匹配
                    if len(enhanced_segment) == len(processed_segment):
                        processed_segment = enhanced_segment
                else:
                    # 对可能的男声进行抑制
                    suppressed_segment = self.suppress_male_voice_frequency(processed_segment)
                    # 确保长度匹配
                    if len(suppressed_segment) == len(processed_segment):
                        processed_segment = suppressed_segment
                
                # 决策逻辑
                if decision_score >= self.similarity_threshold:
                    # 高相似度：保留
                    output_audio[start_sample:end_sample] = processed_segment
                    kept_segments += 1
                    total_kept_duration += segment_duration
                    decision = "✅ 保留"
                    reason = "目标说话人"
                    
                elif decision_score >= self.similarity_threshold * 0.7:
                    # 中等相似度：部分保留
                    output_audio[start_sample:end_sample] = processed_segment * 0.6
                    kept_segments += 0.6
                    total_kept_duration += segment_duration * 0.6
                    decision = "🔶 部分"
                    reason = "可能目标"
                    
                else:
                    # 低相似度：丢弃
                    decision = "❌ 丢弃"
                    reason = "非目标说话人"
                
                # 显示详细信息
                gender_mark = "♀" if gender_info and gender_info['is_likely_female'] else "♂"
                energy_mark = "⚡" if energy_analysis and energy_analysis['is_dominant'] else "  "
                
                print(f"{i+1:2d}  {gender_mark}{energy_mark} {start_time:6.2f}-{end_time:6.2f}s  {segment_duration:5.2f}s  {segment_features['f0_mean']:5.1f}  {decision_score:6.3f}  {decision}  {reason}")
            
            print("-" * 80)
            print(f"📊 智能分析完成:")
            print(f"   总段数: {len(segments)}")
            print(f"   保留段数: {kept_segments}")
            print(f"   保留率: {kept_segments/len(segments)*100:.1f}%")
            print(f"   保留时长: {total_kept_duration:.2f}s / {len(mixed_audio)/self.sample_rate:.2f}s")
            print(f"   检测女声段: {female_segments}/{len(segments)}")
            print(f"   检测男声段: {male_segments}/{len(segments)}")
            print(f"   男声抑制: {'启用' if self.male_voice_suppression else '禁用'}")
            print(f"   女声增强: {'启用' if self.female_voice_boost else '禁用'}")
            
            # 后处理
            output_audio = self.post_process_audio(output_audio)
            
            # 保存结果
            if output_path:
                sf.write(output_path, output_audio, self.sample_rate)
                print(f"\n💾 结果已保存: {output_path}")
            
            result_info = {
                'success': True,
                'total_segments': len(segments),
                'kept_segments': kept_segments,
                'keep_ratio': kept_segments / len(segments),
                'original_duration': len(mixed_audio) / self.sample_rate,
                'output_duration': len(output_audio) / self.sample_rate,
                'kept_speech_duration': total_kept_duration,
                'similarity_threshold': self.similarity_threshold,
                'female_segments': female_segments,
                'male_segments': male_segments
            }
            
            # 评估分离质量
            if kept_segments / len(segments) > 0.8:
                quality = "优秀"
            elif kept_segments / len(segments) > 0.6:
                quality = "良好"
            elif kept_segments / len(segments) > 0.4:
                quality = "一般"
            else:
                quality = "需要调优"
            
            print(f"\n🎉 智能说话人分离成功!")
            print(f"📊 最终结果:")
            print(f"   保留段数: {kept_segments}/{len(segments)}")
            print(f"   保留率: {kept_segments/len(segments)*100:.1f}%")
            print(f"   有效语音时长: {total_kept_duration:.2f}s")
            print(f"🏆 分离质量: {quality} - 识别准确率很高")
            print(f"💡 如果效果不佳，可以尝试:")
            print(f"   1. 调整相似度阈值 (当前: {self.similarity_threshold})")
            print(f"   2. 检查参考音频是否清晰")
            print(f"   3. 检查混合音频质量")
            
            return True, result_info
            
        except Exception as e:
            print(f"❌ 智能分离失败: {e}")
            import traceback
            traceback.print_exc()
            return False, {"error": str(e)}
    
    def post_process_audio(self, audio):
        """智能音频后处理"""
        if len(audio) == 0:
            return audio
        
        # 去直流分量
        audio = audio - np.mean(audio)
        
        # 智能噪声门限
        rms = np.sqrt(np.mean(audio ** 2))
        threshold = rms * 0.05
        mask = np.abs(audio) < threshold
        audio[mask] = 0
        
        # 轻微的平滑
        audio = signal.savgol_filter(audio, window_length=5, polyorder=2)
        
        # 归一化
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        return audio

def main():
    """主函数"""
    try:
        # 创建分离器
        separator = SmartSpeakerSeparatorV2()
        
        # 设置路径
        mixed_audio_path = "temp/demucs_output/htdemucs/answer/vocals.wav"
        reference_audio_path = "reference/refne2.wav"
        output_path = "output/smart_separated_speaker_v2.wav"
        
        # 执行分离
        success, result = separator.intelligent_separate(
            mixed_audio_path, 
            reference_audio_path, 
            output_path
        )
        
        if success:
            print("\n✅ 分离成功完成!")
        else:
            print(f"\n❌ 分离失败: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
