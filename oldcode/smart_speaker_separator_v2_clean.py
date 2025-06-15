#!/usr/bin/env python3
"""
智能语音分段与说话人识别系统 V2 - 简化版
针对主人声能量强、背景男声低沉的场景优化
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

class SmartSpeakerSeparatorV2:
    """智能说话人分离器 V2"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.frame_length = 2048
        self.hop_length = 512
        self.n_mfcc = 13
        
        # 优化参数（针对强女声弱男声）
        self.energy_threshold_ratio = 0.04   # 更低的能量阈值
        self.volume_change_threshold = 0.25  # 更敏感的音量变化
        self.spectral_change_threshold = 0.2 # 更敏感的频谱变化
        self.min_segment_duration = 0.2      # 更短的最小段长
        self.merge_threshold = 0.5           # 更容易合并
        self.similarity_threshold = 0.25     # 更低的相似度阈值
        
        # 性别判断参数
        self.pitch_gender_threshold = 160    # 性别基频阈值
        self.male_suppression_factor = 0.1   # 男声抑制强度
        self.female_boost_factor =  2      # 女声增强强度
        
    def multi_level_segmentation(self, audio):
        """多层次语音分段"""
        print("🔍 执行多层次语音分段...")
        
        # 基于能量的分段
        energy_segments = self.energy_based_segmentation(audio)
        print(f"   能量分段: {len(energy_segments)} 个段")
        
        # 基于音量变化的细分
        volume_segments = self.volume_change_segmentation(audio, energy_segments)
        print(f"   音量细分: {len(volume_segments)} 个段")
        
        # 验证和补充
        final_segments = self.validate_and_supplement_segments(audio, volume_segments)
        print(f"   验证补充: {len(final_segments)} 个段")
        
        return final_segments
    
    def energy_based_segmentation(self, audio):
        """基于能量的分段"""
        frame_size = int(0.025 * self.sample_rate)  # 25ms
        hop_size = int(0.01 * self.sample_rate)     # 10ms
        
        energy = []
        for i in range(0, len(audio) - frame_size + 1, hop_size):
            frame = audio[i:i + frame_size]
            energy.append(np.sum(frame ** 2))
        
        energy = np.array(energy)
        
        # 自适应阈值
        energy_mean = np.mean(energy)
        threshold = energy_mean * self.energy_threshold_ratio
        
        # 找到超过阈值的区域
        above_threshold = energy > threshold
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, is_above in enumerate(above_threshold):
            if is_above and not in_segment:
                start_idx = i
                in_segment = True
            elif not is_above and in_segment:
                start_time = start_idx * hop_size / self.sample_rate
                end_time = i * hop_size / self.sample_rate
                
                if end_time - start_time >= self.min_segment_duration:
                    segments.append((start_time, end_time, np.mean(energy[start_idx:i])))
                
                in_segment = False
        
        # 处理最后一个段
        if in_segment:
            start_time = start_idx * hop_size / self.sample_rate
            end_time = len(above_threshold) * hop_size / self.sample_rate
            if end_time - start_time >= self.min_segment_duration:
                segments.append((start_time, end_time, np.mean(energy[start_idx:])))
        
        return segments
    
    def volume_change_segmentation(self, audio, initial_segments):
        """基于音量变化的细分"""
        refined_segments = []
        
        for start_time, end_time, confidence in initial_segments:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            end_sample = min(end_sample, len(audio))
            
            if end_sample <= start_sample:
                continue
                
            segment_audio = audio[start_sample:end_sample]
            
            # 如果段太短，直接添加
            if len(segment_audio) < self.frame_length * 2:
                refined_segments.append((start_time, end_time, confidence))
                continue
            
            # 计算RMS变化
            frame_size = int(0.02 * self.sample_rate)
            hop_size = int(0.01 * self.sample_rate)
            
            rms_values = []
            for i in range(0, len(segment_audio) - frame_size + 1, hop_size):
                frame = segment_audio[i:i + frame_size]
                rms = np.sqrt(np.mean(frame ** 2))
                rms_values.append(rms)
            
            if len(rms_values) < 3:
                refined_segments.append((start_time, end_time, confidence))
                continue
            
            # 检测音量突变
            rms_values = np.array(rms_values)
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
    
    def validate_and_supplement_segments(self, audio, segments):
        """验证和补充遗漏的语音段"""
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
                start_time = start_idx / self.sample_rate
                end_time = i / self.sample_rate
                
                if end_time - start_time >= 0.1:
                    uncovered_regions.append((start_time, end_time))
                
                in_uncovered = False
        
        # 处理最后一个未覆盖区域
        if in_uncovered:
            start_time = start_idx / self.sample_rate
            end_time = total_samples / self.sample_rate
            if end_time - start_time >= 0.1:
                uncovered_regions.append((start_time, end_time))
        
        # 检查未覆盖区域并补充
        supplemented_segments = list(segments)
        
        for start_time, end_time in uncovered_regions:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            region_audio = audio[start_sample:end_sample]
            
            rms = np.sqrt(np.mean(region_audio ** 2))
            if rms > 0.01:
                supplemented_segments.append((start_time, end_time, rms))
        
        # 按时间排序
        supplemented_segments.sort(key=lambda x: x[0])
        return supplemented_segments
    
    def extract_comprehensive_features(self, audio):
        """提取音频特征"""
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
            
            # 其他特征
            features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))
            features['rms'] = np.sqrt(np.mean(audio ** 2))
            
            return features
            
        except Exception as e:
            return None
    
    def analyze_gender_and_energy(self, segment_audio, reference_audio):
        """分析性别特征和能量占优"""
        try:
            # 基频分析
            f0 = librosa.yin(segment_audio, fmin=50, fmax=500, sr=self.sample_rate)
            f0_clean = f0[f0 > 0]
            
            if len(f0_clean) < 3:
                return {'is_female': False, 'energy_boost': 1.0, 'confidence': 0.0}
            
            f0_mean = np.mean(f0_clean)
            is_female = f0_mean > self.pitch_gender_threshold
            
            # 能量分析
            segment_rms = np.sqrt(np.mean(segment_audio ** 2))
            ref_rms = np.sqrt(np.mean(reference_audio ** 2))
            energy_ratio = segment_rms / ref_rms if ref_rms > 0 else 0
            
            # 频谱分析
            stft = librosa.stft(segment_audio, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            
            # 低频占比（男声特征）
            low_freq_mask = freq_bins < 250
            low_freq_energy = np.sum(magnitude[low_freq_mask, :])
            total_energy = np.sum(magnitude)
            low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
            
            # 决策逻辑
            confidence = 0.0
            energy_boost = 1.0
            
            if is_female:
                confidence += 0.3
                energy_boost = self.female_boost_factor
            
            if low_freq_ratio < 0.25:  # 低频占比低
                confidence += 0.2
            
            if energy_ratio > 0.5:  # 能量占优
                confidence += 0.3
            
            # 如果判断为男声，应用抑制
            if not is_female and low_freq_ratio > 0.35:
                energy_boost = self.male_suppression_factor
                confidence = max(0, confidence - 0.4)
            
            return {
                'is_female': is_female,
                'energy_boost': energy_boost,
                'confidence': confidence,
                'f0_mean': f0_mean,
                'low_freq_ratio': low_freq_ratio,
                'energy_ratio': energy_ratio
            }
            
        except Exception as e:
            return {'is_female': False, 'energy_boost': 1.0, 'confidence': 0.0}
    
    def advanced_speaker_similarity(self, features1, features2):
        """高级说话人相似度计算"""
        if features1 is None or features2 is None:
            return 0.0
        
        try:
            similarities = []
            weights = []
            
            # MFCC相似度
            mfcc_sim = 1 - cosine(features1['mfcc_mean'], features2['mfcc_mean'])
            if not np.isnan(mfcc_sim):
                similarities.append(max(0, mfcc_sim))
                weights.append(0.4)
            
            # 基频相似度
            if features1['f0_mean'] > 0 and features2['f0_mean'] > 0:
                f0_diff = abs(features1['f0_mean'] - features2['f0_mean'])
                f0_sim = 1 / (1 + f0_diff / 30)
                similarities.append(f0_sim)
                weights.append(0.3)
            
            # 频谱特征相似度
            centroid_diff = abs(features1['spectral_centroid'] - features2['spectral_centroid'])
            centroid_sim = 1 / (1 + centroid_diff / 500)
            similarities.append(centroid_sim)
            weights.append(0.2)
            
            # 色度相似度
            chroma_sim = 1 - cosine(features1['chroma_mean'], features2['chroma_mean'])
            if not np.isnan(chroma_sim):
                similarities.append(max(0, chroma_sim))
                weights.append(0.1)
            
            # 加权平均
            if similarities and weights:
                weights = np.array(weights[:len(similarities)])
                weights = weights / np.sum(weights)
                return np.average(similarities, weights=weights)
            else:
                return 0.0
                
        except Exception as e:
            return 0.0
    
    def intelligent_separate(self, mixed_audio_path, reference_audio_path, output_path=None):
        """智能说话人分离主函数"""
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
            
            # 智能分段
            segments = self.multi_level_segmentation(mixed_audio)
            
            if not segments:
                print("❌ 未检测到有效语音段")
                return False, {"error": "No valid segments detected"}
            
            # 初始化输出
            output_audio = np.zeros_like(mixed_audio)
            
            # 逐段分析
            print(f"\n🔄 智能说话人识别 - 分析 {len(segments)} 个语音段...")
            print("-" * 85)
            print("段号 性别 时间范围        时长   基频   相似度 能量增益 综合分   决策   说明")
            print("-" * 85)
            
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
                    print(f"{i+1:2d}  ❓  {start_time:6.2f}-{end_time:6.2f}s  {segment_duration:5.2f}s   -    -      -      -      ❌     特征提取失败")
                    continue
                
                # 性别和能量分析
                gender_energy_info = self.analyze_gender_and_energy(segment_audio, reference_audio)
                
                # 计算相似度
                similarity = self.advanced_speaker_similarity(ref_features, segment_features)
                
                # 综合决策分数
                decision_score = similarity + gender_energy_info['confidence']
                
                # 统计性别
                if gender_energy_info['is_female']:
                    female_segments += 1
                    gender_mark = "♀"
                else:
                    male_segments += 1
                    gender_mark = "♂"
                
                # 决策
                if decision_score >= self.similarity_threshold:
                    # 应用能量调整
                    processed_segment = segment_audio * gender_energy_info['energy_boost']
                    output_audio[start_sample:end_sample] = processed_segment
                    kept_segments += 1
                    total_kept_duration += segment_duration
                    decision = "✅ 保留"
                    reason = "目标说话人"
                    
                elif decision_score >= self.similarity_threshold * 0.6:
                    # 部分保留
                    processed_segment = segment_audio * gender_energy_info['energy_boost'] * 0.5
                    output_audio[start_sample:end_sample] = processed_segment
                    kept_segments += 0.5
                    total_kept_duration += segment_duration * 0.5
                    decision = "🔶 部分"
                    reason = "可能目标"
                    
                else:
                    # 丢弃
                    decision = "❌ 丢弃"
                    reason = "非目标说话人"
                
                # 显示结果
                print(f"{i+1:2d}  {gender_mark}  {start_time:6.2f}-{end_time:6.2f}s  {segment_duration:5.2f}s  {segment_features['f0_mean']:5.1f}  {similarity:6.3f}   {gender_energy_info['energy_boost']:4.1f}   {decision_score:6.3f}  {decision}  {reason}")
            
            print("-" * 85)
            print(f"📊 智能分析完成:")
            print(f"   总段数: {len(segments)}")
            print(f"   保留段数: {kept_segments}")
            print(f"   保留率: {kept_segments/len(segments)*100:.1f}%")
            print(f"   保留时长: {total_kept_duration:.2f}s / {len(mixed_audio)/self.sample_rate:.2f}s")
            print(f"   检测女声段: {female_segments}")
            print(f"   检测男声段: {male_segments}")
            
            # 后处理
            output_audio = self.post_process_audio(output_audio)
            
            # 保存结果
            if output_path:
                sf.write(output_path, output_audio, self.sample_rate)
                print(f"\n💾 结果已保存: {output_path}")
            
            # 评估质量
            if kept_segments / len(segments) > 0.8:
                quality = "优秀"
            elif kept_segments / len(segments) > 0.6:
                quality = "良好"
            elif kept_segments / len(segments) > 0.4:
                quality = "一般"
            else:
                quality = "需要调优"
            
            print(f"\n🎉 智能说话人分离V2成功!")
            print(f"🏆 分离质量: {quality}")
            print(f"💡 优化特点:")
            print(f"   - 针对强女声弱男声场景优化")
            print(f"   - 基于性别特征的智能增强/抑制")
            print(f"   - 更敏感的分段算法")
            print(f"   - 综合能量和频谱特征判断")
            
            result_info = {
                'success': True,
                'total_segments': len(segments),
                'kept_segments': kept_segments,
                'keep_ratio': kept_segments / len(segments),
                'female_segments': female_segments,
                'male_segments': male_segments,
                'quality': quality
            }
            
            return True, result_info
            
        except Exception as e:
            print(f"❌ 智能分离失败: {e}")
            import traceback
            traceback.print_exc()
            return False, {"error": str(e)}
    
    def post_process_audio(self, audio):
        """音频后处理"""
        if len(audio) == 0:
            return audio
        
        # 去直流分量
        audio = audio - np.mean(audio)
        
        # 噪声门限
        rms = np.sqrt(np.mean(audio ** 2))
        threshold = rms * 0.03
        mask = np.abs(audio) < threshold
        audio[mask] = 0
        
        # 轻微平滑
        if len(audio) > 10:
            audio = signal.savgol_filter(audio, window_length=min(5, len(audio)//2*2-1), polyorder=2)
        
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
        mixed_audio_path = "temp/demucs_output/htdemucs/lttgd/vocals.wav"
        reference_audio_path = "reference/lttgd_ref.wav"
        output_path = "output/lltgd.wav"
        
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
