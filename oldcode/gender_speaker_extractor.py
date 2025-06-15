#!/usr/bin/env python3
"""
重新编写的主人声提取算法
基于参考音频进行男女声分离
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.spatial.distance import cosine
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class GenderBasedSpeakerExtractor:
    """
    基于性别特征的说话人提取器
    """
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.frame_length = 2048
        self.hop_length = 512
        self.n_mfcc = 13
        
        # 分析参数
        self.window_duration = 2.0  # 分析窗口长度(秒)
        self.hop_duration = 1.0     # 窗口移动步长(秒)
        
    def analyze_gender_features(self, audio):
        """
        分析音频的性别特征
        
        Args:
            audio: 音频信号
            
        Returns:
            features: 性别特征字典
        """
        if len(audio) < self.sample_rate * 0.5:  # 音频太短
            return None
        
        try:
            # 基频分析
            f0 = librosa.yin(audio, fmin=50, fmax=400, sr=self.sample_rate)
            f0_clean = f0[f0 > 0]
            
            if len(f0_clean) == 0:
                return None
            
            f0_mean = np.mean(f0_clean)
            f0_median = np.median(f0_clean)
            f0_std = np.std(f0_clean)
            
            # MFCC特征
            mfcc = librosa.feature.mfcc(
                y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc,
                n_fft=self.frame_length, hop_length=self.hop_length
            )
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # 频谱特征
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))
            
            # 能量分布分析
            stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            
            # 定义频率带（根据您的描述调整）
            low_freq_mask = (freqs >= 80) & (freqs <= 150)    # 低频带
            mid_freq_mask = (freqs >= 150) & (freqs <= 250)   # 中频带  
            high_freq_mask = (freqs >= 250) & (freqs <= 400)  # 高频带
            
            low_energy = np.mean(magnitude[low_freq_mask, :]) if np.any(low_freq_mask) else 0
            mid_energy = np.mean(magnitude[mid_freq_mask, :]) if np.any(mid_freq_mask) else 0
            high_energy = np.mean(magnitude[high_freq_mask, :]) if np.any(high_freq_mask) else 0
            
            total_energy = low_energy + mid_energy + high_energy
            if total_energy > 0:
                low_ratio = low_energy / total_energy
                mid_ratio = mid_energy / total_energy
                high_ratio = high_energy / total_energy
            else:
                low_ratio = mid_ratio = high_ratio = 0
            
            return {
                'f0_mean': f0_mean,
                'f0_median': f0_median,
                'f0_std': f0_std,
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std,
                'spectral_centroid': spectral_centroid,
                'spectral_bandwidth': spectral_bandwidth,
                'spectral_rolloff': spectral_rolloff,
                'low_energy_ratio': low_ratio,
                'mid_energy_ratio': mid_ratio,
                'high_energy_ratio': high_ratio,
                'rms': np.sqrt(np.mean(audio**2))
            }
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def calculate_similarity(self, features1, features2):
        """
        计算两个特征向量的相似度
        """
        if features1 is None or features2 is None:
            return 0.0
        
        try:
            # MFCC相似度
            mfcc_sim = 1 - cosine(features1['mfcc_mean'], features2['mfcc_mean'])
            mfcc_sim = max(0, mfcc_sim)
            
            # 基频相似度
            f0_diff = abs(features1['f0_mean'] - features2['f0_mean'])
            f0_sim = 1 / (1 + f0_diff / 20)  # 基频相似度
            
            # 频谱质心相似度
            centroid_diff = abs(features1['spectral_centroid'] - features2['spectral_centroid'])
            centroid_sim = 1 / (1 + centroid_diff / 500)
            
            # 能量分布相似度
            energy_diff = abs(features1['mid_energy_ratio'] - features2['mid_energy_ratio'])
            energy_sim = 1 / (1 + energy_diff * 10)
            
            # 加权组合
            similarity = (
                0.4 * mfcc_sim + 
                0.3 * f0_sim + 
                0.2 * centroid_sim + 
                0.1 * energy_sim
            )
            
            return max(0, min(1, similarity))
            
        except Exception as e:
            print(f"相似度计算失败: {e}")
            return 0.0
    
    def apply_gender_filter(self, audio, target_features, strength=0.8):
        """
        应用基于性别特征的滤波器
        """
        if target_features is None:
            return audio
        
        try:
            # 基于目标特征确定滤波参数
            target_f0 = target_features['f0_mean']
            
            # 设计带通滤波器
            if target_f0 > 200:  # 女声（根据您的描述，这里是较高的女声）
                # 保留女声频率范围，抑制男声
                low_cutoff = 150
                high_cutoff = 350
                suppress_low = 80
                suppress_high = 180
            else:  # 男声（根据您的描述，这里是较高的男声）
                # 保留男声频率范围，抑制女声
                low_cutoff = 100
                high_cutoff = 200
                suppress_low = 200
                suppress_high = 300
            
            # 频域滤波
            stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            
            # 创建滤波器掩码
            filter_mask = np.ones_like(magnitude)
            
            # 抑制不需要的频率范围
            suppress_mask = (freqs >= suppress_low) & (freqs <= suppress_high)
            filter_mask[suppress_mask, :] *= (1 - strength)
            
            # 增强目标频率范围
            enhance_mask = (freqs >= low_cutoff) & (freqs <= high_cutoff)
            filter_mask[enhance_mask, :] *= (1 + strength * 0.3)
            
            # 应用滤波器
            filtered_magnitude = magnitude * filter_mask
            
            # 重构音频
            filtered_stft = filtered_magnitude * np.exp(1j * phase)
            filtered_audio = librosa.istft(filtered_stft, hop_length=self.hop_length)
            
            return filtered_audio
            
        except Exception as e:
            print(f"性别滤波失败: {e}")
            return audio
    
    def segment_audio_by_similarity(self, target_audio, reference_features, threshold=0.3):
        """
        根据相似度对音频进行分段
        """
        print(f"🔄 根据相似度分段音频...")
        
        # 计算分段参数
        window_samples = int(self.window_duration * self.sample_rate)
        hop_samples = int(self.hop_duration * self.sample_rate)
        
        segments = []
        confidences = []
        
        # 滑动窗口分析
        for start_sample in range(0, len(target_audio) - window_samples + 1, hop_samples):
            end_sample = start_sample + window_samples
            window_audio = target_audio[start_sample:end_sample]
            
            # 提取窗口特征
            window_features = self.analyze_gender_features(window_audio)
            
            if window_features is not None:
                # 计算相似度
                similarity = self.calculate_similarity(reference_features, window_features)
                
                # 记录段信息
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'similarity': similarity,
                    'audio': window_audio,
                    'features': window_features
                })
                confidences.append(similarity)
        
        print(f"   分析了 {len(segments)} 个音频段")
        print(f"   平均相似度: {np.mean(confidences):.3f}")
        
        # 筛选高相似度段
        target_segments = [seg for seg in segments if seg['similarity'] >= threshold]
        print(f"   找到 {len(target_segments)} 个目标段 (阈值: {threshold})")
        
        return segments, target_segments
    
    def reconstruct_timeline_audio(self, target_audio, segments, target_segments):
        """
        重构时间轴音频，保持原始时长
        """
        print(f"🔄 重构时间轴音频...")
        
        # 初始化输出音频
        output_audio = np.zeros_like(target_audio)
        
        # 为每个时间段分配权重
        for segment in segments:
            start_idx = segment['start_sample']
            end_idx = min(segment['end_sample'], len(target_audio))
            similarity = segment['similarity']
            
            # 获取原始音频段
            original_segment = target_audio[start_idx:end_idx]
            
            if similarity > 0.5:
                # 高相似度：保留并增强
                processed_segment = self.apply_gender_filter(
                    original_segment, segment['features'], strength=0.8
                )
                weight = 1.0
            elif similarity > 0.3:
                # 中等相似度：保留但衰减
                processed_segment = original_segment * 0.7
                weight = 0.7
            elif similarity > 0.1:
                # 低相似度：大幅衰减
                processed_segment = original_segment * 0.2
                weight = 0.2
            else:
                # 极低相似度：基本静音
                processed_segment = original_segment * 0.05
                weight = 0.05
              # 应用到输出音频
            if end_idx > start_idx:
                # 确保长度匹配
                target_len = end_idx - start_idx
                segment_len = len(processed_segment)
                
                if segment_len >= target_len:
                    # 段比目标长，截取
                    output_audio[start_idx:end_idx] = processed_segment[:target_len] * weight
                else:
                    # 段比目标短，填充
                    output_audio[start_idx:start_idx + segment_len] = processed_segment * weight
        
        # 后处理
        output_audio = self.post_process_audio(output_audio)
        
        print(f"✅ 重构完成，时长: {len(output_audio)/self.sample_rate:.2f}秒")
        
        return output_audio
    
    def post_process_audio(self, audio):
        """音频后处理"""
        if len(audio) == 0:
            return audio
        
        # 去直流分量
        audio = audio - np.mean(audio)
        
        # 轻微的动态范围压缩
        threshold = 0.7
        ratio = 3.0
        abs_audio = np.abs(audio)
        mask = abs_audio > threshold
        
        if np.any(mask):
            compression_factor = threshold + (abs_audio[mask] - threshold) / ratio
            audio[mask] = np.sign(audio[mask]) * compression_factor
        
        # 归一化
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        return audio
    
    def extract_target_speaker(self, vocals_path, reference_path, output_path):
        """
        主要的说话人提取流程
        """
        print("🎯 开始基于性别特征的说话人提取")
        print("=" * 60)
        
        try:
            # 1. 加载音频
            print("📁 加载音频文件...")
            target_audio, _ = librosa.load(vocals_path, sr=self.sample_rate)
            reference_audio, _ = librosa.load(reference_path, sr=self.sample_rate)
            
            print(f"   目标音频: {len(target_audio)/self.sample_rate:.2f}秒")
            print(f"   参考音频: {len(reference_audio)/self.sample_rate:.2f}秒")
            
            # 2. 提取参考音频特征
            print("🔍 分析参考音频特征...")
            reference_features = self.analyze_gender_features(reference_audio)
            
            if reference_features is None:
                print("❌ 无法提取参考音频特征")
                return False
            
            print(f"   参考基频: {reference_features['f0_mean']:.1f} Hz")
            print(f"   频谱质心: {reference_features['spectral_centroid']:.1f} Hz")
            
            # 判断性别类型
            if reference_features['f0_mean'] > 180:
                gender_type = "女声"
                print(f"   识别为: {gender_type} (基频较高)")
            else:
                gender_type = "男声"
                print(f"   识别为: {gender_type} (基频较低)")
            
            # 3. 音频分段分析
            all_segments, target_segments = self.segment_audio_by_similarity(
                target_audio, reference_features, threshold=0.3
            )
            
            if not target_segments:
                print("❌ 未找到匹配的音频段")
                return False
            
            # 4. 重构音频
            result_audio = self.reconstruct_timeline_audio(
                target_audio, all_segments, target_segments
            )
            
            # 5. 保存结果
            sf.write(output_path, result_audio, self.sample_rate)
            print(f"💾 结果已保存: {output_path}")
            
            # 6. 生成报告
            print(f"\n📊 提取结果:")
            print(f"   原始时长: {len(target_audio)/self.sample_rate:.2f}秒")
            print(f"   输出时长: {len(result_audio)/self.sample_rate:.2f}秒")
            print(f"   时长保持: {'✅' if abs(len(result_audio)/len(target_audio) - 1) < 0.01 else '⚠️'}")
            print(f"   匹配段数: {len(target_segments)}")
            print(f"   平均相似度: {np.mean([seg['similarity'] for seg in target_segments]):.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 提取失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """测试主函数"""
    extractor = GenderBasedSpeakerExtractor(sample_rate=22050)
    
    # 文件路径
    vocals_path = "temp/demucs_output/htdemucs/answer/vocals.wav"
    reference_path = "reference/refne2.wav"
    output_path = "output/gender_separated_speaker.wav"
    
    # 检查文件存在性
    if not Path(vocals_path).exists():
        print(f"❌ 人声文件不存在: {vocals_path}")
        return
    
    if not Path(reference_path).exists():
        print(f"❌ 参考文件不存在: {reference_path}")
        return
    
    # 执行提取
    success = extractor.extract_target_speaker(vocals_path, reference_path, output_path)
    
    if success:
        print("\n🎉 说话人提取成功!")
    else:
        print("\n❌ 说话人提取失败!")

if __name__ == "__main__":
    main()
