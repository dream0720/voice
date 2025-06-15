#!/usr/bin/env python3
"""
基于语音段分离的说话人提取器
Interview Speaker Extractor

专门用于采访类音频，利用自然静音间隙分段，然后进行说话人识别
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.spatial.distance import cosine
from pathlib import Path

class InterviewSpeakerExtractor:
    """
    采访说话人提取器
    
    核心思路：
    1. 利用静音间隙检测语音段
    2. 对每个语音段提取说话人特征
    3. 与参考音频比较，决定保留或静音
    4. 重构完整音频
    """
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.frame_length = 2048
        self.hop_length = 512
        self.n_mfcc = 13
        
        # 语音活动检测参数
        self.vad_threshold = 0.01    # VAD能量阈值
        self.min_speech_duration = 0.5  # 最小语音段长度（秒）
        self.min_silence_duration = 0.3  # 最小静音间隔（秒）
        
        # 说话人识别参数
        self.similarity_threshold = 0.3  # 相似度阈值
        
    def detect_speech_segments(self, audio):
        """
        检测语音段，基于能量和静音间隔
        
        Args:
            audio: 音频信号
            
        Returns:
            segments: [(start_time, end_time), ...] 语音段列表
        """
        print("🔍 检测语音段...")
        
        # 计算短时能量
        frame_length = int(0.025 * self.sample_rate)  # 25ms
        hop_length = int(0.01 * self.sample_rate)     # 10ms
        
        frames = []
        for i in range(0, len(audio) - frame_length + 1, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sum(frame ** 2)
            frames.append(energy)
        
        frames = np.array(frames)
        
        # 自适应阈值
        mean_energy = np.mean(frames)
        std_energy = np.std(frames)
        threshold = max(self.vad_threshold, mean_energy - 0.5 * std_energy)
        
        print(f"   能量阈值: {threshold:.6f}")
        print(f"   平均能量: {mean_energy:.6f}")
        
        # 语音活动检测
        speech_frames = frames > threshold
        
        # 转换为时间
        frame_times = np.arange(len(speech_frames)) * hop_length / self.sample_rate
        
        # 查找语音段
        segments = []
        in_speech = False
        start_time = 0
        
        for i, is_speech in enumerate(speech_frames):
            current_time = frame_times[i]
            
            if is_speech and not in_speech:
                # 语音开始
                start_time = current_time
                in_speech = True
            elif not is_speech and in_speech:
                # 语音结束
                end_time = current_time
                duration = end_time - start_time
                
                if duration >= self.min_speech_duration:
                    segments.append((start_time, end_time))
                    print(f"   📍 语音段: {start_time:.2f}s - {end_time:.2f}s (时长: {duration:.2f}s)")
                
                in_speech = False
        
        # 处理最后一个段
        if in_speech:
            end_time = len(audio) / self.sample_rate
            duration = end_time - start_time
            if duration >= self.min_speech_duration:
                segments.append((start_time, end_time))
                print(f"   📍 语音段: {start_time:.2f}s - {end_time:.2f}s (时长: {duration:.2f}s)")
        
        print(f"✅ 检测到 {len(segments)} 个语音段")
        return segments
    
    def extract_speaker_features(self, audio_segment):
        """
        提取说话人特征
        
        Args:
            audio_segment: 音频段
            
        Returns:
            features: 特征字典
        """
        if len(audio_segment) == 0:
            return None
        
        try:
            # MFCC特征
            mfcc = librosa.feature.mfcc(
                y=audio_segment, sr=self.sample_rate, n_mfcc=self.n_mfcc,
                n_fft=self.frame_length, hop_length=self.hop_length
            )
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # 基频特征
            try:
                f0 = librosa.yin(audio_segment, fmin=60, fmax=500, sr=self.sample_rate)
                f0_clean = f0[f0 > 0]
                if len(f0_clean) > 10:  # 需要足够的基频点
                    f0_mean = np.mean(f0_clean)
                    f0_std = np.std(f0_clean)
                    f0_median = np.median(f0_clean)
                else:
                    f0_mean = f0_std = f0_median = 0
            except:
                f0_mean = f0_std = f0_median = 0
            
            # 频谱特征
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio_segment, sr=self.sample_rate
            ))
            
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(
                y=audio_segment, sr=self.sample_rate
            ))
            
            # 能量特征
            rms = np.sqrt(np.mean(audio_segment ** 2))
            
            return {
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std,
                'f0_mean': f0_mean,
                'f0_std': f0_std,
                'f0_median': f0_median,
                'spectral_centroid': spectral_centroid,
                'spectral_bandwidth': spectral_bandwidth,
                'rms': rms
            }
            
        except Exception as e:
            print(f"   ⚠️ 特征提取失败: {e}")
            return None
    
    def calculate_speaker_similarity(self, features1, features2):
        """
        计算说话人相似度
        
        Args:
            features1, features2: 特征字典
            
        Returns:
            similarity: 相似度分数 (0-1)
        """
        if features1 is None or features2 is None:
            return 0.0
        
        try:
            similarities = []
            
            # MFCC相似度 (最重要)
            if len(features1['mfcc_mean']) == len(features2['mfcc_mean']):
                mfcc_sim = 1 - cosine(features1['mfcc_mean'], features2['mfcc_mean'])
                similarities.append(('MFCC', mfcc_sim, 0.5))
            
            # 基频相似度
            if features1['f0_mean'] > 0 and features2['f0_mean'] > 0:
                f0_diff = abs(features1['f0_mean'] - features2['f0_mean'])
                f0_sim = 1 / (1 + f0_diff / 30)  # 基频差异容忍度30Hz
                similarities.append(('基频', f0_sim, 0.3))
            
            # 频谱质心相似度
            centroid_diff = abs(features1['spectral_centroid'] - features2['spectral_centroid'])
            centroid_sim = 1 / (1 + centroid_diff / 500)
            similarities.append(('频谱质心', centroid_sim, 0.2))
            
            # 加权平均
            if similarities:
                weighted_sim = sum(sim * weight for _, sim, weight in similarities)
                total_weight = sum(weight for _, sim, weight in similarities)
                final_similarity = weighted_sim / total_weight if total_weight > 0 else 0
            else:
                final_similarity = 0
            
            return max(0, min(1, final_similarity))
            
        except Exception as e:
            print(f"   ⚠️ 相似度计算失败: {e}")
            return 0.0
    
    def extract_target_speaker(self, mixed_audio_path, reference_audio_path, output_path):
        """
        提取目标说话人
        
        Args:
            mixed_audio_path: 混合音频路径
            reference_audio_path: 参考音频路径  
            output_path: 输出路径
            
        Returns:
            success: 是否成功
            result_info: 结果信息
        """
        try:
            print("🎯 基于语音段分离的说话人提取")
            print("=" * 60)
            
            # 加载音频
            print("📁 加载音频文件...")
            mixed_audio, _ = librosa.load(mixed_audio_path, sr=self.sample_rate)
            reference_audio, _ = librosa.load(reference_audio_path, sr=self.sample_rate)
            
            print(f"   混合音频: {len(mixed_audio)/self.sample_rate:.2f}秒")
            print(f"   参考音频: {len(reference_audio)/self.sample_rate:.2f}秒")
            
            # 提取参考音频特征
            print("\n🔍 分析参考音频特征...")
            ref_features = self.extract_speaker_features(reference_audio)
            if ref_features is None:
                print("❌ 无法提取参考音频特征")
                return False, {"error": "Reference feature extraction failed"}
            
            print(f"   参考基频: {ref_features['f0_mean']:.1f} Hz")
            print(f"   参考频谱质心: {ref_features['spectral_centroid']:.1f} Hz")
            
            # 检测语音段
            speech_segments = self.detect_speech_segments(mixed_audio)
            
            if not speech_segments:
                print("❌ 未检测到语音段")
                return False, {"error": "No speech segments detected"}
            
            # 初始化输出音频
            output_audio = np.zeros_like(mixed_audio)
            
            # 逐段分析和决策
            print(f"\n🔄 逐段分析 {len(speech_segments)} 个语音段...")
            print("-" * 60)
            
            kept_segments = 0
            total_kept_duration = 0
            
            for i, (start_time, end_time) in enumerate(speech_segments):
                # 提取音频段
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                
                # 确保索引有效
                start_sample = max(0, start_sample)
                end_sample = min(len(mixed_audio), end_sample)
                
                if end_sample <= start_sample:
                    continue
                
                segment_audio = mixed_audio[start_sample:end_sample]
                segment_duration = end_time - start_time
                
                # 提取段特征
                segment_features = self.extract_speaker_features(segment_audio)
                
                if segment_features is None:
                    print(f"   段 {i+1:2d}: {start_time:6.2f}s-{end_time:6.2f}s | ❌ 特征提取失败 -> 丢弃")
                    continue
                
                # 计算相似度
                similarity = self.calculate_speaker_similarity(ref_features, segment_features)
                
                # 决策
                if similarity >= self.similarity_threshold:
                    # 保留此段
                    output_audio[start_sample:end_sample] = segment_audio
                    kept_segments += 1
                    total_kept_duration += segment_duration
                    decision = "✅ 保留"
                    
                    print(f"   段 {i+1:2d}: {start_time:6.2f}s-{end_time:6.2f}s | "
                          f"相似度: {similarity:.3f} | 基频: {segment_features['f0_mean']:5.1f}Hz | {decision}")
                else:
                    # 丢弃此段（保持静音）
                    decision = "❌ 丢弃"
                    
                    print(f"   段 {i+1:2d}: {start_time:6.2f}s-{end_time:6.2f}s | "
                          f"相似度: {similarity:.3f} | 基频: {segment_features['f0_mean']:5.1f}Hz | {decision}")
            
            print("-" * 60)
            print(f"📊 分析完成:")
            print(f"   总段数: {len(speech_segments)}")
            print(f"   保留段数: {kept_segments}")
            print(f"   保留率: {kept_segments/len(speech_segments)*100:.1f}%")
            print(f"   保留时长: {total_kept_duration:.2f}s / {len(mixed_audio)/self.sample_rate:.2f}s")
            
            # 后处理
            output_audio = self.post_process_audio(output_audio)
            
            # 保存结果
            if output_path:
                sf.write(output_path, output_audio, self.sample_rate)
                print(f"\n💾 结果已保存: {output_path}")
            
            # 结果信息
            result_info = {
                'success': True,
                'total_segments': len(speech_segments),
                'kept_segments': kept_segments,
                'keep_ratio': kept_segments / len(speech_segments),
                'original_duration': len(mixed_audio) / self.sample_rate,
                'output_duration': len(output_audio) / self.sample_rate,
                'kept_speech_duration': total_kept_duration,
                'avg_similarity_threshold': self.similarity_threshold
            }
            
            return True, result_info
            
        except Exception as e:
            print(f"❌ 提取失败: {e}")
            import traceback
            traceback.print_exc()
            return False, {"error": str(e)}
    
    def post_process_audio(self, audio):
        """音频后处理"""
        if len(audio) == 0:
            return audio
        
        # 去直流分量
        audio = audio - np.mean(audio)
        
        # 轻微的噪声门限
        threshold = np.max(np.abs(audio)) * 0.01
        mask = np.abs(audio) < threshold
        audio[mask] = 0
        
        # 归一化
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        return audio

def main():
    """测试主函数"""
    extractor = InterviewSpeakerExtractor(sample_rate=22050)
    
    # 文件路径
    mixed_path = "temp/demucs_output/htdemucs/answer/vocals.wav"
    reference_path = "reference/refne2.wav"
    output_path = "output/gender_separated_speaker.wav"
    
    # 检查文件存在性
    if not Path(mixed_path).exists():
        print(f"❌ 混合音频不存在: {mixed_path}")
        return
    
    if not Path(reference_path).exists():
        print(f"❌ 参考音频不存在: {reference_path}")
        return
    
    # 执行提取
    success, result_info = extractor.extract_target_speaker(
        mixed_path, reference_path, output_path
    )
    
    if success:
        print(f"\n🎉 说话人提取成功!")
        print(f"📊 最终结果:")
        print(f"   保留段数: {result_info['kept_segments']}/{result_info['total_segments']}")
        print(f"   保留率: {result_info['keep_ratio']*100:.1f}%")
        print(f"   有效语音时长: {result_info['kept_speech_duration']:.2f}s")
        
        if result_info['keep_ratio'] > 0.6:
            print("✅ 提取效果: 优秀")
        elif result_info['keep_ratio'] > 0.3:
            print("👍 提取效果: 良好")
        else:
            print("⚠️ 提取效果: 一般，可能需要调整阈值")
    else:
        print(f"\n❌ 说话人提取失败: {result_info.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
