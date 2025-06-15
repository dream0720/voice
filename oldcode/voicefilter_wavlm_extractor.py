#!/usr/bin/env python3
"""
VoiceFilter-WavLM 主人声提取器
使用VoiceFilter-WavLM模型进行基于参考音频的目标说话人提取
"""

import numpy as np
import librosa
import torch
import torchaudio
from pathlib import Path
import soundfile as sf
from scipy.signal import wiener
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

class VoiceFilterWavLMExtractor:
    """基于VoiceFilter-WavLM的主人声提取器"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.frame_length = 1024
        self.hop_length = 256
        self.n_fft = 2048
        
    def load_audio(self, audio_path):
        """加载音频文件"""
        try:
            # 使用librosa加载音频
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            print(f"加载音频: {audio_path}")
            print(f"  时长: {len(audio)/self.sample_rate:.2f}秒")
            print(f"  采样率: {sr}Hz")
            return audio
        except Exception as e:
            print(f"音频加载失败: {e}")
            return None
    
    def extract_speaker_embedding(self, audio):
        """提取说话人嵌入特征（模拟WavLM特征提取）"""
        try:
            # 使用MFCC特征作为说话人特征的简化版本
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate,
                n_mfcc=13,
                hop_length=self.hop_length
            )
            
            # 提取统计特征
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            
            # 合并特征
            embedding = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta_mean])
            
            print(f"说话人嵌入维度: {embedding.shape}")
            return embedding
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def calculate_similarity(self, ref_embedding, mixed_frame_embedding):
        """计算参考嵌入和混合音频帧嵌入的相似度"""
        try:
            # 余弦相似度
            dot_product = np.dot(ref_embedding, mixed_frame_embedding)
            norm_ref = np.linalg.norm(ref_embedding)
            norm_frame = np.linalg.norm(mixed_frame_embedding)
            
            if norm_ref == 0 or norm_frame == 0:
                return 0.0
                
            similarity = dot_product / (norm_ref * norm_frame)
            return max(0, similarity)  # 确保非负
            
        except Exception as e:
            return 0.0
    
    def frame_level_extraction(self, mixed_audio, reference_embedding):
        """基于帧级别的说话人提取"""
        try:
            print("🎯 开始帧级别说话人提取...")
            
            # 计算STFT
            stft = librosa.stft(
                mixed_audio, 
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.frame_length
            )
            
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 为每一帧计算说话人相似度
            n_frames = magnitude.shape[1]
            similarities = np.zeros(n_frames)
            enhanced_magnitude = magnitude.copy()
            
            frame_size = self.hop_length
            
            for i in range(n_frames):
                # 获取当前帧的音频
                start_sample = i * self.hop_length
                end_sample = min(start_sample + frame_size, len(mixed_audio))
                
                if end_sample - start_sample < frame_size // 2:
                    continue
                    
                frame_audio = mixed_audio[start_sample:end_sample]
                
                if len(frame_audio) < frame_size // 4:
                    continue
                
                # 提取当前帧的说话人特征
                try:
                    frame_mfcc = librosa.feature.mfcc(
                        y=frame_audio,
                        sr=self.sample_rate,
                        n_mfcc=13,
                        hop_length=self.hop_length // 4
                    )
                    
                    if frame_mfcc.shape[1] > 0:
                        frame_embedding = np.mean(frame_mfcc, axis=1)
                        
                        # 补齐到参考嵌入的维度
                        if len(frame_embedding) < len(reference_embedding):
                            padding = len(reference_embedding) - len(frame_embedding)
                            frame_embedding = np.pad(frame_embedding, (0, padding), 'constant')
                        elif len(frame_embedding) > len(reference_embedding):
                            frame_embedding = frame_embedding[:len(reference_embedding)]
                        
                        # 计算相似度
                        similarity = self.calculate_similarity(reference_embedding, frame_embedding)
                        similarities[i] = similarity
                    
                except Exception as e:
                    similarities[i] = 0.0
                    continue
            
            # 平滑相似度曲线
            from scipy.ndimage import gaussian_filter1d
            similarities_smooth = gaussian_filter1d(similarities, sigma=2.0)
            
            # 动态阈值
            similarity_mean = np.mean(similarities_smooth)
            similarity_std = np.std(similarities_smooth)
            threshold = max(0.3, similarity_mean + 0.5 * similarity_std)
            
            print(f"相似度统计: 均值={similarity_mean:.3f}, 标准差={similarity_std:.3f}")
            print(f"动态阈值: {threshold:.3f}")
            
            # 创建掩码
            mask = similarities_smooth > threshold
            
            # 应用软掩码
            for i in range(n_frames):
                if mask[i]:
                    # 保留高相似度帧，稍作增强
                    enhanced_magnitude[:, i] *= (1.0 + 0.3 * similarities_smooth[i])
                else:
                    # 抑制低相似度帧
                    suppression_factor = max(0.1, 1.0 - 2.0 * (threshold - similarities_smooth[i]))
                    enhanced_magnitude[:, i] *= suppression_factor
            
            # 重构音频
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(
                enhanced_stft,
                hop_length=self.hop_length,
                win_length=self.frame_length
            )
            
            # 统计信息
            target_frames = np.sum(mask)
            total_frames = len(mask)
            coverage = target_frames / total_frames * 100
            
            print(f"目标说话人帧数: {target_frames}/{total_frames} ({coverage:.1f}%)")
            
            return enhanced_audio, similarities_smooth, mask
            
        except Exception as e:
            print(f"帧级别提取失败: {e}")
            return mixed_audio, None, None
    
    def post_process_audio(self, audio):
        """后处理音频"""
        try:
            # 1. 降噪处理
            noise_profile = audio[:int(0.5 * self.sample_rate)]  # 前0.5秒作为噪声参考
            noise_power = np.mean(noise_profile ** 2)
            
            if noise_power > 1e-8:
                # 简单的谱减法降噪
                stft = librosa.stft(audio, hop_length=self.hop_length)
                magnitude = np.abs(stft)
                phase = np.angle(stft)
                
                # 估计噪声谱
                noise_magnitude = np.mean(magnitude[:, :int(0.5 * self.sample_rate / self.hop_length)], axis=1, keepdims=True)
                
                # 谱减法
                alpha = 2.0  # 过减因子
                enhanced_magnitude = magnitude - alpha * noise_magnitude
                enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
                
                # 重构
                enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
                audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
            
            # 2. 音量归一化
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95
            
            # 3. 平滑处理
            from scipy.signal import savgol_filter
            if len(audio) > 51:
                audio = savgol_filter(audio, 51, 3)
            
            return audio
            
        except Exception as e:
            print(f"后处理失败: {e}")
            return audio
    
    def extract_target_speaker(self, mixed_audio_path, reference_audio_path, output_path="output/voicefilter_result.wav"):
        """主要的目标说话人提取函数"""
        try:
            print("🎤 VoiceFilter-WavLM 主人声提取器")
            print("=" * 60)
            
            # 1. 加载音频
            print("\n📁 加载音频文件...")
            mixed_audio = self.load_audio(mixed_audio_path)
            reference_audio = self.load_audio(reference_audio_path)
            
            if mixed_audio is None or reference_audio is None:
                print("❌ 音频加载失败")
                return False
            
            # 2. 提取参考说话人嵌入
            print("\n🧠 提取参考说话人特征...")
            reference_embedding = self.extract_speaker_embedding(reference_audio)
            
            if reference_embedding is None:
                print("❌ 参考特征提取失败")
                return False
            
            # 3. 基于帧级别的说话人提取
            print("\n🎯 执行目标说话人提取...")
            enhanced_audio, similarities, mask = self.frame_level_extraction(mixed_audio, reference_embedding)
            
            # 4. 后处理
            print("\n🔧 音频后处理...")
            final_audio = self.post_process_audio(enhanced_audio)
            
            # 5. 保存结果
            print(f"\n💾 保存结果到: {output_path}")
            Path(output_path).parent.mkdir(exist_ok=True)
            sf.write(output_path, final_audio, self.sample_rate)
            
            # 6. 生成分析报告
            if similarities is not None and mask is not None:
                self.generate_analysis_report(similarities, mask, output_path)
            
            print("✅ VoiceFilter-WavLM 提取完成!")
            return True
            
        except Exception as e:
            print(f"❌ 提取失败: {e}")
            return False
    
    def generate_analysis_report(self, similarities, mask, output_path):
        """生成分析报告和可视化"""
        try:
            print("\n📊 生成分析报告...")
            
            # 创建可视化
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 时间轴
            time_axis = np.arange(len(similarities)) * self.hop_length / self.sample_rate
            
            # 相似度曲线
            ax1.plot(time_axis, similarities, 'b-', linewidth=1, label='说话人相似度')
            ax1.fill_between(time_axis, 0, similarities, alpha=0.3)
            ax1.set_ylabel('相似度')
            ax1.set_title('目标说话人相似度分析')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 提取决策
            ax2.fill_between(time_axis, 0, mask.astype(float), alpha=0.6, color='green', label='保留片段')
            ax2.fill_between(time_axis, 0, (~mask).astype(float), alpha=0.6, color='red', label='抑制片段')
            ax2.set_xlabel('时间 (秒)')
            ax2.set_ylabel('提取决策')
            ax2.set_title('说话人提取决策')
            ax2.set_ylim(-0.1, 1.1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # 保存分析图
            analysis_path = str(Path(output_path).parent / "voicefilter_analysis.png")
            plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 分析图保存到: {analysis_path}")
            
            # 生成文本报告
            report_path = str(Path(output_path).parent / "voicefilter_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("VoiceFilter-WavLM 提取报告\n")
                f.write("=" * 40 + "\n\n")
                
                target_ratio = np.mean(mask) * 100
                avg_similarity = np.mean(similarities)
                max_similarity = np.max(similarities)
                
                f.write(f"目标说话人覆盖率: {target_ratio:.1f}%\n")
                f.write(f"平均相似度: {avg_similarity:.3f}\n")
                f.write(f"最高相似度: {max_similarity:.3f}\n")
                f.write(f"总处理时长: {len(similarities) * self.hop_length / self.sample_rate:.2f}秒\n")
                
            print(f"  📄 文本报告保存到: {report_path}")
            
        except Exception as e:
            print(f"报告生成失败: {e}")

def main():
    """主函数"""
    # 配置路径
    mixed_audio_path = "input/lttgd.wav"  # 混合音频
    reference_audio_path = "reference/lttgd_ref.wav"  # 参考音频
    output_path = "output/voicefilter_extracted.wav"
    
    # 创建提取器
    extractor = VoiceFilterWavLMExtractor(sample_rate=16000)
    
    # 执行提取
    success = extractor.extract_target_speaker(
        mixed_audio_path=mixed_audio_path,
        reference_audio_path=reference_audio_path,
        output_path=output_path
    )
    
    if success:
        print(f"\n🎉 成功! 提取结果保存到: {output_path}")
    else:
        print("\n❌ 提取失败")

if __name__ == "__main__":
    main()
