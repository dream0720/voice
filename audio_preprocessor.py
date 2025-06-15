#!/usr/bin/env python3
"""
音频预处理降噪器 - 基于信号与系统基础理论
Audio Preprocessing Denoiser - Based on Signals and Systems Theory

包含的经典信号处理方法：
1. 傅里叶变换（FFT）频域分析
2. 带通滤波器设计
3. 谱减法降噪
4. 维纳滤波
5. 时域和频域特征分析
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from pathlib import Path

class AudioPreprocessor:
    """音频预处理降噪器"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        
    def load_audio(self, audio_path):
        """加载音频文件"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            print(f"📁 加载音频: {audio_path}")
            print(f"   采样率: {sr} Hz")
            print(f"   时长: {len(audio)/sr:.2f} 秒")
            print(f"   样本数: {len(audio)}")
            return audio
        except Exception as e:
            print(f"❌ 音频加载失败: {e}")
            return None
    
    def analyze_frequency_spectrum(self, audio, title="频谱分析"):
        """使用傅里叶变换分析频谱"""
        print(f"\n🔍 {title}")
        print("-" * 40)
        
        # 1. 快速傅里叶变换 (FFT)
        N = len(audio)
        audio_fft = fft(audio)
        frequencies = fftfreq(N, 1/self.sample_rate)
        
        # 2. 计算幅度谱和相位谱
        magnitude_spectrum = np.abs(audio_fft)
        phase_spectrum = np.angle(audio_fft)
        power_spectrum = magnitude_spectrum ** 2
        
        # 3. 只取正频率部分
        positive_freq_idx = frequencies >= 0
        freq_positive = frequencies[positive_freq_idx]
        magnitude_positive = magnitude_spectrum[positive_freq_idx]
        power_positive = power_spectrum[positive_freq_idx]
        
        # 4. 分析频谱特征
        dominant_freq_idx = np.argmax(magnitude_positive[1:]) + 1  # 排除直流分量
        dominant_frequency = freq_positive[dominant_freq_idx]
        
        # 计算频谱质心
        spectral_centroid = np.sum(freq_positive * magnitude_positive) / np.sum(magnitude_positive)
        
        # 计算带宽
        spectral_bandwidth = np.sqrt(np.sum(((freq_positive - spectral_centroid) ** 2) * magnitude_positive) / np.sum(magnitude_positive))
        
        print(f"   主导频率: {dominant_frequency:.1f} Hz")
        print(f"   频谱质心: {spectral_centroid:.1f} Hz") 
        print(f"   频谱带宽: {spectral_bandwidth:.1f} Hz")
        
        return {
            'frequencies': freq_positive,
            'magnitude': magnitude_positive,
            'power': power_positive,
            'phase': phase_spectrum,
            'dominant_freq': dominant_frequency,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth
        }
      def design_bandpass_filter(self, low_freq=80, high_freq=8000, filter_order=4):
        """设计带通滤波器"""
        print(f"\n🔧 设计带通滤波器")
        print("-" * 40)
        
        # 归一化截止频率（确保在0-1范围内）
        low_normalized = low_freq / self.nyquist
        high_normalized = min(high_freq / self.nyquist, 0.99)  # 确保小于1
        
        print(f"   通带范围: {low_freq} - {high_freq} Hz")
        print(f"   归一化频率: {low_normalized:.3f} - {high_normalized:.3f}")
        print(f"   滤波器阶数: {filter_order}")
        
        # 使用巴特沃斯滤波器（平坦通带特性）
        try:
            if high_normalized >= 0.99:  # 如果高频接近奈奎斯特频率，使用高通滤波器
                b, a = signal.butter(filter_order, low_normalized, btype='high')
                print("   ✅ 巴特沃斯高通滤波器设计完成（高频接近奈奎斯特频率）")
            else:
                b, a = signal.butter(filter_order, [low_normalized, high_normalized], btype='band')
                print("   ✅ 巴特沃斯带通滤波器设计完成")
            
            # 计算频率响应
            w, h = signal.freqz(b, a, worN=8000)
            frequencies = w * self.sample_rate / (2 * np.pi)
            
            return {
                'coefficients': (b, a),
                'frequencies': frequencies,
                'response': h,
                'type': 'butterworth'
            }
            
        except Exception as e:
            print(f"   ❌ 滤波器设计失败: {e}")
            return None
    
    def apply_bandpass_filter(self, audio, filter_config):
        """应用带通滤波器"""
        try:
            print("\n🌊 应用带通滤波器...")
            
            b, a = filter_config['coefficients']
            
            # 使用零相位滤波（前向和后向滤波）
            filtered_audio = signal.filtfilt(b, a, audio)
            
            print("   ✅ 带通滤波完成")
            
            # 分析滤波效果
            original_energy = np.sum(audio ** 2)
            filtered_energy = np.sum(filtered_audio ** 2)
            energy_ratio = filtered_energy / original_energy
            
            print(f"   能量保持比: {energy_ratio:.3f}")
            
            return filtered_audio
            
        except Exception as e:
            print(f"   ❌ 滤波器应用失败: {e}")
            return audio
    
    def spectral_subtraction_denoise(self, audio, noise_factor=2.0, noise_floor=0.1):
        """谱减法降噪"""
        print(f"\n🔇 谱减法降噪")
        print("-" * 40)
        
        try:
            # 1. 短时傅里叶变换
            frame_length = 1024
            hop_length = 256
            
            # 手动实现STFT（展示傅里叶变换原理）
            n_frames = 1 + (len(audio) - frame_length) // hop_length
            stft_matrix = np.zeros((frame_length // 2 + 1, n_frames), dtype=complex)
            
            print(f"   帧长: {frame_length} 样本")
            print(f"   跳跃长度: {hop_length} 样本") 
            print(f"   总帧数: {n_frames}")
            
            # 汉宁窗函数
            window = np.hanning(frame_length)
            
            for i in range(n_frames):
                start = i * hop_length
                end = start + frame_length
                
                if end <= len(audio):
                    frame = audio[start:end] * window
                    # FFT变换到频域
                    frame_fft = fft(frame)
                    stft_matrix[:, i] = frame_fft[:frame_length // 2 + 1]
            
            # 2. 估计噪声谱（使用前几帧作为噪声参考）
            noise_frames = min(10, n_frames // 4)  # 前25%的帧或前10帧
            noise_spectrum = np.mean(np.abs(stft_matrix[:, :noise_frames]), axis=1)
            
            print(f"   噪声估计帧数: {noise_frames}")
            
            # 3. 谱减法处理
            magnitude = np.abs(stft_matrix)
            phase = np.angle(stft_matrix)
            
            # 对每一帧应用谱减法
            enhanced_magnitude = np.zeros_like(magnitude)
            
            for i in range(n_frames):
                frame_magnitude = magnitude[:, i]
                
                # 谱减法公式: |S'(ω)| = |Y(ω)| - α*|N(ω)|
                enhanced_frame = frame_magnitude - noise_factor * noise_spectrum
                
                # 防止过度抑制，设置噪声底限
                noise_floor_level = noise_floor * frame_magnitude
                enhanced_frame = np.maximum(enhanced_frame, noise_floor_level)
                
                enhanced_magnitude[:, i] = enhanced_frame
            
            # 4. 重构时域信号
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            
            # 逆STFT
            enhanced_audio = np.zeros(len(audio))
            
            for i in range(n_frames):
                start = i * hop_length
                end = start + frame_length
                
                if end <= len(audio):
                    # 逆FFT回到时域
                    frame_spectrum = np.concatenate([
                        enhanced_stft[:, i],
                        np.conj(enhanced_stft[-2:0:-1, i])
                    ])
                    frame_time = ifft(frame_spectrum).real
                    
                    # 重叠相加
                    enhanced_audio[start:end] += frame_time * window
            
            print("   ✅ 谱减法降噪完成")
            
            # 计算信噪比改善
            noise_power = np.mean(audio[:self.sample_rate] ** 2)  # 前1秒作为噪声
            signal_power = np.mean(enhanced_audio ** 2)
            snr_improvement = 10 * np.log10(signal_power / noise_power)
            
            print(f"   估计SNR改善: {snr_improvement:.1f} dB")
            
            return enhanced_audio
            
        except Exception as e:
            print(f"   ❌ 谱减法降噪失败: {e}")
            return audio
    
    def wiener_filter_denoise(self, audio, noise_factor=0.1):
        """维纳滤波降噪"""
        print(f"\n🧠 维纳滤波降噪")
        print("-" * 40)
        
        try:
            # 1. FFT到频域
            audio_fft = fft(audio)
            power_spectrum = np.abs(audio_fft) ** 2
            
            # 2. 估计噪声功率谱（简化方法）
            noise_power = noise_factor * np.mean(power_spectrum)
            
            # 3. 维纳滤波器设计
            # H(ω) = P_s(ω) / (P_s(ω) + P_n(ω))
            # 其中 P_s 是信号功率谱，P_n 是噪声功率谱
            wiener_filter = power_spectrum / (power_spectrum + noise_power)
            
            # 4. 应用滤波器
            filtered_fft = audio_fft * wiener_filter
            filtered_audio = ifft(filtered_fft).real
            
            print(f"   噪声功率估计: {noise_power:.2e}")
            print(f"   滤波器平均增益: {np.mean(wiener_filter):.3f}")
            print("   ✅ 维纳滤波完成")
            
            return filtered_audio
            
        except Exception as e:
            print(f"   ❌ 维纳滤波失败: {e}")
            return audio
    
    def normalize_audio(self, audio, target_level=0.95):
        """音频归一化"""
        print(f"\n📏 音频归一化")
        print("-" * 40)
        
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0:
            normalized_audio = audio / max_amplitude * target_level
            print(f"   原始最大幅度: {max_amplitude:.3f}")
            print(f"   目标幅度: {target_level}")
            print("   ✅ 归一化完成")
            return normalized_audio
        else:
            print("   ⚠️ 音频信号为零，跳过归一化")
            return audio
    
    def generate_analysis_plots(self, original_audio, processed_audio, analysis_results, output_dir):
        """生成分析图表"""
        try:
            print(f"\n📊 生成分析图表...")
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 1. 时域波形对比
            time_axis = np.arange(len(original_audio)) / self.sample_rate
            
            axes[0, 0].plot(time_axis, original_audio, 'b-', alpha=0.7, label='原始音频')
            axes[0, 0].plot(time_axis, processed_audio, 'r-', alpha=0.7, label='处理后音频')
            axes[0, 0].set_xlabel('时间 (秒)')
            axes[0, 0].set_ylabel('幅度')
            axes[0, 0].set_title('时域波形对比')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 原始音频频谱
            original_spectrum = self.analyze_frequency_spectrum(original_audio, "原始音频频谱")
            axes[0, 1].semilogx(original_spectrum['frequencies'], 20*np.log10(original_spectrum['magnitude'] + 1e-10))
            axes[0, 1].set_xlabel('频率 (Hz)')
            axes[0, 1].set_ylabel('幅度 (dB)')
            axes[0, 1].set_title('原始音频频谱')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 处理后音频频谱
            processed_spectrum = self.analyze_frequency_spectrum(processed_audio, "处理后音频频谱")
            axes[0, 2].semilogx(processed_spectrum['frequencies'], 20*np.log10(processed_spectrum['magnitude'] + 1e-10))
            axes[0, 2].set_xlabel('频率 (Hz)')
            axes[0, 2].set_ylabel('幅度 (dB)')
            axes[0, 2].set_title('处理后音频频谱')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. 滤波器频率响应
            if 'filter_config' in analysis_results:
                filter_config = analysis_results['filter_config']
                axes[1, 0].semilogx(filter_config['frequencies'], 20*np.log10(np.abs(filter_config['response'])))
                axes[1, 0].set_xlabel('频率 (Hz)')
                axes[1, 0].set_ylabel('增益 (dB)')
                axes[1, 0].set_title('带通滤波器响应')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 5. 频谱对比
            axes[1, 1].semilogx(original_spectrum['frequencies'], 20*np.log10(original_spectrum['magnitude'] + 1e-10), 
                               'b-', alpha=0.7, label='原始')
            axes[1, 1].semilogx(processed_spectrum['frequencies'], 20*np.log10(processed_spectrum['magnitude'] + 1e-10), 
                               'r-', alpha=0.7, label='处理后')
            axes[1, 1].set_xlabel('频率 (Hz)')
            axes[1, 1].set_ylabel('幅度 (dB)')
            axes[1, 1].set_title('频谱对比')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. 功率谱密度对比
            axes[1, 2].semilogx(original_spectrum['frequencies'], 10*np.log10(original_spectrum['power'] + 1e-10), 
                               'b-', alpha=0.7, label='原始')
            axes[1, 2].semilogx(processed_spectrum['frequencies'], 10*np.log10(processed_spectrum['power'] + 1e-10), 
                               'r-', alpha=0.7, label='处理后')
            axes[1, 2].set_xlabel('频率 (Hz)')
            axes[1, 2].set_ylabel('功率 (dB)')
            axes[1, 2].set_title('功率谱密度对比')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            plot_path = Path(output_dir) / "audio_preprocessing_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   📈 分析图表保存到: {plot_path}")
            
        except Exception as e:
            print(f"   ❌ 图表生成失败: {e}")
    
    def preprocess_audio(self, input_path, output_path="output/preprocessed_audio.wav", 
                        low_freq=80, high_freq=8000, enable_spectral_subtraction=True, 
                        enable_wiener_filter=False):
        """完整的音频预处理流程"""
        try:
            print("🎵 音频预处理降噪器")
            print("=" * 60)
            print("基于信号与系统理论的音频处理")
            print("包含: FFT分析、带通滤波、谱减法降噪等")
            print("=" * 60)
            
            # 创建输出目录
            output_dir = Path(output_path).parent
            output_dir.mkdir(exist_ok=True)
            
            # 1. 加载音频
            original_audio = self.load_audio(input_path)
            if original_audio is None:
                return False
            
            # 2. 原始音频频谱分析
            original_analysis = self.analyze_frequency_spectrum(original_audio, "原始音频分析")
            
            processed_audio = original_audio.copy()
            analysis_results = {}
            
            # 3. 带通滤波器设计和应用
            filter_config = self.design_bandpass_filter(low_freq, high_freq)
            if filter_config:
                processed_audio = self.apply_bandpass_filter(processed_audio, filter_config)
                analysis_results['filter_config'] = filter_config
            
            # 4. 谱减法降噪（可选）
            if enable_spectral_subtraction:
                processed_audio = self.spectral_subtraction_denoise(processed_audio)
            
            # 5. 维纳滤波降噪（可选）
            if enable_wiener_filter:
                processed_audio = self.wiener_filter_denoise(processed_audio)
            
            # 6. 音频归一化
            processed_audio = self.normalize_audio(processed_audio)
            
            # 7. 保存处理后的音频
            print(f"\n💾 保存处理后音频到: {output_path}")
            sf.write(output_path, processed_audio, self.sample_rate)
            
            # 8. 生成分析报告和图表
            self.generate_analysis_plots(original_audio, processed_audio, analysis_results, output_dir)
            
            # 9. 保存处理报告
            report_path = output_dir / "preprocessing_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("音频预处理报告\\n")
                f.write("=" * 30 + "\\n\\n")
                f.write(f"输入文件: {input_path}\\n")
                f.write(f"输出文件: {output_path}\\n")
                f.write(f"采样率: {self.sample_rate} Hz\\n")
                f.write(f"带通滤波范围: {low_freq} - {high_freq} Hz\\n")
                f.write(f"谱减法降噪: {'是' if enable_spectral_subtraction else '否'}\\n")
                f.write(f"维纳滤波: {'是' if enable_wiener_filter else '否'}\\n")
                f.write(f"\\n频谱分析结果:\\n")
                f.write(f"原始音频主导频率: {original_analysis['dominant_freq']:.1f} Hz\\n")
                f.write(f"原始音频频谱质心: {original_analysis['spectral_centroid']:.1f} Hz\\n")
                f.write(f"原始音频频谱带宽: {original_analysis['spectral_bandwidth']:.1f} Hz\\n")
            
            print(f"📄 处理报告保存到: {report_path}")
            print("\\n✅ 音频预处理完成!")
            
            return True
            
        except Exception as e:
            print(f"❌ 音频预处理失败: {e}")
            return False

def main():
    """主函数"""
    # 配置参数
    input_audio = "input/answer.wav"  # 输入音频文件
    output_audio = "output/preprocessed_answer.wav"  # 输出音频文件
    
    # 人声频带范围（根据信号与系统理论）
    human_voice_low = 80    # Hz - 基频下限
    human_voice_high = 8000 # Hz - 谐波上限
    
    # 创建预处理器
    preprocessor = AudioPreprocessor(sample_rate=16000)
    
    # 执行预处理
    success = preprocessor.preprocess_audio(
        input_path=input_audio,
        output_path=output_audio,
        low_freq=human_voice_low,
        high_freq=human_voice_high,
        enable_spectral_subtraction=True,  # 启用谱减法
        enable_wiener_filter=False         # 可选择启用维纳滤波
    )
    
    if success:
        print(f"\\n🎉 预处理成功!")
        print(f"处理后音频: {output_audio}")
        print("可以继续进行后续的人声分离和匹配处理")
    else:
        print("\\n❌ 预处理失败")

if __name__ == "__main__":
    main()
