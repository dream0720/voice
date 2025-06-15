import os
import numpy as np
import librosa
import soundfile as sf
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

from .filters import VoiceEnhancementFilter
from .match_speaker import SpeakerMatcher
from .analyze import AudioAnalyzer

class ProcessPipeline:
    """
    音频处理主流程管道
    
    整合完整的处理流程：
    1. Demucs音源分离
    2. 主人声匹配识别  
    3. 带通滤波增强
    4. 后处理优化
    """
    
    def __init__(self, sample_rate=22050):
        """
        初始化处理管道
        
        Args:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        
        # 初始化各个处理模块
        self.voice_filter = VoiceEnhancementFilter(sample_rate)
        self.speaker_matcher = SpeakerMatcher(sample_rate)
        self.analyzer = AudioAnalyzer(sample_rate)
        
        # 创建输出目录
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "output"
        self.temp_dir = self.project_root / "temp"
        
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
    
    def separate_sources_with_demucs(self, input_audio_path):
        """
        使用Demucs进行音源分离
        
        Args:
            input_audio_path: 输入音频文件路径
            
        Returns:
            separated_files: 分离后的文件路径字典
        """
        print("🎵 开始使用Demucs进行音源分离...")
        
        try:
            # 准备输出目录
            demucs_output = self.temp_dir / "demucs_output"
            demucs_output.mkdir(exist_ok=True)
            
            # 构建Demucs命令
            cmd = [
                "python", "-m", "demucs.separate",
                "--out", str(demucs_output),
                "--name", "htdemucs",  # 使用高质量模型
                str(input_audio_path)
            ]
            
            # 执行Demucs分离
            print("执行命令:", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Demucs执行失败: {result.stderr}")
                # 如果Demucs失败，尝试简单的人声分离
                return self.fallback_vocal_separation(input_audio_path)
            
            # 查找分离后的文件
            audio_name = Path(input_audio_path).stem
            separated_dir = demucs_output / "htdemucs" / audio_name
            
            separated_files = {
                'vocals': separated_dir / "vocals.wav",
                'drums': separated_dir / "drums.wav", 
                'bass': separated_dir / "bass.wav",
                'other': separated_dir / "other.wav"
            }
            
            # 验证文件是否存在
            for key, path in separated_files.items():
                if not path.exists():
                    print(f"警告: 分离文件 {key} 不存在: {path}")
            
            print("✅ Demucs音源分离完成")
            return separated_files
            
        except subprocess.TimeoutExpired:
            print("❌ Demucs处理超时")
            return self.fallback_vocal_separation(input_audio_path)
        except Exception as e:
            print(f"❌ Demucs处理失败: {e}")
            return self.fallback_vocal_separation(input_audio_path)
    
    def fallback_vocal_separation(self, input_audio_path):
        """
        备用人声分离方法 (基于频谱减法)
        
        Args:
            input_audio_path: 输入音频路径
            
        Returns:
            separated_files: 分离文件字典
        """
        print("🔄 使用备用人声分离方法...")
        
        try:
            # 加载音频
            y, sr = librosa.load(input_audio_path, sr=self.sample_rate)
            
            # 如果是立体声，分离左右声道
            if len(y.shape) > 1:
                # 中央人声提取 (Mid-Side处理)
                mid = (y[0] + y[1]) / 2  # 中央信号 (人声)
                side = (y[0] - y[1]) / 2  # 边侧信号 (乐器)
                vocals = mid
            else:
                # 单声道，直接使用原音频作为人声
                vocals = y
            
            # 保存分离结果
            vocals_path = self.temp_dir / "vocals_backup.wav"
            sf.write(vocals_path, vocals, sr)
            
            separated_files = {
                'vocals': vocals_path,
                'drums': None,
                'bass': None, 
                'other': None
            }
            
            print("✅ 备用人声分离完成")
            return separated_files
            
        except Exception as e:
            print(f"❌ 备用人声分离也失败: {e}")
            return {'vocals': None, 'drums': None, 'bass': None, 'other': None}
    
    def match_target_speaker(self, reference_audio_path, separated_vocals_path):
        """
        匹配目标说话人
        
        Args:
            reference_audio_path: 参考音频路径
            separated_vocals_path: 分离出的人声路径
            
        Returns:
            matched_audio: 匹配的目标说话人音频
            confidence: 匹配置信度
        """
        print("🎯 开始主人声匹配...")
        
        try:
            # 加载参考音频
            reference_audio, _ = librosa.load(reference_audio_path, sr=self.sample_rate)
            
            # 加载分离的人声
            if separated_vocals_path and os.path.exists(separated_vocals_path):
                vocals_audio, _ = librosa.load(separated_vocals_path, sr=self.sample_rate)
            else:
                print("❌ 分离的人声文件不存在，使用参考音频")
                return reference_audio, 0.5
            
            # 增强说话人分离效果
            matched_audio, confidence = self.speaker_matcher.enhance_speaker_separation(
                vocals_audio, reference_audio, [vocals_audio]
            )
            
            print(f"✅ 主人声匹配完成，置信度: {confidence:.3f}")
            return matched_audio, confidence
            
        except Exception as e:
            print(f"❌ 主人声匹配失败: {e}")
            # 返回参考音频作为备选
            reference_audio, _ = librosa.load(reference_audio_path, sr=self.sample_rate)
            return reference_audio, 0.3
    
    def apply_voice_enhancement(self, audio_signal, low_cutoff=300, high_cutoff=3400):
        """
        应用人声增强处理
        
        Args:
            audio_signal: 输入音频信号
            low_cutoff: 低截止频率
            high_cutoff: 高截止频率
            
        Returns:
            enhanced_audio: 增强后的音频
        """
        print("🔧 开始应用人声增强...")
        
        try:
            # 应用完整的人声增强流程
            enhanced_audio = self.voice_filter.apply_voice_enhancement(
                audio_signal,
                low_cutoff=low_cutoff,
                high_cutoff=high_cutoff,
                pre_emphasis=True,
                post_processing=True
            )
            
            print("✅ 人声增强处理完成")
            return enhanced_audio
            
        except Exception as e:
            print(f"❌ 人声增强失败: {e}")
            return audio_signal
    
    def post_process_audio(self, audio_signal):
        """
        后处理优化
        
        Args:
            audio_signal: 输入音频信号
            
        Returns:
            processed_audio: 后处理后的音频
        """
        print("🎨 开始后处理优化...")
        
        try:
            processed_audio = audio_signal.copy()
            
            # 1. 动态范围压缩
            processed_audio = self.apply_dynamic_range_compression(processed_audio)
            
            # 2. 噪声门限
            processed_audio = self.apply_noise_gate(processed_audio)
            
            # 3. 最终归一化
            max_val = np.max(np.abs(processed_audio))
            if max_val > 0:
                processed_audio = processed_audio / max_val * 0.95
            
            print("✅ 后处理优化完成")
            return processed_audio
            
        except Exception as e:
            print(f"❌ 后处理失败: {e}")
            return audio_signal
    
    def apply_dynamic_range_compression(self, audio_signal, threshold=0.1, ratio=4.0):
        """
        动态范围压缩
        
        Args:
            audio_signal: 输入音频
            threshold: 压缩阈值
            ratio: 压缩比
            
        Returns:
            compressed_audio: 压缩后的音频
        """
        compressed_audio = audio_signal.copy()
        
        # 找到超过阈值的样本
        over_threshold = np.abs(compressed_audio) > threshold
        
        # 应用压缩
        compressed_audio[over_threshold] = (
            np.sign(compressed_audio[over_threshold]) * 
            (threshold + (np.abs(compressed_audio[over_threshold]) - threshold) / ratio)
        )
        
        return compressed_audio
    
    def apply_noise_gate(self, audio_signal, threshold=0.01):
        """
        噪声门限处理
        
        Args:
            audio_signal: 输入音频
            threshold: 噪声门限
            
        Returns:
            gated_audio: 处理后的音频
        """
        gated_audio = audio_signal.copy()
        
        # 计算滑动平均能量
        window_size = int(0.01 * self.sample_rate)  # 10ms窗口
        energy = np.convolve(audio_signal**2, np.ones(window_size)/window_size, mode='same')
        
        # 应用噪声门限
        gate_mask = energy > threshold**2
        gated_audio[~gate_mask] *= 0.1  # 衰减低能量部分
        
        return gated_audio
    
    def process_audio(self, input_audio_path, reference_audio_path, low_freq=300, high_freq=3400):
        """
        完整的音频处理流程
        
        Args:
            input_audio_path: 输入混合音频路径
            reference_audio_path: 参考主人声音频路径
            low_freq: 低截止频率
            high_freq: 高截止频率
            
        Returns:
            output_path: 处理结果文件路径
        """
        print("🚀 开始完整音频处理流程...")
        
        try:
            # 步骤1: 音源分离
            print("\n📍 步骤1: 音源分离")
            separated_files = self.separate_sources_with_demucs(input_audio_path)
            
            # 步骤2: 主人声匹配
            print("\n📍 步骤2: 主人声匹配")
            matched_audio, confidence = self.match_target_speaker(
                reference_audio_path, 
                separated_files.get('vocals')
            )
            
            # 如果匹配置信度太低，发出警告
            if confidence < 0.4:
                print(f"⚠️  警告: 主人声匹配置信度较低 ({confidence:.3f})")
            
            # 步骤3: 带通滤波增强
            print("\n📍 步骤3: 带通滤波增强")
            enhanced_audio = self.apply_voice_enhancement(
                matched_audio, low_freq, high_freq
            )
            
            # 步骤4: 后处理优化
            print("\n📍 步骤4: 后处理优化")
            final_audio = self.post_process_audio(enhanced_audio)
            
            # 步骤5: 保存结果
            print("\n📍 步骤5: 保存结果")
            output_path = self.output_dir / "enhanced_voice_output.wav"
            sf.write(output_path, final_audio, self.sample_rate)
            
            # 生成处理报告
            self.generate_processing_report(input_audio_path, output_path, confidence)
            
            print(f"🎉 处理完成! 结果保存至: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 处理流程失败: {e}")
            raise e
    
    def generate_processing_report(self, input_path, output_path, confidence):
        """
        生成处理报告
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径  
            confidence: 匹配置信度
        """
        try:
            # 加载音频用于分析
            original_audio, _ = librosa.load(input_path, sr=self.sample_rate)
            processed_audio, _ = librosa.load(output_path, sr=self.sample_rate)
            
            # 生成性能报告
            report = self.analyzer.generate_performance_report(original_audio, processed_audio)
            
            # 创建报告文件
            report_path = self.output_dir / "processing_report.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("🎵 人声增强与主人声提取系统 - 处理报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"输入文件: {input_path}\n")
                f.write(f"输出文件: {output_path}\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("📊 处理效果分析:\n")
                f.write(f"• 主人声匹配置信度: {confidence:.3f}\n")
                f.write(f"• 能量变化: {report['energy_change_db']:.2f} dB\n")
                f.write(f"• 频谱质心变化: {report['spectral_centroid_change_percent']:.1f}%\n")
                f.write(f"• 频谱宽度变化: {report['spectral_spread_change_percent']:.1f}%\n")
                f.write(f"• 原始频谱质心: {report['original_centroid_hz']:.0f} Hz\n")
                f.write(f"• 处理后频谱质心: {report['processed_centroid_hz']:.0f} Hz\n")
                
                f.write("\n🎯 信号与系统知识点体现:\n")
                f.write("• LTI系统: 带通滤波器设计与应用\n")
                f.write("• 频域分析: FFT频谱分析\n")
                f.write("• 卷积运算: 滤波器实现\n")
                f.write("• 冲激响应: 系统特性分析\n")
                f.write("• 音频信号处理: MFCC特征提取\n")
            
            print(f"📋 处理报告已保存: {report_path}")
            
        except Exception as e:
            print(f"⚠️  生成处理报告失败: {e}")

    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            print("🗑️  临时文件清理完成")
        except Exception as e:
            print(f"⚠️  清理临时文件失败: {e}")
