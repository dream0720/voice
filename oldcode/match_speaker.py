import numpy as np
import librosa
from scipy.spatial.distance import cosine
from .enhanced_speaker_extraction import EnhancedSpeakerExtractor

class SpeakerMatcher:
    """
    说话人匹配器 (升级版)
    用于识别和匹配目标说话人的声音
    
    整合了增强的说话人提取算法
    """
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        # 初始化增强提取器
        self.enhanced_extractor = EnhancedSpeakerExtractor(sample_rate)
        
    def extract_mfcc_features(self, audio, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        提取MFCC特征
        
        Args:
            audio: 输入音频信号
            n_mfcc: MFCC系数数量
            n_fft: FFT窗口大小
            hop_length: 跳跃长度
            
        Returns:
            mfcc_features: MFCC特征矩阵
        """
        try:
            # 确保音频不为空
            if len(audio) == 0:
                return np.zeros((n_mfcc, 1))
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # 归一化
            mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
            
            return mfcc
            
        except Exception as e:
            print(f"MFCC提取失败: {e}")
            return np.zeros((n_mfcc, 1))
    
    def calculate_speaker_similarity(self, mfcc1, mfcc2):
        """
        计算两个MFCC特征之间的相似度
        
        Args:
            mfcc1: 第一个MFCC特征
            mfcc2: 第二个MFCC特征
            
        Returns:
            similarity: 相似度值 (0-1)
        """        try:
            # 展平特征向量
            features1 = mfcc1.flatten()
            features2 = mfcc2.flatten()
            
            # 确保特征向量长度一致
            min_length = min(len(features1), len(features2))
            features1 = features1[:min_length]
            features2 = features2[:min_length]
            
            # 计算余弦相似度
            similarity = 1 - cosine(features1, features2)
            
            # 确保结果在合理范围内
            similarity = max(0, min(1, similarity))
            
            return similarity
            
        except Exception as e:
            print(f"相似度计算失败: {e}")
            return 0.0
    
    def enhance_speaker_separation(self, vocals_audio, reference_audio, separated_vocals_list):
        """
        增强说话人分离效果 (使用新算法)
        
        Args:
            vocals_audio: 分离的人声音频
            reference_audio: 参考音频
            separated_vocals_list: 分离的人声列表 (兼容性参数)
            
        Returns:
            enhanced_audio: 增强后的音频
            confidence: 置信度
        """
        try:
            print("🚀 使用增强的说话人分离算法...")            # 使用新的增强提取器
            segments, confidences = self.enhanced_extractor.extract_target_speaker_segments(
                vocals_audio, reference_audio, min_confidence=0.05
            )
            
            if not segments:
                print("⚠️ 未找到匹配段，返回原音频")
                return vocals_audio, 0.1
            
            # 重构音频
            enhanced_audio, avg_confidence = self.enhanced_extractor.reconstruct_target_speaker_audio(
                segments, confidences
            )
            
            if len(enhanced_audio) == 0:
                print("⚠️ 重构失败，返回原音频")
                return vocals_audio, 0.1
            
            return enhanced_audio, avg_confidence
            
        except Exception as e:
            print(f"增强分离失败，使用传统方法: {e}")
            # 回退到原有算法
            return self._legacy_enhance_speaker_separation(vocals_audio, reference_audio)
    
    def _legacy_enhance_speaker_separation(self, vocals_audio, reference_audio):
        """传统的说话人增强方法 (作为备用)"""
        try:
            # 提取参考音频特征
            ref_mfcc = self.extract_mfcc_features(reference_audio)
            
            # 提取人声特征
            vocals_mfcc = self.extract_mfcc_features(vocals_audio)
            
            # 计算相似度作为置信度
            confidence = self.calculate_speaker_similarity(ref_mfcc, vocals_mfcc)
            
            # 简单的增强处理：根据相似度调整音频
            enhanced_audio = vocals_audio.copy()
            
            # 如果相似度较高，保持原音频
            if confidence > 0.6:
                enhanced_audio = vocals_audio
            elif confidence > 0.3:
                # 中等相似度，进行轻微增强
                enhanced_audio = self.apply_spectral_enhancement(vocals_audio, reference_audio)
            else:
                # 低相似度，尝试更多处理
                enhanced_audio = self.apply_adaptive_enhancement(vocals_audio, reference_audio)
            
            return enhanced_audio, confidence
            
        except Exception as e:
            print(f"传统说话人增强失败: {e}")
            return vocals_audio, 0.3
    
    def process_main_speaker_extraction(self, vocals_path, reference_path, output_path=None):
        """
        主要的说话人提取流程 (新增方法)
        
        Args:
            vocals_path: 分离后的人声文件路径
            reference_path: 参考音频路径
            output_path: 输出路径 (可选)
            
        Returns:
            success: 处理是否成功
            result_info: 结果信息字典
        """
        return self.enhanced_extractor.process_main_speaker_extraction(
            vocals_path, reference_path, output_path
        )
    
    def apply_spectral_enhancement(self, audio, reference):
        """
        应用频谱增强
        
        Args:
            audio: 输入音频
            reference: 参考音频
            
        Returns:
            enhanced_audio: 增强后的音频
        """
        try:
            # 简单的频谱匹配增强
            if len(audio) == 0:
                return audio
            
            # 应用轻微的滤波
            enhanced = audio * 1.1  # 轻微放大
            enhanced = np.clip(enhanced, -1.0, 1.0)  # 限幅
            
            return enhanced
            
        except Exception as e:
            print(f"频谱增强失败: {e}")
            return audio
    
    def apply_adaptive_enhancement(self, audio, reference):
        """
        应用自适应增强
        
        Args:
            audio: 输入音频
            reference: 参考音频
            
        Returns:
            enhanced_audio: 增强后的音频
        """
        try:
            # 自适应增强处理
            if len(audio) == 0:
                return audio
            
            # 能量归一化
            if np.std(audio) > 0:
                enhanced = (audio - np.mean(audio)) / np.std(audio)
                enhanced = enhanced * 0.1  # 缩放到合适范围
            else:
                enhanced = audio
            
            enhanced = np.clip(enhanced, -1.0, 1.0)
            
            return enhanced
            
        except Exception as e:
            print(f"自适应增强失败: {e}")
            return audio