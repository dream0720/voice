#!/usr/bin/env python3
"""
增强的主人声提取模块
Enhanced Speaker Extraction Module

基于多特征融合和时序匹配的说话人识别与提取算法
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EnhancedSpeakerExtractor:
    """
    增强的说话人提取器
    
    功能：
    1. 多特征提取（MFCC、频谱特征、音高等）
    2. 滑动窗口时序匹配
    3. 基于相似度的音频分段
    4. 智能音频增强和重构
    """
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.frame_length = 2048
        self.hop_length = 512
        self.n_mfcc = 13
          # 语音活动检测参数
        self.vad_threshold = 0.005  # 降低VAD阈值，更敏感
        self.min_speech_duration = 0.2  # 最小语音段长度(秒) - 降低到0.2秒
        
        # 匹配参数
        self.window_duration = 1.5  # 匹配窗口长度(秒) - 降低到1.5秒
        self.overlap_ratio = 0.8    # 窗口重叠比例 - 增加到80%
        self.similarity_threshold = 0.05  # 相似度阈值 - 进一步降低
        
    def extract_comprehensive_features(self, audio):
        """
        提取综合音频特征
        
        Args:
            audio: 音频信号
            
        Returns:
            features: 特征字典
        """
        if len(audio) == 0:
            return None
            
        try:
            features = {}
            
            # 1. MFCC特征 (核心特征)
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )
            
            # MFCC统计特征
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            features['mfcc_delta'] = np.mean(librosa.feature.delta(mfcc), axis=1)
            features['mfcc_delta2'] = np.mean(librosa.feature.delta(mfcc, order=2), axis=1)
            
            # 2. 频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            features['spectral_centroid'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            features['spectral_rolloff'] = np.mean(spectral_rolloff)
            
            # 3. 过零率
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 4. 能量特征
            rms = librosa.feature.rms(y=audio)[0]
            features['rms'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
              # 5. 基频特征 (音高) - 优化参数
            try:
                f0 = librosa.yin(audio, fmin=60, fmax=500, sr=self.sample_rate, frame_length=self.frame_length)
                f0_clean = f0[f0 > 0]  # 移除无声段
                if len(f0_clean) > 0:
                    features['f0_mean'] = np.mean(f0_clean)
                    features['f0_std'] = np.std(f0_clean)
                    features['f0_range'] = np.ptp(f0_clean)
                    features['f0_median'] = np.median(f0_clean)  # 新增中位数
                else:
                    features['f0_mean'] = 0
                    features['f0_std'] = 0
                    features['f0_range'] = 0
                    features['f0_median'] = 0
            except:
                features['f0_mean'] = 0
                features['f0_std'] = 0
                features['f0_range'] = 0
                features['f0_median'] = 0
            
            # 6. 新增：声谱对比度特征
            try:
                spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
                features['spectral_contrast'] = np.mean(spectral_contrast, axis=1)
            except:
                features['spectral_contrast'] = np.zeros(7)  # 默认7个子带
            
            # 7. 新增：色度特征
            try:
                chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
                features['chroma'] = np.mean(chroma, axis=1)
            except:
                features['chroma'] = np.zeros(12)  # 默认12个色度
            
            return features
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def detect_voice_activity(self, audio):
        """
        语音活动检测 (VAD)
        
        Args:
            audio: 音频信号
            
        Returns:
            voice_segments: 语音段列表 [(start_time, end_time), ...]
        """
        # 计算短时能量
        frame_length = int(0.025 * self.sample_rate)  # 25ms帧
        hop_length = int(0.01 * self.sample_rate)     # 10ms跳跃
        
        # 计算RMS能量
        rms = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # 动态阈值 (基于能量分布)
        energy_threshold = np.percentile(rms, 30)  # 使用30%分位数作为阈值
        energy_threshold = max(energy_threshold, self.vad_threshold)
        
        # 检测语音段
        voice_frames = rms > energy_threshold
        
        # 转换为时间段
        frame_times = librosa.frames_to_time(
            np.arange(len(voice_frames)), 
            sr=self.sample_rate, 
            hop_length=hop_length
        )
        
        # 查找连续的语音段
        voice_segments = []
        start_time = None
        
        for i, is_voice in enumerate(voice_frames):
            if is_voice and start_time is None:
                start_time = frame_times[i]
            elif not is_voice and start_time is not None:
                end_time = frame_times[i]
                if end_time - start_time >= self.min_speech_duration:
                    voice_segments.append((start_time, end_time))
                start_time = None
        
        # 处理最后一段
        if start_time is not None:
            end_time = frame_times[-1]
            if end_time - start_time >= self.min_speech_duration:
                voice_segments.append((start_time, end_time))
        
        return voice_segments
    
    def calculate_segment_similarity(self, features1, features2):
        """
        计算两个特征向量的相似度
        
        Args:
            features1, features2: 特征字典
              Returns:
            similarity: 综合相似度分数 (0-1)
        """
        if features1 is None or features2 is None:
            return 0.0
        
        similarities = []
        weights = []
        
        # 1. MFCC相似度 (最重要的特征)
        try:
            mfcc_sim = 1 - cosine(features1['mfcc_mean'], features2['mfcc_mean'])
            similarities.append(max(0, mfcc_sim))
            weights.append(0.3)  # 降低权重，为其他特征留空间
        except:
            pass
        
        # 2. MFCC差分特征相似度
        try:
            delta_sim = 1 - cosine(features1['mfcc_delta'], features2['mfcc_delta'])
            similarities.append(max(0, delta_sim))
            weights.append(0.15)
        except:
            pass
        
        # 3. 基频相似度 (说话人身份的重要指标)
        try:
            f0_diff = abs(features1['f0_mean'] - features2['f0_mean'])
            f0_sim = 1 / (1 + f0_diff / 30)  # 更敏感的归一化
            similarities.append(f0_sim)
            weights.append(0.2)
        except:
            pass
        
        # 4. 频谱质心相似度
        try:
            centroid_diff = abs(features1['spectral_centroid'] - features2['spectral_centroid'])
            centroid_sim = 1 / (1 + centroid_diff / 800)  # 更敏感
            similarities.append(centroid_sim)
            weights.append(0.15)
        except:
            pass
        
        # 5. 新增：频谱对比度相似度
        try:
            contrast_sim = 1 - cosine(features1['spectral_contrast'], features2['spectral_contrast'])
            similarities.append(max(0, contrast_sim))
            weights.append(0.1)
        except:
            pass
        
        # 6. 新增：色度特征相似度
        try:
            chroma_sim = 1 - cosine(features1['chroma'], features2['chroma'])
            similarities.append(max(0, chroma_sim))
            weights.append(0.05)
        except:
            pass
        
        # 7. 过零率相似度
        try:
            zcr_diff = abs(features1['zcr'] - features2['zcr'])
            zcr_sim = 1 / (1 + zcr_diff * 100)
            similarities.append(zcr_sim)
            weights.append(0.05)
        except:
            pass
        
        if len(similarities) == 0:
            return 0.0
        
        # 加权平均
        weights = np.array(weights[:len(similarities)])
        weights = weights / np.sum(weights)  # 归一化权重
        
        final_similarity = np.average(similarities, weights=weights)
        return final_similarity
    
    def sliding_window_matching(self, target_audio, reference_audio):
        """
        滑动窗口匹配算法
        
        Args:
            target_audio: 目标音频 (长音频)
            reference_audio: 参考音频 (短音频)
            
        Returns:
            match_results: 匹配结果列表 [(start_time, end_time, similarity), ...]
        """
        # 窗口参数
        window_samples = int(self.window_duration * self.sample_rate)
        hop_samples = int(window_samples * (1 - self.overlap_ratio))
        
        # 提取参考音频特征
        print("🔄 提取参考音频特征...")
        ref_features = self.extract_comprehensive_features(reference_audio)
        if ref_features is None:
            return []
        
        # 滑动窗口匹配
        print("🔄 执行滑动窗口匹配...")
        match_results = []
        
        for start_sample in range(0, len(target_audio) - window_samples + 1, hop_samples):
            end_sample = start_sample + window_samples
            window_audio = target_audio[start_sample:end_sample]
            
            # 提取窗口特征
            window_features = self.extract_comprehensive_features(window_audio)
            if window_features is None:
                continue
            
            # 计算相似度
            similarity = self.calculate_segment_similarity(ref_features, window_features)
            
            # 记录结果
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            match_results.append((start_time, end_time, similarity))
        
        return match_results
    
    def extract_target_speaker_segments(self, target_audio, reference_audio, min_confidence=0.05):
        """
        提取目标说话人的音频段
        
        Args:
            target_audio: 目标音频
            reference_audio: 参考音频
            min_confidence: 最小置信度阈值
            
        Returns:
            extracted_segments: 提取的音频段
            confidence_scores: 置信度分数
        """
        print("🎯 开始提取目标说话人音频段...")
        
        # 1. 滑动窗口匹配
        match_results = self.sliding_window_matching(target_audio, reference_audio)
        
        if not match_results:
            print("❌ 没有找到匹配的音频段")
            return [], []
        
        # 2. 筛选高置信度段
        high_confidence_segments = [
            (start, end, sim) for start, end, sim in match_results 
            if sim >= min_confidence
        ]
          print(f"📊 找到 {len(high_confidence_segments)} 个高置信度段 (阈值: {min_confidence})")
        
        if not high_confidence_segments:
            # 如果没有高置信度段，使用最高的几个
            sorted_results = sorted(match_results, key=lambda x: x[2], reverse=True)
            top_count = min(10, len(sorted_results))
            high_confidence_segments = sorted_results[:top_count]
            print(f"📊 使用前 {top_count} 个最高相似度段")
        
        # 3. 提取音频段
        extracted_segments = []
        confidence_scores = []
        
        for start_time, end_time, similarity in high_confidence_segments:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            segment = target_audio[start_sample:end_sample]
            extracted_segments.append(segment)
            confidence_scores.append(similarity)
            
            print(f"   ✅ 段 {len(extracted_segments)}: {start_time:.1f}s-{end_time:.1f}s, 相似度: {similarity:.3f}")
        
        return extracted_segments, confidence_scores
    
    def reconstruct_target_speaker_audio_timeline(self, target_audio, match_results, min_confidence=0.05):
        """
        基于时序重构目标说话人音频 - 保持原始时长，集成性别分离
        
        Args:
            target_audio: 原始目标音频
            match_results: 滑动窗口匹配结果 [(start_time, end_time, similarity), ...]
            min_confidence: 最小置信度阈值
            
        Returns:
            reconstructed_audio: 重构的音频 (与原始时长一致)
            avg_confidence: 平均置信度
        """
        if not match_results:
            return np.zeros_like(target_audio), 0.0
        
        print("🔄 基于时序重构目标说话人音频...")
        
        # 1. 分析目标音频的性别特征
        print("🔍 分析目标说话人性别特征...")
        # 使用前30秒的音频进行性别分析
        analysis_length = min(len(target_audio), int(30 * self.sample_rate))
        gender_info = self.analyze_voice_gender(target_audio[:analysis_length])
        
        print(f"   检测到性别: {gender_info['gender']}")
        print(f"   置信度: {gender_info['confidence']:.3f}")
        print(f"   平均基频: {gender_info['f0_mean']:.1f} Hz")
        
        # 2. 应用全局性别滤波
        print("🔧 应用性别特定滤波...")
        if gender_info['gender'] in ['male', 'female'] and gender_info['confidence'] > 0.4:
            target_audio_filtered = self.apply_gender_specific_filter(
                target_audio, gender_info['gender'], strength=0.7
            )
            print(f"   ✅ 应用了{gender_info['gender']}声滤波")
        else:
            target_audio_filtered = target_audio
            print("   ⚠️ 性别识别不确定，跳过性别滤波")
        
        # 初始化输出音频（与原始音频等长）
        reconstructed_audio = np.zeros_like(target_audio)
        confidence_map = np.zeros(len(target_audio))  # 每个采样点的置信度
        
        # 窗口参数
        window_samples = int(self.window_duration * self.sample_rate)
        hop_samples = int(window_samples * (1 - self.overlap_ratio))
        overlap_samples = window_samples - hop_samples
          valid_segments = 0
        total_confidence = 0.0
        
        # 按时间顺序处理每个匹配结果
        for start_time, end_time, similarity in match_results:
            if similarity < min_confidence:
                continue
                
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # 确保索引有效
            start_sample = max(0, start_sample)
            end_sample = min(len(target_audio), end_sample)
            
            if end_sample <= start_sample:
                continue
            
            # 提取当前窗口的滤波后音频
            window_audio = target_audio_filtered[start_sample:end_sample]
            
            # 根据置信度和性别信息决定处理方式
            if similarity > 0.5:  # 高置信度：增强处理
                # 应用自适应谱减法
                if gender_info['gender'] in ['male', 'female']:
                    processed_audio = self.apply_adaptive_spectral_subtraction(
                        window_audio, None, gender_info['gender']
                    )
                else:
                    processed_audio = self.enhance_audio_segment(window_audio, similarity)
                
                # 额外的增益
                processed_audio = processed_audio * (1.0 + similarity * 0.3)
                weight = similarity
                
            elif similarity > 0.2:  # 中等置信度：保留但处理
                # 轻度谱减法处理
                if gender_info['gender'] in ['male', 'female']:
                    processed_audio = self.apply_adaptive_spectral_subtraction(
                        window_audio, None, gender_info['gender']
                    )
                    processed_audio = processed_audio * 0.8
                else:
                    processed_audio = window_audio * (0.6 + similarity * 0.4)
                weight = similarity * 0.8
                
            else:  # 低置信度：大幅衰减
                processed_audio = window_audio * similarity * 0.3
                weight = similarity * 0.2
            
            # 处理重叠区域的加权混合
            current_len = end_sample - start_sample
            for i in range(current_len):
                global_idx = start_sample + i
                if global_idx >= len(reconstructed_audio):
                    break
                
                # 计算当前位置的权重（考虑窗口内的位置）
                if i < overlap_samples and confidence_map[global_idx] > 0:
                    # 重叠区域：加权平均
                    existing_weight = confidence_map[global_idx]
                    total_weight = existing_weight + weight
                    if total_weight > 0:
                        # 加权混合
                        reconstructed_audio[global_idx] = (
                            reconstructed_audio[global_idx] * existing_weight + 
                            processed_audio[i] * weight
                        ) / total_weight
                        confidence_map[global_idx] = min(1.0, total_weight)
                else:
                    # 非重叠区域：直接赋值
                    if confidence_map[global_idx] == 0:
                        reconstructed_audio[global_idx] = processed_audio[i]
                        confidence_map[global_idx] = weight
            
            valid_segments += 1
            total_confidence += similarity
        
        # 计算平均置信度
        avg_confidence = total_confidence / valid_segments if valid_segments > 0 else 0.0
          # 对未匹配的区域进行处理
        unmatched_mask = confidence_map < 0.01
        if np.any(unmatched_mask):
            # 对未匹配区域应用性别滤波和噪声抑制
            unmatched_audio = target_audio_filtered[unmatched_mask]
            if gender_info['gender'] in ['male', 'female']:
                # 对未匹配区域应用更强的抑制
                suppressed_audio = unmatched_audio * 0.05  # 强烈抑制
            else:
                suppressed_audio = unmatched_audio * 0.1
            reconstructed_audio[unmatched_mask] = suppressed_audio
        
        # 后处理：再次应用性别滤波增强效果
        if gender_info['gender'] in ['male', 'female'] and gender_info['confidence'] > 0.5:
            print("🔧 应用最终性别增强滤波...")
            reconstructed_audio = self.apply_gender_specific_filter(
                reconstructed_audio, gender_info['gender'], strength=0.5
            )
        
        # 常规后处理
        reconstructed_audio = self.post_process_audio(reconstructed_audio)
        
        print(f"✅ 时序重构完成！")
        print(f"   原始时长: {len(target_audio)/self.sample_rate:.2f}秒")
        print(f"   重构时长: {len(reconstructed_audio)/self.sample_rate:.2f}秒")
        print(f"   有效段数: {valid_segments}")
        print(f"   平均置信度: {avg_confidence:.3f}")
        print(f"   性别滤波: {'✅' if gender_info['gender'] in ['male', 'female'] else '❌'}")
        
        return reconstructed_audio, avg_confidence
    
    def reconstruct_target_speaker_audio(self, segments, confidence_scores):
        """
        重构目标说话人音频 (旧方法，保留兼容性)
        
        Args:
            segments: 音频段列表
            confidence_scores: 置信度分数列表
            
        Returns:
            reconstructed_audio: 重构的音频
            avg_confidence: 平均置信度
        """
        if not segments:
            return np.array([]), 0.0
        
        print("🔄 重构目标说话人音频...")
        
        # 1. 按置信度排序
        sorted_pairs = sorted(zip(segments, confidence_scores), key=lambda x: x[1], reverse=True)
        
        # 2. 拼接音频段
        reconstructed_audio = []
        used_confidences = []
        
        for i, (segment, confidence) in enumerate(sorted_pairs):
            # 添加短暂的静音间隔
            if i > 0:
                silence_duration = 0.1  # 100ms静音
                silence_samples = int(silence_duration * self.sample_rate)
                silence = np.zeros(silence_samples)
                reconstructed_audio.append(silence)
            
            # 音频增强处理
            enhanced_segment = self.enhance_audio_segment(segment, confidence)
            reconstructed_audio.append(enhanced_segment)
            used_confidences.append(confidence)
        
        # 3. 合并所有段
        final_audio = np.concatenate(reconstructed_audio)
        avg_confidence = np.mean(used_confidences)
        
        # 4. 后处理
        final_audio = self.post_process_audio(final_audio)
        
        print(f"✅ 重构完成，总时长: {len(final_audio)/self.sample_rate:.2f}秒")
        print(f"📊 平均置信度: {avg_confidence:.3f}")
        
        return final_audio, avg_confidence
    
    def enhance_audio_segment(self, audio_segment, confidence):
        """
        增强音频段
        
        Args:
            audio_segment: 音频段
            confidence: 置信度
            
        Returns:
            enhanced_segment: 增强后的音频段
        """
        enhanced = audio_segment.copy()
        
        # 1. 基于置信度的增益调整
        gain_factor = 0.8 + (confidence * 0.4)  # 0.8-1.2范围
        enhanced = enhanced * gain_factor
        
        # 2. 动态范围压缩
        if confidence > 0.5:
            # 高置信度：轻微压缩
            threshold = 0.7
            ratio = 3.0
        else:
            # 低置信度：更强压缩
            threshold = 0.5
            ratio = 5.0
        
        enhanced = self.apply_compression(enhanced, threshold, ratio)
        
        # 3. 噪声抑制
        if confidence < 0.4:
            enhanced = self.noise_suppression(enhanced)
        
        # 4. 限幅
        enhanced = np.clip(enhanced, -0.95, 0.95)
        
        return enhanced
    
    def apply_compression(self, audio, threshold=0.7, ratio=3.0):
        """应用动态范围压缩"""
        # 简单的软压缩器实现
        abs_audio = np.abs(audio)
        
        # 计算增益减少
        gain_reduction = np.ones_like(abs_audio)
        over_threshold = abs_audio > threshold
        
        if np.any(over_threshold):
            excess = abs_audio[over_threshold] - threshold
            gain_reduction[over_threshold] = 1 - (excess / ratio) / abs_audio[over_threshold]
        
        # 应用压缩
        compressed = audio * gain_reduction
        return compressed
    
    def noise_suppression(self, audio):
        """简单的噪声抑制"""
        # 使用软阈值进行噪声抑制
        threshold = np.percentile(np.abs(audio), 10)  # 10%分位数作为噪声阈值
        
        # 软阈值函数
        sign = np.sign(audio)
        magnitude = np.abs(audio)
        suppressed_magnitude = np.maximum(magnitude - threshold, 0.1 * magnitude)
        
        return sign * suppressed_magnitude
    
    def post_process_audio(self, audio):
        """音频后处理"""
        # 1. 去直流分量
        audio = audio - np.mean(audio)
        
        # 2. 高通滤波 (去除低频噪声)
        sos = signal.butter(3, 80, btype='high', fs=self.sample_rate, output='sos')
        audio = signal.sosfilt(sos, audio)
        
        # 3. 低通滤波 (去除高频噪声)
        sos = signal.butter(3, 8000, btype='low', fs=self.sample_rate, output='sos')
        audio = signal.sosfilt(sos, audio)
        
        # 4. 归一化
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        return audio
    
    def process_main_speaker_extraction(self, vocals_path, reference_path, output_path=None):
        """
        主要的说话人提取流程
        
        Args:
            vocals_path: 分离后的人声文件路径
            reference_path: 参考音频路径
            output_path: 输出路径
            
        Returns:
            success: 处理是否成功
            result_info: 结果信息字典
        """
        try:
            print("🎯 开始主人声提取流程")
            print("=" * 50)
            
            # 1. 加载音频
            print("📁 加载音频文件...")
            vocals_audio, _ = librosa.load(vocals_path, sr=self.sample_rate)
            reference_audio, _ = librosa.load(reference_path, sr=self.sample_rate)
            
            print(f"   人声音频: {len(vocals_audio)/self.sample_rate:.2f}秒")
            print(f"   参考音频: {len(reference_audio)/self.sample_rate:.2f}秒")
            
            # 2. 语音活动检测
            print("🔄 执行语音活动检测...")
            voice_segments = self.detect_voice_activity(vocals_audio)
            print(f"   检测到 {len(voice_segments)} 个语音段")            # 3. 滑动窗口匹配 (获取匹配结果而不是直接提取段)
            print("🔄 提取参考音频特征...")
            match_results = self.sliding_window_matching(vocals_audio, reference_audio)
            
            if not match_results:
                print("❌ 未能找到匹配的音频段")
                return False, {"error": "No matching segments found"}
            
            # 4. 使用新的时序重构方法
            final_audio, avg_confidence = self.reconstruct_target_speaker_audio_timeline(
                vocals_audio, match_results, min_confidence=0.05
            )
            
            # 5. 保存结果
            if output_path:
                sf.write(output_path, final_audio, self.sample_rate)
                print(f"💾 结果已保存: {output_path}")
              # 6. 生成报告
            num_valid_segments = sum(1 for _, _, sim in match_results if sim >= 0.05)
            result_info = {
                "original_duration": len(vocals_audio) / self.sample_rate,
                "extracted_duration": len(final_audio) / self.sample_rate,
                "num_segments": num_valid_segments,
                "avg_confidence": avg_confidence,
                "voice_segments_detected": len(voice_segments),
                "compression_ratio": len(final_audio) / len(vocals_audio),
                "success": True
            }
            
            print("\n📊 提取结果:")
            print(f"   原始时长: {result_info['original_duration']:.2f}秒")
            print(f"   提取时长: {result_info['extracted_duration']:.2f}秒")
            print(f"   压缩比: {result_info['compression_ratio']:.2f}")
            print(f"   平均置信度: {avg_confidence:.3f}")
            print(f"   使用段数: {num_valid_segments}")
            
            return True, result_info
            
        except Exception as e:
            print(f"❌ 主人声提取失败: {e}")
            import traceback
            traceback.print_exc()
            return False, {"error": str(e)}
    
    def extract_target_speaker_segments(self, vocals_audio, reference_audio, min_confidence=0.2):
        """
        提取目标说话人音频段
        
        Args:
            vocals_audio: 分离的人声音频
            reference_audio: 参考音频
            min_confidence: 最小置信度阈值
              Returns:
            segments: 匹配的音频段列表
            confidences: 对应的置信度列表
        """
        try:
            # 使用滑动窗口匹配
            match_results = self.sliding_window_matching(vocals_audio, reference_audio)
            
            # 过滤低置信度的段
            segments = []
            confidences = []
            
            for start_time, end_time, similarity in match_results:
                if similarity >= min_confidence:
                    start_sample = int(start_time * self.sample_rate)
                    end_sample = int(end_time * self.sample_rate)
                    segment = vocals_audio[start_sample:end_sample]
                    segments.append(segment)
                    confidences.append(similarity)
            
            return segments, confidences
            
        except Exception as e:
            print(f"音频段提取失败: {e}")
            return [], []
    
    def reconstruct_target_speaker_audio(self, segments, confidences):
        """
        重构目标说话人音频
        
        Args:
            segments: 音频段列表
            confidences: 置信度列表
            
        Returns:
            reconstructed_audio: 重构的音频
            avg_confidence: 平均置信度
        """
        try:
            if not segments:
                return np.array([]), 0.0
            
            # 连接所有段
            reconstructed_audio = np.concatenate(segments)
            
            # 后处理
            reconstructed_audio = self.post_process_audio(reconstructed_audio)
            
            # 计算平均置信度
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return reconstructed_audio, avg_confidence
            
        except Exception as e:
            print(f"音频重构失败: {e}")
            return np.array([]), 0.0
    
    def analyze_voice_gender(self, audio):
        """
        分析音频的性别特征
        
        Args:
            audio: 音频信号
            
        Returns:
            gender_info: 性别信息字典
        """
        if len(audio) == 0:
            return {'gender': 'unknown', 'confidence': 0.0, 'f0_mean': 0}
        
        try:
            # 基频分析
            f0 = librosa.yin(audio, fmin=60, fmax=500, sr=self.sample_rate)
            f0_clean = f0[f0 > 0]
            
            if len(f0_clean) == 0:
                return {'gender': 'unknown', 'confidence': 0.0, 'f0_mean': 0}
            
            f0_mean = np.mean(f0_clean)
            f0_median = np.median(f0_clean)
            
            # 频域能量分析
            stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            
            # 定义性别特征频率带
            male_freq_mask = (freqs >= 80) & (freqs <= 180)     # 男声基频区
            female_freq_mask = (freqs >= 150) & (freqs <= 350)  # 女声基频区
            
            male_energy = np.mean(magnitude[male_freq_mask, :]) if np.any(male_freq_mask) else 0
            female_energy = np.mean(magnitude[female_freq_mask, :]) if np.any(female_freq_mask) else 0
            
            # 性别判断
            if f0_mean < 150 and male_energy > female_energy * 1.2:
                gender = 'male'
                confidence = min(1.0, (150 - f0_mean) / 70 + 0.3)
            elif f0_mean > 170 and female_energy > male_energy * 1.1:
                gender = 'female'
                confidence = min(1.0, (f0_mean - 170) / 80 + 0.3)
            else:
                # 基于能量比例判断
                if male_energy > female_energy * 1.3:
                    gender = 'male'
                    confidence = 0.6
                elif female_energy > male_energy * 1.2:
                    gender = 'female'
                    confidence = 0.6
                else:
                    gender = 'unknown'
                    confidence = 0.2
            
            return {
                'gender': gender,
                'confidence': confidence,
                'f0_mean': f0_mean,
                'f0_median': f0_median,
                'male_energy': male_energy,
                'female_energy': female_energy
            }
            
        except Exception as e:
            print(f"性别分析失败: {e}")
            return {'gender': 'unknown', 'confidence': 0.0, 'f0_mean': 0}
    
    def apply_gender_specific_filter(self, audio, target_gender, strength=0.8):
        """
        应用基于性别的频域滤波
        
        Args:
            audio: 输入音频
            target_gender: 目标性别 ('male' 或 'female')
            strength: 滤波强度 (0-1)
            
        Returns:
            filtered_audio: 滤波后的音频
        """
        if len(audio) == 0:
            return audio
        
        try:
            # 短时傅里叶变换
            stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 频率轴
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            
            # 创建滤波器掩码
            filter_mask = np.ones_like(magnitude)
            
            if target_gender == 'female':
                # 保留女声，抑制男声
                # 强烈抑制男声基频区域 (80-180 Hz)
                male_mask = (freqs >= 80) & (freqs <= 180)
                filter_mask[male_mask, :] *= (1 - strength * 0.8)
                
                # 轻度抑制低频区域 (60-120 Hz)
                low_male_mask = (freqs >= 60) & (freqs <= 120)
                filter_mask[low_male_mask, :] *= (1 - strength * 0.6)
                
                # 保留女声基频区域 (150-350 Hz)
                female_mask = (freqs >= 150) & (freqs <= 350)
                filter_mask[female_mask, :] *= (1 + strength * 0.3)
                
                # 保留高频共振峰 (1000-3000 Hz)
                high_freq_mask = (freqs >= 1000) & (freqs <= 3000)
                filter_mask[high_freq_mask, :] *= (1 + strength * 0.2)
                
            elif target_gender == 'male':
                # 保留男声，抑制女声
                # 强烈抑制女声基频区域 (200-400 Hz)
                female_mask = (freqs >= 200) & (freqs <= 400)
                filter_mask[female_mask, :] *= (1 - strength * 0.8)
                
                # 保留男声基频区域 (80-180 Hz)
                male_mask = (freqs >= 80) & (freqs <= 180)
                filter_mask[male_mask, :] *= (1 + strength * 0.4)
                
                # 轻度抑制高频区域 (2000-4000 Hz)
                high_freq_mask = (freqs >= 2000) & (freqs <= 4000)
                filter_mask[high_freq_mask, :] *= (1 - strength * 0.3)
            
            # 应用滤波器
            filtered_magnitude = magnitude * filter_mask
            
            # 重构音频
            filtered_stft = filtered_magnitude * np.exp(1j * phase)
            filtered_audio = librosa.istft(filtered_stft, hop_length=self.hop_length)
            
            return filtered_audio
            
        except Exception as e:
            print(f"性别滤波失败: {e}")
            return audio
    
    def apply_adaptive_spectral_subtraction(self, mixed_audio, target_features, target_gender):
        """
        自适应谱减法，结合性别信息
        
        Args:
            mixed_audio: 混合音频
            target_features: 目标说话人特征
            target_gender: 目标性别
            
        Returns:
            enhanced_audio: 增强后的音频  
        """
        try:
            # 短时傅里叶变换
            stft = librosa.stft(mixed_audio, n_fft=self.frame_length, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 频率轴
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            
            # 估计噪声谱（非目标性别的频谱）
            frame_energy = np.sum(magnitude**2, axis=0)
            
            # 根据目标性别调整噪声估计策略
            if target_gender == 'female':
                # 对于女声目标，男声区域作为噪声
                male_freq_mask = (freqs >= 80) & (freqs <= 180)
                noise_estimation_mask = male_freq_mask
            elif target_gender == 'male':
                # 对于男声目标，女声区域作为噪声
                female_freq_mask = (freqs >= 200) & (freqs <= 400)
                noise_estimation_mask = female_freq_mask
            else:
                # 未知性别，使用传统方法
                noise_threshold = np.percentile(frame_energy, 20)
                noise_frames = magnitude[:, frame_energy < noise_threshold]
                if noise_frames.shape[1] > 0:
                    noise_spectrum = np.mean(noise_frames, axis=1, keepdims=True)
                else:
                    noise_spectrum = np.mean(magnitude, axis=1, keepdims=True) * 0.1
                
                # 谱减法
                alpha = 2.0
                beta = 0.01
                enhanced_magnitude = magnitude - alpha * noise_spectrum
                enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
                
                # 重构
                enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
                return librosa.istft(enhanced_stft, hop_length=self.hop_length)
            
            # 基于性别的噪声谱估计
            if np.any(noise_estimation_mask):
                noise_spectrum = np.mean(magnitude[noise_estimation_mask, :], axis=0, keepdims=True).T
                noise_spectrum = np.repeat(noise_spectrum, magnitude.shape[0], axis=1).T
            else:
                noise_spectrum = np.mean(magnitude, axis=1, keepdims=True) * 0.1
            
            # 自适应谱减法参数
            alpha = 2.5  # 更强的噪声抑制
            beta = 0.05  # 适中的残留噪声
            
            # 应用谱减法
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
            
            # 重构音频
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
            
            return enhanced_audio
            
        except Exception as e:
            print(f"自适应谱减法失败: {e}")
            return mixed_audio

def main():
    """测试主函数"""
    # 初始化提取器
    extractor = EnhancedSpeakerExtractor(sample_rate=22050)
    
    # 文件路径
    vocals_path = "temp/demucs_output/htdemucs/answer/vocals.wav"
    reference_path = "reference/refne2.wav"
    output_path = "output/enhanced_main_speaker.wav"
    
    # 执行提取
    success, result_info = extractor.process_main_speaker_extraction(
        vocals_path, reference_path, output_path
    )
    
    if success:
        print("\n🎉 主人声提取成功!")
    else:
        print(f"\n❌ 主人声提取失败: {result_info.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
