#!/usr/bin/env python3
"""
人声匹配器 - 基于参考音频选择最匹配的分离人声
Voice Matcher - Select the best matching separated voice based on reference audio
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import os

class VoiceMatcher:
    """人声匹配器"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.hop_length = 256
        self.n_fft = 1024
        self.n_mfcc = 13
        self.n_mels = 80
        
    def load_audio(self, audio_path):
        """加载音频文件"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            print(f"加载音频: {audio_path}")
            print(f"  时长: {len(audio)/self.sample_rate:.2f}秒")
            return audio
        except Exception as e:
            print(f"音频加载失败 {audio_path}: {e}")
            return None
    
    def convert_audio_format(self, input_path, output_path):
        """使用transwav.py转换音频格式（64位转16位）"""
        try:
            # 读取音频数据
            data, samplerate = sf.read(input_path)
            
            # 查看波形的最大绝对值
            max_val = np.max(np.abs(data))
            print(f"  最大绝对值: {max_val}")
            
            # 如果超出正常范围，先归一化
            if max_val > 1:
                data = data / max_val
            
            # 稳健的归一化
            data = np.clip(data, -1.0, 1.0) * 0.98
            
            # 转换为 int16 类型
            data_int16 = (data * 32767).astype(np.int16)
            
            # 保存音频
            sf.write(output_path, data_int16, samplerate, subtype='PCM_16')
            print(f"  ✅ 格式转换完成: {output_path}")
            return True
            
        except Exception as e:
            print(f"  ❌ 格式转换失败: {e}")
            return False
    
    def extract_comprehensive_features(self, audio):
        """提取综合特征"""
        try:
            features = {}
            
            # 1. MFCC特征
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            features['mfcc_delta'] = np.mean(librosa.feature.delta(mfcc), axis=1)
            
            # 2. Mel频谱特征
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_mean'] = np.mean(mel_spec_db, axis=1)
            features['mel_std'] = np.std(mel_spec_db, axis=1)
            
            # 3. 色度特征
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            
            # 4. 频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]
            
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            
            # 5. 零交叉率
            zcr = librosa.feature.zero_crossing_rate(
                y=audio, hop_length=self.hop_length
            )[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 6. 基频特征
            pitches, magnitudes = librosa.piptrack(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmin=80,
                fmax=400
            )
            pitch_values = pitches[pitches > 0]
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_median'] = np.median(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_median'] = 0
            
            # 7. RMS能量
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # 合并所有特征为向量
            feature_vector = np.concatenate([
                features['mfcc_mean'],
                features['mfcc_std'], 
                features['mfcc_delta'],
                features['mel_mean'][:20],  # 只取前20个mel频带
                features['mel_std'][:20],
                features['chroma_mean'],
                features['chroma_std'],
                [features['spectral_centroid_mean']],
                [features['spectral_centroid_std']],
                [features['spectral_rolloff_mean']],
                [features['spectral_bandwidth_mean']],
                [features['zcr_mean']],
                [features['zcr_std']],
                [features['pitch_mean']],
                [features['pitch_std']],
                [features['pitch_median']],
                [features['rms_mean']],
                [features['rms_std']]
            ])
            
            return feature_vector, features
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None, None
    
    def calculate_similarity_scores(self, ref_features, target_features):
        """计算多维相似度分数"""
        try:
            scores = {}
            
            # 确保特征向量长度一致
            min_len = min(len(ref_features), len(target_features))
            ref_vec = ref_features[:min_len]
            target_vec = target_features[:min_len]
            
            # 1. 余弦相似度
            cos_sim = 1 - cosine(ref_vec, target_vec)
            scores['cosine'] = max(0, cos_sim)
            
            # 2. 皮尔逊相关系数
            try:
                pearson_corr, _ = pearsonr(ref_vec, target_vec)
                if np.isnan(pearson_corr):
                    pearson_corr = 0
                scores['pearson'] = max(0, pearson_corr)
            except:
                scores['pearson'] = 0
            
            # 3. 欧氏距离相似度
            euclidean_dist = np.linalg.norm(ref_vec - target_vec)
            max_possible_dist = np.linalg.norm(ref_vec) + np.linalg.norm(target_vec)
            if max_possible_dist > 0:
                euclidean_sim = 1 - (euclidean_dist / max_possible_dist)
            else:
                euclidean_sim = 0
            scores['euclidean'] = max(0, euclidean_sim)
            
            # 4. 曼哈顿距离相似度
            manhattan_dist = np.sum(np.abs(ref_vec - target_vec))
            max_manhattan = np.sum(np.abs(ref_vec)) + np.sum(np.abs(target_vec))
            if max_manhattan > 0:
                manhattan_sim = 1 - (manhattan_dist / max_manhattan)
            else:
                manhattan_sim = 0
            scores['manhattan'] = max(0, manhattan_sim)
            
            # 5. 综合加权分数
            weights = {
                'cosine': 0.4,
                'pearson': 0.3,
                'euclidean': 0.2,
                'manhattan': 0.1
            }
            
            composite_score = sum(scores[metric] * weight for metric, weight in weights.items())
            scores['composite'] = composite_score
            
            return scores
            
        except Exception as e:
            print(f"相似度计算失败: {e}")
            return {'cosine': 0, 'pearson': 0, 'euclidean': 0, 'manhattan': 0, 'composite': 0}
    
    def match_voices(self, reference_audio_path, separated_voices_dir, output_dir="output"):
        """匹配人声"""
        try:
            print("🎯 开始人声匹配...")
            print("=" * 60)
            
            # 创建输出目录
            Path(output_dir).mkdir(exist_ok=True)
            
            # 1. 加载参考音频
            print("\n📁 加载参考音频...")
            reference_audio = self.load_audio(reference_audio_path)
            if reference_audio is None:
                return None
            
            # 2. 提取参考音频特征
            print("🧠 提取参考音频特征...")
            ref_features, ref_details = self.extract_comprehensive_features(reference_audio)
            if ref_features is None:
                print("❌ 参考音频特征提取失败")
                return None
            
            print(f"参考特征维度: {ref_features.shape}")
            
            # 3. 查找分离的人声文件
            separated_dir = Path(separated_voices_dir)
            voice_files = []
            
            # 查找各种可能的文件格式
            for pattern in ["*.wav", "*.mp3", "*.flac"]:
                voice_files.extend(separated_dir.glob(pattern))
            
            # 也查找SPEAKER_开头的文件
            for pattern in ["SPEAKER_*.wav", "speaker_*.wav"]:
                voice_files.extend(separated_dir.glob(pattern))
            
            if not voice_files:
                print(f"❌ 在 {separated_voices_dir} 中未找到人声文件")
                return None
            
            print(f"\n🔍 找到 {len(voice_files)} 个人声文件:")
            for vf in voice_files:
                print(f"  - {vf.name}")
            
            # 4. 转换音频格式并比较
            print(f"\n🔄 转换音频格式并提取特征...")
            results = []
            
            for i, voice_file in enumerate(voice_files):
                print(f"\n处理文件 {i+1}/{len(voice_files)}: {voice_file.name}")
                
                # 转换格式
                converted_file = Path(output_dir) / f"converted_{voice_file.stem}.wav"
                if self.convert_audio_format(str(voice_file), str(converted_file)):
                    
                    # 加载转换后的音频
                    voice_audio = self.load_audio(str(converted_file))
                    if voice_audio is not None:
                        
                        # 提取特征
                        voice_features, voice_details = self.extract_comprehensive_features(voice_audio)
                        if voice_features is not None:
                            
                            # 计算相似度
                            similarity_scores = self.calculate_similarity_scores(ref_features, voice_features)
                            
                            result = {
                                'file_path': str(voice_file),
                                'converted_path': str(converted_file),
                                'similarity_scores': similarity_scores,
                                'features': voice_features,
                                'feature_details': voice_details
                            }
                            results.append(result)
                            
                            print(f"  相似度分数:")
                            print(f"    余弦相似度: {similarity_scores['cosine']:.3f}")
                            print(f"    皮尔逊相关: {similarity_scores['pearson']:.3f}")
                            print(f"    欧氏距离: {similarity_scores['euclidean']:.3f}")
                            print(f"    综合分数: {similarity_scores['composite']:.3f}")
            
            if not results:
                print("❌ 没有成功处理的人声文件")
                return None
            
            # 5. 选择最佳匹配
            print(f"\n🏆 选择最佳匹配...")
            best_match = max(results, key=lambda x: x['similarity_scores']['composite'])
            
            print(f"最佳匹配: {Path(best_match['file_path']).name}")
            print(f"综合相似度分数: {best_match['similarity_scores']['composite']:.3f}")
            
            # 6. 保存最佳匹配结果
            best_output_path = Path(output_dir) / "best_matched_voice.wav"
            
            # 复制最佳匹配的转换后文件
            import shutil
            shutil.copy2(best_match['converted_path'], best_output_path)
            
            print(f"✅ 最佳匹配结果保存到: {best_output_path}")
            
            # 7. 生成分析报告
            self.generate_matching_report(reference_audio_path, results, best_match, output_dir)
            
            return {
                'best_match': best_match,
                'all_results': results,
                'output_path': str(best_output_path)
            }
            
        except Exception as e:
            print(f"❌ 人声匹配失败: {e}")
            return None
    
    def generate_matching_report(self, reference_path, results, best_match, output_dir):
        """生成匹配分析报告"""
        try:
            print("\n📊 生成分析报告...")
            
            # 1. 文本报告
            report_path = Path(output_dir) / "voice_matching_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("人声匹配分析报告\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"参考音频: {reference_path}\n")
                f.write(f"处理文件数: {len(results)}\n\n")
                
                f.write("所有候选人声相似度分数:\n")
                f.write("-" * 40 + "\n")
                
                # 按综合分数排序
                sorted_results = sorted(results, key=lambda x: x['similarity_scores']['composite'], reverse=True)
                
                for i, result in enumerate(sorted_results):
                    file_name = Path(result['file_path']).name
                    scores = result['similarity_scores']
                    f.write(f"\n{i+1}. {file_name}\n")
                    f.write(f"   综合分数: {scores['composite']:.3f}\n")
                    f.write(f"   余弦相似度: {scores['cosine']:.3f}\n")
                    f.write(f"   皮尔逊相关: {scores['pearson']:.3f}\n")
                    f.write(f"   欧氏距离: {scores['euclidean']:.3f}\n")
                    f.write(f"   曼哈顿距离: {scores['manhattan']:.3f}\n")
                    
                    if result == best_match:
                        f.write("   *** 最佳匹配 ***\n")
                
                f.write(f"\n最终选择: {Path(best_match['file_path']).name}\n")
                f.write(f"最高综合分数: {best_match['similarity_scores']['composite']:.3f}\n")
            
            print(f"  📄 文本报告: {report_path}")
            
            # 2. 可视化报告
            plt.figure(figsize=(12, 8))
            
            # 相似度对比图
            file_names = [Path(result['file_path']).stem for result in results]
            composite_scores = [result['similarity_scores']['composite'] for result in results]
            cosine_scores = [result['similarity_scores']['cosine'] for result in results]
            pearson_scores = [result['similarity_scores']['pearson'] for result in results]
            
            x = np.arange(len(file_names))
            width = 0.25
            
            plt.bar(x - width, composite_scores, width, label='综合分数', alpha=0.8)
            plt.bar(x, cosine_scores, width, label='余弦相似度', alpha=0.8)
            plt.bar(x + width, pearson_scores, width, label='皮尔逊相关', alpha=0.8)
            
            plt.xlabel('候选人声文件')
            plt.ylabel('相似度分数')
            plt.title('人声匹配相似度对比')
            plt.xticks(x, file_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            viz_path = Path(output_dir) / "voice_matching_analysis.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 可视化报告: {viz_path}")
            
        except Exception as e:
            print(f"报告生成失败: {e}")

def main():
    """主函数"""
    # 配置路径
    reference_audio_path = "reference/lttgd_ref.wav"  # 参考音频
    separated_voices_dir = "sepvoice"  # 分离人声文件目录（当前目录，包含SPEAKER_XX.wav）
    output_dir = "output"
    
    print("🎤 人声匹配器 - 基于参考音频选择最佳匹配人声")
    print("=" * 60)
    
    # 创建匹配器
    matcher = VoiceMatcher(sample_rate=16000)
    
    # 执行匹配
    result = matcher.match_voices(
        reference_audio_path=reference_audio_path,
        separated_voices_dir=separated_voices_dir,
        output_dir=output_dir
    )
    
    if result:
        print(f"\n🎉 人声匹配完成!")
        print(f"最佳匹配文件: {result['output_path']}")
        print(f"综合相似度: {result['best_match']['similarity_scores']['composite']:.3f}")
    else:
        print("\n❌ 人声匹配失败")

if __name__ == "__main__":
    main()
