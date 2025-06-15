#!/usr/bin/env python3
"""
简化版人声匹配器 - 基于参考音频选择最匹配的分离人声
Simplified Voice Matcher - Select the best matching separated voice based on reference audio
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import shutil

class SimpleVoiceMatcher:
    """简化版人声匹配器"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.n_fft = 2048
        self.n_mfcc = 13
        
    def load_audio(self, audio_path):
        """加载音频文件"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            print(f"  加载: {Path(audio_path).name} ({len(audio)/self.sample_rate:.1f}s)")
            return audio
        except Exception as e:
            print(f"  ❌ 加载失败 {audio_path}: {e}")
            return None
    
    def convert_audio_format(self, input_path, output_path):
        """转换音频格式（64位转16位）"""
        try:
            data, samplerate = sf.read(input_path)
            
            # 归一化处理
            max_val = np.max(np.abs(data))
            if max_val > 1:
                data = data / max_val
            
            data = np.clip(data, -1.0, 1.0) * 0.98
            data_int16 = (data * 32767).astype(np.int16)
            
            sf.write(output_path, data_int16, samplerate, subtype='PCM_16')
            return True
            
        except Exception as e:
            print(f"  ❌ 转换失败: {e}")
            return False
    
    def extract_simple_features(self, audio):
        """提取简化特征"""
        try:
            # 1. MFCC特征
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            # 2. 频谱质心
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )[0]
            
            # 3. 零交叉率
            zcr = librosa.feature.zero_crossing_rate(
                y=audio, hop_length=self.hop_length
            )[0]
            
            # 4. RMS能量
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            
            # 5. 基频估计（简化版）
            try:
                pitches, magnitudes = librosa.piptrack(
                    y=audio[:min(len(audio), self.sample_rate*10)],  # 只取前10秒
                    sr=self.sample_rate,
                    hop_length=self.hop_length,
                    fmin=80,
                    fmax=400
                )
                pitch_values = pitches[pitches > 0]
                pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
            except:
                pitch_mean = 0
            
            # 合并特征
            features = np.concatenate([
                np.mean(mfcc, axis=1),      # MFCC均值
                np.std(mfcc, axis=1),       # MFCC标准差
                [np.mean(spectral_centroids)],  # 频谱质心均值
                [np.std(spectral_centroids)],   # 频谱质心标准差
                [np.mean(zcr)],             # 零交叉率均值
                [np.std(zcr)],              # 零交叉率标准差
                [np.mean(rms)],             # RMS均值
                [np.std(rms)],              # RMS标准差
                [pitch_mean]                # 基频均值
            ])
            
            return features
            
        except Exception as e:
            print(f"    特征提取失败: {e}")
            return None
    
    def calculate_similarity(self, ref_features, target_features):
        """计算相似度"""
        try:
            # 确保特征向量长度一致
            min_len = min(len(ref_features), len(target_features))
            ref_vec = ref_features[:min_len]
            target_vec = target_features[:min_len]
            
            # 余弦相似度
            cos_sim = 1 - cosine(ref_vec, target_vec)
            cos_sim = max(0, cos_sim)
            
            # 皮尔逊相关系数
            try:
                pearson_corr, _ = pearsonr(ref_vec, target_vec)
                if np.isnan(pearson_corr):
                    pearson_corr = 0
                pearson_corr = max(0, pearson_corr)
            except:
                pearson_corr = 0
            
            # 综合分数
            composite_score = 0.6 * cos_sim + 0.4 * pearson_corr
            
            return {
                'cosine': cos_sim,
                'pearson': pearson_corr,
                'composite': composite_score
            }
            
        except Exception as e:
            print(f"    相似度计算失败: {e}")
            return {'cosine': 0, 'pearson': 0, 'composite': 0}
    
    def match_voices(self, reference_audio_path, separated_voices_dir, output_dir="output"):
        """匹配人声"""
        try:
            print("🎯 简化版人声匹配器")
            print("=" * 50)
            
            # 创建输出目录
            Path(output_dir).mkdir(exist_ok=True)
            
            # 1. 加载参考音频
            print("\n📁 加载参考音频...")
            reference_audio = self.load_audio(reference_audio_path)
            if reference_audio is None:
                return None
            
            # 2. 提取参考音频特征
            print("🧠 提取参考音频特征...")
            ref_features = self.extract_simple_features(reference_audio)
            if ref_features is None:
                print("❌ 参考音频特征提取失败")
                return None
            
            print(f"  参考特征维度: {ref_features.shape}")
            
            # 3. 查找分离的人声文件（去重）
            separated_dir = Path(separated_voices_dir)
            voice_files = set()  # 使用set去重
            
            # 优先选择_fixed版本
            for file in separated_dir.glob("SPEAKER_*_fixed.wav"):
                voice_files.add(file)
            
            # 如果没有_fixed版本，添加原版
            for file in separated_dir.glob("SPEAKER_*.wav"):
                fixed_version = separated_dir / f"{file.stem}_fixed.wav"
                if not fixed_version.exists():
                    voice_files.add(file)
            
            voice_files = sorted(list(voice_files))
            
            if not voice_files:
                print(f"❌ 在 {separated_voices_dir} 中未找到SPEAKER_*.wav文件")
                return None
            
            print(f"\n🔍 找到 {len(voice_files)} 个人声文件:")
            for vf in voice_files:
                print(f"  - {vf.name}")
            
            # 4. 处理每个人声文件
            print(f"\n🔄 处理人声文件...")
            results = []
            
            for i, voice_file in enumerate(voice_files):
                print(f"\n[{i+1}/{len(voice_files)}] {voice_file.name}")
                
                # 转换格式
                converted_file = Path(output_dir) / f"converted_{voice_file.name}"
                if self.convert_audio_format(str(voice_file), str(converted_file)):
                    
                    # 加载转换后的音频
                    voice_audio = self.load_audio(str(converted_file))
                    if voice_audio is not None:
                        
                        # 提取特征
                        voice_features = self.extract_simple_features(voice_audio)
                        if voice_features is not None:
                            
                            # 计算相似度
                            similarity_scores = self.calculate_similarity(ref_features, voice_features)
                            
                            result = {
                                'file_path': str(voice_file),
                                'converted_path': str(converted_file),
                                'similarity_scores': similarity_scores,
                                'features': voice_features
                            }
                            results.append(result)
                            
                            print(f"    余弦相似度: {similarity_scores['cosine']:.3f}")
                            print(f"    皮尔逊相关: {similarity_scores['pearson']:.3f}")
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
            shutil.copy2(best_match['converted_path'], best_output_path)
            
            print(f"✅ 最佳匹配结果保存到: {best_output_path}")
            
            # 7. 生成简单报告
            self.generate_simple_report(results, best_match, output_dir)
            
            return {
                'best_match': best_match,
                'all_results': results,
                'output_path': str(best_output_path)
            }
            
        except Exception as e:
            print(f"❌ 人声匹配失败: {e}")
            return None
    
    def generate_simple_report(self, results, best_match, output_dir):
        """生成简单报告"""
        try:
            print("\n📊 生成分析报告...")
            
            # 文本报告
            report_path = Path(output_dir) / "voice_matching_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("人声匹配分析报告\\n")
                f.write("=" * 30 + "\\n\\n")
                
                # 按综合分数排序
                sorted_results = sorted(results, key=lambda x: x['similarity_scores']['composite'], reverse=True)
                
                for i, result in enumerate(sorted_results):
                    file_name = Path(result['file_path']).name
                    scores = result['similarity_scores']
                    f.write(f"{i+1}. {file_name}\\n")
                    f.write(f"   综合分数: {scores['composite']:.3f}\\n")
                    f.write(f"   余弦相似度: {scores['cosine']:.3f}\\n")
                    f.write(f"   皮尔逊相关: {scores['pearson']:.3f}\\n")
                    
                    if result == best_match:
                        f.write("   *** 最佳匹配 ***\\n")
                    f.write("\\n")
                
                f.write(f"最终选择: {Path(best_match['file_path']).name}\\n")
                f.write(f"最高分数: {best_match['similarity_scores']['composite']:.3f}\\n")
            
            print(f"  📄 报告保存到: {report_path}")
            
            # 可视化
            plt.figure(figsize=(10, 6))
            
            file_names = [Path(result['file_path']).stem for result in results]
            composite_scores = [result['similarity_scores']['composite'] for result in results]
            
            bars = plt.bar(file_names, composite_scores, alpha=0.7)
            
            # 标记最佳匹配
            best_index = results.index(best_match)
            bars[best_index].set_color('red')
            bars[best_index].set_alpha(0.9)
            
            plt.xlabel('候选人声文件')
            plt.ylabel('综合相似度分数')
            plt.title('人声匹配结果对比')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            viz_path = Path(output_dir) / "voice_matching_comparison.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 图表保存到: {viz_path}")
            
        except Exception as e:
            print(f"报告生成失败: {e}")

def main():
    """主函数"""
    # 配置路径
    reference_audio_path = "reference/refne2.wav"  # 参考音频
    separated_voices_dir = "."  # 分离人声文件目录
    output_dir = "output"
    
    # 创建匹配器
    matcher = SimpleVoiceMatcher(sample_rate=16000)
    
    # 执行匹配
    result = matcher.match_voices(
        reference_audio_path=reference_audio_path,
        separated_voices_dir=separated_voices_dir,
        output_dir=output_dir
    )
    
    if result:
        print(f"\\n🎉 人声匹配完成!")
        print(f"最佳匹配文件: {result['output_path']}")
        print(f"综合相似度: {result['best_match']['similarity_scores']['composite']:.3f}")
        
        # 显示所有结果排名
        print("\\n📊 所有候选人声排名:")
        sorted_results = sorted(result['all_results'], 
                              key=lambda x: x['similarity_scores']['composite'], 
                              reverse=True)
        for i, res in enumerate(sorted_results):
            name = Path(res['file_path']).name
            score = res['similarity_scores']['composite']
            print(f"  {i+1}. {name}: {score:.3f}")
    else:
        print("\\n❌ 人声匹配失败")

if __name__ == "__main__":
    main()
