#!/usr/bin/env python3
"""
严格的智能说话人分离测试
使用更严格的阈值来更好地区分男女声
"""

from smart_speaker_separator import SmartSpeakerSeparator
from pathlib import Path

def main():
    print("🎯 严格智能说话人分离测试")
    print("=" * 60)
    
    # 创建严格版本的分离器
    separator = SmartSpeakerSeparator(sample_rate=22050)
    
    # 调整为更严格的参数
    separator.similarity_threshold = 0.7        # 提高相似度阈值
    separator.energy_threshold_ratio = 0.08     # 提高能量阈值
    separator.min_segment_duration = 0.5        # 增加最小段长度
    
    print(f"📋 严格参数设置:")
    print(f"   相似度阈值: {separator.similarity_threshold}")
    print(f"   能量阈值比例: {separator.energy_threshold_ratio}")
    print(f"   最小段长度: {separator.min_segment_duration}s")
    
    # 文件路径
    mixed_path = "temp/demucs_output/htdemucs/answer/vocals.wav"
    reference_path = "reference/refne2.wav"
    output_path = "output/strict_separated_speaker.wav"
    
    # 检查文件
    if not Path(mixed_path).exists():
        print(f"❌ 混合音频不存在: {mixed_path}")
        return
    
    if not Path(reference_path).exists():
        print(f"❌ 参考音频不存在: {reference_path}")
        return
    
    # 执行严格分离
    success, result_info = separator.extract_target_speaker(
        mixed_path, reference_path, output_path
    )
    
    if success:
        print(f"\n🎉 严格分离完成!")
        print(f"📊 对比结果:")
        
        keep_ratio = result_info['keep_ratio']
        kept_duration = result_info['kept_speech_duration']
        total_duration = result_info['original_duration']
        
        print(f"   保留段数: {result_info['kept_segments']:.1f}/{result_info['total_segments']}")
        print(f"   保留率: {keep_ratio*100:.1f}%")
        print(f"   保留时长: {kept_duration:.2f}s / {total_duration:.2f}s")
        print(f"   时长保留率: {kept_duration/total_duration*100:.1f}%")
        
        # 评估分离效果
        if 30 <= keep_ratio*100 <= 70:
            print("🎯 理想效果: 保留率在30-70%，说明成功区分了男女声")
        elif keep_ratio*100 > 80:
            print("⚠️ 可能过度保留: 建议进一步提高阈值")
        elif keep_ratio*100 < 20:
            print("⚠️ 可能过度丢弃: 建议降低阈值")
        else:
            print("👍 良好效果: 保留率合理")
            
        print(f"\n💾 严格分离结果已保存: {output_path}")
        print(f"🔊 建议听取音频确认分离效果")
        
        # 给出进一步优化建议
        if keep_ratio > 0.8:
            print(f"\n💡 如果仍有男声残留，可以尝试:")
            print(f"   1. 提高相似度阈值到 0.8")
            print(f"   2. 增加基频判断权重")
            print(f"   3. 添加性别特定的频域滤波")
    else:
        print(f"\n❌ 严格分离失败: {result_info.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
