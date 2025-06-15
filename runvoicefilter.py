reference = "C:/Users/zzy/Desktop/Courses/xinhaoyuxitong/voice_processing/reference/lttgd_ref.wav"
mixture = "C:/Users/zzy/Desktop/Courses/xinhaoyuxitong/voice_processing/input/lttgd.wav"

hf_token = "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"


from pyannote.audio import Pipeline
import scipy.io.wavfile
import torch
import torchaudio
from pyannote.audio.pipelines.utils.hook import ProgressHook

# 设置 Hugging Face 访问令牌
HF_ACCESS_TOKEN = "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"  # 替换为你的访问令牌

# 实例化 pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speech-separation-ami-1.0",
    use_auth_token=HF_ACCESS_TOKEN
)

# （可选）如果有 GPU，将 pipeline 移动到 GPU
pipeline.to(torch.device("cuda"))

# 加载音频文件（从文件加载）
audio_file = "C:/Users/zzy/Desktop/Courses/xinhaoyuxitong/voice_processing/input/lttgd.wav"  # 替换为你的音频文件路径

# （可选）从内存加载音频以加速处理
waveform, sample_rate = torchaudio.load(audio_file)

# 运行 pipeline，监控进度
with ProgressHook() as hook:
    diarization, sources = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)

# 保存说话者分割结果到 RTTM 文件
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

# 保存每个说话者的分离音频到单独的 WAV 文件
for s, speaker in enumerate(diarization.labels()):
    scipy.io.wavfile.write(f"{speaker}.wav", 16000, sources.data[:, s])