import soundfile as sf
info = sf.info("SPEAKER_00.wav")
print(info)


import soundfile as sf
import numpy as np

# 读取音频数据
data, samplerate = sf.read("output/speaker_output/SPEAKER_00.wav")

# 先查看波形的最大绝对值，便于判断是否超出范围
max_val = np.max(np.abs(data))
print(f"最大绝对值: {max_val}")

# 如果 max_val > 1，说明波形超出正常范围，需要先归一化
if max_val > 1:
    data = data / max_val

# 也可以做更稳健的归一化（留点安全余量，防止裁剪失真）
data = np.clip(data, -1.0, 1.0) * 0.98

# 转换为 int16 类型，避免溢出
data_int16 = (data * 32767).astype(np.int16)

# 保存音频
sf.write("GUISPEAKER_00_fixed.wav", data_int16, samplerate, subtype='PCM_16')

print("转换成功，音质更好，试试看")

