import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt

class BandpassFilter:
    """
    基于信号与系统理论的带通滤波器实现
    
    主要知识点：
    1. LTI系统理论
    2. 冲激响应与频率响应
    3. 卷积运算
    4. 数字滤波器设计
    """
    
    def __init__(self, sample_rate=22050):
        """
        初始化滤波器
        
        Args:
            sample_rate: 采样率 (Hz)
        """
        self.sample_rate = sample_rate
        self.nyquist_freq = sample_rate / 2
        
    def design_butterworth_filter(self, low_cutoff, high_cutoff, order=5):
        """
        设计巴特沃斯带通滤波器
        
        理论基础：
        - 巴特沃斯滤波器具有最平坦的通带响应
        - 采用双线性变换从模拟滤波器转为数字滤波器
        
        Args:
            low_cutoff: 低截止频率 (Hz)
            high_cutoff: 高截止频率 (Hz)
            order: 滤波器阶数
            
        Returns:
            b, a: 滤波器系数 (分子、分母多项式系数)
        """
        # 归一化截止频率 (相对于奈奎斯特频率)
        low_norm = low_cutoff / self.nyquist_freq
        high_norm = high_cutoff / self.nyquist_freq
        
        # 确保频率在有效范围内
        low_norm = max(0.001, min(low_norm, 0.999))
        high_norm = max(0.001, min(high_norm, 0.999))
        
        if low_norm >= high_norm:
            raise ValueError("低截止频率必须小于高截止频率")
        
        # 设计巴特沃斯带通滤波器
        b, a = butter(order, [low_norm, high_norm], btype='band', analog=False)
        
        return b, a
    
    def get_impulse_response(self, b, a, length=1024):
        """
        计算滤波器的冲激响应 h[n]
        
        理论基础：
        - 冲激响应是LTI系统的完整特征
        - h[n] = IDFT{H(ω)}，其中H(ω)是频率响应
        
        Args:
            b, a: 滤波器系数
            length: 冲激响应长度
            
        Returns:
            n: 时间索引
            h: 冲激响应序列
        """
        # 使用冲激信号δ[n]激励系统
        impulse = np.zeros(length)
        impulse[0] = 1.0
        
        # 计算系统对冲激信号的响应
        h = signal.lfilter(b, a, impulse)
        n = np.arange(length)
        
        return n, h
    
    def get_frequency_response(self, b, a, num_points=8192):
        """
        计算滤波器的频率响应 H(ω)
        
        理论基础：
        - 频率响应是冲激响应的傅里叶变换
        - H(ω) = |H(ω)|e^{jφ(ω)}，包含幅度和相位信息
        
        Args:
            b, a: 滤波器系数
            num_points: 频率点数
            
        Returns:
            frequencies: 频率数组 (Hz)
            H: 复数频率响应
        """
        # 计算数字滤波器的频率响应
        w, H = freqz(b, a, worN=num_points, fs=self.sample_rate)
        
        return w, H
    
    def apply_filter(self, audio_signal, b, a, method='filtfilt'):
        """
        应用滤波器到音频信号 (基于卷积运算)
        
        理论基础：
        - 线性时不变系统：y[n] = x[n] * h[n] (卷积)
        - 频域：Y(ω) = X(ω) · H(ω) (点乘)
        
        Args:
            audio_signal: 输入音频信号
            b, a: 滤波器系数
            method: 滤波方法 ('lfilter', 'filtfilt')
            
        Returns:
            filtered_signal: 滤波后的音频信号
        """
        if method == 'filtfilt':
            # 使用零相位滤波 (前向和反向滤波)
            # 优点：无相位失真，但计算量大
            filtered_signal = filtfilt(b, a, audio_signal)
        else:
            # 使用单向滤波
            # 优点：计算效率高，但有相位延迟
            filtered_signal = signal.lfilter(b, a, audio_signal)
        
        return filtered_signal
    
    def design_fir_filter(self, low_cutoff, high_cutoff, num_taps=101, window='hamming'):
        """
        设计FIR (有限冲激响应) 带通滤波器
        
        理论基础：
        - FIR滤波器具有线性相位特性
        - 使用窗函数法设计，截断理想频率响应的冲激响应
        
        Args:
            low_cutoff: 低截止频率 (Hz)
            high_cutoff: 高截止频率 (Hz)
            num_taps: 滤波器长度 (奇数)
            window: 窗函数类型
            
        Returns:
            h: FIR滤波器系数 (冲激响应)
        """
        # 归一化截止频率
        low_norm = low_cutoff / self.nyquist_freq
        high_norm = high_cutoff / self.nyquist_freq
        
        # 设计FIR带通滤波器
        h = signal.firwin(num_taps, [low_norm, high_norm], 
                         window=window, pass_zero=False)
        
        return h
    
    def apply_fir_filter(self, audio_signal, h):
        """
        应用FIR滤波器 (直接卷积实现)
        
        理论基础：
        - y[n] = Σ(k=0 to M-1) h[k] * x[n-k]
        - M为滤波器长度
        
        Args:
            audio_signal: 输入信号
            h: FIR滤波器系数
            
        Returns:
            filtered_signal: 滤波后信号
        """
        # 使用NumPy的卷积函数实现
        filtered_signal = np.convolve(audio_signal, h, mode='same')
        
        return filtered_signal
    
    def analyze_filter_performance(self, b, a, low_cutoff, high_cutoff):
        """
        分析滤波器性能指标
        
        Args:
            b, a: 滤波器系数
            low_cutoff, high_cutoff: 截止频率
            
        Returns:
            performance_metrics: 性能指标字典
        """
        # 获取频率响应
        frequencies, H = self.get_frequency_response(b, a)
        magnitude_db = 20 * np.log10(np.abs(H) + 1e-10)
        
        # 找到通带和阻带
        passband_mask = (frequencies >= low_cutoff) & (frequencies <= high_cutoff)
        
        # 计算性能指标
        passband_ripple = np.max(magnitude_db[passband_mask]) - np.min(magnitude_db[passband_mask])
        passband_gain = np.mean(magnitude_db[passband_mask])
        
        # 找到-3dB点
        target_gain = passband_gain - 3
        low_3db_idx = np.argmin(np.abs(magnitude_db[:len(frequencies)//4] - target_gain))
        high_3db_idx = np.argmin(np.abs(magnitude_db[3*len(frequencies)//4:] - target_gain)) + 3*len(frequencies)//4
        
        performance_metrics = {
            'passband_ripple_db': passband_ripple,
            'passband_gain_db': passband_gain,
            'low_3db_freq': frequencies[low_3db_idx],
            'high_3db_freq': frequencies[high_3db_idx],
            'filter_order': len(b) - 1
        }
        
        return performance_metrics
    
    def plot_filter_characteristics(self, b, a, title="滤波器特性"):
        """
        绘制滤波器特性图
        
        Args:
            b, a: 滤波器系数
            title: 图表标题
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 频率响应 - 幅度
        frequencies, H = self.get_frequency_response(b, a)
        magnitude_db = 20 * np.log10(np.abs(H) + 1e-10)
        
        ax1.plot(frequencies, magnitude_db, 'b-', linewidth=2)
        ax1.set_xlabel('频率 (Hz)')
        ax1.set_ylabel('幅度 (dB)')
        ax1.set_title('幅度响应')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, self.sample_rate/2])
        
        # 频率响应 - 相位
        phase = np.angle(H, deg=True)
        ax2.plot(frequencies, phase, 'r-', linewidth=2)
        ax2.set_xlabel('频率 (Hz)')
        ax2.set_ylabel('相位 (度)')
        ax2.set_title('相位响应')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, self.sample_rate/2])
        
        # 冲激响应
        n, h = self.get_impulse_response(b, a, length=256)
        ax3.stem(n, h, basefmt='b-')
        ax3.set_xlabel('样本 n')
        ax3.set_ylabel('幅度')
        ax3.set_title('冲激响应 h[n]')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 100])
        
        # 零极图
        zeros, poles, _ = signal.tf2zpk(b, a)
        ax4.scatter(np.real(zeros), np.imag(zeros), marker='o', s=50, 
                   facecolors='none', edgecolors='blue', label='零点')
        ax4.scatter(np.real(poles), np.imag(poles), marker='x', s=50, 
                   color='red', label='极点')
        
        # 单位圆
        theta = np.linspace(0, 2*np.pi, 100)
        ax4.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)
        ax4.set_xlabel('实部')
        ax4.set_ylabel('虚部')
        ax4.set_title('零极图')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

class VoiceEnhancementFilter(BandpassFilter):
    """
    专门用于人声增强的滤波器类
    
    针对人声频率特性进行优化：
    - 基频：85-255 Hz (男声) / 165-265 Hz (女声)
    - 共振峰：300-3400 Hz (语音清晰度关键频段)
    """
    
    def __init__(self, sample_rate=22050):
        super().__init__(sample_rate)
        
        # 人声频率特性参数
        self.male_f0_range = (85, 180)      # 男声基频范围
        self.female_f0_range = (165, 265)   # 女声基频范围
        self.speech_range = (300, 3400)     # 语音清晰度频段
        self.telephone_range = (300, 3400)  # 电话语音频段
    
    def design_voice_enhancement_filter(self, voice_type='speech', gender='mixed'):
        """
        设计专门的人声增强滤波器
        
        Args:
            voice_type: 'speech', 'singing', 'telephone'
            gender: 'male', 'female', 'mixed'
            
        Returns:
            b, a: 滤波器系数
        """
        if voice_type == 'speech':
            low_cutoff, high_cutoff = self.speech_range
        elif voice_type == 'telephone':
            low_cutoff, high_cutoff = self.telephone_range
        elif voice_type == 'singing':
            # 歌唱需要更宽的频带
            low_cutoff, high_cutoff = 80, 8000
        else:
            low_cutoff, high_cutoff = 300, 3400
        
        # 根据性别调整低频截止
        if gender == 'male':
            low_cutoff = max(low_cutoff, self.male_f0_range[0])
        elif gender == 'female':
            low_cutoff = max(low_cutoff, self.female_f0_range[0])
        
        return self.design_butterworth_filter(low_cutoff, high_cutoff, order=6)
    
    def apply_voice_enhancement(self, audio_signal, low_cutoff=300, high_cutoff=3400, 
                               pre_emphasis=True, post_processing=True):
        """
        完整的人声增强处理流程
        
        Args:
            audio_signal: 输入音频
            low_cutoff, high_cutoff: 截止频率
            pre_emphasis: 是否应用预加重
            post_processing: 是否应用后处理
            
        Returns:
            enhanced_signal: 增强后的音频
        """
        enhanced_signal = audio_signal.copy()
        
        # 1. 预加重 (突出高频成分)
        if pre_emphasis:
            enhanced_signal = self.apply_pre_emphasis(enhanced_signal)
        
        # 2. 带通滤波
        b, a = self.design_butterworth_filter(low_cutoff, high_cutoff)
        enhanced_signal = self.apply_filter(enhanced_signal, b, a)
        
        # 3. 后处理
        if post_processing:
            enhanced_signal = self.apply_post_processing(enhanced_signal)
        
        return enhanced_signal
    
    def apply_pre_emphasis(self, signal, alpha=0.97):
        """
        应用预加重滤波器：H(z) = 1 - α*z^(-1)
        
        作用：平衡频谱，突出高频成分
        """
        return np.append(signal[0], signal[1:] - alpha * signal[:-1])
    
    def apply_post_processing(self, signal):
        """
        后处理：归一化和动态范围控制
        """
        # 归一化到[-1, 1]
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val * 0.95
        
        return signal
