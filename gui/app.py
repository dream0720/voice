import sys
import os

# PyQt6 imports first
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect, QTimer
from PyQt6.QtGui import QColor

# Then other imports
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from gui.ui_main import Ui_MainWindow

class WorkerThread(QThread):
    """🔄 音频处理工作线程 - 应用信号与系统理论"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished_processing = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, input_path, reference_path, low_freq, high_freq):
        super().__init__()
        self.input_path = input_path
        self.reference_path = reference_path
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.pipeline = None
        
    def run(self):
        """执行音频处理流程"""
        try:
            # 🎯 导入处理管道
            from core.process_pipeline import ProcessPipeline
            
            # 🎯 Phase 1: 初始化处理管道
            self.status_updated.emit("🔄 Initializing processing pipeline...")
            self.progress_updated.emit(5)
            self.pipeline = ProcessPipeline(sample_rate=22050)
            
            # 🎵 Phase 2: 音源分离
            self.status_updated.emit("🎵 Step 1: Source separation with Demucs...")
            self.progress_updated.emit(20)
            separated_files = self.pipeline.separate_sources_with_demucs(self.input_path)
            
            # 🎯 Phase 3: 说话人匹配
            self.status_updated.emit("🎯 Step 2: Matching target speaker...")
            self.progress_updated.emit(40)
            matched_audio, confidence = self.pipeline.match_target_speaker(
                self.reference_path, 
                separated_files.get('vocals')
            )
            
            # 🔧 Phase 4: 语音增强
            self.status_updated.emit("🔧 Step 3: Applying voice enhancement...")
            self.progress_updated.emit(60)
            enhanced_audio = self.pipeline.apply_voice_enhancement(
                matched_audio, self.low_freq, self.high_freq
            )
            
            # 📊 Phase 5: 后处理
            self.status_updated.emit("📊 Step 4: Post-processing optimization...")
            self.progress_updated.emit(80)
            final_audio = self.pipeline.post_process_audio(enhanced_audio)
            
            # 💾 Phase 6: 保存结果
            self.status_updated.emit("💾 Step 5: Saving results...")
            self.progress_updated.emit(95)
            
            import soundfile as sf
            output_path = self.pipeline.output_dir / "enhanced_voice_output.wav"
            sf.write(output_path, final_audio, self.pipeline.sample_rate)
            
            # 生成处理报告
            self.pipeline.generate_processing_report(
                self.input_path, output_path, confidence
            )
            
            # ✅ 完成
            self.status_updated.emit("✅ Voice extraction completed successfully!")
            self.progress_updated.emit(100)
            
            self.finished_processing.emit(str(output_path))
            
        except Exception as e:
            import traceback
            error_msg = f"Processing failed: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            self.error_occurred.emit(error_msg)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # 🎨 初始化变量
        self.input_audio_path = None
        self.reference_audio_path = None
        self.processed_audio_path = None
        self.current_audio_data = None
        self.current_sr = None
        
        # 🎬 设置窗口特效
        self.setup_window_properties()
        
        # 🔗 连接信号槽
        self.connect_signals()
        
        # 📊 初始化可视化
        self.initialize_modern_plots()
        
        # 🎉 显示欢迎界面
        self.show_welcome_state()
    
    def setup_window_properties(self):
        """🎨 设置现代化窗口属性"""
        # 设置窗口标题（虽然使用自定义标题栏，但仍需设置）
        self.setWindowTitle("🎵 Modern Voice Enhancement System")
        
        # 设置最小尺寸
        self.setMinimumSize(1400, 900)
        
        # 窗口居中显示
        self.center_window()
        
        # 添加阴影效果
        self.setStyleSheet("""
            QMainWindow {
                background: transparent;
            }
        """)

    def center_window(self):
        """将窗口居中显示"""
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
    
    def connect_signals(self):
        """🔗 连接所有信号槽"""
        # 📁 文件选择按钮
        self.input_button.clicked.connect(self.select_input_audio)
        self.reference_button.clicked.connect(self.select_reference_audio)
        
        # 🚀 处理控制按钮
        self.process_button.clicked.connect(self.start_processing)
        
        # 🎵 播放控制按钮
        self.play_original_button.clicked.connect(self.play_original_audio)
        self.play_processed_button.clicked.connect(self.play_processed_audio)
        
        # 🎛️ 滑块值变化
        self.low_freq_slider.valueChanged.connect(self.update_filter_visualization)
        self.high_freq_slider.valueChanged.connect(self.update_filter_visualization)
        
        # 📊 标签页切换
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
    
    def initialize_modern_plots(self):
        """📊 初始化现代化图表显示"""
        # 设置matplotlib现代化主题
        plt.style.use('default')
        
        # 初始化所有图表
        self.setup_waveform_plot()
        self.setup_spectrum_plot()
        self.setup_filter_plot()
        self.setup_spectrogram_plot()
        
        # 显示演示数据
        self.show_demo_visualizations()
        
        # 更新滤波器响应
        self.update_filter_visualization()
    
    def setup_waveform_plot(self):
        """🌊 设置波形图表"""
        self.waveform_ax.clear()
        self.waveform_ax.set_facecolor('white')
        
        # 添加知识点标注 - 修复RGBA问题
        self.waveform_ax.text(0.02, 0.98, '📈 Time Domain Analysis', 
                             transform=self.waveform_ax.transAxes,
                             fontsize=12, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.5', 
                                     facecolor='lightblue',
                                     edgecolor='none'))
        
        self.waveform_ax.set_title('Audio Waveform - Time Domain Signal', 
                                  fontsize=16, fontweight='bold', 
                                  color='#2d3748', pad=20)
        self.waveform_ax.set_xlabel('Time (seconds)', fontsize=12, color='#4a5568')
        self.waveform_ax.set_ylabel('Amplitude', fontsize=12, color='#4a5568')
        self.waveform_ax.grid(True, alpha=0.2, color='#a0aec0')
        
        # 美化坐标轴
        for spine in self.waveform_ax.spines.values():
            spine.set_color('#e2e8f0')
            spine.set_linewidth(1)
        self.waveform_ax.tick_params(colors='#718096')
        
        self.waveform_canvas.draw()
    
    def setup_spectrum_plot(self):
        """📊 设置频谱图表"""
        self.spectrum_ax.clear()
        self.spectrum_ax.set_facecolor('white')
        
        # 添加知识点标注 - 修复RGBA问题
        self.spectrum_ax.text(0.02, 0.98, '📊 FFT Analysis', 
                             transform=self.spectrum_ax.transAxes,
                             fontsize=12, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.5', 
                                     facecolor='lightgreen',
                                     edgecolor='none'))
        
        self.spectrum_ax.set_title('Frequency Spectrum - FFT Transform', 
                                  fontsize=16, fontweight='bold', 
                                  color='#2d3748', pad=20)
        self.spectrum_ax.set_xlabel('Frequency (Hz)', fontsize=12, color='#4a5568')
        self.spectrum_ax.set_ylabel('Magnitude (dB)', fontsize=12, color='#4a5568')
        self.spectrum_ax.grid(True, alpha=0.2, color='#a0aec0')
        
        # 美化坐标轴
        for spine in self.spectrum_ax.spines.values():
            spine.set_color('#e2e8f0')
            spine.set_linewidth(1)
        self.spectrum_ax.tick_params(colors='#718096')
        
        self.spectrum_canvas.draw()
    
    def setup_filter_plot(self):
        """🔧 设置滤波器图表"""
        self.filter_ax.clear()
        self.filter_ax.set_facecolor('white')
        
        # 添加知识点标注 - 修复RGBA问题
        self.filter_ax.text(0.02, 0.98, '🔧 LTI System Design', 
                           transform=self.filter_ax.transAxes,
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', 
                                   facecolor='lightcoral',
                                   edgecolor='none'))
        
        self.filter_ax.set_title('Bandpass Filter Response - LTI System', 
                                fontsize=16, fontweight='bold', 
                                color='#2d3748', pad=20)
        self.filter_ax.set_xlabel('Frequency (Hz)', fontsize=12, color='#4a5568')
        self.filter_ax.set_ylabel('Magnitude (dB)', fontsize=12, color='#4a5568')
        self.filter_ax.grid(True, alpha=0.2, color='#a0aec0')
        
        # 美化坐标轴
        for spine in self.filter_ax.spines.values():
            spine.set_color('#e2e8f0')
            spine.set_linewidth(1)
        self.filter_ax.tick_params(colors='#718096')
        
        self.filter_canvas.draw()
    
    def setup_spectrogram_plot(self):
        """🌈 设置时频图表"""
        self.spectrogram_ax.clear()
        self.spectrogram_ax.set_facecolor('white')
        
        # 添加知识点标注 - 修复RGBA问题
        self.spectrogram_ax.text(0.02, 0.98, '🌈 Time-Frequency Analysis', 
                                transform=self.spectrogram_ax.transAxes,
                                fontsize=12, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.5', 
                                        facecolor='plum',
                                        edgecolor='none'))
        
        self.spectrogram_ax.set_title('Spectrogram - STFT Analysis', 
                                     fontsize=16, fontweight='bold', 
                                     color='#2d3748', pad=20)
        self.spectrogram_ax.set_xlabel('Time (seconds)', fontsize=12, color='#4a5568')
        self.spectrogram_ax.set_ylabel('Frequency (Hz)', fontsize=12, color='#4a5568')
        
        # 美化坐标轴
        for spine in self.spectrogram_ax.spines.values():
            spine.set_color('#e2e8f0')
            spine.set_linewidth(1)
        self.spectrogram_ax.tick_params(colors='#718096')
        
        self.spectrogram_canvas.draw()
    
    def show_demo_visualizations(self):
        """🎭 显示演示可视化数据"""
        # 生成演示信号
        t = np.linspace(0, 3, 3000)
        demo_signal = (
            np.sin(2 * np.pi * 440 * t) +  # A4音符
            0.5 * np.sin(2 * np.pi * 880 * t) +  # A5音符
            0.3 * np.sin(2 * np.pi * 220 * t) +  # A3音符
            0.1 * np.random.randn(len(t))  # 噪声
        )
        
        # 显示演示波形
        self.plot_waveform(demo_signal, 1000, "Demo Signal")
        
        # 显示演示频谱
        self.plot_spectrum(demo_signal, 1000, "Demo Signal")
    
    def update_filter_visualization(self):
        """🔧 更新滤波器可视化"""
        try:
            # 获取滤波器参数
            low_freq = self.low_freq_slider.value()
            high_freq = self.high_freq_slider.value()
            
            # 生成频率数组
            frequencies = np.logspace(1, 4, 2000)  # 10Hz to 10kHz
            
            # 设计理想带通滤波器响应
            response = np.ones_like(frequencies)
            
            # 低频滚降 (高通特性)
            low_mask = frequencies < low_freq
            rolloff_low = 40  # dB/decade
            response[low_mask] = (frequencies[low_mask] / low_freq) ** (rolloff_low/20)
            
            # 高频滚降 (低通特性)
            high_mask = frequencies > high_freq
            rolloff_high = 40  # dB/decade
            response[high_mask] = (high_freq / frequencies[high_mask]) ** (rolloff_high/20)
            
            # 清除并重绘
            self.filter_ax.clear()
            self.filter_ax.set_facecolor('white')
            
            # 转换为dB
            magnitude_db = 20 * np.log10(response + 1e-10)
            
            # 绘制主响应曲线
            self.filter_ax.semilogx(frequencies, magnitude_db, 
                                   color='#667eea', linewidth=3, alpha=0.9,
                                   label='Filter Response')
            
            # 标记截止频率
            self.filter_ax.axvline(x=low_freq, color='#f56565', linestyle='--', 
                                  alpha=0.8, linewidth=2,
                                  label=f'Low Cutoff: {low_freq} Hz')
            self.filter_ax.axvline(x=high_freq, color='#f56565', linestyle='--', 
                                  alpha=0.8, linewidth=2,
                                  label=f'High Cutoff: {high_freq} Hz')
            
            # 填充通带区域
            passband_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            self.filter_ax.fill_between(frequencies, magnitude_db, -80, 
                                       where=passband_mask, alpha=0.2, 
                                       color='#48bb78', label='Passband')
            
            # 添加-3dB线
            self.filter_ax.axhline(y=-3, color='#ed8936', linestyle=':', 
                                  alpha=0.7, linewidth=1, label='-3dB Line')
            
            # 设置样式
            self.filter_ax.set_title('Bandpass Filter Response - LTI System Theory', 
                                    fontsize=16, fontweight='bold', 
                                    color='#2d3748', pad=20)
            self.filter_ax.set_xlabel('Frequency (Hz)', fontsize=12, color='#4a5568')
            self.filter_ax.set_ylabel('Magnitude (dB)', fontsize=12, color='#4a5568')
            self.filter_ax.grid(True, alpha=0.2, color='#a0aec0')
            self.filter_ax.set_xlim([10, 10000])
            self.filter_ax.set_ylim([-80, 5])
            
            # 图例
            self.filter_ax.legend(frameon=False, labelcolor='#2d3748', 
                                 loc='upper right', fontsize=10)
            
            # 添加知识点标注 - 修复RGBA问题
            self.filter_ax.text(0.02, 0.98, '🔧 LTI System Design', 
                               transform=self.filter_ax.transAxes,
                               fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', 
                                       facecolor='lightcoral',
                                       edgecolor='none'))
            
            # 添加技术参数标注 - 修复RGBA问题
            params_text = f"""Filter Parameters:
• Type: Butterworth Bandpass
• Order: 5th Order
• Passband: {low_freq}-{high_freq} Hz
• Rolloff: 40 dB/decade"""
            
            self.filter_ax.text(0.65, 0.05, params_text, 
                               transform=self.filter_ax.transAxes,
                               fontsize=9, fontfamily='monospace',
                               bbox=dict(boxstyle='round,pad=0.5', 
                                       facecolor='white',
                                       edgecolor='#e2e8f0'))
            
            # 美化坐标轴
            for spine in self.filter_ax.spines.values():
                spine.set_color('#e2e8f0')
                spine.set_linewidth(1)
            self.filter_ax.tick_params(colors='#718096')
            
            self.filter_canvas.draw()
            
        except Exception as e:
            print(f"🔧 滤波器可视化更新失败: {e}")

    def select_input_audio(self):
        """📁 选择输入音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "🎵 Select Mixed Audio File - Support Multiple Formats",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.aac *.ogg);;All Files (*)"
        )
        
        if file_path:
            self.input_audio_path = file_path
            filename = os.path.basename(file_path)
            
            # 更新标签显示
            self.input_label.setText(f"✅ {filename}")
            self.input_label.setStyleSheet("""
                QLabel {
                    color: #48bb78;
                    font-weight: 600;
                    background: rgba(72, 187, 120, 0.1);
                    padding: 8px 12px;
                    border-radius: 8px;
                    border: 1px solid rgba(72, 187, 120, 0.3);
                }
            """)
            
            # 更新状态栏
            self.statusbar.showMessage(f"🎵 Input audio loaded: {filename}")
            
            # 加载并分析音频
            self.load_and_analyze_audio(file_path, "input")
    
    def select_reference_audio(self):
        """🎤 选择参考音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "🎤 Select Reference Voice Audio - Target Speaker Sample",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.aac *.ogg);;All Files (*)"
        )
        
        if file_path:
            self.reference_audio_path = file_path
            filename = os.path.basename(file_path)
            
            # 更新标签显示
            self.reference_label.setText(f"✅ {filename}")
            self.reference_label.setStyleSheet("""
                QLabel {
                    color: #48bb78;
                    font-weight: 600;
                    background: rgba(72, 187, 120, 0.1);
                    padding: 8px 12px;
                    border-radius: 8px;
                    border: 1px solid rgba(72, 187, 120, 0.3);
                }
            """)
            
            # 更新状态栏
            self.statusbar.showMessage(f"🎤 Reference audio loaded: {filename}")
    
    def load_and_analyze_audio(self, file_path, audio_type):
        """🔍 加载并分析音频文件"""
        try:
            # 显示加载状态
            self.statusbar.showMessage(f"🔄 Loading and analyzing {audio_type} audio...")
            
            # 加载音频数据
            y, sr = librosa.load(file_path, sr=None, duration=30)  # 限制30秒以提高性能
            self.current_audio_data = y
            self.current_sr = sr
            
            # 显示基本信息
            duration = len(y) / sr
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            # 更新所有可视化
            self.plot_waveform(y, sr, f"{audio_type.title()} Audio")
            self.plot_spectrum(y, sr, f"{audio_type.title()} Audio")
            self.plot_spectrogram(y, sr, f"{audio_type.title()} Audio")
            
            # 更新状态信息
            status_text = (f"🎵 {audio_type.title()} audio: {duration:.2f}s, "
                          f"{sr}Hz, {file_size:.1f}MB, {len(y)} samples")
            self.statusbar.showMessage(status_text)
            
        except Exception as e:
            self.show_error_message(
                "Audio Loading Error",
                f"Failed to load {audio_type} audio file:\n\n{str(e)}\n\n"
                "Please check if the file format is supported and try again."
            )
            self.statusbar.showMessage(f"❌ Failed to load {audio_type} audio")
    
    def plot_waveform(self, y, sr, title="Audio"):
        """🌊 绘制音频波形"""
        self.waveform_ax.clear()
        self.waveform_ax.set_facecolor('white')
        
        # 创建时间轴
        time = np.linspace(0, len(y) / sr, len(y))
        
        # 绘制波形 - 使用渐变色
        self.waveform_ax.plot(time, y, color='#667eea', linewidth=0.8, alpha=0.8)
        
        # 添加RMS包络
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = frame_length // 4
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # 插值到原始时间轴
        rms_interp = np.interp(time, times, rms)
        self.waveform_ax.plot(time, rms_interp, color='#f56565', linewidth=2, 
                             alpha=0.7, label='RMS Envelope')
        self.waveform_ax.plot(time, -rms_interp, color='#f56565', linewidth=2, alpha=0.7)
        
        # 设置样式
        self.waveform_ax.set_title(f'{title} - Time Domain Analysis', 
                                  fontsize=16, fontweight='bold', 
                                  color='#2d3748', pad=20)
        self.waveform_ax.set_xlabel('Time (seconds)', fontsize=12, color='#4a5568')
        self.waveform_ax.set_ylabel('Amplitude', fontsize=12, color='#4a5568')
        self.waveform_ax.grid(True, alpha=0.2, color='#a0aec0')
        
        # 添加统计信息 - 修复RGBA问题
        rms_value = np.sqrt(np.mean(y**2))
        peak_value = np.max(np.abs(y))
        zero_crossings = librosa.zero_crossings(y).sum()
        
        stats_text = f"""Audio Statistics:
• RMS: {rms_value:.4f}
• Peak: {peak_value:.4f}
• Duration: {len(y)/sr:.2f}s
• Zero Crossings: {zero_crossings}"""
        
        self.waveform_ax.text(0.02, 0.02, stats_text,
                             transform=self.waveform_ax.transAxes,
                             fontsize=9, fontfamily='monospace',
                             bbox=dict(boxstyle='round,pad=0.5',
                                     facecolor='white',
                                     edgecolor='#e2e8f0'))
        
        # 知识点标注 - 修复RGBA问题
        self.waveform_ax.text(0.02, 0.98, '📈 Time Domain Analysis',
                             transform=self.waveform_ax.transAxes,
                             fontsize=12, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.5',
                                     facecolor='lightblue',
                                     edgecolor='none'))
        
        # 美化坐标轴
        for spine in self.waveform_ax.spines.values():
            spine.set_color('#e2e8f0')
        self.waveform_ax.tick_params(colors='#718096')
        
        self.waveform_canvas.draw()
    
    def plot_spectrum(self, y, sr, title="Audio"):
        """📊 绘制频谱图"""
        self.spectrum_ax.clear()
        self.spectrum_ax.set_facecolor('white')
        
        # 计算FFT
        n_fft = 2048
        Y = np.fft.fft(y, n=n_fft)
        freqs = np.fft.fftfreq(n_fft, 1/sr)
        
        # 只取正频率部分
        positive_freqs = freqs[:n_fft//2]
        magnitude = np.abs(Y[:n_fft//2])
        
        # 转换为dB，添加平滑
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # 平滑处理
        from scipy import signal
        magnitude_db_smooth = signal.savgol_filter(magnitude_db, 51, 3)
        
        # 绘制频谱
        self.spectrum_ax.plot(positive_freqs, magnitude_db_smooth, 
                             color='#48bb78', linewidth=1.5, alpha=0.9)
        
        # 标记重要频率点
        # 找到峰值
        peaks, _ = signal.find_peaks(magnitude_db_smooth, height=-20, distance=50)
        if len(peaks) > 0:
            peak_freqs = positive_freqs[peaks[:5]]  # 显示前5个峰值
            peak_mags = magnitude_db_smooth[peaks[:5]]
            
            self.spectrum_ax.scatter(peak_freqs, peak_mags, 
                                   color='#f56565', s=50, alpha=0.8,
                                   zorder=5, label='Spectral Peaks')
            
            # 标注主要峰值 - 修复RGBA问题
            for freq, mag in zip(peak_freqs[:3], peak_mags[:3]):
                self.spectrum_ax.annotate(f'{freq:.0f}Hz', 
                                        xy=(freq, mag), xytext=(10, 10),
                                        textcoords='offset points',
                                        fontsize=9, color='#2d3748',
                                        bbox=dict(boxstyle='round,pad=0.3',
                                                facecolor='yellow',
                                                alpha=0.7))
        
        # 标记人声频率范围
        self.spectrum_ax.axvspan(85, 255, alpha=0.1, color='#667eea', 
                                label='Fundamental Voice Range')
        self.spectrum_ax.axvspan(255, 2000, alpha=0.1, color='#48bb78', 
                                label='Voice Harmonics')
        
        # 设置样式
        self.spectrum_ax.set_title(f'{title} - Frequency Domain Analysis', 
                                  fontsize=16, fontweight='bold', 
                                  color='#2d3748', pad=20)
        self.spectrum_ax.set_xlabel('Frequency (Hz)', fontsize=12, color='#4a5568')
        self.spectrum_ax.set_ylabel('Magnitude (dB)', fontsize=12, color='#4a5568')
        self.spectrum_ax.grid(True, alpha=0.2, color='#a0aec0')
        self.spectrum_ax.set_xlim([0, sr//2])
        self.spectrum_ax.legend(frameon=False, labelcolor='#2d3748', fontsize=9)
        
        # 知识点标注 - 修复RGBA问题
        self.spectrum_ax.text(0.02, 0.98, '📊 FFT Analysis',
                             transform=self.spectrum_ax.transAxes,
                             fontsize=12, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.5',
                                     facecolor='lightgreen',
                                     edgecolor='none'))
        
        # 美化坐标轴
        for spine in self.spectrum_ax.spines.values():
            spine.set_color('#e2e8f0')
        self.spectrum_ax.tick_params(colors='#718096')
        
        self.spectrum_canvas.draw()
    
    def plot_spectrogram(self, y, sr, title="Audio"):
        """🌈 绘制时频图"""
        try:
            self.spectrogram_ax.clear()
            self.spectrogram_ax.set_facecolor('white')
            
            # 计算STFT
            n_fft = 2048
            hop_length = 512
            D = librosa.amplitude_to_db(
                np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), 
                ref=np.max
            )
            
            # 显示时频图
            img = librosa.display.specshow(
                D, y_axis='hz', x_axis='time', sr=sr,
                hop_length=hop_length, ax=self.spectrogram_ax, 
                cmap='viridis', alpha=0.8
            )
            
            # 设置样式
            self.spectrogram_ax.set_title(f'{title} - Time-Frequency Analysis', 
                                         fontsize=16, fontweight='bold', 
                                         color='#2d3748', pad=20)
            self.spectrogram_ax.set_xlabel('Time (seconds)', fontsize=12, color='#4a5568')
            self.spectrogram_ax.set_ylabel('Frequency (Hz)', fontsize=12, color='#4a5568')
            
            # 添加颜色条
            cbar = plt.colorbar(img, ax=self.spectrogram_ax, format='%+2.0f dB',
                               shrink=0.8, aspect=30)
            cbar.set_label('Magnitude (dB)', fontsize=10, color='#4a5568')
            cbar.ax.tick_params(colors='#718096')
            
            # 知识点标注 - 修复RGBA问题
            self.spectrogram_ax.text(0.02, 0.98, '🌈 STFT Analysis',
                                    transform=self.spectrogram_ax.transAxes,
                                    fontsize=12, fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.5',
                                            facecolor='plum',
                                            edgecolor='none'))
            
            # 美化坐标轴
            for spine in self.spectrogram_ax.spines.values():
                spine.set_color('#e2e8f0')
            self.spectrogram_ax.tick_params(colors='#718096')
            
            self.spectrogram_canvas.draw()
            
        except Exception as e:
            print(f"🌈 时频图绘制失败: {e}")
    
    def start_processing(self):
        """🚀 开始音频处理"""
        # 验证输入
        if not self.input_audio_path:
            self.show_warning_message(
                "Missing Input File", 
                "Please select a mixed audio file before processing."
            )
            return
        
        if not self.reference_audio_path:
            self.show_warning_message(
                "Missing Reference File", 
                "Please select a reference voice audio file before processing."
            )
            return
        
        # 禁用处理按钮并更新UI
        self.process_button.setEnabled(False)
        self.process_button.setText("🔄 Processing...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 添加处理动画效果
        self.animate_processing_start()
        
        # 获取处理参数
        low_freq = self.low_freq_slider.value()
        high_freq = self.high_freq_slider.value()
        
        # 启动处理线程
        self.worker_thread = WorkerThread(
            self.input_audio_path, 
            self.reference_audio_path,
            low_freq, 
            high_freq
        )
        
        # 连接线程信号
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self.statusbar.showMessage)
        self.worker_thread.finished_processing.connect(self.on_processing_finished)
        self.worker_thread.error_occurred.connect(self.on_processing_error)
        
        # 启动线程
        self.worker_thread.start()
    
    def animate_processing_start(self):
        """🎬 处理开始动画"""
        # 进度条淡入动画
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(226, 232, 240, 0.3);
                border: none;
                border-radius: 8px;
                text-align: center;
                color: #2d3748;
                font-weight: bold;
                height: 16px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                           stop:0 #667eea, stop:1 #764ba2);
                border-radius: 8px;
            }
        """)
    
    def on_processing_finished(self, output_path):
        """✅ 处理完成回调"""
        # 恢复按钮状态
        self.process_button.setEnabled(True)
        self.process_button.setText("🚀 Start Voice Extraction")
        self.progress_bar.setVisible(False)
        self.play_processed_button.setEnabled(True)
        
        # 保存输出路径
        self.processed_audio_path = output_path
        
        # 显示成功动画
        self.animate_processing_success()
        
        # 如果有实际输出，加载显示
        if os.path.exists(output_path):
            self.load_and_analyze_audio(output_path, "processed")
        else:
            # 演示模式：显示滤波后的结果
            if self.current_audio_data is not None:
                filtered_audio = self.apply_demo_filter(
                    self.current_audio_data, 
                    self.current_sr
                )
                self.plot_waveform(filtered_audio, self.current_sr, "Processed Audio (Demo)")
                self.plot_spectrum(filtered_audio, self.current_sr, "Processed Audio (Demo)")
        
        # 显示成功消息
        self.show_success_message(
            "🎉 Processing Complete!",
            f"""Voice extraction completed successfully!

📁 Output saved to: {output_path}

🔬 Applied Technologies:
• Neural network source separation
• MFCC-based speaker identification  
• LTI bandpass filter design
• FFT frequency domain processing
• Convolution-based enhancement

🎯 Results:
• Target speaker voice extracted
• Background noise reduced
• Audio quality enhanced"""
        )
    
    def apply_demo_filter(self, y, sr):
        """🔧 应用演示滤波器"""
        try:
            from scipy import signal
            
            # 获取滤波参数
            low_freq = self.low_freq_slider.value()
            high_freq = self.high_freq_slider.value()
            
            # 设计巴特沃斯带通滤波器
            nyquist = sr / 2
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist
            
            # 确保频率在有效范围内
            low_norm = max(0.01, min(0.99, low_norm))
            high_norm = max(0.01, min(0.99, high_norm))
            
            if low_norm >= high_norm:
                high_norm = min(0.99, low_norm + 0.1)
            
            # 设计5阶巴特沃斯滤波器
            b, a = signal.butter(5, [low_norm, high_norm], btype='band')
            
            # 应用零相位滤波
            filtered = signal.filtfilt(b, a, y)
            
            return filtered
            
        except Exception as e:
            print(f"🔧 演示滤波器应用失败: {e}")
            return y  # 返回原始信号
    
    def animate_processing_success(self):
        """🎉 处理成功动画"""
        # 按钮成功颜色动画
        self.process_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #48bb78, stop:1 #38a169);
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: 700;
                color: white;
                min-height: 20px;
            }
        """)
        
        # 2秒后恢复原始样式
        QTimer.singleShot(2000, self.restore_button_style)
    
    def restore_button_style(self):
        """恢复按钮原始样式"""
        self.process_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #667eea, stop:1 #764ba2);
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: 700;
                color: white;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #5a67d8, stop:1 #6b46c1);
            }
        """)
    
    def on_processing_error(self, error_message):
        """❌ 处理错误回调"""
        # 恢复按钮状态
        self.process_button.setEnabled(True)
        self.process_button.setText("🚀 Start Voice Extraction")
        self.progress_bar.setVisible(False)
        
        # 显示错误消息
        self.show_error_message("Processing Error", error_message)
        
        # 更新状态栏
        self.statusbar.showMessage("❌ Processing failed - Please check inputs and try again")
    
    def play_original_audio(self):
        """🎵 播放原始音频"""
        if self.input_audio_path:
            self.statusbar.showMessage("🎵 Audio playback feature coming soon...")
        else:
            self.show_warning_message(
                "No Audio Selected", 
                "Please select an input audio file first."
            )
    
    def play_processed_audio(self):
        """🎵 播放处理后音频"""
        if self.processed_audio_path:
            self.statusbar.showMessage("🎵 Processed audio playback feature coming soon...")
        else:
            self.show_warning_message(
                "No Processed Audio", 
                "Please complete the processing first."
            )
    
    def on_tab_changed(self, index):
        """📊 标签页切换事件"""
        tab_names = ["📈 Waveform", "📊 Spectrum", "🔧 Filter Response", "🌈 Spectrogram"]
        if index < len(tab_names):
            self.statusbar.showMessage(f"Viewing {tab_names[index]} analysis")
    
    def show_welcome_state(self):
        """🎉 显示欢迎状态"""
        welcome_message = (
            "🎵 Welcome to Modern Voice Enhancement System! "
            "Select audio files to begin signal processing analysis."
        )
        self.statusbar.showMessage(welcome_message)
    
    def show_success_message(self, title, message):
        """✅ 显示成功消息"""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStyleSheet("""
            QMessageBox {
                background: white;
                border-radius: 16px;
                font-size: 14px;
                color: #2d3748;
            }
            QMessageBox QLabel {
                color: #2d3748;
                font-size: 14px;
            }
            QMessageBox QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #48bb78, stop:1 #38a169);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
                font-weight: 600;
                min-width: 80px;
            }
            QMessageBox QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #38a169, stop:1 #2f855a);
            }
        """)
        msg_box.exec()
    
    def show_warning_message(self, title, message):
        """⚠️ 显示警告消息"""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStyleSheet("""
            QMessageBox {
                background: white;
                border-radius: 16px;
                font-size: 14px;
                color: #2d3748;
            }
            QMessageBox QLabel {
                color: #2d3748;
                font-size: 14px;
            }
            QMessageBox QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #ed8936, stop:1 #dd6b20);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
                font-weight: 600;
                min-width: 80px;
            }
            QMessageBox QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #dd6b20, stop:1 #c05621);
            }
        """)
        msg_box.exec()
    
    def show_error_message(self, title, message):
        """❌ 显示错误消息"""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStyleSheet("""
            QMessageBox {
                background: white;
                border-radius: 16px;
                font-size: 14px;
                color: #2d3748;
            }
            QMessageBox QLabel {
                color: #2d3748;
                font-size: 14px;
            }
            QMessageBox QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #f56565, stop:1 #e53e3e);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
                font-weight: 600;
                min-width: 80px;
            }
            QMessageBox QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #e53e3e, stop:1 #c53030);
            }
        """)
        msg_box.exec()

def main():
    """🚀 主程序入口"""
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("Modern Voice Enhancement System")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Signal Processing Course Project")
    
    # 启用高DPI支持 - 修复PyQt6属性名称
    try:
        # PyQt6 中的高DPI支持
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
            app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        # 在PyQt6中，高DPI缩放是默认启用的
        print("🎨 High DPI scaling is enabled by default in PyQt6")
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
