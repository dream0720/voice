#!/usr/bin/env python3
"""
Modern Voice Processing GUI Application
======================================

A comprehensive GUI application for voice processing with four main functions:
1. Audio Preprocessing - Noise reduction and filtering
2. Source Separation - Music source separation using Demucs
3. Speaker Separation - Speaker diarization and separation
4. Voice Matching - Reference-based voice matching

Features:
- Modern, intuitive interface
- Real-time console output redirection
- Progress tracking and visualization
- Independent processing modules
- Comprehensive results display
"""

import sys
import os
import io
import traceback
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QPushButton, QTextEdit, QProgressBar, 
    QFileDialog, QMessageBox, QTabWidget, QGroupBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QSplitter, QFrame, QScrollArea
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, 
    QRect, QSize, QUrl
)
from PyQt6.QtGui import (
    QFont, QPixmap, QPalette, QColor, QIcon, QDesktopServices,
    QPainter, QLinearGradient
)

# Matplotlib imports
import matplotlib
matplotlib.use('Qt5Agg')  # 确保使用Qt后端
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
plt.style.use('default')

# Project imports
from modules.preprocessing import AudioPreprocessor
from modules.source_separation import DemucsSourceSeparator
from modules.speaker_separation import SpeakerSeparator
from modules.voice_matching import VoiceMatcher
from modules.utils import PathManager, Logger, validate_audio_file, get_supported_audio_formats


class ConsoleRedirector(io.StringIO):
    """Console output redirector for capturing print statements"""
    
    def __init__(self, text_widget, original_stream=None):
        super().__init__()
        self.text_widget = text_widget
        self.buffer = queue.Queue()
        self.original_stream = original_stream
        
    def write(self, text):
        if text.strip():  # Only add non-empty text
            self.buffer.put(text)
            # Also write to original stream (console)
            if self.original_stream:
                self.original_stream.write(text)
                self.original_stream.flush()
        return len(text)
    
    def flush(self):
        if self.original_stream:
            self.original_stream.flush()


class ProcessingThread(QThread):
    """Generic processing thread for background operations"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        
    def run(self):
        try:
            self.result = self.func(*self.args, **self.kwargs)
            self.result_ready.emit(self.result)
        except Exception as e:
            error_msg = f"Processing error: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)


class MatplotlibWidget(QWidget):
    """Custom matplotlib widget for displaying plots"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Style the figure
        self.figure.patch.set_facecolor('#f0f0f0')
    
    def clear(self):
        """Clear the plot"""
        self.figure.clear()
        self.canvas.draw()
    def plot_preprocessing_results(self, results: Dict[str, Any]):
        """Plot preprocessing analysis results"""
        if not results or 'original_analysis' not in results or 'processed_analysis' not in results:
            return
            
        # Clear previous plots
        self.figure.clear()
        
        try:
            # Get data from results
            original_audio = results['original_audio']
            processed_audio = results['processed_audio']
            original_analysis = results['original_analysis']
            processed_analysis = results['processed_analysis']
            sample_rate = 16000  # Default sample rate
            
            # Create subplots to match the source figure
            axes = self.figure.subplots(3, 2)
            self.figure.suptitle('Audio Preprocessing Analysis', fontsize=16, fontweight='bold')
            
            # Time domain comparison
            time_orig = np.arange(len(original_audio)) / sample_rate
            time_proc = np.arange(len(processed_audio)) / sample_rate
            
            axes[0, 0].plot(time_orig, original_audio, alpha=0.7, color='blue')
            axes[0, 0].set_title('Original Audio - Time Domain')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(time_proc, processed_audio, alpha=0.7, color='red')
            axes[0, 1].set_title('Processed Audio - Time Domain')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Amplitude')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Frequency domain comparison
            orig_freqs = original_analysis['frequencies'][:len(original_audio)//2]
            orig_mag = original_analysis['magnitude_spectrum'][:len(original_audio)//2]
            proc_freqs = processed_analysis['frequencies'][:len(processed_audio)//2]
            proc_mag = processed_analysis['magnitude_spectrum'][:len(processed_audio)//2]
            
            axes[1, 0].semilogy(orig_freqs, orig_mag, alpha=0.7, color='blue')
            axes[1, 0].set_title('Original Audio - Frequency Spectrum')
            axes[1, 0].set_xlabel('Frequency (Hz)')
            axes[1, 0].set_ylabel('Magnitude')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xlim(0, 8000)
            
            axes[1, 1].semilogy(proc_freqs, proc_mag, alpha=0.7, color='red')
            axes[1, 1].set_title('Processed Audio - Frequency Spectrum')
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_ylabel('Magnitude')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xlim(0, 8000)
            
            # Energy distribution comparison
            categories = ['Low\\n(0-1kHz)', 'Mid\\n(1-4kHz)', 'High\\n(4kHz+)']
            orig_ratios = [original_analysis['low_freq_energy_ratio'], 
                          original_analysis['mid_freq_energy_ratio'],
                          original_analysis['high_freq_energy_ratio']]
            proc_ratios = [processed_analysis['low_freq_energy_ratio'],
                          processed_analysis['mid_freq_energy_ratio'], 
                          processed_analysis['high_freq_energy_ratio']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes[2, 0].bar(x - width/2, [r*100 for r in orig_ratios], width, 
                          label='Original', alpha=0.7, color='blue')
            axes[2, 0].bar(x + width/2, [r*100 for r in proc_ratios], width,
                          label='Processed', alpha=0.7, color='red')
            axes[2, 0].set_title('Energy Distribution Comparison')
            axes[2, 0].set_xlabel('Frequency Band')
            axes[2, 0].set_ylabel('Energy Percentage (%)')
            axes[2, 0].set_xticks(x)
            axes[2, 0].set_xticklabels(categories)
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # Spectrogram
            axes[2, 1].specgram(processed_audio, Fs=sample_rate, cmap='viridis')
            axes[2, 1].set_title('Processed Audio - Spectrogram')
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].set_ylabel('Frequency (Hz)')
            
        except Exception as e:
            # Fallback: display error message
            self.figure.text(0.5, 0.5, f'Error displaying results: {str(e)}', 
                           horizontalalignment='center', verticalalignment='center')
        
        self.figure.tight_layout()
        self.canvas.draw()


class ProcessingCard(QFrame):
    """Individual processing card widget"""
    
    def __init__(self, title: str, description: str, card_type: str = "default", icon_path: str = None):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.95);
                border: 2px solid rgba(226, 232, 240, 0.8);
                border-radius: 16px;
                padding: 20px;
                backdrop-filter: blur(20px);
            }
            QFrame:hover {
                border-color: rgba(102, 126, 234, 0.8);
                background: rgba(255, 255, 255, 0.98);
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
            }
        """)
        
        self.title = title
        self.description = description
        self.card_type = card_type
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the card UI"""
        layout = QVBoxLayout()
        
        # Title with icon
        title_layout = QHBoxLayout()
        title_label = QLabel(self.title)
        title_label.setFont(QFont("SF Pro Display", 16, QFont.Weight.Bold))
        title_label.setStyleSheet("""
            color: #2d3748;
            margin-bottom: 8px;
            padding: 0;
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        # Description
        desc_label = QLabel(self.description)
        desc_label.setFont(QFont("SF Pro Display", 11))
        desc_label.setStyleSheet("""
            color: #4a5568;
            margin-bottom: 16px;
            line-height: 1.4;
        """)
        desc_label.setWordWrap(True)
        
        # File selection area
        file_area = QFrame()
        file_area.setStyleSheet("""
            QFrame {
                background: rgba(247, 250, 252, 0.8);
                border: 1px solid rgba(226, 232, 240, 0.6);
                border-radius: 8px;
                padding: 12px;
                margin: 8px 0;
            }
        """)
        file_layout = QVBoxLayout(file_area)
        
        if self.card_type == "voice_matching":
            # Special layout for voice matching
            self.setup_voice_matching_ui(file_layout)
        else:
            # Standard file selection
            self.setup_standard_ui(file_layout)
        
        # Process button
        self.process_button = QPushButton("🚀 开始处理")
        self.process_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 10px;
                font-weight: 600;
                font-size: 13px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #5a67d8, stop:1 #6b46c1);
                transform: translateY(-1px);
            }
            QPushButton:pressed {
                transform: translateY(0);
            }
            QPushButton:disabled {
                background: rgba(160, 174, 192, 0.6);
                color: rgba(255, 255, 255, 0.8);
            }
        """)
        self.process_button.setEnabled(False)
        
        # Results section
        results_layout = QHBoxLayout()
        
        self.view_results_button = QPushButton("📊 查看结果")
        self.view_results_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #f093fb, stop:1 #f5576c);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #e879f9, stop:1 #ef4444);
            }
            QPushButton:disabled {
                background: rgba(160, 174, 192, 0.4);
                color: rgba(255, 255, 255, 0.6);
            }
        """)
        self.view_results_button.setEnabled(False)
        
        self.open_folder_button = QPushButton("📁 打开文件夹")
        self.open_folder_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #4facfe, stop:1 #00f2fe);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #3b82f6, stop:1 #06b6d4);
            }
            QPushButton:disabled {
                background: rgba(160, 174, 192, 0.4);
                color: rgba(255, 255, 255, 0.6);
            }
        """)
        self.open_folder_button.setEnabled(False)
        
        results_layout.addWidget(self.view_results_button)
        results_layout.addWidget(self.open_folder_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 8px;
                text-align: center;
                background: rgba(226, 232, 240, 0.6);
                color: #2d3748;
                font-weight: 500;
                height: 16px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                           stop:0 #667eea, stop:1 #764ba2);
                border-radius: 8px;
            }
        """)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("""
            color: #4a5568;
            font-size: 11px;
            font-weight: 500;
            padding: 4px 0;
        """)
        self.status_label.setVisible(False)
        
        # Add widgets to layout
        layout.addLayout(title_layout)
        layout.addWidget(desc_label)
        layout.addWidget(file_area)
        layout.addWidget(self.process_button)
        layout.addLayout(results_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        # Store for later use
        self.selected_file = None
        self.output_dir = None
        self.processing_results = None
    
    def setup_standard_ui(self, layout):
        """Setup standard file selection UI"""
        file_label = QLabel("📄 输入文件:")
        file_label.setFont(QFont("SF Pro Display", 11, QFont.Weight.Medium))
        file_label.setStyleSheet("color: #4a5568; margin-bottom: 4px;")
        
        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("""
            color: #a0aec0;
            font-style: italic;
            font-size: 11px;
            padding: 4px 0;
        """)
        
        self.select_button = QPushButton("🎵 选择音频文件")
        self.select_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #4facfe, stop:1 #00f2fe);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #3b82f6, stop:1 #06b6d4);
            }
        """)
        
        layout.addWidget(file_label)
        layout.addWidget(self.file_label)
        layout.addWidget(self.select_button)
    
    def setup_voice_matching_ui(self, layout):
        """Setup voice matching specific UI with two buttons"""
        # Reference audio section
        ref_label = QLabel("🎯 参考音频:")
        ref_label.setFont(QFont("SF Pro Display", 11, QFont.Weight.Medium))
        ref_label.setStyleSheet("color: #4a5568; margin-bottom: 4px;")
        
        self.reference_label = QLabel("未选择参考音频")
        self.reference_label.setStyleSheet("""
            color: #a0aec0;
            font-style: italic;
            font-size: 11px;
            padding: 4px 0;
        """)
        
        self.select_reference_button = QPushButton("🎵 选择参考音频")
        self.select_reference_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 12px;
                margin-bottom: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #5a67d8, stop:1 #6b46c1);
            }
        """)
        
        # Candidate audio section
        candidate_label = QLabel("🎭 待匹配音频:")
        candidate_label.setFont(QFont("SF Pro Display", 11, QFont.Weight.Medium))
        candidate_label.setStyleSheet("color: #4a5568; margin-bottom: 4px;")
        
        self.candidate_label = QLabel("未选择待匹配音频")
        self.candidate_label.setStyleSheet("""
            color: #a0aec0;
            font-style: italic;
            font-size: 11px;
            padding: 4px 0;
        """)
        
        self.select_candidates_button = QPushButton("🎪 选择待匹配音频")
        self.select_candidates_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #f093fb, stop:1 #f5576c);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #e879f9, stop:1 #ef4444);
            }
        """)
        
        layout.addWidget(ref_label)
        layout.addWidget(self.reference_label)
        layout.addWidget(self.select_reference_button)
        layout.addWidget(candidate_label)
        layout.addWidget(self.candidate_label)
        layout.addWidget(self.select_candidates_button)
        
        # Initialize attributes for voice matching
        self.reference_file = None
        self.candidate_files = []


class VoiceProcessingApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎵 Voice Processing Suite - 先进语音处理系统")
        self.setGeometry(100, 100, 1600, 1000)
        self.setMinimumSize(1400, 900)
        
        # Initialize processing modules
        self.preprocessor = AudioPreprocessor()
        self.source_separator = DemucsSourceSeparator()
        self.speaker_separator = SpeakerSeparator(hf_token="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")
        self.voice_matcher = VoiceMatcher()
        
        # Setup console redirection
        self.setup_console_redirection()
        
        # Setup UI
        self.setup_ui()
        
        # Setup timer for console updates
        self.console_timer = QTimer()
        self.console_timer.timeout.connect(self.update_console)
        self.console_timer.start(100)  # Update every 100ms
          # Current processing thread
        self.current_thread = None
        
        print("🚀 Voice Processing Suite 初始化成功！")
        print("📋 功能模块：")
        print("   1. 🎛️ 音频预处理 - 噪声消除和信号增强")
        print("   2. 🎼 音源分离 - 分离音乐和人声")
        print("   3. 👥 说话人分离 - 多说话人识别分离")
        print("   4. 🎯 人声匹配 - 基于参考音频的人声匹配")
        print("=" * 60)
    
    def setup_console_redirection(self):
        """Setup console output redirection"""
        self.console_buffer = queue.Queue()
        
        # Keep references to original streams
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create redirectors that write to both GUI and console
        self.stdout_redirector = ConsoleRedirector(None, self.original_stdout)
        self.stderr_redirector = ConsoleRedirector(None, self.original_stderr)
          # Redirect stdout and stderr
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stderr_redirector
    
    def setup_ui(self):
        """Setup the main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Set modern background
        central_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #667eea, stop:0.3 #764ba2, 
                                           stop:0.7 #f093fb, stop:1 #f5576c);
            }
        """)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Processing cards
        left_panel = QScrollArea()
        left_panel.setWidgetResizable(True)
        left_panel.setMaximumWidth(900)
        left_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_panel.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)
        
        left_content = QWidget()
        left_layout = QVBoxLayout()
        left_content.setLayout(left_layout)
        left_panel.setWidget(left_content)
        
        # Title with modern styling
        title_label = QLabel("🎵 Voice Processing Suite")
        title_label.setFont(QFont("SF Pro Display", 28, QFont.Weight.Bold))
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                margin: 30px 20px;
                text-align: center;
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 16px;
                backdrop-filter: blur(20px);
            }
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title_label)
        
        # Processing cards in 2x2 grid
        cards_widget = QWidget()
        cards_widget.setStyleSheet("background: transparent;")
        cards_layout = QGridLayout()
        cards_widget.setLayout(cards_layout)
        cards_layout.setSpacing(20)
        
        # Create processing cards with modern styling
        self.preprocessing_card = ProcessingCard(
            "🎛️ 音频预处理",
            "应用噪声消除、滤波和信号增强，使用FFT分析和频谱减法提升音频质量。",
            "standard"
        )
        
        self.source_separation_card = ProcessingCard(
            "🎼 音源分离",
            "使用先进的Demucs模型将混合音频分离为独立的音轨（人声、鼓声、贝斯、其他）。",
            "standard"
        )
        
        self.speaker_separation_card = ProcessingCard(
            "👥 说话人分离",
            "使用先进的说话人分离技术从多说话人音频中识别并分离出不同的说话人。",
            "standard"
        )
        
        self.voice_matching_card = ProcessingCard(
            "🎯 人声匹配",
            "基于参考音频特征，从分离的音频中匹配和识别最佳匹配的人声。",
            "voice_matching"
        )
        
        # Add cards to grid
        cards_layout.addWidget(self.preprocessing_card, 0, 0)
        cards_layout.addWidget(self.source_separation_card, 0, 1)
        cards_layout.addWidget(self.speaker_separation_card, 1, 0)
        cards_layout.addWidget(self.voice_matching_card, 1, 1)
        
        left_layout.addWidget(cards_widget)
        left_layout.addStretch()
        
        # Connect card signals
        self.setup_card_connections()
        
        # Right panel - Visualization and Console (交换位置)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        right_panel.setStyleSheet("background: transparent;")
        
        # Visualization area (移到上方)
        viz_group = QGroupBox("📊 数据可视化")
        viz_group.setFont(QFont("SF Pro Display", 14, QFont.Weight.Bold))
        viz_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 16px;
                margin-top: 16px;
                padding-top: 20px;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 10px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 8px;
            }
        """)
        viz_layout = QVBoxLayout()
        viz_group.setLayout(viz_layout)
        
        self.matplotlib_widget = MatplotlibWidget()
        self.matplotlib_widget.setStyleSheet("""
            QWidget {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                margin: 10px;
            }
        """)
        viz_layout.addWidget(self.matplotlib_widget)
        
        right_layout.addWidget(viz_group, 2)  # 给可视化更多空间
        
        # Console output (移到下方)
        console_group = QGroupBox("💻 控制台输出")
        console_group.setFont(QFont("SF Pro Display", 14, QFont.Weight.Bold))
        console_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 16px;
                margin-top: 16px;
                padding-top: 20px;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 10px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 8px;
            }
        """)
        console_layout = QVBoxLayout()
        console_group.setLayout(console_layout)
        
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setMaximumHeight(300)
        self.console_output.setStyleSheet("""
            QTextEdit {
                background: rgba(26, 32, 44, 0.95);
                color: #68d391;
                font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
                font-size: 11px;
                border: none;
                border-radius: 12px;
                padding: 16px;
                line-height: 1.4;
                selection-background-color: rgba(104, 211, 145, 0.3);
            }
        """)
        console_layout.addWidget(self.console_output)
        
        # Console controls
        console_controls = QHBoxLayout()
        
        clear_button = QPushButton("🗑️ 清空控制台")
        clear_button.clicked.connect(self.console_output.clear)
        clear_button.setStyleSheet("""
            QPushButton {
                background: rgba(239, 68, 68, 0.8);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 12px;
            }
            QPushButton:hover {
                background: rgba(220, 38, 38, 0.9);
            }
        """)
        
        save_log_button = QPushButton("💾 保存日志")
        save_log_button.clicked.connect(self.save_console_log)
        save_log_button.setStyleSheet("""
            QPushButton {
                background: rgba(59, 130, 246, 0.8);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 12px;
            }
            QPushButton:hover {
                background: rgba(37, 99, 235, 0.9);
            }
        """)
        
        console_controls.addWidget(clear_button)
        console_controls.addWidget(save_log_button)
        console_controls.addStretch()
        
        console_layout.addLayout(console_controls)
        
        right_layout.addWidget(console_group, 1)  # 给控制台较少空间
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 1)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #667eea, stop:0.3 #764ba2, 
                                           stop:0.7 #f093fb, stop:1 #f5576c);
            }
        """)
    
    def save_console_log(self):
        """Save console log to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存控制台日志",
            f"console_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.console_output.toPlainText())
                print(f"✅ 控制台日志已保存到: {file_path}")
            except Exception as e:
                print(f"❌ 保存日志失败: {str(e)}")
                QMessageBox.warning(self, "保存失败", f"无法保存日志文件:\n{str(e)}")
    
    def setup_card_connections(self):
        """Setup connections for processing cards"""
        # Preprocessing card
        self.preprocessing_card.select_button.clicked.connect(
            lambda: self.select_file(self.preprocessing_card)
        )
        self.preprocessing_card.process_button.clicked.connect(
            lambda: self.process_preprocessing()
        )
        self.preprocessing_card.view_results_button.clicked.connect(
            lambda: self.view_preprocessing_results()
        )
        self.preprocessing_card.open_folder_button.clicked.connect(
            lambda: self.open_output_folder(self.preprocessing_card)
        )
        
        # Source separation card
        self.source_separation_card.select_button.clicked.connect(
            lambda: self.select_file(self.source_separation_card)
        )
        self.source_separation_card.process_button.clicked.connect(
            lambda: self.process_source_separation()
        )
        self.source_separation_card.view_results_button.clicked.connect(
            lambda: self.view_source_separation_results()
        )
        self.source_separation_card.open_folder_button.clicked.connect(
            lambda: self.open_output_folder(self.source_separation_card)
        )
        
        # Speaker separation card
        self.speaker_separation_card.select_button.clicked.connect(
            lambda: self.select_file(self.speaker_separation_card)
        )
        self.speaker_separation_card.process_button.clicked.connect(
            lambda: self.process_speaker_separation()
        )
        self.speaker_separation_card.view_results_button.clicked.connect(
            lambda: self.view_speaker_separation_results()
        )
        self.speaker_separation_card.open_folder_button.clicked.connect(
            lambda: self.open_output_folder(self.speaker_separation_card)        )
        
        # Voice matching card - 修改为两个独立的按钮
        self.voice_matching_card.select_reference_button.clicked.connect(
            lambda: self.select_reference_audio()
        )
        self.voice_matching_card.select_candidates_button.clicked.connect(
            lambda: self.select_candidate_audio()
        )
        self.voice_matching_card.process_button.clicked.connect(
            lambda: self.process_voice_matching()
        )
        self.voice_matching_card.view_results_button.clicked.connect(
            lambda: self.view_voice_matching_results()
        )
        self.voice_matching_card.open_folder_button.clicked.connect(
            lambda: self.open_output_folder(self.voice_matching_card)
        )
    
    def select_file(self, card: ProcessingCard):
        """Select input file for processing"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"选择音频文件进行{card.title}",
            "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a);;所有文件 (*)"
        )
        
        if file_path:
            if validate_audio_file(file_path):
                card.selected_file = file_path
                card.file_label.setText(Path(file_path).name)
                card.file_label.setStyleSheet("color: #48bb78; font-weight: 600;")
                card.process_button.setEnabled(True)
                print(f"✅ 已为{card.title}选择文件: {Path(file_path).name}")
            else:
                QMessageBox.warning(self, "无效文件", "所选文件不是有效的音频文件。")
    
    def select_reference_audio(self):
        """Select reference audio file for voice matching"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择参考音频文件",
            "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a);;所有文件 (*)"
        )
        
        if file_path:
            if validate_audio_file(file_path):
                self.voice_matching_card.reference_file = file_path
                self.voice_matching_card.reference_label.setText(f"📄 {Path(file_path).name}")
                self.voice_matching_card.reference_label.setStyleSheet("color: #48bb78; font-weight: 600;")
                print(f"✅ 已选择参考音频: {Path(file_path).name}")
                self.check_voice_matching_ready()
            else:
                QMessageBox.warning(self, "无效文件", "所选文件不是有效的音频文件。")
    
    def select_candidate_audio(self):
        """Select candidate audio files for voice matching"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择待匹配音频文件（可多选）",
            "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a);;所有文件 (*)"
        )
        
        if files:
            valid_files = [f for f in files if validate_audio_file(f)]
            if valid_files:
                self.voice_matching_card.candidate_files = valid_files
                file_names = [Path(f).name for f in valid_files[:3]]  # 显示前3个文件名
                display_text = "📁 " + ", ".join(file_names)
                if len(valid_files) > 3:
                    display_text += f" 等{len(valid_files)}个文件"
                self.voice_matching_card.candidate_label.setText(display_text)
                self.voice_matching_card.candidate_label.setStyleSheet("color: #48bb78; font-weight: 600;")
                print(f"✅ 已选择{len(valid_files)}个待匹配音频文件")
                self.check_voice_matching_ready()
            else:
                QMessageBox.warning(self, "无效文件", "没有选择有效的音频文件。")
    
    def check_voice_matching_ready(self):
        """Check if voice matching is ready to process"""
        card = self.voice_matching_card
        if hasattr(card, 'reference_file') and card.reference_file and \
           hasattr(card, 'candidate_files') and card.candidate_files:
            card.process_button.setEnabled(True)
            print("🎯 人声匹配准备就绪，可以开始处理！")
    
    def process_preprocessing(self):
        """Process audio preprocessing"""
        card = self.preprocessing_card
        if not card.selected_file:
            return
        
        # Setup output directory
        card.output_dir = "output/preprocessing"
        PathManager.ensure_dir(card.output_dir)
        
        # Show progress
        self.show_processing_progress(card, "正在进行音频预处理...")
          # Create processing thread
        self.current_thread = ProcessingThread(
            self.preprocessor.process_audio,
            input_path=card.selected_file,
            output_dir=card.output_dir,
            apply_bandpass=True,
            apply_spectral_subtraction=True,
            apply_wiener=False,
            low_freq=80,
            high_freq=7000  # 使用更保守的高频设置
        )
        
        self.current_thread.result_ready.connect(
            lambda results: self.on_preprocessing_complete(card, results)
        )
        self.current_thread.error_occurred.connect(
            lambda error: self.on_processing_error(card, error)
        )
        
        self.current_thread.start()
    
    def process_source_separation(self):
        """Process source separation"""
        card = self.source_separation_card
        if not card.selected_file:
            return
        
        # Setup output directory
        card.output_dir = "output/demucs_output"
        PathManager.ensure_dir(card.output_dir)
        
        # Show progress
        self.show_processing_progress(card, "正在进行音源分离...")
        
        # Create processing thread
        self.current_thread = ProcessingThread(
            self.source_separator.separate_audio,
            input_path=card.selected_file,
            output_dir=card.output_dir,
            device="cpu"
        )
        
        self.current_thread.result_ready.connect(
            lambda results: self.on_source_separation_complete(card, results)
        )
        self.current_thread.error_occurred.connect(
            lambda error: self.on_processing_error(card, error)
        )
        
        self.current_thread.start()
    
    def process_speaker_separation(self):
        """Process speaker separation"""
        card = self.speaker_separation_card
        if not card.selected_file:
            return
        
        # Setup output directory
        card.output_dir = "output/speaker_output"
        PathManager.ensure_dir(card.output_dir)
        
        # Show progress
        self.show_processing_progress(card, "正在进行说话人分离...")
          # Create processing thread
        self.current_thread = ProcessingThread(
            self.speaker_separator.separate_speakers,  # 修改方法名
            input_path=card.selected_file,
            output_dir=card.output_dir
        )
        
        self.current_thread.result_ready.connect(
            lambda results: self.on_speaker_separation_complete(card, results)
        )
        self.current_thread.error_occurred.connect(            lambda error: self.on_processing_error(card, error)
        )
        
        self.current_thread.start()
    
    def process_voice_matching(self):
        """Process voice matching"""
        card = self.voice_matching_card
        if not hasattr(card, 'reference_file') or not card.reference_file or \
           not hasattr(card, 'candidate_files') or not card.candidate_files:
            QMessageBox.warning(self, "文件未选择", "请先选择参考音频和待匹配音频文件。")
            return
        
        # Setup output directory
        card.output_dir = "output/final_output"
        PathManager.ensure_dir(card.output_dir)
        
        # Show progress
        self.show_processing_progress(card, "正在进行人声匹配分析...")
        
        # Create processing thread
        self.current_thread = ProcessingThread(
            self.voice_matcher.match_voices,
            reference_path=card.reference_file,
            candidate_paths=card.candidate_files,
            output_dir=card.output_dir
        )
        
        self.current_thread.result_ready.connect(
            lambda results: self.on_voice_matching_complete(card, results)
        )
        self.current_thread.error_occurred.connect(
            lambda error: self.on_processing_error(card, error)
        )
        
        self.current_thread.start()
    
    def show_processing_progress(self, card: ProcessingCard, message: str):
        """Show processing progress on card"""
        card.progress_bar.setVisible(True)
        card.progress_bar.setRange(0, 0)  # Indeterminate progress
        card.status_label.setText(message)
        card.status_label.setVisible(True)
        card.process_button.setEnabled(False)
        card.view_results_button.setEnabled(False)
        card.open_folder_button.setEnabled(False)
    
    def hide_processing_progress(self, card: ProcessingCard):
        """Hide processing progress on card"""        
        card.progress_bar.setVisible(False)
        card.status_label.setVisible(False)
        card.process_button.setEnabled(True)
    
    def on_preprocessing_complete(self, card: ProcessingCard, results: Dict[str, Any]):
        """Handle preprocessing completion"""
        self.hide_processing_progress(card)
        
        if results:
            card.processing_results = results
            card.view_results_button.setEnabled(True)
            card.open_folder_button.setEnabled(True)
            
            # Display visualization
            self.matplotlib_widget.plot_preprocessing_results(results)
            
            print(f"✅ 音频预处理完成成功！")
            print(f"📁 输出文件: {results['output_audio_path']}")
            
            # Show success message
            QMessageBox.information(
                self, 
                "处理完成", 
                f"音频预处理已成功完成！\n\n"
                f"输出文件: {Path(results['output_audio_path']).name}\n"
                f"报告文件: {Path(results['report_path']).name}"
            )
        else:
            print("❌ 音频预处理失败！")
            QMessageBox.warning(self, "处理失败", "音频预处理失败！")
    
    def on_source_separation_complete(self, card: ProcessingCard, results: Dict[str, Any]):
        """Handle source separation completion"""
        self.hide_processing_progress(card)
        
        if results and results.get('success'):
            card.processing_results = results
            card.view_results_button.setEnabled(True)
            card.open_folder_button.setEnabled(True)
            
            stems = list(results['separated_files'].keys())
            print(f"✅ 音源分离完成成功！")
            print(f"📁 分离的音轨: {', '.join(stems)}")
            
            # Show success message
            QMessageBox.information(
                self, 
                "处理完成", 
                f"音源分离已成功完成！\n\n"
                f"分离的音轨: {', '.join(stems)}\n"
                f"输出目录: {results['output_dir']}"
            )
        else:
            print("❌ 音源分离失败！")
            QMessageBox.warning(self, "处理失败", "音源分离失败！")
    
    def on_speaker_separation_complete(self, card: ProcessingCard, results: Dict[str, Any]):
        """Handle speaker separation completion"""
        self.hide_processing_progress(card)
        
        if results and results.get('success'):
            card.processing_results = results
            card.view_results_button.setEnabled(True)
            card.open_folder_button.setEnabled(True)
            
            speakers = results.get('speakers', [])
            print(f"✅ 说话人分离完成成功！")
            print(f"👥 发现的说话人: {', '.join(speakers)}")
            
            # Show success message
            QMessageBox.information(
                self, 
                "处理完成", 
                f"说话人分离已成功完成！\n\n"
                f"发现说话人数量: {len(speakers)}\n"
                f"说话人ID: {', '.join(speakers)}"
            )
        else:
            print("❌ 说话人分离失败！")
            QMessageBox.warning(self, "处理失败", "说话人分离失败！")
    
    def on_voice_matching_complete(self, card: ProcessingCard, results: Dict[str, Any]):
        """Handle voice matching completion"""
        self.hide_processing_progress(card)
        
        if results and results.get('success'):
            card.processing_results = results
            card.view_results_button.setEnabled(True)
            card.open_folder_button.setEnabled(True)
            
            best_match = results['best_match_name']
            score = results['best_score']
            
            print(f"✅ 人声匹配完成成功！")
            print(f"🎯 最佳匹配: {best_match} (相似度: {score:.3f})")
            
            # Show success message
            QMessageBox.information(
                self, 
                "处理完成", 
                f"人声匹配已成功完成！\n\n"
                f"最佳匹配: {best_match}\n"
                f"相似度评分: {score:.3f}\n"
                f"输出文件: {Path(results['best_match_output']).name}"
            )
        else:
            print("❌ 人声匹配失败！")
            QMessageBox.warning(self, "处理失败", "人声匹配失败！")
    
    def on_processing_error(self, card: ProcessingCard, error: str):
        """Handle processing errors"""        
        self.hide_processing_progress(card)
        
        print(f"❌ 处理错误: {error}")
        QMessageBox.critical(self, "处理错误", f"处理失败，错误信息:\n\n{error}")
    
    def view_preprocessing_results(self):
        """View preprocessing results"""
        card = self.preprocessing_card
        if card.processing_results:
            # Display visualization
            self.matplotlib_widget.plot_preprocessing_results(card.processing_results)
            
            # Show detailed results dialog
            self.show_results_dialog("预处理结果", card.processing_results)
    
    def view_source_separation_results(self):
        """View source separation results"""
        card = self.source_separation_card
        if card.processing_results:
            results = card.processing_results
            
            # Create results message
            message = f"音源分离结果:\n\n"
            message += f"使用模型: {results['model_used']}\n"
            message += f"输出目录: {results['output_dir']}\n\n"
            message += "分离的文件:\n"
            
            for stem, path in results['separated_files'].items():
                message += f"  {stem.capitalize()}: {Path(path).name}\n"
            
            QMessageBox.information(self, "音源分离结果", message)
    
    def view_speaker_separation_results(self):
        """View speaker separation results"""
        card = self.speaker_separation_card
        if card.processing_results:
            results = card.processing_results
            
            # Create results message
            message = f"说话人分离结果:\n\n"
            message += f"发现说话人数量: {len(results.get('speakers', []))}\n"
            message += f"说话人ID: {', '.join(results.get('speakers', []))}\n\n"
            message += "分离的文件:\n"
            
            for speaker, path in results.get('separated_files', {}).items():
                message += f"  {speaker}: {Path(path).name}\n"
            
            QMessageBox.information(self, "说话人分离结果", message)
    
    def view_voice_matching_results(self):
        """View voice matching results"""
        card = self.voice_matching_card
        if card.processing_results:
            results = card.processing_results
            
            # Create results message
            message = f"人声匹配结果:\n\n"
            message += f"最佳匹配: {results['best_match_name']}\n"
            message += f"相似度评分: {results['best_score']:.4f}\n"
            message += f"输出文件: {Path(results['best_match_output']).name}\n\n"
            
            # Show top similarities
            all_similarities = results.get('all_similarities', {})
            if all_similarities:
                message += "所有候选者排名:\n"
                ranked = sorted(all_similarities.items(), 
                              key=lambda x: x[1]['composite_score'], reverse=True)
                for i, (name, sims) in enumerate(ranked[:5], 1):
                    message += f"  {i}. {name}: {sims['composite_score']:.3f}\n"
            
            QMessageBox.information(self, "人声匹配结果", message)
    
    def show_results_dialog(self, title: str, results: Dict[str, Any]):        
        """Show detailed results dialog"""
        pass
    
    def open_output_folder(self, card: ProcessingCard):
        """Open output folder for a processing card"""
        if card.output_dir and os.path.exists(card.output_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(card.output_dir))
            print(f"📂 已打开输出文件夹: {card.output_dir}")
        else:
            QMessageBox.warning(self, "文件夹未找到", "输出文件夹不存在。")
    
    def update_console(self):
        """Update console output"""
        # Get new output from redirectors
        while not self.stdout_redirector.buffer.empty():
            try:
                text = self.stdout_redirector.buffer.get_nowait()
                self.console_output.append(text.rstrip())
            except queue.Empty:
                break
        
        while not self.stderr_redirector.buffer.empty():
            try:
                text = self.stderr_redirector.buffer.get_nowait()
                self.console_output.append(f"ERROR: {text.rstrip()}")
            except queue.Empty:
                break
        
        # Auto-scroll to bottom
        scrollbar = self.console_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Restore stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Stop any running threads
        if self.current_thread and self.current_thread.isRunning():
            self.current_thread.terminate()
            self.current_thread.wait()
        
        event.accept()


def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Voice Processing Suite")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Voice Processing Team")
    
    # Create and show main window
    window = VoiceProcessingApp()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
