#!/usr/bin/env python3
"""
Modern Voice Processing Suite GUI
===============================

A modern, flat design interface with sidebar navigation and collapsible panels.
"""

import sys
import os
import subprocess
import json
import io
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import time

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import matplotlib
matplotlib.use('Qt5Agg')  # Set matplotlib backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.preprocessing.audio_preprocessor import AudioPreprocessor
from modules.source_separation.demucs_separator import DemucsSourceSeparator
from modules.speaker_separation.speaker_separator import SpeakerSeparator
from modules.voice_matching.voice_matcher import VoiceMatcher
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


class ModernCard(QFrame):
    """Modern card widget with flat design"""
    
    def __init__(self, title: str, icon: str = "🎵", parent=None):
        super().__init__(parent)
        self.title = title
        self.icon = icon
        self.setup_ui()
        
    def setup_ui(self):
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setStyleSheet("""
            ModernCard {
                background-color: #ffffff;
                border-radius: 16px;
                border: 1px solid #f0f0f0;
            }
            ModernCard:hover {
                border: 1px solid #e0e7ff;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Header
        header_layout = QHBoxLayout()
        
        icon_label = QLabel(self.icon)
        icon_label.setStyleSheet("""
            font-size: 24px;
            color: #6366f1;
            font-weight: 500;
        """)
        
        title_label = QLabel(self.title)
        title_label.setStyleSheet("""
            font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
            font-size: 18px;
            font-weight: 600;
            color: #1f2937;
            margin-left: 8px;
        """)
        
        header_layout.addWidget(icon_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Content area
        self.content_area = QVBoxLayout()
        self.content_area.setSpacing(12)
        layout.addLayout(self.content_area)
          # Status and buttons area
        self.status_label = QLabel("准备就绪")
        self.status_label.setStyleSheet("""
            color: #6b7280;
            font-size: 14px;
            padding: 8px 12px;
            background-color: #f9fafb;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        """)
          # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 8px;
                background-color: #f3f4f6;
                height: 8px;
                text-align: center;
                color: #6b7280;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #6366f1;
                border-radius: 8px;
            }
        """)
        
        self.content_area.addWidget(self.status_label)
        self.content_area.addWidget(self.progress_bar)
          # Button area
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(12)
        self.content_area.addLayout(self.button_layout)
        
        # Results button area (initially hidden)
        self.results_layout = QHBoxLayout()
        self.results_layout.setSpacing(8)
        
        # View results button
        self.view_results_btn = QPushButton("📊 查看结果")
        self.view_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 8px;
                font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
            QPushButton:disabled {
                background-color: #e5e7eb;
                color: #9ca3af;
            }
        """)
        self.view_results_btn.setVisible(False)
        self.view_results_btn.setEnabled(False)
        
        # Open folder button
        self.open_folder_btn = QPushButton("📁 打开文件夹")
        self.open_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 8px;
                font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
            QPushButton:disabled {
                background-color: #e5e7eb;
                color: #9ca3af;
            }
        """)
        self.open_folder_btn.setVisible(False)
        self.open_folder_btn.setEnabled(False)
        
        self.results_layout.addWidget(self.view_results_btn)
        self.results_layout.addWidget(self.open_folder_btn)
        self.results_layout.addStretch()
        
        self.content_area.addLayout(self.results_layout)
        
        # Store processing results
        self.processing_results = None
        self.output_dir = None
        
    def add_button(self, text: str, callback, primary: bool = False):
        """Add a modern button to the card"""
        button = QPushButton(text)
        
        if primary:
            button.setStyleSheet("""
                QPushButton {
                    background-color: #6366f1;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 12px;
                    font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
                    font-size: 14px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #5b5bf6;
                }
                QPushButton:pressed {
                    background-color: #4f46e5;
                }
                QPushButton:disabled {
                    background-color: #e5e7eb;
                    color: #9ca3af;
                }
            """)
        else:
            button.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #6b7280;
                    border: 1px solid #d1d5db;
                    padding: 12px 24px;
                    border-radius: 12px;
                    font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
                    font-size: 14px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #f9fafb;
                    border-color: #9ca3af;
                }
                QPushButton:pressed {
                    background-color: #f3f4f6;
                }
            """)
        
        button.clicked.connect(callback)
        self.button_layout.addWidget(button)
        return button


class SidebarButton(QPushButton):
    """Modern sidebar navigation button"""
    
    def __init__(self, text: str, icon: str, parent=None):
        super().__init__(parent)
        self.setText(f"{icon}  {text}")
        self.setCheckable(True)
        self.setStyleSheet("""
            SidebarButton {
                text-align: left;
                padding: 16px 20px;
                border: none;
                border-radius: 12px;
                background-color: transparent;
                color: #6b7280;
                font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
                font-size: 15px;
                font-weight: 500;
                margin: 2px 0;
            }
            SidebarButton:hover {
                background-color: #f8fafc;
                color: #374151;
            }
            SidebarButton:checked {
                background-color: #eef2ff;
                color: #6366f1;
                font-weight: 600;
            }
        """)


class CollapsiblePanel(QFrame):
    """Collapsible panel widget"""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.is_collapsed = False
        self.setup_ui()
        
    def setup_ui(self):
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setStyleSheet("""
            CollapsiblePanel {
                background-color: #ffffff;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        self.header = QPushButton(f"▼ {self.title}")
        self.header.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 16px 20px;
                border: none;
                border-radius: 12px 12px 0 0;
                background-color: #f8fafc;
                color: #374151;
                font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
                font-size: 14px;
                font-weight: 600;
                border-bottom: 1px solid #e5e7eb;
            }
            QPushButton:hover {
                background-color: #f1f5f9;
            }
        """)
        self.header.clicked.connect(self.toggle_collapse)
        layout.addWidget(self.header)
        
        # Content
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(self.content_widget)
        
    def toggle_collapse(self):
        """Toggle panel collapse state"""
        self.is_collapsed = not self.is_collapsed
        self.content_widget.setVisible(not self.is_collapsed)
        
        arrow = "▶" if self.is_collapsed else "▼"
        self.header.setText(f"{arrow} {self.title}")


class ModernConsole(QTextEdit):
    """Modern console widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1f2937;
                color: #f9fafb;
                border: none;
                border-radius: 12px;
                font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                line-height: 1.4;
                padding: 16px;
            }
            QTextEdit QScrollBar:vertical {
                background-color: #374151;
                width: 8px;
                border-radius: 4px;
            }
            QTextEdit QScrollBar::handle:vertical {
                background-color: #6b7280;
                border-radius: 4px;
                min-height: 20px;
            }
            QTextEdit QScrollBar::handle:vertical:hover {
                background-color: #9ca3af;
            }
        """)
        self.setReadOnly(True)
        
    def append_message(self, message: str, level: str = "info"):
        """Append a styled message to console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        colors = {
            "info": "#60a5fa",
            "success": "#34d399", 
            "warning": "#fbbf24",
            "error": "#f87171"
        }
        
        color = colors.get(level, "#f9fafb")
        
        self.append(f'<span style="color: {color};">[{timestamp}] {message}</span>')
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class ProcessingThread(QThread):
    """Background processing thread"""
    progress_update = pyqtSignal(int)
    process_complete = pyqtSignal(dict, str)  # result, output_dir
    process_error = pyqtSignal(str)
    
    def __init__(self, process_type, input_file, processors, **kwargs):
        super().__init__()
        self.process_type = process_type
        self.input_file = input_file
        self.processors = processors
        self.kwargs = kwargs
    
    def run(self):
        """Run the processing in background thread"""
        try:
            if self.process_type == 'preprocessing':
                self._process_preprocessing()
            elif self.process_type == 'source_separation':
                self._process_source_separation()
            elif self.process_type == 'speaker_separation':
                self._process_speaker_separation()
            elif self.process_type == 'voice_matching':
                self._process_voice_matching()
            else:
                self.process_error.emit(f"未知的处理类型: {self.process_type}")
                
        except Exception as e:
            self.process_error.emit(str(e))
    
    def _process_preprocessing(self):
        """Process audio preprocessing"""
        try:
            processor = self.processors.get('preprocessor')
            if not processor:
                self.process_error.emit("预处理器未初始化")
                return
            self.progress_update.emit(10)
            
            # Get output directory
            output_dir = os.path.join("output", "preprocessing")
            os.makedirs(output_dir, exist_ok=True)
            
            # Call the actual preprocessing
            result = processor.process_audio(self.input_file, output_dir)
            
            self.progress_update.emit(100)
            
            self.process_complete.emit(result, output_dir)
            
        except Exception as e:
            self.process_error.emit(f"预处理失败: {str(e)}")
    
    def _process_source_separation(self):
        """Process source separation"""
        try:
            processor = self.processors.get('source_separator')
            if not processor:
                self.process_error.emit("音源分离器未初始化")
                return
            self.progress_update.emit(10)
            
            # Get output directory
            output_dir = os.path.join("output", "source_separation")
            os.makedirs(output_dir, exist_ok=True)
            
            # Call the actual source separation
            result = processor.separate_audio(self.input_file, output_dir)
            
            self.progress_update.emit(100)
            
            self.process_complete.emit(result, output_dir)
            
        except Exception as e:
            self.process_error.emit(f"音源分离失败: {str(e)}")
    
    def _process_speaker_separation(self):
        """Process speaker separation"""
        try:
            processor = self.processors.get('speaker_separator')
            if not processor:
                self.process_error.emit("说话人分离器未初始化")
                return
            self.progress_update.emit(10)
            
            # Get output directory
            output_dir = os.path.join("output", "speaker_separation")
            os.makedirs(output_dir, exist_ok=True)
            
            # Call the actual speaker separation
            result = processor.separate_speakers(self.input_file, output_dir)
            
            self.progress_update.emit(100)
            
            self.process_complete.emit(result, output_dir)
            
        except Exception as e:
            self.process_error.emit(f"说话人分离失败: {str(e)}")
    
    def _process_voice_matching(self):
        """Process voice matching"""
        try:
            processor = self.processors.get('voice_matcher')
            if not processor:
                self.process_error.emit("人声匹配器未初始化")
                return
            
            self.progress_update.emit(10)
            
            # Get reference and candidate files from kwargs
            reference_file = self.kwargs.get('reference_file')
            candidate_files = self.kwargs.get('candidate_files', [])
            if not reference_file or not candidate_files:
                self.process_error.emit("参考音频或候选音频文件缺失")
                return
            
            # Get output directory
            output_dir = os.path.join("output", "voice_matching")
            os.makedirs(output_dir, exist_ok=True)
            
            # Call the actual voice matching
            result = processor.match_voices(reference_file, candidate_files, output_dir)
            
            self.progress_update.emit(100)
            
            self.process_complete.emit(result, output_dir)
            
        except Exception as e:
            self.process_error.emit(f"人声匹配失败: {str(e)}")


class ModernVoiceProcessingApp(QMainWindow):
    """Modern Voice Processing Suite Application"""
    def __init__(self):
        super().__init__()
        self.current_module = None
        self.processors = {}
        
        # Setup console redirection before anything else
        self.setup_console_redirection()
        
        self.setup_processors()
        self.setup_ui()
        self.setup_connections()
          # Setup timer for console updates
        self.console_timer = QTimer()
        self.console_timer.timeout.connect(self.update_console)
        self.console_timer.start(100)  # Update every 100ms
        
    def setup_console_redirection(self):
        """Setup console output redirection"""
        # Keep references to original streams
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create redirectors that write to both GUI and console
        self.stdout_redirector = ConsoleRedirector(None, self.original_stdout)
        self.stderr_redirector = ConsoleRedirector(None, self.original_stderr)
        
        # Redirect stdout and stderr
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stderr_redirector
    def setup_processors(self):
        """Initialize processing modules"""
        try:
            print("🔧 正在初始化处理器...")
            
            # Initialize AudioPreprocessor
            print("   - 初始化音频预处理器...")
            self.processors = {}
            self.processors['preprocessor'] = AudioPreprocessor()
            print("   ✅ 音频预处理器初始化成功")
            
            # Initialize DemucsSourceSeparator
            print("   - 初始化音源分离器...")
            self.processors['source_separator'] = DemucsSourceSeparator()
            print("   ✅ 音源分离器初始化成功")
            
            # Initialize SpeakerSeparator
            print("   - 初始化说话人分离器...")
            self.processors['speaker_separator'] = SpeakerSeparator(hf_token="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")
            print("   ✅ 说话人分离器初始化成功")
            
            # Initialize VoiceMatcher
            print("   - 初始化人声匹配器...")
            self.processors['voice_matcher'] = VoiceMatcher()
            print("   ✅ 人声匹配器初始化成功")
            
            print("🎉 所有处理器初始化完成")
            
        except Exception as e:
            print(f"❌ 处理器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            # 使用简化的处理器字典，避免程序完全崩溃
            self.processors = {
                'preprocessor': None,
                'source_separator': None,
                'speaker_separator': None,
                'voice_matcher': None
            }
            
    def setup_ui(self):
        """Setup modern UI"""
        self.setWindowTitle("Voice Processing Suite - Modern Interface")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set app-wide font
        app_font = QFont("SF Pro Display", 10)
        if not app_font.exactMatch():
            app_font = QFont("Segoe UI", 10)
        self.setFont(app_font)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout - horizontal
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        self.setup_sidebar()
        main_layout.addWidget(self.sidebar, 0)
        
        # Content area
        self.content_area = QWidget()
        self.content_area.setStyleSheet("""
            QWidget {
                background-color: #f8fafc;
            }
        """)
        
        content_layout = QVBoxLayout(self.content_area)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(24)
        
        # Top content - module cards
        self.module_container = QScrollArea()
        self.module_container.setWidgetResizable(True)
        self.module_container.setFrameStyle(QFrame.Shape.NoFrame)
        self.module_container.setStyleSheet("background-color: transparent;")
        
        # Bottom panels - console and visualization
        self.setup_bottom_panels()
        
        content_layout.addWidget(self.module_container, 1)
        content_layout.addWidget(self.bottom_panels, 0)
        
        main_layout.addWidget(self.content_area, 1)
        
        # Initialize with first module
        self.show_module(0)
        
    def setup_sidebar(self):
        """Setup modern sidebar navigation"""
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(280)
        self.sidebar.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-right: 1px solid #e5e7eb;
            }
        """)
        
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(20, 30, 20, 30)
        sidebar_layout.setSpacing(20)
        
        # App title
        title_label = QLabel("Voice Processing")
        title_label.setStyleSheet("""
            font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
            font-size: 24px;
            font-weight: 700;
            color: #111827;
            padding-bottom: 10px;
        """)
        sidebar_layout.addWidget(title_label)
        
        subtitle_label = QLabel("音频处理工具套件")
        subtitle_label.setStyleSheet("""
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 20px;
        """)
        sidebar_layout.addWidget(subtitle_label)
        
        # Navigation buttons
        self.nav_buttons = QButtonGroup()
        
        modules = [
            ("音频预处理", "🎛️"),
            ("音源分离", "🎼"), 
            ("说话人分离", "👥"),
            ("人声匹配", "🎯")
        ]
        
        for i, (name, icon) in enumerate(modules):
            btn = SidebarButton(name, icon)
            self.nav_buttons.addButton(btn, i)
            sidebar_layout.addWidget(btn)
            
        # Set first button as checked
        self.nav_buttons.button(0).setChecked(True)
        
        sidebar_layout.addStretch()
        
        # Footer info
        info_label = QLabel("v2.0 - Modern Edition")
        info_label.setStyleSheet("""
            color: #9ca3af;
            font-size: 12px;
            text-align: center;
        """)
        sidebar_layout.addWidget(info_label)
        
    def setup_bottom_panels(self):
        """Setup collapsible console and visualization panels"""
        self.bottom_panels = QWidget()
        panels_layout = QHBoxLayout(self.bottom_panels)
        panels_layout.setContentsMargins(0, 0, 0, 0)
        panels_layout.setSpacing(24)
        
        # Console panel
        self.console_panel = CollapsiblePanel("控制台输出")
        self.console = ModernConsole()
        self.console.setMaximumHeight(300)
        self.console_panel.content_layout.addWidget(self.console)
        
        # Visualization panel  
        self.viz_panel = CollapsiblePanel("数据可视化")
        self.viz_container = QLabel("可视化内容将在这里显示")
        self.viz_container.setStyleSheet("""
            color: #9ca3af;
            text-align: center;
            padding: 40px;
            background-color: #f9fafb;
            border-radius: 8px;
            border: 2px dashed #d1d5db;
        """)
        self.viz_panel.content_layout.addWidget(self.viz_container)
        
        panels_layout.addWidget(self.console_panel, 1)
        panels_layout.addWidget(self.viz_panel, 1)

    def setup_connections(self):
        """Setup signal connections"""
        self.nav_buttons.buttonClicked.connect(self.on_nav_clicked)
        
        # Add welcome message
        self.console.append_message("🚀 Voice Processing Suite 已启动", "success")
        self.console.append_message("选择左侧模块开始处理", "info")
        
    def update_console(self):
        """Update console output"""
        # Get new output from redirectors
        while not self.stdout_redirector.buffer.empty():
            try:
                text = self.stdout_redirector.buffer.get_nowait()
                self.console.append_message(text.rstrip(), "info")
            except queue.Empty:
                break
        
        while not self.stderr_redirector.buffer.empty():
            try:
                text = self.stderr_redirector.buffer.get_nowait()
                self.console.append_message(f"ERROR: {text.rstrip()}", "error")
            except queue.Empty:
                break
    
    def on_nav_clicked(self, button):
        """Handle navigation button click"""
        module_id = self.nav_buttons.id(button)
        self.show_module(module_id)
    
    def show_module(self, module_id: int):
        """Show specific module interface"""
        self.current_module = module_id
        
        # Create new module widget
        module_widget = QWidget()
        module_layout = QVBoxLayout(module_widget)
        module_layout.setContentsMargins(0, 0, 0, 0)
        module_layout.setSpacing(24)
          # Create appropriate card based on module_id
        if module_id == 0:  # Audio Preprocessing
            card = self.create_preprocessing_card()
            self.preprocessing_card = card
        elif module_id == 1:  # Source Separation
            card = self.create_source_separation_card()
            self.source_separation_card = card
        elif module_id == 2:  # Speaker Separation
            card = self.create_speaker_separation_card()
            self.speaker_separation_card = card
        elif module_id == 3:  # Voice Matching
            card = self.create_voice_matching_card()
            self.voice_matching_card = card
        else:
            card = ModernCard("未知模块")
        
        # Add card to layout
        module_layout.addWidget(card)
        module_layout.addStretch()
        
        # Clear existing content safely
        if self.module_container.widget():
            old_widget = self.module_container.takeWidget()
            if old_widget:
                old_widget.hide()
                QTimer.singleShot(100, old_widget.deleteLater)
        
        # Set the new widget
        self.module_container.setWidget(module_widget)
          
    def create_preprocessing_card(self):
        """Create audio preprocessing module card"""
        card = ModernCard("音频预处理", "🎛️")
        
        # File selection
        file_info = QLabel("选择音频文件进行预处理")
        file_info.setStyleSheet("color: #6b7280; font-size: 14px;")
        card.content_area.addWidget(file_info)
        
        card.select_btn = card.add_button("📁 选择音频文件", self.select_audio_file)
        card.process_btn = card.add_button("🎛️ 开始预处理", self.process_preprocessing, primary=True)
        card.process_btn.setEnabled(False)
        
        # Connect result buttons
        card.view_results_btn.clicked.connect(lambda: self.view_preprocessing_results(card))
        card.open_folder_btn.clicked.connect(lambda: self.open_output_folder(card))
        
        # Store references
        card.selected_file = None
        card.file_info = file_info
        
        return card
          
    def create_source_separation_card(self):
        """Create source separation module card"""
        card = ModernCard("音源分离", "🎼")
        
        file_info = QLabel("使用 Demucs 分离音频为不同音轨")
        file_info.setStyleSheet("color: #6b7280; font-size: 14px;")
        card.content_area.addWidget(file_info)
        
        card.select_btn = card.add_button("📁 选择音频文件", self.select_audio_file)
        card.process_btn = card.add_button("🎼 开始分离", self.process_source_separation, primary=True)
        card.process_btn.setEnabled(False)
        
        # Connect result buttons
        card.view_results_btn.clicked.connect(lambda: self.view_source_separation_results(card))
        card.open_folder_btn.clicked.connect(lambda: self.open_output_folder(card))
        
        card.selected_file = None
        card.file_info = file_info
        
        return card
    
    def create_speaker_separation_card(self):
        """Create speaker separation module card"""
        card = ModernCard("说话人分离", "👥")
        
        file_info = QLabel("使用 Pyannote 进行说话人分离")
        file_info.setStyleSheet("color: #6b7280; font-size: 14px;")
        card.content_area.addWidget(file_info)
        
        card.select_btn = card.add_button("📁 选择音频文件", self.select_audio_file)
        card.process_btn = card.add_button("👥 开始分离", self.process_speaker_separation, primary=True)
        card.process_btn.setEnabled(False)
        
        # Connect result buttons
        card.view_results_btn.clicked.connect(lambda: self.view_speaker_separation_results(card))
        card.open_folder_btn.clicked.connect(lambda: self.open_output_folder(card))
        
        card.selected_file = None
        card.file_info = file_info
        
        return card
        
    def create_voice_matching_card(self):
        """Create voice matching module card"""
        card = ModernCard("人声匹配", "🎯")
        
        # Reference audio
        ref_info = QLabel("参考音频：未选择")
        ref_info.setStyleSheet("color: #6b7280; font-size: 14px;")
        card.content_area.addWidget(ref_info)
        
        card.select_ref_btn = card.add_button("🎵 选择参考音频", self.select_reference_audio)
        
        # Candidate audios
        cand_info = QLabel("待匹配音频：未选择")
        cand_info.setStyleSheet("color: #6b7280; font-size: 14px;")
        card.content_area.addWidget(cand_info)
        card.select_cand_btn = card.add_button("🎪 选择待匹配音频", self.select_candidate_audio)
        card.process_btn = card.add_button("🎯 开始匹配", self.process_voice_matching, primary=True)
        card.process_btn.setEnabled(False)
        
        # Connect result buttons
        card.view_results_btn.clicked.connect(lambda: self.view_voice_matching_results(card))
        card.open_folder_btn.clicked.connect(lambda: self.open_output_folder(card))
        
        card.reference_file = None
        card.candidate_files = []
        card.ref_info = ref_info
        card.cand_info = cand_info
        
        return card
    
    def select_audio_file(self):
        """Select audio file for current module"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a);;所有文件 (*)"
        )
        
        if file_path:
            # Get current active card based on the sender button
            sender = self.sender()
            card = sender.parent()
            while card and not isinstance(card, ModernCard):
                card = card.parent()
                
            if card:
                card.selected_file = file_path
                if hasattr(card, 'file_info'):
                    card.file_info.setText(f"已选择：{Path(file_path).name}")
                    card.file_info.setStyleSheet("color: #059669; font-size: 14px; font-weight: 500;")
                card.process_btn.setEnabled(True)
                
                self.console.append_message(f"已选择文件：{Path(file_path).name}", "success")
    
    def select_reference_audio(self):
        """Select reference audio for voice matching"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择参考音频",
            "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a);;所有文件 (*)"
        )
        
        if file_path:
            # Get current active card based on the sender button
            sender = self.sender()
            card = sender.parent()
            while card and not isinstance(card, ModernCard):
                card = card.parent()
                
            if card:
                card.reference_file = file_path
                if hasattr(card, 'ref_info'):
                    card.ref_info.setText(f"参考音频：{Path(file_path).name}")
                    card.ref_info.setStyleSheet("color: #059669; font-size: 14px; font-weight: 500;")
                self.check_voice_matching_ready(card)
    
    def select_candidate_audio(self):
        """Select candidate audio files for voice matching"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择待匹配音频文件（可多选）",
            "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a);;所有文件 (*)"
        )
        
        if files:
            # Get current active card based on the sender button
            sender = self.sender()
            card = sender.parent()
            while card and not isinstance(card, ModernCard):
                card = card.parent()
                
            if card:
                card.candidate_files = files
                
                file_names = [Path(f).name for f in files[:3]]
                display_text = ", ".join(file_names)
                if len(files) > 3:
                    display_text += f" 等{len(files)}个文件"
                    
                if hasattr(card, 'cand_info'):
                    card.cand_info.setText(f"待匹配音频：{display_text}")
                    card.cand_info.setStyleSheet("color: #059669; font-size: 14px; font-weight: 500;")
                self.check_voice_matching_ready(card)    
    
    def reset_preprocessing_ui(self):
        """Reset preprocessing UI state"""
        try:
            if hasattr(self, 'current_processing_card') and self.current_processing_card:
                card = self.current_processing_card
                card.process_btn.setEnabled(True)
                card.status_label.setText("准备就绪")
                card.progress_bar.setVisible(False)
        except:
            pass
    def on_preprocessing_success(self, result, output_dir):
        """Handle successful preprocessing completion"""
        try:
            if hasattr(self, 'current_processing_card') and self.current_processing_card:
                card = self.current_processing_card
                card.processing_results = result
                card.output_dir = output_dir
                card.view_results_btn.setVisible(True)
                card.view_results_btn.setEnabled(True)
                card.open_folder_btn.setVisible(True)
                card.open_folder_btn.setEnabled(True)
                card.process_btn.setEnabled(True)
                card.progress_bar.setVisible(False)
                
                # Update visualization if available
                if result and isinstance(result, dict):
                    visualization_files = []
                    
                    # Look for visualization files in output directory
                    if output_dir:
                        viz_extensions = ['.png', '.jpg', '.jpeg', '.pdf']
                        for ext in viz_extensions:
                            viz_files = list(Path(output_dir).glob(f"*{ext}"))
                            visualization_files.extend([str(f) for f in viz_files])
                    
                    # Also check if result contains visualization paths
                    if 'visualization_path' in result:
                        visualization_files.append(result['visualization_path'])
                    elif 'visualizations' in result:
                        visualization_files.extend(result['visualizations'])
                    
                    if visualization_files:
                        self.update_visualization(visualization_files)
                        self.console.append_message(f"📊 已加载 {len(visualization_files)} 个可视化文件", "success")
                
        except Exception as e:
            self.console.append_message(f"❌ 设置结果按钮失败：{str(e)}", "error")
    def get_current_card(self):
        """Get the currently active processing card"""
        try:
            # Return the card of the currently active module
            if self.current_module == 0:  # Audio Preprocessing
                return self.preprocessing_card
            elif self.current_module == 1:  # Source Separation
                return self.source_separation_card
            elif self.current_module == 2:  # Speaker Separation
                return self.speaker_separation_card
            elif self.current_module == 3:  # Voice Matching
                return self.voice_matching_card
            else:
                return None
        except Exception as e:
            self.console.append_message(f"❌ 获取当前卡片失败：{str(e)}", "error")
            return None
    
    def update_progress(self, progress):
        """Update progress bar"""
        try:
            card = self.get_current_card()
            if card and hasattr(card, 'progress_bar'):
                card.progress_bar.setValue(progress)
        except Exception as e:
            self.console.append_message(f"❌ 更新进度失败：{str(e)}", "error")
    
    def handle_processing_complete(self, result, output_dir):
        """Handle processing completion"""
        try:
            self.console.append_message(f"✅ 处理完成!", "success")
            self.console.append_message(f"📁 输出目录: {output_dir}", "info")
            
            card = self.get_current_card()
            if card:
                card.processing_results = result
                card.output_dir = output_dir
                card.view_results_btn.setVisible(True)
                card.view_results_btn.setEnabled(True)
                card.open_folder_btn.setVisible(True)
                card.open_folder_btn.setEnabled(True)
                card.process_btn.setEnabled(True)
                card.progress_bar.setVisible(False)
                
        except Exception as e:
            self.console.append_message(f"❌ 处理完成回调失败：{str(e)}", "error")
    
    def handle_processing_error(self, error_msg):
        """Handle processing error"""
        try:
            self.console.append_message(f"❌ 处理失败: {error_msg}", "error")
            
            card = self.get_current_card()
            if card:
                card.process_btn.setEnabled(True)
                card.progress_bar.setVisible(False)
                
        except Exception as e:
            self.console.append_message(f"❌ 错误处理回调失败：{str(e)}", "error")
    
    def check_voice_matching_ready(self, card):
        """Check if voice matching is ready to process"""
        try:
            if not hasattr(card, 'reference_file') or not hasattr(card, 'candidate_files'):
                return False, "❌ 卡片信息错误"
                
            if not card.reference_file:
                return False, "❌ 请先选择参考音频"
                
            if not card.candidate_files or len(card.candidate_files) == 0:
                return False, "❌ 请先选择待匹配音频"
                
            # Validate files exist
            if not os.path.exists(card.reference_file):
                return False, "❌ 参考音频文件不存在"
                
            for candidate_file in card.candidate_files:
                if not os.path.exists(candidate_file):
                    return False, f"❌ 候选音频文件不存在: {Path(candidate_file).name}"
            
            # Enable process button if everything is ready
            if hasattr(card, 'process_btn'):
                card.process_btn.setEnabled(True)
                    
            return True, "✅ 准备就绪"
            
        except Exception as e:
            return False, f"❌ 检查状态失败: {str(e)}"
    
    # ===== Audio Preprocessing Methods =====
    
    def process_preprocessing(self):
        """Process audio preprocessing"""
        try:
            print("🔥 process_preprocessing 被调用了!")
            self.console.append_message("🔥 开始执行音频预处理...", "info")
              # Check if processors are initialized
            if not self.processors.get('preprocessor'):
                self.console.append_message("❌ 音频预处理器未初始化，请重启应用", "error")
                return
            
            # Get current active card
            card = self.get_current_card()
                
            if not card or not hasattr(card, 'selected_file') or not card.selected_file:
                self.console.append_message("❌ 请先选择音频文件", "error")
                return
                
            # Validate file exists
            if not os.path.exists(card.selected_file):
                self.console.append_message("❌ 选择的文件不存在", "error")
                return
                
            self.console.append_message(f"📁 处理文件: {card.selected_file}", "info")
            
            # Store current card reference
            self.current_processing_card = card
            
            # Setup UI for processing
            card.process_btn.setEnabled(False)
            card.progress_bar.setVisible(True)
            card.progress_bar.setValue(0)
            card.view_results_btn.setVisible(False)
            card.open_folder_btn.setVisible(False)
            
            # Create processing thread
            self.processing_thread = ProcessingThread(
                'preprocessing',
                card.selected_file,
                self.processors
            )
            self.processing_thread.progress_update.connect(self.update_progress)
            self.processing_thread.process_complete.connect(self.handle_processing_complete)
            self.processing_thread.process_error.connect(self.handle_processing_error)
            self.processing_thread.start()
            
        except Exception as e:
            self.console.append_message(f"❌ 音频预处理启动失败：{str(e)}", "error")
            print(f"❌ process_preprocessing 异常：{str(e)}")
            # Re-enable button on error
            card = self.get_current_card()
            if card:
                card.process_btn.setEnabled(True)
                card.progress_bar.setVisible(False)
                
    # ===== Source Separation Methods =====
    
    def process_source_separation(self):
        """Process source separation"""
        try:
            print("🔥 process_source_separation 被调用了!")
            self.console.append_message("🔥 开始执行音源分离...", "info")
            
            # Check if processors are initialized
            if not self.processors.get('source_separator'):
                self.console.append_message("❌ 音源分离器未初始化，请重启应用", "error")
                return
            
            # Get current active card
            card = self.get_current_card()
                
            if not card or not hasattr(card, 'selected_file') or not card.selected_file:
                self.console.append_message("❌ 请先选择音频文件", "error")
                return
                
            # Validate file exists
            if not os.path.exists(card.selected_file):
                self.console.append_message("❌ 选择的文件不存在", "error")
                return
                
            self.console.append_message(f"📁 处理文件: {card.selected_file}", "info")
            
            # Store current card reference
            self.current_processing_card = card
            
            # Disable UI during processing
            card.process_btn.setEnabled(False)
            card.status_label.setText("正在处理...")
            card.progress_bar.setVisible(True)
            card.progress_bar.setValue(0)
            
            # Start processing in thread
            self.processing_thread = threading.Thread(
                target=self._process_source_separation_thread,
                args=(card.selected_file,)
            )
            self.processing_thread.start()
            
        except Exception as e:
            error_msg = f"❌ 启动音源分离失败: {str(e)}"
            self.console.append_message(error_msg, "error")
            import traceback
            traceback.print_exc()
    def _process_source_separation_thread(self, input_file):
        """Source separation thread"""
        try:
            print(f"🎼 开始音源分离线程，输入文件: {input_file}")
            QTimer.singleShot(100, lambda: self.update_processing_status("🎼 开始音源分离...", 10, "info"))
            
            # Check if source_separator is available
            if not self.processors.get('source_separator'):
                raise Exception("音源分离器未初始化")
                
            # Process with DemucsSourceSeparator
            QTimer.singleShot(500, lambda: self.update_processing_status("🤖 加载 Demucs 模型...", 30, "info"))
            
            # 创建输出目录
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 输出目录: {output_dir}")
            
            QTimer.singleShot(800, lambda: self.update_processing_status("🎵 分离音轨中...", 60, "info"))
            print("🔧 调用音源分离器...")
            result = self.processors['source_separator'].separate_audio(
                input_path=input_file,
                output_dir=output_dir
            )
            print(f"✅ 音源分离完成，结果: {result}")
            if result and result.get('separated_files'):
                track_count = len(result['separated_files'])
                QTimer.singleShot(100, lambda: self.update_processing_status(f"✅ 音源分离完成！分离出 {track_count} 个音轨", 100, "success"))
                
                # Store results and show result buttons
                QTimer.singleShot(200, lambda: self.on_source_separation_success(result, output_dir))
                
            else:
                QTimer.singleShot(100, lambda: self.update_processing_status(f"❌ 音源分离失败：结果为空", None, "error"))
                
        except Exception as e:
            error_msg = f"❌ 音源分离出错：{str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            QTimer.singleShot(100, lambda: self.update_processing_status(error_msg, None, "error"))
        finally:
            # Re-enable UI
            print("🔄 重置UI状态...")
            QTimer.singleShot(300, lambda: self.reset_source_separation_ui())
    
    def reset_source_separation_ui(self):
        """Reset source separation UI state"""
        try:
            if hasattr(self, 'current_processing_card') and self.current_processing_card:
                card = self.current_processing_card
                card.process_btn.setEnabled(True)
                card.status_label.setText("准备就绪")
                card.progress_bar.setVisible(False)
        except:
            pass
    def on_source_separation_success(self, result, output_dir):
        """Handle successful source separation completion"""
        try:
            if hasattr(self, 'current_processing_card') and self.current_processing_card:
                card = self.current_processing_card
                card.processing_results = result
                card.output_dir = output_dir
                card.view_results_btn.setVisible(True)
                card.view_results_btn.setEnabled(True)
                card.open_folder_btn.setVisible(True)
                card.open_folder_btn.setEnabled(True)
                card.process_btn.setEnabled(True)
                card.progress_bar.setVisible(False)
                
                # Update visualization if available
                if result and isinstance(result, dict):
                    visualization_files = []
                    
                    # Look for visualization files in output directory
                    if output_dir:
                        viz_extensions = ['.png', '.jpg', '.jpeg', '.pdf']
                        for ext in viz_extensions:
                            viz_files = list(Path(output_dir).glob(f"**/*{ext}"))
                            visualization_files.extend([str(f) for f in viz_files])
                    
                    # Also check if result contains visualization paths
                    if 'visualization_path' in result:
                        visualization_files.append(result['visualization_path'])
                    elif 'visualizations' in result:
                        visualization_files.extend(result['visualizations'])
                    
                    if visualization_files:
                        self.update_visualization(visualization_files)
                        self.console.append_message(f"📊 已加载 {len(visualization_files)} 个可视化文件", "success")
                
        except Exception as e:
            self.console.append_message(f"❌ 设置结果按钮失败：{str(e)}", "error")
    # ===== Speaker Separation Methods =====
    
    def process_speaker_separation(self):
        """Process speaker separation"""
        try:
            print("🔥 process_speaker_separation 被调用了!")
            self.console.append_message("🔥 开始执行说话人分离...", "info")
            
            # Check if processors are initialized
            if not self.processors.get('speaker_separator'):
                self.console.append_message("❌ 说话人分离器未初始化，请重启应用", "error")
                return
            
            # Get current active card
            card = self.get_current_card()
                
            if not card or not hasattr(card, 'selected_file') or not card.selected_file:
                self.console.append_message("❌ 请先选择音频文件", "error")
                return
                
            # Validate file exists
            if not os.path.exists(card.selected_file):
                self.console.append_message("❌ 选择的文件不存在", "error")
                return
            
            self.console.append_message(f"📁 处理文件: {card.selected_file}", "info")
            
            # Store current card reference
            self.current_processing_card = card
            
            # Disable UI during processing
            card.process_btn.setEnabled(False)
            card.status_label.setText("正在处理...")
            card.progress_bar.setVisible(True)
            card.progress_bar.setValue(0)
            
            # Start processing in thread
            self.processing_thread = threading.Thread(
                target=self._process_speaker_separation_thread,
                args=(card.selected_file,)
            )
            self.processing_thread.start()
            
        except Exception as e:
            error_msg = f"❌ 启动说话人分离失败: {str(e)}"
            self.console.append_message(error_msg, "error")
            import traceback
            traceback.print_exc()
        
    def _process_speaker_separation_thread(self, input_file):
        """Speaker separation thread"""
        try:
            QTimer.singleShot(100, lambda: self.update_processing_status("👥 开始说话人分离...", 10, "info"))
            
            # Process with SpeakerSeparator
            QTimer.singleShot(500, lambda: self.update_processing_status("🤖 加载 Pyannote 模型...", 30, "info"))
            
            # 创建输出目录
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            QTimer.singleShot(800, lambda: self.update_processing_status("🗣️ 识别说话人...", 60, "info"))
            result = self.processors['speaker_separator'].separate_speakers(
                input_path=input_file,
                output_dir=output_dir
            )
            if result and result.get('separated_files'):
                speaker_count = len(result['separated_files'])
                QTimer.singleShot(100, lambda: self.update_processing_status(f"✅ 说话人分离完成！识别出 {speaker_count} 个说话人", 100, "success"))
                
                # Store results and show result buttons
                QTimer.singleShot(200, lambda: self.on_speaker_separation_success(result, output_dir))
                
            else:
                QTimer.singleShot(100, lambda: self.update_processing_status(f"❌ 说话人分离失败：结果为空", None, "error"))
                
        except Exception as e:
            QTimer.singleShot(100, lambda: self.update_processing_status(f"❌ 说话人分离出错：{str(e)}", None, "error"))
        finally:
            # Re-enable UI
            QTimer.singleShot(300, lambda: self.reset_speaker_separation_ui())
              
              
    def reset_speaker_separation_ui(self):
        """Reset speaker separation UI state"""
        try:
            if hasattr(self, 'current_processing_card') and self.current_processing_card:
                card = self.current_processing_card
                card.process_btn.setEnabled(True)
                card.status_label.setText("准备就绪")
                card.progress_bar.setVisible(False)
        except:
            pass
    def on_speaker_separation_success(self, result, output_dir):
        """Handle successful speaker separation completion"""
        try:
            if hasattr(self, 'current_processing_card') and self.current_processing_card:
                card = self.current_processing_card
                card.processing_results = result
                card.output_dir = output_dir
                card.view_results_btn.setVisible(True)
                card.view_results_btn.setEnabled(True)
                card.open_folder_btn.setVisible(True)
                card.open_folder_btn.setEnabled(True)
                card.process_btn.setEnabled(True)
                card.progress_bar.setVisible(False)
                
                # Update visualization if available
                if result and isinstance(result, dict):
                    visualization_files = []
                    
                    # Look for visualization files in output directory
                    if output_dir:
                        viz_extensions = ['.png', '.jpg', '.jpeg', '.pdf']
                        for ext in viz_extensions:
                            viz_files = list(Path(output_dir).glob(f"**/*{ext}"))
                            visualization_files.extend([str(f) for f in viz_files])
                    
                    # Also check if result contains visualization paths
                    if 'visualization_path' in result:
                        visualization_files.append(result['visualization_path'])
                    elif 'visualizations' in result:
                        visualization_files.extend(result['visualizations'])
                    
                    if visualization_files:
                        self.update_visualization(visualization_files)
                        self.console.append_message(f"📊 已加载 {len(visualization_files)} 个可视化文件", "success")
                
        except Exception as e:
            self.console.append_message(f"❌ 设置结果按钮失败：{str(e)}", "error")
    # ===== Voice Matching Methods =====
    
    def process_voice_matching(self):
        """Process voice matching"""
        try:
            print("🔥 process_voice_matching 被调用了!")
            self.console.append_message("🔥 开始执行人声匹配...", "info")
            
            # Check if processors are initialized
            if not self.processors.get('voice_matcher'):
                self.console.append_message("❌ 人声匹配器未初始化，请重启应用", "error")
                return
            
            # Get current active card
            card = self.get_current_card()
                
            if not card or not hasattr(card, 'reference_file') or not hasattr(card, 'candidate_files'):
                self.console.append_message("❌ 卡片信息错误", "error")
                return
                
            if not card.reference_file or not card.candidate_files:
                self.console.append_message("❌ 请先选择参考音频和待匹配音频", "error")
                return
                
            # Validate files exist
            if not os.path.exists(card.reference_file):
                self.console.append_message("❌ 参考音频文件不存在", "error")
                return
                
            for candidate_file in card.candidate_files:
                if not os.path.exists(candidate_file):
                    self.console.append_message(f"❌ 候选音频文件不存在: {Path(candidate_file).name}", "error")
                    return
                
            self.console.append_message(f"📁 参考音频: {Path(card.reference_file).name}", "info")
            self.console.append_message(f"📁 待匹配音频: {len(card.candidate_files)} 个文件", "info")
            
            # Store current card reference
            self.current_processing_card = card
            
            # Disable UI during processing
            card.process_btn.setEnabled(False)
            card.status_label.setText("正在处理...")
            card.progress_bar.setVisible(True)
            card.progress_bar.setValue(0)
            
            # Start processing in thread
            self.processing_thread = threading.Thread(
                target=self._process_voice_matching_thread,
                args=(card.reference_file, card.candidate_files)
            )
            self.processing_thread.start()
            
        except Exception as e:
            error_msg = f"❌ 启动人声匹配失败: {str(e)}"
            print(error_msg)
            self.console.append_message(error_msg, "error")
            import traceback
            traceback.print_exc()
        
    def _process_voice_matching_thread(self, reference_file, candidate_files):
        """Voice matching thread"""
        try:
            QTimer.singleShot(100, lambda: self.update_processing_status("🎯 开始人声匹配分析...", 10, "info"))
            
            # Process with VoiceMatcher
            QTimer.singleShot(500, lambda: self.update_processing_status("🔬 提取音频特征...", 30, "info"))
            
            # 创建输出目录
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            QTimer.singleShot(800, lambda: self.update_processing_status("🎵 计算相似度...", 60, "info"))
            result = self.processors['voice_matcher'].match_voices(
                reference_path=reference_file,
                candidate_paths=candidate_files,
                output_dir=output_dir
            )
            if result and result.get('success'):
                best_name = result.get('best_match_name', 'Unknown')
                best_score = result.get('best_score', 0)
                if isinstance(best_score, (int, float)):
                    score_text = f"{best_score:.3f}"
                else:
                    score_text = str(best_score)
                QTimer.singleShot(100, lambda: self.update_processing_status(f"✅ 人声匹配完成！最佳匹配：{best_name} (评分: {score_text})", 100, "success"))
                
                # Store results and show result buttons
                QTimer.singleShot(200, lambda: self.on_voice_matching_success(result, output_dir))
                
            else:
                QTimer.singleShot(100, lambda: self.update_processing_status(f"❌ 人声匹配失败：结果为空", None, "error"))
                
        except Exception as e:
            QTimer.singleShot(100, lambda: self.update_processing_status(f"❌ 人声匹配出错：{str(e)}", None, "error"))
        finally:
            # Re-enable UI
            QTimer.singleShot(300, lambda: self.reset_voice_matching_ui())
    
    def reset_voice_matching_ui(self):
        """Reset voice matching UI state"""
        try:
            if hasattr(self, 'current_processing_card') and self.current_processing_card:
                card = self.current_processing_card
                card.process_btn.setEnabled(True)
                card.status_label.setText("准备就绪")
                card.progress_bar.setVisible(False)
        except:
            pass
    def on_voice_matching_success(self, result, output_dir):
        """Handle successful voice matching completion"""
        try:
            if hasattr(self, 'current_processing_card') and self.current_processing_card:
                card = self.current_processing_card
                card.processing_results = result
                card.output_dir = output_dir
                card.view_results_btn.setVisible(True)
                card.view_results_btn.setEnabled(True)
                card.open_folder_btn.setVisible(True)
                card.open_folder_btn.setEnabled(True)
                card.process_btn.setEnabled(True)
                card.progress_bar.setVisible(False)
                
                # Update visualization if available
                if result and isinstance(result, dict):
                    visualization_files = []
                    
                    # Look for visualization files in output directory
                    if output_dir:
                        viz_extensions = ['.png', '.jpg', '.jpeg', '.pdf']
                        for ext in viz_extensions:
                            viz_files = list(Path(output_dir).glob(f"**/*{ext}"))
                            visualization_files.extend([str(f) for f in viz_files])
                    
                    # Also check if result contains visualization paths
                    if 'visualization_path' in result:
                        visualization_files.append(result['visualization_path'])
                    elif 'visualizations' in result:
                        visualization_files.extend(result['visualizations'])
                    
                    if visualization_files:
                        self.update_visualization(visualization_files)
                        self.console.append_message(f"📊 已加载 {len(visualization_files)} 个可视化文件", "success")
                
        except Exception as e:
            self.console.append_message(f"❌ 设置结果按钮失败：{str(e)}", "error")
    
    def show_custom_dialog(self, title: str, content: str, dialog_type: str = "info"):
        """Show a custom dialog with proper styling
        
        Args:
            title: Dialog title
            content: Dialog content
            dialog_type: Type of dialog ('info', 'warning', 'error')
        """
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setFixedSize(500, 350)
        
        # Icon mapping
        icons = {
            'info': '💡',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅'
        }
        icon = icons.get(dialog_type, '💡')
        
        dialog.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
            }
            QLabel {
                color: #374151;
                font-size: 14px;
                margin: 4px 0;
            }
            QTextEdit {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 12px;
                background-color: #f9fafb;
                color: #374151;
                font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #5855eb;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Title
        title_label = QLabel(f"{icon} {title}")
        title_label.setStyleSheet("font-size: 18px; font-weight: 600; color: #1f2937; margin-bottom: 8px;")
        layout.addWidget(title_label)
        
        # Content
        content_edit = QTextEdit()
        content_edit.setPlainText(content)
        content_edit.setReadOnly(True)
        layout.addWidget(content_edit)
        
        # OK button
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        layout.addLayout(button_layout)
        
        dialog.exec()
            
    # ===== Result Display Methods =====
    
    def view_preprocessing_results(self, card):
        """Display preprocessing results in detail dialog"""
        if not hasattr(card, 'processing_results') or not card.processing_results:
            QMessageBox.information(self, "提示", "暂无处理结果")
            return
            
        result = card.processing_results
        
        # Create detailed results dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("音频预处理结果")
        dialog.setFixedSize(600, 500)        
        dialog.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
            }
            QLabel {
                color: #374151;
                font-size: 14px;
                margin: 4px 0;
            }
            QTextEdit {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 12px;
                background-color: #f9fafb;
                color: #374151;
                font-family: 'Monaco', 'Consolas', monospace;
                font-size: 13px;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Title
        title = QLabel("🎛️ 音频预处理结果详情")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #1f2937; margin-bottom: 8px;")
        layout.addWidget(title)
        
        # Results content
        results_text = QTextEdit()
        results_text.setReadOnly(True)
        
        # Build results display
        content = []
        content.append("=" * 50)
        content.append("📊 处理结果摘要")
        content.append("=" * 50)
        
        if result.get('output_audio_path'):
            content.append(f"✅ 输出文件: {Path(result['output_audio_path']).name}")
            
        content.append(f"\n⏰ 处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("=" * 50)
        
        results_text.setPlainText("\n".join(content))
        layout.addWidget(results_text)
        
        # Close button
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #5b5bf6;
            }
        """)
        close_btn.clicked.connect(dialog.accept)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        dialog.exec()
    def view_source_separation_results(self, card):
        """Display source separation results"""
        if not hasattr(card, 'processing_results') or not card.processing_results:
            self.show_custom_dialog("提示", "暂无处理结果", "info")
            return
            
        result = card.processing_results
        
        content = f"""🎼 音源分离处理完成！

📁 输出目录: {result.get('output_dir', 'N/A')}

📊 处理结果:
• 输入文件: {Path(result.get('input_path', '')).name if result.get('input_path') else 'N/A'}
• 分离模型: {result.get('model_used', 'N/A')}

🎵 分离音轨:"""

        if result.get('separated_files'):
            for stem, path in result['separated_files'].items():
                content += f"\n• {stem.capitalize()}: {Path(path).name}"
        else:
            content += "\n• 暂无分离文件信息"
            
        content += f"\n\n📄 详细报告: {Path(result.get('report_path', '')).name if result.get('report_path') else 'N/A'}"
        
        self.show_custom_dialog("音源分离结果", content, "success")
        
    def view_speaker_separation_results(self, card):
        """Display speaker separation results"""
        if not hasattr(card, 'processing_results') or not card.processing_results:
            self.show_custom_dialog("提示", "暂无处理结果", "info")
            return
            
        result = card.processing_results
        
        content = f"""👥 说话人分离处理完成！

📁 输出目录: {result.get('output_dir', 'N/A')}

📊 处理结果:
• 输入文件: {Path(result.get('input_path', '')).name if result.get('input_path') else 'N/A'}
• 识别说话人数: {len(result.get('speakers', []))}

🗣️ 分离说话人:"""

        if result.get('separated_files'):
            for speaker, path in result['separated_files'].items():
                content += f"\n• {speaker}: {Path(path).name}"
        else:
            content += "\n• 暂无分离文件信息"
            
        content += f"\n\n📄 时间轴文件: {Path(result.get('diarization_file', '')).name if result.get('diarization_file') else 'N/A'}"
        content += f"\n📄 详细报告: {Path(result.get('report_path', '')).name if result.get('report_path') else 'N/A'}"
        
        self.show_custom_dialog("说话人分离结果", content, "success")
        
    def view_voice_matching_results(self, card):
        """Display voice matching results"""
        if not hasattr(card, 'processing_results') or not card.processing_results:
            self.show_custom_dialog("提示", "暂无处理结果", "info")
            return
            
        result = card.processing_results
        
        content = f"""🎯 人声匹配处理完成！

📁 输出目录: {result.get('output_dir', 'N/A')}

📊 匹配结果:
• 参考音频: {Path(result.get('reference_path', '')).name if result.get('reference_path') else 'N/A'}
• 候选音频数量: {len(result.get('candidate_paths', []))}"""

        best_match = result.get('best_match_name', 'N/A')
        best_score = result.get('best_score', 0)
        if isinstance(best_score, (int, float)):
            score_text = f"{best_score:.3f}"
        else:
            score_text = str(best_score)
            
        content += f"""

🏆 最佳匹配:
• 文件名: {best_match}
• 相似度评分: {score_text}
• 输出文件: {Path(result.get('best_match_output', '')).name if result.get('best_match_output') else 'N/A'}

📄 详细报告: {Path(result.get('report_path', '')).name if result.get('report_path') else 'N/A'}
📊 可视化图表: {Path(result.get('visualization_path', '')).name if result.get('visualization_path') else 'N/A'}"""
        
        self.show_custom_dialog("人声匹配结果", content, "success")
    
    def open_output_folder(self, card):
        """Open output folder in file explorer"""
        if not hasattr(card, 'output_dir') or not card.output_dir:
            self.show_custom_dialog("提示", "暂无输出目录信息", "info")
            return
            
        output_dir = Path(card.output_dir)
        if not output_dir.exists():
            self.show_custom_dialog("警告", f"输出目录不存在：{output_dir}", "warning")
            return
            
        try:
            # Windows
            if sys.platform == "win32":
                os.startfile(str(output_dir))
            # macOS
            elif sys.platform == "darwin":
                subprocess.run(["open", str(output_dir)])
            # Linux
            else:
                subprocess.run(["xdg-open", str(output_dir)])
                
            self.console.append_message(f"📁 已打开输出文件夹：{output_dir}", "success")
            
        except Exception as e:
            self.show_custom_dialog("错误", f"无法打开文件夹：{str(e)}", "error")
            self.console.append_message(f"❌ 打开文件夹失败：{str(e)}", "error")
            
    # ===== Utility Methods =====
    
    def update_processing_status(self, message, progress, level):
        """Update processing status and progress"""
        try:
            if hasattr(self, 'current_processing_card') and self.current_processing_card:
                card = self.current_processing_card
                card.status_label.setText(message)
                if progress is not None:
                    card.progress_bar.setValue(progress)
            
            self.console.append_message(message, level)
            
        except Exception as e:
            print(f"❌ 更新状态失败: {e}")
    
    def update_visualization(self, visualization_files):
        """Update visualization panel with images"""
        try:
            if not visualization_files:
                self.console.append_message("⚠️ 没有可视化文件需要加载", "warning")
                return
                
            self.console.append_message(f"📊 正在加载 {len(visualization_files)} 个可视化文件...", "info")
            
            # Ensure viz_panel exists and has content_layout
            if not hasattr(self, 'viz_panel') or not self.viz_panel:
                self.console.append_message("❌ 可视化面板未初始化", "error")
                return
                
            if not hasattr(self.viz_panel, 'content_layout') or not self.viz_panel.content_layout:
                self.console.append_message("❌ 可视化面板布局未初始化", "error")
                return
                
            # Clear existing visualizations safely
            while self.viz_panel.content_layout.count():
                child = self.viz_panel.content_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                        
            # Create matplotlib figure
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            fig = Figure(figsize=(12, 8), facecolor='white')
            canvas = FigureCanvas(fig)
            
            # Load and display visualizations
            valid_files = [f for f in visualization_files if os.path.exists(f) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
            
            if not valid_files:
                self.console.append_message("❌ 没有找到有效的可视化文件", "error")
                return
                
            if len(valid_files) == 1:
                # Single visualization
                img_path = valid_files[0]
                try:
                    import matplotlib.image as mpimg
                    img = mpimg.imread(img_path)
                    ax = fig.add_subplot(111)
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(Path(img_path).stem, fontsize=14, fontweight='bold')
                    self.console.append_message(f"✅ 已加载可视化: {Path(img_path).name}", "success")
                except Exception as e:
                    self.console.append_message(f"❌ 加载图片失败 {Path(img_path).name}: {e}", "error")
                    return
            else:
                # Multiple visualizations in subplots
                rows = (len(valid_files) + 1) // 2
                for i, img_path in enumerate(valid_files):
                    try:
                        import matplotlib.image as mpimg
                        img = mpimg.imread(img_path)
                        ax = fig.add_subplot(rows, 2, i+1)
                        ax.imshow(img)
                        ax.axis('off')
                        ax.set_title(Path(img_path).stem, fontsize=12, fontweight='bold')
                    except Exception as e:
                        self.console.append_message(f"❌ 加载图片失败 {Path(img_path).name}: {e}", "error")
                        
            fig.tight_layout()
            canvas.draw()
            
            # Add canvas to layout
            self.viz_panel.content_layout.addWidget(canvas)
            
            # Expand visualization panel if collapsed
            if hasattr(self.viz_panel, 'is_collapsed') and self.viz_panel.is_collapsed:
                self.viz_panel.toggle_collapse()
                
            self.console.append_message("✅ 可视化面板已更新", "success")
            
        except Exception as e:
            self.console.append_message(f"❌ 可视化更新失败：{str(e)}", "error")
            import traceback
            traceback.print_exc()
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Clean up any running threads
        if hasattr(self, 'processing_thread') and self.processing_thread.isRunning():
            self.console.append_message("⏹️ 正在停止处理线程...", "warning")
            self.processing_thread.quit()
            self.processing_thread.wait(3000)  # Wait up to 3 seconds
            
        # Restore original stdout/stderr
        if hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout
        if hasattr(self, 'original_stderr'):
            sys.stderr = self.original_stderr
            
        event.accept()


def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
        
    print(f"❌ 未处理的异常: {exc_type.__name__}: {exc_value}")
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_traceback)


def main():
    """Main function"""
    # Set global exception handler
    sys.excepthook = handle_exception
    
    try:
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("Voice Processing Suite")
        app.setApplicationVersion("2.0")
        app.setOrganizationName("Voice Processing Team")
        
        print("🚀 启动现代化Voice Processing Suite...")
        
        # Create and show main window
        print("🏗️ 创建主窗口...")
        window = ModernVoiceProcessingApp()
        
        print("🖼️ 显示窗口...")
        window.show()
        
        print("✅ 应用启动成功！")
        
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
