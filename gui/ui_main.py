from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class GlassCard(QWidget):
    """毛玻璃效果卡片组件"""
    
    def __init__(self, title="", icon="", parent=None):
        super().__init__(parent)
        self.title = title
        self.icon = icon
        self.hover_animation = None
        self.setup_ui()
        self.setup_animations()
    
    def setup_ui(self):
        """设置UI"""
        self.setFixedHeight(180)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(24, 24, 24, 24)
        self.layout.setSpacing(16)
        
        # 标题行
        if self.title or self.icon:
            title_layout = QHBoxLayout()
            
            if self.icon:
                icon_label = QLabel(self.icon)
                icon_label.setStyleSheet("""
                    QLabel {
                        font-size: 24px;
                        color: #667eea;
                    }
                """)
                title_layout.addWidget(icon_label)
            
            if self.title:
                title_label = QLabel(self.title)
                title_label.setStyleSheet("""
                    QLabel {
                        color: #2d3748;
                        font-size: 18px;
                        font-weight: 700;
                        background: transparent;
                    }
                """)
                title_layout.addWidget(title_label)
            
            title_layout.addStretch()
            self.layout.addLayout(title_layout)
    
    def setup_animations(self):
        """设置动画效果"""
        self.hover_animation = QPropertyAnimation(self, b"geometry")
        self.hover_animation.setDuration(300)
        self.hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    def add_widget(self, widget):
        """添加子组件"""
        self.layout.addWidget(widget)
    
    def add_layout(self, layout):
        """添加布局"""
        self.layout.addLayout(layout)
    
    def paintEvent(self, event):
        """自定义绘制毛玻璃效果"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 毛玻璃背景
        rect = self.rect()
        painter.setBrush(QBrush(QColor(255, 255, 255, 220)))
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        painter.drawRoundedRect(rect, 20, 20)
        
        # 边框光效
        gradient = QLinearGradient(0, 0, rect.width(), rect.height())
        gradient.setColorAt(0, QColor(103, 126, 234, 80))
        gradient.setColorAt(1, QColor(118, 75, 162, 80))
        painter.setPen(QPen(QBrush(gradient), 2))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 19, 19)

class ModernButton(QPushButton):
    """现代化按钮组件"""
    
    def __init__(self, text="", icon="", button_type="primary", parent=None):
        super().__init__(text, parent)
        self.button_type = button_type
        self.icon = icon
        self.click_animation = None
        self.setup_animations()
        self.apply_style()
    
    def setup_animations(self):
        """设置动画"""
        self.click_animation = QPropertyAnimation(self, b"geometry")
        self.click_animation.setDuration(150)
        self.click_animation.setEasingCurve(QEasingCurve.Type.OutBack)
    
    def apply_style(self):
        """应用样式"""
        base_style = """
            QPushButton {
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                color: white;
                min-height: 20px;
            }
        """
        
        if self.button_type == "primary":
            style = base_style + """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                               stop:0 #667eea, stop:1 #764ba2);
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                               stop:0 #5a67d8, stop:1 #6b46c1);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                               stop:0 #4c51bf, stop:1 #553c9a);
                }
            """
        elif self.button_type == "success":
            style = base_style + """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                               stop:0 #48bb78, stop:1 #38a169);
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                               stop:0 #38a169, stop:1 #2f855a);
                }
            """
        elif self.button_type == "danger":
            style = base_style + """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                               stop:0 #f56565, stop:1 #e53e3e);
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                               stop:0 #e53e3e, stop:1 #c53030);
                }
            """
        
        self.setStyleSheet(style)
    
    def mousePressEvent(self, event):
        """鼠标按下动画"""
        if self.click_animation:
            original = self.geometry()
            smaller = QRect(
                original.x() + 2, original.y() + 2,
                original.width() - 4, original.height() - 4
            )
            self.click_animation.setStartValue(original)
            self.click_animation.setEndValue(smaller)
            self.click_animation.start()
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放动画"""
        if self.click_animation:
            QTimer.singleShot(100, lambda: self.click_animation.start())
        super().mouseReleaseEvent(event)

class ModernSlider(QWidget):
    """现代化滑块组件"""
    valueChanged = pyqtSignal(int)
    
    def __init__(self, min_val=0, max_val=100, default_val=50, suffix="", label="", parent=None):
        super().__init__(parent)
        self.suffix = suffix
        self.label = label
        self.setup_ui(min_val, max_val, default_val)
    
    def setup_ui(self, min_val, max_val, default_val):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)  # 增加间距
        layout.setContentsMargins(0, 8, 0, 8)  # 增加上下边距
        
        # 标签和数值显示
        if self.label:
            header_layout = QHBoxLayout()
            
            label_widget = QLabel(self.label)
            label_widget.setStyleSheet("""
                QLabel {
                    color: #4a5568;
                    font-size: 14px;
                    font-weight: 600;
                    background: transparent;
                    padding: 4px 0;
                }
            """)
            
            self.value_label = QLabel(f"{default_val}{self.suffix}")
            self.value_label.setStyleSheet("""
                QLabel {
                    color: #667eea;
                    font-size: 14px;
                    font-weight: 700;
                    background: rgba(102, 126, 234, 0.1);
                    padding: 6px 14px;
                    border-radius: 10px;
                    min-width: 60px;
                    text-align: center;
                }
            """)
            
            header_layout.addWidget(label_widget)
            header_layout.addStretch()
            header_layout.addWidget(self.value_label)
            layout.addLayout(header_layout)
        
        # 滑块容器 - 增加高度
        slider_container = QWidget()
        slider_container.setFixedHeight(40)  # 增加高度
        slider_layout = QVBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 8, 0, 8)
        
        # 滑块
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default_val)
        self.slider.setFixedHeight(24)  # 设置固定高度
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #e2e8f0;
                height: 10px;
                border-radius: 5px;
                margin: 0px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #667eea, stop:1 #764ba2);
                border: 4px solid white;
                width: 24px;
                height: 24px;
                margin: -7px 0;
                border-radius: 16px;
                box-shadow: 0 3px 6px rgba(0,0,0,0.2);
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #5a67d8, stop:1 #6b46c1);
                border: 4px solid white;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }
            QSlider::handle:horizontal:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #4c51bf, stop:1 #553c9a);
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                           stop:0 #667eea, stop:1 #764ba2);
                border-radius: 5px;
            }
            QSlider::add-page:horizontal {
                background: #e2e8f0;
                border-radius: 5px;
            }
        """)
        
        self.slider.valueChanged.connect(self.on_value_changed)
        slider_layout.addWidget(self.slider)
        layout.addWidget(slider_container)

    def on_value_changed(self, value):
        """值变化处理"""
        if hasattr(self, 'value_label'):
            self.value_label.setText(f"{value}{self.suffix}")
        self.valueChanged.emit(value)
    
    def value(self):
        return self.slider.value()
    
    def setValue(self, value):
        self.slider.setValue(value)

class ModernProgressBar(QProgressBar):
    """现代化进度条"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(8)
        self.setStyleSheet("""
            QProgressBar {
                background: #e2e8f0;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                           stop:0 #667eea, stop:1 #764ba2);
                border-radius: 4px;
            }
        """)

class ModernTabWidget(QTabWidget):
    """现代化标签页组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
                margin-top: 8px;
            }
            QTabBar::tab {
                background: transparent;
                color: #718096;
                padding: 12px 24px;
                margin-right: 8px;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #667eea, stop:1 #764ba2);
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: rgba(102, 126, 234, 0.1);
                color: #667eea;
            }
        """)

class CustomTitleBar(QWidget):
    """自定义标题栏"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(50)
        self.setup_ui()
        self.start_pos = None
        
    def setup_ui(self):
        """设置标题栏UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(15)
        
        # 应用图标和标题
        icon_title_layout = QHBoxLayout()
        
        # 图标
        icon_label = QLabel("🎵")
        icon_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: #667eea;
                background: transparent;
            }
        """)
        
        # 标题
        title_label = QLabel("Voice Enhancement System")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 16px;
                font-weight: 700;
                background: transparent;
                padding-left: 8px;
            }
        """)
        
        icon_title_layout.addWidget(icon_label)
        icon_title_layout.addWidget(title_label)
        
        # 窗口控制按钮
        control_layout = QHBoxLayout()
        control_layout.setSpacing(8)
        
        # 最小化按钮
        self.min_btn = QPushButton("−")
        self.min_btn.setFixedSize(35, 35)
        self.min_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 17px;
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.3);
            }
        """)
        self.min_btn.clicked.connect(self.minimize_window)
        
        # 最大化按钮
        self.max_btn = QPushButton("□")
        self.max_btn.setFixedSize(35, 35)
        self.max_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 17px;
                color: white;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.3);
            }
        """)
        self.max_btn.clicked.connect(self.toggle_maximize)
        
        # 关闭按钮
        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(35, 35)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 17px;
                color: white;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ff5f57;
            }
            QPushButton:pressed {
                background: #e04b42;
            }
        """)
        self.close_btn.clicked.connect(self.close_window)
        
        control_layout.addWidget(self.min_btn)
        control_layout.addWidget(self.max_btn)
        control_layout.addWidget(self.close_btn)
        
        layout.addLayout(icon_title_layout)
        layout.addStretch()
        layout.addLayout(control_layout)
        
        # 设置标题栏样式
        self.setStyleSheet("""
            CustomTitleBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #667eea, stop:1 #764ba2);
                border-top-left-radius: 15px;
                border-top-right-radius: 15px;
            }
        """)
    
    def minimize_window(self):
        """最小化窗口"""
        if self.parent:
            self.parent.showMinimized()
    
    def toggle_maximize(self):
        """切换最大化状态"""
        if self.parent:
            if self.parent.isMaximized():
                self.parent.showNormal()
                self.max_btn.setText("□")
            else:
                self.parent.showMaximized()
                self.max_btn.setText("❐")
    
    def close_window(self):
        """关闭窗口"""
        if self.parent:
            self.parent.close()
    
    def mousePressEvent(self, event):
        """鼠标按下事件 - 开始拖拽"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = event.globalPosition().toPoint()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 拖拽窗口"""
        if event.buttons() == Qt.MouseButton.LeftButton and self.start_pos:
            if self.parent and not self.parent.isMaximized():
                delta = event.globalPosition().toPoint() - self.start_pos
                self.parent.move(self.parent.pos() + delta)
                self.start_pos = event.globalPosition().toPoint()
    
    def mouseDoubleClickEvent(self, event):
        """双击事件 - 切换最大化"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_maximize()

class Ui_MainWindow:
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 1000)
        MainWindow.setMinimumSize(QSize(1400, 900))
        
        # 设置无边框窗口
        MainWindow.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        MainWindow.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # 主容器
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # 设置主容器样式
        self.centralwidget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #f7fafc, stop:0.5 #edf2f7, stop:1 #e2e8f0);
                font-family: 'SF Pro Display', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                border-radius: 15px;
            }
        """)
        
        # 主布局
        self.main_layout = QVBoxLayout(self.centralwidget)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 添加自定义标题栏
        self.title_bar = CustomTitleBar(MainWindow)
        self.main_layout.addWidget(self.title_bar)
        
        # 内容区域
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(32)
        content_layout.setContentsMargins(40, 20, 40, 40)
        
        # 创建界面组件
        self.create_header(content_layout)
        self.create_content_area(content_layout)
        
        self.main_layout.addWidget(content_widget)
        self.create_status_bar(MainWindow)
        
        MainWindow.setCentralWidget(self.centralwidget)
    
    def create_header(self, parent_layout):
        """创建顶部标题区域"""
        header_layout = QHBoxLayout()
        
        # 左侧标题
        title_layout = QVBoxLayout()
        
        main_title = QLabel("🎵 Voice Enhancement System")
        main_title.setStyleSheet("""
            QLabel {
                color: #2d3748;
                font-size: 28px;
                font-weight: 800;
                background: transparent;
                margin: 0;
            }
        """)
        
        subtitle = QLabel("Advanced Signal Processing & AI-Powered Voice Extraction")
        subtitle.setStyleSheet("""
            QLabel {
                color: #718096;
                font-size: 15px;
                font-weight: 500;
                background: transparent;
                margin: 0;
            }
        """)
        
        title_layout.addWidget(main_title)
        title_layout.addWidget(subtitle)
        title_layout.setSpacing(6)
        
        # 右侧知识点标签
        knowledge_layout = QHBoxLayout()
        knowledge_points = ["LTI System", "FFT Analysis", "Bandpass Filter", "Convolution"]
        
        for point in knowledge_points:
            label = QLabel(point)
            label.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                               stop:0 #667eea, stop:1 #764ba2);
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: 600;
                }
            """)
            knowledge_layout.addWidget(label)
        
        knowledge_layout.addStretch()
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        header_layout.addLayout(knowledge_layout)
        
        parent_layout.addLayout(header_layout)
    
    def create_content_area(self, parent_layout):
        """创建主内容区域"""
        content_layout = QHBoxLayout()
        content_layout.setSpacing(32)
        
        # 左侧控制面板
        self.create_control_panel(content_layout)
        
        # 右侧可视化面板
        self.create_visualization_panel(content_layout)
        
        self.main_layout.addLayout(content_layout)
    
    def create_control_panel(self, parent_layout):
        """创建左侧控制面板"""
        control_container = QWidget()
        control_container.setFixedWidth(450)  # 增加宽度以适应更大的滑块
        control_layout = QVBoxLayout(control_container)
        control_layout.setSpacing(10)  # 减小间距，让卡片排列更紧凑
        
        # 文件选择卡片
        file_card = GlassCard("File Selection", "📁")
        file_card.setFixedHeight(200)  # 增加高度
        
        self.input_button = ModernButton("🎵 Select Mixed Audio", button_type="primary")
        self.input_label = QLabel("No audio file selected")
        self.input_label.setStyleSheet("""
            QLabel {
                color: #a0aec0;
                font-size: 13px;
                background: transparent;
                padding: 8px 0;
            }
        """)
        
        self.reference_button = ModernButton("🎤 Select Reference Voice", button_type="success")
        self.reference_label = QLabel("No reference file selected")
        self.reference_label.setStyleSheet("""
            QLabel {
                color: #a0aec0;
                font-size: 13px;
                background: transparent;
                padding: 8px 0;
            }
        """)
        
        file_card.add_widget(self.input_button)
        file_card.add_widget(self.input_label)
        file_card.add_widget(self.reference_button)
        file_card.add_widget(self.reference_label)
        
        # 参数控制卡片
        params_card = GlassCard("Filter Parameters", "⚙️")
        params_card.setFixedHeight(260)  # 增加高度以适应更大的滑块，确保内容完全显示
        
        self.low_freq_slider = ModernSlider(20, 2000, 300, " Hz", "Low-cut Frequency")
        self.high_freq_slider = ModernSlider(2000, 8000, 3400, " Hz", "High-cut Frequency")
        
        params_card.add_widget(self.low_freq_slider)
        params_card.add_widget(self.high_freq_slider)
        
        # 处理控制卡片
        process_card = GlassCard("Processing Control", "🚀")
        process_card.setFixedHeight(160)  # 设置固定高度以避免遮盖其他卡片
        
        self.process_button = ModernButton("🚀 Start Voice Extraction", button_type="primary")
        self.process_button.setFixedHeight(50)
        self.process_button.setStyleSheet(self.process_button.styleSheet() + """
            QPushButton {
                font-size: 16px;
                font-weight: 700;
            }
        """)
        
        self.progress_bar = ModernProgressBar()
        self.progress_bar.setVisible(False)
        
        process_card.add_widget(self.process_button)
        process_card.add_widget(self.progress_bar)
        
        # 播放控制卡片
        playback_card = GlassCard("Playback Control", "🎧")
        playback_card.setFixedHeight(140)  # 设置固定高度以保持布局一致性
        
        playback_layout = QHBoxLayout()
        self.play_original_button = ModernButton("▶️ Original", button_type="primary")
        self.play_processed_button = ModernButton("▶️ Processed", button_type="success")
        self.play_processed_button.setEnabled(False)
        
        playback_layout.addWidget(self.play_original_button)
        playback_layout.addWidget(self.play_processed_button)
        playback_card.add_layout(playback_layout)
        
        # 添加所有卡片
        control_layout.addWidget(file_card)
        control_layout.addWidget(params_card)
        control_layout.addWidget(process_card)
        control_layout.addWidget(playback_card)
        control_layout.addStretch()
        
        parent_layout.addWidget(control_container)
    
    def create_visualization_panel(self, parent_layout):
        """创建右侧可视化面板"""
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setSpacing(24)
        
        # 可视化标题
        viz_title = QLabel("Real-time Audio Analysis")
        viz_title.setStyleSheet("""
            QLabel {
                color: #2d3748;
                font-size: 20px;
                font-weight: 700;
                background: transparent;
                margin-bottom: 16px;
            }
        """)
        viz_layout.addWidget(viz_title)
        
        # 标签页组件
        self.tab_widget = ModernTabWidget()
        
        # 1. 波形分析
        self.waveform_widget = QWidget()
        waveform_layout = QVBoxLayout(self.waveform_widget)
        self.waveform_figure = Figure(figsize=(10, 5), dpi=100)
        self.waveform_figure.patch.set_facecolor((0,0,0,0)) # 修正透明背景
        self.waveform_canvas = FigureCanvas(self.waveform_figure)
        self.waveform_canvas.setStyleSheet("background: transparent;")
        waveform_layout.addWidget(self.waveform_canvas)
        self.tab_widget.addTab(self.waveform_widget, "Waveform")
        
        # 2. 频谱分析
        self.spectrum_widget = QWidget()
        spectrum_layout = QVBoxLayout(self.spectrum_widget)
        self.spectrum_figure = Figure(figsize=(10, 5), dpi=100)
        self.spectrum_figure.patch.set_facecolor((0,0,0,0)) # 修正透明背景
        self.spectrum_canvas = FigureCanvas(self.spectrum_figure)
        self.spectrum_canvas.setStyleSheet("background: transparent;")
        spectrum_layout.addWidget(self.spectrum_canvas)
        self.tab_widget.addTab(self.spectrum_widget, "Spectrum")
        
        # 3. 滤波器响应
        self.filter_widget = QWidget()
        filter_layout = QVBoxLayout(self.filter_widget)
        self.filter_figure = Figure(figsize=(10, 5), dpi=100)
        self.filter_figure.patch.set_facecolor((0,0,0,0)) # 修正透明背景
        self.filter_canvas = FigureCanvas(self.filter_figure)
        self.filter_canvas.setStyleSheet("background: transparent;")
        filter_layout.addWidget(self.filter_canvas)
        self.tab_widget.addTab(self.filter_widget, "Filter Response")
        
        # 4. 时频分析
        self.spectrogram_widget = QWidget()
        spectrogram_layout = QVBoxLayout(self.spectrogram_widget)
        self.spectrogram_figure = Figure(figsize=(10, 5), dpi=100)
        self.spectrogram_figure.patch.set_facecolor((0,0,0,0)) # 修正透明背景
        self.spectrogram_canvas = FigureCanvas(self.spectrogram_figure)
        self.spectrogram_canvas.setStyleSheet("background: transparent;")
        spectrogram_layout.addWidget(self.spectrogram_canvas)
        self.tab_widget.addTab(self.spectrogram_widget, "Spectrogram")
        
        viz_layout.addWidget(self.tab_widget)
        parent_layout.addWidget(viz_container)
        
        # 初始化图表
        self.init_plots()
    
    def init_plots(self):
        """初始化所有图表"""
        plt.style.use('default')
        
        # 设置现代化颜色主题
        colors = {
            'primary': '#667eea',
            'success': '#48bb78',
            'warning': '#ed8936',
            'danger': '#f56565',
            'info': '#4299e1',
            'text': '#2d3748',
            'muted': '#718096'
        }
        
        # 1. 波形图
        self.waveform_ax = self.waveform_figure.add_subplot(111)
        self.waveform_ax.set_facecolor((0,0,0,0)) # 修正透明背景
        self.waveform_ax.set_title('Time Domain Analysis', fontsize=16, fontweight='bold', 
                                   color=colors['text'], pad=20)
        self.waveform_ax.set_xlabel('Time (s)', fontsize=12, color=colors['muted'])
        self.waveform_ax.set_ylabel('Amplitude', fontsize=12, color=colors['muted'])
        self.waveform_ax.grid(True, alpha=0.2, color=colors['muted'])
        self.waveform_ax.spines['top'].set_visible(False)
        self.waveform_ax.spines['right'].set_visible(False)
        for spine in self.waveform_ax.spines.values():
            spine.set_color(colors['muted'])
            spine.set_alpha(0.3)
        self.waveform_figure.tight_layout()
        
        # 2. 频谱图
        self.spectrum_ax = self.spectrum_figure.add_subplot(111)
        self.spectrum_ax.set_facecolor((0,0,0,0)) # 修正透明背景
        self.spectrum_ax.set_title('Frequency Domain Analysis', fontsize=16, fontweight='bold',
                                   color=colors['text'], pad=20)
        self.spectrum_ax.set_xlabel('Frequency (Hz)', fontsize=12, color=colors['muted'])
        self.spectrum_ax.set_ylabel('Magnitude (dB)', fontsize=12, color=colors['muted'])
        self.spectrum_ax.grid(True, alpha=0.2, color=colors['muted'])
        self.spectrum_ax.spines['top'].set_visible(False)
        self.spectrum_ax.spines['right'].set_visible(False)
        for spine in self.spectrum_ax.spines.values():
            spine.set_color(colors['muted'])
            spine.set_alpha(0.3)
        self.spectrum_figure.tight_layout()
        
        # 3. 滤波器响应图
        self.filter_ax = self.filter_figure.add_subplot(111)
        self.filter_ax.set_facecolor((0,0,0,0)) # 修正透明背景
        self.filter_ax.set_title('Bandpass Filter Response', fontsize=16, fontweight='bold',
                                 color=colors['text'], pad=20)
        self.filter_ax.set_xlabel('Frequency (Hz)', fontsize=12, color=colors['muted'])
        self.filter_ax.set_ylabel('Magnitude (dB)', fontsize=12, color=colors['muted'])
        self.filter_ax.grid(True, alpha=0.2, color=colors['muted'])
        self.filter_ax.spines['top'].set_visible(False)
        self.filter_ax.spines['right'].set_visible(False)
        for spine in self.filter_ax.spines.values():
            spine.set_color(colors['muted'])
            spine.set_alpha(0.3)
        self.filter_figure.tight_layout()
        
        # 4. 时频图
        self.spectrogram_ax = self.spectrogram_figure.add_subplot(111)
        self.spectrogram_ax.set_facecolor((0,0,0,0)) # 修正透明背景
        self.spectrogram_ax.set_title('Time-Frequency Analysis', fontsize=16, fontweight='bold',
                                      color=colors['text'], pad=20)
        self.spectrogram_ax.set_xlabel('Time (s)', fontsize=12, color=colors['muted'])
        self.spectrogram_ax.set_ylabel('Frequency (Hz)', fontsize=12, color=colors['muted'])
        self.spectrogram_ax.spines['top'].set_visible(False)
        self.spectrogram_ax.spines['right'].set_visible(False)
        for spine in self.spectrogram_ax.spines.values():
            spine.set_color(colors['muted'])
            spine.set_alpha(0.3)
        self.spectrogram_figure.tight_layout()
        
        # 更新所有画布
        for canvas in [self.waveform_canvas, self.spectrum_canvas, 
                      self.filter_canvas, self.spectrogram_canvas]:
            canvas.draw()
    
    def create_status_bar(self, MainWindow):
        """创建状态栏"""
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setStyleSheet("""
            QStatusBar {
                background: rgba(255, 255, 255, 0.8);
                color: #4a5568;
                border: none;
                padding: 8px 16px;
                font-size: 13px;
            }
        """)
        self.statusbar.showMessage("System Ready - Select audio files to begin processing")
        MainWindow.setStatusBar(self.statusbar)
