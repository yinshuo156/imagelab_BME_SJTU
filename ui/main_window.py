from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from .projects import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ImageLab')
        self.setMinimumSize(1200, 800)
        
        # 创建中心部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # 添加标题
        title_label = QLabel("ImageLab")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50; margin: 20px;")
        self.layout.addWidget(title_label)
        
        # 创建项目选择组
        self.create_project_group()
        
        # 创建主要内容区域
        self.stack_widget = QStackedWidget()
        self.layout.addWidget(self.stack_widget)
        
        # 添加各个项目页面
        self.project1 = HistogramProject()
        self.project2 = ConvolutionProject()
        self.project3 = MorphologyProject()
        self.project4 = SkeletonProject()
        
        self.stack_widget.addWidget(self.project1)
        self.stack_widget.addWidget(self.project2)
        self.stack_widget.addWidget(self.project3)
        self.stack_widget.addWidget(self.project4)
        
        # 添加版权信息
        self.add_copyright()
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                margin-top: 12px;
                padding: 15px;
                background-color: #fafafa;
            }
            QPushButton {
                background-color: #f8f9fa;
                color: #212529;
                border: 1px solid #dee2e6;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #dee2e6;
            }
            QPushButton:pressed {
                background-color: #dee2e6;
            }
            QLabel {
                color: #212529;
            }
        """)
        
    def create_project_group(self):
        group = QGroupBox("项目选择")
        layout = QHBoxLayout()
        
        projects = [
            ("1 - Histogram & Threshold", "直方图分析与阈值分割"),
            ("2 - Convolution & Filters", "卷积与滤波处理"),
            ("3 - Morphological Ops", "形态学操作"),
            ("4 - Skeleton & Distance", "骨架提取与距离变换")
        ]
        
        for i, (name, tooltip) in enumerate(projects):
            btn = QPushButton(name)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda x, idx=i: self.switch_project(idx))
            layout.addWidget(btn)
            
        group.setLayout(layout)
        self.layout.addWidget(group)
        
    def switch_project(self, index):
        """切换项目并更新UI"""
        self.stack_widget.setCurrentIndex(index)
        
    def add_copyright(self):
        copyright = QLabel("© 2024 Shuo Yin (yinelon@gmail.com) All Rights Reserved")
        copyright.setAlignment(Qt.AlignCenter)
        copyright.setStyleSheet("color: #7f8c8d; padding: 10px;")
        self.layout.addWidget(copyright) 