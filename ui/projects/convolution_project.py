from .base_project import BaseProject
from utils.image_processor import ImageProcessor
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class ConvolutionProject(BaseProject):
    def __init__(self):
        super().__init__()
        self.setup_specific_ui()
        
    def setup_specific_ui(self):
        """设置卷积项目特定的UI元素"""
        control_layout = self.findChild(QVBoxLayout)
        
        # 边缘检测组
        edge_group = QGroupBox("2-边缘检测")
        edge_layout = QVBoxLayout(edge_group)
        
        self.roberts_btn = QPushButton("Roberts算子")
        self.prewitt_btn = QPushButton("Prewitt算子")
        self.sobel_btn = QPushButton("Sobel算子")
        
        self.roberts_btn.clicked.connect(lambda: self.apply_edge_detection('roberts'))
        self.prewitt_btn.clicked.connect(lambda: self.apply_edge_detection('prewitt'))
        self.sobel_btn.clicked.connect(lambda: self.apply_edge_detection('sobel'))
        
        edge_layout.addWidget(self.roberts_btn)
        edge_layout.addWidget(self.prewitt_btn)
        edge_layout.addWidget(self.sobel_btn)
        
        # 滤波组
        filter_group = QGroupBox("2-图像滤波")
        filter_layout = QVBoxLayout(filter_group)
        
        # 高斯滤波参数
        gaussian_layout = QHBoxLayout()
        gaussian_layout.addWidget(QLabel("高斯滤波:"))
        self.gaussian_size = QSpinBox()
        self.gaussian_size.setRange(3, 15)
        self.gaussian_size.setSingleStep(2)
        self.gaussian_size.setValue(3)
        self.gaussian_sigma = QDoubleSpinBox()
        self.gaussian_sigma.setRange(0.1, 5.0)
        self.gaussian_sigma.setSingleStep(0.1)
        self.gaussian_sigma.setValue(1.0)
        self.gaussian_btn = QPushButton("应用高斯滤波")
        
        gaussian_layout.addWidget(QLabel("核大小:"))
        gaussian_layout.addWidget(self.gaussian_size)
        gaussian_layout.addWidget(QLabel("Sigma:"))
        gaussian_layout.addWidget(self.gaussian_sigma)
        gaussian_layout.addWidget(self.gaussian_btn)
        
        # 中值滤波参数
        median_layout = QHBoxLayout()
        median_layout.addWidget(QLabel("中值滤波:"))
        self.median_size = QSpinBox()
        self.median_size.setRange(3, 15)
        self.median_size.setSingleStep(2)
        self.median_size.setValue(3)
        self.median_btn = QPushButton("应用中值滤波")
        
        median_layout.addWidget(QLabel("核大小:"))
        median_layout.addWidget(self.median_size)
        median_layout.addWidget(self.median_btn)
        
        filter_layout.addLayout(gaussian_layout)
        filter_layout.addLayout(median_layout)
        
        # 添加到控制面板
        control_layout.insertWidget(2, edge_group)
        control_layout.insertWidget(3, filter_group)
        
        # 连接信号
        self.gaussian_btn.clicked.connect(self.apply_gaussian)
        self.median_btn.clicked.connect(self.apply_median)
        
    def apply_edge_detection(self, operator):
        if not self.image:
            return

        def process():
            arr = self.qimage_to_numpy(self.image)
            result = ImageProcessor.edge_detection(arr, operator)
            self.update_result(result, f"{operator.title()} Edge Detection")

        self.process_with_loading(process)

    def apply_gaussian(self):
        if not self.image:
            return

        def process():
            arr = self.qimage_to_numpy(self.image)
            result = ImageProcessor.gaussian_filter(
                arr,
                self.gaussian_size.value(),
                self.gaussian_sigma.value()
            )
            params = {
                "kernel_size": self.gaussian_size.value(),
                "sigma": self.gaussian_sigma.value()
            }
            self.update_result(result, "Gaussian Filter", params)

        self.process_with_loading(process)

    def apply_median(self):
        """应用中值滤波"""
        if self.image:
            arr = self.qimage_to_numpy(self.image)
            result = ImageProcessor.median_filter(
                arr,
                self.median_size.value()
            )
            self.processed_image = self.numpy_to_qimage(result)
            self.processed_label.setPixmap(
                QPixmap.fromImage(self.processed_image).scaled(400, 400, Qt.KeepAspectRatio)) 