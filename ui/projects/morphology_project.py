from .base_project import BaseProject
from utils.image_processor import ImageProcessor
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class MorphologyProject(BaseProject):
    def __init__(self):
        super().__init__()
        self.setup_specific_ui()
        
    def setup_specific_ui(self):
        """设置形态学项目特定的UI元素"""
        control_layout = self.findChild(QVBoxLayout)
        
        # 结构元素设置
        kernel_group = QGroupBox("3-结构元素设置")
        kernel_layout = QHBoxLayout(kernel_group)
        
        self.kernel_size = QSpinBox()
        self.kernel_size.setRange(3, 15)
        self.kernel_size.setSingleStep(2)
        self.kernel_size.setValue(3)
        
        kernel_layout.addWidget(QLabel("核大小:"))
        kernel_layout.addWidget(self.kernel_size)
        
        # 基本操作组
        basic_group = QGroupBox("3-基本形态学操作")
        basic_layout = QVBoxLayout(basic_group)
        
        self.dilate_btn = QPushButton("膨胀")
        self.erode_btn = QPushButton("腐蚀")
        self.open_btn = QPushButton("开运算")
        self.close_btn = QPushButton("闭运算")
        
        basic_layout.addWidget(self.dilate_btn)
        basic_layout.addWidget(self.erode_btn)
        basic_layout.addWidget(self.open_btn)
        basic_layout.addWidget(self.close_btn)
        
        # 添加到控制面板
        control_layout.insertWidget(2, kernel_group)
        control_layout.insertWidget(3, basic_group)
        
        # 连接信号
        self.dilate_btn.clicked.connect(self.apply_dilation)
        self.erode_btn.clicked.connect(self.apply_erosion)
        self.open_btn.clicked.connect(self.apply_opening)
        self.close_btn.clicked.connect(self.apply_closing)
        
    def get_kernel(self):
        """获取当前设置的结构元素"""
        size = self.kernel_size.value()
        return np.ones((size, size), dtype=np.uint8)
        
    def apply_dilation(self):
        if not self.image:
            return

        def process():
            arr = self.qimage_to_numpy(self.image)
            kernel = self.get_kernel()
            result = ImageProcessor.binary_dilation(arr, kernel)
            self.update_result(result, "Dilation", {"kernel_size": kernel.shape[0]})

        self.process_with_loading(process)

    def apply_erosion(self):
        if not self.image:
            return

        def process():
            arr = self.qimage_to_numpy(self.image)
            kernel = self.get_kernel()
            result = ImageProcessor.binary_erosion(arr, kernel)
            self.update_result(result, "Erosion", {"kernel_size": kernel.shape[0]})

        self.process_with_loading(process)

    def apply_opening(self):
        """应用开运算"""
        if self.image:
            arr = self.qimage_to_numpy(self.image)
            result = ImageProcessor.morphological_opening(arr, self.get_kernel())
            self.processed_image = self.numpy_to_qimage(result)
            self.processed_label.setPixmap(
                QPixmap.fromImage(self.processed_image).scaled(400, 400, Qt.KeepAspectRatio))
            
    def apply_closing(self):
        """应用闭运算"""
        if self.image:
            arr = self.qimage_to_numpy(self.image)
            result = ImageProcessor.morphological_closing(arr, self.get_kernel())
            self.processed_image = self.numpy_to_qimage(result)
            self.processed_label.setPixmap(
                QPixmap.fromImage(self.processed_image).scaled(400, 400, Qt.KeepAspectRatio)) 