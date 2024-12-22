from .base_project import BaseProject
from utils.image_processor import ImageProcessor
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class SkeletonProject(BaseProject):
    def __init__(self):
        super().__init__()
        self.setup_specific_ui()
        self.binary_image = None
        self.skeleton_image = None
        self.distance_map = None
        
    def setup_specific_ui(self):
        """设置骨架提取项目特定的UI元素"""
        control_layout = self.findChild(QVBoxLayout)
        
        # 预处理组
        preprocess_group = QGroupBox("预处理")
        preprocess_layout = QVBoxLayout()
        
        # Otsu二值化
        self.otsu_btn = QPushButton("Otsu二值化")
        self.otsu_btn.clicked.connect(self.apply_otsu)
        preprocess_layout.addWidget(self.otsu_btn)
        
        # 手动二值化
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(127)
        self.threshold_value = QSpinBox()
        self.threshold_value.setRange(0, 255)
        self.threshold_value.setValue(127)
        
        self.threshold_slider.valueChanged.connect(self.threshold_value.setValue)
        self.threshold_value.valueChanged.connect(self.threshold_slider.setValue)
        self.threshold_slider.valueChanged.connect(self.apply_manual_threshold)
        
        threshold_layout.addWidget(QLabel("手动阈值:"))
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value)
        preprocess_layout.addLayout(threshold_layout)
        
        preprocess_group.setLayout(preprocess_layout)
        
        # 形态学操作组
        morph_group = QGroupBox("形态学操作")
        morph_layout = QVBoxLayout()
        
        # 进度显示
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(100)
        morph_layout.addWidget(self.progress_text)
        
        # 操作按钮
        self.distance_btn = QPushButton("1. Distance Transform")
        self.skeleton_btn = QPushButton("2. Skeleton Extraction")
        self.restore_btn = QPushButton("3. Skeleton Restoration")
        
        morph_layout.addWidget(self.distance_btn)
        morph_layout.addWidget(self.skeleton_btn)
        morph_layout.addWidget(self.restore_btn)
        
        morph_group.setLayout(morph_layout)
        
        # 添加到控制面板
        control_layout.insertWidget(2, preprocess_group)
        control_layout.insertWidget(3, morph_group)
        
        # 连接信号
        self.distance_btn.clicked.connect(self.apply_distance_transform)
        self.skeleton_btn.clicked.connect(self.apply_skeleton)
        self.restore_btn.clicked.connect(self.apply_restoration)
        
    def log_progress(self, text):
        """更新进度显示"""
        self.progress_text.append(text)
        self.progress_text.verticalScrollBar().setValue(
            self.progress_text.verticalScrollBar().maximum()
        )
        QApplication.processEvents()
        
    def apply_otsu(self):
        """应用Otsu二值化"""
        if not self.image:
            return
            
        try:
            self.log_progress("正在进行Otsu二值化...")
            arr = self.qimage_to_numpy(self.image)
            threshold = ImageProcessor.otsu_threshold(arr)
            self.binary_image = ImageProcessor.manual_threshold(arr, threshold)
            self.update_result(self.binary_image, "Binary")
            self.threshold_value.setValue(threshold)
            self.log_progress(f"二值化完成，阈值: {threshold}")
            
        except Exception as e:
            self.log_progress(f"错误: {str(e)}")
            QMessageBox.critical(self, "错误", str(e))
            
    def apply_manual_threshold(self):
        """应用手动二值化"""
        if not self.image:
            return
            
        try:
            arr = self.qimage_to_numpy(self.image)
            threshold = self.threshold_value.value()
            self.binary_image = ImageProcessor.manual_threshold(arr, threshold)
            self.update_result(self.binary_image, "Binary")
            self.log_progress(f"手动二值化完成，阈值: {threshold}")
            
        except Exception as e:
            self.log_progress(f"错误: {str(e)}")
            QMessageBox.critical(self, "错误", str(e))
            
    def apply_distance_transform(self):
        """应用距离变换"""
        if self.binary_image is None:
            QMessageBox.warning(self, "警告", "请先进行二值化处理！")
            return
            
        try:
            self.log_progress("开始计算距离变换...")
            self.distance_map = ImageProcessor.distance_transform(self.binary_image)
            self.update_result(self.distance_map, "Distance")
            self.log_progress("距离变换计算完成")
            
        except Exception as e:
            self.log_progress(f"错误: {str(e)}")
            QMessageBox.critical(self, "错误", str(e))
            
    def apply_skeleton(self):
        """应用骨架提取"""
        if self.binary_image is None:
            QMessageBox.warning(self, "警告", "请先进行二值化处理！")
            return
            
        try:
            self.log_progress("开始提取骨架...")
            
            # 使用优化后的骨架提取算法
            def progress_callback(iteration):
                self.log_progress(f"骨架提取迭代 {iteration}")
                
            self.skeleton_image = ImageProcessor.fast_skeleton(
                self.binary_image,
                progress_callback
            )
            self.update_result(self.skeleton_image, "Skeleton")
            self.log_progress("骨架提取完成")
            
        except Exception as e:
            self.log_progress(f"错误: {str(e)}")
            QMessageBox.critical(self, "错误", str(e))
            
    def apply_restoration(self):
        """应用骨架重建"""
        if self.skeleton_image is None or self.distance_map is None:
            QMessageBox.warning(self, "警告", "请先完成距离变换和骨架提取！")
            return
            
        try:
            self.log_progress("开始骨架重建...")
            
            def progress_callback(current, total):
                if current % 100 == 0:
                    self.log_progress(f"重建进度: {current}/{total}")
                    
            result = ImageProcessor.skeleton_restoration(
                self.skeleton_image,
                self.distance_map,
                progress_callback
            )
            self.update_result(result, "Restored")
            self.log_progress("骨架重建完成")
            
        except Exception as e:
            self.log_progress(f"错误: {str(e)}")
            QMessageBox.critical(self, "错误", str(e))
            
    def load_image(self):
        """重写加载图片方法，重置状态"""
        if super().load_image():
            self.binary_image = None
            self.skeleton_image = None
            self.distance_map = None
            self.progress_text.clear()
            self.log_progress("图像加载完成，请进行二值化处理")