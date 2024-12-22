from .base_project import BaseProject
from utils.image_processor import ImageProcessor
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class HistogramProject(BaseProject):
    def __init__(self):
        super().__init__()
        self.setup_specific_ui()
        self._update_timer = QTimer()  # 添加定时器
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.delayed_update)
        
    def setup_specific_ui(self):
        """设置直方图项目特定的UI元素"""
        control_layout = self.findChild(QVBoxLayout)
        
        # 添加直方图显示
        self.figure, self.ax = plt.subplots(figsize=(4, 3))
        self.canvas = FigureCanvas(self.figure)
        control_layout.insertWidget(2, self.canvas)
        
        # 阈值操作组
        threshold_group = QGroupBox("Threshold")
        threshold_layout = QVBoxLayout(threshold_group)
        
        # 阈值选择方式
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Manual", "Otsu", "Entropy"])
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        threshold_layout.addWidget(self.method_combo)
        
        # 手动阈值控件组
        self.manual_widget = QWidget()
        manual_layout = QHBoxLayout(self.manual_widget)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(127)
        self.threshold_slider.setMinimumWidth(200)  # 设置最小宽度
        
        self.threshold_value = QSpinBox()
        self.threshold_value.setRange(0, 255)
        self.threshold_value.setValue(127)
        
        self.threshold_slider.valueChanged.connect(self.threshold_value.setValue)
        self.threshold_value.valueChanged.connect(self.threshold_slider.setValue)
        self.threshold_slider.valueChanged.connect(self.apply_threshold)
        
        manual_layout.addWidget(QLabel("Value:"))
        manual_layout.addWidget(self.threshold_slider)
        manual_layout.addWidget(self.threshold_value)
        
        threshold_layout.addWidget(self.manual_widget)
        
        control_layout.insertWidget(3, threshold_group)
        
    def on_method_changed(self, method):
        """当阈值方法改变时"""
        self.manual_widget.setVisible(method == "Manual")
        if method != "Manual":
            self.apply_current_method()
    
    def apply_current_method(self):
        if not self.image or not self.start_processing():
            return

        try:
            method = self.method_combo.currentText()
            arr = self.qimage_to_numpy(self.image)

            if method == "Manual":
                threshold = self.threshold_value.value()
                result = ImageProcessor.manual_threshold(arr, threshold)
                operation_name = "Manual Threshold"
                params = {"threshold": threshold}
            elif method == "Otsu":
                threshold = ImageProcessor.otsu_threshold(arr)
                result = ImageProcessor.manual_threshold(arr, threshold)
                operation_name = "Otsu Threshold"
                params = {"threshold": threshold}
            else:  # Entropy
                threshold = ImageProcessor.entropy_threshold(arr)
                result = ImageProcessor.manual_threshold(arr, threshold)
                operation_name = "Entropy Threshold"
                params = {"threshold": threshold}

            self.update_result(result, operation_name, params)

            if method != "Manual":
                self.threshold_value.setValue(threshold)
                self.threshold_value.blockSignals(True)
                self.threshold_slider.setValue(threshold)
                self.threshold_value.blockSignals(False)

        finally:
            self.finish_processing()
    
    def apply_threshold(self):
        """优化实时更新逻辑"""
        if self.method_combo.currentText() == "Manual":
            # 使用定时器延迟更新，避免频繁计算
            self._update_timer.start(100)
            
    def delayed_update(self):
        """延迟更新处理结果"""
        self.apply_current_method()

    def load_image(self):
        """重写加载图片方法，添加直方图显示和自动处理"""
        if super().load_image():
            self.update_histogram()
            # 自动执行Otsu阈值分割
            self.method_combo.setCurrentText("Otsu")
            self.apply_current_method()
            
    def update_histogram(self):
        """优化直方图显示"""
        if not self.image:
            return
            
        try:
            arr = self.qimage_to_numpy(self.image)
            if arr is None:
                return
                
            histogram = ImageProcessor.calculate_histogram(arr)
            
            self.ax.clear()
            self.ax.bar(range(256), histogram, width=1)
            self.ax.set_title("Histogram")
            
            # 减少重绘频率
            self.canvas.draw_idle()
            
        except Exception as e:
            QMessageBox.warning(self, "警告", f"更新直方图失败: {str(e)}") 