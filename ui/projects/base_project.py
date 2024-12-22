from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np

class BaseProject(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.processed_image = None
        self.is_processing = False
        self.setup_ui()

    def setup_ui(self):
        """设置基础UI布局"""
        layout = QHBoxLayout(self)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 图片操作按钮组
        btn_group = QGroupBox("图像操作")
        btn_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("打开图片")
        self.save_btn = QPushButton("保存结果")
        
        self.load_btn.clicked.connect(self.load_image)
        self.save_btn.clicked.connect(self.save_image)
        
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.save_btn)
        btn_group.setLayout(btn_layout)
        
        control_layout.addWidget(btn_group)
        control_layout.addStretch()
        
        # 右��图片显示区域
        display_panel = QWidget()
        display_layout = QHBoxLayout(display_panel)
        
        self.original_label = QLabel()
        self.processed_label = QLabel()
        self.original_label.setMinimumSize(400, 400)
        self.processed_label.setMinimumSize(400, 400)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 1px solid #dee2e6; background: white;")
        self.processed_label.setStyleSheet("border: 1px solid #dee2e6; background: white;")
        
        display_layout.addWidget(self.original_label)
        display_layout.addWidget(self.processed_label)
        
        # 添加到主布局
        layout.addWidget(control_panel, 1)
        layout.addWidget(display_panel, 4)

    def start_processing(self):
        """开始处理"""
        if self.is_processing:
            return False
        self.is_processing = True
        self.processed_label.setText("Processing...")
        QApplication.processEvents()
        return True

    def finish_processing(self):
        """结束处理"""
        self.is_processing = False
        QApplication.processEvents()

    def process_with_loading(self, func, *args, **kwargs):
        """通用的处理���数包装器"""
        if not self.start_processing():
            return
        try:
            self.clear_result()
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            return None
        finally:
            self.finish_processing()

    def clear_result(self):
        """清除处理结果"""
        self.processed_image = None
        self.processed_label.clear()
        self.processed_label.setText("Waiting for processing...")

    def load_image(self):
        """加载图片"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "图片文件 (*.png *.jpg *.bmp)")
            
            if not file_path:
                return False
                
            # 使用QImageReader进行优化的图像加载
            reader = QImageReader(file_path)
            reader.setAutoTransform(True)
            
            if reader.canRead():
                self.image = reader.read()
                if self.image.isNull():
                    raise Exception("无法读取图像数据")
                    
                # 转换为灰度图
                if self.image.format() != QImage.Format_Grayscale8:
                    self.image = self.image.convertToFormat(QImage.Format_Grayscale8)
                
                # 优化图像显示
                scaled_pixmap = QPixmap.fromImage(self.image).scaled(
                    400, 400, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                self.original_label.setPixmap(scaled_pixmap)
                self.original_label.setAlignment(Qt.AlignCenter)
                
                # 清除旧结果
                self.processed_label.clear()
                self.processed_label.setText("等待处理...")
                self.processed_label.setAlignment(Qt.AlignCenter)
                
                return True
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图片失败: {str(e)}")
            return False

    def save_image(self):
        """保存处理后的图片"""
        if self.processed_image:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图片", "", "PNG图片 (*.png);;JPEG图片 (*.jpg)")
            if file_path:
                self.processed_image.save(file_path)

    def qimage_to_numpy(self, qimage):
        """将QImage转换为numpy数组"""
        try:
            # 确保图像格式正确
            if qimage.format() != QImage.Format_Grayscale8:
                qimage = qimage.convertToFormat(QImage.Format_Grayscale8)
            
            width = qimage.width()
            height = qimage.height()
            
            # 直接获取字节数据
            ptr = qimage.constBits()
            ptr.setsize(height * width)
            
            # 避免复制数据
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width))
            return arr.copy()  # 返回副本以避免内存问题
            
        except Exception as e:
            QMessageBox.critical(None, "错误", f"图像转换失败: {str(e)}")
            return None

    def numpy_to_qimage(self, arr):
        """将numpy数组转换为QImage"""
        height, width = arr.shape
        bytes_per_line = width
        return QImage(arr.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

    def update_result(self, result, operation_name, params=None):
        """更新处理结果"""
        if result is None:
            return
            
        try:
            self.processed_image = self.numpy_to_qimage(result)
            pixmap = QPixmap.fromImage(self.processed_image)
            self.processed_label.setPixmap(
                pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新结果失败: {str(e)}")