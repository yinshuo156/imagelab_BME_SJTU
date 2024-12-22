from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor

class LoadingSpinner(QWidget):
    def __init__(self, parent=None, centerOnParent=True):
        super().__init__(parent)
        self.centerOnParent = centerOnParent
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 设置动画参数
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.setInterval(50)  # 20fps
        
        # 设置大小
        self.setFixedSize(100, 100)
        
        # 初始隐藏
        self.hide()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制半透明背景
        painter.fillRect(self.rect(), QColor(255, 255, 255, 200))
        
        # 绘制加载动画
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.angle)
        
        painter.setPen(Qt.NoPen)
        for i in range(8):
            painter.rotate(45)
            alpha = int(255 * ((i + 1) / 8))
            painter.setBrush(QColor(70, 136, 250, alpha))
            painter.drawRoundedRect(-4, -20, 8, 8, 4, 4)
            
    def rotate(self):
        self.angle = (self.angle + 45) % 360
        self.update()
        
    def showEvent(self, event):
        if self.centerOnParent:
            self.move(
                self.parentWidget().width() / 2 - self.width() / 2,
                self.parentWidget().height() / 2 - self.height() / 2
            )
        self.timer.start()
        
    def hideEvent(self, event):
        self.timer.stop() 