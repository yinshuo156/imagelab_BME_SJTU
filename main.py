import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ui.main_window import MainWindow

def exception_hook(exctype, value, traceback):
    """捕获未处理的异常"""
    print('Exception:', exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)

if __name__ == '__main__':
    # 设置异常钩子
    sys._excepthook = sys.excepthook
    sys.excepthook = exception_hook
    
    # 创建应用
    app = QApplication(sys.argv)
    
    # 设置应用信息
    app.setApplicationName("ImageLab")
    app.setApplicationDisplayName("ImageLab - Digital Image Processing")
    app.setOrganizationName("Shuo Yin")
    app.setOrganizationDomain("yinelon@gmail.com")
    
    # 设置应用图标
    app_icon = QIcon("resources/icon.png")
    app.setWindowIcon(app_icon)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_()) 