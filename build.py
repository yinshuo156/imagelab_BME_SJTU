import sys
import os
from PyInstaller.__main__ import run

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义图标路径
icon_path = os.path.join(current_dir, 'resources', 'icon.png')

# 定义PyInstaller打包参数
options = [
    'main.py',  # 主程序文件
    '--name=ImageLab',  # 生成的exe名称
    '--windowed',  # 使用GUI模式
    '--onefile',  # 打包成单个文件
    f'--icon={icon_path}',  # 设置图标
    '--clean',  # 清理临时文件
    '--add-data=resources;resources',  # 添加资源文件
    # 添加所需的包
    '--hidden-import=numpy',
    '--hidden-import=PyQt5',
    '--hidden-import=matplotlib',
]

# 运行打包命令
run(options) 