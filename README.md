# ImageLab - [计算机辅助手术与治疗技术](https://oc.sjtu.edu.cn/courses/70195)Project

*12/22/2025 sjtu.ys@sjtu.edu.cn*

## 1. 内容说明

- 文件夹code内是python程序原代码

  ```
  ImageLab/
  ├── main.py                  # 主程序入口
  ├── resources/               # 资源文件夹
  │   └── icon.png            # 程序图标
  ├── ui/                      # 用户界面模块
  │   ├── main_window.py      # 主窗口
  │   └── projects/           # 各个项目的UI实现
  │       ├── base_project.py     # 基础项目类
  │       ├── histogram_project.py # 直方图项目
  │       ├── convolution_project.py # 卷积项目
  │       ├── morphology_project.py # 形态学项目
  │       └── skeleton_project.py  # 骨架提取项目
  └── utils/                   # 工具模块
      └── image_processor.py   # 图像处理算法实现
  ```
- 文件夹exe内是打包好的可运行程序——ImageLab.exe ，双击即可运行

## 2. 设计过程

- 界面实现

  - 使用PyQt5构建图形界面
  - 采用Qt Designer设计UI,生成ui文件
  - 通过PyQt5的uic模块将ui文件转换为Python代码
- 图像处理功能

  - 基于OpenCV-Python实现图像读取与处理
  - 使用NumPy进行像素级操作
  - 支持图像滤波、边缘检测、颜色调整等基础功能
  - 实现图像格式转换与保存
- 文件处理

  - 支持拖拽导入图片
  - 实现批量处理功能
  - 自动保存处理结果
- 程序打包

  - 使用PyInstaller将Python程序打包为exe
  - 采用--noconsole模式运行
  - 包含所有依赖库与资源文件

## 3. 技术实现与优化

### 3.1 直方图分析与阈值分割

- 实现了基于 NumPy 的高效直方图计算
- 集成多种阈值分割方法:
  - 手动阈值：支持实时预览的滑块控制
  - Otsu 阈值：基于类间方差最大化原理
  - 熵阈值：利用图像信息熵进行自适应分割
- 优化了阈值计算过程，使用向量化操作提升性能

### 3.2 卷积与滤波处理

- 实现了通用的 2D 卷积操作框架
- 边缘检测算子:
  - Roberts: 适用于检测局部细节
  - Prewitt: 提供方向性边缘检测
  - Sobel: 增强了边缘检测的抗噪性能
- 图像平滑:
  - 高斯滤波：实现可调节核大小和 sigma 的平滑效果
  - 中值滤波：有效去除椒盐噪声
- 采用 padding 处理边界问题，保持输出图像尺寸

### 3.3 形态学操作

- 基础操作:
  - 膨胀：扩展前景区域
  - 腐蚀：收缩前景区域
- 复合操作:
  - 开运算：消除小的突出部分
  - 闭运算：填充小的孔洞
- 支持自定义结构元素大小，提高操作灵活性

### 3.4 骨架提取与重建

- 距离变换:
  - 采用两遍扫描法计算欧氏距离
  - 实现了前向和后向扫描的优化
- 骨架提取:
  - 基于 Zhang-Suen 细化算法
  - 通过迭代的击中击不中变换保持拓扑结构
- 骨架重建:
  - 结合距离信息的自适应重建
  - 使用双层高斯权重实现平滑过渡
  - 通过参数调优优化重建效果

### 3.5 Qt 界面

- 采用模块化设计，将 UI 和处理逻辑分离
- 实现了实时预览功能
- 添加了处理进度反馈
- 优化了图像显示的缩放和对齐
- 统一了界面风格，提升用户体验
