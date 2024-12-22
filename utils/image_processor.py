import numpy as np
from typing import Union, Tuple, List

class ImageProcessor:
    @staticmethod
    def calculate_histogram(image):
        """优化直方图计算"""
        if len(image.shape) == 3:
            image = image.mean(axis=2).astype(np.uint8)
        # 使用numpy的histogram函数替代循环
        histogram, _ = np.histogram(image, bins=256, range=(0, 256))
        return histogram
    
    @staticmethod
    def otsu_threshold(image):
        """优化Otsu阈值计算"""
        if len(image.shape) == 3:
            image = image.mean(axis=2).astype(np.uint8)
            
        histogram = ImageProcessor.calculate_histogram(image)
        total = histogram.sum()
        
        # 使用累积和加速计算
        weight_1 = np.cumsum(histogram)
        weight_2 = total - weight_1
        
        # 使用向量化操作替代循环
        sum_1 = np.cumsum(histogram * np.arange(256))
        
        # 避免除零警告
        mean_1 = np.zeros_like(weight_1, dtype=float)
        mean_2 = np.zeros_like(weight_2, dtype=float)
        
        # 只在权重不为0的位置计算均值
        valid_idx1 = weight_1 > 0
        valid_idx2 = weight_2 > 0
        
        mean_1[valid_idx1] = sum_1[valid_idx1] / weight_1[valid_idx1]
        mean_2[valid_idx2] = (sum_1[-1] - sum_1[valid_idx2]) / weight_2[valid_idx2]
        
        # 计算类间方差
        variance = weight_1 * weight_2 * (mean_1 - mean_2) ** 2
        
        # 处理无效值
        variance = np.nan_to_num(variance)
        
        return np.argmax(variance)
    
    @staticmethod
    def manual_threshold(image, threshold):
        """优化手动阈值处理"""
        return np.where(image > threshold, 255, 0).astype(np.uint8)
    
    @staticmethod
    def entropy_threshold(image: np.ndarray) -> int:
        """基于熵的阈值分割"""
        histogram = ImageProcessor.calculate_histogram(image)
        histogram = histogram / np.sum(histogram)  # 归一化
        
        max_entropy = 0
        best_threshold = 0
        
        for threshold in range(1, 255):
            # 计算前景和背景的概率
            p1 = histogram[:threshold].sum()
            p2 = histogram[threshold:].sum()
            
            if p1 == 0 or p2 == 0:
                continue
                
            # 计算前景和背景的熵
            h1 = -np.sum(histogram[:threshold] * np.log2(histogram[:threshold] + 1e-10)) / p1
            h2 = -np.sum(histogram[threshold:] * np.log2(histogram[threshold:] + 1e-10)) / p2
            
            entropy = h1 + h2
            if entropy > max_entropy:
                max_entropy = entropy
                best_threshold = threshold
                
        return best_threshold

    # Project 2: 卷积和滤波相关方法
    @staticmethod
    def convolution2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2D卷积操作"""
        if len(image.shape) == 3:
            image = image.mean(axis=2).astype(np.uint8)
            
        h, w = image.shape
        k_h, k_w = kernel.shape
        pad_h = k_h // 2
        pad_w = k_w // 2
        
        # 填充图像边界
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        output = np.zeros_like(image, dtype=float)
        
        # 执行卷积
        for i in range(h):
            for j in range(w):
                output[i, j] = np.sum(
                    padded[i:i+k_h, j:j+k_w] * kernel
                )
                
        return np.clip(output, 0, 255).astype(np.uint8)

    # 边缘检测算子
    ROBERTS_X = np.array([[1, 0], [0, -1]])
    ROBERTS_Y = np.array([[0, 1], [-1, 0]])
    
    PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    @staticmethod
    def edge_detection(image: np.ndarray, operator: str = 'sobel') -> np.ndarray:
        """边缘检测
        operator: 'roberts', 'prewitt', 或 'sobel'
        """
        if operator == 'roberts':
            gx = ImageProcessor.convolution2d(image, ImageProcessor.ROBERTS_X)
            gy = ImageProcessor.convolution2d(image, ImageProcessor.ROBERTS_Y)
        elif operator == 'prewitt':
            gx = ImageProcessor.convolution2d(image, ImageProcessor.PREWITT_X)
            gy = ImageProcessor.convolution2d(image, ImageProcessor.PREWITT_Y)
        else:  # sobel
            gx = ImageProcessor.convolution2d(image, ImageProcessor.SOBEL_X)
            gy = ImageProcessor.convolution2d(image, ImageProcessor.SOBEL_Y)
            
        magnitude = np.sqrt(gx.astype(float)**2 + gy.astype(float)**2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    @staticmethod
    def gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
        """高斯滤波"""
        # 生成高斯核
        k = kernel_size // 2
        x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
        kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))
        kernel = kernel / kernel.sum()
        
        return ImageProcessor.convolution2d(image, kernel)

    @staticmethod
    def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """中值滤波"""
        if len(image.shape) == 3:
            image = image.mean(axis=2).astype(np.uint8)
            
        h, w = image.shape
        k = kernel_size // 2
        output = np.zeros_like(image)
        
        padded = np.pad(image, ((k, k), (k, k)), mode='reflect')
        
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.median(window)
                
        return output.astype(np.uint8)

    # Project 3: 形态学操作
    @staticmethod
    def binary_dilation(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
        """二值膨胀操作"""
        if kernel is None:
            kernel = np.ones((3, 3), dtype=np.uint8)
            
        h, w = image.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h//2, k_w//2
        
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        output = np.zeros_like(image)
        
        for i in range(h):
            for j in range(w):
                if np.any(padded[i:i+k_h, j:j+k_w] * kernel):
                    output[i, j] = 255
                    
        return output

    @staticmethod
    def binary_erosion(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
        """二值腐蚀操作"""
        if kernel is None:
            kernel = np.ones((3, 3), dtype=np.uint8)
            
        h, w = image.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h//2, k_w//2
        
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        output = np.zeros_like(image)
        
        for i in range(h):
            for j in range(w):
                if np.all((padded[i:i+k_h, j:j+k_w] * kernel) == (kernel * 255)):
                    output[i, j] = 255
                    
        return output

    @staticmethod
    def morphological_opening(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
        """开运算：先腐蚀后膨胀"""
        eroded = ImageProcessor.binary_erosion(image, kernel)
        return ImageProcessor.binary_dilation(eroded, kernel)

    @staticmethod
    def morphological_closing(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
        """闭运算：先膨胀后腐蚀"""
        dilated = ImageProcessor.binary_dilation(image, kernel)
        return ImageProcessor.binary_erosion(dilated, kernel)

    # Project 4: 骨架提取和距离变换
    @staticmethod
    def distance_transform(image: np.ndarray) -> np.ndarray:
        """优化的距离变换"""
        # 确保输入是二值图像
        binary = image > 127
        h, w = binary.shape
        
        # 初始化距离图
        inf = h + w
        dist = np.full(binary.shape, inf, dtype=float)
        dist[binary == 0] = 0
        
        # 前向扫描
        for i in range(1, h):
            for j in range(1, w):
                if binary[i, j]:
                    dist[i, j] = min(
                        dist[i-1, j-1] + 1.414,  # 对角线距离
                        dist[i-1, j] + 1,        # 上方距离
                        dist[i, j-1] + 1         # 左侧距离
                    )
        
        # 后向扫描
        for i in range(h-2, -1, -1):
            for j in range(w-2, -1, -1):
                if binary[i, j]:
                    dist[i, j] = min(
                        dist[i, j],
                        dist[i+1, j+1] + 1.414,  # 对角线距离
                        dist[i+1, j] + 1,        # 下方距离
                        dist[i, j+1] + 1         # 右侧距离
                    )
        
        # 归一化到0-255
        if dist.max() > 0:
            dist = dist * (255.0 / dist.max())
            
        return dist.astype(np.uint8)

    @staticmethod
    def binary_hit_or_miss(image, kernel1, kernel2):
        """击中击不中变换"""
        h, w = image.shape
        k_h, k_w = kernel1.shape
        pad_h, pad_w = k_h//2, k_w//2
        
        padded = np.pad(image > 127, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        output = np.zeros_like(image)
        
        for i in range(h):
            for j in range(w):
                window = padded[i:i+k_h, j:j+k_w]
                if (np.all(window[kernel1 == 1] == 1) and 
                    np.all(window[kernel2 == 1] == 0)):
                    output[i, j] = 255
        return output

    @staticmethod
    def fast_skeleton(image, progress_callback=None):
        """改进的骨架提取算法 - 使用Zhang-Suen细化算法"""
        # 确保输入是二值图像
        binary = (image > 127)  # 白色为前景(1)，黑色为背景(0)
        
        def neighbors_8(x, y, image):
            """获取8邻域像素"""
            return [
                image[x-1, y],   # P2 (North)
                image[x-1, y+1], # P3
                image[x, y+1],   # P4 (East)
                image[x+1, y+1], # P5
                image[x+1, y],   # P6 (South)
                image[x+1, y-1], # P7
                image[x, y-1],   # P8 (West)
                image[x-1, y-1]  # P9
            ]

        def transitions(neighbors):
            """计算0到1的跳变次数"""
            n = neighbors + [neighbors[0]]
            return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

        def first_subiteration(image):
            rows, cols = image.shape
            candidates = []
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if not image[i, j]:  # 跳过背景点
                        continue
                    
                    P = neighbors_8(i, j, image)
                    if (2 <= sum(P) <= 6 and      # 条件1: 保持连通性
                        transitions(P) == 1 and    # 条件2: 端点检测
                        P[0] * P[2] * P[4] == 0 and  # 条件3: 防止过度删除
                        P[2] * P[4] * P[6] == 0):    # 条件4: 保持形状特征
                        candidates.append((i, j))
            
            return candidates

        def second_subiteration(image):
            rows, cols = image.shape
            candidates = []
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if not image[i, j]:
                        continue
                    
                    P = neighbors_8(i, j, image)
                    if (2 <= sum(P) <= 6 and
                        transitions(P) == 1 and
                        P[0] * P[2] * P[6] == 0 and  # 修改条件以保持对称性
                        P[0] * P[4] * P[6] == 0):
                        candidates.append((i, j))
            
            return candidates

        # 主循环
        skeleton = binary.copy()
        iteration = 0
        changed = True
        
        while changed and iteration < 100:
            if progress_callback:
                progress_callback(iteration)
            
            changed = False
            
            # 第一次子迭代
            candidates = first_subiteration(skeleton)
            if candidates:
                changed = True
                for x, y in candidates:
                    skeleton[x, y] = False
            
            # 第二次子迭代
            candidates = second_subiteration(skeleton)
            if candidates:
                changed = True
                for x, y in candidates:
                    skeleton[x, y] = False
            
            iteration += 1
        
        # 返回结果，保持与输入图像相同的方向
        return skeleton.astype(np.uint8) * 255

    @staticmethod
    def skeleton_restoration(skeleton, dist_transform, progress_callback=None):
        """改进的骨架重建算法 - 修复尺寸匹配问题"""
        h, w = skeleton.shape
        restored = np.zeros_like(skeleton, dtype=float)
        
        # 获取骨架点
        points = np.argwhere(skeleton > 0)
        total = len(points)
        
        if total == 0:
            return restored.astype(np.uint8)
        
        # 计算距离变换的归一化值
        max_dist = np.max(dist_transform)
        if max_dist == 0:
            max_dist = 1
        dist_norm = dist_transform.astype(float) / max_dist
        
        # 创建形状保持重建
        for idx, (y, x) in enumerate(points):
            if progress_callback and idx % 100 == 0:
                progress_callback(idx, total)
            
            # 使用0.28作为基础重建半径系数
            local_radius = int(dist_transform[y, x] * 0.28)
            # 增加外部平滑区域
            smooth_radius = int(local_radius * 1.2)  # 额外20%的平滑区域
            radius = max(2, smooth_radius)
            
            # 定义局部区域
            y_min = max(0, y - radius)
            y_max = min(h, y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(w, x + radius + 1)
            
            # 使用改进的椭圆形掩码
            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
            
            # 计算到中心点的距离
            dist_sq = ((yy - y)**2 + (xx - x)**2) / (local_radius**2)
            
            # 创建双层高斯权重
            # 内层：主要形状
            inner_weight = np.exp(-dist_sq * 2.8)
            
            # 外层：平滑过渡
            outer_weight = np.exp(-dist_sq * 1.5)
            
            # 组合权重
            weight = np.where(dist_sq <= 1,
                            inner_weight,
                            outer_weight * np.exp(-(dist_sq - 1) * 2))
            
            # 应用距离信息的权重
            dist_weight = dist_norm[y, x]
            weight = weight * (0.85 + 0.15 * dist_weight)
            
            # 更新重建结果
            region = restored[y_min:y_max, x_min:x_max]
            region[:] = np.maximum(region, weight * 255)
        
        # 创建高斯核
        kernel_size = 3
        sigma = 0.8
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - center
                y = j - center
                kernel[i, j] = np.exp(-(x*x + y*y)/(2*sigma*sigma))
        kernel = kernel / kernel.sum()
        
        # 应用高斯平滑
        smoothed = np.zeros_like(restored)
        padded = np.pad(restored, ((1, 1), (1, 1)), mode='reflect')
        
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                smoothed[i, j] = np.sum(window * kernel)
        
        # 使用平滑的阈值处理
        threshold = np.max(smoothed) * 0.42
        restored = (smoothed > threshold).astype(np.uint8) * 255
        
        return restored