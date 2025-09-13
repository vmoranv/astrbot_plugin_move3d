try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError as e:
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(e)

# rembg相关变量初始化
REMBG_AVAILABLE = False
REMBG_IMPORT_ERROR = None
rembg_remove = None
PILImage = None

# 尝试导入rembg库
try:
    from rembg import remove
    from PIL import Image
    REMBG_AVAILABLE = True
except ImportError as e:
    REMBG_AVAILABLE = False
    REMBG_IMPORT_ERROR = str(e)

from typing import Any, Optional
import os

# 全局变量存储导入的模块
rembg_remove: Optional[Any] = remove if 'remove' in locals() else None
PILImage: Optional[Any] = Image if 'Image' in locals() else None

def _check_rembg_import() -> bool:
    """检查并尝试导入rembg库"""
    global REMBG_AVAILABLE, REMBG_IMPORT_ERROR, rembg_remove, PILImage
    if REMBG_AVAILABLE and rembg_remove is not None and PILImage is not None:
        return True
    
    try:
        from rembg import remove
        from PIL import Image
        # 将导入的模块保存到全局变量中
        rembg_remove = remove
        PILImage = Image
        REMBG_AVAILABLE = True
        REMBG_IMPORT_ERROR = None
        return True
    except ImportError as e:
        REMBG_AVAILABLE = False
        REMBG_IMPORT_ERROR = str(e)
        return False

def check_dependencies():
    """检查依赖是否可用"""
    return CV2_AVAILABLE

def get_import_error():
    """获取导入错误信息"""
    return CV2_IMPORT_ERROR if not CV2_AVAILABLE else None

def check_rembg_dependencies() -> bool:
    """检查rembg依赖是否可用"""
    # 每次都尝试导入，以应对运行时安装的情况
    _check_rembg_import()
    return REMBG_AVAILABLE

def get_rembg_import_error() -> Optional[str]:
    """获取rembg导入错误信息"""
    return REMBG_IMPORT_ERROR if not REMBG_AVAILABLE else None

def remove_background(image_path: str, output_path: str) -> None:
    """
    移除图片背景并保存结果
    :param image_path: 输入图片路径
    :param output_path: 输出图片路径
    """
    # 确保rembg已导入
    if not check_rembg_dependencies():
        raise ImportError(f"rembg库未正确导入: {get_rembg_import_error()}")
    
    # 类型检查确保PILImage不为None
    if PILImage is None or rembg_remove is None:
        raise ImportError("PIL或rembg模块未正确导入")
        
    try:
        # 打开图片并转换为RGBA模式
        input_image: Any = PILImage.open(image_path).convert('RGBA')
        
        # 使用rembg移除背景
        output_image: Any = rembg_remove(input_image)
        
        # 保存处理后的图片
        # 明确调用PIL Image的save方法
        output_image.save(output_path, format='PNG')
    except Exception as e:
        raise RuntimeError(f"处理图片时发生错误: {str(e)}")

def sketch_effect(image_path, output_path, blur_kernel_size):
    """
    将指定的图片转换为素描/简笔画风格并保存，带有透明背景和加深的线条。

    参数:
    image_path (str): 输入图片的路径。
    output_path (str): 输出简笔画图片的保存路径。
    blur_kernel_size (int): 高斯模糊的核大小，必须为奇数。
    """
    if not CV2_AVAILABLE:
        raise ImportError(f"缺少必要的依赖库: {CV2_IMPORT_ERROR}")
        
    # 确保模糊核大小为奇数
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
        
    # 1. 读取原始图像
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        raise FileNotFoundError(f"无法读取图片，请检查路径是否正确: {image_path}")

    # 2. 将图片转换为灰度图
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 3. 对灰度图进行反色处理
    inverted_gray_image = cv2.bitwise_not(gray_image)

    # 4. 对反色图像进行高斯模糊
    blurred_image = cv2.GaussianBlur(inverted_gray_image, (blur_kernel_size, blur_kernel_size), 0)

    # 5. 将原始灰度图与模糊后的反色图进行颜色减淡混合
    inverted_blurred_image = cv2.bitwise_not(blurred_image)
    sketch_image = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)
    
    # 6. 调整对比度和亮度使线条更清晰，但不过度
    # 创建一个对比度增强的查找表
    alpha = 1.2  # 降低对比度增强程度以保持线条连贯性
    beta = -20   # 减少亮度降低以保持细节
    sketch_image = cv2.convertScaleAbs(sketch_image, alpha=alpha, beta=beta)
    
    # 7. 创建透明背景的RGBA图像
    # 线条应该是白色，背景是透明的
    # 先创建一个白色的图像作为基础
    white_image = np.full_like(sketch_image, 255, dtype=np.uint8)
    
    # 创建alpha通道 - 线条区域为不透明(255)，背景区域为透明(0)
    # 使用较低的阈值以保留更多细节和线条连贯性
    _, alpha_channel = cv2.threshold(sketch_image, 230, 255, cv2.THRESH_BINARY_INV)
    
    # 将灰度图像转换为3通道白色图像
    sketch_bgr = cv2.cvtColor(white_image, cv2.COLOR_GRAY2BGR)
    
    # 合并BGR和Alpha通道
    result_image = cv2.merge([sketch_bgr, alpha_channel])
    
    # 8. 保存最终的简笔画图片（PNG格式支持透明度）
    cv2.imwrite(output_path, result_image)
    
    return output_path

def line_art_effect(image_path, output_path, threshold1, threshold2):
    """
    使用Canny边缘检测将图片转换为线稿风格，带有透明背景和加深的线条。

    参数:
    image_path (str): 输入图片的路径。
    output_path (str): 输出线稿图片的保存路径。
    threshold1 (int): Canny边缘检测的第一个阈值。
    threshold2 (int): Canny边缘检测的第二个阈值。
    """
    if not CV2_AVAILABLE:
        raise ImportError(f"缺少必要的依赖库: {CV2_IMPORT_ERROR}")
        
    # 读取图片
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if original_image is None:
        raise FileNotFoundError(f"无法读取图片，请检查路径是否正确: {image_path}")
    
    # 如果图片有alpha通道，分离出来
    if original_image.shape[2] == 4:
        bgr_image = original_image[:, :, :3]
        alpha_channel = original_image[:, :, 3]
    else:
        bgr_image = original_image
        alpha_channel = None
        
    # 转为灰度图
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊降噪，但不过度模糊以保持线条清晰
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    
    # 使用Canny边缘检测，调整阈值以获得更好的线条连贯性
    edges = cv2.Canny(blurred_image, threshold1, threshold2)
    
    # 减少腐蚀操作或降低强度以保持线条连贯性
    kernel = np.ones((1,1), np.uint8)  # 使用更小的内核
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # 增加闭运算来连接断开的线条，使区域封闭
    closing_kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, closing_kernel)
    
    # Canny输出的是黑底白线，我们要保持黑线白底，但添加透明背景
    # 创建alpha通道 - 白色线条区域为不透明(255)，黑色背景区域为透明(0)
    alpha_channel = edges  # 线条区域为白色（255），背景为黑色（0）
    
    # 创建三通道黑色图像（反转 edges 图像以获得黑线）
    black_lines = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    
    # 合并BGR和Alpha通道
    result_image = cv2.merge([black_lines, alpha_channel])
    
    # 保存为PNG格式（支持透明度）
    cv2.imwrite(output_path, result_image)
    
    return output_path

def sketch_3d_effect(image_path, output_path, threshold1=80, threshold2=160):
    """
    将指定的图片转换为3D效果简笔画并放在深蓝色背景上。
    包含黑色线条简笔画和亮红色放大简笔画，都具有透明背景。

    参数:
    image_path (str): 输入图片的路径。
    output_path (str): 输出3D效果图片的保存路径。
    threshold1 (int): Canny边缘检测的第一个阈值，默认为80。
    threshold2 (int): Canny边缘检测的第二个阈值，默认为160。
    """
    if not CV2_AVAILABLE:
        raise ImportError(f"缺少必要的依赖库: {CV2_IMPORT_ERROR}")
    
    # 1. 读取原始图像
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        raise FileNotFoundError(f"无法读取图片，请检查路径是否正确: {image_path}")
    
    # 获取原始图像尺寸
    height, width = original_image.shape[:2]
    
    # 2. 创建透明背景 (尺寸更大以容纳伸出的线条)
    background_size = (int(width * 1.8), int(height * 1.8))  # 增大背景以容纳伸出的线条
    background = np.zeros((background_size[1], background_size[0], 4), dtype=np.uint8)  # 透明背景 (BGRA)
    
    # 3. 创建蓝色背景方块 (#1A01EA -> BGR(234, 1, 26))
    blue_background_size = (int(width * 1.1), int(height * 1.1))  # 减小蓝色背景
    blue_background = np.full((blue_background_size[1], blue_background_size[0], 3), (234, 1, 26), dtype=np.uint8)
    
    # 4. 将蓝色背景方块放置在透明背景中央
    bg_h, bg_w = background.shape[:2]
    blue_h, blue_w = blue_background.shape[:2]
    
    # 计算居中位置
    y_offset = (bg_h - blue_h) // 2
    x_offset = (bg_w - blue_w) // 2
    
    # 确保不会越界
    y1, y2 = max(0, y_offset), min(bg_h, y_offset + blue_h)
    x1, x2 = max(0, x_offset), min(bg_w, x_offset + blue_w)
    
    # 计算实际要复制的区域大小
    copy_h = y2 - y1
    copy_w = x2 - x1
    
    # 计算蓝色背景中对应的区域
    blue_y1 = max(0, -y_offset)
    blue_y2 = blue_y1 + copy_h
    blue_x1 = max(0, -x_offset)
    blue_x2 = blue_x1 + copy_w
    
    # 只在有效区域内复制蓝色背景
    if copy_h > 0 and copy_w > 0:
        background[y1:y2, x1:x2, :3] = blue_background[blue_y1:blue_y2, blue_x1:blue_x2]
        background[y1:y2, x1:x2, 3] = 255  # 设置alpha通道为不透明
    
    # 5. 创建临时文件用于存储黑色简笔画（带透明背景）
    temp_black_sketch = image_path.replace('.jpg', '_temp_black.png')
    temp_red_sketch = image_path.replace('.jpg', '_temp_red.png')
    
    try:
        # 6. 生成黑色简笔画（使用 Canny 参数）
        line_art_effect(image_path, temp_black_sketch, threshold1, threshold2)
        
        # 7. 读取黑色简笔画（带透明背景）
        black_sketch_rgba = cv2.imread(temp_black_sketch, cv2.IMREAD_UNCHANGED)
        if black_sketch_rgba is None:
            raise FileNotFoundError(f"无法读取临时黑色简笔画: {temp_black_sketch}")
        
        # 提取黑色简笔画的BGR和Alpha通道
        black_sketch_bgr = black_sketch_rgba[:, :, :3].copy()
        black_sketch_alpha = black_sketch_rgba[:, :, 3]
        
        # 8. 生成亮红色简笔画（使用相同的 Canny 参数）
        line_art_effect(image_path, temp_red_sketch, threshold1, threshold2)
        
        # 9. 读取亮红色简笔画并转换为指定红色（#D72E7A -> BGR(122, 46, 215)）
        red_sketch_rgba = cv2.imread(temp_red_sketch, cv2.IMREAD_UNCHANGED)
        if red_sketch_rgba is None:
            raise FileNotFoundError(f"无法读取临时红色简笔画: {temp_red_sketch}")
        
        # 创建指定红色版本
        red_sketch_bgr = red_sketch_rgba[:, :, :3].copy()
        red_sketch_alpha = red_sketch_rgba[:, :, 3]
        
        # 将白色区域变为指定红色
        red_mask = red_sketch_alpha > 0
        red_sketch_bgr[red_mask] = [122, 46, 215]  # BGR格式的#D72E7A
        
        # 10. 调整红色简笔画尺寸（放大1.3倍，使其伸出蓝色背景）
        red_sketch_h, red_sketch_w = red_sketch_bgr.shape[:2]
        new_dimensions = (int(red_sketch_w * 1.3), int(red_sketch_h * 1.3))  # 增大放大倍数
        resized_red_sketch_bgr = cv2.resize(red_sketch_bgr, new_dimensions, interpolation=cv2.INTER_LINEAR)
        resized_red_sketch_alpha = cv2.resize(red_sketch_alpha, new_dimensions, interpolation=cv2.INTER_LINEAR)
        
        # 11. 将黑色简笔画放置在蓝色背景上（居中）
        black_h, black_w = black_sketch_bgr.shape[:2]
        
        # 计算居中位置（相对于蓝色背景）
        y_offset = (blue_h - black_h) // 2 + (bg_h - blue_h) // 2  # 蓝色背景在透明背景中的起始位置 + 黑色简笔画在蓝色背景中的偏移
        x_offset = (blue_w - black_w) // 2 + (bg_w - blue_w) // 2
        
        # 确保不会越界
        y1, y2 = max(0, y_offset), min(bg_h, y_offset + black_h)
        x1, x2 = max(0, x_offset), min(bg_w, x_offset + black_w)
        
        # 计算实际要复制的区域大小
        copy_h = y2 - y1
        copy_w = x2 - x1
        
        # 计算简笔画中对应的区域
        black_y1 = max(0, -y_offset + (bg_h - blue_h) // 2)
        black_y2 = black_y1 + copy_h
        black_x1 = max(0, -x_offset + (bg_w - blue_w) // 2)
        black_x2 = black_x1 + copy_w
        
        # 只在有效区域内复制黑色简笔画
        if copy_h > 0 and copy_w > 0:
            # 创建ROI区域
            roi = background[y1:y2, x1:x2]
            
            # 创建黑色简笔画的掩码
            black_mask = black_sketch_alpha[black_y1:black_y2, black_x1:black_x2, np.newaxis] / 255.0
            
            # 混合背景和黑色简笔画
            background[y1:y2, x1:x2, :3] = (1.0 - black_mask) * roi[:, :, :3] + black_mask * black_sketch_bgr[black_y1:black_y2, black_x1:black_x2]
            # 更新alpha通道
            background[y1:y2, x1:x2, 3] = np.maximum(roi[:, :, 3], (black_mask[:, :, 0] * 255).astype(np.uint8))
        
        # 12. 将放大后的红色简笔画放置在透明背景上（居中，部分会伸出蓝色背景）
        red_h, red_w = resized_red_sketch_bgr.shape[:2]
        
        # 计算居中位置
        y_offset = (bg_h - red_h) // 2
        x_offset = (bg_w - red_w) // 2
        
        # 确保不会越界
        y1, y2 = max(0, y_offset), min(bg_h, y_offset + red_h)
        x1, x2 = max(0, x_offset), min(bg_w, x_offset + red_w)
        
        # 计算实际要复制的区域大小
        copy_h = y2 - y1
        copy_w = x2 - x1
        
        # 计算红色简笔画中对应的区域
        red_y1 = max(0, -y_offset)
        red_y2 = red_y1 + copy_h
        red_x1 = max(0, -x_offset)
        red_x2 = red_x1 + copy_w
        
        # 只在有效区域内复制红色简笔画
        if copy_h > 0 and copy_w > 0:
            # 创建ROI区域
            roi = background[y1:y2, x1:x2]
            
            # 创建红色简笔画的掩码
            red_mask = resized_red_sketch_alpha[red_y1:red_y2, red_x1:red_x2, np.newaxis] / 255.0
            
            # 混合背景和红色简笔画
            background[y1:y2, x1:x2, :3] = (1.0 - red_mask) * roi[:, :, :3] + red_mask * resized_red_sketch_bgr[red_y1:red_y2, red_x1:red_x2]
            # 更新alpha通道
            background[y1:y2, x1:x2, 3] = np.maximum(roi[:, :, 3], (red_mask[:, :, 0] * 255).astype(np.uint8))
        
        # 13. 保存最终的3D效果图片（PNG格式支持透明度）
        cv2.imwrite(output_path, background)
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_black_sketch):
            os.remove(temp_black_sketch)
        if os.path.exists(temp_red_sketch):
            os.remove(temp_red_sketch)
    
    return output_path

def c3d_effect(image_path, output_path, contrast=1.2, shadow_scale=1.1, shadow_blur=2.5, shadow_offset_x=-10):
    """
    将图片转换为c3d效果：去色->调整对比度->渐变映射为指定颜色->添加阴影效果
    
    参数:
    image_path (str): 输入图片路径
    output_path (str): 输出图片路径
    contrast (float): 对比度调整值，默认为1.2
    shadow_scale (float): 阴影放大倍数，默认为1.1
    shadow_blur (float): 高斯模糊强度，默认为2.5
    shadow_offset_x (int): 阴影水平偏移像素，默认为-10（向左偏移）
    """
    if not CV2_AVAILABLE:
        raise ImportError(f"缺少必要的依赖库: {CV2_IMPORT_ERROR}")
    
    # 1. 读取原始图像
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        raise FileNotFoundError(f"无法读取图片，请检查路径是否正确: {image_path}")
    
    height, width = original_image.shape[:2]
    
    # 2. 创建透明背景 (尺寸更大以容纳阴影偏移)
    bg_width = int(width * 1.5)
    bg_height = int(height * 1.5)
    background = np.zeros((bg_height, bg_width, 4), dtype=np.uint8)  # 透明背景
    
    # 3. 创建蓝色背景方块 (#1A01EA -> BGR(234, 1, 26))
    blue_bg_size = (int(width * 1.1), int(height * 1.1))
    blue_background = np.full((blue_bg_size[1], blue_bg_size[0], 3), (234, 1, 26), dtype=np.uint8)
    
    # 4. 将蓝色背景放置在透明背景中央
    bg_h, bg_w = background.shape[:2]
    blue_h, blue_w = blue_background.shape[:2]
    y_offset = (bg_h - blue_h) // 2
    x_offset = (bg_w - blue_w) // 2
    
    y1, y2 = max(0, y_offset), min(bg_h, y_offset + blue_h)
    x1, x2 = max(0, x_offset), min(bg_w, x_offset + blue_w)
    
    copy_h = y2 - y1
    copy_w = x2 - x1
    
    if copy_h > 0 and copy_w > 0:
        blue_y1 = max(0, -y_offset)
        blue_y2 = blue_y1 + copy_h
        blue_x1 = max(0, -x_offset)
        blue_x2 = blue_x1 + copy_w
        
        background[y1:y2, x1:x2, :3] = blue_background[blue_y1:blue_y2, blue_x1:blue_x2]
        background[y1:y2, x1:x2, 3] = 255
    
    # 5. 处理主图像
    # 5.1 转换为灰度图（饱和度设为0）
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # 5.2 调整对比度
    adjusted_image = cv2.convertScaleAbs(gray_image, alpha=contrast, beta=0)
    
    # 5.3 转换回BGR格式
    adjusted_bgr = cv2.cvtColor(adjusted_image, cv2.COLOR_GRAY2BGR)
    
    # 5.4 渐变映射为 #DE3097 颜色 (BGR: 151, 48, 222)
    # 创建颜色映射
    color_mapped = np.zeros_like(adjusted_bgr)
    color_mapped[:, :] = [151, 48, 222]  # #DE3097 的BGR值
    
    # 使用灰度值作为alpha通道来混合颜色
    gray_normalized = adjusted_image.astype(np.float32) / 255.0
    gray_3channel = np.stack([gray_normalized] * 3, axis=2)
    
    # 将原始灰度信息与目标颜色混合
    for i in range(3):
        color_mapped[:, :, i] = (adjusted_bgr[:, :, i].astype(np.float32) / 255.0 * color_mapped[:, :, i].astype(np.float32) / 255.0 * 255).astype(np.uint8)
    
    # 更精确的颜色映射：保持亮度变化但使用目标色调
    hsv_image = cv2.cvtColor(adjusted_bgr, cv2.COLOR_BGR2HSV)
    # #DE3097 的HSV值大约是 (320, 0.79, 0.87)，OpenCV中色调范围是0-179
    hsv_image[:, :, 0] = 160  # 色调设为 #DE3097 的色调 (320/2 = 160)
    hsv_image[:, :, 1] = 200  # 饱和度设为较高值
    color_mapped = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    # 6. 创建阴影图
    # 6.1 创建纯色阴影图像 (#040B64 -> BGR: 100, 11, 4))
    shadow_color = [100, 11, 4]  # #040B64 的BGR值
    shadow_image = np.full((height, width, 3), shadow_color, dtype=np.uint8)
    
    # 6.2 使用原图的alpha通道（基于亮度）
    _, alpha_channel = cv2.threshold(adjusted_image, 1, 255, cv2.THRESH_BINARY)
    
    # 6.3 放大阴影图
    new_shadow_size = (int(width * shadow_scale), int(height * shadow_scale))
    resized_shadow = cv2.resize(shadow_image, new_shadow_size, interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(alpha_channel, new_shadow_size, interpolation=cv2.INTER_LINEAR)
    
    # 6.4 高斯模糊
    blur_size = int(shadow_blur * 2) + 1  # 确保是奇数
    if blur_size % 2 == 0:
        blur_size += 1
    if blur_size > 1:
        resized_shadow = cv2.GaussianBlur(resized_shadow, (blur_size, blur_size), 0)
    
    # 7. 放置阴影图（向左偏移）
    shadow_h, shadow_w = resized_shadow.shape[:2]
    
    # 计算阴影位置（相对于蓝色背景，向左偏移）
    shadow_y_offset = (bg_h - shadow_h) // 2
    shadow_x_offset = (bg_w - shadow_w) // 2 + shadow_offset_x  # 向左偏移
    
    shadow_y1, shadow_y2 = max(0, shadow_y_offset), min(bg_h, shadow_y_offset + shadow_h)
    shadow_x1, shadow_x2 = max(0, shadow_x_offset), min(bg_w, shadow_x_offset + shadow_w)
    
    shadow_copy_h = shadow_y2 - shadow_y1
    shadow_copy_w = shadow_x2 - shadow_x1
    
    if shadow_copy_h > 0 and shadow_copy_w > 0:
        shadow_y1_local = max(0, -shadow_y_offset)
        shadow_y2_local = shadow_y1_local + shadow_copy_h
        shadow_x1_local = max(0, -shadow_x_offset)
        shadow_x2_local = shadow_x1_local + shadow_copy_w
        
        # 创建阴影的alpha通道
        shadow_alpha_mask = resized_alpha[shadow_y1_local:shadow_y2_local, shadow_x1_local:shadow_x2_local]
        shadow_alpha_norm = shadow_alpha_mask.astype(np.float32) / 255.0
        shadow_alpha_3channel = np.stack([shadow_alpha_norm] * 3, axis=2)
        
        # 混合阴影到背景
        roi = background[shadow_y1:shadow_y2, shadow_x1:shadow_x2]
        background[shadow_y1:shadow_y2, shadow_x1:shadow_x2, :3] = (
            (1.0 - shadow_alpha_3channel) * roi[:, :, :3] +
            shadow_alpha_3channel * resized_shadow[shadow_y1_local:shadow_y2_local, shadow_x1_local:shadow_x2_local]
        ).astype(np.uint8)
        
        # 更新alpha通道
        background[shadow_y1:shadow_y2, shadow_x1:shadow_x2, 3] = np.maximum(
            roi[:, :, 3],
            shadow_alpha_mask
        )
    
    # 8. 放置主图像（渐变映射后的图像）
    main_y_offset = (bg_h - height) // 2
    main_x_offset = (bg_w - width) // 2
    
    main_y1, main_y2 = max(0, main_y_offset), min(bg_h, main_y_offset + height)
    main_x1, main_x2 = max(0, main_x_offset), min(bg_w, main_x_offset + width)
    
    main_copy_h = main_y2 - main_y1
    main_copy_w = main_x2 - main_x1
    
    if main_copy_h > 0 and main_copy_w > 0:
        main_y1_local = max(0, -main_y_offset)
        main_y2_local = main_y1_local + main_copy_h
        main_x1_local = max(0, -main_x_offset)
        main_x2_local = main_x1_local + main_copy_w
        
        # 创建主图像的alpha通道
        main_alpha_mask = alpha_channel[main_y1_local:main_y2_local, main_x1_local:main_x2_local]
        main_alpha_norm = main_alpha_mask.astype(np.float32) / 255.0
        main_alpha_3channel = np.stack([main_alpha_norm] * 3, axis=2)
        
        # 混合主图像到背景
        roi = background[main_y1:main_y2, main_x1:main_x2]
        background[main_y1:main_y2, main_x1:main_x2, :3] = (
            (1.0 - main_alpha_3channel) * roi[:, :, :3] +
            main_alpha_3channel * color_mapped[main_y1_local:main_y2_local, main_x1_local:main_x2_local]
        ).astype(np.uint8)
        
        # 更新alpha通道
        background[main_y1:main_y2, main_x1:main_x2, 3] = np.maximum(
            roi[:, :, 3],
            main_alpha_mask
        )
    
    # 9. 保存最终结果
    cv2.imwrite(output_path, background)
    
    return output_path
