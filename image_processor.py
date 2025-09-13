# 常量定义
# 颜色常量 (BGR格式)
COLOR_RED_DARK = (234, 1, 26)          # #1A01EA
COLOR_RED_BRIGHT = (122, 46, 215)      # #D72E7A
COLOR_PINK = (151, 48, 222)            # #DE3097
COLOR_BLUE_DARK = (100, 11, 4)         # #040B64

# 尺寸缩放常量
SKETCH_3D_BG_SCALE = 1.8                # 3D草图背景缩放
SKETCH_3D_RED_SCALE = 1.3               # 3D草图红色线条缩放
SKETCH_3D_BLUE_SCALE = 1.1              # 3D草图蓝色背景缩放
C3D_BG_SCALE = 1.5                      # C3D效果背景缩放
C3D_BLUE_SCALE = 1.1                    # C3D效果蓝色背景缩放
C3D_SHADOW_SCALE = 1.1                  # C3D效果阴影缩放

# 图像处理常量
C3D_DEFAULT_CONTRAST = 1.2              # C3D默认对比度
C3D_DEFAULT_SHADOW_BLUR = 2.5           # C3D默认阴影模糊
C3D_DEFAULT_SHADOW_OFFSET = -10         # C3D默认阴影偏移

# HSV颜色值
HSV_PINK_HUE = 160                      # #DE3097的色调 (320/2 = 160)
HSV_PINK_SATURATION = 200              # #DE3097的饱和度

# 依赖导入
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
    CV2_IMPORT_ERROR = None
except ImportError as e:
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(e)

from typing import Any, Optional
import os

# rembg模块 - 使用惰性加载
_rembg_remove = None
_PILImage = None
_REMBG_AVAILABLE = None
_REMBG_IMPORT_ERROR = None

def _check_rembg_import() -> bool:
    """检查并尝试导入rembg库"""
    global _rembg_remove, _PILImage, _REMBG_AVAILABLE, _REMBG_IMPORT_ERROR
    
    if _REMBG_AVAILABLE and _rembg_remove is not None and _PILImage is not None:
        return True
    
    return _import_rembg_modules()

def _import_rembg_modules() -> bool:
    """导入rembg相关模块并更新全局变量"""
    return _perform_rembg_import()

def _perform_rembg_import() -> bool:
    """执行rembg模块导入操作"""
    try:
        from rembg import remove
        from PIL import Image
        # 将导入的模块保存到全局变量中
        global _rembg_remove, _PILImage, _REMBG_AVAILABLE, _REMBG_IMPORT_ERROR
        _rembg_remove = remove
        _PILImage = Image
        _REMBG_AVAILABLE = True
        _REMBG_IMPORT_ERROR = None
        return True
    except ImportError as e:
        global _REMBG_AVAILABLE, _REMBG_IMPORT_ERROR
        _REMBG_AVAILABLE = False
        _REMBG_IMPORT_ERROR = str(e)
        return False

def check_dependencies():
    """检查依赖是否可用"""
    return CV2_AVAILABLE

def get_import_error():
    """获取导入错误信息"""
    return None if CV2_AVAILABLE else CV2_IMPORT_ERROR

def check_rembg_dependencies() -> bool:
    """检查rembg依赖是否可用"""
    # 每次都尝试导入，以应对运行时安装的情况
    _check_rembg_import()
    return _REMBG_AVAILABLE or False

def get_rembg_import_error() -> Optional[str]:
    """获取rembg导入错误信息"""
    return None if _REMBG_AVAILABLE else _REMBG_IMPORT_ERROR

def remove_background(image_path: str, output_path: str) -> None:
    """
    移除图片背景并保存结果
    :param image_path: 输入图片路径
    :param output_path: 输出图片路径
    """
    # 确保rembg已导入
    if not check_rembg_dependencies():
        raise ImportError(f"rembg库未正确导入: {get_rembg_import_error()}")
    
    # 类型检查确保_PILImage不为None
    if _PILImage is None or _rembg_remove is None:
        raise ImportError("PIL或rembg模块未正确导入")
        
    try:
        # 打开图片并转换为RGBA模式
        input_image: Any = _PILImage.open(image_path).convert('RGBA')
        
        # 使用rembg移除背景
        output_image: Any = _rembg_remove(input_image)
        
        # 保存处理后的图片
        # 明确调用PIL Image的save方法
        output_image.save(output_path, format='PNG')
    except Exception as e:
        raise RuntimeError(f"处理图片时发生错误: {str(e)}") from e

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

def _create_transparent_background(width: int, height: int, scale: float) -> np.ndarray:
    """创建透明背景"""
    bg_width = int(width * scale)
    bg_height = int(height * scale)
    return np.zeros((bg_height, bg_width, 4), dtype=np.uint8)

def _create_colored_background(width: int, height: int, color: tuple, scale: float) -> np.ndarray:
    """创建指定颜色的背景"""
    bg_width = int(width * scale)
    bg_height = int(height * scale)
    return np.full((bg_height, bg_width, 3), color, dtype=np.uint8)

def _overlay_image_with_alpha(background: np.ndarray, foreground: np.ndarray,
                            foreground_alpha: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
    """将前景图像叠加到背景上，使用alpha通道混合"""
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]
    
    # 计算位置
    y1, y2 = max(0, y_offset), min(bg_h, y_offset + fg_h)
    x1, x2 = max(0, x_offset), min(bg_w, x_offset + fg_w)
    
    copy_h = y2 - y1
    copy_w = x2 - x1
    
    if copy_h > 0 and copy_w > 0:
        # 计算前景图像中对应的区域
        fg_y1 = max(0, -y_offset)
        fg_y2 = fg_y1 + copy_h
        fg_x1 = max(0, -x_offset)
        fg_x2 = fg_x1 + copy_w
        
        # 创建掩码
        fg_mask = foreground_alpha[fg_y1:fg_y2, fg_x1:fg_x2, np.newaxis] / 255.0
        
        # 混合图像
        roi = background[y1:y2, x1:x2]
        background[y1:y2, x1:x2, :3] = (
            (1.0 - fg_mask) * roi[:, :, :3] +
            fg_mask * foreground[fg_y1:fg_y2, fg_x1:fg_x2]
        )
        
        # 更新alpha通道
        background[y1:y2, x1:x2, 3] = np.maximum(
            roi[:, :, 3],
            (fg_mask[:, :, 0] * 255).astype(np.uint8)
        )
    
    return background

def _generate_line_art_in_memory(image_path: str, threshold1: int, threshold2: int) -> tuple:
    """在内存中生成线稿效果，返回BGR图像和alpha通道"""
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

    # 使用高斯模糊降噪
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    
    # 使用Canny边缘检测
    edges = cv2.Canny(blurred_image, threshold1, threshold2)
    
    # 减少腐蚀操作
    kernel = np.ones((1,1), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # 增加闭运算来连接断开的线条
    closing_kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, closing_kernel)
    
    # 创建三通道黑色图像
    black_lines = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    
    return black_lines, edges

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
    将指定的图片转换为3D效果简笔画并放在深红色背景上。
    包含黑色线条简笔画和亮红色放大简笔画，都具有透明背景。

    参数:
    image_path (str): 输入图片的路径。
    output_path (str): 输出3D效果图片的保存路径。
    threshold1 (int): Canny边缘检测的第一个阈值，默认为80。
    threshold2 (int): Canny边缘检测的第二个阈值，默认为160。
    """
    if not CV2_AVAILABLE:
        raise ImportError(f"缺少必要的依赖库: {CV2_IMPORT_ERROR}")
    
    # 1. 读取原始图像并获取尺寸
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        raise FileNotFoundError(f"无法读取图片，请检查路径是否正确: {image_path}")
    
    height, width = original_image.shape[:2]
    
    # 2. 创建透明背景
    background = _create_transparent_background(width, height, SKETCH_3D_BG_SCALE)
    bg_h, bg_w = background.shape[:2]
    
    # 3. 创建红色背景并放置在透明背景中央
    red_background = _create_colored_background(width, height, COLOR_RED_DARK, SKETCH_3D_BLUE_SCALE)
    red_h, red_w = red_background.shape[:2]
    
    # 计算红色背景在透明背景中的位置
    red_y_offset = (bg_h - red_h) // 2
    red_x_offset = (bg_w - red_w) // 2
    
    # 将红色背景叠加到透明背景上
    red_alpha = np.full((red_h, red_w), 255, dtype=np.uint8)
    background = _overlay_image_with_alpha(background, red_background, red_alpha,
                                          red_x_offset, red_y_offset)
    
    # 4. 在内存中生成黑色简笔画
    black_sketch_bgr, black_sketch_alpha = _generate_line_art_in_memory(image_path, threshold1, threshold2)
    black_h, black_w = black_sketch_bgr.shape[:2]
    
    # 5. 在内存中生成红色简笔画
    red_sketch_bgr, red_sketch_alpha = _generate_line_art_in_memory(image_path, threshold1, threshold2)
    
    # 将红色简笔画改为指定颜色
    red_mask = red_sketch_alpha > 0
    red_sketch_bgr[red_mask] = COLOR_RED_BRIGHT
    
    # 6. 调整红色简笔画尺寸（放大）
    red_sketch_h, red_sketch_w = red_sketch_bgr.shape[:2]
    new_dimensions = (int(red_sketch_w * SKETCH_3D_RED_SCALE), int(red_sketch_h * SKETCH_3D_RED_SCALE))
    resized_red_sketch_bgr = cv2.resize(red_sketch_bgr, new_dimensions, interpolation=cv2.INTER_LINEAR)
    resized_red_sketch_alpha = cv2.resize(red_sketch_alpha, new_dimensions, interpolation=cv2.INTER_LINEAR)
    
    # 7. 将黑色简笔画放置在红色背景上（居中）
    black_y_offset = (bg_h - black_h) // 2
    black_x_offset = (bg_w - black_w) // 2
    background = _overlay_image_with_alpha(background, black_sketch_bgr, black_sketch_alpha,
                                          black_x_offset, black_y_offset)
    
    # 8. 将放大后的红色简笔画放置在透明背景上（居中）
    red_h, red_w = resized_red_sketch_bgr.shape[:2]
    red_y_offset = (bg_h - red_h) // 2
    red_x_offset = (bg_w - red_w) // 2
    background = _overlay_image_with_alpha(background, resized_red_sketch_bgr, resized_red_sketch_alpha,
                                          red_x_offset, red_y_offset)
    
    # 9. 保存最终的3D效果图片
    cv2.imwrite(output_path, background)
    
    return output_path

def _apply_color_mapping(image: np.ndarray, target_color: tuple) -> np.ndarray:
    """应用颜色映射到图像"""
    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 设置目标颜色的色调和饱和度
    hsv_image[:, :, 0] = HSV_PINK_HUE
    hsv_image[:, :, 1] = HSV_PINK_SATURATION
    
    # 转换回BGR
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def _create_shadow_effect(image: np.ndarray, shadow_color: tuple, scale: float,
                         blur: float, offset_x: int) -> tuple:
    """创建阴影效果"""
    height, width = image.shape[:2]
    
    # 创建阴影图像
    shadow_image = np.full((height, width, 3), shadow_color, dtype=np.uint8)
    
    # 创建alpha通道（基于亮度）
    _, alpha_channel = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    
    # 放大阴影
    new_shadow_size = (int(width * scale), int(height * scale))
    resized_shadow = cv2.resize(shadow_image, new_shadow_size, interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(alpha_channel, new_shadow_size, interpolation=cv2.INTER_LINEAR)
    
    # 应用高斯模糊
    blur_size = int(blur * 2) + 1
    if blur_size % 2 == 0:
        blur_size += 1
    if blur_size > 1:
        resized_shadow = cv2.GaussianBlur(resized_shadow, (blur_size, blur_size), 0)
    
    return resized_shadow, resized_alpha

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
    
    # 1. 读取原始图像并获取尺寸
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        raise FileNotFoundError(f"无法读取图片，请检查路径是否正确: {image_path}")
    
    height, width = original_image.shape[:2]
    
    # 2. 创建透明背景
    background = _create_transparent_background(width, height, C3D_BG_SCALE)
    bg_h, bg_w = background.shape[:2]
    
    # 3. 创建红色背景并放置在透明背景中央
    red_background = _create_colored_background(width, height, COLOR_RED_DARK, C3D_BLUE_SCALE)
    red_h, red_w = red_background.shape[:2]
    
    # 计算红色背景在透明背景中的位置
    red_y_offset = (bg_h - red_h) // 2
    red_x_offset = (bg_w - red_w) // 2
    
    # 将红色背景叠加到透明背景上
    red_alpha = np.full((red_h, red_w), 255, dtype=np.uint8)
    background = _overlay_image_with_alpha(background, red_background, red_alpha,
                                          red_x_offset, red_y_offset)
    
    # 4. 处理主图像
    # 4.1 转换为灰度图并调整对比度
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    adjusted_image = cv2.convertScaleAbs(gray_image, alpha=contrast, beta=0)
    
    # 4.2 转换回BGR格式
    adjusted_bgr = cv2.cvtColor(adjusted_image, cv2.COLOR_GRAY2BGR)
    
    # 4.3 应用颜色映射（移除了死代码，直接使用HSV方法）
    color_mapped = _apply_color_mapping(adjusted_bgr, COLOR_PINK)
    
    # 5. 创建阴影效果
    resized_shadow, resized_alpha = _create_shadow_effect(
        adjusted_image, COLOR_BLUE_DARK, shadow_scale, shadow_blur, shadow_offset_x
    )
    
    # 6. 放置阴影图（向左偏移）
    shadow_h, shadow_w = resized_shadow.shape[:2]
    shadow_y_offset = (bg_h - shadow_h) // 2
    shadow_x_offset = (bg_w - shadow_w) // 2 + shadow_offset_x
    background = _overlay_image_with_alpha(background, resized_shadow, resized_alpha,
                                          shadow_x_offset, shadow_y_offset)
    
    # 7. 放置主图像（颜色映射后的图像）
    main_y_offset = (bg_h - height) // 2
    main_x_offset = (bg_w - width) // 2
    _, alpha_channel = cv2.threshold(adjusted_image, 1, 255, cv2.THRESH_BINARY)
    background = _overlay_image_with_alpha(background, color_mapped, alpha_channel,
                                          main_x_offset, main_y_offset)
    
    # 8. 保存最终结果
    cv2.imwrite(output_path, background)
    
    return output_path
