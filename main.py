from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.core.message.components import Image, Reply
import os
import tempfile
import base64
from typing import Dict, Any
from .image_processor import sketch_effect, line_art_effect, sketch_3d_effect, c3d_effect, check_dependencies, get_import_error, remove_background, check_rembg_dependencies, get_rembg_import_error

# 尝试导入rembg库
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError as e:
    REMBG_AVAILABLE = False
    REMBG_IMPORT_ERROR = str(e)

def check_rembg_dependencies():
    """检查rembg依赖是否可用"""
    return REMBG_AVAILABLE

def get_rembg_import_error():
    """获取rembg导入错误信息"""
    return None if REMBG_AVAILABLE else REMBG_IMPORT_ERROR

@register("jb", "vmoranv", "一个将图片转换为裸眼3D简笔画的插件", "1.0.0")
class MyPlugin(Star):
    def __init__(self, context: Context, config: Dict[str, Any]):
        super().__init__(context)
        self.config = config

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""
        if not check_dependencies():
            logger.warning("图片处理依赖库未安装或系统缺少必要库文件，插件功能将无法使用。请确保已安装 opencv-python 和 numpy，并且系统包含必要的库文件。")
    
    def check_and_report_rembg(self, event: AstrMessageEvent):
        """检查rembg依赖并报告错误"""
        if not check_rembg_dependencies():
            error_msg = get_rembg_import_error()
            return False, f"缺少必要的依赖库: {error_msg}。\n请确保已安装 rembg 和 Pillow：pip install rembg Pillow"
        return True, None
            
    async def _extract_image_from_event(self, event: AstrMessageEvent) -> tuple[str | None, bool]:
        """从事件中提取图片数据"""
        image_data_base64 = None
        image_found = False
        
        # 从当前对话上下文中获取图片信息
        if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
            for comp in event.message_obj.message:
                if isinstance(comp, Image):
                    try:
                        image_data_base64 = await comp.convert_to_base64()
                        if image_data_base64:
                            image_found = True
                            break
                    except (IOError, ValueError, OSError) as e:
                        logger.warning(f"转换当前消息中的图片到base64失败: {e}")
                    except Exception as e:
                        logger.error(f"处理当前消息中的图片时出现未预期的错误: {e}")
                elif isinstance(comp, Reply):
                    # 处理引用消息中的图片
                    if comp.chain:
                        for reply_comp in comp.chain:
                            if isinstance(reply_comp, Image):
                                try:
                                    image_data_base64 = await reply_comp.convert_to_base64()
                                    if image_data_base64:
                                        image_found = True
                                        logger.info("从引用消息中获取到图片")
                                        break
                                except (IOError, ValueError, OSError) as e:
                                    logger.warning(f"转换引用消息中的图片到base64失败: {e}")
                                except Exception as e:
                                    logger.error(f"处理引用消息中的图片时出现未预期的错误: {e}")
        
        return image_data_base64, image_found

    async def _process_image_with_temp_files(self, event: AstrMessageEvent, image_data_base64: str,
                                           output_suffix: str, process_func):
        """处理图片的通用方法，包含临时文件管理"""
        input_path = None
        output_path = None
        try:
            # 将base64数据转换为bytes
            if image_data_base64:
                image_data = base64.b64decode(image_data_base64)
            else:
                yield event.plain_result("无法获取图片数据!")
                return
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_input:
                tmp_input.write(image_data)
                input_path = tmp_input.name
                
            # 创建输出文件
            output_path = input_path.replace('.jpg', output_suffix)
            
            # 调用处理函数
            process_func(input_path, output_path)
            
            # 发送结果
            result_image = Image.fromFileSystem(output_path)
            yield event.chain_result([result_image])
            
        except Exception as e:
            logger.error(f"处理图片时出错: {e}")
            yield event.plain_result(f"处理图片时出错: {str(e)}")
        finally:
            # 清理临时文件
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)

    @filter.command("jb")
    async def image_to_sketch(self, event: AstrMessageEvent, _=None):
        """将引用的图片转换为简笔画""" 
        # 检查依赖
        if not check_dependencies():
            error_msg = get_import_error()
            yield event.plain_result(f"缺少必要的依赖库: {error_msg}。\n请确保已安装 opencv-python 和 numpy，并且系统包含必要的库文件。")
            return
            
        # 直接从配置中获取参数，而不是通过插件名称作为键
        effect_type = self.config.get("effect_type", "sketch")
        blur_kernel_size = max(1, int(self.config.get("blur_kernel_size", 21)))
        canny_threshold1 = max(1, int(self.config.get("canny_threshold1", 80)))
        canny_threshold2 = max(1, int(self.config.get("canny_threshold2", 100)))
        
        # 确保阈值2大于阈值1
        if canny_threshold2 <= canny_threshold1:
            canny_threshold2 = canny_threshold1 + 80
            
        # 调试日志：输出当前使用的配置参数
        logger.info(f"当前插件配置: {self.config}")
        logger.info(f"效果类型: {effect_type}")
        logger.info(f"模糊核大小: {blur_kernel_size}")
        logger.info(f"Canny阈值1: {canny_threshold1}")
        logger.info(f"Canny阈值2: {canny_threshold2}")
            
        # 提取图片数据
        image_data_base64, image_found = await self._extract_image_from_event(event)
        
        if not image_found:
            yield event.plain_result("请引用一张图片并发送 /jb 命令来转换为简笔画!")
            return
            
        input_path = None
        output_path = None
        try:
            # 将base64数据转换为bytes
            if image_data_base64:
                image_data = base64.b64decode(image_data_base64)
            else:
                yield event.plain_result("无法获取图片数据!")
                return
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_input:
                tmp_input.write(image_data)
                input_path = tmp_input.name
                
            # 创建输出文件
            output_path = input_path.replace('.jpg', '_sketch.png')  # 使用PNG格式以支持透明度
            
            # 根据配置选择效果类型
            if effect_type == "sketch":
                # 转换为素描效果
                sketch_effect(input_path, output_path, blur_kernel_size)
            else:
                # 转换为线稿效果
                line_art_effect(input_path, output_path, canny_threshold1, canny_threshold2)
            
            # 发送结果
            # 使用 Image.fromFileSystem 发送图片
            result_image = Image.fromFileSystem(output_path)
            yield event.chain_result([result_image])
            
        except Exception as e:
            logger.error(f"处理图片时出错: {e}")
            yield event.plain_result(f"处理图片时出错: {str(e)}")
        finally:
            # 清理临时文件
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)

    @filter.command("jb3d")
    async def image_to_sketch_3d(self, event: AstrMessageEvent, _=None):
        """将引用的图片转换为3D效果简笔画""" 
        # 检查依赖
        if not check_dependencies():
            error_msg = get_import_error()
            yield event.plain_result(f"缺少必要的依赖库: {error_msg}。\n请确保已安装 opencv-python 和 numpy，并且系统包含必要的库文件。")
            return
            
        # 从配置中获取 Canny 参数
        canny_threshold1 = max(1, int(self.config.get("canny_threshold1", 80)))
        canny_threshold2 = max(1, int(self.config.get("canny_threshold2", 100)))
        
        # 确保阈值2大于阈值1
        if canny_threshold2 <= canny_threshold1:
            canny_threshold2 = canny_threshold1 + 20
            
        # 调试日志：输出当前使用的配置参数
        logger.info(f"当前插件配置: {self.config}")
        logger.info(f"Canny阈值1: {canny_threshold1}")
        logger.info(f"Canny阈值2: {canny_threshold2}")
            
        # 提取图片数据
        image_data_base64, image_found = await self._extract_image_from_event(event)
        
        if not image_found:
            yield event.plain_result("请引用一张图片并发送 /jb3d 命令来转换为3D简笔画!")
            return
            
        input_path = None
        output_path = None
        try:
            # 将base64数据转换为bytes
            if image_data_base64:
                image_data = base64.b64decode(image_data_base64)
            else:
                yield event.plain_result("无法获取图片数据!")
                return
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_input:
                tmp_input.write(image_data)
                input_path = tmp_input.name
                
            # 创建输出文件 (使用PNG格式)
            output_path = input_path.replace('.jpg', '_sketch_3d.png')
            
            # 转换为3D效果简笔画，使用从配置中获取的Canny参数
            sketch_3d_effect(input_path, output_path, canny_threshold1, canny_threshold2)
            
            # 发送结果
            result_image = Image.fromFileSystem(output_path)
            yield event.chain_result([result_image])
            
        except Exception as e:
            logger.error(f"处理图片时出错: {e}")
            yield event.plain_result(f"处理图片时出错: {str(e)}")
        finally:
            # 清理临时文件
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)

    @filter.command("rembg")
    async def remove_background(self, event: AstrMessageEvent, _=None):
        """移除引用图片的背景"""
        # 检查rembg依赖
        is_available, error_msg = self.check_and_report_rembg(event)
        if not is_available:
            yield event.plain_result(error_msg)
            return
            
        # 提取图片数据
        image_data_base64, image_found = await self._extract_image_from_event(event)
        
        if not image_found:
            yield event.plain_result("请引用一张图片并发送 /rembg 命令来移除背景!")
            return
            
        input_path = None
        output_path = None
        try:
            # 将base64数据转换为bytes
            if image_data_base64:
                image_data = base64.b64decode(image_data_base64)
            else:
                yield event.plain_result("无法获取图片数据!")
                return
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_input:
                tmp_input.write(image_data)
                input_path = tmp_input.name
                
            # 创建输出文件
            output_path = input_path.replace('.png', '_no_bg.png')
            
            # 使用rembg移除背景
            remove_background(input_path, output_path)
            
            # 发送结果
            result_image = Image.fromFileSystem(output_path)
            yield event.chain_result([result_image])
            
        except Exception as e:
            logger.error(f"处理图片时出错: {e}")
            yield event.plain_result(f"处理图片时出错: {str(e)}")
        finally:
            # 清理临时文件
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)

    @filter.command("c3d")
    async def image_to_c3d(self, event: AstrMessageEvent):
        """将引用的图片转换为c3d效果"""
        # 检查依赖
        if not check_dependencies():
            error_msg = get_import_error()
            yield event.plain_result(f"缺少必要的依赖库: {error_msg}。\n请确保已安装 opencv-python 和 numpy，并且系统包含必要的库文件。")
            return
            
        # 从配置中获取参数
        contrast = max(0.1, float(self.config.get("c3d_contrast", 1.2)))
        shadow_scale = max(1.0, float(self.config.get("c3d_shadow_scale", 1.1)))
        shadow_blur = max(0.1, float(self.config.get("c3d_shadow_blur", 2.5)))
        shadow_offset_x = int(self.config.get("c3d_shadow_offset_x", -10))
            
        # 调试日志：输出当前使用的配置参数
        logger.info(f"当前插件配置: {self.config}")
        logger.info(f"C3D对比度: {contrast}")
        logger.info(f"C3D阴影放大: {shadow_scale}")
        logger.info(f"C3D阴影模糊: {shadow_blur}")
        logger.info(f"C3D阴影偏移X: {shadow_offset_x}")
            
        # 提取图片数据
        image_data_base64, image_found = await self._extract_image_from_event(event)
        
        if not image_found:
            yield event.plain_result("请引用一张图片并发送 /c3d 命令来转换为c3d效果!")
            return
            
        input_path = None
        output_path = None
        try:
            # 将base64数据转换为bytes
            if image_data_base64:
                image_data = base64.b64decode(image_data_base64)
            else:
                yield event.plain_result("无法获取图片数据!")
                return
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_input:
                tmp_input.write(image_data)
                input_path = tmp_input.name
                
            # 创建输出文件
            output_path = input_path.replace('.jpg', '_c3d.png')
            
            # 转换为c3d效果
            c3d_effect(input_path, output_path, contrast, shadow_scale, shadow_blur, shadow_offset_x)
            
            # 发送结果
            result_image = Image.fromFileSystem(output_path)
            yield event.chain_result([result_image])
            
        except Exception as e:
            logger.error(f"处理图片时出错: {e}")
            yield event.plain_result(f"处理图片时出错: {str(e)}")
        finally:
            # 清理临时文件
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""