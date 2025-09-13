# move3d

AstrBot 简笔画插件

将用户引用的图片转换为简笔画风格裸眼3D图的 AstrBot 插件。

## 功能

- 将用户引用的图片转换为简笔画风格
- 通过 `/jb` 命令触发基础简笔画功能
- 通过 `/jb3d` 命令触发3D立体简笔画功能
- 通过 `/c3d` 命令触发特殊颜色映射3D效果
- 通过 `/rembg` 命令移除图片背景
- 支持素描风格和线稿风格两种效果
- 可配置模糊核大小和边缘检测参

## 使用方法

### 基础简笔画 (/jb)
1. 在聊天中引用一张图片
2. 发送 `/jb` 命令
3. 等待插件处理并返回简笔画图片

### 3D立体简笔画 (/jb3d)
1. 在聊天中引用一张图片
2. 发送 `/jb3d` 命令
3. 等待插件处理并返回3D立体效果简笔画图片

### 特殊颜色映射3D效果 (/c3d)
1. 在聊天中引用一张图片
2. 发送 `/c3d` 命令
3. 等待插件处理并返回特殊颜色映射的3D效果图片

### 背景移除 (/rembg)
1. 在聊天中引用一张图片
2. 发送 `/rembg` 命令
3. 等待插件处理并返回移除背景的图片

## 配置选项

插件支持以下配置选项，可以通过 AstrBot 管理面板进行配置：

- **effect_type**：选择素描风格或线稿风格
- **blur_kernel_size**：调整素描效果的模糊程度（仅在素描风格下生效）
- **canny_threshold1**：调整线稿效果的边缘检测第一个阈值（仅在线稿风格下生效）
- **canny_threshold2**：调整线稿效果的边缘检测第二个阈值（仅在线稿风格下生效）
- **c3d_contrast**：调整C3D效果的对比度强度（仅在C3D效果下生效）
- **c3d_shadow_scale**：调整C3D效果的阴影放大倍数（仅在C3D效果下生效）
- **c3d_shadow_blur**：调整C3D效果的阴影模糊强度（仅在C3D效果下生效）
- **c3d_shadow_offset_x**：调整C3D效果的阴影水平偏移像素（仅在C3D效果下生效）

## Docker 环境中的特殊说明

如果您在 Docker 环境中运行 AstrBot（例如使用官方 Docker 镜像），需要安装额外的系统依赖库才能正常使用此插件。

### Python 依赖

AstrBot 会自动从 `requirements.txt` 文件安装以下 Python 依赖：
- opencv-python
- numpy
- onnxruntime
- rembg
- pillow

如果依赖安装后仍然出现问题，请手动安装：
```bash
pip install opencv-python numpy rembg onnxruntime pillow
```

### 系统依赖安装

即使 Python 依赖已安装，OpenCV 仍然需要系统库才能正常工作：

1. **进入 Docker 容器**：
   ```bash
   docker exec -it astrbot /bin/bash
   ```

2. **安装必要的系统库**：
   ```bash
   # 对于基于 Debian/Ubuntu 的系统
   apt-get update
   apt-get install -y libgl1-mesa-glx libglib2.0-dev libgomp1
   
   # 对于 Alpine Linux
   apk add mesa-gl libgomp
   ```

3. **重启 AstrBot 容器**：
   ```bash
   docker restart astrbot
   ```

## 支持

[帮助文档](https://astrbot.app)

## 作者

[vmoranv](https://github.com/vmoranv)