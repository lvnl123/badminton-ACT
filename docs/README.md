# TrackNetV3 with Attention + MMPose - 羽毛球轨迹追踪与姿态检测

基于TrackNetV3并添加CBAM（Convolutional Block Attention Module）注意力机制的羽毛球轨迹追踪系统，集成MMPose进行球员姿态检测。

## 项目结构

```
TrackNetV3_Attention/
├── TrackNetAttention.py    # 带有CBAM注意力机制的TrackNetV3模型
├── ball_detect.py          # 球检测和轨迹追踪主程序
├── pose_detect.py          # MMPose姿态检测模块
├── visualize_combined.py   # 轨迹和姿态联合可视化
├── run_combined.py         # 完整流程运行脚本（轨迹+姿态）
├── denoise.py              # 轨迹平滑和去噪
├── kalman_filter.py        # 卡尔曼滤波器实现
├── utils.py                # JSON读写工具函数
├── visualize.py            # 轨迹可视化
├── run.py                  # 轨迹检测运行脚本
├── ball_track_attention.pt # 模型权重文件
└── results/                # 结果输出目录
```

## 核心特性

### 1. CBAM注意力机制

实现了CBAM（Convolutional Block Attention Module）注意力机制，包含：

- **通道注意力（Channel Attention）**：通过全局平均池化和最大池化来捕获通道间的依赖关系
- **空间注意力（Spatial Attention）**：通过空间上下文信息来捕获特征图中的重要区域

注意力模块被添加到编码器和解码器的每个阶段，帮助模型更好地关注羽毛球的关键特征。

### 2. 轨迹追踪流程

1. **球检测**：使用TrackNetV3 with Attention模型检测每一帧中的羽毛球位置
2. **异常检测**：基于帧间距离检测异常检测点
3. **轨迹平滑**：使用卡尔曼滤波器对轨迹进行平滑处理
4. **可视化**：将检测到的轨迹绘制在视频上

### 3. MMPose姿态检测

集成了MMPose进行球员姿态检测：

- **人体姿态估计**：使用MMPose的预训练模型检测17个人体关键点
- **双人检测**：自动识别视频中的两名球员
- **骨骼可视化**：将检测到的骨骼连接可视化在视频上
- **联合可视化**：同时显示羽毛球轨迹和球员姿态

## 使用方法

### 快速开始

#### 方式1：仅轨迹检测

```bash
cd TrackNetV3_Attention
python run.py --video <视频路径> --result_dir <结果保存路径>
```

#### 方式2：轨迹检测 + 姿态检测（推荐）

```bash
cd TrackNetV3_Attention
python run_combined.py --video <视频路径> --result_dir <结果保存路径>
```

### 参数说明

#### run.py（仅轨迹检测）

- `--video`: 输入视频路径（必需）
- `--result_dir`: 结果保存路径（默认：./results）
- `--model`: 模型权重文件路径（默认：ball_track_attention.pt）
- `--num_frames`: 输入序列的帧数（默认：3）
- `--traj_len`: 可视化时显示的轨迹长度（默认：10）

#### run_combined.py（轨迹+姿态检测）

- `--video`: 输入视频路径（必需）
- `--result_dir`: 结果保存路径（默认：./results）
- `--model`: 模型权重文件路径（默认：ball_track_attention.pt）
- `--num_frames`: 输入序列的帧数（默认：3）
- `--threshold`: 检测阈值（0.0-1.0，默认：0.5）
- `--traj_len`: 可视化时显示的轨迹长度（默认：10）
- `--device`: 设备选择（cuda/cpu，默认：cuda）

### 示例

#### 仅轨迹检测

```bash
python run.py --video ../test2.mp4 --result_dir ./results --traj_len 15
```

#### 轨迹+姿态检测

```bash
python run_combined.py --video ../test2.mp4 --result_dir ./results --traj_len 15 --threshold 0.3
```

## 输出结果

### 仅轨迹检测模式

运行完成后，会在`results`目录下生成：

1. **loca_info/<视频名>/**: 原始检测结果（JSON格式）
2. **loca_info_denoise/<视频名>/**: 平滑后的轨迹（JSON格式）
3. **<视频名>_with_trajectory_attention.mp4**: 带有轨迹标注的视频

### 轨迹+姿态检测模式

运行完成后，会在`results`目录下生成：

1. **loca_info/<视频名>/**: 原始球检测结果（JSON格式）
2. **loca_info_denoise/<视频名>/**: 平滑后的球轨迹（JSON格式）
3. **<视频名>_poses.npy**: 姿态检测数据（NumPy数组，形状：[帧数, 2, 17, 2]）
4. **<视频名>_with_trajectory_attention.mp4**: 带有球轨迹标注的视频
5. **<视频名>_combined.mp4**: 同时显示球轨迹和球员姿态的视频（推荐）

## 模型架构

TrackNetV3 with Attention模型在原始TrackNetV3的基础上，在以下位置添加了CBAM注意力模块：

- 编码器阶段1（64通道）
- 编码器阶段2（128通道）
- 编码器阶段3（256通道）
- 编码器阶段4（512通道）
- 解码器阶段1（256通道）
- 解码器阶段2（128通道）
- 解码器阶段3（64通道）

## 技术细节

### CBAM注意力模块

```python
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)
```

### 卡尔曼滤波器

用于轨迹平滑，参数包括：
- `max_gap`: 最大允许的连续缺失帧数（默认：8）
- `process_noise`: 过程噪声协方差（默认：5.0）
- `measurement_noise`: 测量噪声协方差（默认：20.0）

## 依赖项

### 基础依赖

- PyTorch
- OpenCV
- NumPy
- Pandas
- tqdm

### MMPose依赖（仅用于姿态检测）

- mmpose
- mmdet
- mmcv

安装MMPose依赖：

```bash
pip install mmpose
pip install mmdet
pip install mmcv
```

## 注意事项

1. 模型权重文件需要与模型架构匹配
2. 如果使用预训练的TrackNetV3权重，attention层会随机初始化
3. 视频处理速度取决于GPU性能
4. MMPose首次运行时会自动下载预训练模型，需要网络连接
5. 姿态检测需要较多显存，建议使用GPU运行
6. 如果显存不足，可以降低视频分辨率或使用CPU运行（速度较慢）
