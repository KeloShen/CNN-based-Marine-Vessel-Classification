# 基于VGG13的海面舰船图像二分类

本项目使用VGG13网络实现对海面舰船图像的二分类任务（船类和非船类）。

## 环境要求

- Python 3.7+
- PyTorch 1.7.0+
- torchvision 0.8.0+
- 其他依赖见requirements.txt

## 项目结构

```
.
├── data/               # 数据集目录
│   ├── train/         # 训练集
│   │   ├── sea/      # 非船类图像
│   │   └── ship/     # 船类图像
│   └── val/          # 验证集
│       ├── sea/      # 非船类图像
│       └── ship/     # 船类图像
├── model.py           # VGG13网络模型定义
├── train.py          # 训练脚本
├── test.py           # 测试脚本
├── draw_plots.py     # 绘制训练曲线脚本
├── requirements.txt   # 项目依赖
└── README.md         # 项目说明
```

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 训练模型：
```bash
python train.py --dataset_root ./data --epochs 50 --batch_size 16 --lr 0.001
```

3. 测试模型：
```bash
python test.py --weights_path weights/vgg13_best.pth
```

4. 绘制训练曲线：
```bash
python draw_plots.py --json_path training_statistics.json --save_dir plots
```

## 参数说明

### 训练参数
- `--dataset_root`: 数据集根目录
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--lr`: 学习率
- `--num_classes`: 分类数量（默认为2）

### 测试参数
- `--weights_path`: 模型权重文件路径
- `--img_paths`: 测试图像路径模式
- `--batch_size`: 测试批次大小

### 绘图参数
- `--json_path`: 训练统计数据文件路径
- `--save_dir`: 图像保存目录

## 模型架构

VGG13网络包含：
- 10个卷积层
- 5个最大池化层
- 3个全连接层
- ReLU激活函数
- Dropout正则化

## 性能指标

测试集上的性能指标包括：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数（F1-Score）
- 推理时间
