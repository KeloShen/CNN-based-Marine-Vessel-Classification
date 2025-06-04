# 基于VGG13的海面舰船图像二分类

本项目使用VGG13网络实现对海面舰船图像的二分类任务（船类和非船类）。

## 环境要求

- Python 3.7+
- PyTorch 1.7.0+
- torchvision 0.8.0+
- 其他依赖见requirements.txt

## 项目结构
```
├── data/               # 原始数据集目录
│   ├── sea/           # 非船类图像
│   └── ship/          # 船类图像
├── model.py           # VGG13网络模型定义
├── train.py           # 训练脚本
├── test.py            # 测试脚本
├── split_dataset.py   # 数据集划分脚本
├── draw_plots.py      # 绘制训练曲线脚本
├── check_classification.py  # 分类结果检查脚本
├── run_all.py         # 一键运行所有流程的脚本
├── ship_classification.ipynb  # Colab notebook文件
├── requirements.txt    # 项目依赖
└── README.md          # 项目说明

注：运行split_dataset.py后，data目录结构将变为：
data/
├── train/            # 训练集（由split_dataset.py生成）
│   ├── sea/         # 非船类图像
│   └── ship/        # 船类图像
└── val/             # 验证集（由split_dataset.py生成）
    ├── sea/         # 非船类图像
    └── ship/        # 船类图像
```

## 使用方法

### 方法一：使用Google Colab（推荐）

1. 点击下面的链接在Colab中打开notebook：
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/你的用户名/仓库名/blob/main/ship_classification.ipynb)

2. 在Colab中运行notebook：
   - 点击"代码执行程序"→"更改运行时类型"，选择"GPU"
   - 按顺序运行所有单元格
   - notebook中包含了完整的数据准备、训练和评估流程

### 方法二：本地运行

1. 克隆仓库：
```bash
git clone https://github.com/你的用户名/仓库名.git
cd 仓库名
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 准备数据：
   - 将海面图片数据放入 `data/sea/` 目录
   - 将舰船图片数据放入 `data/ship/` 目录

4. 运行完整流程：
```bash
python run_all.py --data_root ./data/ --epochs 100 --batch_size 64 --lr 0.0001
```

或者分步运行：

1. 划分数据集：
```bash
python split_dataset.py --dataset_root ./data/ --train_ratio 0.8
```

2. 训练模型：
```bash
python train.py --dataset_root ./data/ --epochs 100 --batch_size 64 --lr 0.0001
```

3. 测试模型：
```bash
python test.py --weights_path weights/vgg13_best.pth --dataset_root ./data/
```

4. 检查分类结果：
```bash
python check_classification.py --weights_path weights/vgg13_best.pth --dataset_root ./data/ --conf_thr 0.9
```

5. 绘制训练曲线：
```bash
python draw_plots.py
```

## 输出结果

运行完成后，将生成以下文件：

- `weights/vgg13_best.pth`：训练好的模型权重
- `plots/`：训练过程的损失和准确率曲线
- `low_confidence_predictions/`：低置信度和错误分类的样本

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
