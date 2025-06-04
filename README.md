# 基于VGG13的海面舰船图像二分类

本项目使用VGG13网络实现对海面舰船图像的二分类任务（船类和非船类）。

## 环境要求

- Python 3.7+
- PyTorch 1.7.0+
- torchvision 0.8.0+
- 其他依赖见requirements.txt

## 项目结构
```
├── data/                    # 数据集目录
│   ├── sea/                # 非船类图像
│   └── ship/               # 船类图像
├── source_codes/           # 源代码目录
│   ├── model.py           # VGG13网络模型定义
│   ├── train.py           # 训练脚本
│   ├── test.py            # 测试脚本
│   ├── test_pipeline.py   # 完整流程测试脚本
│   ├── split_dataset.py   # 数据集划分脚本
│   ├── draw_plots.py      # 绘制训练曲线脚本
│   ├── check_classification.py  # 分类结果检查脚本
│   └── run_all.py         # 一键运行所有流程的脚本
├── weights/                # 模型权重保存目录
├── plots/                  # 训练曲线图保存目录
├── results/                # 其他结果文件目录
├── colab.ipynb            # Colab notebook文件
├── requirements.txt        # 项目依赖
└── README.md              # 项目说明

注：运行split_dataset.py后，data目录结构将变为：
data/
├── train/                 # 训练集（由split_dataset.py生成）
│   ├── sea/              # 非船类图像
│   └── ship/             # 船类图像
└── val/                  # 验证集（由split_dataset.py生成）
    ├── sea/              # 非船类图像
    └── ship/             # 船类图像
```

## 使用方法

### 方法一：使用Google Colab（推荐）

1. 在Google Colab中打开`colab.ipynb`文件：
   - 访问 https://github.com/KeloShen/CNN-based-Marine-Vessel-Classification
   - 打开`colab.ipynb`文件
   - 点击"在Colab中打开"按钮

2. 配置运行环境：
   - 点击"代码执行程序"→"更改运行时类型"
   - 选择"GPU"作为硬件加速器
   - 确保有足够的运行时内存

3. 运行notebook：
   - 按顺序执行所有代码单元格
   - 观察训练过程和输出结果
   - 所有结果会自动保存到指定目录


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
python source_codes/run_all.py --data_root ./data/ --epochs 100 --batch_size 64 --lr 0.0001
```

或者分步运行：

1. 划分数据集：
```bash
python source_codes/split_dataset.py --dataset_root ./data/ --train_ratio 0.8
```

2. 训练模型：
```bash
python source_codes/train.py --dataset_root ./data/ --epochs 100 --batch_size 64 --lr 0.0001
```

3. 测试模型：
```bash
python source_codes/test.py --weights_path weights/vgg13_best.pth --dataset_root ./data/
```

4. 检查分类结果：
```bash
python source_codes/check_classification.py --weights_path weights/vgg13_best.pth --dataset_root ./data/ --conf_thr 0.9
```

5. 绘制训练曲线：
```bash
python source_codes/draw_plots.py
```

### 方法三：运行测试流程

为了验证整个项目的正确性，我们提供了完整的测试流程：

```bash
python source_codes/test_pipeline.py --data_root ./data/ --test_weights_dir ./test_weights/ --test_plots_dir ./test_plots/
```

测试流程包括：
1. 数据集划分测试（验证8:2的训练验证比例）
2. 模型训练测试（使用小规模数据集快速验证）
3. 模型评估测试（验证准确率计算）
4. 分类结果检查测试（验证低置信度样本识别）
5. 训练曲线绘制测试

测试参数说明：
- `--data_root`: 测试数据集根目录
- `--test_weights_dir`: 测试权重保存目录
- `--test_plots_dir`: 测试图表保存目录
- `--test_batch_size`: 测试批次大小（默认8）
- `--test_epochs`: 测试训练轮数（默认2）

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
