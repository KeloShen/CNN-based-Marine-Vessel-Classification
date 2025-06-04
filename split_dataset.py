import os
import random
import shutil
import argparse
from glob import glob
from pathlib import Path
from tqdm import tqdm

def get_arguments():
    """定义数据集划分参数
    
    Returns:
        Namespace: 包含数据集路径和训练集比例的参数
    """
    parser = argparse.ArgumentParser(description='数据集划分参数设置')
    parser.add_argument('--dataset_root',
                        default='./data',
                        type=str,
                        help='数据集根目录路径')
    parser.add_argument('--train_ratio',
                        default=0.8,
                        type=float,
                        help='训练集比例(0.5~1.0之间)')
    return parser.parse_args()

def split_dataset(dataset_root: str, train_ratio: float = 0.8):
    """划分数据集为训练集和验证集
    
    Args:
        dataset_root: 数据集根目录
        train_ratio: 训练集比例
    """
    # 验证输入参数
    assert os.path.exists(dataset_root) and os.path.isdir(dataset_root), \
        f'数据集目录 {dataset_root} 不存在!'
    assert 0.5 < train_ratio < 1, f'训练集比例 {train_ratio} 无效! 应在0.5到1.0之间'

    # 获取所有样本路径
    print("正在收集数据集信息...")
    neg_samples = glob(os.path.join(dataset_root, 'sea/*.png'), recursive=False)
    pos_samples = glob(os.path.join(dataset_root, 'ship/*.png'), recursive=False)
    
    # 打印数据集统计信息
    num_neg = len(neg_samples)
    num_pos = len(pos_samples)
    print(f"找到 {num_neg} 张非船只(sea)图片")
    print(f"找到 {num_pos} 张船只(ship)图片")
    print(f"总计 {num_neg + num_pos} 张图片")
    
    # 随机打乱数据
    random.shuffle(neg_samples)
    random.shuffle(pos_samples)

    # 计算划分数量
    num_neg_train = round(num_neg * train_ratio)
    num_pos_train = round(num_pos * train_ratio)
    
    # 划分数据集
    neg_train = neg_samples[:num_neg_train]
    neg_val = neg_samples[num_neg_train:]
    pos_train = pos_samples[:num_pos_train]
    pos_val = pos_samples[num_pos_train:]

    # 创建目录结构
    print("\n创建目录结构...")
    train_dir = os.path.join(dataset_root, 'train')
    val_dir = os.path.join(dataset_root, 'val')
    
    # 清理已存在的目录
    for dir_path in [train_dir, val_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    
    # 创建新目录
    for dir_path in [
        os.path.join(train_dir, 'sea'),
        os.path.join(train_dir, 'ship'),
        os.path.join(val_dir, 'sea'),
        os.path.join(val_dir, 'ship')
    ]:
        os.makedirs(dir_path)

    # 复制文件
    print("\n正在复制训练集文件...")
    for src in tqdm(neg_train, desc="复制非船只训练图片"):
        dst = os.path.join(train_dir, 'sea', Path(src).name)
        shutil.copyfile(src, dst)
    
    for src in tqdm(pos_train, desc="复制船只训练图片"):
        dst = os.path.join(train_dir, 'ship', Path(src).name)
        shutil.copyfile(src, dst)

    print("\n正在复制验证集文件...")
    for src in tqdm(neg_val, desc="复制非船只验证图片"):
        dst = os.path.join(val_dir, 'sea', Path(src).name)
        shutil.copyfile(src, dst)
    
    for src in tqdm(pos_val, desc="复制船只验证图片"):
        dst = os.path.join(val_dir, 'ship', Path(src).name)
        shutil.copyfile(src, dst)

    # 打印划分结果
    print("\n数据集划分完成!")
    print(f"训练集:")
    print(f"- 非船只图片: {len(neg_train)}张")
    print(f"- 船只图片: {len(pos_train)}张")
    print(f"验证集:")
    print(f"- 非船只图片: {len(neg_val)}张")
    print(f"- 船只图片: {len(pos_val)}张")

if __name__ == "__main__":
    args = get_arguments()
    split_dataset(args.dataset_root, args.train_ratio) 