import os
import subprocess
import argparse
import sys

def run_command(cmd):
    """运行命令并打印输出"""
    print(f"\n执行命令: {cmd}\n")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode('utf-8'))
    if stderr:
        print("错误信息:", stderr.decode('utf-8'))
    return process.returncode

def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    parser = argparse.ArgumentParser(description='海面舰船图像分类完整流程')
    parser.add_argument('--data_root', type=str, required=True, help='数据集根目录路径')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    args = parser.parse_args()

    # 创建必要的目录
    os.makedirs(os.path.join(root_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "low_confidence_predictions"), exist_ok=True)

    # 1. 数据集划分
    print("\n=== 第1步：数据集划分 ===")
    cmd = f"python {os.path.join(current_dir, 'split_dataset.py')} --dataset_root {args.data_root} --train_ratio {args.train_ratio}"
    if run_command(cmd) != 0:
        print("数据集划分失败！")
        return

    # 2. 训练模型
    print("\n=== 第2步：训练模型 ===")
    cmd = f"python {os.path.join(current_dir, 'train.py')} --dataset_root {args.data_root} --epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr}"
    if run_command(cmd) != 0:
        print("模型训练失败！")
        return

    # 3. 测试模型
    print("\n=== 第3步：测试模型 ===")
    val_dir = os.path.join(args.data_root, "val")
    weights_path = os.path.join(root_dir, "weights", "vgg13_best.pth")
    cls_index = os.path.join(root_dir, "class_index.json")
    cmd = f"python {os.path.join(current_dir, 'test.py')} --weights_path {weights_path} --img_paths {val_dir} --batch_size {args.batch_size} --cls_index {cls_index}"
    if run_command(cmd) != 0:
        print("模型测试失败！")
        return

    # 4. 检查分类结果
    print("\n=== 第4步：检查分类结果 ===")
    cmd = f"python {os.path.join(current_dir, 'check_classification.py')} --weights_path {weights_path} --img_paths {val_dir} --batch_size {args.batch_size} --conf_thr 0.9 --cls_index {cls_index}"
    if run_command(cmd) != 0:
        print("分类结果检查失败！")
        return

    # 5. 绘制训练曲线
    print("\n=== 第5步：绘制训练曲线 ===")
    stats_path = os.path.join(root_dir, "training_statistics.json")
    plots_dir = os.path.join(root_dir, "plots")
    cmd = f"python {os.path.join(current_dir, 'draw_plots.py')} --json_path {stats_path} --save_dir {plots_dir}"
    if run_command(cmd) != 0:
        print("训练曲线绘制失败！")
        return

    print("\n=== 所有步骤已完成 ===")
    print("\n结果文件位置：")
    print(f"- 模型权重：{weights_path}")
    print(f"- 训练曲线：{plots_dir}")
    print(f"- 分类结果：{os.path.join(root_dir, 'low_confidence_predictions')}")

if __name__ == '__main__':
    main() 