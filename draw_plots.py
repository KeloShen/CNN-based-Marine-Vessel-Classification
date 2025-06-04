import os
import json
import argparse
import matplotlib.pyplot as plt

def draw_plots(json_path: str, save_dir: str):
    """绘制训练过程中的损失和准确率曲线"""
    
    # 读取训练统计数据
    with open(json_path, 'r') as f:
        statistics = json.load(f)

    loss = statistics['loss']
    accuracy = statistics['accuracy']
    time = statistics.get('time', None)  # 时间数据可能不存在

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 绘制损失曲线
    plt.figure(1)
    plt.plot(range(len(loss)), loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'train_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制准确率曲线
    plt.figure(2)
    plt.plot(range(len(accuracy)), accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'val_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 如果有时间数据，绘制时间曲线
    if time is not None:
        plt.figure(3)
        plt.plot(range(len(time)), time)
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.title('Training Time per Epoch')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'training_time.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Plots have been saved to {save_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Draw training curves')
    parser.add_argument('--json_path', type=str, default='training_statistics.json',
                        help='path to training statistics json file')
    parser.add_argument('--save_dir', type=str, default='plots',
                        help='directory to save plots')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    draw_plots(args.json_path, args.save_dir) 