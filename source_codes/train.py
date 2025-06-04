import os
import json
import torch
import torch.nn as nn
import argparse
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import vgg13
from tqdm import tqdm
import sys

def main(args):
    # 获取当前脚本所在目录和项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 数据集加载
    data_root = args.dataset_root
    assert os.path.exists(data_root), f"{data_root} path does not exist."

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_root, "train"),
        transform=data_transform["train"]
    )
    train_num = len(train_dataset)

    # 保存类别信息
    class_idx = train_dataset.class_to_idx
    class_dict = dict((v, k) for k, v in class_idx.items())
    json_str = json.dumps(class_dict, indent=4)
    with open(os.path.join(root_dir, "class_index.json"), "w") as json_file:
        json_file.write(json_str)

    # 数据加载器
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f"Using {nw} dataloader workers")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_root, "val"),
        transform=data_transform["val"]
    )
    val_num = len(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw
    )

    print(f"Using {train_num} images for training, {val_num} images for validation.")

    # 创建模型
    model = vgg13(num_classes=args.num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练参数
    epochs = args.epochs
    best_acc = 0.0
    save_path = os.path.join(root_dir, "weights", "vgg13_best.pth")
    os.makedirs(os.path.join(root_dir, "weights"), exist_ok=True)
    
    # 记录训练过程
    train_steps = len(train_loader)
    loss_history = []
    acc_history = []
    time_history = []

    # 开始训练
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        epoch_start_time = time.time()

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch+1}/{epochs}] loss:{loss:.3f}"

        # 验证阶段
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = f"valid epoch[{epoch+1}/{epochs}]"

        val_accurate = acc / val_num
        epoch_time = time.time() - epoch_start_time
        
        print(f'[epoch {epoch+1}] train_loss: {running_loss/train_steps:.3f}  '
              f'val_accuracy: {val_accurate:.3f}  '
              f'time: {epoch_time:.2f}s')

        # 记录训练数据
        loss_history.append(running_loss / train_steps)
        acc_history.append(val_accurate)
        time_history.append(epoch_time)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    # 保存训练统计数据
    statistics = {
        "loss": loss_history,
        "accuracy": acc_history,
        "time": time_history
    }
    with open(os.path.join(root_dir, "training_statistics.json"), "w") as f:
        json.dump(statistics, f, indent=4)

    print("Training finished!")
    print(f"Best accuracy: {best_acc}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train VGG13 on Ship Classification')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--dataset_root', type=str, default='./data',
                        help='dataset root path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args) 