import os
import json
import argparse
import shutil
import torch
from PIL import Image
from torchvision import transforms
from glob import glob
from pathlib import Path

from model import vgg13


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 获取测试集图片路径
    val_dir = args.img_paths
    img_path_list = []
    for class_name in ['sea', 'ship']:
        pattern = os.path.join(val_dir, class_name, '*.png')
        img_path_list.extend(glob(pattern))
    
    assert len(img_path_list) > 0, f"找不到任何图片文件在 {val_dir} 目录下"

    # 加载类别信息
    json_path = args.cls_index
    assert os.path.exists(json_path), f"文件 '{json_path}' 不存在"
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    class_indict_reverse = {v: k for k, v in class_indict.items()}
    
    # 获取真实标签
    ground_truths = [int(class_indict_reverse[Path(x).parent.name])
                     for x in img_path_list]

    # 加载VGG13模型
    model = vgg13(num_classes=args.num_classes).to(device)
    weights_path = args.weights_path
    assert os.path.exists(weights_path), f"权重文件 '{weights_path}' 不存在"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 设置为评估模式
    model.eval()
    batch_size = args.batch_size

    # 存储低置信度预测和错误预测
    lower_conf_preds = []  # 存储低置信度的预测结果
    wrong_preds = set()    # 存储错误分类的图片路径

    print("开始检查分类结果...")
    with torch.no_grad():
        for ids in range(0, len(img_path_list), batch_size):
            # 准备批次数据
            img_list = []
            start_idx = ids
            end_idx = min(ids + batch_size, len(img_path_list))
            
            # 加载和预处理图片
            for img_path in img_path_list[start_idx:end_idx]:
                assert os.path.exists(img_path), f"图片 '{img_path}' 不存在"
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)
            
            batch_imgs = torch.stack(img_list, dim=0)
            batch_labels = ground_truths[start_idx:end_idx]

            # 模型预测
            outputs = model(batch_imgs.to(device))
            predict = torch.softmax(outputs, dim=1)
            probs, classes = torch.max(predict, dim=1)

            # 检查每个预测结果
            for idx, (prob, cls) in enumerate(zip(probs, classes)):
                img_path = img_path_list[start_idx + idx]
                pred_class = class_indict[str(cls.item())]
                pred_prob = prob.item()

                print(f"图片: {img_path}")
                print(f"预测类别: {pred_class}")
                print(f"置信度: {pred_prob:.4f}\n")

                # 检查置信度是否低于阈值
                if pred_prob < args.conf_thr:
                    lower_conf_preds.append([
                        img_path,
                        pred_class,
                        pred_prob
                    ])

            # 找出分类错误的样本
            batch_preds = classes.cpu().numpy().tolist()
            # 找出假阳性(FP)和假阴性(FN)样本
            FP = [i for i, (g, p) in enumerate(zip(batch_labels, batch_preds))
                  if g == 0 and p == 1]  # 实际为非船只，预测为船只
            FN = [i for i, (g, p) in enumerate(zip(batch_labels, batch_preds))
                  if g == 1 and p == 0]  # 实际为船只，预测为非船只
            
            # 将错误预测的图片路径添加到集合中
            for i in FP + FN:
                wrong_preds.add(img_path_list[start_idx + i])

    # 保存低置信度的预测结果
    low_probs_dir = 'low_confidence_predictions'
    if os.path.exists(low_probs_dir):
        shutil.rmtree(low_probs_dir)
    os.makedirs(low_probs_dir)

    print(f"\n保存低置信度预测结果到 '{low_probs_dir}' 目录...")
    for pred in lower_conf_preds:
        src_path = pred[0]
        pred_class = pred[1]
        pred_prob = pred[2]
        dst_name = f'{Path(src_path).stem}_{pred_class}_{pred_prob:.4f}.png'
        dst_path = os.path.join(low_probs_dir, dst_name)
        shutil.copyfile(src_path, dst_path)

    print(f"\n分类错误的图片:")
    for wrong_pred in sorted(wrong_preds):
        print(wrong_pred)

    # 打印统计信息
    print(f"\n统计信息:")
    print(f"总样本数: {len(img_path_list)}")
    print(f"低置信度样本数: {len(lower_conf_preds)}")
    print(f"分类错误样本数: {len(wrong_preds)}")


def parse_args():
    parser = argparse.ArgumentParser(description='检查VGG13模型的分类结果')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='类别数量')
    parser.add_argument('--img_paths', type=str, default='data/val/*/*.png',
                        help='测试图片路径模式')
    parser.add_argument('--cls_index', type=str, default='class_index.json',
                        help='类别索引文件路径')
    parser.add_argument('--weights_path', type=str, default='weights/vgg13_best.pth',
                        help='模型权重文件路径')
    parser.add_argument('--conf_thr', type=float, default=0.9,
                        help='置信度阈值(0.5~1.0)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args) 