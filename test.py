import os
import json
import torch
import time
import argparse
from PIL import Image
from torchvision import transforms
from model import vgg13
from glob import glob

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 获取测试图像路径
    test_paths = args.img_paths
    img_path_list = glob(test_paths, recursive=True)

    # 加载类别信息
    json_path = args.cls_index
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    class_indict_reverse = {v: k for k, v in class_indict.items()}
    
    # 获取真实标签
    ground_truths = [int(class_indict_reverse[x.split('/')[-2]])
                     for x in img_path_list]

    # 加载模型
    model = vgg13(num_classes=args.num_classes).to(device)
    weights_path = args.weights_path
    assert os.path.exists(weights_path), f"file: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 评估模式
    model.eval()
    batch_size = args.batch_size
    
    # 性能指标统计
    TPs, TNs, FPs, FNs = 0, 0, 0, 0
    total_time = 0.0
    
    with torch.no_grad():
        for ids in range(0, len(img_path_list), batch_size):
            img_list = []
            start_idx = ids
            end_idx = min(ids + batch_size, len(img_path_list))
            
            # 批量处理图像
            for img_path in img_path_list[start_idx:end_idx]:
                assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)
                
            batch_imgs = torch.stack(img_list, dim=0)
            batch_labels = ground_truths[start_idx:end_idx]
            
            # 计时开始
            start_time = time.time()
            
            # 模型预测
            outputs = model(batch_imgs.to(device))
            predict = torch.softmax(outputs, dim=1)
            probs, classes = torch.max(predict, dim=1)
            
            # 计时结束
            end_time = time.time()
            total_time += (end_time - start_time)
            
            # 打印预测结果
            for idx, (prob, cls) in enumerate(zip(probs, classes)):
                print(f"image: {img_path_list[start_idx + idx]}  "
                      f"class: {class_indict[str(cls.item())]}  "
                      f"prob: {prob.item():.3f}")
            
            # 统计性能指标
            batch_preds = classes.cpu().numpy().tolist()
            for gt, pred in zip(batch_labels, batch_preds):
                if gt == pred == 1:
                    TPs += 1
                elif gt == pred == 0:
                    TNs += 1
                elif gt == 0 and pred == 1:
                    FPs += 1
                else:  # gt == 1 and pred == 0
                    FNs += 1

    # 计算性能指标
    total_samples = len(img_path_list)
    accuracy = (TPs + TNs) / total_samples
    precision = TPs / (TPs + FPs) if (TPs + FPs) > 0 else 0
    recall = TPs / (TPs + FNs) if (TPs + FNs) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_time_per_image = total_time / total_samples

    # 打印性能指标
    print("\nTest Results:")
    print(f"Total samples: {total_samples}")
    print(f"True Positives: {TPs}")
    print(f"True Negatives: {TNs}")
    print(f"False Positives: {FPs}")
    print(f"False Negatives: {FNs}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average time per image: {avg_time_per_image*1000:.2f}ms")
    print(f"Total inference time: {total_time:.2f}s")

def parse_args():
    parser = argparse.ArgumentParser(description='Test VGG13 on Ship Classification')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--img_paths', type=str, default='data/val/*/*.png',
                        help='path pattern to test images')
    parser.add_argument('--cls_index', type=str, default='class_index.json',
                        help='path to class index json file')
    parser.add_argument('--weights_path', type=str, default='weights/vgg13_best.pth',
                        help='path to model weights')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for testing')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args) 