import os
import shutil
import unittest
from pathlib import Path
import json
import torch
from PIL import Image
import numpy as np
import subprocess
import sys

class TestShipClassificationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        print("\n=== 设置测试环境 ===")
        # 创建测试数据目录
        cls.test_root = "test_data"
        cls.data_dir = os.path.join(cls.test_root, "data")
        
        # 创建原始数据目录
        cls.sea_dir = os.path.join(cls.data_dir, "sea")
        cls.ship_dir = os.path.join(cls.data_dir, "ship")
        
        # 创建必要的目录
        os.makedirs(cls.sea_dir, exist_ok=True)
        os.makedirs(cls.ship_dir, exist_ok=True)
        os.makedirs("weights", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        
        # 创建测试图片
        cls._create_test_images()
        
        # 创建类别索引文件
        cls._create_class_index()
        print("测试环境设置完成！")

    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        print("\n=== 清理测试环境 ===")
        if os.path.exists(cls.test_root):
            shutil.rmtree(cls.test_root)
        if os.path.exists("weights"):
            shutil.rmtree("weights")
        if os.path.exists("plots"):
            shutil.rmtree("plots")
        if os.path.exists("low_confidence_predictions"):
            shutil.rmtree("low_confidence_predictions")
        if os.path.exists("training_statistics.json"):
            os.remove("training_statistics.json")
        if os.path.exists("class_index.json"):
            os.remove("class_index.json")
        print("测试环境清理完成！")

    @classmethod
    def _create_test_images(cls):
        """创建测试用的图片"""
        print("创建测试图片...")
        # 创建测试图片
        for i in range(10):
            # 创建非船只图片（蓝色为主）
            img_sea = Image.fromarray(
                np.random.randint(0, 100, (224, 224, 3), dtype=np.uint8)
            )
            img_sea.save(os.path.join(cls.sea_dir, f"sea_{i}.jpg"))
            
            # 创建船只图片（灰色为主）
            img_ship = Image.fromarray(
                np.random.randint(100, 255, (224, 224, 3), dtype=np.uint8)
            )
            img_ship.save(os.path.join(cls.ship_dir, f"ship_{i}.jpg"))
        
        print(f"创建完成：sea目录 {10}张图片")
        print(f"         ship目录 {10}张图片")

    @classmethod
    def _create_class_index(cls):
        """创建类别索引文件"""
        print("创建类别索引文件...")
        class_index = {
            "0": "sea",
            "1": "ship"
        }
        with open("class_index.json", "w") as f:
            json.dump(class_index, f)
        print("类别索引文件创建完成！")

    def _run_command(self, cmd):
        """运行命令并返回结果"""
        print(f"\n执行命令: {cmd}")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stdout.decode('utf-8'))
        if stderr:
            print("错误信息:", stderr.decode('utf-8'))
        return process.returncode

    def test_1_split_dataset(self):
        """测试数据集划分"""
        print("\n=== 测试数据集划分 ===")
        cmd = f"python split_dataset.py --dataset_root {self.data_dir} --train_ratio 0.8"
        self.assertEqual(self._run_command(cmd), 0)
        
        # 验证目录结构
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "train", "sea")))
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "train", "ship")))
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "val", "sea")))
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "val", "ship")))
        print("数据集划分测试完成！")

    def test_2_train_model(self):
        """测试模型训练"""
        print("\n=== 测试模型训练 ===")
        cmd = f"python train.py --dataset_root {self.data_dir} --epochs 2 --batch_size 2 --lr 0.001"
        self.assertEqual(self._run_command(cmd), 0)
        
        # 验证权重文件是否生成
        self.assertTrue(os.path.exists("weights/vgg13_best.pth"))
        # 验证训练统计文件是否生成
        self.assertTrue(os.path.exists("training_statistics.json"))
        print("模型训练测试完成！")

    def test_3_test_model(self):
        """测试模型评估"""
        print("\n=== 测试模型评估 ===")
        cmd = f"python test.py --weights_path weights/vgg13_best.pth --img_paths {self.data_dir}/val/*/*.* --batch_size 2 --cls_index class_index.json"
        self.assertEqual(self._run_command(cmd), 0)
        print("模型评估测试完成！")

    def test_4_check_classification(self):
        """测试分类结果检查"""
        print("\n=== 测试分类结果检查 ===")
        cmd = f"python check_classification.py --weights_path weights/vgg13_best.pth --img_paths {self.data_dir}/val/*/*.* --batch_size 2 --conf_thr 0.9 --cls_index class_index.json"
        self.assertEqual(self._run_command(cmd), 0)
        
        # 验证是否生成了低置信度预测结果目录
        self.assertTrue(os.path.exists("low_confidence_predictions"))
        print("分类结果检查测试完成！")

    def test_5_draw_plots(self):
        """测试绘图功能"""
        print("\n=== 测试绘图功能 ===")
        cmd = "python draw_plots.py --json_path training_statistics.json --save_dir plots"
        self.assertEqual(self._run_command(cmd), 0)
        
        # 验证是否生成了图表
        self.assertTrue(os.path.exists("plots/train_loss.png"))
        self.assertTrue(os.path.exists("plots/val_accuracy.png"))
        print("绘图功能测试完成！")

    def test_6_full_pipeline(self):
        """测试完整流程"""
        print("\n=== 测试完整流程 ===")
        cmd = f"python run_all.py --data_root {self.data_dir} --epochs 2 --batch_size 2 --lr 0.001"
        self.assertEqual(self._run_command(cmd), 0)
        print("完整流程测试完成！")

if __name__ == '__main__':
    print("开始运行测试...")
    print("Python版本:", sys.version)
    print("PyTorch版本:", torch.__version__)
    print("CUDA是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA设备:", torch.cuda.get_device_name(0))
    unittest.main(verbosity=2) 