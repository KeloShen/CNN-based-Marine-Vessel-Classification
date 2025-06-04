import os
import shutil
import unittest
from pathlib import Path
import json
import torch
from PIL import Image
import numpy as np

class TestShipClassificationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建测试数据目录
        cls.test_root = "test_data"
        cls.data_dir = os.path.join(cls.test_root, "data")
        cls.sea_dir = os.path.join(cls.data_dir, "sea")
        cls.ship_dir = os.path.join(cls.data_dir, "ship")
        
        # 创建必要的目录
        os.makedirs(cls.sea_dir, exist_ok=True)
        os.makedirs(cls.ship_dir, exist_ok=True)
        
        # 创建测试图片
        cls._create_test_images()
        
        # 创建类别索引文件
        cls._create_class_index()

    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
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

    @classmethod
    def _create_test_images(cls):
        """创建测试用的图片"""
        # 创建一些随机图片用于测试
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

    @classmethod
    def _create_class_index(cls):
        """创建类别索引文件"""
        class_index = {
            "0": "sea",
            "1": "ship"
        }
        with open("class_index.json", "w") as f:
            json.dump(class_index, f)

    def test_1_split_dataset(self):
        """测试数据集划分"""
        print("\n测试数据集划分...")
        cmd = f"python split_dataset.py --dataset_root {self.data_dir} --train_ratio 0.8"
        self.assertEqual(os.system(cmd), 0)
        
        # 验证目录结构
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "train", "sea")))
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "train", "ship")))
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "val", "sea")))
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "val", "ship")))

    def test_2_train_model(self):
        """测试模型训练"""
        print("\n测试模型训练...")
        cmd = f"python train.py --dataset_root {self.data_dir} --epochs 2 --batch_size 2 --lr 0.001"
        self.assertEqual(os.system(cmd), 0)
        
        # 验证权重文件是否生成
        self.assertTrue(os.path.exists("weights/vgg13_best.pth"))
        
        # 验证训练统计文件是否生成
        self.assertTrue(os.path.exists("training_statistics.json"))

    def test_3_test_model(self):
        """测试模型评估"""
        print("\n测试模型评估...")
        cmd = f"python test.py --weights_path weights/vgg13_best.pth --img_paths {self.data_dir}/val/*/*.* --batch_size 2 --cls_index class_index.json"
        self.assertEqual(os.system(cmd), 0)

    def test_4_check_classification(self):
        """测试分类结果检查"""
        print("\n测试分类结果检查...")
        cmd = f"python check_classification.py --weights_path weights/vgg13_best.pth --img_paths {self.data_dir}/val/*/*.* --batch_size 2 --conf_thr 0.9 --cls_index class_index.json"
        self.assertEqual(os.system(cmd), 0)
        
        # 验证是否生成了低置信度预测结果目录
        self.assertTrue(os.path.exists("low_confidence_predictions"))

    def test_5_draw_plots(self):
        """测试绘图功能"""
        print("\n测试绘图功能...")
        cmd = "python draw_plots.py --json_path training_statistics.json --save_dir plots"
        self.assertEqual(os.system(cmd), 0)
        
        # 验证是否生成了图表
        self.assertTrue(os.path.exists("plots/train_loss.png"))
        self.assertTrue(os.path.exists("plots/val_accuracy.png"))

    def test_6_full_pipeline(self):
        """测试完整流程"""
        print("\n测试完整流程...")
        cmd = f"python run_all.py --data_root {self.data_dir} --epochs 2 --batch_size 2 --lr 0.001"
        self.assertEqual(os.system(cmd), 0)

if __name__ == '__main__':
    unittest.main(verbosity=2) 