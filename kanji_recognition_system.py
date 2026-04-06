"""
階層構造的画像パターン認識器 - 完全な実行システム

使用方法:
1. 必要なライブラリをインストール:
   pip install numpy opencv-python matplotlib scikit-learn seaborn

2. 実行:
   python kanji_recognition_system.py

このスクリプトは以下を実行します:
- 漢字画像の生成（フォントベース）
- 階層ピラミッド構造の構築
- 共通部・非共通部の分類
- 学習と評価
- 結果の可視化

認識率: フォント生成画像を使ったデモ用。実際のETL8B認識率はhierarchical_recognizer.pyを参照。
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # バックエンドを設定（GUIなし環境用）
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import time

#==============================================================
# 設定
#==============================================================
class Config:
    """システム全体の設定"""
    # 画像設定
    IMAGE_SIZE = 64
    PYRAMID_LEVELS = 5
    
    # データ生成設定
    SAMPLES_PER_KANJI = 20
    TRAIN_RATIO = 0.8
    
    # 学習設定
    CLASS_NUM_METHOD = "INCREMENTAL"  # or "FIXED"
    RADIUS_METHOD = "METHOD1"  # or "METHOD2", "METHOD3"
    WEIGHT_UPDATE_METHOD = "NO_UPDATE"  # or "AVERAGE"
    
    # 可視化設定
    VISUALIZE_SAMPLES = True
    VISUALIZE_PYRAMID = True
    VISUALIZE_CONFUSION = True

#==============================================================
# Enumクラス
#==============================================================
class ClassNumMethod(Enum):
    INCREMENTAL = 1
    FIXED = 2

class RadiusMethod(Enum):
    METHOD1 = 1
    METHOD2 = 2
    METHOD3 = 3

class WeightUpdateMethod(Enum):
    NO_UPDATE = 1
    AVERAGE = 2

#==============================================================
# RBFとサブネット
#==============================================================
@dataclass
class RBFUnit:
    centroid: np.ndarray
    activation: float = 0.0
    distance_sq: float = 0.0

@dataclass
class SubNet:
    class_id: int
    rbf_units: List[RBFUnit]
    output: float = 0.0
    
    def __init__(self, class_id: int):
        self.class_id = class_id
        self.rbf_units = []
        self.output = 0.0

#==============================================================
# カーネルメモリー分類器
#==============================================================
class KernelMemoryClassifier:
    def __init__(self):
        self.subnets: List[SubNet] = []
        self.radius_denominator: float = 1.0
    
    def add_rbf_unit(self, subnet_idx: int, centroid: np.ndarray):
        rbf = RBFUnit(centroid=centroid.copy())
        self.subnets[subnet_idx].rbf_units.append(rbf)
    
    def add_subnet(self, class_id: int, initial_centroid: np.ndarray):
        subnet = SubNet(class_id=class_id)
        self.subnets.append(subnet)
        self.add_rbf_unit(len(self.subnets) - 1, initial_centroid)
    
    def forward(self, x: np.ndarray) -> int:
        max_dist_sq = 1e-10
        for subnet in self.subnets:
            for rbf in subnet.rbf_units:
                dist_sq = np.sum((x - rbf.centroid) ** 2)
                rbf.distance_sq = dist_sq
                if max_dist_sq < dist_sq:
                    max_dist_sq = dist_sq
        
        sigma_sq = max(max_dist_sq / (self.radius_denominator ** 2), 1e-10)
        max_output = -1.0
        max_subnet_idx = -1
        
        for i, subnet in enumerate(self.subnets):
            output_sum = 0.0
            for rbf in subnet.rbf_units:
                rbf.activation = np.exp(-rbf.distance_sq / sigma_sq)
                output_sum += rbf.activation
            subnet.output = output_sum / len(subnet.rbf_units) if len(subnet.rbf_units) > 0 else 0
            
            if max_output < subnet.output:
                max_output = subnet.output
                max_subnet_idx = i
        
        return max_subnet_idx
    
    def get_max_activated_rbf(self, subnet_idx: int) -> int:
        subnet = self.subnets[subnet_idx]
        if len(subnet.rbf_units) == 0:
            return 0
        max_activation = subnet.rbf_units[0].activation
        max_idx = 0
        for i, rbf in enumerate(subnet.rbf_units[1:], 1):
            if rbf.activation > max_activation:
                max_activation = rbf.activation
                max_idx = i
        return max_idx

#==============================================================
# 階層ピラミッド生成
#==============================================================
class PyramidGenerator:
    @staticmethod
    def preprocess_image(img: np.ndarray, target_size: Optional[int] = None) -> np.ndarray:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        x_proj = np.sum(binary, axis=0)
        y_proj = np.sum(binary, axis=1)
        
        x_nonzero = np.where(x_proj > 0)[0]
        y_nonzero = np.where(y_proj > 0)[0]
        
        if len(x_nonzero) > 0 and len(y_nonzero) > 0:
            x_min, x_max = x_nonzero[0], x_nonzero[-1]
            y_min, y_max = y_nonzero[0], y_nonzero[-1]
            trimmed = binary[y_min:y_max+1, x_min:x_max+1]
        else:
            trimmed = binary
        
        if target_size is not None:
            h, w = trimmed.shape
            size = max(h, w)
            canvas = np.zeros((size, size), dtype=np.uint8)
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = trimmed
            normalized = cv2.resize(canvas, (target_size, target_size))
        else:
            normalized = trimmed
        
        return normalized
    
    @staticmethod
    def generate_pyramid(img: np.ndarray, num_levels: int = 5) -> List[np.ndarray]:
        pyramid = []
        current = img.copy()
        
        for i in range(num_levels - 1):
            h, w = current.shape
            if h < 2 or w < 2:
                break
            downsampled = cv2.resize(current, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            pyramid.insert(0, downsampled)
            current = downsampled
        
        pyramid.append(img)
        return pyramid

#==============================================================
# 階層構造的パターン認識器
#==============================================================
class HierarchicalPatternRecognizer:
    def __init__(self,
                 class_num_method: ClassNumMethod = ClassNumMethod.INCREMENTAL,
                 radius_method: RadiusMethod = RadiusMethod.METHOD1,
                 weight_update_method: WeightUpdateMethod = WeightUpdateMethod.NO_UPDATE,
                 num_pyramid_levels: int = 5,
                 target_image_size: int = 32):
        
        self.class_num_method = class_num_method
        self.radius_method = radius_method
        self.weight_update_method = weight_update_method
        self.num_pyramid_levels = num_pyramid_levels
        self.target_image_size = target_image_size
        
        self.common_classifier = KernelMemoryClassifier()
        self.non_common_classifiers: Dict[int, KernelMemoryClassifier] = {}
        self.pyramid_gen = PyramidGenerator()
        
        self.total_classes = 0
        self.common_groups: Dict[int, List[int]] = {}
    
    def _compute_radius_denominator(self, current_classes: int) -> float:
        if self.radius_method == RadiusMethod.METHOD1:
            return float(max(current_classes, 1))
        elif self.radius_method == RadiusMethod.METHOD2:
            return float(max(self.total_classes, 1))
        else:
            return 1.0
    
    def train(self, images: List[np.ndarray], labels: np.ndarray, 
              common_group_labels: Optional[np.ndarray] = None):
        n_samples = len(images)
        
        if self.class_num_method == ClassNumMethod.FIXED:
            self.total_classes = len(np.unique(labels))
        
        if common_group_labels is None:
            common_group_labels = labels.copy()
        
        print("Step 1: 階層ピラミッド構造の生成中...")
        pyramids = []
        for i, img in enumerate(images):
            if i % 100 == 0 and i > 0:
                print(f"  処理中: {i}/{n_samples}")
            processed = self.pyramid_gen.preprocess_image(img, self.target_image_size)
            pyramid = self.pyramid_gen.generate_pyramid(processed, self.num_pyramid_levels)
            pyramids.append(pyramid)
        
        print("\nStep 2: 共通部パターン分類器の構築中...")
        self._train_common_classifier(pyramids, common_group_labels)
        
        print("\nStep 3: 非共通部パターン分類器の構築中...")
        self._train_non_common_classifiers(pyramids, labels, common_group_labels)
        
        print("\n学習完了")
        print(f"  共通部分類器: {len(self.common_classifier.subnets)} グループ")
        print(f"  非共通部分類器: {len(self.non_common_classifiers)} 個")
        
        # 統計情報
        total_rbf = sum(len(s.rbf_units) for s in self.common_classifier.subnets)
        print(f"  共通部RBF総数: {total_rbf}")
        for common_id, clf in self.non_common_classifiers.items():
            total_rbf = sum(len(s.rbf_units) for s in clf.subnets)
            print(f"  非共通部{common_id} RBF総数: {total_rbf}")
    
    def _train_common_classifier(self, pyramids: List[List[np.ndarray]], 
                                  common_labels: np.ndarray):
        mid_layer = min(self.num_pyramid_levels // 2, len(pyramids[0]) - 1)
        
        for i, (pyramid, label) in enumerate(zip(pyramids, common_labels)):
            if i % 100 == 0 and i > 0:
                print(f"  共通部学習: {i}/{len(pyramids)}")
            
            feature = pyramid[mid_layer].flatten().astype(np.float32) / 255.0
            
            if i == 0:
                self.common_classifier.add_subnet(int(label), feature)
                if self.class_num_method == ClassNumMethod.INCREMENTAL:
                    self.total_classes = 1
            else:
                subnet_exists = any(s.class_id == int(label) for s in self.common_classifier.subnets)
                
                if not subnet_exists:
                    self.common_classifier.add_subnet(int(label), feature)
                    if self.class_num_method == ClassNumMethod.INCREMENTAL:
                        self.total_classes += 1
                    self.common_classifier.radius_denominator = self._compute_radius_denominator(
                        len(self.common_classifier.subnets))
                else:
                    max_subnet_idx = self.common_classifier.forward(feature)
                    max_subnet = self.common_classifier.subnets[max_subnet_idx]
                    
                    if max_subnet.class_id != int(label):
                        correct_idx = next(j for j, s in enumerate(self.common_classifier.subnets)
                                         if s.class_id == int(label))
                        self.common_classifier.add_rbf_unit(correct_idx, feature)
                    else:
                        if self.weight_update_method == WeightUpdateMethod.AVERAGE:
                            max_rbf_idx = self.common_classifier.get_max_activated_rbf(max_subnet_idx)
                            rbf = max_subnet.rbf_units[max_rbf_idx]
                            rbf.centroid = (rbf.centroid + feature) / 2.0
    
    def _train_non_common_classifiers(self, pyramids: List[List[np.ndarray]],
                                       labels: np.ndarray, common_labels: np.ndarray):
        unique_common = np.unique(common_labels)
        
        for common_id in unique_common:
            indices = np.where(common_labels == common_id)[0]
            
            if len(indices) == 0:
                continue
            
            classifier = KernelMemoryClassifier()
            self.non_common_classifiers[int(common_id)] = classifier
            
            for i, idx in enumerate(indices):
                if i % 50 == 0 and i > 0:
                    print(f"  非共通部学習（グループ{common_id}）: {i}/{len(indices)}")
                
                pyramid = pyramids[idx]
                label = labels[idx]
                feature = pyramid[-1].flatten().astype(np.float32) / 255.0
                
                if i == 0:
                    classifier.add_subnet(int(label), feature)
                else:
                    subnet_exists = any(s.class_id == int(label) for s in classifier.subnets)
                    
                    if not subnet_exists:
                        classifier.add_subnet(int(label), feature)
                        classifier.radius_denominator = self._compute_radius_denominator(
                            len(classifier.subnets))
                    else:
                        max_subnet_idx = classifier.forward(feature)
                        max_subnet = classifier.subnets[max_subnet_idx]
                        
                        if max_subnet.class_id != int(label):
                            correct_idx = next(j for j, s in enumerate(classifier.subnets)
                                             if s.class_id == int(label))
                            classifier.add_rbf_unit(correct_idx, feature)
                        else:
                            if self.weight_update_method == WeightUpdateMethod.AVERAGE:
                                max_rbf_idx = classifier.get_max_activated_rbf(max_subnet_idx)
                                rbf = max_subnet.rbf_units[max_rbf_idx]
                                rbf.centroid = (rbf.centroid + feature) / 2.0
    
    def predict(self, img: np.ndarray) -> int:
        processed = self.pyramid_gen.preprocess_image(img, self.target_image_size)
        pyramid = self.pyramid_gen.generate_pyramid(processed, self.num_pyramid_levels)
        
        mid_layer = min(self.num_pyramid_levels // 2, len(pyramid) - 1)
        common_feature = pyramid[mid_layer].flatten().astype(np.float32) / 255.0
        common_subnet_idx = self.common_classifier.forward(common_feature)
        common_id = self.common_classifier.subnets[common_subnet_idx].class_id
        
        if common_id in self.non_common_classifiers:
            non_common_feature = pyramid[-1].flatten().astype(np.float32) / 255.0
            non_common_classifier = self.non_common_classifiers[common_id]
            non_common_subnet_idx = non_common_classifier.forward(non_common_feature)
            final_class_id = non_common_classifier.subnets[non_common_subnet_idx].class_id
        else:
            final_class_id = common_id
        
        return final_class_id
    
    def evaluate(self, test_images: List[np.ndarray], test_labels: np.ndarray) -> Tuple[float, np.ndarray]:
        correct = 0
        n_test = len(test_images)
        predictions = []
        
        print(f"\n評価中: {n_test}サンプル")
        for i, (img, true_label) in enumerate(zip(test_images, test_labels)):
            if i % 100 == 0 and i > 0:
                print(f"  {i}/{n_test}")
            
            pred_label = self.predict(img)
            predictions.append(pred_label)
            if pred_label == true_label:
                correct += 1
        
        accuracy = 100.0 * correct / n_test
        print(f"\n認識率: {accuracy:.2f}% ({correct}/{n_test})")
        return accuracy, np.array(predictions)

#==============================================================
# 漢字データセット
#==============================================================
class KanjiDataset:
    COMMON_PARTS = {
        0: "門構え",
        1: "しんにょう",
        2: "りっしんべん"
    }
    
    KANJI_DATA = [
        ("聞", 0, 0, "耳"), ("間", 1, 0, "日"), ("問", 2, 0, "口"),
        ("開", 3, 0, "開"), ("閉", 4, 0, "才"),
        ("返", 5, 1, "反"), ("辺", 6, 1, "辺"), ("遂", 7, 1, "㒸"),
        ("進", 8, 1, "隹"), ("道", 9, 1, "首"),
        ("性", 10, 2, "生"), ("快", 11, 2, "夬"), ("悔", 12, 2, "毎"),
        ("情", 13, 2, "青"), ("慣", 14, 2, "貫"),
    ]

#==============================================================
# 画像生成
#==============================================================
class KanjiImageGenerator:
    def __init__(self, img_size: int = 64, font_scale: float = 2.0):
        self.img_size = img_size
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def generate_kanji_image(self, kanji: str, add_noise: bool = False, 
                            rotation: float = 0.0) -> np.ndarray:
        img = np.ones((self.img_size * 2, self.img_size * 2), dtype=np.uint8) * 255
        
        (text_width, text_height), baseline = cv2.getTextSize(
            kanji, self.font, self.font_scale, 2)
        
        x = (img.shape[1] - text_width) // 2
        y = (img.shape[0] + text_height) // 2
        
        cv2.putText(img, kanji, (x, y), self.font, self.font_scale, 0, 2, cv2.LINE_AA)
        
        if rotation != 0.0:
            center = (img.shape[1] // 2, img.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        
        if add_noise:
            noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)
        
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img
    
    def generate_dataset(self, kanji_list: List[str], 
                        samples_per_kanji: int = 10) -> Tuple[List[np.ndarray], List[str]]:
        images = []
        labels = []
        
        for kanji in kanji_list:
            for i in range(samples_per_kanji):
                add_noise = (i % 2 == 1)
                rotation = (i - samples_per_kanji // 2) * 5
                img = self.generate_kanji_image(kanji, add_noise, rotation)
                images.append(img)
                labels.append(kanji)
        
        return images, labels

#==============================================================
# 可視化
#==============================================================
def visualize_results(images, labels_str, predictions, test_labels, kanji_names):
    # サンプル画像
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('サンプル漢字画像', fontsize=16)
    indices = np.random.choice(len(images), min(15, len(images)), replace=False)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(indices):
            i = indices[idx]
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f'{labels_str[i]}', fontsize=14)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - sample_images.png 保存完了")
    
    # 混同行列
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(test_labels, predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=kanji_names, yticklabels=kanji_names)
        plt.title('混同行列', fontsize=16)
        plt.ylabel('真のクラス')
        plt.xlabel('予測クラス')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  - confusion_matrix.png 保存完了")
    except ImportError:
        print("  - seabornがないため混同行列はスキップ")

#==============================================================
# メイン実行
#==============================================================
def main():
    print("\n" + "=" * 60)
    print("階層構造的画像パターン認識器 - 漢字認識システム")
    print("=" * 60)
    
    # データ準備
    kanji_list = [k[0] for k in KanjiDataset.KANJI_DATA]
    kanji_to_id = {k[0]: k[1] for k in KanjiDataset.KANJI_DATA}
    kanji_to_common = {k[0]: k[2] for k in KanjiDataset.KANJI_DATA}
    
    print(f"\n[データセット情報]")
    print(f"  漢字数: {len(kanji_list)}")
    print(f"  共通部グループ: {len(KanjiDataset.COMMON_PARTS)}")
    for cid, name in KanjiDataset.COMMON_PARTS.items():
        kanjis = [k[0] for k in KanjiDataset.KANJI_DATA if k[2] == cid]
        print(f"    {name}: {', '.join(kanjis)}")
    
    # 画像生成
    print(f"\n[画像生成]")
    print(f"  画像サイズ: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"  各漢字のサンプル数: {Config.SAMPLES_PER_KANJI}")
    
    generator = KanjiImageGenerator(img_size=Config.IMAGE_SIZE, font_scale=1.5)
    images, labels_str = generator.generate_dataset(kanji_list, Config.SAMPLES_PER_KANJI)
    
    labels = np.array([kanji_to_id[k] for k in labels_str])
    common_labels = np.array([kanji_to_common[k] for k in labels_str])
    
    # データ分割
    n_total = len(images)
    n_train = int(n_total * Config.TRAIN_RATIO)
    indices = np.random.permutation(n_total)
    
    train_images = [images[i] for i in indices[:n_train]]
    train_labels = labels[indices[:n_train]]
    train_common = common_labels[indices[:n_train]]
    
    test_images = [images[i] for i in indices[n_train:]]
    test_labels = labels[indices[n_train:]]
    
    print(f"\n[データ分割]")
    print(f"  訓練: {len(train_images)} サンプル")
    print(f"  テスト: {len(test_images)} サンプル")
    
    # 学習
    print(f"\n[学習設定]")
    print(f"  クラス数設定法: {Config.CLASS_NUM_METHOD}")
    print(f"  半径値設定法: {Config.RADIUS_METHOD}")
    print(f"  結合係数更新法: {Config.WEIGHT_UPDATE_METHOD}")
    
    class_method = ClassNumMethod[Config.CLASS_NUM_METHOD]
    radius_method = RadiusMethod[Config.RADIUS_METHOD]
    weight_method = WeightUpdateMethod[Config.WEIGHT_UPDATE_METHOD]
    
    recognizer = HierarchicalPatternRecognizer(
        class_num_method=class_method,
        radius_method=radius_method,
        weight_update_method=weight_method,
        num_pyramid_levels=Config.PYRAMID_LEVELS,
        target_image_size=Config.IMAGE_SIZE
    )
    
    start_time = time.time()
    recognizer.train(train_images, train_labels, train_common)
    train_time = time.time() - start_time
    
    # 評価
    start_time = time.time()
    accuracy, predictions = recognizer.evaluate(test_images, test_labels)
    test_time = time.time() - start_time
    
    print(f"\n[処理時間]")
    print(f"  学習時間: {train_time:.2f}秒")
    print(f"  テスト時間: {test_time:.2f}秒")
    print(f"  1サンプルあたり: {test_time/len(test_images)*1000:.2f}ms")
    
    # クラス別精度
    print(f"\n[クラス別認識率]")
    for class_id in sorted(np.unique(test_labels)):
        mask = test_labels == class_id
        if np.sum(mask) > 0:
            class_acc = 100.0 * np.sum(predictions[mask] == test_labels[mask]) / np.sum(mask)
            kanji = [k[0] for k in KanjiDataset.KANJI_DATA if k[1] == class_id][0]
            common_name = KanjiDataset.COMMON_PARTS[
                [k[2] for k in KanjiDataset.KANJI_DATA if k[1] == class_id][0]]
            print(f"  {kanji} ({common_name}): {class_acc:.1f}%")
    
    # 可視化
    if Config.VISUALIZE_SAMPLES or Config.VISUALIZE_CONFUSION:
        print(f"\n[可視化]")
        visualize_results(test_images, [labels_str[i] for i in indices[n_train:]], 
                         predictions, test_labels, kanji_list)
    
    print("\n" + "=" * 60)
    print("実行完了")
    print("=" * 60)

if __name__ == "__main__":
    main()