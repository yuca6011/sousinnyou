#!/usr/bin/env python3
"""
MLP (多層パーセプトロン) ETL8B分類器
階層構造的パターン認識器との比較実験用

認識率（15クラス漢字認識）:
  30サンプル/クラス (32x32、135テスト):  77.04%
  30サンプル/クラス (64x64、135テスト):  88.15%
  全サンプル (2415サンプル、725テスト):  92.28%
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class ETL8BDataset(Dataset):
    """ETL8Bデータセット用PyTorchデータセット"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 32x32画像を1024次元ベクトルに平坦化
        image = image.flatten().astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return torch.tensor(image), torch.tensor(label, dtype=torch.long)

class MLP(nn.Module):
    """多層パーセプトロン分類器"""
    
    def __init__(self, input_size=1024, hidden_sizes=[512, 256, 128], num_classes=5, dropout=0.3):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 隠れ層構築
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # 出力層
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # パラメータ初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)
    
    def count_parameters(self):
        """パラメータ数をカウント"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLPTrainer:
    """MLP学習・評価クラス"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        
    def train(self, train_loader, val_loader=None, epochs=100, lr=0.001, weight_decay=1e-4):
        """学習実行"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        best_accuracy = 0.0
        start_time = time.time()
        
        print(f"学習開始 - エポック数: {epochs}")
        print(f"モデルサイズ: {self.model.count_parameters():,}パラメータ")
        
        for epoch in range(epochs):
            # 学習フェーズ
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100.0 * correct / total
            
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_accuracy)
            
            # 検証フェーズ
            val_accuracy = 0.0
            if val_loader:
                val_accuracy, _, _, _ = self.evaluate(val_loader)
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
            
            # 進捗表示（10エポックごと）
            if (epoch + 1) % 10 == 0:
                print(f"エポック {epoch+1:3d}/{epochs}: "
                      f"損失={epoch_loss:.4f}, "
                      f"訓練精度={epoch_accuracy:.2f}%, "
                      f"検証精度={val_accuracy:.2f}%")
        
        training_time = time.time() - start_time
        print(f"学習完了 - 時間: {training_time:.2f}秒")
        print(f"最高検証精度: {best_accuracy:.2f}%")
        
        return {
            'training_time': training_time,
            'best_accuracy': best_accuracy,
            'final_train_accuracy': epoch_accuracy
        }
    
    def evaluate(self, test_loader):
        """評価実行"""
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        inference_time = time.time() - start_time
        accuracy = 100.0 * correct / total
        
        return accuracy, all_predictions, all_targets, inference_time
    
    def plot_training_curves(self, save_path=None):
        """学習曲線のプロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 損失曲線
        ax1.plot(self.train_losses)
        ax1.set_title('学習損失')
        ax1.set_xlabel('エポック')
        ax1.set_ylabel('損失')
        ax1.grid(True)
        
        # 精度曲線
        ax2.plot(self.train_accuracies)
        ax2.set_title('訓練精度')
        ax2.set_xlabel('エポック')
        ax2.set_ylabel('精度 (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"学習曲線を保存: {save_path}")
        else:
            plt.show()
        
        plt.close()

def load_etl_data(base_path, target_classes, max_samples=50):
    """ETL8Bデータ読み込み（階層構造認識器と共通）"""
    images = []
    labels = []
    
    print(f"データ読み込み中...")
    
    for class_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(base_path, class_hex)
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        
        if len(image_files) > max_samples:
            np.random.seed(42)
            image_files = np.random.choice(image_files, max_samples, replace=False)
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != (32, 32):
                    img = cv2.resize(img, (32, 32))
                img = img.astype(np.float64) / 255.0
                images.append(img)
                labels.append(class_idx)
        
        decimal = int(class_hex, 16)
        char = chr(decimal) if decimal < 0x10000 else f'[{decimal}]'
        print(f"{class_hex} -> {char} ({len([l for l in labels if l == class_idx])}サンプル)")
    
    print(f"読み込み完了: {len(images)}サンプル")
    return np.array(images), np.array(labels)

def run_mlp_experiment(target_classes, max_samples=50, experiment_name="MLP実験"):
    """MLP実験実行"""
    print("=" * 80)
    print(f"{experiment_name} - {len(target_classes)}クラス分類")
    print("=" * 80)
    
    # データ読み込み
    base_path = "./ETL8B-img-full"
    images, labels = load_etl_data(base_path, target_classes, max_samples)
    
    # 訓練・テスト分割（階層構造認識器と同一設定）
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"\n訓練データ: {len(train_images)}サンプル")
    print(f"テストデータ: {len(test_images)}サンプル")
    
    # データセット作成
    train_dataset = ETL8BDataset(train_images, train_labels)
    test_dataset = ETL8BDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # MLPモデル作成
    num_classes = len(target_classes)
    model = MLP(input_size=1024, hidden_sizes=[512, 256, 128], 
                num_classes=num_classes, dropout=0.3)
    
    trainer = MLPTrainer(model, device)
    
    # 学習実行
    print(f"\n{'='*80}")
    print(f"学習開始")
    print(f"{'='*80}")
    
    train_results = trainer.train(train_loader, test_loader, epochs=100, lr=0.001)
    
    # 評価実行
    print(f"\n{'='*80}")
    print(f"最終評価")
    print(f"{'='*80}")
    
    test_accuracy, predictions, targets, inference_time = trainer.evaluate(test_loader)
    
    # 結果表示
    print(f"\n{'='*80}")
    print(f"結果")
    print(f"{'='*80}")
    print(f"学習時間: {train_results['training_time']:.2f}秒")
    print(f"推論時間: {inference_time:.4f}秒")
    print(f"1サンプルあたり推論時間: {inference_time/len(test_images)*1000:.2f}ms")
    print(f"テスト精度: {test_accuracy:.2f}%")
    print(f"モデルサイズ: {model.count_parameters():,}パラメータ")
    
    # 予測分布
    from collections import Counter
    pred_dist = Counter(predictions)
    true_dist = Counter(targets)
    print(f"\n予測分布: {dict(pred_dist)}")
    print(f"真値分布: {dict(true_dist)}")
    
    # 学習曲線保存
    curve_path = f"./mlp_training_curves_{len(target_classes)}classes.png"
    trainer.plot_training_curves(curve_path)
    
    return {
        'accuracy': test_accuracy,
        'training_time': train_results['training_time'],
        'inference_time': inference_time,
        'model_parameters': model.count_parameters(),
        'predictions': predictions,
        'targets': targets
    }

def main():
    """メイン実行関数"""
    print("MLP vs 階層構造的パターン認識器 比較実験")
    print("=" * 80)
    
    # 5クラス実験
    print("\n🔥 5クラス実験")
    target_classes_5 = ['3a2c', '3a2e', '3a4e', '3a5d', '3a6e']
    results_5 = run_mlp_experiment(target_classes_5, max_samples=50, 
                                  experiment_name="MLP 5クラス実験")
    
    # 15クラス実験
    print("\n🔥 15クラス実験")
    target_classes_15 = ['3a2c', '3a2e', '3a4e', '3a5d', '3a6e', '3a7e', 
                         '3a51', '3a59', '3a60', '3a62', '3a64', '3a72', 
                         '3b4f', '3b6b', '3b6c']
    results_15 = run_mlp_experiment(target_classes_15, max_samples=30, 
                                   experiment_name="MLP 15クラス実験")
    
    # 比較サマリー
    print("\n" + "=" * 80)
    print("📊 MLP実験結果サマリー")
    print("=" * 80)
    print(f"5クラス  - 精度: {results_5['accuracy']:.2f}%, "
          f"学習時間: {results_5['training_time']:.2f}s, "
          f"パラメータ: {results_5['model_parameters']:,}")
    print(f"15クラス - 精度: {results_15['accuracy']:.2f}%, "
          f"学習時間: {results_15['training_time']:.2f}s, "
          f"パラメータ: {results_15['model_parameters']:,}")
    
    print("\n🔄 階層構造的パターン認識器と比較してください:")
    print("- 階層構造 5クラス: 86.67%, 0.04秒")
    print("- 階層構造 15クラス: 71.85%, 0.06秒")

if __name__ == "__main__":
    main()