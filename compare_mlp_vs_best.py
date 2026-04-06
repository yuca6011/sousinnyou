#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最良モデル vs MLP 比較実験

最良モデル（PNN + HOG + データ拡張）とMLP（多層パーセプトロン）の性能を比較

認識率（15クラス、各30サンプル、450サンプル、135テスト）:
  Best Model (PNN+HOG+Aug): 88.89%（学習6.11秒）
  MLP:                      88.15%（学習12.67秒）
"""

import numpy as np
import cv2
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from kanji_best_standalone import KanjiBestRecognizer, load_etl8b_data


# ============================================================================
# MLP実装
# ============================================================================

class ETL8BDataset(Dataset):
    """PyTorch Dataset for ETL8B"""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].flatten().astype(np.float32)
        label = self.labels[idx]
        return torch.tensor(image), torch.tensor(label, dtype=torch.long)


class MLP(nn.Module):
    """多層パーセプトロン分類器"""

    def __init__(self, input_size=4096, hidden_sizes=[1024, 512, 256],
                 num_classes=15, dropout=0.3):
        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        # 隠れ層
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # 出力層
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)

    def count_parameters(self):
        """パラメータ数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPTrainer:
    """MLP学習・評価"""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.train_accuracies = []

    def train(self, train_loader, val_loader=None, epochs=100,
              lr=0.001, weight_decay=1e-4):
        """学習"""
        optimizer = optim.Adam(self.model.parameters(),
                              lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        print(f"\nMLP学習開始")
        print(f"  エポック数: {epochs}")
        print(f"  パラメータ数: {self.model.count_parameters():,}")
        print(f"  学習率: {lr}")

        best_accuracy = 0.0

        for epoch in range(epochs):
            # 訓練
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_accuracy = 100.0 * correct / total
            avg_loss = total_loss / len(train_loader)

            self.train_losses.append(avg_loss)
            self.train_accuracies.append(train_accuracy)

            # 検証
            if val_loader is not None and (epoch + 1) % 10 == 0:
                val_accuracy = self.evaluate(val_loader)
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy

                print(f"  Epoch {epoch+1:3d}/{epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"Train Acc={train_accuracy:.2f}%, "
                      f"Val Acc={val_accuracy:.2f}%")
            elif (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"Train Acc={train_accuracy:.2f}%")

        return best_accuracy

    def evaluate(self, test_loader):
        """評価"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def predict(self, test_loader):
        """予測"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)


# ============================================================================
# 比較実験
# ============================================================================

def experiment_best_model(train_images, train_labels, test_images, test_labels):
    """最良モデルの実験"""
    print("\n" + "="*80)
    print("実験1: 最良モデル（PNN + HOG + データ拡張）")
    print("="*80)

    recognizer = KanjiBestRecognizer()

    # 学習
    start_time = time.time()
    recognizer.train(train_images, train_labels)
    train_time = time.time() - start_time

    # 評価
    start_time = time.time()
    accuracy, predictions = recognizer.evaluate(test_images, test_labels)
    eval_time = time.time() - start_time

    cm = confusion_matrix(test_labels, predictions)

    print(f"\n結果:")
    print(f"  学習時間: {train_time:.2f}秒")
    print(f"  評価時間: {eval_time:.2f}秒")
    print(f"  認識率: {accuracy:.2f}%")

    return {
        'method': 'Best Model (PNN+HOG+Aug)',
        'accuracy': float(accuracy),
        'train_time': float(train_time),
        'eval_time': float(eval_time),
        'predictions': predictions.tolist(),
        'confusion_matrix': cm.tolist()
    }


def experiment_mlp(train_images, train_labels, test_images, test_labels,
                   device='cpu', epochs=100):
    """MLPの実験"""
    print("\n" + "="*80)
    print("実験2: MLP（多層パーセプトロン）")
    print("="*80)

    # データセット作成
    train_dataset = ETL8BDataset(train_images, train_labels)
    test_dataset = ETL8BDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # モデル作成
    model = MLP(input_size=64*64,
                hidden_sizes=[1024, 512, 256],
                num_classes=15,
                dropout=0.3)

    trainer = MLPTrainer(model, device=device)

    # 学習
    start_time = time.time()
    trainer.train(train_loader, val_loader=test_loader,
                 epochs=epochs, lr=0.001)
    train_time = time.time() - start_time

    # 評価
    start_time = time.time()
    accuracy = trainer.evaluate(test_loader)
    predictions = trainer.predict(test_loader)
    eval_time = time.time() - start_time

    cm = confusion_matrix(test_labels, predictions)

    print(f"\n結果:")
    print(f"  学習時間: {train_time:.2f}秒")
    print(f"  評価時間: {eval_time:.2f}秒")
    print(f"  認識率: {accuracy:.2f}%")

    return {
        'method': 'MLP (Deep Neural Network)',
        'accuracy': float(accuracy),
        'train_time': float(train_time),
        'eval_time': float(eval_time),
        'predictions': predictions.tolist(),
        'confusion_matrix': cm.tolist(),
        'train_losses': trainer.train_losses,
        'train_accuracies': trainer.train_accuracies
    }


def visualize_comparison(best_results, mlp_results, class_info, output_dir):
    """結果の可視化"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 認識率比較
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    methods = [best_results['method'], mlp_results['method']]
    accuracies = [best_results['accuracy'], mlp_results['accuracy']]
    colors = ['#27ae60', '#3498db']

    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=2)
    ax.set_ylabel('認識率 (%)', fontsize=13, fontweight='bold')
    ax.set_title('最良モデル vs MLP - 認識率比較', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # 2. 時間比較
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    train_times = [best_results['train_time'], mlp_results['train_time']]
    eval_times = [best_results['eval_time'], mlp_results['eval_time']]

    bars = axes[0].bar(methods, train_times, color=colors, alpha=0.8,
                       edgecolor='black')
    axes[0].set_ylabel('時間 (秒)', fontsize=11, fontweight='bold')
    axes[0].set_title('学習時間比較', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for bar, t in zip(bars, train_times):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{t:.1f}s', ha='center', va='bottom', fontsize=10)

    bars = axes[1].bar(methods, eval_times, color=colors, alpha=0.8,
                       edgecolor='black')
    axes[1].set_ylabel('時間 (秒)', fontsize=11, fontweight='bold')
    axes[1].set_title('評価時間比較', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    for bar, t in zip(bars, eval_times):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{t:.3f}s', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # 3. 混同行列比較
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    cm_best = np.array(best_results['confusion_matrix'])
    cm_mlp = np.array(mlp_results['confusion_matrix'])

    sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens', ax=axes[0],
                cbar_kws={'label': 'Count'}, square=True)
    axes[0].set_title(f'最良モデル 混同行列\n認識率: {best_results["accuracy"]:.2f}%',
                     fontsize=13, fontweight='bold')
    axes[0].set_xlabel('予測ラベル', fontsize=11)
    axes[0].set_ylabel('真のラベル', fontsize=11)

    sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                cbar_kws={'label': 'Count'}, square=True)
    axes[1].set_title(f'MLP 混同行列\n認識率: {mlp_results["accuracy"]:.2f}%',
                     fontsize=13, fontweight='bold')
    axes[1].set_xlabel('予測ラベル', fontsize=11)
    axes[1].set_ylabel('真のラベル', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # 4. MLP学習曲線
    if 'train_losses' in mlp_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(mlp_results['train_losses']) + 1)

        axes[0].plot(epochs, mlp_results['train_losses'], 'b-', linewidth=2)
        axes[0].set_xlabel('エポック', fontsize=11)
        axes[0].set_ylabel('損失', fontsize=11)
        axes[0].set_title('MLP 学習曲線 - 損失', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, mlp_results['train_accuracies'], 'g-', linewidth=2)
        axes[1].set_xlabel('エポック', fontsize=11)
        axes[1].set_ylabel('訓練精度 (%)', fontsize=11)
        axes[1].set_title('MLP 学習曲線 - 精度', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mlp_training_curve.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

    print(f"\n可視化完了: {output_dir}")


def generate_report(best_results, mlp_results, class_info, output_path):
    """比較レポート生成"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(" 最良モデル vs MLP 比較実験レポート\n")
        f.write("="*100 + "\n\n")

        f.write("1. 実験設定\n")
        f.write("-"*100 + "\n")
        f.write(f"  対象クラス数: {len(class_info)}\n")
        f.write(f"  画像サイズ: 64x64\n")
        f.write(f"  訓練データ: 315サンプル\n")
        f.write(f"  テストデータ: 135サンプル\n\n")

        f.write("2. モデル詳細\n")
        f.write("-"*100 + "\n")
        f.write("  最良モデル:\n")
        f.write("    - アーキテクチャ: PNN + HOG特徴 + データ拡張\n")
        f.write("    - HOG特徴: 9方向、8×8セル、階層的抽出\n")
        f.write("    - データ拡張: 5倍（弾性変形・回転・スケール）\n")
        f.write("    - 分類器: 確率的ニューラルネットワーク（適応的σ）\n")
        f.write("    - 学習方法: ノンパラメトリック（反復学習なし）\n\n")

        f.write("  MLP:\n")
        f.write("    - アーキテクチャ: 多層パーセプトロン\n")
        f.write("    - 入力: 64×64=4096次元（画素値の平坦化）\n")
        f.write("    - 隠れ層: [1024, 512, 256]\n")
        f.write("    - 活性化関数: ReLU\n")
        f.write("    - 正則化: Dropout(0.3) + BatchNorm + L2正則化\n")
        f.write("    - 最適化: Adam\n")
        f.write("    - エポック数: 100\n\n")

        f.write("3. 認識率比較\n")
        f.write("-"*100 + "\n")
        f.write(f"{'モデル':<40} {'認識率':<15}\n")
        f.write("-"*100 + "\n")
        f.write(f"{best_results['method']:<40} {best_results['accuracy']:>6.2f}%\n")
        f.write(f"{mlp_results['method']:<40} {mlp_results['accuracy']:>6.2f}%\n")
        f.write("\n")

        diff = best_results['accuracy'] - mlp_results['accuracy']
        if diff > 0:
            f.write(f"  → 最良モデルがMLPより {diff:.2f}% 高い精度\n\n")
        else:
            f.write(f"  → MLPが最良モデルより {-diff:.2f}% 高い精度\n\n")

        f.write("4. 処理時間比較\n")
        f.write("-"*100 + "\n")
        f.write(f"{'モデル':<40} {'学習時間':<15} {'評価時間':<15}\n")
        f.write("-"*100 + "\n")
        f.write(f"{best_results['method']:<40} {best_results['train_time']:>10.2f}秒 "
                f"{best_results['eval_time']:>10.3f}秒\n")
        f.write(f"{mlp_results['method']:<40} {mlp_results['train_time']:>10.2f}秒 "
                f"{mlp_results['eval_time']:>10.3f}秒\n")
        f.write("\n")

        train_speedup = mlp_results['train_time'] / best_results['train_time']
        eval_speedup = mlp_results['eval_time'] / best_results['eval_time']

        f.write(f"  学習時間: 最良モデルはMLPの {1/train_speedup:.2f}倍の速度\n")
        f.write(f"  評価時間: 最良モデルはMLPの {1/eval_speedup:.2f}倍の速度\n\n")

        f.write("5. 特徴比較\n")
        f.write("-"*100 + "\n")
        f.write("  最良モデルの特徴:\n")
        f.write("    ✓ 特徴工学による高精度\n")
        f.write("    ✓ 少ないデータで高性能（データ拡張活用）\n")
        f.write("    ✓ 高速な学習（反復学習不要）\n")
        f.write("    ✓ ハイパーパラメータチューニング不要\n")
        f.write("    ✓ 解釈可能（HOG特徴、階層ピラミッド）\n")
        f.write("    × 特徴設計に専門知識が必要\n\n")

        f.write("  MLPの特徴:\n")
        f.write("    ✓ 特徴設計不要（End-to-End学習）\n")
        f.write("    ✓ 大規模データで強力\n")
        f.write("    ✓ GPU活用で高速化可能\n")
        f.write("    × 多くのデータが必要\n")
        f.write("    × 長い学習時間\n")
        f.write("    × ハイパーパラメータ調整が必要\n")
        f.write("    × 解釈が困難（ブラックボックス）\n\n")

        f.write("6. 結論\n")
        f.write("-"*100 + "\n")
        f.write("  本実験の条件（15クラス、各30サンプル）では、\n")

        if best_results['accuracy'] > mlp_results['accuracy']:
            f.write("  最良モデル（PNN + HOG + データ拡張）がMLPを上回る結果となった。\n\n")
            f.write("  理由:\n")
            f.write("    1. HOG特徴が漢字の形状を効果的に捉える\n")
            f.write("    2. データ拡張により少量データでも高性能\n")
            f.write("    3. PNNがノンパラメトリックで過学習しにくい\n")
        else:
            f.write("  MLPが最良モデルを上回る結果となった。\n\n")
            f.write("  理由:\n")
            f.write("    1. 深層学習の表現力\n")
            f.write("    2. End-to-End学習の柔軟性\n")

        f.write("\n")
        f.write("7. 生成ファイル\n")
        f.write("-"*100 + "\n")
        f.write("  - accuracy_comparison.png: 認識率比較\n")
        f.write("  - time_comparison.png: 処理時間比較\n")
        f.write("  - confusion_matrices.png: 混同行列比較\n")
        f.write("  - mlp_training_curve.png: MLP学習曲線\n")
        f.write("  - comparison_results.json: 詳細結果（JSON）\n")
        f.write("  - comparison_report.txt: このレポート\n")
        f.write("\n")

    print(f"レポート生成完了: {output_path}")


def main():
    """メイン実行"""
    print("="*80)
    print("最良モデル vs MLP 比較実験")
    print("="*80)

    # データ読み込み
    target_classes = [
        '3a2c', '3a2e', '3a4e', '3a6e', '3a51',
        '3a60', '3a64', '3a72', '3b4f', '3b6b',
        '4f40', '3c23', '3d26', '3d3b', '3d3e'
    ]

    images, labels, class_info = load_etl8b_data(
        "./ETL8B-img-full", target_classes, max_samples=30
    )

    # 訓練・テスト分割
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"\n訓練データ: {len(train_images)}サンプル")
    print(f"テストデータ: {len(test_images)}サンプル")

    # デバイス設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")

    # 実験1: 最良モデル
    best_results = experiment_best_model(
        train_images, train_labels, test_images, test_labels
    )

    # 実験2: MLP
    mlp_results = experiment_mlp(
        train_images, train_labels, test_images, test_labels,
        device=device, epochs=100
    )

    # 結果保存
    output_dir = "./results/mlp_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # JSON保存
    results = {
        'best_model': best_results,
        'mlp': mlp_results
    }
    with open(os.path.join(output_dir, 'comparison_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 可視化
    visualize_comparison(best_results, mlp_results, class_info, output_dir)

    # レポート生成
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    generate_report(best_results, mlp_results, class_info, report_path)

    # サマリー表示
    print("\n" + "="*80)
    print("比較実験完了！")
    print("="*80)
    print(f"結果は {output_dir}/ に保存されました\n")

    print("認識率:")
    print(f"  最良モデル: {best_results['accuracy']:.2f}%")
    print(f"  MLP:        {mlp_results['accuracy']:.2f}%")

    print("\n学習時間:")
    print(f"  最良モデル: {best_results['train_time']:.2f}秒")
    print(f"  MLP:        {mlp_results['train_time']:.2f}秒")


if __name__ == "__main__":
    main()
