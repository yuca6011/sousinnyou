# -*- coding: utf-8 -*-
"""
全サンプルでの比較実験
- Best Model（データ拡張あり）
- Best Model（データ拡張なし）
- MLP

各クラス161サンプル、合計2415サンプルで実験

認識率（15クラス、2415サンプル、725テスト）:
  Best Model（拡張なし）: 97.38% ★最高精度（学習5.51秒）
  MLP:                    92.28%（学習63.00秒）
  Best Model（拡張あり）: 82.90%（学習113.16秒）
"""

import os
import sys
import time
import numpy as np
import cv2
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_etl8b_data(data_dir, target_classes, max_samples=None):
    """
    ETL8Bデータを読み込む

    Args:
        data_dir: データディレクトリ
        target_classes: 対象クラスのリスト
        max_samples: 各クラスの最大サンプル数（Noneの場合は全サンプル）
    """
    images = []
    labels = []
    class_info = {}

    print(f"\n{'='*80}")
    print(f"ETL8Bデータ読み込み")
    print(f"{'='*80}")

    for label_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(data_dir, class_hex)

        if not os.path.exists(class_dir):
            print(f"警告: {class_dir} が見つかりません")
            continue

        # 画像ファイルを読み込み
        image_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])

        if max_samples is not None:
            image_files = image_files[:max_samples]

        class_images = []
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                # 64x64にリサイズ
                img = cv2.resize(img, (64, 64))
                class_images.append(img)

        images.extend(class_images)
        labels.extend([label_idx] * len(class_images))

        class_info[label_idx] = {
            'hex': class_hex,
            'count': len(class_images)
        }

        print(f"  クラス {label_idx:2d} ({class_hex}): {len(class_images):4d} サンプル")

    images = np.array(images)
    labels = np.array(labels)

    print(f"\n  合計: {len(images)} サンプル、{len(target_classes)} クラス")
    print(f"{'='*80}\n")

    return images, labels, class_info


class BestModelWithAugmentation:
    """最良モデル（データ拡張あり）"""

    def __init__(self):
        # kanji_best_standaloneから必要な部分をインポート
        from kanji_best_standalone import KanjiBestRecognizer
        self.recognizer = KanjiBestRecognizer()
        self.recognizer.augmentation_factor = 5  # データ拡張5倍

    def fit(self, X, y):
        """学習"""
        self.recognizer.train(X, y)

    def predict(self, X):
        """予測"""
        if len(X.shape) == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])

        predictions = []
        for img in X:
            pred = self.recognizer.predict(img)
            predictions.append(pred)

        return np.array(predictions)


class BestModelWithoutAugmentation:
    """最良モデル（データ拡張なし）"""

    def __init__(self):
        from kanji_best_standalone import KanjiBestRecognizer
        self.recognizer = KanjiBestRecognizer()
        self.recognizer.augmentation_factor = 1  # データ拡張なし

    def fit(self, X, y):
        """学習"""
        self.recognizer.train(X, y)

    def predict(self, X):
        """予測"""
        if len(X.shape) == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])

        predictions = []
        for img in X:
            pred = self.recognizer.predict(img)
            predictions.append(pred)

        return np.array(predictions)


class MLP(nn.Module):
    """多層パーセプトロン"""

    def __init__(self, input_size=4096, hidden_sizes=[1024, 512, 256], num_classes=15):
        super(MLP, self).__init__()

        layers = []
        in_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_size = hidden_size

        layers.append(nn.Linear(in_size, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MLPWrapper:
    """MLPラッパー"""

    def __init__(self, num_classes=15):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_losses = []
        self.train_accuracies = []

    def fit(self, X, y, epochs=100, batch_size=32, lr=0.001):
        """学習"""
        # データを平坦化
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0

        # PyTorchテンソルに変換
        X_tensor = torch.FloatTensor(X_flat)
        y_tensor = torch.LongTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # モデル作成
        self.model = MLP(input_size=4096, num_classes=self.num_classes).to(self.device)

        # 損失関数と最適化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # 学習
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100.0 * correct / total

            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def predict(self, X):
        """予測"""
        self.model.eval()

        # データを平坦化
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        X_tensor = torch.FloatTensor(X_flat).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()


def plot_comparison_results(results, output_dir):
    """比較結果の可視化"""

    os.makedirs(output_dir, exist_ok=True)

    # 1. 認識率比較
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [r['method'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    bars = ax.bar(range(len(methods)), accuracies, color=colors, alpha=0.8)

    # 値をバーの上に表示
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Recognition Accuracy Comparison (Full Dataset: 2415 samples)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison_full.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 処理時間比較
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    train_times = [r['train_time'] for r in results]
    eval_times = [r['eval_time'] for r in results]

    # 学習時間
    bars1 = ax1.bar(range(len(methods)), train_times, color=colors, alpha=0.8)
    for bar, time in zip(bars1, train_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Training Time', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # 評価時間
    bars2 = ax2.bar(range(len(methods)), eval_times, color=colors, alpha=0.8)
    for bar, time in zip(bars2, eval_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Evaluation Time', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison_full.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 混同行列比較
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (result, ax) in enumerate(zip(results, axes)):
        cm = np.array(result['confusion_matrix'])
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(result['method'], fontsize=12, fontweight='bold')

        # カラーバー
        plt.colorbar(im, ax=ax)

        # 軸ラベル
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

        # 数値表示（小さいフォントで）
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_full.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. MLP学習曲線
    mlp_result = [r for r in results if 'train_losses' in r][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(mlp_result['train_losses']) + 1)

    # 損失
    ax1.plot(epochs, mlp_result['train_losses'], 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('MLP Training Loss (Full Dataset)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 精度
    ax2.plot(epochs, mlp_result['train_accuracies'], 'g-', linewidth=2, label='Training Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('MLP Training Accuracy (Full Dataset)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mlp_training_curve_full.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n可視化結果を {output_dir}/ に保存しました")


def generate_report(results, output_dir, class_info):
    """レポート生成"""

    report_path = os.path.join(output_dir, 'full_dataset_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(" 全サンプル比較実験レポート\n")
        f.write("="*100 + "\n\n")

        f.write("1. 実験設定\n")
        f.write("-"*100 + "\n")
        f.write(f"  対象クラス数: 15\n")
        f.write(f"  画像サイズ: 64x64\n")

        # サンプル数計算
        total_samples = sum([info['count'] for info in class_info.values()])
        train_samples = int(total_samples * 0.7)
        test_samples = total_samples - train_samples

        f.write(f"  訓練データ: {train_samples}サンプル\n")
        f.write(f"  テストデータ: {test_samples}サンプル\n")
        f.write(f"  合計: {total_samples}サンプル（各クラス約161サンプル）\n\n")

        f.write("2. モデル詳細\n")
        f.write("-"*100 + "\n")
        f.write("  Best Model（データ拡張あり）:\n")
        f.write("    - HOG特徴抽出\n")
        f.write("    - データ拡張: 5倍（弾性変形・回転・スケール）\n")
        f.write("    - PNN分類器（適応的σ）\n")
        f.write("    - 階層ピラミッド: 5層\n\n")

        f.write("  Best Model（データ拡張なし）:\n")
        f.write("    - HOG特徴抽出\n")
        f.write("    - データ拡張: なし\n")
        f.write("    - PNN分類器（適応的σ）\n")
        f.write("    - 階層ピラミッド: 5層\n\n")

        f.write("  MLP:\n")
        f.write("    - 入力: 64×64=4096次元\n")
        f.write("    - 隠れ層: [1024, 512, 256]\n")
        f.write("    - 活性化関数: ReLU\n")
        f.write("    - 正則化: Dropout(0.3) + BatchNorm + L2\n")
        f.write("    - エポック数: 100\n\n")

        f.write("3. 認識率比較\n")
        f.write("-"*100 + "\n")
        f.write(f"{'モデル':<50} {'認識率':<20}\n")
        f.write("-"*100 + "\n")

        best_acc = max([r['accuracy'] for r in results])
        for result in results:
            marker = " ★最高精度" if result['accuracy'] == best_acc else ""
            f.write(f"{result['method']:<50} {result['accuracy']:>6.2f}%{marker}\n")

        # 差分計算
        f.write("\n  差分分析:\n")
        aug_acc = [r for r in results if 'With Augmentation' in r['method']][0]['accuracy']
        no_aug_acc = [r for r in results if 'No Augmentation' in r['method']][0]['accuracy']
        mlp_acc = [r for r in results if 'MLP' in r['method']][0]['accuracy']

        f.write(f"    データ拡張の効果: {aug_acc - no_aug_acc:+.2f}%\n")
        f.write(f"    拡張あり vs MLP: {aug_acc - mlp_acc:+.2f}%\n")
        f.write(f"    拡張なし vs MLP: {no_aug_acc - mlp_acc:+.2f}%\n\n")

        f.write("4. 処理時間比較\n")
        f.write("-"*100 + "\n")
        f.write(f"{'モデル':<50} {'学習時間':<20} {'評価時間':<20}\n")
        f.write("-"*100 + "\n")

        for result in results:
            f.write(f"{result['method']:<50} {result['train_time']:>10.2f}秒      {result['eval_time']:>10.3f}秒\n")

        f.write("\n  速度比較:\n")
        aug_train = [r for r in results if 'With Augmentation' in r['method']][0]['train_time']
        mlp_train = [r for r in results if 'MLP' in r['method']][0]['train_time']

        f.write(f"    学習時間比（拡張あり/MLP）: {aug_train/mlp_train:.2f}倍\n")

        aug_eval = [r for r in results if 'With Augmentation' in r['method']][0]['eval_time']
        mlp_eval = [r for r in results if 'MLP' in r['method']][0]['eval_time']

        f.write(f"    評価時間比（拡張あり/MLP）: {aug_eval/mlp_eval:.2f}倍\n\n")

        f.write("5. 30サンプル実験との比較\n")
        f.write("-"*100 + "\n")
        f.write("  30サンプル実験結果（参考）:\n")
        f.write("    - Best Model（拡張あり）: 88.89% (450サンプル中135テスト)\n")
        f.write("    - MLP: 88.15% (450サンプル中135テスト)\n\n")

        f.write("  全サンプル実験結果:\n")
        f.write(f"    - Best Model（拡張あり）: {aug_acc:.2f}% (2415サンプル中{test_samples}テスト)\n")
        f.write(f"    - Best Model（拡張なし）: {no_aug_acc:.2f}% (2415サンプル中{test_samples}テスト)\n")
        f.write(f"    - MLP: {mlp_acc:.2f}% (2415サンプル中{test_samples}テスト)\n\n")

        f.write("6. 結論\n")
        f.write("-"*100 + "\n")
        f.write("  全サンプル（2415サンプル）を使用した実験では:\n\n")

        if aug_acc > mlp_acc:
            f.write(f"  ✓ Best Model（拡張あり）がMLPより {aug_acc - mlp_acc:.2f}% 高い精度\n")
        else:
            f.write(f"  ✓ MLPがBest Model（拡張あり）より {mlp_acc - aug_acc:.2f}% 高い精度\n")

        if aug_acc > no_aug_acc:
            f.write(f"  ✓ データ拡張により {aug_acc - no_aug_acc:.2f}% の精度向上\n")

        f.write("\n  データ量の影響:\n")
        f.write("    - サンプル数が増加（450→2415、約5.4倍）\n")
        f.write("    - より信頼性の高い評価が可能\n")
        f.write("    - 学習時間は増加するが、精度も向上\n\n")

        f.write("7. 生成ファイル\n")
        f.write("-"*100 + "\n")
        f.write("  - accuracy_comparison_full.png: 認識率比較\n")
        f.write("  - time_comparison_full.png: 処理時間比較\n")
        f.write("  - confusion_matrices_full.png: 混同行列比較\n")
        f.write("  - mlp_training_curve_full.png: MLP学習曲線\n")
        f.write("  - full_dataset_results.json: 詳細結果（JSON）\n")
        f.write("  - full_dataset_report.txt: このレポート\n\n")

    print(f"\nレポートを {report_path} に保存しました")


def main():
    print("\n" + "="*100)
    print(" 全サンプル比較実験")
    print("="*100)
    print("\n比較内容:")
    print("  1. Best Model（HOG + データ拡張あり）")
    print("  2. Best Model（HOG + データ拡張なし）")
    print("  3. MLP（Deep Neural Network）")
    print()

    # データ読み込み（全サンプル）
    target_classes = [
        '3a2c', '3a2e', '3a4e', '3a6e', '3a51',
        '3a60', '3a64', '3a72', '3b4f', '3b6b',
        '4f40', '3c23', '3d26', '3d3b', '3d3e'
    ]

    images, labels, class_info = load_etl8b_data(
        "./ETL8B-img-full", target_classes, max_samples=None  # 全サンプル
    )

    # 訓練・テスト分割
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"\n訓練データ: {len(train_images)}サンプル")
    print(f"テストデータ: {len(test_images)}サンプル")

    results = []

    # 1. Best Model（データ拡張あり）
    print("\n" + "="*100)
    print("1. Best Model（データ拡張あり）")
    print("="*100)

    model1 = BestModelWithAugmentation()

    start_time = time.time()
    model1.fit(train_images, train_labels)
    train_time1 = time.time() - start_time

    start_time = time.time()
    predictions1 = model1.predict(test_images)
    eval_time1 = time.time() - start_time

    accuracy1 = accuracy_score(test_labels, predictions1) * 100
    cm1 = confusion_matrix(test_labels, predictions1)

    print(f"\n認識率: {accuracy1:.2f}%")
    print(f"学習時間: {train_time1:.2f}秒")
    print(f"評価時間: {eval_time1:.3f}秒")

    results.append({
        'method': 'Best Model (With Augmentation)',
        'accuracy': accuracy1,
        'train_time': train_time1,
        'eval_time': eval_time1,
        'predictions': predictions1.tolist(),
        'confusion_matrix': cm1.tolist()
    })

    # 2. Best Model（データ拡張なし）
    print("\n" + "="*100)
    print("2. Best Model（データ拡張なし）")
    print("="*100)

    model2 = BestModelWithoutAugmentation()

    start_time = time.time()
    model2.fit(train_images, train_labels)
    train_time2 = time.time() - start_time

    start_time = time.time()
    predictions2 = model2.predict(test_images)
    eval_time2 = time.time() - start_time

    accuracy2 = accuracy_score(test_labels, predictions2) * 100
    cm2 = confusion_matrix(test_labels, predictions2)

    print(f"\n認識率: {accuracy2:.2f}%")
    print(f"学習時間: {train_time2:.2f}秒")
    print(f"評価時間: {eval_time2:.3f}秒")

    results.append({
        'method': 'Best Model (No Augmentation)',
        'accuracy': accuracy2,
        'train_time': train_time2,
        'eval_time': eval_time2,
        'predictions': predictions2.tolist(),
        'confusion_matrix': cm2.tolist()
    })

    # 3. MLP
    print("\n" + "="*100)
    print("3. MLP（Deep Neural Network）")
    print("="*100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    model3 = MLPWrapper(num_classes=15)

    start_time = time.time()
    model3.fit(train_images, train_labels, epochs=100, batch_size=32, lr=0.001)
    train_time3 = time.time() - start_time

    start_time = time.time()
    predictions3 = model3.predict(test_images)
    eval_time3 = time.time() - start_time

    accuracy3 = accuracy_score(test_labels, predictions3) * 100
    cm3 = confusion_matrix(test_labels, predictions3)

    print(f"\n認識率: {accuracy3:.2f}%")
    print(f"学習時間: {train_time3:.2f}秒")
    print(f"評価時間: {eval_time3:.3f}秒")

    results.append({
        'method': 'MLP (Deep Neural Network)',
        'accuracy': accuracy3,
        'train_time': train_time3,
        'eval_time': eval_time3,
        'predictions': predictions3.tolist(),
        'confusion_matrix': cm3.tolist(),
        'train_losses': model3.train_losses,
        'train_accuracies': model3.train_accuracies
    })

    # 結果まとめ
    print("\n" + "="*100)
    print(" 実験結果まとめ")
    print("="*100)
    print(f"\n{'モデル':<40} {'認識率':>10} {'学習時間':>12} {'評価時間':>12}")
    print("-"*100)

    for result in results:
        print(f"{result['method']:<40} {result['accuracy']:>9.2f}% {result['train_time']:>10.2f}秒 {result['eval_time']:>10.3f}秒")

    # 結果保存
    output_dir = 'results/full_dataset'
    os.makedirs(output_dir, exist_ok=True)

    # JSON保存
    with open(os.path.join(output_dir, 'full_dataset_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 可視化
    plot_comparison_results(results, output_dir)

    # レポート生成
    generate_report(results, output_dir, class_info)

    print("\n" + "="*100)
    print(" 実験完了")
    print("="*100)
    print(f"\n結果は {output_dir}/ に保存されました")


if __name__ == '__main__':
    main()
