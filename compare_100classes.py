# -*- coding: utf-8 -*-
"""
100クラス比較実験: ベストモデル(HOG+PNN) vs MLP

左右分割漢字100クラス（にんべん42、さんずい35、ごんべん14、他9）で
ベストモデル（データ拡張なし）とMLPを比較する。

使い方:
    python3 compare_100classes.py
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

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 100クラスのJISコードリスト
TARGET_CLASSES_100 = [
    '304c', '304d', '314b', '3155', '3169',
    '322f', '3239', '323d', '323e', '323f',
    '3241', '324f', '325d', '3324', '3368',
    '3441', '3525', '352d', '3544', '3559',
    '3576', '3579', '3621', '3731', '3738',
    '3757', '3768', '3769', '376f', '3772',
    '383a', '3844', '3850', '386c', '386d',
    '386e', '3875', '3941', '3956', '3a2c',
    '3a2e', '3a4e', '3a51', '3a60', '3a64',
    '3a6e', '3a72', '3b45', '3b48', '3b4f',
    '3b6b', '3b6c', '3b6d', '3b6e', '3b77',
    '3c23', '3c5a', '3c72', '3d24', '3d26',
    '3d3b', '3d3e', '3e43', '3f2e', '3f3c',
    '3f4e', '4036', '4075', '417c', '4226',
    '422c', '422f', '423e', '424e', '4265',
    '4353', '436d', '4463', '4464', '4541',
    '4572', '462f', '4724', '4748', '4749',
    '475c', '4877', '4936', '4955', '4a29',
    '4a58', '4a5d', '4b21', '4b7e', '4c7d',
    '4d4e', '4d61', '4e2e', '4e63', '4f40',
]


def jis_to_unicode(jis_hex):
    """JISコードをUnicode文字に変換"""
    try:
        jis_code = int(jis_hex, 16)
        j1 = (jis_code >> 8) & 0xFF
        j2 = jis_code & 0xFF
        if j1 % 2 == 0:
            s1 = ((j1 - 0x21) >> 1) + 0x81
            if s1 > 0x9f:
                s1 += 0x40
            s2 = j2 + 0x7e
        else:
            s1 = ((j1 - 0x21) >> 1) + 0x81
            if s1 > 0x9f:
                s1 += 0x40
            if j2 < 0x60:
                s2 = j2 + 0x1f
            else:
                s2 = j2 + 0x20
        return bytes([s1, s2]).decode('shift_jis')
    except Exception:
        return '?'


def load_etl8b_data(data_dir, target_classes, max_samples=None):
    """ETL8Bデータを読み込む"""
    images = []
    labels = []
    class_info = {}

    print(f"\n{'='*80}")
    print(f"ETL8Bデータ読み込み ({len(target_classes)}クラス)")
    print(f"{'='*80}")

    for label_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(data_dir, class_hex)
        if not os.path.exists(class_dir):
            print(f"警告: {class_dir} が見つかりません")
            continue

        image_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
        if max_samples is not None:
            image_files = image_files[:max_samples]

        class_images = []
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                class_images.append(img)

        images.extend(class_images)
        labels.extend([label_idx] * len(class_images))

        char = jis_to_unicode(class_hex)
        class_info[label_idx] = {
            'hex': class_hex,
            'char': char,
            'count': len(class_images)
        }

        if (label_idx + 1) % 10 == 0 or label_idx == 0:
            print(f"  クラス {label_idx:3d} ({class_hex} {char}): {len(class_images):4d} サンプル")

    images = np.array(images)
    labels = np.array(labels)

    print(f"  ...")
    print(f"  合計: {len(images)} サンプル、{len(target_classes)} クラス")
    print(f"{'='*80}\n")

    return images, labels, class_info


class BestModel:
    """ベストモデル（HOG+PNN+左右分割+多数決投票、データ拡張なし）"""

    def __init__(self):
        from kanji_best_standalone import KanjiBestRecognizer
        self.recognizer = KanjiBestRecognizer()
        self.recognizer.augmentation_factor = 1

    def fit(self, X, y):
        self.recognizer.train(X, y)

    def predict(self, X):
        if len(X.shape) == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
        predictions = []
        total = len(X)
        for i, img in enumerate(X):
            pred = self.recognizer.predict(img)
            predictions.append(pred)
            if (i + 1) % 500 == 0:
                print(f"    予測中: {i+1}/{total}")
        return np.array(predictions)


class MLP(nn.Module):
    """多層パーセプトロン"""

    def __init__(self, input_size=4096, hidden_sizes=None, num_classes=100):
        super(MLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [1024, 512, 256]

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

    def __init__(self, num_classes=100):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_losses = []
        self.train_accuracies = []

    def fit(self, X, y, epochs=100, batch_size=64, lr=0.001):
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        X_tensor = torch.FloatTensor(X_flat)
        y_tensor = torch.LongTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = MLP(input_size=X_flat.shape[1], num_classes=self.num_classes).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

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

            scheduler.step()

            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100.0 * correct / total
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def predict(self, X):
        self.model.eval()
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        X_tensor = torch.FloatTensor(X_flat).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()


def plot_comparison_results(results, output_dir, num_classes, class_info):
    """比較結果の可視化"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 認識率比較
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = [r['method'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax.bar(range(len(methods)), accuracies, color=colors[:len(methods)], alpha=0.8)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Recognition Accuracy Comparison ({num_classes} classes)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison_100.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 処理時間比較
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    train_times = [r['train_time'] for r in results]
    eval_times = [r['eval_time'] for r in results]

    bars1 = ax1.bar(range(len(methods)), train_times, color=colors[:len(methods)], alpha=0.8)
    for bar, t in zip(bars1, train_times):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{t:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Training Time', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    bars2 = ax2.bar(range(len(methods)), eval_times, color=colors[:len(methods)], alpha=0.8)
    for bar, t in zip(bars2, eval_times):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{t:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Evaluation Time', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison_100.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 混同行列（100クラスなので数値は表示せずヒートマップのみ）
    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 7))
    if len(results) == 1:
        axes = [axes]

    for idx, (result, ax) in enumerate(zip(results, axes)):
        cm = np.array(result['confusion_matrix'])
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f"{result['method']}\n({result['accuracy']:.2f}%)", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_100.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. MLP学習曲線
    mlp_results = [r for r in results if 'train_losses' in r]
    if mlp_results:
        mlp_result = mlp_results[0]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(mlp_result['train_losses']) + 1)

        ax1.plot(epochs, mlp_result['train_losses'], 'b-', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'MLP Training Loss ({num_classes} classes)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, mlp_result['train_accuracies'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title(f'MLP Training Accuracy ({num_classes} classes)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mlp_training_curve_100.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 5. 偏グループ別精度（ベストモデル）
    best_result = [r for r in results if 'Best' in r['method']][0]
    plot_radical_accuracy(best_result, class_info, output_dir)

    print(f"\n可視化結果を {output_dir}/ に保存しました")


def plot_radical_accuracy(result, class_info, output_dir):
    """偏グループ別の精度を可視化"""
    # 偏グループの定義
    radical_groups = {
        'にんべん(亻)': [],
        'さんずい(氵)': [],
        'ごんべん(言)': [],
        'その他': [],
    }

    # JISコードから偏グループへのマッピング（100クラスリストの順序）
    ninben = {
        '304c', '304d', '322f', '323d', '323e', '323f', '3241', '3559',
        '3621', '3738', '376f', '3772', '3844', '3875', '3a6e', '3b45',
        '3b48', '3b77', '3c5a', '3d24', '3d3b', '3f2e', '3f4e', '417c',
        '4226', '422f', '423e', '424e', '4265', '4463', '4464', '4541',
        '462f', '4724', '475c', '4877', '4936', '4955', '4a29', '4a58',
        '4a5d', '4e63',
    }
    sanzui = {
        '314b', '3155', '3169', '3239', '324f', '3324', '3368', '3441',
        '3525', '3579', '3768', '3769', '383a', '3850', '3941', '3a2e',
        '3a51', '3c23', '3c72', '3e43', '3f3c', '4036', '4075', '422c',
        '4353', '436d', '4572', '4748', '4749', '4b21', '4b7e', '4c7d',
        '4d4e', '4d61', '4e2e',
    }
    gonben = {
        '325d', '352d', '3544', '3576', '3731', '3757', '386c', '386d',
        '386e', '3956', '3b6c', '3b6d', '3b6e', '4f40',
    }

    cm = np.array(result['confusion_matrix'])
    predictions = np.array(result['predictions'])

    for label_idx, info in class_info.items():
        jis = info['hex']
        # そのクラスの正解数/合計を計算
        if label_idx < cm.shape[0]:
            correct = cm[label_idx, label_idx]
            total = cm[label_idx].sum()
            acc = 100.0 * correct / total if total > 0 else 0

            if jis in ninben:
                radical_groups['にんべん(亻)'].append(acc)
            elif jis in sanzui:
                radical_groups['さんずい(氵)'].append(acc)
            elif jis in gonben:
                radical_groups['ごんべん(言)'].append(acc)
            else:
                radical_groups['その他'].append(acc)

    # 棒グラフ
    fig, ax = plt.subplots(figsize=(10, 6))
    group_names = []
    group_means = []
    group_counts = []

    for name, accs in radical_groups.items():
        if accs:
            group_names.append(name)
            group_means.append(np.mean(accs))
            group_counts.append(len(accs))

    colors = ['#3498db', '#2ecc71', '#e67e22', '#95a5a6']
    bars = ax.bar(range(len(group_names)), group_means, color=colors[:len(group_names)], alpha=0.8)

    for bar, mean, count in zip(bars, group_means, group_counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{mean:.1f}%\n({count} cls)', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax.set_title('Best Model: Accuracy by Radical Group', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names, fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radical_accuracy_100.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(results, output_dir, class_info, num_classes):
    """レポート生成"""
    report_path = os.path.join(output_dir, 'comparison_100_report.txt')

    total_samples = sum([info['count'] for info in class_info.values()])
    train_samples = int(total_samples * 0.7)
    test_samples = total_samples - train_samples

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f" 100クラス比較実験レポート: ベストモデル vs MLP\n")
        f.write("=" * 100 + "\n\n")

        f.write("1. 実験設定\n")
        f.write("-" * 100 + "\n")
        f.write(f"  対象クラス数: {num_classes}\n")
        f.write(f"  画像サイズ: 64x64\n")
        f.write(f"  訓練データ: {train_samples}サンプル (70%)\n")
        f.write(f"  テストデータ: {test_samples}サンプル (30%)\n")
        f.write(f"  合計: {total_samples}サンプル\n")
        f.write(f"  データ分割: stratified, random_state=42\n\n")

        f.write("  選定基準: 左右分割構造（IDS ⿰判定）の漢字\n")
        f.write("  偏グループ構成:\n")
        f.write("    - にんべん(亻): 42クラス\n")
        f.write("    - さんずい(氵): 35クラス\n")
        f.write("    - ごんべん(言): 14クラス\n")
        f.write("    - その他(木,扌,土,日,女,礻,彳): 9クラス\n\n")

        f.write("2. モデル詳細\n")
        f.write("-" * 100 + "\n")
        f.write("  ベストモデル (HOG+PNN):\n")
        f.write("    - HOG特徴抽出 (9方向, 8x8セル)\n")
        f.write("    - 階層ピラミッド: 5段\n")
        f.write("    - 左右分割 + 多数決投票 (全体・偏・旁)\n")
        f.write("    - PNN分類器 (適応的σ, K-means代表点)\n")
        f.write("    - データ拡張: なし\n\n")
        f.write("  MLP (Deep Neural Network):\n")
        f.write("    - 入力: 64x64=4096次元\n")
        f.write("    - 隠れ層: [1024, 512, 256]\n")
        f.write("    - 活性化関数: ReLU\n")
        f.write("    - 正則化: Dropout(0.3) + BatchNorm + L2\n")
        f.write("    - 学習率スケジューラ: StepLR(step=30, gamma=0.5)\n")
        f.write("    - エポック数: 100, バッチサイズ: 64\n\n")

        f.write("3. 認識率比較\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'モデル':<50} {'認識率':<20}\n")
        f.write("-" * 100 + "\n")

        best_acc = max([r['accuracy'] for r in results])
        for result in results:
            marker = " ★最高精度" if result['accuracy'] == best_acc else ""
            f.write(f"  {result['method']:<48} {result['accuracy']:>6.2f}%{marker}\n")

        # 差分
        best_result = [r for r in results if 'Best' in r['method']][0]
        mlp_result = [r for r in results if 'MLP' in r['method']][0]
        diff = best_result['accuracy'] - mlp_result['accuracy']
        f.write(f"\n  差分 (Best - MLP): {diff:+.2f}%\n\n")

        f.write("4. 処理時間比較\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'モデル':<50} {'学習時間':<20} {'評価時間':<20}\n")
        f.write("-" * 100 + "\n")
        for result in results:
            f.write(f"  {result['method']:<48} {result['train_time']:>10.2f}秒      {result['eval_time']:>10.2f}秒\n")

        f.write(f"\n5. 15クラス実験との比較\n")
        f.write("-" * 100 + "\n")
        f.write("  15クラス実験結果（参考）:\n")
        f.write("    - Best Model（拡張なし）: 97.38% (2415サンプル)\n")
        f.write("    - MLP:                    92.28% (2415サンプル)\n\n")
        f.write(f"  100クラス実験結果:\n")
        f.write(f"    - Best Model（拡張なし）: {best_result['accuracy']:.2f}% ({total_samples}サンプル)\n")
        f.write(f"    - MLP:                    {mlp_result['accuracy']:.2f}% ({total_samples}サンプル)\n\n")

        f.write("6. 生成ファイル\n")
        f.write("-" * 100 + "\n")
        f.write("  - accuracy_comparison_100.png: 認識率比較\n")
        f.write("  - time_comparison_100.png: 処理時間比較\n")
        f.write("  - confusion_matrices_100.png: 混同行列比較\n")
        f.write("  - mlp_training_curve_100.png: MLP学習曲線\n")
        f.write("  - radical_accuracy_100.png: 偏グループ別精度\n")
        f.write("  - comparison_100_results.json: 詳細結果(JSON)\n")
        f.write("  - comparison_100_report.txt: このレポート\n\n")

    print(f"\nレポートを {report_path} に保存しました")


def main():
    print("\n" + "=" * 100)
    print(" 100クラス比較実験: ベストモデル(HOG+PNN) vs MLP")
    print("=" * 100)

    num_classes = len(TARGET_CLASSES_100)
    print(f"\n対象クラス数: {num_classes}")
    print(f"比較内容:")
    print(f"  1. Best Model (HOG+PNN+左右分割+多数決投票)")
    print(f"  2. MLP (Deep Neural Network)")

    # データ読み込み（全サンプル）
    images, labels, class_info = load_etl8b_data(
        "./ETL8B-img-full", TARGET_CLASSES_100, max_samples=None
    )

    # 訓練・テスト分割
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"\n訓練データ: {len(train_images)}サンプル")
    print(f"テストデータ: {len(test_images)}サンプル")

    results = []

    # 1. ベストモデル（データ拡張なし）
    print("\n" + "=" * 100)
    print("1. Best Model (HOG+PNN、データ拡張なし)")
    print("=" * 100)

    model1 = BestModel()

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
    print(f"評価時間: {eval_time1:.2f}秒")

    results.append({
        'method': 'Best Model (HOG+PNN)',
        'accuracy': accuracy1,
        'train_time': train_time1,
        'eval_time': eval_time1,
        'predictions': predictions1.tolist(),
        'confusion_matrix': cm1.tolist()
    })

    # 2. MLP
    print("\n" + "=" * 100)
    print("2. MLP (Deep Neural Network)")
    print("=" * 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    model2 = MLPWrapper(num_classes=num_classes)

    start_time = time.time()
    model2.fit(train_images, train_labels, epochs=100, batch_size=64, lr=0.001)
    train_time2 = time.time() - start_time

    start_time = time.time()
    predictions2 = model2.predict(test_images)
    eval_time2 = time.time() - start_time

    accuracy2 = accuracy_score(test_labels, predictions2) * 100
    cm2 = confusion_matrix(test_labels, predictions2)

    print(f"\n認識率: {accuracy2:.2f}%")
    print(f"学習時間: {train_time2:.2f}秒")
    print(f"評価時間: {eval_time2:.2f}秒")

    results.append({
        'method': 'MLP (DNN)',
        'accuracy': accuracy2,
        'train_time': train_time2,
        'eval_time': eval_time2,
        'predictions': predictions2.tolist(),
        'confusion_matrix': cm2.tolist(),
        'train_losses': model2.train_losses,
        'train_accuracies': model2.train_accuracies
    })

    # 結果まとめ
    print("\n" + "=" * 100)
    print(" 実験結果まとめ")
    print("=" * 100)
    print(f"\n{'モデル':<40} {'認識率':>10} {'学習時間':>12} {'評価時間':>12}")
    print("-" * 80)

    for result in results:
        print(f"{result['method']:<40} {result['accuracy']:>9.2f}% {result['train_time']:>10.2f}秒 {result['eval_time']:>10.2f}秒")

    diff = results[0]['accuracy'] - results[1]['accuracy']
    print(f"\n差分 (Best Model - MLP): {diff:+.2f}%")

    # 結果保存
    output_dir = 'results/comparison_100'
    os.makedirs(output_dir, exist_ok=True)

    # JSON保存（predictions/confusion_matrixは大きいので別途）
    save_results = []
    for r in results:
        save_r = {k: v for k, v in r.items()}
        save_results.append(save_r)

    with open(os.path.join(output_dir, 'comparison_100_results.json'), 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    # 可視化
    plot_comparison_results(results, output_dir, num_classes, class_info)

    # レポート生成
    generate_report(results, output_dir, class_info, num_classes)

    print("\n" + "=" * 100)
    print(" 実験完了")
    print("=" * 100)
    print(f"\n結果は {output_dir}/ に保存されました")


if __name__ == '__main__':
    main()
