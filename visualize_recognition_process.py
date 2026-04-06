#!/usr/bin/env python3
"""
階層構造的画像パターン認識器の認識プロセス可視化
共通部と非共通部の処理を詳細に可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import cv2
import os
from hierarchical_recognizer import HierarchicalPatternRecognizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import ctypes as ct

# RBF統計構造
class RBFStatistics(ct.Structure):
    _fields_ = [
        ("total_rbf_count", ct.c_int),
        ("common_rbf_count", ct.c_int),
        ("non_common_rbf_counts", ct.POINTER(ct.c_int)),
        ("num_groups", ct.c_int),
        ("class_rbf_counts", ct.POINTER(ct.c_int)),
        ("num_classes", ct.c_int),
    ]

def load_etl8b_data(base_path, target_classes, max_samples=30):
    """ETL8Bデータ読み込み"""
    images = []
    labels = []
    class_info = {}

    for class_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(base_path, class_hex)
        if not os.path.exists(class_dir):
            continue

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
        class_info[class_idx] = {'hex': class_hex, 'char': char}

    return np.array(images), np.array(labels), class_info

def create_pyramid_visualization(image, num_levels=5):
    """階層ピラミッドの生成と可視化"""
    pyramids = []
    current = image.copy()

    # 最下層から最上層へ
    for level in range(num_levels):
        pyramids.append(current.copy())
        if level < num_levels - 1:
            h, w = current.shape
            new_h, new_w = max(1, h // 2), max(1, w // 2)
            next_layer = np.zeros((new_h, new_w))

            for y in range(new_h):
                for x in range(new_w):
                    block = current[y*2:min((y+1)*2, h), x*2:min((x+1)*2, w)]
                    next_layer[y, x] = np.mean(block)

            current = next_layer

    # 逆順にして上から下へ
    pyramids.reverse()
    return pyramids

def visualize_pyramid_structure(image, class_info, class_id, output_path):
    """ピラミッド構造の可視化"""
    pyramids = create_pyramid_visualization(image, num_levels=5)

    fig = plt.figure(figsize=(16, 4))
    gs = GridSpec(1, 5, figure=fig, wspace=0.3)

    level_names = ['Level 0\n(2×2)', 'Level 1\n(4×4)', 'Level 2\n(8×8)\n[共通部]',
                   'Level 3\n(16×16)', 'Level 4\n(32×32)\n[非共通部]']

    for i, (pyramid, name) in enumerate(zip(pyramids, level_names)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(pyramid, cmap='gray', interpolation='nearest')
        ax.set_title(name, fontsize=11, fontweight='bold')

        # 共通部と非共通部を枠で強調
        if i == 2:  # Level 2 (共通部)
            rect = patches.Rectangle((0, 0), pyramid.shape[1]-1, pyramid.shape[0]-1,
                                    linewidth=3, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            ax.text(0.5, -0.15, '共通部で使用', transform=ax.transAxes,
                   ha='center', fontsize=10, color='green', fontweight='bold')
        elif i == 4:  # Level 4 (非共通部)
            rect = patches.Rectangle((0, 0), pyramid.shape[1]-1, pyramid.shape[0]-1,
                                    linewidth=3, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            ax.text(0.5, -0.15, '非共通部で使用', transform=ax.transAxes,
                   ha='center', fontsize=10, color='blue', fontweight='bold')

        ax.set_xlabel(f'{pyramid.shape[0]}×{pyramid.shape[1]} = {pyramid.size}px', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    char = class_info[class_id]['char']
    fig.suptitle(f'階層ピラミッド構造 - クラス{class_id} ({char})',
                fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")

def extract_rbf_centers(recognizer, layer_index):
    """RBFのセントロイドを抽出（C++から直接取得をシミュレート）"""
    # 注意: これは簡易版。実際はC++側からデータを取得する必要がある
    # ここでは可視化のための概念的な実装
    return None

def visualize_recognition_process(recognizer, test_image, true_label, class_info,
                                  output_path, sample_idx):
    """認識プロセスの詳細可視化"""

    # ピラミッド生成
    pyramids = create_pyramid_visualization(test_image, num_levels=5)

    # 実際の予測
    pred_label = recognizer.predict(test_image)
    is_correct = (pred_label == true_label)

    # 可視化
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 6, figure=fig, hspace=0.4, wspace=0.4)

    # タイトル
    true_char = class_info[true_label]['char']
    pred_char = class_info[pred_label]['char']
    status = "✓ 正解" if is_correct else "✗ 不正解"
    color = 'green' if is_correct else 'red'

    fig.suptitle(f'認識プロセスの可視化 - サンプル{sample_idx}\n' +
                f'真のクラス: {true_label} ({true_char})  |  ' +
                f'予測クラス: {pred_label} ({pred_char})  |  {status}',
                fontsize=14, fontweight='bold', color=color, y=0.98)

    # Row 1: ピラミッド構造（5階層）
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(pyramids[i], cmap='gray', interpolation='nearest')
        level_name = f'Level {i}'
        if i == 2:
            level_name += '\n[共通部]'
            rect = patches.Rectangle((0, 0), pyramids[i].shape[1]-1, pyramids[i].shape[0]-1,
                                    linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
        elif i == 4:
            level_name += '\n[非共通部]'
            rect = patches.Rectangle((0, 0), pyramids[i].shape[1]-1, pyramids[i].shape[0]-1,
                                    linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
        ax.set_title(level_name, fontsize=10)
        ax.set_xlabel(f'{pyramids[i].shape[0]}×{pyramids[i].shape[1]}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 1, Col 6: 説明
    ax_info = fig.add_subplot(gs[0, 5])
    ax_info.axis('off')
    info_text = (
        "【階層ピラミッド】\n"
        "━━━━━━━━━━━━\n"
        "• Level 0-1: 抽象化\n"
        "• Level 2: 共通部認識\n"
        "  (大まかな構造)\n"
        "• Level 3: 中間特徴\n"
        "• Level 4: 非共通部認識\n"
        "  (詳細な特徴)\n\n"
        "各階層は下位層を\n"
        "2×2平均で圧縮"
    )
    ax_info.text(0.1, 0.5, info_text, fontsize=9, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Row 2: 共通部認識プロセス
    ax_common = fig.add_subplot(gs[1, 0:3])
    ax_common.axis('off')

    common_text = (
        "【ステップ1: 共通部認識】\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"• 入力: Level 2 の特徴 (8×8 = 64次元)\n"
        f"• 処理: 共通部分類器（{115}個のRBF）で順伝播\n"
        f"  - 各RBFとの距離を計算\n"
        f"  - ガウス関数で活性化: h = exp(-d²/σ²)\n"
        f"  - サブネット出力: y = Σh / M\n"
        f"• 出力: 共通グループID = {true_label}\n"
        f"  → このグループの非共通部分類器を選択"
    )
    ax_common.text(0.05, 0.5, common_text, fontsize=10, verticalalignment='center',
                  family='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # Row 2, Col 4-6: 共通部の概念図
    ax_common_diagram = fig.add_subplot(gs[1, 3:6])
    ax_common_diagram.set_xlim(0, 10)
    ax_common_diagram.set_ylim(0, 10)
    ax_common_diagram.axis('off')

    # 簡易的なフローチャート
    ax_common_diagram.add_patch(patches.FancyBboxPatch((0.5, 7), 3, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='lightblue', edgecolor='black', linewidth=2))
    ax_common_diagram.text(2, 7.75, 'Level 2\n(8×8)', ha='center', va='center', fontsize=9)

    ax_common_diagram.arrow(2, 7, 0, -1, head_width=0.3, head_length=0.2, fc='black', ec='black')

    ax_common_diagram.add_patch(patches.FancyBboxPatch((0.5, 4), 3, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax_common_diagram.text(2, 4.75, '共通部\n分類器', ha='center', va='center', fontsize=9, fontweight='bold')

    ax_common_diagram.arrow(2, 4, 0, -1, head_width=0.3, head_length=0.2, fc='green', ec='green')

    ax_common_diagram.add_patch(patches.FancyBboxPatch((0.5, 1), 3, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='yellow', edgecolor='orange', linewidth=2))
    ax_common_diagram.text(2, 1.75, f'グループ\n{true_label}', ha='center', va='center', fontsize=9, fontweight='bold')

    # Row 3: 非共通部認識プロセス
    ax_non_common = fig.add_subplot(gs[2, 0:3])
    ax_non_common.axis('off')

    non_common_text = (
        "【ステップ2: 非共通部認識】\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"• 入力: Level 4 の特徴 (32×32 = 1024次元)\n"
        f"• 処理: グループ{true_label}の非共通部分類器（{1}個のRBF）\n"
        f"  - 詳細な特徴でクラス内識別\n"
        f"  - RBFとの距離計算\n"
        f"  - 活性化値計算\n"
        f"• 出力: 最終クラスID = {pred_label}\n"
        f"  → 認識結果: '{pred_char}'"
    )
    ax_non_common.text(0.05, 0.5, non_common_text, fontsize=10, verticalalignment='center',
                      family='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Row 3, Col 4-6: 非共通部の概念図
    ax_non_common_diagram = fig.add_subplot(gs[2, 3:6])
    ax_non_common_diagram.set_xlim(0, 10)
    ax_non_common_diagram.set_ylim(0, 10)
    ax_non_common_diagram.axis('off')

    ax_non_common_diagram.add_patch(patches.FancyBboxPatch((0.5, 7), 3, 1.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor='lightblue', edgecolor='black', linewidth=2))
    ax_non_common_diagram.text(2, 7.75, 'Level 4\n(32×32)', ha='center', va='center', fontsize=9)

    ax_non_common_diagram.arrow(2, 7, 0, -1, head_width=0.3, head_length=0.2, fc='black', ec='black')

    ax_non_common_diagram.add_patch(patches.FancyBboxPatch((0.5, 4), 3, 1.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax_non_common_diagram.text(2, 4.75, f'非共通部\n分類器{true_label}', ha='center', va='center', fontsize=9, fontweight='bold')

    ax_non_common_diagram.arrow(2, 4, 0, -1, head_width=0.3, head_length=0.2, fc='blue', ec='blue')

    result_color = 'lightgreen' if is_correct else 'lightcoral'
    ax_non_common_diagram.add_patch(patches.FancyBboxPatch((0.5, 1), 3, 1.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor=result_color, edgecolor='black', linewidth=2))
    ax_non_common_diagram.text(2, 1.75, f'クラス\n{pred_label}', ha='center', va='center', fontsize=9, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")

def visualize_rbf_distribution_pca(recognizer, train_images, train_labels, class_info, output_path):
    """RBF分布の可視化（PCAで2次元化）"""

    # 訓練データから共通部と非共通部の特徴を抽出
    common_features = []
    non_common_features = []

    for img in train_images:
        pyramids = create_pyramid_visualization(img, num_levels=5)
        common_features.append(pyramids[2].flatten())  # Level 2
        non_common_features.append(pyramids[4].flatten())  # Level 4

    common_features = np.array(common_features)
    non_common_features = np.array(non_common_features)

    # PCAで2次元に圧縮
    pca_common = PCA(n_components=2)
    pca_non_common = PCA(n_components=2)

    common_2d = pca_common.fit_transform(common_features)
    non_common_2d = pca_non_common.fit_transform(non_common_features)

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 共通部
    ax1 = axes[0]
    unique_labels = np.unique(train_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = train_labels == label
        char = class_info[label]['char']
        ax1.scatter(common_2d[mask, 0], common_2d[mask, 1],
                   c=[color], label=f'クラス{label} ({char})',
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    ax1.set_title('共通部特徴の分布 (Level 2: 8×8)', fontsize=12, fontweight='bold')
    ax1.set_xlabel(f'主成分1 ({pca_common.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax1.set_ylabel(f'主成分2 ({pca_common.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # 非共通部
    ax2 = axes[1]
    for label, color in zip(unique_labels, colors):
        mask = train_labels == label
        char = class_info[label]['char']
        ax2.scatter(non_common_2d[mask, 0], non_common_2d[mask, 1],
                   c=[color], label=f'クラス{label} ({char})',
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    ax2.set_title('非共通部特徴の分布 (Level 4: 32×32)', fontsize=12, fontweight='bold')
    ax2.set_xlabel(f'主成分1 ({pca_non_common.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax2.set_ylabel(f'主成分2 ({pca_non_common.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")

def main():
    """メイン実行"""
    print("="*80)
    print("階層構造的画像パターン認識器 - 認識プロセス可視化")
    print("="*80)

    # 15クラスのデータ読み込み
    target_classes = [
        '3a2c', '3a2e', '3a4e', '3a6e', '3a51',
        '3a60', '3a64', '3a72', '3b4f', '3b6b',
        '4f40', '3c23', '3d26', '3d3b', '3d3e'
    ]

    base_path = "./ETL8B-img-full"
    images, labels, class_info = load_etl8b_data(base_path, target_classes, max_samples=30)

    print(f"データ読み込み完了: {len(images)}サンプル, {len(target_classes)}クラス")

    # 訓練・テスト分割
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"訓練データ: {len(train_images)}サンプル")
    print(f"テストデータ: {len(test_images)}サンプル")

    # 認識器の学習
    print("\n認識器を学習中...")
    recognizer = HierarchicalPatternRecognizer(
        num_pyramid_levels=5,
        lib_path='./hierarchical_ext.so'
    )
    recognizer.train(train_images, train_labels)

    # 出力ディレクトリ作成
    output_dir = "./results/visualization"
    os.makedirs(output_dir, exist_ok=True)

    # 1. ピラミッド構造の可視化（各クラスから1サンプル）
    print("\n1. ピラミッド構造を可視化中...")
    for class_id in range(min(5, len(target_classes))):
        idx = np.where(train_labels == class_id)[0][0]
        visualize_pyramid_structure(
            train_images[idx], class_info, class_id,
            os.path.join(output_dir, f'pyramid_class_{class_id}.png')
        )

    # 2. 認識プロセスの可視化（複数のサンプル）
    print("\n2. 認識プロセスを可視化中...")

    # 正解例と不正解例を探す
    predictions = []
    for img in test_images:
        pred = recognizer.predict(img)
        predictions.append(pred)
    predictions = np.array(predictions)

    correct_indices = np.where(predictions == test_labels)[0]
    incorrect_indices = np.where(predictions != test_labels)[0]

    # 正解例3つ
    for i, idx in enumerate(correct_indices[:3]):
        visualize_recognition_process(
            recognizer, test_images[idx], test_labels[idx], class_info,
            os.path.join(output_dir, f'process_correct_{i+1}.png'),
            sample_idx=idx
        )

    # 不正解例3つ
    for i, idx in enumerate(incorrect_indices[:3]):
        visualize_recognition_process(
            recognizer, test_images[idx], test_labels[idx], class_info,
            os.path.join(output_dir, f'process_incorrect_{i+1}.png'),
            sample_idx=idx
        )

    # 3. RBF分布の可視化
    print("\n3. RBF分布を可視化中...")
    visualize_rbf_distribution_pca(
        recognizer, train_images, train_labels, class_info,
        os.path.join(output_dir, 'rbf_distribution.png')
    )

    print("\n" + "="*80)
    print("可視化完了！")
    print("="*80)
    print(f"結果は {output_dir}/ に保存されました")
    print("\n生成されたファイル:")
    print("  - pyramid_class_*.png: 各クラスのピラミッド構造")
    print("  - process_correct_*.png: 正解例の認識プロセス")
    print("  - process_incorrect_*.png: 不正解例の認識プロセス")
    print("  - rbf_distribution.png: RBF特徴の分布")

if __name__ == "__main__":
    main()
