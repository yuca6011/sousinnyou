# -*- coding: utf-8 -*-
"""
左右分割（偏・旁分離）の可視化

垂直投影による分割点検出プロセスと結果を可視化します。
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']


def load_sample_images(data_dir, target_classes, num_samples=5):
    """サンプル画像を読み込む"""
    images = []
    class_labels = []

    for class_hex in target_classes[:num_samples]:
        class_dir = os.path.join(data_dir, class_hex)
        if not os.path.exists(class_dir):
            continue

        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        if len(image_files) > 0:
            img_path = os.path.join(class_dir, image_files[0])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                img = img.astype(np.float32) / 255.0
                images.append(img)
                class_labels.append(class_hex)

    return images, class_labels


def split_left_right_with_info(image):
    """
    左右分割を実施し、詳細情報を返す

    Returns:
        left_img, right_img, split_col, vertical_projection, v_smooth
    """
    # 垂直投影の計算
    vertical_projection = np.sum(image, axis=0)

    # 平滑化は行わない（hierarchical_recognizer_lr_improved.pyに合わせる）
    v_smooth = vertical_projection.copy()

    # 中央付近で最大値（最も白い列 = 空白部分）を検出
    width = image.shape[1]
    center = width // 2
    center_start = max(0, center - width // 4)
    center_end = min(width, center + width // 4)

    v_center = v_smooth[center_start:center_end]
    split_col = center_start + np.argmax(v_center)

    # 画像を分割
    left_img = image[:, :split_col]
    right_img = image[:, split_col:]

    # 正方形化
    def to_square(img):
        if img.size == 0:
            return np.zeros((8, 8), dtype=np.float32)

        h, w = img.shape
        size = max(h, w, 8)

        square = np.zeros((size, size), dtype=np.float32)

        y_offset = (size - h) // 2
        x_offset = (size - w) // 2

        square[y_offset:y_offset+h, x_offset:x_offset+w] = img

        return square

    left_square = to_square(left_img)
    right_square = to_square(right_img)

    return left_square, right_square, split_col, vertical_projection, v_smooth


def visualize_split(image, class_label, output_path):
    """単一画像の分割を可視化"""

    left_img, right_img, split_col, v_proj, v_smooth = split_left_right_with_info(image)

    fig = plt.figure(figsize=(16, 4))

    # 1. 元画像
    ax1 = plt.subplot(1, 5, 1)
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Original Image\nClass: {class_label}', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. 垂直投影グラフ
    ax2 = plt.subplot(1, 5, 2)
    x = np.arange(len(v_proj))
    ax2.plot(x, v_proj, 'b-', linewidth=2, label='Projection')
    ax2.axvline(x=split_col, color='g', linestyle='--', linewidth=2, label=f'Split at {split_col}')

    # 中央領域を塗りつぶし（center ± width/4）
    width = len(v_proj)
    center = width // 2
    center_start = max(0, center - width // 4)
    center_end = min(width, center + width // 4)
    ax2.axvspan(center_start, center_end, alpha=0.1, color='yellow', label='Search Area')

    ax2.set_xlabel('Column Position', fontsize=10)
    ax2.set_ylabel('Vertical Projection', fontsize=10)
    ax2.set_title('Vertical Projection\n& Split Detection', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. 分割点を示した画像
    ax3 = plt.subplot(1, 5, 3)
    img_with_line = np.stack([image]*3, axis=-1)
    img_with_line[:, split_col, :] = [0, 1, 0]  # 緑色の線
    if split_col > 0:
        img_with_line[:, split_col-1, :] = [0, 1, 0]
    if split_col < image.shape[1] - 1:
        img_with_line[:, split_col+1, :] = [0, 1, 0]

    ax3.imshow(img_with_line)
    ax3.set_title('Split Line\n(Green)', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 4. 左側画像（偏）
    ax4 = plt.subplot(1, 5, 4)
    ax4.imshow(left_img, cmap='gray')
    ax4.set_title(f'Left Part (Hen)\n{left_img.shape[0]}x{left_img.shape[1]}',
                  fontsize=12, fontweight='bold')
    ax4.axis('off')

    # 5. 右側画像（旁）
    ax5 = plt.subplot(1, 5, 5)
    ax5.imshow(right_img, cmap='gray')
    ax5.set_title(f'Right Part (Tsukuri)\n{right_img.shape[0]}x{right_img.shape[1]}',
                  fontsize=12, fontweight='bold')
    ax5.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"保存: {output_path}")


def visualize_all_samples(images, class_labels, output_dir):
    """全サンプルの分割を1枚の画像に可視化"""

    n_samples = len(images)

    fig = plt.figure(figsize=(20, 4 * n_samples))

    for idx, (image, class_label) in enumerate(zip(images, class_labels)):
        left_img, right_img, split_col, v_proj, v_smooth = split_left_right_with_info(image)

        # 元画像
        ax1 = plt.subplot(n_samples, 5, idx*5 + 1)
        ax1.imshow(image, cmap='gray')
        if idx == 0:
            ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'Class {class_label}', fontsize=11, fontweight='bold')
        ax1.axis('off')

        # 垂直投影
        ax2 = plt.subplot(n_samples, 5, idx*5 + 2)
        x = np.arange(len(v_proj))
        ax2.plot(x, v_proj, 'b-', linewidth=2)
        ax2.axvline(x=split_col, color='g', linestyle='--', linewidth=2)

        width = len(v_proj)
        center = width // 2
        center_start = max(0, center - width // 4)
        center_end = min(width, center + width // 4)
        ax2.axvspan(center_start, center_end, alpha=0.1, color='yellow')

        if idx == 0:
            ax2.set_title('Vertical Projection', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Column', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 分割線付き画像
        ax3 = plt.subplot(n_samples, 5, idx*5 + 3)
        img_with_line = np.stack([image]*3, axis=-1)
        img_with_line[:, split_col, :] = [0, 1, 0]
        if split_col > 0:
            img_with_line[:, split_col-1, :] = [0, 1, 0]
        if split_col < image.shape[1] - 1:
            img_with_line[:, split_col+1, :] = [0, 1, 0]

        ax3.imshow(img_with_line)
        if idx == 0:
            ax3.set_title('Split Line', fontsize=12, fontweight='bold')
        ax3.axis('off')

        # 左側
        ax4 = plt.subplot(n_samples, 5, idx*5 + 4)
        ax4.imshow(left_img, cmap='gray')
        if idx == 0:
            ax4.set_title('Left (Hen)', fontsize=12, fontweight='bold')
        ax4.axis('off')

        # 右側
        ax5 = plt.subplot(n_samples, 5, idx*5 + 5)
        ax5.imshow(right_img, cmap='gray')
        if idx == 0:
            ax5.set_title('Right (Tsukuri)', fontsize=12, fontweight='bold')
        ax5.axis('off')

    plt.suptitle('Left-Right Split Visualization (Projection Method)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'all_samples_split.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n全サンプル可視化を保存: {output_path}")


def create_explanation_figure(output_dir):
    """分割アルゴリズムの説明図を作成"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # サンプル画像を作成（簡単な例）
    sample = np.zeros((64, 64), dtype=np.float32)

    # 左側に「イ」のような形
    sample[20:50, 15:20] = 0.8
    sample[30:35, 20:28] = 0.8

    # 右側に「可」のような形
    sample[20:25, 40:55] = 0.8
    sample[30:35, 40:55] = 0.8
    sample[40:45, 40:55] = 0.8
    sample[25:40, 45:46] = 0.8

    # 1. 元画像
    axes[0, 0].imshow(sample, cmap='gray')
    axes[0, 0].set_title('Step 1: Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. 垂直投影計算
    v_proj = np.sum(sample, axis=0)
    axes[0, 1].plot(v_proj, 'b-', linewidth=2)
    axes[0, 1].set_title('Step 2: Calculate Vertical Projection\nV(x) = Σ I(x,y)',
                         fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Column Position (x)', fontsize=12)
    axes[0, 1].set_ylabel('Projection Sum', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 探索範囲の設定
    width = len(v_proj)
    center = width // 2
    center_start = max(0, center - width // 4)
    center_end = min(width, center + width // 4)

    axes[0, 2].plot(v_proj, 'b-', linewidth=2, label='Projection')
    axes[0, 2].axvspan(center_start, center_end, alpha=0.2, color='yellow',
                       label=f'Search Area\n(center ± W/4)')
    axes[0, 2].set_title('Step 3: Define Search Range\nCenter ± Width/4',
                         fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Column Position (x)', fontsize=12)
    axes[0, 2].set_ylabel('Projection Sum', fontsize=12)
    axes[0, 2].legend(fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 最大値検出（最も白い列 = 空白部分）
    v_center = v_proj[center_start:center_end]
    split_col = center_start + np.argmax(v_center)

    axes[1, 0].plot(v_proj, 'b-', linewidth=2, label='Projection')
    axes[1, 0].axvspan(center_start, center_end, alpha=0.2, color='yellow',
                       label='Search Area')
    axes[1, 0].axvline(x=split_col, color='g', linestyle='--', linewidth=3,
                       label=f'Split Point: {split_col}')
    axes[1, 0].scatter([split_col], [v_proj[split_col]], color='g', s=200,
                       marker='o', zorder=5)
    axes[1, 0].set_title('Step 4: Find Maximum (Whitest Column)\nx_split = argmax(V_center)',
                         fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Column Position (x)', fontsize=12)
    axes[1, 0].set_ylabel('Projection Sum', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 分割結果
    img_with_line = np.stack([sample]*3, axis=-1)
    img_with_line[:, split_col, :] = [0, 1, 0]
    if split_col > 0:
        img_with_line[:, split_col-1, :] = [0, 1, 0]
    if split_col < sample.shape[1] - 1:
        img_with_line[:, split_col+1, :] = [0, 1, 0]

    axes[1, 1].imshow(img_with_line)
    axes[1, 1].set_title('Step 5: Apply Split\nGreen Line Shows Division',
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # 6. 分割後の画像
    left_img = sample[:, :split_col]
    right_img = sample[:, split_col:]

    combined = np.zeros((64, 64*2 + 10), dtype=np.float32)
    combined[:, :left_img.shape[1]] = left_img
    combined[:, left_img.shape[1]+10:left_img.shape[1]+10+right_img.shape[1]] = right_img

    axes[1, 2].imshow(combined, cmap='gray')
    axes[1, 2].axvline(x=left_img.shape[1] + 5, color='r', linestyle='-', linewidth=3)
    axes[1, 2].set_title('Step 6: Left and Right Parts\n(Separated for Visualization)',
                         fontsize=14, fontweight='bold')
    axes[1, 2].text(left_img.shape[1]//2, -3, 'Left (Hen)',
                    ha='center', fontsize=12, fontweight='bold', color='blue')
    axes[1, 2].text(left_img.shape[1]+10+right_img.shape[1]//2, -3, 'Right (Tsukuri)',
                    ha='center', fontsize=12, fontweight='bold', color='red')
    axes[1, 2].axis('off')

    plt.suptitle('Left-Right Split Algorithm (Projection Method)',
                 fontsize=18, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'split_algorithm_explanation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"アルゴリズム説明図を保存: {output_path}")


def main():
    print("\n" + "="*80)
    print(" 左右分割（偏・旁分離）可視化")
    print("="*80 + "\n")

    # 出力ディレクトリ
    output_dir = 'results/split_visualization'
    os.makedirs(output_dir, exist_ok=True)

    # 対象クラス
    target_classes = [
        '3a2c', '3a2e', '3a4e', '3a6e', '3a51',
        '3a60', '3a64', '3a72', '3b4f', '3b6b',
        '4f40', '3c23', '3d26', '3d3b', '3d3e'
    ]

    # サンプル画像を読み込み
    print("サンプル画像を読み込み中...")
    images, class_labels = load_sample_images("./ETL8B-img-full", target_classes, num_samples=6)

    if len(images) == 0:
        print("エラー: 画像が読み込めませんでした")
        return

    print(f"読み込み完了: {len(images)}サンプル\n")

    # 各サンプルの個別可視化
    print("個別サンプルの可視化中...")
    for idx, (image, class_label) in enumerate(zip(images, class_labels)):
        output_path = os.path.join(output_dir, f'split_sample_{idx+1}_{class_label}.png')
        visualize_split(image, class_label, output_path)

    # 全サンプルをまとめた可視化
    print("\n全サンプルをまとめた可視化中...")
    visualize_all_samples(images, class_labels, output_dir)

    # アルゴリズム説明図
    print("\nアルゴリズム説明図を作成中...")
    create_explanation_figure(output_dir)

    print("\n" + "="*80)
    print(" 可視化完了")
    print("="*80)
    print(f"\n結果は {output_dir}/ に保存されました\n")
    print("生成されたファイル:")
    print(f"  - split_sample_*.png: 個別サンプルの可視化（{len(images)}ファイル）")
    print(f"  - all_samples_split.png: 全サンプルまとめ")
    print(f"  - split_algorithm_explanation.png: アルゴリズム説明図")


if __name__ == '__main__':
    main()
