#!/usr/bin/env python3
"""
左右分割検知のテスト
白い境界を検出して漢字を左（偏）と右（旁）に分割

※このファイルは分割アルゴリズムのテスト用。認識率の測定は行わない。
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def detect_split_column(image):
    """
    垂直投影を使って左右の分割位置を検出

    Args:
        image: 2D numpy array (grayscale, 0.0-1.0)

    Returns:
        split_col: 分割列の位置
    """
    # 垂直投影を計算（各列の合計値）
    # 値が小さいほど暗い（黒い）、大きいほど明るい（白い）
    vertical_projection = np.sum(image, axis=0)

    # 中央付近で最大値（最も白い列）を探す
    height, width = image.shape
    center = width // 2
    search_start = max(0, center - width // 4)
    search_end = min(width, center + width // 4)

    # 検索範囲内で最大値を探す
    search_range = range(search_start, search_end)
    if len(search_range) == 0:
        return center

    split_col = max(search_range, key=lambda x: vertical_projection[x])

    return split_col

def split_left_right(image, split_col):
    """
    画像を左右に分割

    Args:
        image: 2D numpy array
        split_col: 分割列

    Returns:
        left_part, right_part: 左右の画像
    """
    left_part = image[:, :split_col]
    right_part = image[:, split_col+1:]

    return left_part, right_part

def visualize_split(image, split_col, left_part, right_part, title, output_path):
    """分割結果の可視化"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: オリジナル画像と投影
    ax1 = axes[0, 0]
    ax1.imshow(image, cmap='gray')
    ax1.axvline(x=split_col, color='red', linewidth=2, linestyle='--', label='分割位置')
    ax1.set_title('元画像 + 分割位置', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_xlabel(f'分割列: {split_col}')

    # 垂直投影
    ax2 = axes[0, 1]
    vertical_projection = np.sum(image, axis=0)
    ax2.plot(vertical_projection)
    ax2.axvline(x=split_col, color='red', linewidth=2, linestyle='--')
    ax2.set_title('垂直投影（列ごとの合計）', fontsize=12)
    ax2.set_xlabel('列の位置')
    ax2.set_ylabel('投影値（高い=白い）')
    ax2.grid(True, alpha=0.3)

    # 元画像（拡大）
    ax3 = axes[0, 2]
    ax3.imshow(image, cmap='gray')
    ax3.set_title('元画像', fontsize=12)

    # Row 2: 左部分と右部分
    ax4 = axes[1, 0]
    ax4.imshow(left_part, cmap='gray')
    ax4.set_title(f'左部分（偏）\nサイズ: {left_part.shape}',
                 fontsize=12, fontweight='bold', color='green')
    ax4.set_xlabel('共通部で使用')

    ax5 = axes[1, 1]
    ax5.imshow(right_part, cmap='gray')
    ax5.set_title(f'右部分（旁）\nサイズ: {right_part.shape}',
                 fontsize=12, fontweight='bold', color='blue')
    ax5.set_xlabel('非共通部で使用')

    # 左右を並べて表示
    ax6 = axes[1, 2]
    # 左右のサイズを揃える
    max_width = max(left_part.shape[1], right_part.shape[1])
    left_padded = np.pad(left_part, ((0, 0), (0, max_width - left_part.shape[1])),
                        constant_values=1.0)
    right_padded = np.pad(right_part, ((0, 0), (0, max_width - right_part.shape[1])),
                         constant_values=1.0)
    combined = np.hstack([left_padded, right_padded])
    ax6.imshow(combined, cmap='gray')
    ax6.axvline(x=left_padded.shape[1]-0.5, color='red', linewidth=2, linestyle='--')
    ax6.set_title('左右を並べて表示', fontsize=12)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")

def load_and_test_split(base_path, target_classes, num_samples=5):
    """複数のサンプルで分割をテスト"""
    output_dir = "./results/lr_split_test"
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("左右分割検知テスト")
    print("="*80)

    for class_idx, class_hex in enumerate(target_classes[:5]):  # 最初の5クラスをテスト
        class_dir = os.path.join(base_path, class_hex)
        if not os.path.exists(class_dir):
            continue

        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        if len(image_files) == 0:
            continue

        # 各クラスから数サンプル取得
        for i, img_file in enumerate(image_files[:num_samples]):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                # 正規化
                if img.shape != (32, 32):
                    img = cv2.resize(img, (32, 32))
                img = img.astype(np.float64) / 255.0

                # 分割位置検出
                split_col = detect_split_column(img)
                left_part, right_part = split_left_right(img, split_col)

                # 文字情報
                decimal = int(class_hex, 16)
                char = chr(decimal) if decimal < 0x10000 else f'[{decimal}]'

                # 可視化
                title = f'クラス {class_idx} ({class_hex}): {char} - サンプル {i+1}'
                output_path = os.path.join(output_dir,
                                          f'split_class{class_idx}_{i+1}.png')

                visualize_split(img, split_col, left_part, right_part,
                              title, output_path)

                print(f"  {char}: 分割位置={split_col}, "
                      f"左サイズ={left_part.shape}, 右サイズ={right_part.shape}")

    print("\n" + "="*80)
    print("テスト完了！")
    print("="*80)
    print(f"結果は {output_dir}/ に保存されました")

def main():
    # 15クラスのデータ
    target_classes = [
        '3a2c',  # 根
        '3a2e',  # 混
        '3a4e',  # 採
        '3a6e',  # 作
        '3a51',  # 済
        '3a60',  # 材
        '3a64',  # 坂
        '3a72',  # 昨
        '3b4f',  # 始
        '3b6b',  # 視
        '4f40',  # 論
        '3c23',  # 治
        '3d26',  # 拾
        '3d3b',  # 住
        '3d3e',  # 従
    ]

    base_path = "./ETL8B-img-full"
    load_and_test_split(base_path, target_classes, num_samples=3)

if __name__ == "__main__":
    main()
