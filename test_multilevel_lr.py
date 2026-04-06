#!/usr/bin/env python3
"""
マルチレベル左右分割型階層構造的画像パターン認識器の実験
15クラスで性能評価

認識率（15クラス、各30サンプル、32x32画像、135テスト）:
  マルチレベル左右分割型: 40.74%（学習0.13秒、評価0.07秒）
"""

import numpy as np
import cv2
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from hierarchical_recognizer_multilevel_lr import MultiLevelLeftRightRecognizer

def load_etl8b_data(base_path, target_classes, max_samples=30):
    """ETL8Bデータ読み込み"""
    images = []
    labels = []
    class_info = {}

    print("データ読み込み中...")
    for class_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(base_path, class_hex)
        if not os.path.exists(class_dir):
            print(f"  警告: {class_dir} が見つかりません")
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
        print(f"  クラス {class_idx:2d} ({class_hex}): {char} - {len([l for l in labels if l == class_idx])}サンプル")

    print(f"\n読み込み完了: {len(images)}サンプル, {len(target_classes)}クラス\n")
    return np.array(images), np.array(labels), class_info

def main():
    """メイン実行"""
    print("="*80)
    print("マルチレベル左右分割型階層構造的画像パターン認識器 - 15クラス実験")
    print("="*80)
    print()

    # 15クラスのデータ
    target_classes = [
        '3a2c', '3a2e', '3a4e', '3a6e', '3a51',
        '3a60', '3a64', '3a72', '3b4f', '3b6b',
        '4f40', '3c23', '3d26', '3d3b', '3d3e'
    ]

    base_path = "./ETL8B-img-full"
    max_samples = 30

    # データ読み込み
    images, labels, class_info = load_etl8b_data(base_path, target_classes, max_samples)

    if len(images) == 0:
        print("エラー: データが読み込めませんでした")
        return

    # 訓練・テスト分割
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"訓練データ: {len(train_images)}サンプル")
    print(f"テストデータ: {len(test_images)}サンプル")
    print()

    # 認識器の構築
    recognizer = MultiLevelLeftRightRecognizer(
        num_pyramid_levels=5,
        lib_path='./hierarchical_ext.so'
    )

    # 学習
    print("\n" + "="*80)
    print("学習開始")
    print("="*80)
    start_time = time.time()
    recognizer.train(train_images, train_labels)
    train_time = time.time() - start_time

    # 評価
    print("\n" + "="*80)
    print("評価開始")
    print("="*80)
    start_time = time.time()
    accuracy, predictions = recognizer.evaluate(test_images, test_labels)
    eval_time = time.time() - start_time

    # 混同行列
    cm = confusion_matrix(test_labels, predictions)

    # クラス別精度
    print("\n" + "="*80)
    print("クラス別認識率")
    print("="*80)
    for class_id in sorted(class_info.keys()):
        info = class_info[class_id]
        mask = test_labels == class_id
        if np.sum(mask) > 0:
            class_correct = np.sum(predictions[mask] == class_id)
            class_total = np.sum(mask)
            class_acc = 100.0 * class_correct / class_total
            print(f"  クラス {class_id:2d} ({info['char']}): "
                  f"{class_acc:6.2f}% ({class_correct:2d}/{class_total:2d})")

    # レベル別の詳細分析（オプション）
    print("\n" + "="*80)
    print("認識器構成")
    print("="*80)
    print("Level 4 (32×32): 最高解像度、重み=3")
    print("Level 3 (16×16): 中解像度、重み=2")
    print("Level 2 (8×8):   低解像度、重み=1")
    print("→ 重み付き多数決で最終判定")

    # 結果サマリー
    print("\n" + "="*80)
    print("実験結果サマリー")
    print("="*80)
    print(f"総認識率:     {accuracy:.2f}%")
    print(f"学習時間:     {train_time:.4f}秒")
    print(f"評価時間:     {eval_time:.4f}秒")
    print(f"クラス数:     {len(target_classes)}")
    print(f"処理レベル数: 3 (Level 2, 3, 4)")
    print()

    print("="*80)
    print("実験完了")
    print("="*80)

if __name__ == "__main__":
    main()
