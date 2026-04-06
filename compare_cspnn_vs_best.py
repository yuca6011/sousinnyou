#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
100クラス比較実験: CS-PNN(Algorithm1) vs Best Model(旧版)

比較対象:
  (A) CS-PNN Algorithm 1 (kanji_best_improved.py) — 今回実装
  (B) Best Model 旧版 (kanji_best_standalone.py) — 96.67%実績
"""

import os
import sys
import time
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# 100クラスリスト（compare_100classes_v2.py と同一）
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


def load_etl8b_data(data_dir, target_classes):
    """ETL8Bデータ読み込み"""
    images, labels = [], []
    for label_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(data_dir, class_hex)
        if not os.path.exists(class_dir):
            print(f"  警告: {class_dir} が見つかりません")
            continue
        image_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
        for img_file in image_files:
            img = cv2.imread(os.path.join(class_dir, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img, (64, 64)))
                labels.append(label_idx)
    return np.array(images), np.array(labels)


def main():
    print("=" * 80)
    print(" 100クラス比較実験: CS-PNN(Algorithm1) vs Best Model(旧版)")
    print("=" * 80)

    # データ読み込み
    print("\nデータ読み込み中...")
    images, labels = load_etl8b_data("./ETL8B-img-full", TARGET_CLASSES_100)
    n_classes = len(np.unique(labels))
    print(f"  合計: {len(images)}サンプル, {n_classes}クラス")

    # 訓練・テスト分割（既存実験と同一条件）
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"  訓練: {len(train_images)}, テスト: {len(test_images)}")

    results = []

    # =============================================
    # (A) CS-PNN Algorithm 1 (kanji_best_improved.py)
    # =============================================
    print("\n" + "=" * 80)
    print(" (A) CS-PNN Algorithm 1 (kanji_best_improved.py)")
    print("=" * 80)

    from kanji_best_improved import KanjiBestRecognizer as ImprovedRecognizer

    model_a = ImprovedRecognizer()

    start = time.time()
    model_a.train(train_images, train_labels)
    train_time_a = time.time() - start

    start = time.time()
    accuracy_a, predictions_a = model_a.evaluate(test_images, test_labels)
    eval_time_a = time.time() - start

    info_a = model_a.get_model_info()
    f1_a = f1_score(test_labels, predictions_a, average='macro')

    results.append({
        'name': 'CS-PNN Algorithm 1',
        'accuracy': accuracy_a,
        'f1_macro': f1_a,
        'train_time': train_time_a,
        'eval_time': eval_time_a,
        'info': info_a,
    })

    # =============================================
    # (B) Best Model 旧版 (kanji_best_standalone.py)
    # =============================================
    print("\n" + "=" * 80)
    print(" (B) Best Model 旧版 (kanji_best_standalone.py)")
    print("=" * 80)

    from kanji_best_standalone import KanjiBestRecognizer as OldRecognizer

    model_b = OldRecognizer()

    start = time.time()
    model_b.train(train_images, train_labels)
    train_time_b = time.time() - start

    start = time.time()
    accuracy_b, predictions_b = model_b.evaluate(test_images, test_labels)
    eval_time_b = time.time() - start

    info_b = model_b.get_model_info() if hasattr(model_b, 'get_model_info') else {
        'algorithm': 'adaptive PNN (K-means + mean(NN)/sqrt(2))',
        'sigma_method': 'adaptive',
    }
    f1_b = f1_score(test_labels, predictions_b, average='macro')

    results.append({
        'name': 'Best Model (旧版)',
        'accuracy': accuracy_b,
        'f1_macro': f1_b,
        'train_time': train_time_b,
        'eval_time': eval_time_b,
        'info': info_b,
    })

    # =============================================
    # 結果比較
    # =============================================
    print("\n" + "=" * 80)
    print(" 比較結果サマリー")
    print("=" * 80)

    print(f"\n{'モデル':<30} {'認識率':>10} {'F1-macro':>10} {'学習時間':>12} {'評価時間':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<30} {r['accuracy']:>9.2f}% {r['f1_macro']:>9.4f} "
              f"{r['train_time']:>10.2f}秒 {r['eval_time']:>10.2f}秒")

    print(f"\n差分: {results[0]['accuracy'] - results[1]['accuracy']:+.2f}% "
          f"(CS-PNN Algorithm1 - 旧版)")

    # モデル詳細
    print("\n--- モデル詳細 ---")
    for r in results:
        print(f"\n[{r['name']}]")
        for k, v in r['info'].items():
            print(f"  {k}: {v}")

    # クラス別精度比較
    print("\n--- クラス別精度（上位/下位5クラス差分） ---")
    per_class_a = []
    per_class_b = []
    for cls_idx in range(n_classes):
        mask = test_labels == cls_idx
        if mask.sum() > 0:
            acc_a = 100.0 * np.sum(predictions_a[mask] == cls_idx) / mask.sum()
            acc_b = 100.0 * np.sum(predictions_b[mask] == cls_idx) / mask.sum()
            per_class_a.append(acc_a)
            per_class_b.append(acc_b)
        else:
            per_class_a.append(0)
            per_class_b.append(0)

    diffs = [(i, per_class_a[i] - per_class_b[i], per_class_a[i], per_class_b[i])
             for i in range(n_classes)]
    diffs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'クラス':>6} {'CS-PNN':>10} {'旧版':>10} {'差分':>10}")
    print("-" * 40)
    for i, diff, acc_a, acc_b in diffs[:5]:
        print(f"  {TARGET_CLASSES_100[i]:>6} {acc_a:>9.1f}% {acc_b:>9.1f}% {diff:>+9.1f}%")
    print("  ...")
    for i, diff, acc_a, acc_b in diffs[-5:]:
        print(f"  {TARGET_CLASSES_100[i]:>6} {acc_a:>9.1f}% {acc_b:>9.1f}% {diff:>+9.1f}%")

    # CS-PNNが勝ったクラス数
    wins_a = sum(1 for d in diffs if d[1] > 0)
    wins_b = sum(1 for d in diffs if d[1] < 0)
    ties = sum(1 for d in diffs if d[1] == 0)
    print(f"\nCS-PNN勝利: {wins_a}クラス, 旧版勝利: {wins_b}クラス, 引き分け: {ties}クラス")

    print("\n" + "=" * 80)
    print(" 実験完了")
    print("=" * 80)


if __name__ == '__main__':
    main()
