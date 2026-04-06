#!/usr/bin/env python3
"""
左右分割型階層構造的画像パターン認識器の実験
元の階層型、左右分割型（現行版）、左右分割型（改善版）の3手法を比較

認識率（15クラス、各30サンプル、32x32画像、135テスト）:
  元の階層構造的認識器:          73.33%
  左右分割型認識器（改善版）:    63.70%
  左右分割型認識器（現行版）:    40.74%
"""

import numpy as np
import cv2
import os
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 認識器のインポート
from hierarchical_recognizer import HierarchicalPatternRecognizer
from hierarchical_recognizer_lr import LeftRightSplitRecognizer
from hierarchical_recognizer_lr_improved import ImprovedLeftRightSplitRecognizer

def load_etl8b_data(base_path, target_classes, max_samples=30):
    """ETL8Bデータ読み込み"""
    images = []
    labels = []
    class_info = {}

    print("データ読み込み中...")
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
        print(f"  クラス {class_idx} ({class_hex}): {char} - {len([l for l in labels if l == class_idx])}サンプル")

    print(f"読み込み完了: {len(images)}サンプル, {len(target_classes)}クラス\n")
    return np.array(images), np.array(labels), class_info

def experiment_original(train_images, train_labels, test_images, test_labels):
    """元の階層型認識器"""
    print("\n" + "="*80)
    print("実験1: 元の階層構造的画像パターン認識器")
    print("="*80)

    recognizer = HierarchicalPatternRecognizer(
        num_pyramid_levels=5,
        lib_path='./hierarchical_ext.so'
    )

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
    print(f"  学習時間: {train_time:.4f}秒")
    print(f"  評価時間: {eval_time:.4f}秒")
    print(f"  認識率: {accuracy:.2f}%")

    return {
        'method': 'Original Hierarchical',
        'accuracy': float(accuracy),
        'train_time': float(train_time),
        'eval_time': float(eval_time),
        'predictions': predictions.tolist(),
        'confusion_matrix': cm.tolist()
    }

def experiment_lr_split(train_images, train_labels, test_images, test_labels):
    """左右分割型認識器（現行版）"""
    print("\n" + "="*80)
    print("実験2: 左右分割型階層構造的画像パターン認識器（現行版）")
    print("="*80)

    recognizer = LeftRightSplitRecognizer(
        num_pyramid_levels=5,
        lib_path='./hierarchical_ext.so'
    )

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
    print(f"  学習時間: {train_time:.4f}秒")
    print(f"  評価時間: {eval_time:.4f}秒")
    print(f"  認識率: {accuracy:.2f}%")

    return {
        'method': 'LR-Split (Current)',
        'accuracy': float(accuracy),
        'train_time': float(train_time),
        'eval_time': float(eval_time),
        'predictions': predictions.tolist(),
        'confusion_matrix': cm.tolist()
    }

def experiment_lr_split_improved(train_images, train_labels, test_images, test_labels):
    """左右分割型認識器（改善版）"""
    print("\n" + "="*80)
    print("実験3: 左右分割型階層構造的画像パターン認識器（改善版）")
    print("="*80)

    recognizer = ImprovedLeftRightSplitRecognizer(
        num_pyramid_levels=5,
        lib_path='./hierarchical_ext.so'
    )

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
    print(f"  学習時間: {train_time:.4f}秒")
    print(f"  評価時間: {eval_time:.4f}秒")
    print(f"  認識率: {accuracy:.2f}%")

    return {
        'method': 'LR-Split (Improved)',
        'accuracy': float(accuracy),
        'train_time': float(train_time),
        'eval_time': float(eval_time),
        'predictions': predictions.tolist(),
        'confusion_matrix': cm.tolist()
    }

def visualize_comparison(original_results, lr_results, lr_improved_results, class_info, output_dir):
    """3手法の比較可視化"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 認識率比較
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    methods = ['Original\nHierarchical', 'LR-Split\n(Current)', 'LR-Split\n(Improved)']
    accuracies = [original_results['accuracy'], lr_results['accuracy'], lr_improved_results['accuracy']]
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    bars = ax.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Recognition Accuracy Comparison', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 混同行列比較
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    cm_orig = np.array(original_results['confusion_matrix'])
    cm_lr = np.array(lr_results['confusion_matrix'])
    cm_lr_improved = np.array(lr_improved_results['confusion_matrix'])

    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                cbar_kws={'label': 'Count'}, square=True)
    axes[0].set_title('Original Hierarchical\nConfusion Matrix', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)

    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                cbar_kws={'label': 'Count'}, square=True)
    axes[1].set_title('LR-Split (Current)\nConfusion Matrix', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)

    sns.heatmap(cm_lr_improved, annot=True, fmt='d', cmap='Greens', ax=axes[2],
                cbar_kws={'label': 'Count'}, square=True)
    axes[2].set_title('LR-Split (Improved)\nConfusion Matrix', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Predicted Label', fontsize=11)
    axes[2].set_ylabel('True Label', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 時間比較
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    train_times = [original_results['train_time'], lr_results['train_time'],
                   lr_improved_results['train_time']]
    eval_times = [original_results['eval_time'], lr_results['eval_time'],
                  lr_improved_results['eval_time']]

    bars = axes[0].bar(methods, train_times, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Time (seconds)', fontsize=11)
    axes[0].set_title('Training Time', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for bar, t in zip(bars, train_times):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{t:.3f}s', ha='center', va='bottom', fontsize=10)

    bars = axes[1].bar(methods, eval_times, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Time (seconds)', fontsize=11)
    axes[1].set_title('Evaluation Time', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    for bar, t in zip(bars, eval_times):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{t:.4f}s', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n可視化完了: {output_dir}")

def generate_report(original_results, lr_results, lr_improved_results, class_info, output_path):
    """レポート生成"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" 階層構造的画像パターン認識器 比較実験レポート\n")
        f.write(" 元の階層型 vs 左右分割型（現行版） vs 左右分割型（改善版）\n")
        f.write("="*80 + "\n\n")

        f.write("1. 実験設定\n")
        f.write("-"*80 + "\n")
        f.write(f"  対象クラス数: {len(class_info)}\n")
        f.write(f"  画像サイズ: 32x32\n")
        f.write(f"  ピラミッド階層数: 5\n\n")

        f.write("  対象クラス:\n")
        for idx in sorted(class_info.keys()):
            info = class_info[idx]
            # 偏情報を追加
            from hierarchical_recognizer_lr_improved import ImprovedLeftRightSplitRecognizer
            radical = ImprovedLeftRightSplitRecognizer.CLASS_TO_RADICAL.get(idx, '不明')
            f.write(f"    クラス {idx:2d} ({info['hex']}): {info['char']} ({radical})\n")
        f.write("\n")

        f.write("2. 認識率比較\n")
        f.write("-"*80 + "\n")
        f.write(f"{'手法':<35} {'認識率':<15}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'元の階層構造的認識器':<30} {original_results['accuracy']:>6.2f}%\n")
        f.write(f"{'左右分割型認識器（現行版）':<30} {lr_results['accuracy']:>6.2f}%\n")
        f.write(f"{'左右分割型認識器（改善版）':<30} {lr_improved_results['accuracy']:>6.2f}%\n")
        f.write("\n")

        best_accuracy = max(original_results['accuracy'], lr_results['accuracy'],
                           lr_improved_results['accuracy'])
        if lr_improved_results['accuracy'] == best_accuracy:
            diff_from_orig = lr_improved_results['accuracy'] - original_results['accuracy']
            diff_from_lr = lr_improved_results['accuracy'] - lr_results['accuracy']
            f.write(f"  ✓ 改善版LR分割型が最高精度\n")
            f.write(f"    - 元の階層型より {diff_from_orig:+.2f}%\n")
            f.write(f"    - 現行版LR分割型より {diff_from_lr:+.2f}%\n\n")
        elif original_results['accuracy'] == best_accuracy:
            diff = original_results['accuracy'] - lr_improved_results['accuracy']
            f.write(f"  ✓ 元の階層型が最高精度 ({diff:.2f}% 差)\n\n")

        f.write("3. 処理時間比較\n")
        f.write("-"*80 + "\n")
        f.write(f"{'手法':<35} {'学習時間':<15} {'評価時間':<15}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'元の階層構造的認識器':<30} {original_results['train_time']:>10.4f}秒 "
                f"{original_results['eval_time']:>10.4f}秒\n")
        f.write(f"{'左右分割型認識器（現行版）':<30} {lr_results['train_time']:>10.4f}秒 "
                f"{lr_results['eval_time']:>10.4f}秒\n")
        f.write(f"{'左右分割型認識器（改善版）':<30} {lr_improved_results['train_time']:>10.4f}秒 "
                f"{lr_improved_results['eval_time']:>10.4f}秒\n")
        f.write("\n")

        f.write("4. 特徴\n")
        f.write("-"*80 + "\n")
        f.write("  元の階層構造的認識器:\n")
        f.write("    - Level 2 (8×8) で共通部認識\n")
        f.write("    - Level 4 (32×32) で非共通部認識\n")
        f.write("    - 階層的な特徴表現\n\n")
        f.write("  左右分割型認識器（現行版）:\n")
        f.write("    - 画像を自動的に左（偏）と右（旁）に分割\n")
        f.write("    - 各クラスを独立した偏グループとして扱う（15グループ）\n")
        f.write("    - 問題: 同じ偏を持つクラスが別グループになる\n\n")
        f.write("  左右分割型認識器（改善版）:\n")
        f.write("    - 画像を自動的に左（偏）と右（旁）に分割\n")
        f.write("    - 実際の偏でグループ化（8グループ）\n")
        f.write("    - 同じ偏を持つクラスが同じグループで学習\n")
        f.write("    - 偏認識の学習データが増加\n\n")

        f.write("5. 偏グループ情報\n")
        f.write("-"*80 + "\n")
        from hierarchical_recognizer_lr_improved import ImprovedLeftRightSplitRecognizer
        radical_groups = {}
        for class_id, radical in ImprovedLeftRightSplitRecognizer.CLASS_TO_RADICAL.items():
            if class_id < len(class_info):
                if radical not in radical_groups:
                    radical_groups[radical] = []
                radical_groups[radical].append(class_id)

        for radical, classes in sorted(radical_groups.items()):
            class_chars = [class_info[c]['char'] for c in classes]
            f.write(f"  {radical}: クラス {classes} ({', '.join(class_chars)})\n")
        f.write("\n")

        f.write("6. 生成ファイル\n")
        f.write("-"*80 + "\n")
        f.write("  - accuracy_comparison.png: 認識率比較\n")
        f.write("  - confusion_matrices.png: 混同行列比較\n")
        f.write("  - time_comparison.png: 処理時間比較\n")
        f.write("  - comparison_report.txt: このレポート\n")
        f.write("\n")

    print(f"レポート生成完了: {output_path}")

def main():
    """メイン実行"""
    print("="*80)
    print("階層構造的画像パターン認識器 比較実験")
    print("元の階層型 vs 左右分割型（現行版） vs 左右分割型（改善版）")
    print("="*80)

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

    # 訓練・テスト分割
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"訓練データ: {len(train_images)}サンプル")
    print(f"テストデータ: {len(test_images)}サンプル")

    # 実験1: 元の階層型
    original_results = experiment_original(train_images, train_labels,
                                          test_images, test_labels)

    # 実験2: 左右分割型（現行版）
    lr_results = experiment_lr_split(train_images, train_labels,
                                    test_images, test_labels)

    # 実験3: 左右分割型（改善版）
    lr_improved_results = experiment_lr_split_improved(train_images, train_labels,
                                                       test_images, test_labels)

    # 結果保存
    output_dir = "./results/lr_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # JSON保存
    with open(os.path.join(output_dir, 'original_results.json'), 'w', encoding='utf-8') as f:
        json.dump(original_results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'lr_results.json'), 'w', encoding='utf-8') as f:
        json.dump(lr_results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'lr_improved_results.json'), 'w', encoding='utf-8') as f:
        json.dump(lr_improved_results, f, indent=2, ensure_ascii=False)

    # 可視化
    visualize_comparison(original_results, lr_results, lr_improved_results, class_info, output_dir)

    # レポート生成
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    generate_report(original_results, lr_results, lr_improved_results, class_info, report_path)

    print("\n" + "="*80)
    print("比較実験完了！")
    print("="*80)
    print(f"結果は {output_dir}/ に保存されました")
    print("\n認識率サマリー:")
    print(f"  元の階層型:               {original_results['accuracy']:.2f}%")
    print(f"  左右分割型（現行版）:     {lr_results['accuracy']:.2f}%")
    print(f"  左右分割型（改善版）:     {lr_improved_results['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
