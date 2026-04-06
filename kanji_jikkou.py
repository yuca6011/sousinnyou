#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kanji_jikkou.py
特徴効果測定実験（hierarchical_recognizer_optimized.py）

認識率（15クラス、各30サンプル、64x64画像、135テスト）:
  Exp1 Baseline（画素のみ）:    22.22%
  Exp2 HOG特徴のみ:            48.89%
  Exp3 Gabor特徴のみ:          36.30%
  Exp4 データ拡張のみ:         45.93%
  Exp5 HOG + データ拡張:       69.63% ★最良
  Exp6 HOG + Gabor:            38.52%
  Exp7 全特徴(HOG+Gabor+Aug):  34.07%
"""

import numpy as np
import cv2
import os
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from hierarchical_recognizer_optimized import HierarchicalPatternRecognizer

def load_etl8b_data(base_path, target_classes, max_samples=30):
    images = []
    labels = []
    class_info = {}

    print("Loading data...")
    for class_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(base_path, class_hex)
        if not os.path.exists(class_dir):
            print(f"  Warning: {class_dir} not found")
            continue

        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]

        if len(image_files) > max_samples:
            np.random.seed(42)
            image_files = np.random.choice(image_files, max_samples, replace=False)

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != (64, 64):
                    img = cv2.resize(img, (64, 64))
                img = img.astype(np.float64) / 255.0
                images.append(img)
                labels.append(class_idx)

        decimal = int(class_hex, 16)
        char = chr(decimal) if decimal < 0x10000 else f'[{decimal}]'
        class_info[class_idx] = {'hex': class_hex, 'char': char}
        print(f"  Class {class_idx} ({class_hex}): {char} - {len([l for l in labels if l == class_idx])} samples")

    print(f"Loaded: {len(images)} samples, {len(target_classes)} classes\\n")
    return np.array(images), np.array(labels), class_info


def run_experiment(exp_name, train_images, train_labels, test_images, test_labels, **kwargs):
    print("\\n" + "="*80)
    print(f"Experiment: {exp_name}")
    print("="*80)

    recognizer = HierarchicalPatternRecognizer(**kwargs)

    start_time = time.time()
    recognizer.train(train_images, train_labels)
    train_time = time.time() - start_time

    start_time = time.time()
    accuracy, predictions = recognizer.evaluate(test_images, test_labels)
    eval_time = time.time() - start_time

    cm = confusion_matrix(test_labels, predictions)

    print(f"\\nResults:")
    print(f"  Training time: {train_time:.4f}s")
    print(f"  Evaluation time: {eval_time:.4f}s")
    print(f"  Accuracy: {accuracy:.2f}%")

    return {
        'method': exp_name,
        'accuracy': float(accuracy),
        'train_time': float(train_time),
        'eval_time': float(eval_time),
        'predictions': predictions.tolist(),
        'confusion_matrix': cm.tolist()
    }


def visualize_results(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Accuracy comparison
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))

    methods = [r['method'] for r in all_results[:7]]
    accuracies = [r['accuracy'] for r in all_results[:7]]

    colors = ['#95a5a6', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#27ae60']

    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Feature Effect Measurement - Accuracy Comparison', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=11)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Time comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    train_times = [r['train_time'] for r in all_results[:7]]
    eval_times = [r['eval_time'] for r in all_results[:7]]

    axes[0].bar(methods, train_times, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    for i, t in enumerate(train_times):
        axes[0].text(i, t, f'{t:.2f}s', ha='center', va='bottom', fontsize=9)

    axes[1].bar(methods, eval_times, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[1].set_title('Evaluation Time Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    for i, t in enumerate(eval_times):
        axes[1].text(i, t, f'{t:.3f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 3. Confusion matrices (Baseline vs Full)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    cm_baseline = np.array(all_results[0]['confusion_matrix'])
    cm_full = np.array(all_results[6]['confusion_matrix'])

    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                cbar_kws={'label': 'Count'}, square=True)
    axes[0].set_title(f'Baseline Confusion Matrix\\nAccuracy: {all_results[0]["accuracy"]:.2f}%',
                     fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)

    sns.heatmap(cm_full, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                cbar_kws={'label': 'Count'}, square=True)
    axes[1].set_title(f'Full Features Confusion Matrix\\nAccuracy: {all_results[6]["accuracy"]:.2f}%',
                     fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 4. Split method comparison
    if len(all_results) > 7 and isinstance(all_results[7], dict) and 'projection' in all_results[7]:
        split_results = all_results[7]
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        split_methods = list(split_results.keys())
        split_accuracies = [split_results[m]['accuracy'] for m in split_methods]

        bars = ax.bar(split_methods, split_accuracies, color=['#e74c3c', '#3498db', '#2ecc71'],
                     alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title('Split Method Comparison (Full Features)', fontsize=15, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        for bar, acc in zip(bars, split_accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'split_method_comparison.png'), dpi=200, bbox_inches='tight')
        plt.close()

    print(f"\\nVisualization completed: {output_dir}")


def generate_report(all_results, class_info, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\\n")
        f.write(" Feature Effect Measurement Experiment Report\\n")
        f.write(" hierarchical_recognizer_optimized.py\\n")
        f.write("="*100 + "\\n\\n")

        f.write("1. Experimental Setup\\n")
        f.write("-"*100 + "\\n")
        f.write(f"  Number of classes: {len(class_info)}\\n")
        f.write(f"  Image size: 64x64\\n")
        f.write(f"  Pyramid levels: 5\\n")
        f.write(f"  Augmentation factor: 5x (when enabled)\\n\\n")

        f.write("  Target classes:\\n")
        for idx in sorted(class_info.keys()):
            info = class_info[idx]
            f.write(f"    Class {idx:2d} ({info['hex']}): {info['char']}\\n")
        f.write("\\n")

        f.write("2. Accuracy Comparison (Experiments 1-7)\\n")
        f.write("-"*100 + "\\n")
        f.write(f"{'Exp':<5} {'Method':<35} {'Accuracy':<12} {'Train Time':<12} {'Eval Time':<12}\\n")
        f.write("-"*100 + "\\n")

        for i, result in enumerate(all_results[:7], 1):
            f.write(f"{i:<5} {result['method']:<35} {result['accuracy']:>6.2f}% "
                   f"{result['train_time']:>10.4f}s {result['eval_time']:>10.4f}s\\n")
        f.write("\\n")

        best_idx = max(range(7), key=lambda i: all_results[i]['accuracy'])
        baseline_acc = all_results[0]['accuracy']
        best_acc = all_results[best_idx]['accuracy']
        improvement = best_acc - baseline_acc

        f.write("3. Analysis\\n")
        f.write("-"*100 + "\\n")
        f.write(f"  Baseline accuracy: {baseline_acc:.2f}%\\n")
        f.write(f"  Best accuracy: {best_acc:.2f}% (Exp {best_idx+1}: {all_results[best_idx]['method']})\\n")
        f.write(f"  Improvement: {improvement:+.2f}%\\n\\n")

        f.write("  Feature effects:\\n")
        hog_effect = all_results[1]['accuracy'] - baseline_acc
        gabor_effect = all_results[2]['accuracy'] - baseline_acc
        aug_effect = all_results[3]['accuracy'] - baseline_acc
        hog_aug_effect = all_results[4]['accuracy'] - baseline_acc
        hog_gabor_effect = all_results[5]['accuracy'] - baseline_acc

        f.write(f"    - HOG only:         {hog_effect:+.2f}% (vs baseline)\\n")
        f.write(f"    - Gabor only:       {gabor_effect:+.2f}%\\n")
        f.write(f"    - Augmentation only:{aug_effect:+.2f}%\\n")
        f.write(f"    - HOG + Aug:        {hog_aug_effect:+.2f}%\\n")
        f.write(f"    - HOG + Gabor:      {hog_gabor_effect:+.2f}%\\n")
        f.write(f"    - Full features:    {improvement:+.2f}%\\n\\n")

        if len(all_results) > 7 and isinstance(all_results[7], dict):
            split_results = all_results[7]
            f.write("4. Split Method Comparison (Full Features)\\n")
            f.write("-"*100 + "\\n")
            f.write(f"{'Method':<20} {'Accuracy':<12} {'Train Time':<12} {'Eval Time':<12}\\n")
            f.write("-"*100 + "\\n")

            for method, result in split_results.items():
                f.write(f"{method:<20} {result['accuracy']:>6.2f}% "
                       f"{result['train_time']:>10.4f}s {result['eval_time']:>10.4f}s\\n")
            f.write("\\n")

            best_split = max(split_results.keys(), key=lambda k: split_results[k]['accuracy'])
            f.write(f"  Best split method: {best_split} ({split_results[best_split]['accuracy']:.2f}%)\\n\\n")

        f.write("5. Conclusion\\n")
        f.write("-"*100 + "\\n")
        f.write("  This experiment revealed the effectiveness of each feature in hierarchical_recognizer_optimized.py.\\n")
        f.write(f"  The most effective setting is '{all_results[best_idx]['method']}'.\\n")
        f.write(f"  Compared to baseline, it achieved {improvement:.2f}% improvement.\\n\\n")

        f.write("6. Generated Files\\n")
        f.write("-"*100 + "\\n")
        f.write("  - accuracy_comparison.png\\n")
        f.write("  - time_comparison.png\\n")
        f.write("  - confusion_matrices.png\\n")
        f.write("  - split_method_comparison.png\\n")
        f.write("  - feature_comparison.json\\n")
        f.write("  - experiment_report.txt\\n")
        f.write("\\n")

    print(f"Report generated: {output_path}")


def main():
    print("="*100)
    print("Feature Effect Measurement Experiment")
    print("hierarchical_recognizer_optimized.py")
    print("="*100)

    target_classes = [
        '3a2c', '3a2e', '3a4e', '3a6e', '3a51',
        '3a60', '3a64', '3a72', '3b4f', '3b6b',
        '4f40', '3c23', '3d26', '3d3b', '3d3e'
    ]

    base_path = "./ETL8B-img-full"
    max_samples = 30

    images, labels, class_info = load_etl8b_data(base_path, target_classes, max_samples)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"\\nTraining data: {len(train_images)} samples")
    print(f"Test data: {len(test_images)} samples\\n")

    all_results = []

    # Experiment 1: Baseline
    all_results.append(run_experiment(
        "Baseline (Pixels only)",
        train_images, train_labels, test_images, test_labels,
        num_pyramid_levels=5,
        split_method='projection',
        pnn_sigma_method='adaptive',
        use_enhanced_features=False,
        use_data_augmentation=False,
        augmentation_factor=1
    ))

    # Experiment 2: HOG only
    all_results.append(run_experiment(
        "HOG Features Only",
        train_images, train_labels, test_images, test_labels,
        num_pyramid_levels=5,
        split_method='projection',
        pnn_sigma_method='adaptive',
        use_enhanced_features=True,
        feature_types=['hog'],
        use_data_augmentation=False,
        augmentation_factor=1
    ))

    # Experiment 3: Gabor only
    all_results.append(run_experiment(
        "Gabor Features Only",
        train_images, train_labels, test_images, test_labels,
        num_pyramid_levels=5,
        split_method='projection',
        pnn_sigma_method='adaptive',
        use_enhanced_features=True,
        feature_types=['gabor'],
        use_data_augmentation=False,
        augmentation_factor=1
    ))

    # Experiment 4: Augmentation only
    all_results.append(run_experiment(
        "Data Augmentation Only",
        train_images, train_labels, test_images, test_labels,
        num_pyramid_levels=5,
        split_method='projection',
        pnn_sigma_method='adaptive',
        use_enhanced_features=False,
        use_data_augmentation=True,
        augmentation_factor=5
    ))

    # Experiment 5: HOG + Augmentation
    all_results.append(run_experiment(
        "HOG + Augmentation",
        train_images, train_labels, test_images, test_labels,
        num_pyramid_levels=5,
        split_method='projection',
        pnn_sigma_method='adaptive',
        use_enhanced_features=True,
        feature_types=['hog'],
        use_data_augmentation=True,
        augmentation_factor=5
    ))

    # Experiment 6: HOG + Gabor (no aug)
    all_results.append(run_experiment(
        "HOG + Gabor (No Aug)",
        train_images, train_labels, test_images, test_labels,
        num_pyramid_levels=5,
        split_method='projection',
        pnn_sigma_method='adaptive',
        use_enhanced_features=True,
        feature_types=['hog', 'gabor'],
        use_data_augmentation=False,
        augmentation_factor=1
    ))

    # Experiment 7: Full features
    all_results.append(run_experiment(
        "Full Features (HOG+Gabor+Aug)",
        train_images, train_labels, test_images, test_labels,
        num_pyramid_levels=5,
        split_method='projection',
        pnn_sigma_method='adaptive',
        use_enhanced_features=True,
        feature_types=['hog', 'gabor'],
        use_data_augmentation=True,
        augmentation_factor=5
    ))

    # Experiment 8: Split method comparison
    print("\\n" + "="*80)
    print("Experiment 8: Split Method Comparison (projection vs fixed vs adaptive)")
    print("="*80)

    split_results = {}
    for split_method in ['projection', 'fixed', 'adaptive']:
        print(f"\\n--- Split method: {split_method} ---")

        result = run_experiment(
            f'Split: {split_method}',
            train_images, train_labels, test_images, test_labels,
            num_pyramid_levels=5,
            split_method=split_method,
            pnn_sigma_method='adaptive',
            use_enhanced_features=True,
            feature_types=['hog', 'gabor'],
            use_data_augmentation=True,
            augmentation_factor=5
        )
        split_results[split_method] = result

    all_results.append(split_results)

    # Save results
    output_dir = "./results/kanji_jikkou"
    os.makedirs(output_dir, exist_ok=True)

    json_output = {
        'experiments_1_to_7': all_results[:7],
        'experiment_8_split_methods': split_results
    }
    with open(os.path.join(output_dir, 'feature_comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    visualize_results(all_results, output_dir)

    report_path = os.path.join(output_dir, 'experiment_report.txt')
    generate_report(all_results, class_info, report_path)

    print("\\n" + "="*100)
    print("Experiment completed!")
    print("="*100)
    print(f"Results saved to {output_dir}/\\n")

    print("Accuracy Summary:")
    for i, result in enumerate(all_results[:7], 1):
        print(f"  Exp {i}: {result['method']:<35} {result['accuracy']:>6.2f}%")

    print("\\nSplit Method Comparison:")
    for method, result in split_results.items():
        print(f"  {method:<20} {result['accuracy']:>6.2f}%")


if __name__ == "__main__":
    main()
