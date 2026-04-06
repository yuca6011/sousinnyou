#!/usr/bin/env python3
"""
階層型パターン認識器 vs MLP 比較実験（15クラス）
左右分離型の漢字を用いた比較実験

認識率（15クラス、各30サンプル、32x32画像、135テスト）:
  MLP:                   77.04%（学習2.56秒）
  階層型パターン認識器:  73.33%（学習0.009秒）
"""

import numpy as np
import json
import time
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 階層型パターン認識器のインポート
from hierarchical_recognizer import HierarchicalPatternRecognizer, ImagePreprocessor
import ctypes as ct

# MLPのインポート
import torch
from mlp_etl8b import MLP, MLPTrainer, ETL8BDataset, load_etl_data
from torch.utils.data import DataLoader

# RBF統計構造（C++から取得）
class RBFStatistics(ct.Structure):
    _fields_ = [
        ("total_rbf_count", ct.c_int),
        ("common_rbf_count", ct.c_int),
        ("non_common_rbf_counts", ct.POINTER(ct.c_int)),
        ("num_groups", ct.c_int),
        ("class_rbf_counts", ct.POINTER(ct.c_int)),
        ("num_classes", ct.c_int),
    ]

def get_rbf_statistics(recognizer):
    """階層型パターン認識器のRBF統計を取得"""
    recognizer.lib.getRBFStatistics.argtypes = [ct.c_void_p]
    recognizer.lib.getRBFStatistics.restype = ct.POINTER(RBFStatistics)
    recognizer.lib.freeRBFStatistics.argtypes = [ct.POINTER(RBFStatistics)]
    recognizer.lib.freeRBFStatistics.restype = None

    stats_ptr = recognizer.lib.getRBFStatistics(recognizer.c_recognizer)
    stats = stats_ptr.contents

    non_common_rbfs = [stats.non_common_rbf_counts[i] for i in range(stats.num_groups)]
    class_rbfs = [stats.class_rbf_counts[i] for i in range(stats.num_classes)]

    result = {
        'total_rbf': stats.total_rbf_count,
        'common_rbf': stats.common_rbf_count,
        'non_common_rbf_per_group': non_common_rbfs,
        'rbf_per_class': class_rbfs,
        'total_non_common_rbf': sum(non_common_rbfs)
    }

    recognizer.lib.freeRBFStatistics(stats_ptr)
    return result

def load_etl8b_data(base_path, target_classes, max_samples=30):
    """ETL8Bデータ読み込み"""
    images = []
    labels = []
    class_info = {}

    print(f"データ読み込み中: {base_path}")

    for class_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(base_path, class_hex)
        if not os.path.exists(class_dir):
            print(f"警告: {class_dir} が見つかりません")
            continue

        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]

        if len(image_files) > max_samples:
            np.random.seed(42)
            image_files = np.random.choice(image_files, max_samples, replace=False)

        class_images = []
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            import cv2
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != (32, 32):
                    img = cv2.resize(img, (32, 32))
                img = img.astype(np.float64) / 255.0
                images.append(img)
                labels.append(class_idx)
                class_images.append(img)

        # 文字コードから文字を取得
        decimal = int(class_hex, 16)
        char = chr(decimal) if decimal < 0x10000 else f'[{decimal}]'
        class_info[class_idx] = {
            'hex': class_hex,
            'char': char,
            'count': len(class_images)
        }
        print(f"  クラス {class_idx:2d} ({class_hex}): {char} - {len(class_images)}サンプル")

    print(f"読み込み完了: {len(images)}サンプル, {len(target_classes)}クラス")
    return np.array(images), np.array(labels), class_info

def experiment_hierarchical(images, labels, class_info, num_pyramid_levels=5):
    """階層型パターン認識器の実験"""
    print("\n" + "="*80)
    print("階層型パターン認識器実験")
    print("="*80)

    # 訓練・テスト分割
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"訓練データ: {len(train_images)}サンプル")
    print(f"テストデータ: {len(test_images)}サンプル")
    print(f"ピラミッド階層数: {num_pyramid_levels}")

    # 認識器初期化
    start_time = time.time()
    recognizer = HierarchicalPatternRecognizer(
        num_pyramid_levels=num_pyramid_levels,
        lib_path='./hierarchical_ext.so'
    )

    # 学習
    print("\n学習開始...")
    train_start = time.time()
    recognizer.train(train_images, train_labels)
    train_time = time.time() - train_start

    # RBF統計取得
    rbf_stats = get_rbf_statistics(recognizer)

    # 評価（evaluate メソッドを使用）
    print("\n評価開始...")
    eval_start = time.time()
    accuracy, predictions = recognizer.evaluate(test_images, test_labels)
    eval_time = time.time() - eval_start

    correct = np.sum(predictions == test_labels)

    # クラス別精度
    class_accuracies = {}
    for class_idx in range(len(class_info)):
        mask = test_labels == class_idx
        if np.sum(mask) > 0:
            class_acc = np.mean(predictions[mask] == test_labels[mask]) * 100
            class_accuracies[class_idx] = {
                'accuracy': class_acc,
                'correct': np.sum(predictions[mask] == test_labels[mask]),
                'total': np.sum(mask)
            }

    # 混同行列
    cm = confusion_matrix(test_labels, predictions)

    # 結果表示
    print(f"\n{'='*80}")
    print(f"階層型パターン認識器 結果")
    print(f"{'='*80}")
    print(f"総合認識率: {accuracy:.2f}% ({correct}/{len(test_labels)})")
    print(f"学習時間: {train_time:.4f}秒")
    print(f"評価時間: {eval_time:.4f}秒")
    print(f"平均予測時間: {eval_time/len(test_images)*1000:.2f}ミリ秒/画像")
    print(f"\nRBF統計:")
    print(f"  総RBF数: {rbf_stats['total_rbf']}")
    print(f"  共通部RBF数: {rbf_stats['common_rbf']}")
    print(f"  非共通部RBF数: {rbf_stats['total_non_common_rbf']}")

    print(f"\nクラス別認識率:")
    for class_idx in sorted(class_accuracies.keys()):
        stats = class_accuracies[class_idx]
        char = class_info[class_idx]['char']
        print(f"  クラス {class_idx:2d} ({char}): {stats['accuracy']:6.2f}% "
              f"({stats['correct']}/{stats['total']})")

    # numpy型をPython標準型に変換
    class_accuracies_json = {}
    for k, v in class_accuracies.items():
        class_accuracies_json[int(k)] = {
            'accuracy': float(v['accuracy']),
            'correct': int(v['correct']),
            'total': int(v['total'])
        }

    return {
        'method': 'Hierarchical',
        'accuracy': float(accuracy),
        'correct': int(correct),
        'total': int(len(test_labels)),
        'train_time': float(train_time),
        'eval_time': float(eval_time),
        'avg_prediction_time': float(eval_time / len(test_images)),
        'rbf_stats': {
            'total_rbf': int(rbf_stats['total_rbf']),
            'common_rbf': int(rbf_stats['common_rbf']),
            'non_common_rbf_per_group': [int(x) for x in rbf_stats['non_common_rbf_per_group']],
            'rbf_per_class': [int(x) for x in rbf_stats['rbf_per_class']],
            'total_non_common_rbf': int(rbf_stats['total_non_common_rbf'])
        },
        'class_accuracies': class_accuracies_json,
        'confusion_matrix': [[int(x) for x in row] for row in cm.tolist()],
        'predictions': [int(x) for x in predictions],
        'targets': [int(x) for x in test_labels],
        'num_train_samples': int(len(train_images)),
        'num_test_samples': int(len(test_images))
    }

def experiment_mlp(images, labels, class_info, epochs=100, batch_size=32):
    """MLPの実験"""
    print("\n" + "="*80)
    print("MLP実験")
    print("="*80)

    # 訓練・テスト分割（階層型と同じseed）
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"訓練データ: {len(train_images)}サンプル")
    print(f"テストデータ: {len(test_images)}サンプル")

    # データセット作成
    train_dataset = ETL8BDataset(train_images, train_labels)
    test_dataset = ETL8BDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # モデル作成
    num_classes = len(class_info)
    model = MLP(input_size=1024, hidden_sizes=[512, 256, 128],
                num_classes=num_classes, dropout=0.3)

    trainer = MLPTrainer(model, device)

    # 学習
    print("\n学習開始...")
    train_results = trainer.train(train_loader, test_loader, epochs=epochs, lr=0.001)

    # 評価
    print("\n評価開始...")
    test_accuracy, predictions, targets, inference_time = trainer.evaluate(test_loader)

    # クラス別精度
    predictions_np = np.array(predictions)
    targets_np = np.array(targets)
    class_accuracies = {}
    for class_idx in range(num_classes):
        mask = targets_np == class_idx
        if np.sum(mask) > 0:
            class_acc = np.mean(predictions_np[mask] == targets_np[mask]) * 100
            class_accuracies[class_idx] = {
                'accuracy': class_acc,
                'correct': int(np.sum(predictions_np[mask] == targets_np[mask])),
                'total': int(np.sum(mask))
            }

    # 混同行列
    cm = confusion_matrix(targets, predictions)

    # 結果表示
    print(f"\n{'='*80}")
    print(f"MLP 結果")
    print(f"{'='*80}")
    print(f"総合認識率: {test_accuracy:.2f}%")
    print(f"学習時間: {train_results['training_time']:.4f}秒")
    print(f"評価時間: {inference_time:.4f}秒")
    print(f"平均予測時間: {inference_time/len(test_images)*1000:.2f}ミリ秒/画像")
    print(f"モデルパラメータ数: {model.count_parameters():,}")

    print(f"\nクラス別認識率:")
    for class_idx in sorted(class_accuracies.keys()):
        stats = class_accuracies[class_idx]
        char = class_info[class_idx]['char']
        print(f"  クラス {class_idx:2d} ({char}): {stats['accuracy']:6.2f}% "
              f"({stats['correct']}/{stats['total']})")

    # numpy型をPython標準型に変換
    class_accuracies_json = {}
    for k, v in class_accuracies.items():
        class_accuracies_json[int(k)] = {
            'accuracy': float(v['accuracy']),
            'correct': int(v['correct']),
            'total': int(v['total'])
        }

    return {
        'method': 'MLP',
        'accuracy': float(test_accuracy),
        'train_time': float(train_results['training_time']),
        'eval_time': float(inference_time),
        'avg_prediction_time': float(inference_time / len(test_images)),
        'model_parameters': int(model.count_parameters()),
        'class_accuracies': class_accuracies_json,
        'confusion_matrix': [[int(x) for x in row] for row in cm.tolist()],
        'predictions': [int(x) for x in predictions],
        'targets': [int(x) for x in targets],
        'num_train_samples': int(len(train_images)),
        'num_test_samples': int(len(test_images))
    }

def visualize_comparison(hierarchical_results, mlp_results, class_info, output_dir):
    """比較結果の可視化"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 総合精度比較
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    methods = ['Hierarchical\nRecognizer', 'MLP']
    accuracies = [hierarchical_results['accuracy'], mlp_results['accuracy']]
    colors = ['#2ecc71', '#3498db']

    bars = ax.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Overall Recognition Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # 値を表示
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. クラス別精度比較
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    class_indices = sorted(class_info.keys())
    hierarchical_accs = [hierarchical_results['class_accuracies'][i]['accuracy'] for i in class_indices]
    mlp_accs = [mlp_results['class_accuracies'][i]['accuracy'] for i in class_indices]

    x = np.arange(len(class_indices))
    width = 0.35

    bars1 = ax.bar(x - width/2, hierarchical_accs, width, label='Hierarchical',
                   color='#2ecc71', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, mlp_accs, width, label='MLP',
                   color='#3498db', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Recognition Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n{class_info[i]['char']}" for i in class_indices], fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 混同行列比較
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    cm_h = np.array(hierarchical_results['confusion_matrix'])
    cm_m = np.array(mlp_results['confusion_matrix'])

    # 階層型
    sns.heatmap(cm_h, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                cbar_kws={'label': 'Count'}, square=True)
    axes[0].set_title('Hierarchical Recognizer\nConfusion Matrix', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)

    # MLP
    sns.heatmap(cm_m, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                cbar_kws={'label': 'Count'}, square=True)
    axes[1].set_title('MLP\nConfusion Matrix', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. 時間・モデルサイズ比較
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 学習時間
    train_times = [hierarchical_results['train_time'], mlp_results['train_time']]
    bars = axes[0].bar(methods, train_times, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Time (seconds)', fontsize=11)
    axes[0].set_title('Training Time', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for bar, time_val in zip(bars, train_times):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s', ha='center', va='bottom', fontsize=10)

    # 評価時間
    eval_times = [hierarchical_results['eval_time'], mlp_results['eval_time']]
    bars = axes[1].bar(methods, eval_times, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Time (seconds)', fontsize=11)
    axes[1].set_title('Evaluation Time', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    for bar, time_val in zip(bars, eval_times):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.4f}s', ha='center', va='bottom', fontsize=10)

    # モデルサイズ
    model_sizes = [hierarchical_results['rbf_stats']['total_rbf'],
                   mlp_results['model_parameters']]
    size_labels = ['RBFs', 'Parameters']
    bars = axes[2].bar(methods, model_sizes, color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Count', fontsize=11)
    axes[2].set_title('Model Size', fontsize=12, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(axis='y', alpha=0.3)
    for bar, size_val, label in zip(bars, model_sizes, size_labels):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{size_val:,}\n{label}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_size_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n可視化完了: {output_dir}")

def generate_report(hierarchical_results, mlp_results, class_info, output_path):
    """比較レポート生成"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" 階層型パターン認識器 vs MLP - 15クラス比較実験レポート\n")
        f.write("="*80 + "\n\n")

        # 実験設定
        f.write("1. 実験設定\n")
        f.write("-"*80 + "\n")
        f.write(f"  対象クラス数: {len(class_info)}\n")
        f.write(f"  訓練データ数: {hierarchical_results['num_train_samples']}\n")
        f.write(f"  テストデータ数: {hierarchical_results['num_test_samples']}\n")
        f.write(f"  画像サイズ: 32x32\n\n")

        f.write("  対象クラス:\n")
        for idx in sorted(class_info.keys()):
            info = class_info[idx]
            f.write(f"    クラス {idx:2d} ({info['hex']}): {info['char']}\n")
        f.write("\n")

        # 総合結果
        f.write("2. 総合結果\n")
        f.write("-"*80 + "\n")
        f.write(f"{'手法':<25} {'認識率':<15} {'学習時間':<15} {'評価時間':<15}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'階層型パターン認識器':<20} {hierarchical_results['accuracy']:>6.2f}% "
                f"{hierarchical_results['train_time']:>10.4f}秒 "
                f"{hierarchical_results['eval_time']:>10.4f}秒\n")
        f.write(f"{'MLP':<20} {mlp_results['accuracy']:>6.2f}% "
                f"{mlp_results['train_time']:>10.4f}秒 "
                f"{mlp_results['eval_time']:>10.4f}秒\n")
        f.write("\n")

        # モデルサイズ
        f.write("3. モデルサイズ\n")
        f.write("-"*80 + "\n")
        f.write(f"  階層型パターン認識器:\n")
        f.write(f"    総RBF数: {hierarchical_results['rbf_stats']['total_rbf']}\n")
        f.write(f"    共通部RBF数: {hierarchical_results['rbf_stats']['common_rbf']}\n")
        f.write(f"    非共通部RBF数: {hierarchical_results['rbf_stats']['total_non_common_rbf']}\n\n")
        f.write(f"  MLP:\n")
        f.write(f"    パラメータ数: {mlp_results['model_parameters']:,}\n\n")

        # クラス別精度
        f.write("4. クラス別認識率\n")
        f.write("-"*80 + "\n")
        f.write(f"{'クラス':<10} {'文字':<8} {'階層型':<15} {'MLP':<15} {'差分':<10}\n")
        f.write("-"*80 + "\n")
        for idx in sorted(class_info.keys()):
            char = class_info[idx]['char']
            h_acc = hierarchical_results['class_accuracies'][idx]['accuracy']
            m_acc = mlp_results['class_accuracies'][idx]['accuracy']
            diff = h_acc - m_acc
            f.write(f"{idx:>2} ({class_info[idx]['hex']:<6}) {char:<5} "
                   f"{h_acc:>6.2f}% ({hierarchical_results['class_accuracies'][idx]['correct']}/{hierarchical_results['class_accuracies'][idx]['total']})  "
                   f"{m_acc:>6.2f}% ({mlp_results['class_accuracies'][idx]['correct']}/{mlp_results['class_accuracies'][idx]['total']})  "
                   f"{diff:>+6.2f}%\n")
        f.write("\n")

        # 比較分析
        f.write("5. 比較分析\n")
        f.write("-"*80 + "\n")

        acc_diff = hierarchical_results['accuracy'] - mlp_results['accuracy']
        if acc_diff > 0:
            f.write(f"  ✓ 階層型パターン認識器が {acc_diff:.2f}% 高精度\n")
        else:
            f.write(f"  ✓ MLPが {-acc_diff:.2f}% 高精度\n")

        time_ratio = mlp_results['train_time'] / hierarchical_results['train_time']
        f.write(f"  ✓ 階層型の学習時間はMLPの約{time_ratio:.0f}倍高速（{hierarchical_results['train_time']:.4f}秒 vs {mlp_results['train_time']:.4f}秒）\n")

        size_ratio = mlp_results['model_parameters'] / hierarchical_results['rbf_stats']['total_rbf']
        f.write(f"  ✓ 階層型のモデルサイズはMLPの約{size_ratio:.0f}倍コンパクト（{hierarchical_results['rbf_stats']['total_rbf']} RBFs vs {mlp_results['model_parameters']:,}パラメータ）\n")

        f.write("\n")
        f.write("6. 生成ファイル\n")
        f.write("-"*80 + "\n")
        f.write("  - accuracy_comparison.png: 総合精度比較\n")
        f.write("  - per_class_accuracy_comparison.png: クラス別精度比較\n")
        f.write("  - confusion_matrix_comparison.png: 混同行列比較\n")
        f.write("  - time_size_comparison.png: 時間・モデルサイズ比較\n")
        f.write("  - comparison_report.txt: このレポート\n")
        f.write("  - hierarchical_results.json: 階層型の詳細結果\n")
        f.write("  - mlp_results.json: MLPの詳細結果\n")
        f.write("\n")

    print(f"\nレポート生成完了: {output_path}")

def main():
    """メイン実行関数"""
    print("="*80)
    print("階層型パターン認識器 vs MLP 比較実験 (15クラス)")
    print("="*80)

    # 15クラス設定（左右分離型漢字）
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

    # データ読み込み
    base_path = "./ETL8B-img-full"
    max_samples = 30

    images, labels, class_info = load_etl8b_data(base_path, target_classes, max_samples)

    # 階層型パターン認識器実験
    hierarchical_results = experiment_hierarchical(images, labels, class_info)

    # MLP実験
    mlp_results = experiment_mlp(images, labels, class_info, epochs=100, batch_size=32)

    # 結果保存
    output_dir = "./results/comparison"
    os.makedirs(output_dir, exist_ok=True)

    # JSON保存
    with open(os.path.join(output_dir, 'hierarchical_results.json'), 'w', encoding='utf-8') as f:
        json.dump(hierarchical_results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'mlp_results.json'), 'w', encoding='utf-8') as f:
        json.dump(mlp_results, f, indent=2, ensure_ascii=False)

    # 可視化
    visualize_comparison(hierarchical_results, mlp_results, class_info, output_dir)

    # レポート生成
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    generate_report(hierarchical_results, mlp_results, class_info, report_path)

    print("\n" + "="*80)
    print("比較実験完了！")
    print("="*80)
    print(f"結果は {output_dir}/ に保存されました")

if __name__ == "__main__":
    main()
