#!/usr/bin/env python3
"""
analyze_etl8b_15classes.py
ETL8B 15クラス実験の詳細分析
- RBF数の統計
- 処理時間の詳細計測
- 共通部/非共通部の可視化

認識率（15クラス、各30サンプル、32x32画像、135テスト）:
  階層型パターン認識器: 77.04%（総RBF数129、学習0.011秒）
"""

import sys
import numpy as np
import cv2
import time
import matplotlib
matplotlib.use('Agg')  # GUIなし環境用
import matplotlib.pyplot as plt
import seaborn as sns
from hierarchical_recognizer import (
    HierarchicalPatternRecognizer,
    load_etl8b_data,
    ClassNumMethod,
    RadiusMethod,
    WeightUpdateMethod
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import ctypes as ct

class RBFStatistics(ct.Structure):
    """RBF統計情報の構造体"""
    _fields_ = [
        ('total_rbf_count', ct.c_int),
        ('common_rbf_count', ct.c_int),
        ('non_common_rbf_counts', ct.POINTER(ct.c_int)),
        ('num_groups', ct.c_int),
        ('class_rbf_counts', ct.POINTER(ct.c_int)),
        ('num_classes', ct.c_int)
    ]

def get_rbf_statistics(recognizer):
    """RBF統計を取得"""
    if recognizer.c_recognizer is None:
        return None

    # C++関数の定義
    recognizer.lib.getRBFStatistics.argtypes = [ct.c_void_p]
    recognizer.lib.getRBFStatistics.restype = ct.POINTER(RBFStatistics)
    recognizer.lib.freeRBFStatistics.argtypes = [ct.POINTER(RBFStatistics)]
    recognizer.lib.freeRBFStatistics.restype = None

    # 統計取得
    stats_ptr = recognizer.lib.getRBFStatistics(recognizer.c_recognizer)
    stats = stats_ptr.contents

    # Python辞書に変換
    result = {
        'total_rbf': stats.total_rbf_count,
        'common_rbf': stats.common_rbf_count,
        'non_common_rbf_per_group': [stats.non_common_rbf_counts[i]
                                      for i in range(stats.num_groups)],
        'rbf_per_class': [stats.class_rbf_counts[i]
                          for i in range(stats.num_classes)]
    }

    # メモリ解放
    recognizer.lib.freeRBFStatistics(stats_ptr)

    return result

def visualize_pyramid(img, num_levels=5, save_path='pyramid.png'):
    """階層ピラミッド構造を可視化"""
    fig, axes = plt.subplots(1, num_levels, figsize=(15, 3))
    fig.suptitle('Hierarchical Pyramid Structure', fontsize=16)

    current = img.copy()
    for level in range(num_levels - 1, -1, -1):
        ax = axes[num_levels - 1 - level]
        ax.imshow(current, cmap='gray')
        ax.set_title(f'Level {level}\n{current.shape[0]}x{current.shape[1]}')
        ax.axis('off')

        if level > 0:
            # ダウンサンプリング
            h, w = current.shape
            new_h, new_w = h // 2, w // 2
            if new_h < 1: new_h = 1
            if new_w < 1: new_w = 1
            current = cv2.resize(current, (new_w, new_h), interpolation=cv2.INTER_AREA)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ピラミッド可視化を保存: {save_path}")

def visualize_rbf_distribution(rbf_stats, class_info, save_path='rbf_distribution.png'):
    """RBF分布を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 総RBF数
    ax = axes[0, 0]
    categories = ['Common Part', 'Non-Common Parts']
    values = [rbf_stats['common_rbf'],
              rbf_stats['total_rbf'] - rbf_stats['common_rbf']]
    ax.bar(categories, values, color=['skyblue', 'lightcoral'])
    ax.set_title('Total RBF Distribution')
    ax.set_ylabel('Number of RBFs')
    for i, v in enumerate(values):
        ax.text(i, v + max(values)*0.02, str(v), ha='center', fontsize=12, fontweight='bold')

    # 2. グループごとのRBF数
    ax = axes[0, 1]
    groups = [f'Group {i}' for i in range(len(rbf_stats['non_common_rbf_per_group']))]
    values = rbf_stats['non_common_rbf_per_group']
    ax.bar(groups, values, color='lightgreen')
    ax.set_title('RBFs per Common Group')
    ax.set_ylabel('Number of RBFs')
    ax.set_xlabel('Common Group')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. クラスごとのRBF数
    ax = axes[1, 0]
    class_ids = sorted(class_info.keys())
    class_labels = [f'{class_info[i]}' for i in class_ids]
    values = [rbf_stats['rbf_per_class'][i] for i in class_ids]

    colors = plt.cm.tab20(np.linspace(0, 1, len(class_ids)))
    bars = ax.bar(range(len(class_ids)), values, color=colors)
    ax.set_title('RBFs per Class')
    ax.set_ylabel('Number of RBFs')
    ax.set_xlabel('Class')
    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels(class_labels, rotation=45, ha='right')

    # 4. 統計情報テキスト
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"""
RBF Statistics Summary

Total RBFs: {rbf_stats['total_rbf']}
Common Part RBFs: {rbf_stats['common_rbf']}
Non-Common RBFs: {rbf_stats['total_rbf'] - rbf_stats['common_rbf']}

Average RBFs per Class: {np.mean(rbf_stats['rbf_per_class']):.1f}
Max RBFs in a Class: {np.max(rbf_stats['rbf_per_class'])}
Min RBFs in a Class: {np.min(rbf_stats['rbf_per_class'])}
Std Dev: {np.std(rbf_stats['rbf_per_class']):.2f}
"""
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  RBF分布を保存: {save_path}")

def visualize_confusion_matrix(cm, class_info, save_path='confusion_matrix.png'):
    """混同行列のヒートマップ"""
    plt.figure(figsize=(12, 10))

    class_labels = [class_info[i] for i in sorted(class_info.keys())]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix - 15 Classes', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  混同行列を保存: {save_path}")

def visualize_sample_images(images, labels, class_info, predictions=None,
                            save_path='sample_images.png', n_samples=15):
    """サンプル画像の可視化"""
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('Sample Images from 15 Classes', fontsize=16)

    # 各クラスから1サンプルずつ
    shown = set()
    idx = 0

    for i in range(len(images)):
        label = labels[i]
        if label not in shown and idx < 15:
            ax = axes[idx // 5, idx % 5]
            ax.imshow(images[i], cmap='gray')

            title = f'Class {label}: {class_info[label]}'
            if predictions is not None:
                pred = predictions[i]
                if pred == label:
                    title += '\n✓'
                    ax.add_patch(plt.Rectangle((0, 0), images[i].shape[1], images[i].shape[0],
                                              fill=False, edgecolor='green', linewidth=3))
                else:
                    title += f'\n✗ → {class_info[pred]}'
                    ax.add_patch(plt.Rectangle((0, 0), images[i].shape[1], images[i].shape[0],
                                              fill=False, edgecolor='red', linewidth=3))

            ax.set_title(title, fontsize=10)
            ax.axis('off')
            shown.add(label)
            idx += 1

            if idx >= 15:
                break

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  サンプル画像を保存: {save_path}")

def analyze_experiment(base_path='./ETL8B-img-full', max_samples=30, output_dir='./results/analysis'):
    """完全な実験分析"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print(" ETL8B 15クラス実験 - 詳細分析")
    print("=" * 100)
    print()

    # 15クラスの漢字
    target_classes = [
        '3a2c', '3a2e', '3a4e', '3a59', '3a6e',
        '3c23', '3c72', '3c78', '3d26', '3d3b',
        '3d3e', '436d', '3b26', '3b33', '3b45'
    ]

    print(f"対象クラス数: {len(target_classes)}")
    print(f"サンプル数/クラス: {max_samples}")
    print(f"出力ディレクトリ: {output_dir}")
    print()

    # データ読み込み
    print("-" * 100)
    print("[1/6] データ読み込み中...")
    print("-" * 100)

    time_start = time.time()
    images, labels, class_info = load_etl8b_data(
        base_path, target_classes, max_samples
    )
    time_load = time.time() - time_start

    print(f"  読み込み時間: {time_load:.3f}秒")
    print(f"  総サンプル数: {len(images)}")
    print()

    # 訓練・テスト分割
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"  訓練データ: {len(train_images)}サンプル")
    print(f"  テストデータ: {len(test_images)}サンプル")
    print()

    # 認識器の構築
    print("-" * 100)
    print("[2/6] 階層構造的認識器の学習中...")
    print("-" * 100)

    recognizer = HierarchicalPatternRecognizer(
        num_pyramid_levels=5,
        target_image_size=32,
        class_num_method=ClassNumMethod.INCREMENTAL,
        radius_method=RadiusMethod.METHOD1,
        weight_update_method=WeightUpdateMethod.NO_UPDATE
    )

    common_labels = train_labels.copy()

    time_start = time.time()
    recognizer.train(train_images, train_labels, common_labels)
    time_train = time.time() - time_start

    print(f"  学習時間: {time_train:.3f}秒")
    print()

    # RBF統計の取得
    print("-" * 100)
    print("[3/6] RBF統計の取得中...")
    print("-" * 100)

    rbf_stats = get_rbf_statistics(recognizer)

    if rbf_stats:
        print(f"  総RBF数: {rbf_stats['total_rbf']}")
        print(f"  共通部RBF数: {rbf_stats['common_rbf']}")
        print(f"  非共通部RBF数: {rbf_stats['total_rbf'] - rbf_stats['common_rbf']}")
        print()

        print("  グループごとのRBF数:")
        for i, count in enumerate(rbf_stats['non_common_rbf_per_group']):
            print(f"    グループ {i}: {count}個")
        print()

        print("  クラスごとのRBF数:")
        for class_id in sorted(class_info.keys()):
            rbf_count = rbf_stats['rbf_per_class'][class_id]
            print(f"    クラス {class_id} ({class_info[class_id]}): {rbf_count}個")
        print()

    # 評価
    print("-" * 100)
    print("[4/6] モデルの評価中...")
    print("-" * 100)

    time_start = time.time()
    accuracy, predictions = recognizer.evaluate(test_images, test_labels)
    time_eval = time.time() - time_start

    print(f"  評価時間: {time_eval:.3f}秒")
    print(f"  平均予測時間: {(time_eval/len(test_images))*1000:.2f}ミリ秒/画像")
    print(f"  認識率: {accuracy:.2f}%")
    print()

    # 混同行列
    cm = confusion_matrix(test_labels, predictions)

    # 可視化
    print("-" * 100)
    print("[5/6] 可視化の生成中...")
    print("-" * 100)

    # ピラミッド構造の可視化（最初のサンプル）
    visualize_pyramid(test_images[0] if len(test_images) > 0 else images[0],
                     num_levels=5,
                     save_path=f'{output_dir}/pyramid_structure.png')

    # RBF分布の可視化
    if rbf_stats:
        visualize_rbf_distribution(rbf_stats, class_info,
                                  save_path=f'{output_dir}/rbf_distribution.png')

    # 混同行列の可視化
    visualize_confusion_matrix(cm, class_info,
                              save_path=f'{output_dir}/confusion_matrix.png')

    # サンプル画像の可視化
    visualize_sample_images(test_images, test_labels, class_info, predictions,
                           save_path=f'{output_dir}/sample_images.png')

    print()

    # レポート作成
    print("-" * 100)
    print("[6/6] レポート作成中...")
    print("-" * 100)

    report_path = f'{output_dir}/analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(" ETL8B 15クラス実験 - 詳細分析レポート\n")
        f.write("=" * 100 + "\n\n")

        f.write("1. 実験設定\n")
        f.write("-" * 100 + "\n")
        f.write(f"  対象クラス数: {len(target_classes)}\n")
        f.write(f"  サンプル数/クラス: {max_samples}\n")
        f.write(f"  総サンプル数: {len(images)}\n")
        f.write(f"  訓練データ: {len(train_images)}サンプル\n")
        f.write(f"  テストデータ: {len(test_images)}サンプル\n")
        f.write(f"  ピラミッド階層数: 5\n")
        f.write(f"  画像サイズ: 32x32\n")
        f.write("\n")

        f.write("2. 処理時間\n")
        f.write("-" * 100 + "\n")
        f.write(f"  データ読み込み: {time_load:.3f}秒\n")
        f.write(f"  学習時間: {time_train:.3f}秒\n")
        f.write(f"  評価時間: {time_eval:.3f}秒\n")
        f.write(f"  平均予測時間: {(time_eval/len(test_images))*1000:.2f}ミリ秒/画像\n")
        f.write("\n")

        if rbf_stats:
            f.write("3. RBF統計\n")
            f.write("-" * 100 + "\n")
            f.write(f"  総RBF数: {rbf_stats['total_rbf']}\n")
            f.write(f"  共通部RBF数: {rbf_stats['common_rbf']}\n")
            f.write(f"  非共通部RBF数: {rbf_stats['total_rbf'] - rbf_stats['common_rbf']}\n")
            f.write(f"  平均RBF数/クラス: {np.mean(rbf_stats['rbf_per_class']):.1f}\n")
            f.write(f"  最大RBF数: {np.max(rbf_stats['rbf_per_class'])}\n")
            f.write(f"  最小RBF数: {np.min(rbf_stats['rbf_per_class'])}\n")
            f.write("\n")

            f.write("  グループごとのRBF数:\n")
            for i, count in enumerate(rbf_stats['non_common_rbf_per_group']):
                f.write(f"    グループ {i}: {count}個\n")
            f.write("\n")

            f.write("  クラスごとのRBF数:\n")
            for class_id in sorted(class_info.keys()):
                rbf_count = rbf_stats['rbf_per_class'][class_id]
                f.write(f"    クラス {class_id} ({class_info[class_id]}): {rbf_count}個\n")
            f.write("\n")

        f.write("4. 認識結果\n")
        f.write("-" * 100 + "\n")
        f.write(f"  総合認識率: {accuracy:.2f}%\n")
        f.write(f"  正解数: {np.sum(predictions == test_labels)}/{len(test_labels)}\n")
        f.write("\n")

        f.write("  クラス別認識率:\n")
        for class_id in sorted(class_info.keys()):
            mask = test_labels == class_id
            if np.sum(mask) > 0:
                class_correct = np.sum(predictions[mask] == class_id)
                class_total = np.sum(mask)
                class_acc = 100.0 * class_correct / class_total
                f.write(f"    クラス {class_id} ({class_info[class_id]}): "
                       f"{class_acc:.2f}% ({class_correct}/{class_total})\n")
        f.write("\n")

        f.write("5. 混同行列\n")
        f.write("-" * 100 + "\n")
        f.write(str(cm) + "\n")
        f.write("\n")

        f.write("6. 生成ファイル\n")
        f.write("-" * 100 + "\n")
        f.write(f"  - pyramid_structure.png: ピラミッド階層構造の可視化\n")
        f.write(f"  - rbf_distribution.png: RBF数の分布\n")
        f.write(f"  - confusion_matrix.png: 混同行列ヒートマップ\n")
        f.write(f"  - sample_images.png: サンプル画像\n")
        f.write(f"  - analysis_report.txt: このレポート\n")
        f.write("\n")

    print(f"  レポートを保存: {report_path}")
    print()

    print("=" * 100)
    print(" 分析完了")
    print("=" * 100)
    print(f"\n結果は {output_dir}/ に保存されました。\n")

    return {
        'accuracy': accuracy,
        'rbf_stats': rbf_stats,
        'time_load': time_load,
        'time_train': time_train,
        'time_eval': time_eval
    }

def main():
    """メイン関数"""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("使用方法:")
        print("  python analyze_etl8b_15classes.py [base_path] [max_samples] [output_dir]")
        print()
        print("例:")
        print("  python analyze_etl8b_15classes.py")
        print("  python analyze_etl8b_15classes.py ./ETL8B-img-full 30 ./results")
        sys.exit(0)

    base_path = sys.argv[1] if len(sys.argv) > 1 else './ETL8B-img-full'
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    output_dir = sys.argv[3] if len(sys.argv) > 3 else './results/analysis'

    try:
        analyze_experiment(base_path, max_samples, output_dir)
    except Exception as e:
        print(f"\nエラー: {e}")
        print("\nC++ライブラリのコンパイルを確認してください:")
        print("  make")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
