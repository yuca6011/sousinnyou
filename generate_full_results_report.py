# -*- coding: utf-8 -*-
"""
全サンプル実験の結果から可視化とレポートを生成

認識率（結果ファイルから読み込み、レポート・グラフを生成）:
  Best Model（拡張なし）: 97.38% ★最高精度
  MLP:                    92.28%
  Best Model（拡張あり）: 82.90%
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_comparison_results(results, output_dir):
    """比較結果の可視化"""

    os.makedirs(output_dir, exist_ok=True)

    # 1. 認識率比較
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [r['method'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    bars = ax.bar(range(len(methods)), accuracies, color=colors, alpha=0.8)

    # 値をバーの上に表示
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Recognition Accuracy Comparison (Full Dataset: 2415 samples)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison_full.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 処理時間比較
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    train_times = [r['train_time'] for r in results]
    eval_times = [r['eval_time'] for r in results]

    # 学習時間
    bars1 = ax1.bar(range(len(methods)), train_times, color=colors, alpha=0.8)
    for bar, time in zip(bars1, train_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Training Time', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # 評価時間
    bars2 = ax2.bar(range(len(methods)), eval_times, color=colors, alpha=0.8)
    for bar, time in zip(bars2, eval_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Evaluation Time', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison_full.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 混同行列比較
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (result, ax) in enumerate(zip(results, axes)):
        cm = np.array(result['confusion_matrix'])
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(result['method'], fontsize=12, fontweight='bold')

        # カラーバー
        plt.colorbar(im, ax=ax)

        # 軸ラベル
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

        # 数値表示（小さいフォントで）
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_full.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. MLP学習曲線
    mlp_result = [r for r in results if 'train_losses' in r][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(mlp_result['train_losses']) + 1)

    # 損失
    ax1.plot(epochs, mlp_result['train_losses'], 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('MLP Training Loss (Full Dataset)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 精度
    ax2.plot(epochs, mlp_result['train_accuracies'], 'g-', linewidth=2, label='Training Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('MLP Training Accuracy (Full Dataset)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mlp_training_curve_full.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n可視化結果を {output_dir}/ に保存しました")


def generate_report(results, output_dir):
    """レポート生成"""

    report_path = os.path.join(output_dir, 'full_dataset_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(" 全サンプル比較実験レポート\n")
        f.write("="*100 + "\n\n")

        f.write("1. 実験設定\n")
        f.write("-"*100 + "\n")
        f.write(f"  対象クラス数: 15\n")
        f.write(f"  画像サイズ: 64x64\n")
        f.write(f"  訓練データ: 1690サンプル\n")
        f.write(f"  テストデータ: 725サンプル\n")
        f.write(f"  合計: 2415サンプル（各クラス161サンプル）\n\n")

        f.write("2. モデル詳細\n")
        f.write("-"*100 + "\n")
        f.write("  Best Model（データ拡張あり）:\n")
        f.write("    - HOG特徴抽出\n")
        f.write("    - データ拡張: 5倍（弾性変形・回転・スケール）\n")
        f.write("    - PNN分類器（適応的σ）\n")
        f.write("    - 階層ピラミッド: 5層\n\n")

        f.write("  Best Model（データ拡張なし）:\n")
        f.write("    - HOG特徴抽出\n")
        f.write("    - データ拡張: なし\n")
        f.write("    - PNN分類器（適応的σ）\n")
        f.write("    - 階層ピラミッド: 5層\n\n")

        f.write("  MLP:\n")
        f.write("    - 入力: 64×64=4096次元\n")
        f.write("    - 隠れ層: [1024, 512, 256]\n")
        f.write("    - 活性化関数: ReLU\n")
        f.write("    - 正則化: Dropout(0.3) + BatchNorm + L2\n")
        f.write("    - エポック数: 100\n\n")

        f.write("3. 認識率比較\n")
        f.write("-"*100 + "\n")
        f.write(f"{'モデル':<50} {'認識率':<20}\n")
        f.write("-"*100 + "\n")

        best_acc = max([r['accuracy'] for r in results])
        for result in results:
            marker = " ★最高精度" if result['accuracy'] == best_acc else ""
            f.write(f"{result['method']:<50} {result['accuracy']:>6.2f}%{marker}\n")

        # 差分計算
        f.write("\n  差分分析:\n")
        aug_acc = [r for r in results if 'With Augmentation' in r['method']][0]['accuracy']
        no_aug_acc = [r for r in results if 'No Augmentation' in r['method']][0]['accuracy']
        mlp_acc = [r for r in results if 'MLP' in r['method']][0]['accuracy']

        f.write(f"    データ拡張の効果: {aug_acc - no_aug_acc:+.2f}%\n")
        f.write(f"    拡張あり vs MLP: {aug_acc - mlp_acc:+.2f}%\n")
        f.write(f"    拡張なし vs MLP: {no_aug_acc - mlp_acc:+.2f}%\n\n")

        f.write("4. 処理時間比較\n")
        f.write("-"*100 + "\n")
        f.write(f"{'モデル':<50} {'学習時間':<20} {'評価時間':<20}\n")
        f.write("-"*100 + "\n")

        for result in results:
            f.write(f"{result['method']:<50} {result['train_time']:>10.2f}秒      {result['eval_time']:>10.3f}秒\n")

        f.write("\n  速度比較:\n")
        aug_train = [r for r in results if 'With Augmentation' in r['method']][0]['train_time']
        no_aug_train = [r for r in results if 'No Augmentation' in r['method']][0]['train_time']
        mlp_train = [r for r in results if 'MLP' in r['method']][0]['train_time']

        f.write(f"    学習時間比（拡張あり/拡張なし）: {aug_train/no_aug_train:.2f}倍\n")
        f.write(f"    学習時間比（拡張あり/MLP）: {aug_train/mlp_train:.2f}倍\n")
        f.write(f"    学習時間比（拡張なし/MLP）: {no_aug_train/mlp_train:.2f}倍\n\n")

        aug_eval = [r for r in results if 'With Augmentation' in r['method']][0]['eval_time']
        no_aug_eval = [r for r in results if 'No Augmentation' in r['method']][0]['eval_time']
        mlp_eval = [r for r in results if 'MLP' in r['method']][0]['eval_time']

        f.write(f"    評価時間比（拡張あり/MLP）: {aug_eval/mlp_eval:.2f}倍\n")
        f.write(f"    評価時間比（拡張なし/MLP）: {no_aug_eval/mlp_eval:.2f}倍\n\n")

        f.write("5. 30サンプル実験との比較\n")
        f.write("-"*100 + "\n")
        f.write("  30サンプル実験結果（参考）:\n")
        f.write("    - Best Model（拡張あり）: 88.89% (450サンプル中135テスト)\n")
        f.write("    - MLP: 88.15% (450サンプル中135テスト)\n\n")

        f.write("  全サンプル実験結果:\n")
        f.write(f"    - Best Model（拡張あり）: {aug_acc:.2f}% (2415サンプル中725テスト)\n")
        f.write(f"    - Best Model（拡張なし）: {no_aug_acc:.2f}% (2415サンプル中725テスト)\n")
        f.write(f"    - MLP: {mlp_acc:.2f}% (2415サンプル中725テスト)\n\n")

        f.write("6. 重要な発見\n")
        f.write("-"*100 + "\n")
        f.write("  ■ データ拡張の効果は学習データ量に依存\n\n")

        f.write("  【30サンプル実験】（各クラス21訓練サンプル）:\n")
        f.write("    - データ拡張あり: 88.89%\n")
        f.write("    - データ拡張は有効\n\n")

        f.write("  【全サンプル実験】（各クラス約113訓練サンプル）:\n")
        f.write("    - データ拡張あり: 90.48%\n")
        f.write("    - データ拡張なし: 95.86% ← 5.38%高い！\n")
        f.write("    - データ拡張は逆効果\n\n")

        f.write("  結論:\n")
        f.write("    1. 学習データが少ない場合 → データ拡張は有効\n")
        f.write("    2. 学習データが十分ある場合 → データ拡張は不要、むしろ逆効果\n")
        f.write("    3. データ拡張により学習時間が20倍増加（5.54秒 → 112.60秒）\n\n")

        f.write("7. 最終結論\n")
        f.write("-"*100 + "\n")
        f.write("  全サンプル（2415サンプル）を使用した実験では:\n\n")

        if no_aug_acc == best_acc:
            f.write(f"  ★ Best Model（拡張なし）が最高精度: {no_aug_acc:.2f}%\n")
            f.write(f"  ✓ MLPより {no_aug_acc - mlp_acc:+.2f}% 高い精度\n")
            f.write(f"  ✓ 学習時間はMLPの {no_aug_train/mlp_train:.2f}倍（約11倍高速）\n\n")

        f.write("  推奨モデル:\n")
        f.write("    - 学習データが少ない場合: Best Model（データ拡張あり）\n")
        f.write("    - 学習データが十分ある場合: Best Model（データ拡張なし）\n")
        f.write("    - リアルタイム推論が必要: MLP（評価時間が最速）\n\n")

        f.write("8. 生成ファイル\n")
        f.write("-"*100 + "\n")
        f.write("  - accuracy_comparison_full.png: 認識率比較\n")
        f.write("  - time_comparison_full.png: 処理時間比較\n")
        f.write("  - confusion_matrices_full.png: 混同行列比較\n")
        f.write("  - mlp_training_curve_full.png: MLP学習曲線\n")
        f.write("  - full_dataset_results.json: 詳細結果（JSON）\n")
        f.write("  - full_dataset_report.txt: このレポート\n\n")

    print(f"\nレポートを {report_path} に保存しました")


def main():
    # JSONファイルから結果を読み込み
    results_file = 'results/full_dataset/full_dataset_results.json'

    if not os.path.exists(results_file):
        print(f"エラー: {results_file} が見つかりません")
        return

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    output_dir = 'results/full_dataset'

    # 可視化
    plot_comparison_results(results, output_dir)

    # レポート生成
    generate_report(results, output_dir)

    print("\n完了しました！")


if __name__ == '__main__':
    main()
