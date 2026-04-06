#!/usr/bin/env python3
"""
test_etl8b_15classes.py
ETL8B漢字データを使用した15クラス実験

認識率（15クラス、各30サンプル、32x32画像、135テスト）:
  複数設定を比較（結果はresults/に出力）
"""

import sys
import numpy as np
import time
from hierarchical_recognizer import (
    HierarchicalPatternRecognizer,
    load_etl8b_data,
    ClassNumMethod,
    RadiusMethod,
    WeightUpdateMethod
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def print_results_table(results):
    """結果をテーブル形式で表示"""
    print("\n" + "=" * 100)
    print(" 実験結果サマリー")
    print("=" * 100)
    print(f"{'設定名':<40} {'認識率':>10} {'学習時間':>12} {'予測時間':>12}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<40} {r['accuracy']:>9.2f}% {r['train_time']:>10.2f}s {r['pred_time']:>10.4f}s")
    print("=" * 100)

def experiment_basic(base_path='./ETL8B-img-full', max_samples=30):
    """
    実験1: 基本テスト (15クラス、30サンプル/クラス)
    """
    print("=" * 100)
    print(" 実験1: 基本テスト (15クラス)")
    print("=" * 100)
    print()

    # 15クラスの漢字を選択
    target_classes = [
        '3a2c', '3a2e', '3a4e', '3a59', '3a6e',  # 5クラス
        '3c23', '3c72', '3c78', '3d26', '3d3b',  # 5クラス
        '3d3e', '436d', '3b26', '3b33', '3b45'   # 5クラス
    ]

    print(f"対象クラス数: {len(target_classes)}")
    print(f"サンプル数/クラス: {max_samples}")
    print()

    # データ読み込み
    images, labels, class_info = load_etl8b_data(
        base_path, target_classes, max_samples
    )

    if len(images) == 0:
        print("エラー: データが読み込めませんでした")
        return None

    # 訓練・テスト分割 (70%/30%)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"訓練データ: {len(train_images)}サンプル")
    print(f"テストデータ: {len(test_images)}サンプル")
    print()

    # 認識器の構築
    recognizer = HierarchicalPatternRecognizer(
        num_pyramid_levels=5,
        target_image_size=32,
        class_num_method=ClassNumMethod.INCREMENTAL,
        radius_method=RadiusMethod.METHOD1,
        weight_update_method=WeightUpdateMethod.NO_UPDATE
    )

    # 学習
    common_labels = train_labels.copy()
    start_time = time.time()
    recognizer.train(train_images, train_labels, common_labels)
    train_time = time.time() - start_time

    # 評価
    start_time = time.time()
    accuracy, predictions = recognizer.evaluate(test_images, test_labels)
    pred_time = (time.time() - start_time) / len(test_images)

    # クラス別認識率
    print("\n" + "-" * 100)
    print("クラス別認識率:")
    print("-" * 100)
    for class_id, char in sorted(class_info.items()):
        mask = test_labels == class_id
        if np.sum(mask) > 0:
            class_correct = np.sum(predictions[mask] == class_id)
            class_total = np.sum(mask)
            class_acc = 100.0 * class_correct / class_total
            print(f"  クラス{class_id:2d} ({char}): {class_acc:6.2f}% ({class_correct}/{class_total})")

    # 混同行列
    print("\n" + "-" * 100)
    print("混同行列:")
    print("-" * 100)
    cm = confusion_matrix(test_labels, predictions)
    print(cm)

    print("\n" + "=" * 100)
    print(f"総合認識率: {accuracy:.2f}%")
    print(f"学習時間: {train_time:.2f}秒")
    print(f"平均予測時間: {pred_time*1000:.2f}ミリ秒/画像")
    print("=" * 100)

    return {
        'name': '基本設定 (INCREMENTAL + METHOD1)',
        'accuracy': accuracy,
        'train_time': train_time,
        'pred_time': pred_time,
        'predictions': predictions,
        'test_labels': test_labels,
        'class_info': class_info
    }

def experiment_comparison(base_path='./ETL8B-img-full', max_samples=30):
    """
    実験2: 複数設定での比較実験
    """
    print("\n" + "=" * 100)
    print(" 実験2: 複数設定での比較")
    print("=" * 100)
    print()

    # 15クラスの漢字
    target_classes = [
        '3a2c', '3a2e', '3a4e', '3a59', '3a6e',
        '3c23', '3c72', '3c78', '3d26', '3d3b',
        '3d3e', '436d', '3b26', '3b33', '3b45'
    ]

    # データ読み込み
    images, labels, class_info = load_etl8b_data(
        base_path, target_classes, max_samples
    )

    if len(images) == 0:
        print("エラー: データが読み込めませんでした")
        return []

    # 訓練・テスト分割
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"訓練データ: {len(train_images)}サンプル")
    print(f"テストデータ: {len(test_images)}サンプル")
    print()

    # 比較する設定
    configs = [
        {
            'name': '設定1: INCREMENTAL + METHOD1 + NO_UPDATE (5層)',
            'class_num': ClassNumMethod.INCREMENTAL,
            'radius': RadiusMethod.METHOD1,
            'update': WeightUpdateMethod.NO_UPDATE,
            'levels': 5,
            'size': 32
        },
        {
            'name': '設定2: FIXED + METHOD2 + AVERAGE (5層)',
            'class_num': ClassNumMethod.FIXED,
            'radius': RadiusMethod.METHOD2,
            'update': WeightUpdateMethod.AVERAGE,
            'levels': 5,
            'size': 32
        },
        {
            'name': '設定3: INCREMENTAL + METHOD1 + AVERAGE (7層)',
            'class_num': ClassNumMethod.INCREMENTAL,
            'radius': RadiusMethod.METHOD1,
            'update': WeightUpdateMethod.AVERAGE,
            'levels': 7,
            'size': 32
        },
        {
            'name': '設定4: INCREMENTAL + METHOD1 + NO_UPDATE (5層, 48x48)',
            'class_num': ClassNumMethod.INCREMENTAL,
            'radius': RadiusMethod.METHOD1,
            'update': WeightUpdateMethod.NO_UPDATE,
            'levels': 5,
            'size': 48
        }
    ]

    results = []

    for idx, config in enumerate(configs, 1):
        print(f"\n{'-' * 100}")
        print(f"[{idx}/{len(configs)}] {config['name']}")
        print('-' * 100)

        try:
            recognizer = HierarchicalPatternRecognizer(
                num_pyramid_levels=config['levels'],
                target_image_size=config['size'],
                class_num_method=config['class_num'],
                radius_method=config['radius'],
                weight_update_method=config['update']
            )

            common_labels = train_labels.copy()

            # 学習
            start_time = time.time()
            recognizer.train(train_images, train_labels, common_labels)
            train_time = time.time() - start_time

            # 評価
            start_time = time.time()
            accuracy, predictions = recognizer.evaluate(test_images, test_labels)
            pred_time = (time.time() - start_time) / len(test_images)

            results.append({
                'name': config['name'],
                'accuracy': accuracy,
                'train_time': train_time,
                'pred_time': pred_time
            })

            print(f"認識率: {accuracy:.2f}%")
            print(f"学習時間: {train_time:.2f}秒")
            print(f"平均予測時間: {pred_time*1000:.2f}ミリ秒/画像")

        except Exception as e:
            print(f"\nエラー: {e}")
            import traceback
            traceback.print_exc()

    return results

def experiment_sample_variation(base_path='./ETL8B-img-full'):
    """
    実験3: サンプル数変動テスト
    """
    print("\n" + "=" * 100)
    print(" 実験3: サンプル数変動テスト")
    print("=" * 100)
    print()

    target_classes = [
        '3a2c', '3a2e', '3a4e', '3a59', '3a6e',
        '3c23', '3c72', '3c78', '3d26', '3d3b',
        '3d3e', '436d', '3b26', '3b33', '3b45'
    ]

    sample_counts = [20, 30, 40, 50]
    results = []

    for n_samples in sample_counts:
        print(f"\n{'-' * 100}")
        print(f"サンプル数: {n_samples}/クラス")
        print('-' * 100)

        # データ読み込み
        images, labels, class_info = load_etl8b_data(
            base_path, target_classes, n_samples
        )

        if len(images) == 0:
            print("エラー: データが読み込めませんでした")
            continue

        # 訓練・テスト分割
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=0.3, random_state=42, stratify=labels
        )

        print(f"訓練データ: {len(train_images)}サンプル")
        print(f"テストデータ: {len(test_images)}サンプル")

        # 認識器
        recognizer = HierarchicalPatternRecognizer(
            num_pyramid_levels=5,
            target_image_size=32,
            class_num_method=ClassNumMethod.INCREMENTAL
        )

        common_labels = train_labels.copy()

        # 学習
        start_time = time.time()
        recognizer.train(train_images, train_labels, common_labels)
        train_time = time.time() - start_time

        # 評価
        start_time = time.time()
        accuracy, _ = recognizer.evaluate(test_images, test_labels)
        pred_time = (time.time() - start_time) / len(test_images)

        results.append({
            'name': f'サンプル数: {n_samples}/クラス',
            'n_samples': n_samples,
            'accuracy': accuracy,
            'train_time': train_time,
            'pred_time': pred_time
        })

        print(f"認識率: {accuracy:.2f}%")
        print(f"学習時間: {train_time:.2f}秒")

    return results

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python test_etl8b_15classes.py basic      - 基本テスト")
        print("  python test_etl8b_15classes.py compare    - 複数設定比較")
        print("  python test_etl8b_15classes.py sample     - サンプル数変動")
        print("  python test_etl8b_15classes.py all        - すべての実験")
        print()
        print("例:")
        print("  python test_etl8b_15classes.py basic")
        print("  python test_etl8b_15classes.py all")
        sys.exit(1)

    mode = sys.argv[1]
    base_path = sys.argv[2] if len(sys.argv) > 2 else './ETL8B-img-full'

    try:
        if mode == 'basic':
            result = experiment_basic(base_path, max_samples=30)

        elif mode == 'compare':
            results = experiment_comparison(base_path, max_samples=30)
            print_results_table(results)

        elif mode == 'sample':
            results = experiment_sample_variation(base_path)
            print_results_table(results)

        elif mode == 'all':
            print("\n" + "=" * 100)
            print(" 15クラス漢字認識 - 完全実験")
            print("=" * 100)

            all_results = []

            # 実験1
            result1 = experiment_basic(base_path, max_samples=30)
            if result1:
                all_results.append(result1)

            # 実験2
            results2 = experiment_comparison(base_path, max_samples=30)
            all_results.extend(results2)

            # 実験3
            results3 = experiment_sample_variation(base_path)
            all_results.extend(results3)

            # 全体サマリー
            print_results_table(all_results)

        else:
            print(f"不明なモード: {mode}")
            sys.exit(1)

    except Exception as e:
        print(f"\nエラー: {e}")
        print("\nC++ライブラリのコンパイルを確認してください:")
        print("  make")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
