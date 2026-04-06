#!/usr/bin/env python3
"""
test_etl8b.py
ETL8B漢字データを使用した階層構造的パターン認識器のテスト

認識率: 4クラス少数テスト用（基本動作確認）
"""

import sys
import numpy as np
from hierarchical_recognizer import (
    HierarchicalPatternRecognizer,
    load_etl8b_data,
    ClassNumMethod,
    RadiusMethod,
    WeightUpdateMethod
)
from sklearn.model_selection import train_test_split

def test_simple(base_path='./ETL8B-img-full', max_samples=30):
    """シンプルなテスト"""
    print("=" * 80)
    print(" ETL8B漢字認識テスト - シンプル版")
    print("=" * 80)
    print()
    
    # 少数クラスでテスト
    target_classes = ['3a2c', '3a2e', '3a4e', '3a59']
    
    # データ読み込み
    images, labels, class_info = load_etl8b_data(
        base_path, target_classes, max_samples
    )
    
    if len(images) == 0:
        print("エラー: データが読み込めませんでした")
        return
    
    # 分割
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"訓練: {len(train_images)}, テスト: {len(test_images)}")
    
    # 認識器
    recognizer = HierarchicalPatternRecognizer(
        num_pyramid_levels=5,
        target_image_size=32,
        class_num_method=ClassNumMethod.INCREMENTAL
    )
    
    # 学習
    common_labels = train_labels.copy()
    recognizer.train(train_images, train_labels, common_labels)
    
    # 評価
    accuracy, predictions = recognizer.evaluate(test_images, test_labels)
    
    print(f"\n最終認識率: {accuracy:.2f}%")
    
    # クラス別
    print("\nクラス別認識率:")
    for class_id, char in sorted(class_info.items()):
        mask = test_labels == class_id
        if np.sum(mask) > 0:
            class_acc = 100.0 * np.sum(predictions[mask] == class_id) / np.sum(mask)
            print(f"  {char}: {class_acc:.2f}%")

def test_full(base_path='./ETL8B-img-full', max_samples=50):
    """フルテスト（複数設定）"""
    print("=" * 80)
    print(" ETL8B漢字認識テスト - フル版")
    print("=" * 80)
    print()
    
    # 12クラス
    target_classes = [
        '3a2c', '3a2e', '3a4e', '3a59', '3a6e', '436d',
        '3c23', '3c72', '3c78', '3d26', '3d3b', '3d3e'
    ]
    
    # データ読み込み
    images, labels, class_info = load_etl8b_data(
        base_path, target_classes, max_samples
    )
    
    if len(images) == 0:
        print("エラー: データが読み込めませんでした")
        return
    
    # 分割
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"訓練: {len(train_images)}, テスト: {len(test_images)}")
    print()
    
    # 複数設定でテスト
    configs = [
        {
            'name': 'INCREMENTAL + METHOD1 + NO_UPDATE',
            'class_num': ClassNumMethod.INCREMENTAL,
            'radius': RadiusMethod.METHOD1,
            'update': WeightUpdateMethod.NO_UPDATE,
            'levels': 5
        },
        {
            'name': 'FIXED + METHOD2 + AVERAGE',
            'class_num': ClassNumMethod.FIXED,
            'radius': RadiusMethod.METHOD2,
            'update': WeightUpdateMethod.AVERAGE,
            'levels': 5
        },
        {
            'name': 'INCREMENTAL + METHOD1 + AVERAGE (7層)',
            'class_num': ClassNumMethod.INCREMENTAL,
            'radius': RadiusMethod.METHOD1,
            'update': WeightUpdateMethod.AVERAGE,
            'levels': 7
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"設定: {config['name']}")
        print('='*80)
        
        recognizer = HierarchicalPatternRecognizer(
            num_pyramid_levels=config['levels'],
            target_image_size=32,
            class_num_method=config['class_num'],
            radius_method=config['radius'],
            weight_update_method=config['update']
        )
        
        common_labels = train_labels.copy()
        recognizer.train(train_images, train_labels, common_labels)
        
        accuracy, predictions = recognizer.evaluate(test_images, test_labels)
        
        results.append({
            'name': config['name'],
            'accuracy': accuracy
        })
    
    # サマリー
    print(f"\n{'='*80}")
    print(" 結果サマリー")
    print('='*80)
    for r in results:
        print(f"{r['name']}: {r['accuracy']:.2f}%")
    print()

def test_comparative(base_path='./ETL8B-img-full'):
    """比較テスト（サンプル数を変えて）"""
    print("=" * 80)
    print(" ETL8B漢字認識テスト - 比較版")
    print("=" * 80)
    print()
    
    target_classes = ['3a2c', '3a2e', '3a4e', '3a59', '3a6e', '436d']
    sample_counts = [10, 20, 30, 50]
    
    results = []
    
    for n_samples in sample_counts:
        print(f"\n{'='*80}")
        print(f"サンプル数: {n_samples}")
        print('='*80)
        
        images, labels, class_info = load_etl8b_data(
            base_path, target_classes, n_samples
        )
        
        if len(images) == 0:
            continue
        
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        recognizer = HierarchicalPatternRecognizer(
            num_pyramid_levels=5,
            target_image_size=32,
            class_num_method=ClassNumMethod.INCREMENTAL
        )
        
        common_labels = train_labels.copy()
        recognizer.train(train_images, train_labels, common_labels)
        accuracy, _ = recognizer.evaluate(test_images, test_labels)
        
        results.append({
            'n_samples': n_samples,
            'accuracy': accuracy
        })
    
    # サマリー
    print(f"\n{'='*80}")
    print(" サンプル数と認識率の関係")
    print('='*80)
    for r in results:
        print(f"サンプル数 {r['n_samples']:2d}: {r['accuracy']:.2f}%")
    print()

def main():
    """メイン"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python test_etl8b.py simple [path] [n]  - シンプルテスト")
        print("  python test_etl8b.py full [path] [n]    - フルテスト")
        print("  python test_etl8b.py compare [path]     - 比較テスト")
        print()
        print("例:")
        print("  python test_etl8b.py simple ./ETL8B-img-full 30")
        print("  python test_etl8b.py full")
        print("  python test_etl8b.py compare")
        sys.exit(1)
    
    mode = sys.argv[1]
    base_path = sys.argv[2] if len(sys.argv) > 2 else './ETL8B-img-full'
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    try:
        if mode == 'simple':
            test_simple(base_path, max_samples)
        elif mode == 'full':
            test_full(base_path, max_samples)
        elif mode == 'compare':
            test_comparative(base_path)
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