#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文準拠パラメータ実験: 12パターン全組み合わせ

クラス数設定法(2) × 半径値設定法(3) × 結合係数更新法(2) = 12パターン

クラス数設定法:
  INCREMENTAL(法1): qを動的増加（学習中にクラスが増えるごとにq+1）
  FIXED(法2): q=総クラス数で固定

半径値設定法:
  METHOD1: σ = d_max / q （入力と全代表点の最大距離 / q）
  METHOD2: σ = d_max / N_c （入力と全代表点の最大距離 / 総クラス数）
  METHOD3: σ = 固定値（学習時に計算した適応的σ）

結合係数更新法:
  NO_UPDATE: セントロイド更新なし
  AVERAGE: 正分類時にセントロイドを平均値で更新

使い方:
    python3 experiment_paper_methods.py
"""

import os
import sys
import time
import numpy as np
import cv2
import json
from enum import IntEnum
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 列挙型定義
# ============================================================================

class ClassNumMethod(IntEnum):
    """クラス数設定法"""
    INCREMENTAL = 0  # 法1: 動的増加
    FIXED = 1        # 法2: 固定

class RadiusMethod(IntEnum):
    """半径値設定法"""
    METHOD1 = 0  # σ = d_max,M / q
    METHOD2 = 1  # σ = d'_max,M / N_c
    METHOD3 = 2  # 任意の値（適応的σ）

class WeightUpdateMethod(IntEnum):
    """結合係数更新法"""
    NO_UPDATE = 0  # 更新なし
    AVERAGE = 1    # 平均値で更新


# ============================================================================
# 論文準拠カーネルメモリー分類器
# ============================================================================

class PaperKernelMemory:
    """
    論文準拠のカーネルメモリー分類器

    学習アルゴリズム:
      1. 最初のサンプルでサブネットを作成（RBFユニット1つ）
      2. 各学習サンプルに対して順伝播を実行
      3. 誤分類 → 正しいクラスに新しいRBFユニットを追加
      4. 正分類 + AVERAGE → 最近のセントロイドを平均値で更新

    順伝播:
      1. 入力xと全セントロイド間の距離²を計算
      2. d_max² = 最大距離²を求める
      3. σ² = d_max² / radius_denom² （METHOD1/2の場合）
      4. h_k(j) = exp(-dist² / σ²)
      5. y_k = mean(h_k(j)) （クラスkの全RBF活性化の平均）
      6. 予測 = argmax(y_k)
    """

    def __init__(self, class_num_method, radius_method, weight_update_method,
                 total_classes, fixed_sigma=1.0):
        self.class_num_method = class_num_method
        self.radius_method = radius_method
        self.weight_update_method = weight_update_method
        self.total_classes = total_classes
        self.fixed_sigma = fixed_sigma

        # サブネット: {class_id: np.ndarray (n_centroids, dim)}
        self.subnets = {}
        self.radius_denom = 1.0

        # 高速化用キャッシュ
        self._all_centroids = None
        self._class_ranges = None
        self._cache_valid = False

    def _update_radius_denom(self):
        """radius_denom を更新"""
        if self.radius_method == RadiusMethod.METHOD3:
            # METHOD3: σは固定値を使うのでradius_denomは使わない
            self.radius_denom = 1.0
        elif self.class_num_method == ClassNumMethod.INCREMENTAL:
            if self.radius_method == RadiusMethod.METHOD1:
                # INCREMENTAL + METHOD1: q = 現在のサブネット数
                self.radius_denom = max(1.0, float(len(self.subnets)))
            else:
                # INCREMENTAL + METHOD2: N_c = 総クラス数
                self.radius_denom = float(self.total_classes)
        else:
            # FIXED: 常に総クラス数
            self.radius_denom = float(self.total_classes)

    def _rebuild_cache(self):
        """高速化用のセントロイド行列を再構築"""
        if not self.subnets:
            self._cache_valid = False
            return

        arrays = []
        ranges = {}
        idx = 0
        for cls in sorted(self.subnets.keys()):
            n = len(self.subnets[cls])
            ranges[cls] = (idx, idx + n)
            arrays.append(self.subnets[cls])
            idx += n

        self._all_centroids = np.vstack(arrays)
        self._class_ranges = ranges
        self._cache_valid = True

    def _forward(self, x):
        """順伝播: クラスごとのスコアを返す"""
        if not self.subnets:
            return {}

        if not self._cache_valid:
            self._rebuild_cache()

        # 全セントロイドとの距離²を一括計算
        diff = x - self._all_centroids
        dists_sq = np.sum(diff * diff, axis=1)
        max_dist_sq = np.max(dists_sq)

        # σ²の計算
        if self.radius_method == RadiusMethod.METHOD3:
            sigma_sq = self.fixed_sigma ** 2
        else:
            if self.radius_denom > 0 and max_dist_sq > 0:
                sigma_sq = max_dist_sq / (self.radius_denom ** 2)
            else:
                sigma_sq = 1.0

        sigma_sq = max(sigma_sq, 1e-10)

        # RBF活性化
        activations = np.exp(-dists_sq / sigma_sq)

        # クラスごとの平均活性化
        class_scores = {}
        for cls, (start, end) in self._class_ranges.items():
            class_scores[cls] = np.mean(activations[start:end])

        return class_scores

    def fit(self, X, y):
        """論文アルゴリズムに従った学習"""
        self.subnets = {}
        self._cache_valid = False

        classes = np.unique(y)

        # クラスごとにインデックスを準備（クラス順に処理）
        class_indices = {}
        for cls in sorted(classes):
            class_indices[cls] = np.where(y == cls)[0]

        total_samples = len(X)
        processed = 0

        for cls in sorted(classes):
            indices = class_indices[cls]

            for i, idx in enumerate(indices):
                x = X[idx]
                target = y[idx]

                if target not in self.subnets:
                    # 新クラスの最初のサンプル → サブネット作成
                    self.subnets[target] = x.copy().reshape(1, -1)
                    self._cache_valid = False
                    self._update_radius_denom()
                    processed += 1
                    continue

                # 順伝播
                scores = self._forward(x)
                if scores:
                    predicted = max(scores, key=scores.get)
                else:
                    predicted = -1

                if predicted != target:
                    # 誤分類 → セントロイド追加
                    self.subnets[target] = np.vstack([
                        self.subnets[target], x.copy().reshape(1, -1)
                    ])
                    self._cache_valid = False
                elif self.weight_update_method == WeightUpdateMethod.AVERAGE:
                    # 正分類 + AVERAGE → 最近セントロイドを更新
                    centroids = self.subnets[target]
                    dists = np.sum((centroids - x) ** 2, axis=1)
                    nearest_idx = np.argmin(dists)
                    centroids[nearest_idx] = (centroids[nearest_idx] + x) / 2.0
                    self._cache_valid = False

                processed += 1

            # 各クラス処理後にradius_denom更新（INCREMENTAL用）
            if self.class_num_method == ClassNumMethod.INCREMENTAL and \
               self.radius_method == RadiusMethod.METHOD1:
                self.radius_denom = float(len(self.subnets))
                self._cache_valid = False

        # 最終的なradius_denom
        self._update_radius_denom()
        self._rebuild_cache()

        total_centroids = sum(len(c) for c in self.subnets.values())
        return total_centroids

    def predict(self, x):
        """単一サンプルの予測"""
        scores = self._forward(x)
        if not scores:
            return 0
        return max(scores, key=scores.get)

    def predict_batch(self, X):
        """バッチ予測"""
        return np.array([self.predict(x) for x in X])


# ============================================================================
# 特徴抽出（kanji_best_standalone.py から再利用）
# ============================================================================

def extract_all_features(images):
    """全画像から全体・左・右の特徴を抽出"""
    from kanji_best_standalone import (
        HierarchicalPyramid, LeftRightSplitter, HOGFeatureExtractor
    )

    pyramid_gen = HierarchicalPyramid(num_levels=5)
    splitter = LeftRightSplitter(split_method='projection')
    hog_extractor = HOGFeatureExtractor()

    def extract_hierarchical(image):
        pyramid = pyramid_gen.generate_pyramid(image)
        feats = []
        for level_img in pyramid:
            f = hog_extractor.extract(level_img)
            feats.append(f)
        return np.concatenate(feats)

    whole_features = []
    left_features = []
    right_features = []

    for i, img in enumerate(images):
        # 前処理
        if img.max() > 1.0:
            img = img.astype(np.float64) / 255.0
        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64))

        # 左右分割
        left, right = splitter.split(img)

        # 特徴抽出
        whole_features.append(extract_hierarchical(img))
        left_features.append(extract_hierarchical(left))
        right_features.append(extract_hierarchical(right))

        if (i + 1) % 1000 == 0:
            print(f"    特徴抽出: {i+1}/{len(images)}")

    return np.array(whole_features), np.array(left_features), np.array(right_features)


# ============================================================================
# データ読み込み
# ============================================================================

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


def load_etl8b_data(data_dir, target_classes, max_samples=None):
    """ETL8Bデータを読み込む"""
    images = []
    labels = []

    for label_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(data_dir, class_hex)
        if not os.path.exists(class_dir):
            continue

        image_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
        if max_samples is not None:
            image_files = image_files[:max_samples]

        for img_file in image_files:
            img = cv2.imread(os.path.join(class_dir, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                images.append(img)
                labels.append(label_idx)

    return np.array(images), np.array(labels)


# ============================================================================
# 実験の実行
# ============================================================================

def compute_adaptive_sigma(features):
    """適応的σを計算（METHOD3用）: 全代表点の最近傍距離平均 / sqrt(2)"""
    # サンプルが多すぎる場合はサブサンプル
    n = len(features)
    if n > 2000:
        idx = np.random.RandomState(42).choice(n, 2000, replace=False)
        features = features[idx]

    distances = cdist(features, features, metric='euclidean')
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    sigma = np.mean(min_distances) / np.sqrt(2)
    return max(sigma, 0.01)


def run_single_experiment(config, train_feats, train_labels, test_feats, test_labels,
                          total_classes, fixed_sigma):
    """単一パラメータ設定で実験を実行（全体・左・右の多数決投票）"""
    cnm, rm, wum = config

    results = {}

    # 3つの分類器（全体・左・右）
    for part_name, (tr_f, ts_f) in [
        ('whole', (train_feats[0], test_feats[0])),
        ('left',  (train_feats[1], test_feats[1])),
        ('right', (train_feats[2], test_feats[2])),
    ]:
        pnn = PaperKernelMemory(
            class_num_method=cnm,
            radius_method=rm,
            weight_update_method=wum,
            total_classes=total_classes,
            fixed_sigma=fixed_sigma
        )

        t0 = time.time()
        n_centroids = pnn.fit(tr_f, train_labels)
        train_time = time.time() - t0

        t0 = time.time()
        preds = pnn.predict_batch(ts_f)
        eval_time = time.time() - t0

        acc = accuracy_score(test_labels, preds) * 100

        results[part_name] = {
            'preds': preds,
            'accuracy': acc,
            'train_time': train_time,
            'eval_time': eval_time,
            'n_centroids': n_centroids,
        }

    # 多数決投票
    whole_preds = results['whole']['preds']
    left_preds = results['left']['preds']
    right_preds = results['right']['preds']

    vote_preds = []
    for w, l, r in zip(whole_preds, left_preds, right_preds):
        votes = [w, l, r]
        vote_preds.append(max(set(votes), key=votes.count))
    vote_preds = np.array(vote_preds)

    vote_acc = accuracy_score(test_labels, vote_preds) * 100
    total_train = sum(r['train_time'] for r in results.values())
    total_eval = sum(r['eval_time'] for r in results.values())
    total_centroids = sum(r['n_centroids'] for r in results.values())

    return {
        'accuracy_whole': results['whole']['accuracy'],
        'accuracy_left': results['left']['accuracy'],
        'accuracy_right': results['right']['accuracy'],
        'accuracy_vote': vote_acc,
        'train_time': total_train,
        'eval_time': total_eval,
        'n_centroids': total_centroids,
        'predictions': vote_preds,
    }


def method_name(cnm, rm, wum):
    """設定名を文字列に変換"""
    cn = "INC" if cnm == ClassNumMethod.INCREMENTAL else "FIX"
    r = ["M1(d/q)", "M2(d/Nc)", "M3(fixed)"][rm]
    w = "noUpd" if wum == WeightUpdateMethod.NO_UPDATE else "avg"
    return f"{cn}_{r}_{w}"


def main():
    print("=" * 100)
    print(" 論文準拠パラメータ実験: 12パターン全組み合わせ")
    print("=" * 100)

    # データ読み込み
    print("\n[1/4] データ読み込み中...")
    images, labels = load_etl8b_data("./ETL8B-img-full", TARGET_CLASSES_100)
    num_classes = len(np.unique(labels))
    print(f"  {len(images)}サンプル, {num_classes}クラス")

    # 訓練・テスト分割
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"  訓練: {len(train_images)}, テスト: {len(test_images)}")

    # 特徴抽出
    print("\n[2/4] 特徴抽出中...")
    t0 = time.time()
    train_whole, train_left, train_right = extract_all_features(train_images)
    test_whole, test_left, test_right = extract_all_features(test_images)
    feat_time = time.time() - t0
    print(f"  特徴抽出完了: {feat_time:.1f}秒")
    print(f"  全体特徴次元: {train_whole.shape[1]}")
    print(f"  左特徴次元: {train_left.shape[1]}")
    print(f"  右特徴次元: {train_right.shape[1]}")

    # 正規化（全体のみ）
    scaler_whole = StandardScaler()
    train_whole = scaler_whole.fit_transform(train_whole)
    test_whole = scaler_whole.transform(test_whole)

    # METHOD3用のσを事前計算
    print("\n  METHOD3用の適応的σを計算中...")
    fixed_sigma_whole = compute_adaptive_sigma(train_whole)
    fixed_sigma_left = compute_adaptive_sigma(train_left)
    fixed_sigma_right = compute_adaptive_sigma(train_right)
    print(f"  σ(whole)={fixed_sigma_whole:.4f}, σ(left)={fixed_sigma_left:.4f}, σ(right)={fixed_sigma_right:.4f}")
    # 全分類器共通の平均σを使用
    fixed_sigma = np.mean([fixed_sigma_whole, fixed_sigma_left, fixed_sigma_right])
    print(f"  平均σ(METHOD3用): {fixed_sigma:.4f}")

    # 12パターン実験
    print("\n[3/4] 12パターン実験実行中...")
    print("=" * 100)

    configs = []
    for cnm in ClassNumMethod:
        for rm in RadiusMethod:
            for wum in WeightUpdateMethod:
                configs.append((cnm, rm, wum))

    all_results = []

    for exp_idx, (cnm, rm, wum) in enumerate(configs):
        name = method_name(cnm, rm, wum)
        print(f"\n--- 実験 {exp_idx+1}/12: {name} ---")

        result = run_single_experiment(
            config=(cnm, rm, wum),
            train_feats=(train_whole, train_left, train_right),
            test_feats=(test_whole, test_left, test_right),
            train_labels=train_labels,
            test_labels=test_labels,
            total_classes=num_classes,
            fixed_sigma=fixed_sigma,
        )
        result['name'] = name
        result['config'] = {
            'class_num_method': int(cnm),
            'radius_method': int(rm),
            'weight_update_method': int(wum),
        }
        all_results.append(result)

        print(f"  全体: {result['accuracy_whole']:.2f}% | "
              f"左: {result['accuracy_left']:.2f}% | "
              f"右: {result['accuracy_right']:.2f}% | "
              f"投票: {result['accuracy_vote']:.2f}%")
        print(f"  代表点数: {result['n_centroids']} | "
              f"学習: {result['train_time']:.1f}秒 | "
              f"評価: {result['eval_time']:.1f}秒")

    # 結果まとめ
    print("\n" + "=" * 100)
    print(" 実験結果まとめ")
    print("=" * 100)

    # ソート（投票精度降順）
    sorted_results = sorted(all_results, key=lambda r: r['accuracy_vote'], reverse=True)

    print(f"\n{'#':>2} {'設定':<25} {'全体':>7} {'左':>7} {'右':>7} "
          f"{'投票':>7} {'代表点':>8} {'学習':>8} {'評価':>8}")
    print("-" * 100)

    for i, r in enumerate(sorted_results):
        marker = " ★" if i == 0 else ""
        print(f"{i+1:2d} {r['name']:<25} "
              f"{r['accuracy_whole']:>6.2f}% {r['accuracy_left']:>6.2f}% "
              f"{r['accuracy_right']:>6.2f}% {r['accuracy_vote']:>6.2f}%{marker} "
              f"{r['n_centroids']:>7d} {r['train_time']:>7.1f}s {r['eval_time']:>7.1f}s")

    # 参考: 現行モデルとMLP
    print(f"\n  参考: 現行ベストモデル(adaptive σ + K-means) = 89.32%")
    print(f"  参考: MLP (DNN)                             = 92.05%")

    best = sorted_results[0]
    diff_best = best['accuracy_vote'] - 89.32
    diff_mlp = best['accuracy_vote'] - 92.05
    print(f"\n  最良結果: {best['name']} = {best['accuracy_vote']:.2f}%")
    print(f"  現行比: {diff_best:+.2f}%, MLP比: {diff_mlp:+.2f}%")

    # 結果保存
    print("\n[4/4] 結果保存中...")
    output_dir = 'results/paper_methods'
    os.makedirs(output_dir, exist_ok=True)

    # JSON保存
    save_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != 'predictions'}
        save_results.append(sr)

    with open(os.path.join(output_dir, 'paper_methods_results.json'), 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    # 可視化
    plot_results(sorted_results, output_dir)

    print(f"\n結果を {output_dir}/ に保存しました")
    print("=" * 100)


def plot_results(sorted_results, output_dir):
    """結果の可視化"""

    # 1. 全12パターンの精度比較
    fig, ax = plt.subplots(figsize=(14, 7))
    names = [r['name'] for r in sorted_results]
    accs = [r['accuracy_vote'] for r in sorted_results]

    colors = []
    for r in sorted_results:
        rm = r['config']['radius_method']
        if rm == 0:
            colors.append('#3498db')  # METHOD1: 青
        elif rm == 1:
            colors.append('#2ecc71')  # METHOD2: 緑
        else:
            colors.append('#e74c3c')  # METHOD3: 赤

    bars = ax.bar(range(len(names)), accs, color=colors, alpha=0.85)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{acc:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 参考線
    ax.axhline(y=89.32, color='gray', linestyle='--', alpha=0.7, label='Current Best (89.32%)')
    ax.axhline(y=92.05, color='orange', linestyle='--', alpha=0.7, label='MLP (92.05%)')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Paper Methods: All 12 Combinations (100 classes, majority vote)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    # 凡例用ダミー
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.85, label='M1: d_max/q'),
        Patch(facecolor='#2ecc71', alpha=0.85, label='M2: d_max/Nc'),
        Patch(facecolor='#e74c3c', alpha=0.85, label='M3: fixed sigma'),
        plt.Line2D([0], [0], color='gray', linestyle='--', label='Current Best (89.32%)'),
        plt.Line2D([0], [0], color='orange', linestyle='--', label='MLP (92.05%)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paper_methods_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 全体/左/右/投票の内訳
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (key, title) in zip(axes.flat, [
        ('accuracy_whole', 'Whole Classifier'),
        ('accuracy_left', 'Left Classifier (Hen)'),
        ('accuracy_right', 'Right Classifier (Tsukuri)'),
        ('accuracy_vote', 'Majority Vote'),
    ]):
        vals = [r[key] for r in sorted_results]
        bars = ax.barh(range(len(names)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Accuracy (%)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.axvline(x=92.05, color='orange', linestyle='--', alpha=0.5)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paper_methods_detail.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 代表点数と精度の関係
    fig, ax = plt.subplots(figsize=(10, 6))
    centroids = [r['n_centroids'] for r in sorted_results]
    ax.scatter(centroids, accs, c=colors, s=100, alpha=0.85, edgecolors='black')

    for i, r in enumerate(sorted_results):
        ax.annotate(r['name'], (centroids[i], accs[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.axhline(y=92.05, color='orange', linestyle='--', alpha=0.5, label='MLP (92.05%)')
    ax.set_xlabel('Total Centroids (3 classifiers)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Centroids vs Accuracy', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'centroids_vs_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
