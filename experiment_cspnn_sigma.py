#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS-PNN論文準拠σ決定法の実験

CS-PNN論文「Automatic Construction of Pattern Classifiers Capable of
Continuous Incremental Learning and Unlearning Tasks Based on
Compact-Sized Probabilistic Neural Network」のσ決定方式を適用。

CS-PNNのσ決定法:
  σ²(x) = max_v2(x) / r_den²
  h_j = exp(-||x - c_j||² / σ²)  ← 2σ²ではなくσ²
  y_k = mean(h_j for j in class k)

max_v2(x) = 入力xと全RBF中心の距離²の最大値（入力ごとに動的）
r_den = クラス数（or そのバリエーション）

比較する設定:
  1. 現行adaptive (baseline)
  2. CS-PNN dynamic (r_den=Nc)
  3. CS-PNN dynamic (r_den=√Nc)
  4. CS-PNN dynamic (r_den=√(2Nc))
  5. CS-PNN dynamic (r_den=∛Nc)
  6. OrigPNN fixed (σ=d_max_all/Nc)
  7. OrigPNN fixed (σ=d_max_all/√Nc)
  8. クラス別σ

使い方:
    python3 experiment_cspnn_sigma.py
"""

import os
import sys
import time
import numpy as np
import cv2
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 100クラスリスト
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


# ============================================================================
# PNN分類器（複数のσ方式対応）
# ============================================================================

class MultiSigmaPNN:
    """
    K-means代表点 + 複数のσ計算方式に対応するPNN

    sigma_type:
      'adaptive'       : σ = mean(NN距離) / √2（現行最良モデル）
      'class_specific'  : クラスごとに σ_c = mean(NN距離_c) / √2
      'cspnn_dynamic'   : CS-PNN方式。推論時に動的計算: σ²=max_v2/r_den²
      'origpnn_fixed'   : OrigPNN方式。σ = d_max(全体) / r_den、固定

    rbf_formula:
      'standard'  : exp(-d² / (2σ²))  ← kanji_best_standaloneの方式
      'cspnn'     : exp(-d² / σ²)     ← CS-PNN / OrigPNNの方式
    """

    def __init__(self, sigma_type='adaptive', r_den_value=None,
                 rbf_formula='standard', exemplar_ratio=0.2):
        self.sigma_type = sigma_type
        self.r_den_value = r_den_value
        self.rbf_formula = rbf_formula
        self.exemplar_ratio = exemplar_ratio

        self.exemplars = None     # {cls: np.array (n, dim)}
        self.sigma = None         # float or dict
        self.n_classes = 0

        # 高速化用キャッシュ
        self._all_centroids = None
        self._class_ranges = None

    def fit(self, X, y):
        """K-meansで代表点を選択し、σを計算"""
        classes = np.unique(y)
        self.n_classes = len(classes)
        self.exemplars = {}

        # K-means代表点選択
        for cls in classes:
            X_cls = X[y == cls]
            if len(X_cls) > 5:
                n_ex = max(1, int(len(X_cls) * self.exemplar_ratio))
                km = KMeans(n_clusters=n_ex, random_state=42, n_init=10)
                km.fit(X_cls)
                self.exemplars[cls] = km.cluster_centers_
            else:
                self.exemplars[cls] = X_cls.copy()

        # キャッシュ構築
        self._build_cache()

        # σの事前計算（adaptive/class_specific/origpnn_fixed）
        if self.sigma_type == 'adaptive':
            self.sigma = self._compute_adaptive_sigma()
        elif self.sigma_type == 'class_specific':
            self.sigma = self._compute_class_specific_sigma()
        elif self.sigma_type == 'origpnn_fixed':
            self.sigma = self._compute_origpnn_sigma(X)
        # cspnn_dynamic は推論時に計算するので事前計算不要

    def _build_cache(self):
        """全代表点の行列とクラス範囲を構築"""
        arrays = []
        ranges = {}
        idx = 0
        for cls in sorted(self.exemplars.keys()):
            n = len(self.exemplars[cls])
            ranges[cls] = (idx, idx + n)
            arrays.append(self.exemplars[cls])
            idx += n
        self._all_centroids = np.vstack(arrays)
        self._class_ranges = ranges

    def _compute_adaptive_sigma(self):
        """現行方式: σ = mean(最近傍距離) / √2"""
        all_ex = self._all_centroids
        if len(all_ex) <= 1:
            return 1.0
        dists = cdist(all_ex, all_ex, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        nn_dists = np.min(dists, axis=1)
        return max(np.mean(nn_dists) / np.sqrt(2), 0.01)

    def _compute_class_specific_sigma(self):
        """クラスごとの適応的σ"""
        sigmas = {}
        for cls, ex in self.exemplars.items():
            if len(ex) > 1:
                dists = cdist(ex, ex, metric='euclidean')
                np.fill_diagonal(dists, np.inf)
                nn_dists = np.min(dists, axis=1)
                sigmas[cls] = max(np.mean(nn_dists) / np.sqrt(2), 0.01)
            else:
                sigmas[cls] = 1.0
        return sigmas

    def _compute_origpnn_sigma(self, X):
        """OrigPNN方式: σ = d_max(全学習データ) / r_den"""
        # 全学習データからd_maxを計算（サブサンプル）
        n = len(X)
        if n > 3000:
            idx = np.random.RandomState(42).choice(n, 3000, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X
        d_max = np.max(pdist(X_sub, metric='euclidean'))
        r_den = self.r_den_value if self.r_den_value else float(self.n_classes)
        return d_max / r_den

    def predict_batch(self, X):
        """バッチ予測"""
        if self.sigma_type == 'cspnn_dynamic':
            return self._predict_batch_cspnn(X)
        else:
            return self._predict_batch_fixed(X)

    def _predict_batch_fixed(self, X):
        """固定σでのバッチ予測"""
        all_c = self._all_centroids
        # 全テストサンプルと全代表点の距離²
        dists_sq = cdist(X, all_c, metric='sqeuclidean')

        predictions = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            d_sq = dists_sq[i]
            # クラスごとのスコア計算
            best_cls = -1
            best_score = -1.0
            for cls, (start, end) in self._class_ranges.items():
                if isinstance(self.sigma, dict):
                    s = self.sigma[cls]
                else:
                    s = self.sigma

                if self.rbf_formula == 'cspnn':
                    acts = np.exp(-d_sq[start:end] / (s * s))
                else:
                    acts = np.exp(-d_sq[start:end] / (2.0 * s * s))

                score = np.mean(acts)
                if score > best_score:
                    best_score = score
                    best_cls = cls

            predictions[i] = best_cls
        return predictions

    def _predict_batch_cspnn(self, X):
        """CS-PNN動的σでのバッチ予測"""
        all_c = self._all_centroids
        r_den = self.r_den_value if self.r_den_value else float(self.n_classes)

        # 全テストサンプルと全代表点の距離²
        dists_sq = cdist(X, all_c, metric='sqeuclidean')

        predictions = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            d_sq = dists_sq[i]
            max_v2 = np.max(d_sq)

            if max_v2 < 1e-10 or r_den < 1e-10:
                predictions[i] = 0
                continue

            # CS-PNN: σ² = max_v2 / r_den², exp(-d²/σ²)
            sigma_sq = max_v2 / (r_den * r_den)
            acts = np.exp(-d_sq / sigma_sq)

            best_cls = -1
            best_score = -1.0
            for cls, (start, end) in self._class_ranges.items():
                score = np.mean(acts[start:end])
                if score > best_score:
                    best_score = score
                    best_cls = cls

            predictions[i] = best_cls
        return predictions


# ============================================================================
# 特徴抽出
# ============================================================================

def extract_all_features(images):
    """全画像から全体・左・右の特徴を抽出"""
    from kanji_best_standalone import (
        HierarchicalPyramid, LeftRightSplitter, HOGFeatureExtractor
    )
    pyramid_gen = HierarchicalPyramid(num_levels=5)
    splitter = LeftRightSplitter(split_method='projection')
    hog_ext = HOGFeatureExtractor()

    def extract_hier(img):
        pyramid = pyramid_gen.generate_pyramid(img)
        return np.concatenate([hog_ext.extract(lv) for lv in pyramid])

    whole_f, left_f, right_f = [], [], []
    for i, img in enumerate(images):
        if img.max() > 1.0:
            img = img.astype(np.float64) / 255.0
        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64))
        left, right = splitter.split(img)
        whole_f.append(extract_hier(img))
        left_f.append(extract_hier(left))
        right_f.append(extract_hier(right))
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(images)}")

    return np.array(whole_f), np.array(left_f), np.array(right_f)


def load_etl8b_data(data_dir, target_classes):
    """ETL8Bデータ読み込み"""
    images, labels = [], []
    for idx, hex_code in enumerate(target_classes):
        d = os.path.join(data_dir, hex_code)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith('.png'):
                img = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(cv2.resize(img, (64, 64)))
                    labels.append(idx)
    return np.array(images), np.array(labels)


# ============================================================================
# 実験本体
# ============================================================================

def run_config(name, train_feats, test_feats, train_labels, test_labels, nc,
               sigma_type, r_den_value, rbf_formula):
    """1設定分の実験（全体・左・右の多数決投票）"""
    parts = ['whole', 'left', 'right']
    preds_per_part = {}
    accs = {}
    total_time = 0.0

    for p_idx, part in enumerate(parts):
        pnn = MultiSigmaPNN(
            sigma_type=sigma_type,
            r_den_value=r_den_value,
            rbf_formula=rbf_formula,
            exemplar_ratio=0.2,
        )
        pnn.fit(train_feats[p_idx], train_labels)

        t0 = time.time()
        preds = pnn.predict_batch(test_feats[p_idx])
        dt = time.time() - t0
        total_time += dt

        acc = accuracy_score(test_labels, preds) * 100
        preds_per_part[part] = preds
        accs[part] = acc

    # 多数決投票
    vote = []
    for w, l, r in zip(preds_per_part['whole'],
                       preds_per_part['left'],
                       preds_per_part['right']):
        v = [w, l, r]
        vote.append(max(set(v), key=v.count))
    vote = np.array(vote)
    acc_vote = accuracy_score(test_labels, vote) * 100

    return {
        'name': name,
        'acc_whole': accs['whole'],
        'acc_left': accs['left'],
        'acc_right': accs['right'],
        'acc_vote': acc_vote,
        'eval_time': total_time,
    }


def main():
    print("=" * 100)
    print(" CS-PNN σ決定法実験")
    print("=" * 100)

    # データ読み込み
    print("\n[1/3] データ読み込み・特徴抽出...")
    images, labels = load_etl8b_data("./ETL8B-img-full", TARGET_CLASSES_100)
    nc = len(np.unique(labels))
    print(f"  {len(images)}サンプル, {nc}クラス")

    tr_img, ts_img, tr_lbl, ts_lbl = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    t0 = time.time()
    tr_w, tr_l, tr_r = extract_all_features(tr_img)
    ts_w, ts_l, ts_r = extract_all_features(ts_img)
    print(f"  特徴抽出: {time.time()-t0:.1f}秒 (dim={tr_w.shape[1]})")

    # 全特徴を正規化（全体・左・右それぞれ個別に）
    sc_w, sc_l, sc_r = StandardScaler(), StandardScaler(), StandardScaler()
    tr_w = sc_w.fit_transform(tr_w)
    ts_w = sc_w.transform(ts_w)
    tr_l = sc_l.fit_transform(tr_l)
    ts_l = sc_l.transform(ts_l)
    tr_r = sc_r.fit_transform(tr_r)
    ts_r = sc_r.transform(ts_r)

    train_feats = [tr_w, tr_l, tr_r]
    test_feats = [ts_w, ts_l, ts_r]

    # 実験設定
    print("\n[2/3] 実験実行中...")
    configs = [
        # (名前, sigma_type, r_den_value, rbf_formula)
        ("A: Adaptive(baseline)",
         'adaptive', None, 'standard'),
        ("B: ClassSpecific",
         'class_specific', None, 'standard'),
        ("C: CSPNN r=Nc",
         'cspnn_dynamic', float(nc), 'cspnn'),
        ("D: CSPNN r=√Nc",
         'cspnn_dynamic', np.sqrt(nc), 'cspnn'),
        ("E: CSPNN r=√(2Nc)",
         'cspnn_dynamic', np.sqrt(2 * nc), 'cspnn'),
        ("F: CSPNN r=∛Nc",
         'cspnn_dynamic', nc ** (1.0/3.0), 'cspnn'),
        ("G: CSPNN r=2",
         'cspnn_dynamic', 2.0, 'cspnn'),
        ("H: OrigPNN σ=dmax/Nc",
         'origpnn_fixed', float(nc), 'cspnn'),
        ("I: OrigPNN σ=dmax/√Nc",
         'origpnn_fixed', np.sqrt(nc), 'cspnn'),
        ("J: OrigPNN σ=dmax/√(2Nc)",
         'origpnn_fixed', np.sqrt(2 * nc), 'cspnn'),
    ]

    results = []
    for i, (name, st, rd, rbf) in enumerate(configs):
        print(f"\n  [{i+1}/{len(configs)}] {name}")
        r = run_config(name, train_feats, test_feats, tr_lbl, ts_lbl,
                       nc, st, rd, rbf)
        results.append(r)
        print(f"    全体={r['acc_whole']:.2f}% 左={r['acc_left']:.2f}% "
              f"右={r['acc_right']:.2f}% 投票={r['acc_vote']:.2f}% "
              f"({r['eval_time']:.1f}s)")

    # 結果まとめ
    print("\n" + "=" * 100)
    print(" 結果まとめ（投票精度降順）")
    print("=" * 100)

    results_sorted = sorted(results, key=lambda r: r['acc_vote'], reverse=True)

    print(f"\n{'#':>2} {'設定':<30} {'全体':>7} {'左':>7} {'右':>7} {'投票':>7}")
    print("-" * 75)
    for i, r in enumerate(results_sorted):
        m = " ★" if i == 0 else ""
        print(f"{i+1:2d} {r['name']:<30} "
              f"{r['acc_whole']:>6.2f}% {r['acc_left']:>6.2f}% "
              f"{r['acc_right']:>6.2f}% {r['acc_vote']:>6.2f}%{m}")

    print(f"\n  参考: 前回ベストモデル(全体のみ正規化) = 89.32%")
    print(f"  参考: MLP                             = 92.05%")

    best = results_sorted[0]
    print(f"\n  最良: {best['name']} = {best['acc_vote']:.2f}%")
    print(f"  MLP比: {best['acc_vote'] - 92.05:+.2f}%")

    # 保存
    output_dir = 'results/cspnn_sigma'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'cspnn_sigma_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 可視化
    fig, ax = plt.subplots(figsize=(14, 7))
    names = [r['name'] for r in results_sorted]
    accs = [r['acc_vote'] for r in results_sorted]
    acc_w = [r['acc_whole'] for r in results_sorted]
    acc_l = [r['acc_left'] for r in results_sorted]
    acc_r = [r['acc_right'] for r in results_sorted]

    x = np.arange(len(names))
    w = 0.2
    ax.bar(x - 1.5*w, acc_w, w, label='Whole', alpha=0.8, color='#3498db')
    ax.bar(x - 0.5*w, acc_l, w, label='Left', alpha=0.8, color='#2ecc71')
    ax.bar(x + 0.5*w, acc_r, w, label='Right', alpha=0.8, color='#e67e22')
    ax.bar(x + 1.5*w, accs, w, label='Vote', alpha=0.9, color='#e74c3c')

    for i, v in enumerate(accs):
        ax.text(x[i] + 1.5*w, v + 0.5, f'{v:.1f}', ha='center', fontsize=8, fontweight='bold')

    ax.axhline(y=89.32, color='gray', linestyle='--', alpha=0.6, label='Prev Best (89.32%)')
    ax.axhline(y=92.05, color='purple', linestyle='--', alpha=0.6, label='MLP (92.05%)')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('CS-PNN Sigma Methods Comparison (100 classes, K-means centroids, all normalized)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cspnn_sigma_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  結果を {output_dir}/ に保存しました")
    print("=" * 100)


if __name__ == '__main__':
    main()
