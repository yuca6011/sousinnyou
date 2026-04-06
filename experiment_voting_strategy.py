#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投票戦略の追加実験

前回実験で判明した問題:
- 左特徴（偏）が5-18%と極端に低い（同じ偏の漢字を100クラス集めたため）
- 全体のみで96.13%達成（MLP 92.05%を上回る）
- 3者投票が逆効果（96.13% → 91.18%に低下）

本実験では以下の投票戦略を比較:
1. 全体のみ（投票なし）
2. 全体+右の2者投票（一致なら採用、不一致なら全体優先）
3. 重み付き投票（全体重視）
4. スコアベース統合（PNNスコアの加重平均）
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


class ScorePNN:
    """スコア（クラス別確率）を返すPNN"""

    def __init__(self, sigma_type='origpnn_fixed', r_den_value=None,
                 rbf_formula='cspnn', exemplar_ratio=0.2):
        self.sigma_type = sigma_type
        self.r_den_value = r_den_value
        self.rbf_formula = rbf_formula
        self.exemplar_ratio = exemplar_ratio
        self.exemplars = None
        self.n_classes = 0
        self._all_centroids = None
        self._class_ranges = None
        self.sigma = None
        self.classes_ = None

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.n_classes = len(classes)
        self.exemplars = {}

        for cls in classes:
            X_cls = X[y == cls]
            if len(X_cls) > 5:
                n_ex = max(1, int(len(X_cls) * self.exemplar_ratio))
                km = KMeans(n_clusters=n_ex, random_state=42, n_init=10)
                km.fit(X_cls)
                self.exemplars[cls] = km.cluster_centers_
            else:
                self.exemplars[cls] = X_cls.copy()

        self._build_cache()

        if self.sigma_type == 'adaptive':
            self.sigma = self._compute_adaptive_sigma()
        elif self.sigma_type == 'origpnn_fixed':
            self.sigma = self._compute_origpnn_sigma(X)

    def _build_cache(self):
        arrays, ranges = [], {}
        idx = 0
        for cls in sorted(self.exemplars.keys()):
            n = len(self.exemplars[cls])
            ranges[cls] = (idx, idx + n)
            arrays.append(self.exemplars[cls])
            idx += n
        self._all_centroids = np.vstack(arrays)
        self._class_ranges = ranges

    def _compute_adaptive_sigma(self):
        all_ex = self._all_centroids
        if len(all_ex) <= 1:
            return 1.0
        dists = cdist(all_ex, all_ex, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        nn_dists = np.min(dists, axis=1)
        return max(np.mean(nn_dists) / np.sqrt(2), 0.01)

    def _compute_origpnn_sigma(self, X):
        n = len(X)
        if n > 3000:
            idx = np.random.RandomState(42).choice(n, 3000, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X
        d_max = np.max(pdist(X_sub, metric='euclidean'))
        r_den = self.r_den_value if self.r_den_value else float(self.n_classes)
        return d_max / r_den

    def predict_scores(self, X):
        """各クラスのスコアを返す (n_samples, n_classes)"""
        all_c = self._all_centroids
        dists_sq = cdist(X, all_c, metric='sqeuclidean')
        n_samples = len(X)
        n_cls = len(self.classes_)
        scores = np.zeros((n_samples, n_cls))

        for i in range(n_samples):
            d_sq = dists_sq[i]
            for c_idx, cls in enumerate(sorted(self._class_ranges.keys())):
                start, end = self._class_ranges[cls]
                s = self.sigma
                if self.rbf_formula == 'cspnn':
                    acts = np.exp(-d_sq[start:end] / (s * s))
                else:
                    acts = np.exp(-d_sq[start:end] / (2.0 * s * s))
                scores[i, c_idx] = np.mean(acts)
        return scores

    def predict(self, X):
        scores = self.predict_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]


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


def main():
    print("=" * 100)
    print(" 投票戦略実験（σ=dmax/√Nc, 全特徴正規化）")
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
    print(f"  特徴抽出: {time.time()-t0:.1f}秒")

    # 全特徴を正規化
    sc_w, sc_l, sc_r = StandardScaler(), StandardScaler(), StandardScaler()
    tr_w = sc_w.fit_transform(tr_w)
    ts_w = sc_w.transform(ts_w)
    tr_l = sc_l.fit_transform(tr_l)
    ts_l = sc_l.transform(ts_l)
    tr_r = sc_r.fit_transform(tr_r)
    ts_r = sc_r.transform(ts_r)

    # 上位2設定で実験
    sigma_configs = [
        ("OrigPNN dmax/sqrt(Nc)", 'origpnn_fixed', np.sqrt(nc), 'cspnn'),
        ("Adaptive(baseline)", 'adaptive', None, 'standard'),
    ]

    print("\n[2/3] 実験実行...")
    all_results = []

    for cfg_name, sigma_type, r_den, rbf in sigma_configs:
        print(f"\n{'='*80}")
        print(f"  σ設定: {cfg_name}")
        print(f"{'='*80}")

        # 各パートのPNNを学習し、スコアを取得
        pnns = {}
        scores_train = {}
        scores_test = {}
        preds = {}
        accs = {}

        feats = {'whole': (tr_w, ts_w), 'left': (tr_l, ts_l), 'right': (tr_r, ts_r)}

        for part, (tr_f, ts_f) in feats.items():
            pnn = ScorePNN(
                sigma_type=sigma_type,
                r_den_value=r_den,
                rbf_formula=rbf,
                exemplar_ratio=0.2,
            )
            pnn.fit(tr_f, tr_lbl)
            pnns[part] = pnn

            sc = pnn.predict_scores(ts_f)
            scores_test[part] = sc
            pred = pnn.classes_[np.argmax(sc, axis=1)]
            preds[part] = pred
            accs[part] = accuracy_score(ts_lbl, pred) * 100
            print(f"  {part}: {accs[part]:.2f}%")

        # 投票戦略
        strategies = {}

        # 1. 全体のみ
        strategies['(1) Whole only'] = preds['whole']

        # 2. 3者多数決（従来）
        vote3 = []
        for w, l, r in zip(preds['whole'], preds['left'], preds['right']):
            v = [w, l, r]
            vote3.append(max(set(v), key=v.count))
        strategies['(2) 3-way vote (W+L+R)'] = np.array(vote3)

        # 3. 全体+右の2者投票（一致なら採用、不一致なら全体優先）
        vote_wr = []
        for w, r in zip(preds['whole'], preds['right']):
            vote_wr.append(w if w != r else w)  # 一致→採用、不一致→全体
        strategies['(3) W+R agree->adopt, else W'] = np.array(vote_wr)
        # Note: この場合は全体のみと同じ結果になる

        # 4. 全体+右 2者多数決 + 一致/不一致 判定
        # 一致→その値、不一致→全体を採用（これは全体のみと同じ）
        # なので代わりに、一致なら採用、不一致ならスコアで判断
        vote_wr_score = []
        for i in range(len(ts_lbl)):
            w, r = preds['whole'][i], preds['right'][i]
            if w == r:
                vote_wr_score.append(w)
            else:
                # スコアで判断（全体のスコアを優先）
                w_score = scores_test['whole'][i, w]
                r_score = scores_test['right'][i, r]
                if w_score >= r_score:
                    vote_wr_score.append(w)
                else:
                    vote_wr_score.append(r)
        strategies['(4) W+R agree or score-based'] = np.array(vote_wr_score)

        # 5. スコア加重統合（全体:右 = 2:1）
        w_weight, r_weight = 2.0, 1.0
        combined_21 = w_weight * scores_test['whole'] + r_weight * scores_test['right']
        strategies['(5) Score W:R=2:1'] = pnns['whole'].classes_[np.argmax(combined_21, axis=1)]

        # 6. スコア加重統合（全体:右 = 3:1）
        w_weight, r_weight = 3.0, 1.0
        combined_31 = w_weight * scores_test['whole'] + r_weight * scores_test['right']
        strategies['(6) Score W:R=3:1'] = pnns['whole'].classes_[np.argmax(combined_31, axis=1)]

        # 7. スコア加重統合（全体:左:右 = 3:0.5:1）
        combined_wlr = (3.0 * scores_test['whole'] +
                        0.5 * scores_test['left'] +
                        1.0 * scores_test['right'])
        strategies['(7) Score W:L:R=3:0.5:1'] = pnns['whole'].classes_[np.argmax(combined_wlr, axis=1)]

        # 8. スコア加重統合（全体:左:右 = 3:0:1）
        combined_w0r = 3.0 * scores_test['whole'] + 1.0 * scores_test['right']
        strategies['(8) Score W:L:R=3:0:1'] = pnns['whole'].classes_[np.argmax(combined_w0r, axis=1)]

        # 9. スコア正規化統合（全体+右をそれぞれ正規化してから加重）
        # 各パートのスコアを行ごとに合計1に正規化
        sw_norm = scores_test['whole'] / (scores_test['whole'].sum(axis=1, keepdims=True) + 1e-30)
        sr_norm = scores_test['right'] / (scores_test['right'].sum(axis=1, keepdims=True) + 1e-30)
        combined_norm = 2.0 * sw_norm + 1.0 * sr_norm
        strategies['(9) Normalized Score W:R=2:1'] = pnns['whole'].classes_[np.argmax(combined_norm, axis=1)]

        # 10. 全体+右+左 正規化スコア統合
        sl_norm = scores_test['left'] / (scores_test['left'].sum(axis=1, keepdims=True) + 1e-30)
        combined_norm_all = 3.0 * sw_norm + 0.5 * sl_norm + 1.0 * sr_norm
        strategies['(10) Norm Score W:L:R=3:0.5:1'] = pnns['whole'].classes_[np.argmax(combined_norm_all, axis=1)]

        # 結果表示
        print(f"\n  {'戦略':<40} {'精度':>8}")
        print(f"  {'-'*50}")

        cfg_results = []
        for sname, spred in strategies.items():
            acc = accuracy_score(ts_lbl, spred) * 100
            marker = " ★" if acc >= max(accs['whole'], 92.05) else ""
            print(f"  {sname:<40} {acc:>7.2f}%{marker}")
            cfg_results.append({
                'sigma': cfg_name,
                'strategy': sname,
                'accuracy': acc,
                'acc_whole': accs['whole'],
                'acc_left': accs['left'],
                'acc_right': accs['right'],
            })
        all_results.extend(cfg_results)

    # 最終まとめ
    print("\n" + "=" * 100)
    print(" 全結果まとめ（精度降順）")
    print("=" * 100)
    all_results.sort(key=lambda r: r['accuracy'], reverse=True)

    print(f"\n{'#':>2} {'σ設定':<30} {'投票戦略':<40} {'精度':>8}")
    print("-" * 85)
    for i, r in enumerate(all_results[:15]):
        marker = " ★" if i == 0 else ""
        print(f"{i+1:2d} {r['sigma']:<30} {r['strategy']:<40} {r['accuracy']:>7.2f}%{marker}")

    print(f"\n  参考: 前回ベストモデル(全体のみ正規化, 3者投票) = 89.32%")
    print(f"  参考: MLP                                     = 92.05%")

    best = all_results[0]
    print(f"\n  最良: {best['sigma']} + {best['strategy']} = {best['accuracy']:.2f}%")
    print(f"  MLP比: {best['accuracy'] - 92.05:+.2f}%")

    # 保存
    output_dir = 'results/cspnn_sigma'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'voting_strategy_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 可視化
    fig, ax = plt.subplots(figsize=(14, 8))
    # 上位設定のみプロット
    top_results = all_results[:15]
    names = [f"{r['sigma'][:12]}\n{r['strategy']}" for r in top_results]
    vals = [r['accuracy'] for r in top_results]
    colors = ['#e74c3c' if v >= 92.05 else '#3498db' for v in vals]

    bars = ax.barh(range(len(names)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(x=92.05, color='purple', linestyle='--', linewidth=2, label='MLP (92.05%)')
    ax.axvline(x=89.32, color='gray', linestyle='--', linewidth=1, label='Prev Best (89.32%)')

    for i, v in enumerate(vals):
        ax.text(v + 0.2, i, f'{v:.2f}%', va='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Voting Strategy Comparison (100 classes, all features normalized)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(85, 100)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'voting_strategy_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  結果を {output_dir}/ に保存しました")
    print("=" * 100)


if __name__ == '__main__':
    main()
