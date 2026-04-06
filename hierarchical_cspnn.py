#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS-PNN × 階層構造的画像パターン認識器

基盤論文:
  [1] Hoya & Morita (arXiv:2501.00725)
      "Automatic Construction of Pattern Classifiers Capable of Continuous
       Incremental Learning and Unlearning Tasks Based on Compact-Sized
       Probabilistic Neural Network"
  [2] 保谷・西脇 (2025)
      "階層構造的画像パターン認識器"

論文[2]の階層ピラミッド＋構造分解の枠組みに、論文[1]のCS-PNNを分類器として
組み込んだ統合システム。複数の設定を切り替えて比較可能。

使用方法:
    python hierarchical_cspnn.py                  # 15クラスで全構成比較
    python hierarchical_cspnn.py --classes 100    # 100クラス
    python hierarchical_cspnn.py --config 1       # 構成1のみ実行
"""

import numpy as np
import cv2
import os
import json
import time
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 第1部: 前処理（論文[2] Step 1）
# ============================================================================

class ImagePreprocessor:
    """
    画像前処理（論文[2] 3.1節 Step 1）

    処理手順:
      1. グレースケール化
      2. 射影法による余白トリミング（2値化せず、閾値ベース）
      3. 正方形化（ゼロパディング）
      4. リサイズ
      5. 正規化（[-1,1] or [0,1]）
    """

    @staticmethod
    def trim_margins(img, threshold=5):
        """
        射影法による余白トリミング

        論文[2]: "原画像のx,y両軸への射影による上下左右余白のトリミング処理"
        ※ 2値化は行わず、グレースケール値の閾値で判定
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # uint8に変換（float入力対応）
        if gray.dtype in (np.float32, np.float64):
            gray = (gray * 255).astype(np.uint8)

        # 射影: 各行・各列の画素値合計
        x_proj = np.sum(gray, axis=0)
        y_proj = np.sum(gray, axis=1)

        # 閾値以上の画素が存在する行・列を検出
        x_nonzero = np.where(x_proj > threshold * gray.shape[0])[0]
        y_nonzero = np.where(y_proj > threshold * gray.shape[1])[0]

        if len(x_nonzero) > 0 and len(y_nonzero) > 0:
            x_min, x_max = x_nonzero[0], x_nonzero[-1]
            y_min, y_max = y_nonzero[0], y_nonzero[-1]
            trimmed = gray[y_min:y_max+1, x_min:x_max+1]
        else:
            trimmed = gray

        return trimmed

    @staticmethod
    def squarify(img):
        """
        正方形化（論文[2]: "縦横サイズを等しく、すなわち(M,M)または(N,N)に変形"）
        ゼロパディングで正方形に変換
        """
        h, w = img.shape[:2]
        size = max(h, w)
        if size == 0:
            return np.zeros((8, 8), dtype=img.dtype)
        canvas = np.zeros((size, size), dtype=img.dtype)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img
        return canvas

    @staticmethod
    def normalize(img, method='pm1'):
        """
        画素値正規化

        method:
          'pm1': [-1, 1] — 論文[1]の実験設定に準拠
          '01':  [0, 1]
        """
        img = img.astype(np.float64)
        if img.max() > 1.0:
            img = img / 255.0

        if method == 'pm1':
            return (img - 0.5) * 2.0  # [0,1] → [-1,1]
        elif method == '01':
            return img
        return img

    @classmethod
    def preprocess(cls, img, target_size=64, norm='pm1'):
        """前処理パイプライン"""
        trimmed = cls.trim_margins(img)
        squared = cls.squarify(trimmed)
        resized = cv2.resize(squared, (target_size, target_size),
                             interpolation=cv2.INTER_AREA)
        normalized = cls.normalize(resized, method=norm)
        return normalized


# ============================================================================
# 第2部: 階層ピラミッド（論文[2] Step 1）
# ============================================================================

class PyramidGenerator:
    """
    階層ピラミッド構造生成

    論文[2]: "複数回の量子化を施すことにより得られる異なる複数の解像度からなる
             階層ピラミッド構造を生成する"

    method:
      'quantize': 論文[2]準拠 — 2x2ブロック平均による量子化
      'gaussian': 比較用 — ガウシアンぼかし+面積補間リサイズ
    """

    def __init__(self, num_levels=5, method='quantize'):
        self.num_levels = num_levels
        self.method = method

    def _quantize_2x2(self, img):
        """
        2x2ブロック平均による量子化（論文[2]の基本手法）

        論文[2]: "ピラミッドの底に該当する画像データそのものを量子化して得られる
                  新たな画像データ"
        C++実装(hierarchical_recognizer.cpp): 2x2ブロックの平均
        """
        h, w = img.shape
        # 奇数サイズの場合は切り捨て
        h2, w2 = (h // 2) * 2, (w // 2) * 2
        img_even = img[:h2, :w2]
        return img_even.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))

    def _gaussian_downsample(self, img):
        """ガウシアンぼかし+リサイズ（比較用）"""
        blurred = gaussian_filter(img, sigma=1.0)
        h, w = blurred.shape
        return cv2.resize(blurred, (w // 2, h // 2),
                          interpolation=cv2.INTER_AREA)

    def generate(self, img):
        """
        階層ピラミッド生成

        Returns:
            list: [原画像(最高解像度), ..., 最低解像度]
                  例: [64x64, 32x32, 16x16, 8x8, 4x4]
        """
        pyramid = [img]
        current = img.copy()

        for _ in range(self.num_levels - 1):
            if current.shape[0] < 2 or current.shape[1] < 2:
                break
            if self.method == 'quantize':
                current = self._quantize_2x2(current)
            else:  # gaussian
                current = self._gaussian_downsample(current)
            pyramid.append(current)

        return pyramid


# ============================================================================
# 第3部: 左右分割（改良版参照）
# ============================================================================

class LeftRightSplitter:
    """
    左右分割（偏・旁の分離）

    改良版のargmax projection方式を使用:
      垂直射影の中央±W/4範囲で最大値（最も白い列）を分割点とする
    """

    def detect_split_column(self, image):
        """分割位置の検出（argmax projection方式）"""
        vertical_projection = np.sum(image, axis=0)
        if len(vertical_projection) == 0:
            return image.shape[1] // 2

        width = len(vertical_projection)
        center = width // 2
        center_region_start = max(0, center - width // 4)
        center_region_end = min(width, center + width // 4)
        center_region = vertical_projection[center_region_start:center_region_end]

        if len(center_region) == 0:
            return image.shape[1] // 2

        local_max_idx = np.argmax(center_region)
        return center_region_start + local_max_idx

    def split(self, image):
        """画像を左右に分割し、それぞれ正方形化"""
        split_col = self.detect_split_column(image)

        left = image[:, :split_col]
        right = image[:, split_col:]

        if left.size == 0 or right.size == 0:
            mid = image.shape[1] // 2
            left = image[:, :mid]
            right = image[:, mid:]

        return self._to_square(left), self._to_square(right)

    def _to_square(self, image):
        """ゼロパディングで正方形化"""
        h, w = image.shape
        size = max(h, w)
        if size == 0:
            return np.zeros((8, 8), dtype=image.dtype)
        square = np.zeros((size, size), dtype=image.dtype)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = image
        return square


# ============================================================================
# 第4部: 特徴抽出
# ============================================================================

class FeatureExtractor:
    """
    ピラミッドからの特徴抽出

    method:
      'raw': 論文[2]準拠 — 各階層の画素値をflatten連結
             64²+32²+16²+8²+4² = 5456次元
      'hog': 比較用 — HOG特徴抽出（改良版と同等）
             1764+324+36+0+80 = 2204次元
    """

    def __init__(self, method='raw'):
        self.method = method

    def extract_from_pyramid(self, pyramid):
        """ピラミッド全階層から特徴ベクトルを抽出"""
        if self.method == 'raw':
            return self._extract_raw(pyramid)
        elif self.method == 'hog':
            return self._extract_hog(pyramid)
        return self._extract_raw(pyramid)

    def _extract_raw(self, pyramid):
        """生画素値のflatten連結（論文[2]準拠）"""
        features = []
        for level in pyramid:
            features.append(level.flatten())
        return np.concatenate(features)

    def _extract_hog(self, pyramid):
        """HOG特徴（比較用）"""
        from skimage.feature import hog
        features = []
        for level in pyramid:
            if level.shape[0] >= 16 and level.shape[1] >= 16:
                h = hog(level, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=False,
                        feature_vector=True)
                features.append(h)
            elif level.size > 0:
                features.append(level.flatten())
        return np.concatenate(features) if features else np.array([])


# ============================================================================
# 第5部: CS-PNN分類器（論文[1] Algorithm 1-4、設定可変）
# ============================================================================

class CSPNNClassifier:
    """
    CS-PNN分類器（設定可変）

    論文[1]のAlgorithm 1-4を実装。以下のオプションを比較可能:

    sigma_method:
      'cspnn_dynamic': σ = dmax/k — 論文[1]式(3)、サンプルごとに動的計算
      'origpnn_static': σ = D_max/Nc — Morita et al. (2023)式(2)、全サンプル共通
      'fixed': 固定値

    aggregation:
      'sum':  ok = Σ hj — 論文[1]のCS-PNN
      'mean': ok = (1/Mk) Σ hj — オリジナルPNN

    centroid_update:
      'update':    cJ = (cJ + x)/2 — 論文[1]式(4)
      'no_update': 何もしない

    class_count_method:
      'dynamic': q=1から開始、クラス出現ごとにインクリメント — 論文[2]クラス数設定法1
      'fixed':   q=Nc（全クラス数）で固定 — 論文[2]クラス数設定法2
    """

    def __init__(self, sigma_method='cspnn_dynamic', aggregation='sum',
                 centroid_update='update', class_count_method='dynamic',
                 fixed_sigma=1.0):
        self.sigma_method = sigma_method
        self.aggregation = aggregation
        self.centroid_update = centroid_update
        self.class_count_method = class_count_method
        self.fixed_sigma = fixed_sigma

        self.n_classes = 0
        self._all_centroids = None
        self._class_ranges = None
        self._static_sigma = None  # origpnn_static用

    def fit(self, X, y):
        """
        Algorithm 1: Construction/Reconstruction of a CS-PNN

        ワンパスで訓練データを処理しネットワークを自動構築する。
        """
        unique_classes = np.unique(y)
        total_classes = len(unique_classes)

        # セントロイド管理
        centroids = []
        centroid_class = []
        known_classes = []

        # --- 初期化 (Algorithm 1, lines 2-5) ---
        centroids.append(X[0].copy())
        centroid_class.append(y[0])
        known_classes.append(y[0])

        # --- 残りのサンプルを順に処理 (Algorithm 1, lines 9-26) ---
        for i in range(1, len(X)):
            x_i = X[i]
            t_i = y[i]

            # 新クラスの場合: 出力ユニット＋RBFを追加 (lines 10-12)
            if t_i not in known_classes:
                known_classes.append(t_i)
                centroids.append(x_i.copy())
                centroid_class.append(t_i)
                continue

            # 既知クラスの場合: 分類を試みる (lines 13-25)
            C = np.array(centroids)

            # dmax: 入力と全セントロイド間の最大距離 (line 14)
            dists = np.linalg.norm(C - x_i, axis=1)
            dmax = np.max(dists)

            # σの決定 (line 16, 式(3))
            sigma = self._compute_sigma_train(dmax, known_classes, total_classes)

            # RBF活性値 hj(x) = exp(-||x - cj||² / σ²) (line 17)
            dists_sq = dists ** 2
            sigma_sq = sigma * sigma + 1e-30
            acts = np.exp(-dists_sq / sigma_sq)

            # 各SubNetの出力 (line 18)
            class_labels_arr = np.array(centroid_class)
            class_scores = {}
            for cls in known_classes:
                mask = (class_labels_arr == cls)
                if self.aggregation == 'sum':
                    class_scores[cls] = np.sum(acts[mask])
                else:  # mean
                    n_units = np.sum(mask)
                    class_scores[cls] = np.sum(acts[mask]) / max(n_units, 1)

            # 最大出力のクラス K = argmax(ok) (line 18)
            K = max(class_scores, key=class_scores.get)

            if K != t_i:
                # 誤分類 → 新RBFを追加 (lines 19-20)
                centroids.append(x_i.copy())
                centroid_class.append(t_i)
            else:
                # 正分類 → セントロイド更新 (lines 22-23)
                if self.centroid_update == 'update':
                    mask = (class_labels_arr == K)
                    indices_in_K = np.where(mask)[0]
                    acts_in_K = acts[indices_in_K]
                    J = indices_in_K[np.argmax(acts_in_K)]
                    # 式(4): cJ = (cJ + x) / 2
                    centroids[J] = (centroids[J] + x_i) / 2.0

        # --- 構築完了: キャッシュ作成 ---
        self._finalize(centroids, centroid_class)

        # origpnn_static: 構築後にD_maxを計算
        if self.sigma_method == 'origpnn_static':
            self._compute_static_sigma()

    def _compute_sigma_train(self, dmax, known_classes, total_classes):
        """訓練時のσ計算"""
        if self.sigma_method == 'cspnn_dynamic':
            if self.class_count_method == 'dynamic':
                k = len(known_classes)
            else:  # fixed
                k = total_classes
            return dmax / k if (dmax > 0 and k > 0) else 1e-10

        elif self.sigma_method == 'origpnn_static':
            # 訓練中は暫定的にdmax/Ncを使用
            k = total_classes
            return dmax / k if (dmax > 0 and k > 0) else 1e-10

        elif self.sigma_method == 'fixed':
            return self.fixed_sigma

        return 1e-10

    def _compute_static_sigma(self):
        """origpnn_static用: 全セントロイド間のD_maxからσを計算"""
        C = self._all_centroids
        if len(C) > 3000:
            # メモリ節約: ランダムサブサンプル
            idx = np.random.RandomState(42).choice(len(C), 3000, replace=False)
            C_sub = C[idx]
        else:
            C_sub = C
        dists = cdist(C_sub, C_sub, metric='euclidean')
        D_max = np.max(dists)
        self._static_sigma = D_max / self.n_classes if self.n_classes > 0 else 1.0

    def _finalize(self, centroids, centroid_class):
        """構築結果からキャッシュを作成"""
        class_centroids = {}
        for c, cls in zip(centroids, centroid_class):
            if cls not in class_centroids:
                class_centroids[cls] = []
            class_centroids[cls].append(c)

        arrays, ranges = [], {}
        idx = 0
        for cls in sorted(class_centroids.keys()):
            arr = np.array(class_centroids[cls])
            n = len(arr)
            ranges[cls] = (idx, idx + n)
            arrays.append(arr)
            idx += n

        self._all_centroids = np.vstack(arrays)
        self._class_ranges = ranges
        self.n_classes = len(ranges)

    def predict_scores(self, X):
        """
        Algorithm 4: Testing of a CS-PNN（バッチ対応）

        各テストサンプルについて:
          - dmax計算 → σ設定 → RBF活性値 → クラススコア
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        all_c = self._all_centroids
        n_samples = len(X)

        # 距離行列 (n_samples, n_centroids)
        dists = cdist(X, all_c, metric='euclidean')
        dists_sq = dists ** 2

        # σの計算
        if self.sigma_method == 'cspnn_dynamic':
            # 各サンプルごとのdmax (Algorithm 4, line 3)
            dmax = np.max(dists, axis=1)  # (n_samples,)
            k = self.n_classes
            sigma = np.maximum(dmax / k, 1e-30)  # (n_samples,)
            sigma_sq = (sigma * sigma)[:, np.newaxis]  # (n_samples, 1)
        elif self.sigma_method == 'origpnn_static':
            sigma = self._static_sigma
            sigma_sq = np.full((n_samples, 1), sigma * sigma)
        elif self.sigma_method == 'fixed':
            sigma_sq = np.full((n_samples, 1), self.fixed_sigma ** 2)
        else:
            sigma_sq = np.ones((n_samples, 1))

        # RBF活性値
        acts = np.exp(-dists_sq / (sigma_sq + 1e-30))

        # クラスごとのスコア
        sorted_classes = sorted(self._class_ranges.keys())
        n_cls = len(sorted_classes)
        scores = np.zeros((n_samples, n_cls))

        for c_idx, cls in enumerate(sorted_classes):
            start, end = self._class_ranges[cls]
            if self.aggregation == 'sum':
                scores[:, c_idx] = np.sum(acts[:, start:end], axis=1)
            else:  # mean
                n_units = end - start
                scores[:, c_idx] = np.sum(acts[:, start:end], axis=1) / max(n_units, 1)

        return scores

    def predict(self, x):
        """単一サンプル予測"""
        scores = self.predict_scores(x)
        sorted_classes = sorted(self._class_ranges.keys())
        return sorted_classes[np.argmax(scores[0])]

    def predict_batch(self, X):
        """バッチ予測"""
        scores = self.predict_scores(X)
        sorted_classes = sorted(self._class_ranges.keys())
        return np.array([sorted_classes[i] for i in np.argmax(scores, axis=1)])

    def get_info(self):
        """モデル情報"""
        info = {
            'sigma_method': self.sigma_method,
            'aggregation': self.aggregation,
            'centroid_update': self.centroid_update,
            'class_count_method': self.class_count_method,
            'n_classes': self.n_classes,
            'n_centroids': len(self._all_centroids) if self._all_centroids is not None else 0,
        }
        if self.sigma_method == 'origpnn_static' and self._static_sigma is not None:
            info['static_sigma'] = f'{self._static_sigma:.4f}'
        return info


# ============================================================================
# 第6部: 統合パイプライン
# ============================================================================

class HierarchicalCSPNN:
    """
    CS-PNN × 階層構造的画像パターン認識器（統合パイプライン）

    config辞書で全設定を制御:
      pyramid_method: 'quantize' | 'gaussian'
      feature_method: 'raw' | 'hog'
      sigma_method:   'cspnn_dynamic' | 'origpnn_static' | 'fixed'
      aggregation:    'sum' | 'mean'
      centroid_update: 'update' | 'no_update'
      class_count_method: 'dynamic' | 'fixed'
      voting:         'weighted_score' | 'majority_vote' | 'whole_only'
      normalization:  'zscore' | 'none'
      norm_range:     'pm1' | '01'
      pyramid_levels: int (default 5)
      target_size:    int (default 64)
    """

    DEFAULT_CONFIG = {
        'pyramid_method': 'quantize',
        'feature_method': 'raw',
        'sigma_method': 'cspnn_dynamic',
        'aggregation': 'sum',
        'centroid_update': 'update',
        'class_count_method': 'dynamic',
        'voting': 'weighted_score',
        'normalization': 'zscore',
        'norm_range': 'pm1',
        'pyramid_levels': 5,
        'target_size': 64,
        'n_groups': 'auto',  # hierarchical投票用: 'auto' または整数
    }

    def __init__(self, config=None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        c = self.config

        # コンポーネント初期化
        self.pyramid_gen = PyramidGenerator(c['pyramid_levels'], c['pyramid_method'])
        self.splitter = LeftRightSplitter()
        self.feature_ext = FeatureExtractor(c['feature_method'])

        # 分類器・スケーラー
        self.classifiers = {}
        self.scalers = {}
        self.trained = False

    def _make_classifier(self):
        """設定に基づきCSPNNClassifierを生成"""
        c = self.config
        return CSPNNClassifier(
            sigma_method=c['sigma_method'],
            aggregation=c['aggregation'],
            centroid_update=c['centroid_update'],
            class_count_method=c['class_count_method'],
        )

    def _preprocess_image(self, img):
        """画像の前処理"""
        return ImagePreprocessor.preprocess(
            img,
            target_size=self.config['target_size'],
            norm=self.config['norm_range']
        )

    def _extract_features_single(self, img):
        """1画像から全体・左・右の特徴を抽出"""
        proc = self._preprocess_image(img)
        left, right = self.splitter.split(proc)

        # 各パートのピラミッド生成・特徴抽出
        feat_whole = self.feature_ext.extract_from_pyramid(
            self.pyramid_gen.generate(proc))
        feat_left = self.feature_ext.extract_from_pyramid(
            self.pyramid_gen.generate(left))
        feat_right = self.feature_ext.extract_from_pyramid(
            self.pyramid_gen.generate(right))

        return feat_whole, feat_left, feat_right

    def _extract_all_features(self, images):
        """全画像から特徴を一括抽出"""
        whole_f, left_f, right_f = [], [], []
        for i, img in enumerate(images):
            fw, fl, fr = self._extract_features_single(img)
            whole_f.append(fw)
            left_f.append(fl)
            right_f.append(fr)
            if (i + 1) % 2000 == 0:
                print(f"    特徴抽出: {i+1}/{len(images)}")

        return {
            'whole': np.array(whole_f),
            'left': np.array(left_f),
            'right': np.array(right_f),
        }

    def _discover_groups(self, left_features, labels):
        """
        左特徴から偏グループを自動発見する

        手順:
          1. 各クラスの左特徴の平均ベクトル（クラスプロトタイプ）を計算
          2. クラスプロトタイプ間の距離行列を計算
          3. 凝集型階層的クラスタリングでグループを自動決定

        Returns:
            label_to_group: dict — 文字ラベル → グループID
            n_groups: int — 発見されたグループ数
        """
        unique_labels = np.unique(labels)

        # 各クラスの左特徴の平均ベクトルを計算
        prototypes = []
        for lbl in unique_labels:
            mask = (labels == lbl)
            proto = np.mean(left_features[mask], axis=0)
            prototypes.append(proto)
        prototypes = np.array(prototypes)

        # 凝集型階層的クラスタリング
        dist_matrix = pdist(prototypes, metric='euclidean')
        Z = linkage(dist_matrix, method='ward')

        n_groups_cfg = self.config.get('n_groups', 'auto')

        if n_groups_cfg == 'auto':
            # 自動決定: クラスタ間距離の中央値を閾値として使用
            threshold = np.median(Z[:, 2]) * 1.0
            group_ids = fcluster(Z, t=threshold, criterion='distance')
        else:
            # 指定グループ数に分割
            group_ids = fcluster(Z, t=int(n_groups_cfg), criterion='maxclust')

        # ラベル→グループIDのマッピングを構築
        label_to_group = {}
        for i, lbl in enumerate(unique_labels):
            label_to_group[lbl] = int(group_ids[i])

        n_groups = len(set(group_ids))
        return label_to_group, n_groups

    def train(self, images, labels):
        """学習"""
        n_classes = len(np.unique(labels))

        # 1. 特徴抽出
        features = self._extract_all_features(images)

        # 2. 正規化
        if self.config['normalization'] == 'zscore':
            for part in ['whole', 'left', 'right']:
                sc = StandardScaler()
                features[part] = sc.fit_transform(features[part])
                self.scalers[part] = sc

        voting = self.config['voting']

        if voting == 'hierarchical':
            self._train_hierarchical(features, labels)
        else:
            # 3. 既存方式: 分類器訓練
            parts_to_train = ['whole', 'left', 'right']
            if voting == 'whole_only':
                parts_to_train = ['whole']

            for part in parts_to_train:
                clf = self._make_classifier()
                clf.fit(features[part], labels)
                self.classifiers[part] = clf

        self.trained = True

    def _train_hierarchical(self, features, labels):
        """
        階層的絞り込み分類の学習

        Stage 1: 左特徴 → 偏グループ（自動発見）
        Stage 2: グループごとに右特徴 → 文字クラス
        """
        # 偏グループの自動発見
        self._label_to_group, self._n_groups = self._discover_groups(
            features['left'], labels)

        # グループラベルの配列を作成
        group_labels = np.array([self._label_to_group[lbl] for lbl in labels])

        # グループの内容を表示
        groups = {}
        for lbl, gid in self._label_to_group.items():
            if gid not in groups:
                groups[gid] = []
            groups[gid].append(lbl)
        print(f"  自動発見グループ数: {self._n_groups}")
        for gid in sorted(groups.keys()):
            members = sorted(groups[gid])
            print(f"    グループ{gid}: {len(members)}クラス — {members[:10]}"
                  f"{'...' if len(members) > 10 else ''}")

        # Stage 1: 左分類器（左特徴 → グループID）
        stage1_clf = self._make_classifier()
        stage1_clf.fit(features['left'], group_labels)
        self.classifiers['stage1_left'] = stage1_clf
        print(f"  Stage 1 (左→グループ): RBF数={stage1_clf.get_info()['n_centroids']}")

        # Stage 2: グループごとの右分類器（右特徴 → 文字クラス）
        self._stage2_classifiers = {}
        for gid in sorted(groups.keys()):
            mask = (group_labels == gid)
            if np.sum(mask) == 0:
                continue
            right_feat_group = features['right'][mask]
            labels_group = labels[mask]

            # グループ内に1クラスしかない場合は分類器不要
            if len(np.unique(labels_group)) == 1:
                self._stage2_classifiers[gid] = {
                    'type': 'single',
                    'label': np.unique(labels_group)[0]
                }
                print(f"  Stage 2 グループ{gid}: 1クラスのみ — 分類器不要")
            else:
                clf = self._make_classifier()
                clf.fit(right_feat_group, labels_group)
                self._stage2_classifiers[gid] = {
                    'type': 'classifier',
                    'clf': clf
                }
                info = clf.get_info()
                print(f"  Stage 2 グループ{gid}: {info['n_classes']}クラス, "
                      f"RBF数={info['n_centroids']}")

    def predict_batch(self, images):
        """バッチ予測"""
        if not self.trained:
            raise RuntimeError("モデルが訓練されていません")

        # 特徴抽出
        features = self._extract_all_features(images)

        # 正規化
        if self.config['normalization'] == 'zscore':
            for part in features:
                if part in self.scalers:
                    features[part] = self.scalers[part].transform(features[part])

        voting = self.config['voting']
        sorted_classes = sorted(self.classifiers['whole']._class_ranges.keys())

        if voting == 'whole_only':
            scores = self.classifiers['whole'].predict_scores(features['whole'])
            return np.array([sorted_classes[i] for i in np.argmax(scores, axis=1)])

        elif voting == 'weighted_score':
            # 全体×2 + 右×1（正規化後加重統合）
            scores_w = self.classifiers['whole'].predict_scores(features['whole'])
            scores_r = self.classifiers['right'].predict_scores(features['right'])

            # 行ごとに正規化（合計=1の確率分布化）
            scores_w = scores_w / (scores_w.sum(axis=1, keepdims=True) + 1e-30)
            scores_r = scores_r / (scores_r.sum(axis=1, keepdims=True) + 1e-30)

            combined = 2.0 * scores_w + 1.0 * scores_r
            return np.array([sorted_classes[i] for i in np.argmax(combined, axis=1)])

        elif voting == 'majority_vote':
            # 3分類器の多数決
            preds_w = self.classifiers['whole'].predict_batch(features['whole'])
            preds_l = self.classifiers['left'].predict_batch(features['left'])
            preds_r = self.classifiers['right'].predict_batch(features['right'])

            results = []
            for pw, pl, pr in zip(preds_w, preds_l, preds_r):
                votes = Counter([pw, pl, pr])
                results.append(votes.most_common(1)[0][0])
            return np.array(results)

        elif voting == 'hierarchical':
            return self._predict_hierarchical(features)

        return np.zeros(len(images), dtype=int)

    def _predict_hierarchical(self, features):
        """
        階層的絞り込み分類のテスト

        Stage 1: 左分類器でグループを予測
        Stage 2: グループに対応する右分類器で文字クラスを予測
        """
        n_samples = len(features['left'])

        # Stage 1: グループ予測
        group_preds = self.classifiers['stage1_left'].predict_batch(features['left'])

        # Stage 2: グループごとに右分類器で文字クラスを予測
        results = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            gid = group_preds[i]
            if gid in self._stage2_classifiers:
                s2 = self._stage2_classifiers[gid]
                if s2['type'] == 'single':
                    results[i] = s2['label']
                else:
                    results[i] = s2['clf'].predict(features['right'][i])
            else:
                # 未知のグループ（学習時に見なかったグループID）
                # → 最も近いグループにフォールバック
                known_gids = list(self._stage2_classifiers.keys())
                results[i] = self._stage2_classifiers[known_gids[0]].get(
                    'label', 0) if self._stage2_classifiers[known_gids[0]]['type'] == 'single' \
                    else self._stage2_classifiers[known_gids[0]]['clf'].predict(
                        features['right'][i])

        return results

    def evaluate(self, images, labels, verbose=True):
        """評価（各パート個別精度も計算）"""
        if not self.trained:
            raise RuntimeError("モデルが訓練されていません")

        # 特徴抽出
        features = self._extract_all_features(images)

        if self.config['normalization'] == 'zscore':
            for part in features:
                if part in self.scalers:
                    features[part] = self.scalers[part].transform(features[part])

        part_accuracies = {}

        if self.config['voting'] == 'hierarchical':
            # Stage 1 精度（グループ分類）
            group_preds = self.classifiers['stage1_left'].predict_batch(
                features['left'])
            true_groups = np.array([self._label_to_group.get(lbl, -1)
                                    for lbl in labels])
            stage1_acc = 100.0 * np.sum(group_preds == true_groups) / len(labels)
            part_accuracies['stage1_group'] = stage1_acc

            # 統合精度
            predictions = self._predict_hierarchical(features)
            final_acc = 100.0 * np.sum(predictions == labels) / len(labels)
            part_accuracies['combined'] = final_acc

            # 各パート表示用にダミー値
            part_accuracies['whole'] = 0
            part_accuracies['left'] = stage1_acc
            part_accuracies['right'] = final_acc
        else:
            # 既存方式
            for part in self.classifiers:
                if part.startswith('stage'):
                    continue
                feat_key = part
                if feat_key in features:
                    preds = self.classifiers[part].predict_batch(features[feat_key])
                    acc = 100.0 * np.sum(preds == labels) / len(labels)
                    part_accuracies[part] = acc

            predictions = self.predict_batch(images)
            final_acc = 100.0 * np.sum(predictions == labels) / len(labels)
            part_accuracies['combined'] = final_acc

        return part_accuracies, predictions

    def get_config_name(self):
        """設定の短縮名"""
        c = self.config
        parts = [
            c['pyramid_method'][:5],
            c['feature_method'],
            c['sigma_method'].split('_')[0],
            c['aggregation'],
            c['centroid_update'][:3],
            c['class_count_method'][:3],
            c['voting'].split('_')[0],
            c['normalization'][:3],
        ]
        return '_'.join(parts)

    def get_summary(self):
        """モデルサマリ"""
        info = {'config': self.config}
        if self.trained:
            total_centroids = 0
            for part, clf in self.classifiers.items():
                ci = clf.get_info()
                info[f'{part}_centroids'] = ci['n_centroids']
                total_centroids += ci['n_centroids']
            if hasattr(self, '_stage2_classifiers'):
                for gid, s2 in self._stage2_classifiers.items():
                    if s2['type'] == 'classifier':
                        ci = s2['clf'].get_info()
                        info[f'stage2_g{gid}_centroids'] = ci['n_centroids']
                        total_centroids += ci['n_centroids']
            info['total_centroids'] = total_centroids
            if hasattr(self, '_n_groups'):
                info['n_groups'] = self._n_groups
        return info


# ============================================================================
# 第7部: 実験実行
# ============================================================================

def load_etl8b_data(base_path, target_classes, max_samples=None):
    """ETL8Bデータ読み込み"""
    images, labels = [], []
    for class_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(base_path, class_hex)
        if not os.path.exists(class_dir):
            print(f"  警告: {class_dir} が存在しません")
            continue
        image_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
        if max_samples and len(image_files) > max_samples:
            np.random.seed(42)
            image_files = list(np.random.choice(image_files, max_samples, replace=False))
        for img_file in image_files:
            img = cv2.imread(os.path.join(class_dir, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(class_idx)
    return np.array(images), np.array(labels)


def generate_experiment_configs():
    """12構成の実験設定を生成"""
    configs = []

    # 1. 論文準拠フル構成
    configs.append(('paper_full', {
        'pyramid_method': 'quantize', 'feature_method': 'raw',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'sum',
        'centroid_update': 'update', 'class_count_method': 'dynamic',
        'voting': 'weighted_score', 'normalization': 'zscore',
    }))

    # 2. 論文準拠 + 多数決投票
    configs.append(('paper_majority', {
        'pyramid_method': 'quantize', 'feature_method': 'raw',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'sum',
        'centroid_update': 'update', 'class_count_method': 'dynamic',
        'voting': 'majority_vote', 'normalization': 'zscore',
    }))

    # 3. 論文準拠 + 全体のみ
    configs.append(('paper_whole_only', {
        'pyramid_method': 'quantize', 'feature_method': 'raw',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'sum',
        'centroid_update': 'update', 'class_count_method': 'dynamic',
        'voting': 'whole_only', 'normalization': 'zscore',
    }))

    # 4. セントロイド更新なし
    configs.append(('paper_no_update', {
        'pyramid_method': 'quantize', 'feature_method': 'raw',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'sum',
        'centroid_update': 'no_update', 'class_count_method': 'dynamic',
        'voting': 'weighted_score', 'normalization': 'zscore',
    }))

    # 5. mean集約
    configs.append(('paper_mean_agg', {
        'pyramid_method': 'quantize', 'feature_method': 'raw',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'mean',
        'centroid_update': 'update', 'class_count_method': 'dynamic',
        'voting': 'weighted_score', 'normalization': 'zscore',
    }))

    # 6. OrigPNN方式（静的σ + mean + 更新なし）
    configs.append(('origpnn_full', {
        'pyramid_method': 'quantize', 'feature_method': 'raw',
        'sigma_method': 'origpnn_static', 'aggregation': 'mean',
        'centroid_update': 'no_update', 'class_count_method': 'fixed',
        'voting': 'weighted_score', 'normalization': 'zscore',
    }))

    # 7. OrigPNN方式 + 多数決
    configs.append(('origpnn_majority', {
        'pyramid_method': 'quantize', 'feature_method': 'raw',
        'sigma_method': 'origpnn_static', 'aggregation': 'mean',
        'centroid_update': 'no_update', 'class_count_method': 'fixed',
        'voting': 'majority_vote', 'normalization': 'zscore',
    }))

    # 8. 固定クラス数（クラス数設定法2）
    configs.append(('paper_fixed_k', {
        'pyramid_method': 'quantize', 'feature_method': 'raw',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'sum',
        'centroid_update': 'update', 'class_count_method': 'fixed',
        'voting': 'weighted_score', 'normalization': 'zscore',
    }))

    # 9. 正規化なし
    configs.append(('paper_no_norm', {
        'pyramid_method': 'quantize', 'feature_method': 'raw',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'sum',
        'centroid_update': 'update', 'class_count_method': 'dynamic',
        'voting': 'weighted_score', 'normalization': 'none',
    }))

    # 10. ガウシアンピラミッド + 生画素
    configs.append(('gaussian_raw', {
        'pyramid_method': 'gaussian', 'feature_method': 'raw',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'sum',
        'centroid_update': 'update', 'class_count_method': 'dynamic',
        'voting': 'weighted_score', 'normalization': 'zscore',
    }))

    # 11. 量子化ピラミッド + HOG
    configs.append(('quant_hog', {
        'pyramid_method': 'quantize', 'feature_method': 'hog',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'sum',
        'centroid_update': 'update', 'class_count_method': 'dynamic',
        'voting': 'weighted_score', 'normalization': 'zscore',
    }))

    # 12. ガウシアン + HOG（改良版ベースライン相当）
    configs.append(('baseline_gauss_hog', {
        'pyramid_method': 'gaussian', 'feature_method': 'hog',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'sum',
        'centroid_update': 'update', 'class_count_method': 'dynamic',
        'voting': 'weighted_score', 'normalization': 'zscore',
    }))

    # --- 階層的絞り込み構成 (13-16) ---

    # 13. 階層的: 量子化 + 生画素
    configs.append(('hier_quant_raw', {
        'pyramid_method': 'quantize', 'feature_method': 'raw',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'sum',
        'centroid_update': 'update', 'class_count_method': 'dynamic',
        'voting': 'hierarchical', 'normalization': 'zscore',
    }))

    # 14. 階層的: 量子化 + HOG
    configs.append(('hier_quant_hog', {
        'pyramid_method': 'quantize', 'feature_method': 'hog',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'sum',
        'centroid_update': 'update', 'class_count_method': 'dynamic',
        'voting': 'hierarchical', 'normalization': 'zscore',
    }))

    # 15. 階層的: ガウシアン + HOG
    configs.append(('hier_gauss_hog', {
        'pyramid_method': 'gaussian', 'feature_method': 'hog',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'sum',
        'centroid_update': 'update', 'class_count_method': 'dynamic',
        'voting': 'hierarchical', 'normalization': 'zscore',
    }))

    # 16. 階層的: ガウシアン + 生画素
    configs.append(('hier_gauss_raw', {
        'pyramid_method': 'gaussian', 'feature_method': 'raw',
        'sigma_method': 'cspnn_dynamic', 'aggregation': 'sum',
        'centroid_update': 'update', 'class_count_method': 'dynamic',
        'voting': 'hierarchical', 'normalization': 'zscore',
    }))

    return configs


def run_experiments(n_classes=15, config_indices=None):
    """実験実行"""
    print("=" * 85)
    print(f" CS-PNN × 階層構造的画像パターン認識器 実験 ({n_classes}クラス)")
    print("=" * 85)

    # データ読み込み
    # 15クラス
    target_classes_15 = [
        '3a2c', '3a2e', '3a4e', '3a6e', '3a51',
        '3a60', '3a64', '3a72', '3b4f', '3b6b',
        '4f40', '3c23', '3d26', '3d3b', '3d3e'
    ]

    # 100クラス（lr_kanji_list_100_codes.pyから）
    target_classes_100_file = os.path.join('results', 'lr_selection',
                                           'lr_kanji_list_100_codes.py')
    if n_classes == 100 and os.path.exists(target_classes_100_file):
        # 100クラスリストを読み込み
        with open(target_classes_100_file, 'r') as f:
            content = f.read()
        # リストを抽出
        import ast
        start = content.find('[')
        end = content.find(']') + 1
        target_classes = ast.literal_eval(content[start:end])
    elif n_classes <= 15:
        target_classes = target_classes_15[:n_classes]
    else:
        target_classes = target_classes_15

    base_path = './ETL8B-img-full'
    images, labels = load_etl8b_data(base_path, target_classes)
    if len(images) == 0:
        print("エラー: データが読み込めませんでした")
        return

    print(f"データ: {len(images)}サンプル, {len(target_classes)}クラス")

    # 訓練・テスト分割
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"訓練: {len(train_imgs)}, テスト: {len(test_imgs)}")
    print()

    # 実験構成
    all_configs = generate_experiment_configs()
    if config_indices:
        all_configs = [(n, c) for i, (n, c) in enumerate(all_configs)
                       if (i + 1) in config_indices]

    # 結果格納
    results = []

    for idx, (name, cfg) in enumerate(all_configs):
        print(f"\n{'─' * 85}")
        print(f" [{idx+1}/{len(all_configs)}] {name}")
        print(f"   ピラミッド={cfg.get('pyramid_method','quantize')}, "
              f"特徴={cfg.get('feature_method','raw')}, "
              f"σ={cfg.get('sigma_method','cspnn_dynamic')}, "
              f"集約={cfg.get('aggregation','sum')}, "
              f"更新={cfg.get('centroid_update','update')}, "
              f"投票={cfg.get('voting','weighted_score')}")
        print(f"{'─' * 85}")

        model = HierarchicalCSPNN(cfg)

        # 学習
        t0 = time.time()
        model.train(train_imgs, train_labels)
        train_time = time.time() - t0

        # 評価
        t0 = time.time()
        part_accs, preds = model.evaluate(test_imgs, test_labels)
        eval_time = time.time() - t0

        # RBF数集計
        summary = model.get_summary()
        total_rbf = summary.get('total_centroids', 0)

        result = {
            'name': name,
            'config': cfg,
            'whole_acc': part_accs.get('whole', 0),
            'left_acc': part_accs.get('left', 0),
            'right_acc': part_accs.get('right', 0),
            'combined_acc': part_accs.get('combined', 0),
            'total_rbf': total_rbf,
            'train_time': train_time,
            'eval_time': eval_time,
        }
        results.append(result)

        print(f"\n  結果: 全体={part_accs.get('whole',0):.2f}%, "
              f"左={part_accs.get('left',0):.2f}%, "
              f"右={part_accs.get('right',0):.2f}%, "
              f"統合={part_accs['combined']:.2f}%")
        print(f"  RBF数={total_rbf}, 学習={train_time:.1f}秒, 評価={eval_time:.1f}秒")

    # 比較テーブル出力
    print("\n")
    print("=" * 100)
    print(f" 比較結果 ({n_classes}クラス, {len(test_imgs)}テストサンプル)")
    print("=" * 100)
    print(f" {'#':>2}  {'設定名':<22} {'全体':>7} {'左':>7} {'右':>7} "
          f"{'統合':>7} {'RBF数':>7} {'学習':>7} {'評価':>7}")
    print(f" {'':>2}  {'':<22} {'(%)':>7} {'(%)':>7} {'(%)':>7} "
          f"{'(%)':>7} {'':>7} {'(秒)':>7} {'(秒)':>7}")
    print("-" * 100)

    for i, r in enumerate(results):
        print(f" {i+1:>2}  {r['name']:<22} "
              f"{r['whole_acc']:>6.2f}% {r['left_acc']:>6.2f}% "
              f"{r['right_acc']:>6.2f}% {r['combined_acc']:>6.2f}% "
              f"{r['total_rbf']:>7d} {r['train_time']:>6.1f}s "
              f"{r['eval_time']:>6.1f}s")

    # 最良構成
    best = max(results, key=lambda x: x['combined_acc'])
    print("-" * 100)
    print(f" 最良: {best['name']} — 統合精度 {best['combined_acc']:.2f}%")
    print("=" * 100)

    # 結果保存
    save_dir = os.path.join('results', 'hierarchical_cspnn')
    os.makedirs(save_dir, exist_ok=True)

    # JSON保存
    results_json = []
    for r in results:
        rj = {k: v for k, v in r.items() if k != 'config'}
        rj['config'] = {k: str(v) for k, v in r['config'].items()}
        results_json.append(rj)

    json_path = os.path.join(save_dir, f'results_{n_classes}cls.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"\n結果を保存: {json_path}")

    return results


# ============================================================================
# メイン
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='CS-PNN × 階層構造的画像パターン認識器')
    parser.add_argument('--classes', type=int, default=15,
                        help='クラス数 (default: 15)')
    parser.add_argument('--config', type=int, nargs='*', default=None,
                        help='実行する構成番号 (例: --config 1 2 3)')
    args = parser.parse_args()

    run_experiments(n_classes=args.classes, config_indices=args.config)
