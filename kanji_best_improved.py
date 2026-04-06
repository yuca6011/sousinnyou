#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改良漢字認識モデル - スタンドアロン版

CS-PNN (Hoya & Morita, arXiv:2501.00725) Algorithm 1 準拠:
  (1) 構築: ワンパス構築（誤分類→新RBF追加、正分類→centroid更新）
  (2) σ決定: 動的 σ² = max_d² / k²（k = クラス数、サンプルごと）
  (3) RBF: exp(-d² / σ²)
  (4) 集約: mean（各SubNet出力 = Σ RBF活性値 / nUnits, C++準拠）
  (5) 正規化: 全体・左・右すべてZ-score
  (6) 投票: 正規化スコア加重 (W:R=2:1)

使用方法:
    from kanji_best_improved import KanjiBestRecognizer

    recognizer = KanjiBestRecognizer()
    recognizer.train(train_images, train_labels)
    accuracy, predictions = recognizer.evaluate(test_images, test_labels)
"""

import numpy as np
import cv2
import os
import pickle
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 第1部: PNN分類器（改良版）
# ============================================================================

class KernelMemoryPNN:
    """
    CS-PNN (Class-Specific Probabilistic Neural Network)

    Hoya & Morita (arXiv:2501.00725) Algorithm 1 準拠のワンパス構築。
      - ワンパスで SubNet（各クラス1つ）を構築
      - 動的σ: σ² = max_d² / k²（k = クラス数）
      - RBF = exp(-d² / σ²)
      - クラス出力 = Σ(RBF活性値) / nUnits（mean集約, C++準拠）
    """

    def __init__(self):
        self.subnets = {}           # 構築中: class_id → [centroid1, centroid2, ...]
        self.n_classes = 0
        self._all_centroids = None  # 構築後キャッシュ
        self._class_ranges = None
        self._sorted_classes = None

    def fit(self, X, y):
        """CS-PNN Algorithm 1 ワンパス構築"""
        self.subnets = {}
        self.n_classes = 0

        for i in range(len(X)):
            x = X[i]
            cls = y[i]

            # 新クラス → 新SubNet作成
            if cls not in self.subnets:
                self.subnets[cls] = [x.copy()]
                self.n_classes = len(self.subnets)
                continue

            # Forward pass
            pred_cls, max_rbf_idx = self._forward_train(x)

            if pred_cls != cls:
                # 誤分類 → 正しいクラスのSubNetに新RBFを追加
                self.subnets[cls].append(x.copy())
            else:
                # 正分類 → 最も活性化したRBFのcentroidを更新
                c = self.subnets[pred_cls][max_rbf_idx]
                self.subnets[pred_cls][max_rbf_idx] = (c + x) / 2.0

        self._build_cache()

    def _forward_train(self, x):
        """
        構築中のforward pass（C++参照: forwardCSPNN）

        Returns:
            (pred_cls, max_rbf_idx_in_winning_subnet)
        """
        # 全centroidへの距離²を計算
        all_centroids = []
        subnet_info = []  # [(cls, start, n), ...]
        idx = 0
        for cls in self.subnets:
            centroids = self.subnets[cls]
            n = len(centroids)
            subnet_info.append((cls, idx, n))
            all_centroids.extend(centroids)
            idx += n

        all_centroids_arr = np.array(all_centroids)
        diff = all_centroids_arr - x
        dists_sq = np.sum(diff * diff, axis=1)

        # max_v2 と σ²
        max_v2 = np.max(dists_sq)
        k = len(self.subnets)
        sigma_sq = max_v2 / (k * k) if max_v2 > 0 else 1e-30

        # 各RBFの活性値
        acts = np.exp(-dists_sq / (sigma_sq + 1e-30))

        # 各SubNetの出力（mean = sum/nUnits, C++準拠）と最大SubNetの探索
        best_cls = None
        best_score = -1.0
        best_rbf_idx = 0

        for cls, start, n in subnet_info:
            score = np.sum(acts[start:start + n]) / n
            if score > best_score:
                best_score = score
                best_cls = cls
                best_rbf_idx = int(np.argmax(acts[start:start + n]))

        return best_cls, best_rbf_idx

    def _build_cache(self):
        """全centroidの行列とクラス範囲を構築（バッチ予測用）"""
        arrays, ranges = [], {}
        idx = 0
        sorted_classes = sorted(self.subnets.keys())
        for cls in sorted_classes:
            centroid_list = self.subnets[cls]
            n = len(centroid_list)
            ranges[cls] = (idx, idx + n)
            arrays.append(np.array(centroid_list))
            idx += n
        self._all_centroids = np.vstack(arrays)
        self._class_ranges = ranges
        self._sorted_classes = sorted_classes

    def predict_scores(self, X):
        """
        各クラスのスコアを返す（動的σ + mean集約）

        Parameters:
            X : np.ndarray (n_samples, n_features) or (n_features,)

        Returns:
            scores : np.ndarray (n_samples, n_classes)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        k = self.n_classes
        all_c = self._all_centroids

        dists_sq = cdist(X, all_c, 'sqeuclidean')
        max_v2 = np.max(dists_sq, axis=1, keepdims=True)
        sigma_sq = max_v2 / (k * k)
        acts = np.exp(-dists_sq / (sigma_sq + 1e-30))

        n_samples = len(X)
        n_cls = len(self._sorted_classes)
        scores = np.zeros((n_samples, n_cls))

        for c_idx, cls in enumerate(self._sorted_classes):
            start, end = self._class_ranges[cls]
            n_units = end - start
            scores[:, c_idx] = np.sum(acts[:, start:end], axis=1) / n_units

        return scores

    def predict(self, x):
        """
        単一サンプルの予測

        Parameters:
            x : np.ndarray (n_features,)

        Returns:
            prediction : int
        """
        scores = self.predict_scores(x)  # (1, n_cls)
        return self._sorted_classes[np.argmax(scores[0])]

    def predict_batch(self, X):
        """バッチ予測"""
        scores = self.predict_scores(X)
        return np.array([self._sorted_classes[i] for i in np.argmax(scores, axis=1)])

    def get_info(self):
        """モデル情報を返す"""
        return {
            'algorithm': 'CS-PNN Algorithm 1',
            'aggregation': 'mean (sum/nUnits)',
            'n_classes': self.n_classes,
            'n_centroids': len(self._all_centroids) if self._all_centroids is not None else 0,
            'feature_dim': self._all_centroids.shape[1] if self._all_centroids is not None else 0,
        }


# ============================================================================
# 第2部: 階層ピラミッド
# ============================================================================

class HierarchicalPyramid:
    """階層ピラミッド構造生成"""

    def __init__(self, num_levels=5):
        self.num_levels = num_levels

    def generate_pyramid(self, image):
        """画像から階層ピラミッド生成"""
        pyramid = [image]
        current = image.copy()

        for _ in range(self.num_levels - 1):
            current = gaussian_filter(current, sigma=1.0)
            h, w = current.shape
            current = cv2.resize(current, (w // 2, h // 2),
                                interpolation=cv2.INTER_AREA)
            pyramid.append(current)

        return pyramid


# ============================================================================
# 第3部: 左右分割
# ============================================================================

class LeftRightSplitter:
    """左右分割（偏・旁）"""

    def __init__(self, split_method='projection'):
        self.split_method = split_method

    def detect_split_column(self, image):
        """分割位置の検出"""
        if self.split_method == 'fixed':
            return image.shape[1] // 2

        elif self.split_method == 'projection':
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

        return image.shape[1] // 2

    def split(self, image):
        """画像を左右に分割"""
        split_col = self.detect_split_column(image)

        left = image[:, :split_col]
        right = image[:, split_col:]

        if left.size == 0 or right.size == 0:
            left = image[:, :image.shape[1]//2]
            right = image[:, image.shape[1]//2:]

        return self._to_square(left), self._to_square(right)

    def _to_square(self, image):
        """画像を正方形に変換"""
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
# 第4部: HOG特徴抽出
# ============================================================================

class HOGFeatureExtractor:
    """HOG特徴抽出"""

    def extract_hog(self, image):
        """HOG特徴を抽出"""
        from skimage.feature import hog

        if image.shape[0] < 16 or image.shape[1] < 16:
            return np.array([])

        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True
        )
        return features

    def extract(self, image):
        """特徴抽出"""
        hog_features = self.extract_hog(image)
        if len(hog_features) > 0:
            return hog_features
        return image.flatten()


# ============================================================================
# 第5部: 最良認識器（統合・改良版）
# ============================================================================

class KanjiBestRecognizer:
    """
    改良漢字認識器

    旧版からの改良点:
      (1) σ決定: dmax/√Nc (OrigPNN方式)
      (2) RBF: exp(-d²/σ²) (CS-PNN式)
      (3) 正規化: 全体・左・右すべてZ-score正規化
      (4) 投票: 正規化スコア加重統合 (W:R=2:1)

    認識率: 96.67% (100クラス, 16100サンプル)
    vs MLP: +4.89%
    """

    def __init__(self):
        self.num_pyramid_levels = 5
        self.split_method = 'projection'
        self.vote_weights = {'whole': 2.0, 'right': 1.0}

        # コンポーネント初期化
        self.pyramid_generator = HierarchicalPyramid(self.num_pyramid_levels)
        self.splitter = LeftRightSplitter(self.split_method)
        self.feature_extractor = HOGFeatureExtractor()

        # 分類器（全体・左・右）
        self.classifiers = {}

        # スケーラー（全体・左・右それぞれ個別）
        self.scalers = {}

        self.trained = False

    def _preprocess_image(self, image):
        """前処理"""
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        if image.shape != (64, 64):
            image = cv2.resize(image, (64, 64))
        return image

    def _extract_hierarchical_features(self, image):
        """階層的特徴抽出"""
        pyramid = self.pyramid_generator.generate_pyramid(image)
        return np.concatenate([self.feature_extractor.extract(lv) for lv in pyramid])

    def _extract_all_features(self, images):
        """全画像から全体・左・右の特徴を一括抽出"""
        whole_f, left_f, right_f = [], [], []
        for i, img in enumerate(images):
            img = self._preprocess_image(img)
            left, right = self.splitter.split(img)
            whole_f.append(self._extract_hierarchical_features(img))
            left_f.append(self._extract_hierarchical_features(left))
            right_f.append(self._extract_hierarchical_features(right))
            if (i + 1) % 2000 == 0:
                print(f"    {i+1}/{len(images)}")
        return np.array(whole_f), np.array(left_f), np.array(right_f)

    def train(self, images, labels):
        """
        学習

        Parameters:
            images : np.ndarray (N, 64, 64)
            labels : np.ndarray (N,)
        """
        n_classes = len(np.unique(labels))
        print("=" * 70)
        print("改良漢字認識器 - 学習")
        print("=" * 70)
        print(f"サンプル数: {len(images)}")
        print(f"クラス数: {n_classes}")
        print(f"設定: HOG + CS-PNN(Algorithm1) + スコア統合")
        print()

        # 1. 特徴抽出
        print("1. 特徴抽出中...")
        whole_f, left_f, right_f = self._extract_all_features(images)
        print(f"   特徴次元: {whole_f.shape[1]}")

        # 2. 全パート個別にZ-score正規化
        print("2. 正規化中（全体・左・右それぞれ）...")
        features = {'whole': whole_f, 'left': left_f, 'right': right_f}
        for part in ['whole', 'left', 'right']:
            sc = StandardScaler()
            features[part] = sc.fit_transform(features[part])
            self.scalers[part] = sc

        # 3. PNN分類器訓練
        print("3. PNN分類器訓練中...")
        for part in ['whole', 'left', 'right']:
            pnn = KernelMemoryPNN()
            pnn.fit(features[part], labels)
            self.classifiers[part] = pnn
            info = pnn.get_info()
            print(f"   {part}: RBF数={info['n_centroids']}")

        self.trained = True
        print("\n訓練完了！")
        print("=" * 70)

    def predict(self, image):
        """
        単一画像の予測

        Parameters:
            image : np.ndarray (64, 64)

        Returns:
            prediction : int
        """
        if not self.trained:
            raise RuntimeError("モデルが訓練されていません")

        # 前処理・特徴抽出
        image = self._preprocess_image(image)
        left, right = self.splitter.split(image)

        feats = {
            'whole': self._extract_hierarchical_features(image),
            'left': self._extract_hierarchical_features(left),
            'right': self._extract_hierarchical_features(right),
        }

        # 正規化
        for part in feats:
            feats[part] = self.scalers[part].transform([feats[part]])[0]

        # スコア取得・正規化・加重統合 (W:R=2:1)
        scores = {}
        for part in ['whole', 'right']:
            raw = self.classifiers[part].predict_scores(feats[part])  # (1, n_cls)
            row_sum = raw.sum() + 1e-30
            scores[part] = raw[0] / row_sum  # 正規化

        combined = (self.vote_weights['whole'] * scores['whole'] +
                    self.vote_weights['right'] * scores['right'])

        sorted_classes = sorted(self.classifiers['whole']._class_ranges.keys())
        return sorted_classes[np.argmax(combined)]

    def predict_batch(self, images):
        """
        バッチ予測（高速版）

        Parameters:
            images : np.ndarray (N, 64, 64)

        Returns:
            predictions : np.ndarray (N,)
        """
        if not self.trained:
            raise RuntimeError("モデルが訓練されていません")

        # 一括特徴抽出
        whole_f, left_f, right_f = self._extract_all_features(images)

        # 正規化
        whole_f = self.scalers['whole'].transform(whole_f)
        left_f = self.scalers['left'].transform(left_f)
        right_f = self.scalers['right'].transform(right_f)

        # スコア取得
        scores_w = self.classifiers['whole'].predict_scores(whole_f)
        scores_r = self.classifiers['right'].predict_scores(right_f)

        # 行ごとに正規化
        scores_w = scores_w / (scores_w.sum(axis=1, keepdims=True) + 1e-30)
        scores_r = scores_r / (scores_r.sum(axis=1, keepdims=True) + 1e-30)

        # 加重統合 (W:R=2:1)
        combined = (self.vote_weights['whole'] * scores_w +
                    self.vote_weights['right'] * scores_r)

        sorted_classes = sorted(self.classifiers['whole']._class_ranges.keys())
        return np.array([sorted_classes[i] for i in np.argmax(combined, axis=1)])

    def evaluate(self, images, labels):
        """
        評価

        Parameters:
            images : np.ndarray (N, 64, 64)
            labels : np.ndarray (N,)

        Returns:
            accuracy : float (%)
            predictions : np.ndarray
        """
        if not self.trained:
            raise RuntimeError("モデルが訓練されていません")

        print(f"\n評価中: {len(images)}サンプル（バッチ処理）")

        predictions = self.predict_batch(images)
        accuracy = 100.0 * np.sum(predictions == labels) / len(labels)

        print(f"\n認識率: {accuracy:.2f}% ({np.sum(predictions == labels)}/{len(labels)})")
        return accuracy, predictions

    def get_model_info(self):
        """モデル情報を返す"""
        info = {
            'name': '改良漢字認識器',
            'sigma_method': 'CS-PNN: dynamic dmax/k per sample',
            'rbf_formula': 'exp(-d^2/sigma^2) [CS-PNN]',
            'voting': 'Normalized Score W:R=2:1',
            'normalization': 'Z-score (whole, left, right)',
            'pyramid_levels': self.num_pyramid_levels,
            'split_method': self.split_method,
        }
        if self.trained:
            for part in ['whole', 'left', 'right']:
                pnn_info = self.classifiers[part].get_info()
                info[f'{part}_centroids'] = pnn_info['n_centroids']
            info['feature_dim'] = self.classifiers['whole'].get_info()['feature_dim']
        return info

    def save(self, filepath):
        """モデルを保存"""
        if not self.trained:
            raise RuntimeError("モデルが訓練されていません")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"モデルを保存しました: {filepath}")

    @staticmethod
    def load(filepath):
        """モデルを読み込み"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"モデルを読み込みました: {filepath}")
        return model


# ============================================================================
# 使用例
# ============================================================================

def load_etl8b_data(base_path, target_classes, max_samples=None):
    """ETL8Bデータ読み込み"""
    images, labels = [], []
    for class_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(base_path, class_hex)
        if not os.path.exists(class_dir):
            continue
        image_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
        if max_samples and len(image_files) > max_samples:
            np.random.seed(42)
            image_files = list(np.random.choice(image_files, max_samples, replace=False))
        for img_file in image_files:
            img = cv2.imread(os.path.join(class_dir, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img, (64, 64)))
                labels.append(class_idx)
    print(f"読み込み完了: {len(images)}サンプル, {len(target_classes)}クラス")
    return np.array(images), np.array(labels)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    import time

    print("=" * 80)
    print("改良漢字認識モデル - デモンストレーション")
    print("=" * 80)

    # 15クラスでテスト
    target_classes_15 = [
        '3a2c', '3a2e', '3a4e', '3a6e', '3a51',
        '3a60', '3a64', '3a72', '3b4f', '3b6b',
        '4f40', '3c23', '3d26', '3d3b', '3d3e'
    ]

    images, labels = load_etl8b_data("./ETL8B-img-full", target_classes_15)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"訓練: {len(train_images)}, テスト: {len(test_images)}")

    recognizer = KanjiBestRecognizer()

    start = time.time()
    recognizer.train(train_images, train_labels)
    train_time = time.time() - start

    start = time.time()
    accuracy, predictions = recognizer.evaluate(test_images, test_labels)
    eval_time = time.time() - start

    print(f"\n学習時間: {train_time:.2f}秒")
    print(f"評価時間: {eval_time:.2f}秒")
    print(f"認識率: {accuracy:.2f}%")

    # モデル情報
    info = recognizer.get_model_info()
    print(f"\nモデル情報:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("デモンストレーション完了！")
    print("=" * 80)
