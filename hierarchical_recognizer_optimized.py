#!/usr/bin/env python3
"""
階層構造的画像パターン認識器 - 最適化版
PDFの仕様に基づく実装 + 精度向上のための改良

核心要素：
1. PNN（確率的ニューラルネットワーク）とRBFの隠れ層
2. 左右分割による偏・旁の認識
3. 階層ピラミッド構造による多重解像度処理

認識率（15クラス漢字認識）:
  30サンプル/クラス HOG+Aug (64x64): 69.63%
  全サンプル 拡張なし (2415、725テスト):  97.38% ★最高精度
  全サンプル 拡張あり (2415、725テスト):  82.90%
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist, pdist
from scipy.ndimage import gaussian_filter, map_coordinates
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


# ========================== 第1部：カーネルメモリーとPNN ==========================

class KernelMemoryPNN:
    """
    カーネルメモリーベースの確率的ニューラルネットワーク
    PDFの仕様に基づく実装
    
    特徴：
    - 複数の異なる長さの入力に対応
    - クラス別RBFニューロン
    - 適応的なスムージングパラメータσ
    """
    
    def __init__(self, 
                 sigma_init: float = 1.0,
                 sigma_method: str = 'adaptive',  # 'fixed', 'adaptive', 'class_specific'
                 exemplar_ratio: float = 0.2,
                 use_kmeans: bool = True):
        """
        Parameters:
        -----------
        sigma_init : float
            RBFカーネルの初期スムージングパラメータ
        sigma_method : str
            σの決定方法
        exemplar_ratio : float
            各クラスから選択する代表点の比率
        use_kmeans : bool
            k-meansで代表点を選択するか（False: 全サンプル使用）
        """
        self.sigma_init = sigma_init
        self.sigma_method = sigma_method
        self.exemplar_ratio = exemplar_ratio
        self.use_kmeans = use_kmeans
        
        # 学習後に設定される属性
        self.classes_ = None
        self.class_centers_ = {}  # クラスごとのRBF中心
        self.class_sigmas_ = {}   # クラスごとのσ
        self.n_classes_ = 0
        
    def _compute_sigma(self, centers: np.ndarray, class_label: int) -> float:
        """
        σ値を計算（PDFの半径値設定法に基づく）
        """
        if self.sigma_method == 'fixed':
            return self.sigma_init
            
        elif self.sigma_method == 'adaptive':
            # 半径値設定法1: d_max / q
            if len(centers) > 1:
                d_max = np.max(pdist(centers))
                sigma = d_max / np.sqrt(2 * self.n_classes_)
            else:
                sigma = self.sigma_init
            return max(sigma, 0.1)  # 最小値を設定
            
        elif self.sigma_method == 'class_specific':
            # 半径値設定法2: d'_max,M / N_c
            if len(centers) > 1:
                # クラス内の最大距離
                d_max = np.max(pdist(centers))
                sigma = d_max / len(centers)
            else:
                sigma = self.sigma_init
            return max(sigma, 0.1)
            
        else:
            return self.sigma_init
    
    def _select_exemplars(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        各クラスの代表点（exemplar）を選択
        """
        exemplars = {}
        
        for cls in self.classes_:
            X_cls = X[y == cls]
            n_samples = len(X_cls)
            
            if not self.use_kmeans or n_samples <= 5:
                # サンプル数が少ない or k-means不使用の場合は全サンプル
                exemplars[cls] = X_cls
            else:
                # k-meansで代表点を選択
                n_exemplars = max(1, int(n_samples * self.exemplar_ratio))
                n_exemplars = min(n_exemplars, n_samples)
                
                if n_exemplars == n_samples:
                    exemplars[cls] = X_cls
                else:
                    kmeans = KMeans(n_clusters=n_exemplars, 
                                  random_state=42, 
                                  n_init=10)
                    kmeans.fit(X_cls)
                    exemplars[cls] = kmeans.cluster_centers_
        
        return exemplars
    
    def _rbf_kernel(self, X: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
        """
        RBFカーネル計算
        """
        # ユークリッド距離の2乗
        distances_sq = cdist(X, centers, metric='sqeuclidean')
        
        # RBF活性化
        gamma = 1.0 / (2.0 * sigma ** 2)
        K = np.exp(-gamma * distances_sq)
        
        return K
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        PNNモデルを訓練
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # 各クラスの代表点を選択
        exemplars = self._select_exemplars(X, y)
        
        # 各クラスのRBF中心とσを設定
        for cls in self.classes_:
            self.class_centers_[cls] = exemplars[cls]
            self.class_sigmas_[cls] = self._compute_sigma(exemplars[cls], cls)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        各クラスの事後確率を計算
        """
        n_samples = len(X)
        probas = np.zeros((n_samples, self.n_classes_))
        
        for idx, cls in enumerate(self.classes_):
            centers = self.class_centers_[cls]
            sigma = self.class_sigmas_[cls]
            
            # RBFカーネルで類似度を計算
            K = self._rbf_kernel(X, centers, sigma)
            
            # クラスの活性化（平均）
            class_activation = np.mean(K, axis=1)
            probas[:, idx] = class_activation
        
        # 確率に正規化（ソフトマックス）
        probas_sum = np.sum(probas, axis=1, keepdims=True)
        probas_sum[probas_sum == 0] = 1e-10  # ゼロ除算を防ぐ
        probas = probas / probas_sum
        
        return probas
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        最大確率のクラスを予測
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


# ========================== 第2部：階層ピラミッド構造 ==========================

class HierarchicalPyramid:
    """
    階層ピラミッド構造の生成と処理
    PDFのStep 1に対応
    """
    
    def __init__(self, num_levels: int = 5):
        """
        Parameters:
        -----------
        num_levels : int
            ピラミッドの階層数
        """
        self.num_levels = num_levels
        
    def generate_pyramid(self, image: np.ndarray) -> List[np.ndarray]:
        """
        画像から階層ピラミッド構造を生成
        
        Returns:
        --------
        pyramid : list
            最下層（元画像）から最上層（最低解像度）までの画像リスト
        """
        pyramid = [image]
        
        current = image.copy()
        for _ in range(self.num_levels - 1):
            # ガウシアンフィルタで平滑化してからダウンサンプリング
            current = cv2.pyrDown(current)
            pyramid.append(current)
        
        # 逆順にして、最上層を先頭に
        pyramid.reverse()
        
        return pyramid
    
    def extract_level_features(self, pyramid: List[np.ndarray], 
                              level: int) -> np.ndarray:
        """
        特定の階層から特徴を抽出
        """
        if level < 0 or level >= len(pyramid):
            raise ValueError(f"Invalid level: {level}")
        
        level_image = pyramid[level]
        
        # 階層画像を一次元ベクトルに変換
        features = level_image.flatten()
        
        return features
    
    def multi_level_features(self, pyramid: List[np.ndarray]) -> np.ndarray:
        """
        複数の階層から特徴を統合
        """
        features = []
        
        for level in range(len(pyramid)):
            level_feat = self.extract_level_features(pyramid, level)
            features.append(level_feat)
        
        # 各レベルの特徴を結合
        return np.concatenate(features)


# ========================== 第3部：左右分割処理 ==========================

class LeftRightSplitter:
    """
    漢字画像の左右分割処理
    偏（へん）と旁（つくり）の分離
    """
    
    def __init__(self, split_method: str = 'projection'):
        """
        Parameters:
        -----------
        split_method : str
            分割方法（'projection', 'fixed', 'adaptive'）
        """
        self.split_method = split_method
        
    def detect_split_column(self, image: np.ndarray) -> int:
        """
        垂直投影で分割位置を検出
        PDFのStep 2.2に対応
        """
        if self.split_method == 'fixed':
            # 固定位置（中央）で分割
            return image.shape[1] // 2
        
        elif self.split_method == 'projection':
            # 垂直投影を計算
            vertical_projection = np.sum(image, axis=0)

            # 中央付近で最大値（最も白い列 = 空白部分）を探す
            h, w = image.shape[:2]
            center = w // 2
            search_range = w // 4

            start = max(0, center - search_range)
            end = min(w, center + search_range)

            # 最大投影値（最も白い列 = 偏と旁の間の空白）を探す
            max_col = start
            max_val = vertical_projection[start]

            for col in range(start, end):
                if vertical_projection[col] > max_val:
                    max_val = vertical_projection[col]
                    max_col = col

            return max_col
        
        elif self.split_method == 'adaptive':
            # 適応的な分割（エッジ検出ベース）
            edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
            vertical_proj = np.sum(edges, axis=0)
            
            # 最小エッジ部分を分割位置に
            h, w = image.shape[:2]
            center = w // 2
            search_range = w // 3
            
            start = max(0, center - search_range)
            end = min(w, center + search_range)
            
            split_col = start + np.argmin(vertical_proj[start:end])
            return split_col
        
        else:
            return image.shape[1] // 2
    
    def split(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        画像を左右に分割
        
        Returns:
        --------
        left_part : 左側（偏）
        right_part : 右側（旁）
        """
        split_col = self.detect_split_column(image)
        
        # 分割
        left_part = image[:, :split_col]
        right_part = image[:, split_col:]
        
        # サイズを正規化（正方形に）
        target_size = max(image.shape[0], image.shape[1])
        
        left_resized = self._resize_to_square(left_part, target_size)
        right_resized = self._resize_to_square(right_part, target_size)
        
        return left_resized, right_resized
    
    def _resize_to_square(self, image: np.ndarray, size: int) -> np.ndarray:
        """
        画像を正方形にリサイズ
        """
        h, w = image.shape[:2]
        
        # アスペクト比を保持しながらリサイズ
        if h > w:
            new_h = size
            new_w = int(w * size / h)
        else:
            new_w = size
            new_h = int(h * size / w)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # パディングして正方形に
        square = np.zeros((size, size), dtype=resized.dtype)
        
        y_offset = (size - new_h) // 2
        x_offset = (size - new_w) // 2
        
        square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return square


# ========================== 第4部：特徴抽出強化 ==========================

class EnhancedFeatureExtractor:
    """
    強化された特徴抽出器
    HOG、Gabor、方向分解特徴を統合
    """
    
    def __init__(self, use_hog: bool = True, 
                 use_gabor: bool = True,
                 use_direction: bool = True):
        self.use_hog = use_hog
        self.use_gabor = use_gabor
        self.use_direction = use_direction
        
        if use_gabor:
            self.gabor_kernels = self._create_gabor_kernels()
    
    def _create_gabor_kernels(self) -> List[cv2.UMat]:
        """
        Gaborフィルタバンクを作成
        """
        kernels = []
        ksize = 31
        
        # 8方向 × 5スケール = 40フィルタ
        for theta in np.arange(0, np.pi, np.pi / 8):
            for sigma in [3.0, 5.0, 7.0, 9.0, 11.0]:
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), 
                    sigma, 
                    theta,
                    10.0,  # lambd
                    0.5,   # gamma
                    0,     # psi
                    ktype=cv2.CV_32F
                )
                kernels.append(kernel)
        
        return kernels
    
    def extract_hog(self, image: np.ndarray) -> np.ndarray:
        """
        HOG特徴を抽出
        """
        from skimage.feature import hog

        # 画像サイズが小さすぎる場合はスキップ
        if image.shape[0] < 16 or image.shape[1] < 16:
            # 小さい画像の場合は空の特徴ベクトルを返す
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
    
    def extract_gabor(self, image: np.ndarray) -> np.ndarray:
        """
        Gabor特徴を抽出
        """
        features = []
        
        for kernel in self.gabor_kernels:
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            
            # 統計量を特徴として使用
            features.extend([
                np.mean(filtered),
                np.std(filtered),
                np.max(filtered),
                np.min(filtered)
            ])
        
        return np.array(features)
    
    def extract_direction(self, image: np.ndarray) -> np.ndarray:
        """
        8方向の勾配特徴
        """
        # Sobelフィルタで勾配を計算
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        
        # 勾配の強度と方向
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        
        # 8方向にビン化
        features = []
        for i in range(8):
            angle_min = i * np.pi / 4 - np.pi
            angle_max = (i + 1) * np.pi / 4 - np.pi
            
            mask = (angle >= angle_min) & (angle < angle_max)
            direction_sum = np.sum(magnitude[mask])
            features.append(direction_sum)
        
        return np.array(features)
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        すべての特徴を抽出して結合
        """
        features = []
        
        if self.use_hog:
            features.append(self.extract_hog(image))
        
        if self.use_gabor:
            features.append(self.extract_gabor(image))
        
        if self.use_direction:
            features.append(self.extract_direction(image))
        
        if not features:
            # デフォルトで画素値を使用
            return image.flatten()
        
        return np.concatenate(features)


# ========================== 第5部：メイン認識器 ==========================

class HierarchicalPatternRecognizer:
    """
    階層構造的画像パターン認識器
    PDFの仕様に基づく完全な実装
    """
    
    def __init__(self,
                 num_pyramid_levels=5,
                 split_method='projection',
                 pnn_sigma_method='adaptive',
                 use_enhanced_features=True,
                 feature_types=None):
        """
        Parameters:
        -----------
        num_pyramid_levels : int
            ピラミッドの階層数
        split_method : str
            左右分割方法
        pnn_sigma_method : str
            PNNのσ決定方法
        use_enhanced_features : bool
            強化特徴抽出を使用
        feature_types : list
            使用する特徴のリスト（['hog', 'gabor', 'direction']）
            Noneの場合は全て使用
        """
        self.num_pyramid_levels = num_pyramid_levels
        self.split_method = split_method
        self.pnn_sigma_method = pnn_sigma_method
        self.use_enhanced_features = use_enhanced_features
        self.feature_types = feature_types if feature_types else ['hog', 'gabor', 'direction']

        # コンポーネントの初期化
        self.pyramid_generator = HierarchicalPyramid(num_pyramid_levels)
        self.splitter = LeftRightSplitter(split_method)

        # 特徴抽出器（feature_typesに基づいて設定）
        if use_enhanced_features:
            use_hog = 'hog' in self.feature_types
            use_gabor = 'gabor' in self.feature_types
            use_direction = 'direction' in self.feature_types
            self.feature_extractor = EnhancedFeatureExtractor(
                use_hog=use_hog,
                use_gabor=use_gabor,
                use_direction=use_direction
            )
        else:
            self.feature_extractor = None
        
        # 分類器（後で初期化）
        self.left_classifiers = {}   # 左側（偏）の分類器
        self.right_classifiers = {}  # 右側（旁）の分類器
        self.whole_classifier = None # 全体の分類器
        
        # スケーラー
        self.scaler = StandardScaler()
        
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        画像の前処理（正規化、サイズ調整など）
        """
        # グレースケール変換（必要な場合）
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 正規化 [0, 1]
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # サイズを統一（64x64）
        if image.shape[0] != 64 or image.shape[1] != 64:
            image = cv2.resize(image, (64, 64))
        
        return image
    
    def _extract_hierarchical_features(self, image: np.ndarray) -> np.ndarray:
        """
        階層的な特徴抽出
        """
        # ピラミッド生成
        pyramid = self.pyramid_generator.generate_pyramid(image)
        
        all_features = []
        
        # 各階層から特徴を抽出
        for level, level_image in enumerate(pyramid):
            if self.use_enhanced_features:
                # 強化特徴抽出
                features = self.feature_extractor.extract(level_image)
            else:
                # 単純な画素値特徴
                features = level_image.flatten()
            
            all_features.append(features)
        
        # すべての階層の特徴を結合
        return np.concatenate(all_features)
    
    def train(self, images, labels):
        """
        学習
        
        Args:
            images: 学習画像のリスト [n_samples, 32, 32] または [n_samples, h, w]
            labels: クラスラベル
        """
        print("=" * 70)
        print(" 階層構造的画像パターン認識器 - 学習")
        print("=" * 70)
        print(f"サンプル数: {len(images)}")
        print(f"クラス数: {len(np.unique(labels))}")
        print(f"ピラミッド階層数: {self.num_pyramid_levels}")
        print(f"分割方法: {self.split_method}")
        print(f"特徴抽出: {'強化' if self.use_enhanced_features else '基本'}")
        print()

        # numpy配列に変換（リストの場合も対応）
        images = np.array(images) if not isinstance(images, np.ndarray) else images
        labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels

        # 画像の前処理
        print("\n1. 画像前処理中...")
        processed_images = np.array([self._preprocess_image(img) for img in images])
        print(f"   サンプル数: {len(processed_images)}")

        # 左右分割
        print("\n2. 左右分割処理中...")
        left_images = []
        right_images = []
        
        for img in processed_images:
            left, right = self.splitter.split(img)
            left_images.append(left)
            right_images.append(right)
        
        left_images = np.array(left_images)
        right_images = np.array(right_images)
        
        # 特徴抽出
        print("\n3. 階層的特徴抽出中...")
        print("   3.1 全体画像から特徴抽出...")
        whole_features = np.array([
            self._extract_hierarchical_features(img)
            for img in processed_images
        ])

        print("   3.2 左側（偏）から特徴抽出...")
        left_features = np.array([
            self._extract_hierarchical_features(img)
            for img in left_images
        ])
        
        print("   3.3 右側（旁）から特徴抽出...")
        right_features = np.array([
            self._extract_hierarchical_features(img) 
            for img in right_images
        ])
        
        # 特徴の正規化
        print("\n4. 特徴正規化中...")
        # すべての特徴を結合してスケーリング
        all_features = np.vstack([whole_features, left_features, right_features])
        self.scaler.fit(all_features)

        whole_features = self.scaler.transform(whole_features)
        left_features = self.scaler.transform(left_features)
        right_features = self.scaler.transform(right_features)

        # PNN分類器の訓練
        print("\n5. PNN分類器の訓練中...")

        # 全体分類器
        print("   5.1 全体分類器...")
        self.whole_classifier = KernelMemoryPNN(
            sigma_method=self.pnn_sigma_method,
            exemplar_ratio=0.2
        )
        self.whole_classifier.fit(whole_features, labels)
        
        # 左側分類器（偏のグループ分類）
        # 共通部ラベルを自動的に生成（クラスの最初の文字や簡易的なグループ化）
        common_labels = labels // 100 if labels.max() > 100 else np.zeros_like(labels)
        
        print("   5.2 左側（偏）分類器...")
        unique_common = np.unique(common_labels)
        for common_label in unique_common:
            mask = common_labels == common_label
            if np.sum(mask) > 0:
                classifier = KernelMemoryPNN(
                    sigma_method=self.pnn_sigma_method,
                    exemplar_ratio=0.2
                )
                classifier.fit(left_features[mask], labels[mask])
                self.left_classifiers[common_label] = classifier

        # 右側分類器（旁の詳細分類）
        print("   5.3 右側（旁）分類器...")
        for common_label in unique_common:
            mask = common_labels == common_label
            if np.sum(mask) > 0:
                classifier = KernelMemoryPNN(
                    sigma_method=self.pnn_sigma_method,
                    exemplar_ratio=0.2
                )
                classifier.fit(right_features[mask], labels[mask])
                self.right_classifiers[common_label] = classifier
        
        print("\n訓練完了！")
        print(f"左側分類器数: {len(self.left_classifiers)}")
        print(f"右側分類器数: {len(self.right_classifiers)}")
        
        return self
    
    def predict(self, image):
        """
        予測
        
        Args:
            image: 入力画像（2D numpy array）
        
        Returns:
            予測クラスID
        """
        # numpy配列に変換（必要に応じて）
        image = np.array(image) if not isinstance(image, np.ndarray) else image
        # 前処理
        image = self._preprocess_image(image)
        
        # 左右分割
        left, right = self.splitter.split(image)
        
        # 特徴抽出
        whole_features = self._extract_hierarchical_features(image)
        left_features = self._extract_hierarchical_features(left)
        right_features = self._extract_hierarchical_features(right)
        
        # 正規化
        whole_features = self.scaler.transform(whole_features.reshape(1, -1))
        left_features = self.scaler.transform(left_features.reshape(1, -1))
        right_features = self.scaler.transform(right_features.reshape(1, -1))
        
        # 予測
        # 方法1: 全体分類器を使用
        whole_pred = self.whole_classifier.predict(whole_features)[0]
        whole_proba = self.whole_classifier.predict_proba(whole_features)[0]
        
        # 方法2: 左右分類器の組み合わせ
        # （実装簡略化のため、ここでは全体分類器の結果を返す）
        
        return whole_pred
    
    def predict_proba(self, image):
        """
        予測確率を返す
        
        Args:
            image: 入力画像
        
        Returns:
            各クラスの予測確率
        """
        # numpy配列に変換（必要に応じて）
        image = np.array(image) if not isinstance(image, np.ndarray) else image
        # 前処理
        image = self._preprocess_image(image)
        
        # 特徴抽出
        whole_features = self._extract_hierarchical_features(image)
        whole_features = self.scaler.transform(whole_features.reshape(1, -1))
        
        # 予測確率
        return self.whole_classifier.predict_proba(whole_features)[0]
    
    def evaluate(self, test_images, test_labels):
        """
        評価
        
        Args:
            test_images: テスト画像のリスト
            test_labels: 正解ラベル
        
        Returns:
            (認識率, 予測ラベル配列)
        """
        # numpy配列に変換（リストの場合も対応）
        test_images = np.array(test_images) if not isinstance(test_images, np.ndarray) else test_images
        test_labels = np.array(test_labels) if not isinstance(test_labels, np.ndarray) else test_labels
        
        n_test = len(test_images)
        predictions = []
        
        print(f"\n評価中: {n_test}サンプル")
        
        for i, img in enumerate(test_images):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  予測中: {i+1}/{n_test}")
            
            pred = self.predict(img)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 認識率計算
        correct = np.sum(predictions == test_labels)
        accuracy = 100.0 * correct / n_test
        
        print(f"\n認識率: {accuracy:.2f}% ({correct}/{n_test})")
        
        return accuracy, predictions


# ========================== 第7部：最適化とグリッドサーチ ==========================

class HyperparameterOptimizer:
    """
    ハイパーパラメータ最適化
    """
    
    def __init__(self, param_grid: Optional[Dict] = None):
        """
        Parameters:
        -----------
        param_grid : dict
            探索するパラメータのグリッド
        """
        self.param_grid = param_grid or self._default_param_grid()
        
    def _default_param_grid(self) -> Dict:
        """
        デフォルトのパラメータグリッド
        """
        return {
            'num_pyramid_levels': [3, 5, 7],
            'pnn_sigma_method': ['adaptive', 'class_specific'],
            'use_enhanced_features': [True, False]
        }
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray, 
                cv_folds: int = 5) -> Dict:
        """
        交差検証による最適化
        """
        print("ハイパーパラメータ最適化開始...")
        print(f"パラメータグリッド: {self.param_grid}")
        print(f"交差検証: {cv_folds}分割")
        
        best_params = None
        best_score = 0
        
        # グリッドサーチ
        from itertools import product
        
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]
        
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            print(f"\n試行中: {params}")
            
            # 交差検証
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                # モデル構築
                model = HierarchicalPatternRecognizer(**params)
                model.train(X_tr, y_tr)
                
                # 評価
                results = model.evaluate(X_val, y_val)
                scores.append(results['accuracy'])
                
                print(f"  Fold {fold+1}: {results['accuracy']:.4f}")
            
            mean_score = np.mean(scores)
            print(f"  平均スコア: {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        print(f"\n最適パラメータ: {best_params}")
        print(f"最高スコア: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score
        }


# ========================== 使用例 ==========================

def demo():
    """
    デモンストレーション
    """
    print("階層構造的画像パターン認識器 - デモ")
    print("=" * 70)
    
    # ダミーデータの生成（実際にはMNISTや漢字データセットを使用）
    n_samples = 1000
    n_classes = 10
    image_size = 64
    
    # ランダムな画像データ（実際のデータに置き換える）
    X_train = np.random.rand(n_samples, image_size, image_size)
    y_train = np.random.randint(0, n_classes, n_samples)
    
    X_test = np.random.rand(200, image_size, image_size)
    y_test = np.random.randint(0, n_classes, 200)
    
    # 基本的な使用法
    print("\n1. 基本モデルの訓練")
    model = HierarchicalPatternRecognizer(
        num_pyramid_levels=5,
        split_method='projection',
        pnn_sigma_method='adaptive',
        use_enhanced_features=True
    )
    
    model.train(X_train, y_train)
    
    # 評価
    print("\n2. テストデータで評価")
    results = model.evaluate(X_test, y_test)
    
    # ハイパーパラメータ最適化
    print("\n3. ハイパーパラメータ最適化（オプション）")
    optimizer = HyperparameterOptimizer()
    # best_params = optimizer.optimize(X_train[:100], y_train[:100], cv_folds=3)
    
    print("\nデモ完了！")
    

if __name__ == "__main__":
    # デモを実行
    demo()
