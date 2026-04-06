#!/usr/bin/env python3
"""
マルチレベル左右分割型階層構造的画像パターン認識器
各レベル（8×8、16×16、32×32）で左右分割認識を行い、結果を統合

認識率（15クラス、各30サンプル、32x32画像、135テスト）:
  マルチレベル左右分割型: 40.74%（学習0.13秒、評価0.07秒）
"""

import numpy as np
from hierarchical_recognizer import HierarchicalPatternRecognizer
import cv2
from collections import Counter

class MultiLevelLeftRightRecognizer:
    """
    マルチレベル左右分割型認識器

    各解像度レベルで左右分割認識を行い、優先度付き統合で最終判定

    処理フロー：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    入力画像（32×32）
        ↓
    [左右分割]
        ↓
    左側          右側
        ↓            ↓
    各レベルで処理:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Level 4 (32×32): 左（共通部） × 右（非共通部） → クラス候補A
    Level 3 (16×16): 左（共通部） × 右（非共通部） → クラス候補B
    Level 2 (8×8):   左（共通部） × 右（非共通部） → クラス候補C
        ↓
    [統合判定] 優先度: Level 4 > Level 3 > Level 2
        ↓
    最終クラスID
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    def __init__(self, num_pyramid_levels=5, lib_path='./hierarchical_ext.so'):
        """
        Args:
            num_pyramid_levels: ピラミッド階層数（内部的に使用）
            lib_path: C++ライブラリのパス
        """
        self.num_pyramid_levels = num_pyramid_levels
        self.lib_path = lib_path

        # 各レベル（2, 3, 4）の認識器
        # level_sizes[level] = (height, width)
        self.level_sizes = {
            2: (8, 8),
            3: (16, 16),
            4: (32, 32)
        }

        # 左側（共通部）認識器: {level: recognizer}
        self.left_recognizers = {}

        # 右側（非共通部）認識器: {level: {left_group: recognizer}}
        self.right_recognizers = {}

        # クラス情報
        self.num_classes = 0

    def detect_split_column(self, image):
        """
        垂直投影を使って左右の分割位置を検出

        Args:
            image: 2D numpy array (0.0-1.0)

        Returns:
            split_col: 分割列の位置
        """
        # 垂直投影を計算
        vertical_projection = np.sum(image, axis=0)

        # 中央付近で最小値（最も黒い列）を探す
        height, width = image.shape
        center = width // 2
        search_start = max(0, center - width // 4)
        search_end = min(width, center + width // 4)

        search_range = range(search_start, search_end)
        if len(search_range) == 0:
            return center

        # 最小値（最も白い列）を探す
        split_col = max(search_range, key=lambda x: vertical_projection[x])
        return split_col

    def split_left_right(self, image):
        """
        画像を左右に分割

        Args:
            image: 2D numpy array

        Returns:
            left_part, right_part: 左右の画像（元のサイズを維持）
        """
        split_col = self.detect_split_column(image)
        left_part = image[:, :split_col]
        right_part = image[:, split_col+1:]

        # 正方形にリサイズ
        size = max(left_part.shape[0], left_part.shape[1],
                   right_part.shape[0], right_part.shape[1])

        left_square = np.ones((size, size), dtype=np.float64)
        right_square = np.ones((size, size), dtype=np.float64)

        left_square[:left_part.shape[0], :left_part.shape[1]] = left_part
        right_square[:right_part.shape[0], :right_part.shape[1]] = right_part

        # 32x32にリサイズ
        left_resized = cv2.resize(left_square, (32, 32))
        right_resized = cv2.resize(right_square, (32, 32))

        return left_resized, right_resized

    def resize_to_level(self, images, level):
        """
        画像を指定レベルのサイズにリサイズ

        Args:
            images: 画像配列 [n_samples, 32, 32]
            level: レベル（2, 3, 4）

        Returns:
            リサイズされた画像配列
        """
        target_size = self.level_sizes[level]
        resized = []

        for img in images:
            resized_img = cv2.resize(img, target_size)
            resized.append(resized_img)

        return np.array(resized)

    def train(self, images, labels):
        """
        学習

        Args:
            images: 学習画像のリスト [n_samples, 32, 32]
            labels: クラスラベル
        """
        n_samples = len(images)
        self.num_classes = len(np.unique(labels))

        print("="*70)
        print(" マルチレベル左右分割型階層構造的画像パターン認識器 - 学習")
        print("="*70)
        print(f"サンプル数: {n_samples}")
        print(f"クラス数: {self.num_classes}")
        print(f"処理レベル: Level 2 (8×8), Level 3 (16×16), Level 4 (32×32)")
        print()

        # 全画像を左右に分割
        print("画像を左右に分割中...")
        left_images = []
        right_images = []

        for i, img in enumerate(images):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  分割中: {i+1}/{n_samples}")
            left, right = self.split_left_right(img)
            left_images.append(left)
            right_images.append(right)

        left_images = np.array(left_images)
        right_images = np.array(right_images)

        # 各レベルで左側の認識器を構築
        left_labels = labels.copy()  # 各クラスの左側を独立したグループとして扱う

        # 各レベルで認識器を構築
        for level in [2, 3, 4]:
            print(f"\n{'='*70}")
            print(f" Level {level} ({self.level_sizes[level][0]}×{self.level_sizes[level][1]}) の認識器を構築")
            print(f"{'='*70}")

            # 左右の画像をこのレベルにリサイズ
            level_left_images = self.resize_to_level(left_images, level)
            level_right_images = self.resize_to_level(right_images, level)

            # 左側（共通部）認識器を学習
            print(f"\nLevel {level} - 左側（共通部）認識器を構築中...")
            print(f"  → {self.level_sizes[level][0]}×{self.level_sizes[level][1]}の特徴で偏グループを識別")

            # 共通部ラベル：全ての偏を1つの大きなグループとして扱う
            left_common_labels = np.zeros(len(left_labels), dtype=np.int32)

            left_recognizer = HierarchicalPatternRecognizer(
                num_pyramid_levels=self.num_pyramid_levels,
                lib_path=self.lib_path
            )
            left_recognizer.train(level_left_images, left_labels, left_common_labels)

            self.left_recognizers[level] = left_recognizer

            # 右側（非共通部）認識器を各偏グループごとに構築
            print(f"\nLevel {level} - 右側（非共通部）認識器を構築中...")
            print(f"  → {self.level_sizes[level][0]}×{self.level_sizes[level][1]}の特徴でクラスを識別")

            unique_left_groups = np.unique(left_labels)
            self.right_recognizers[level] = {}

            for left_group in unique_left_groups:
                if (left_group + 1) % 5 == 0 or left_group == 0:
                    print(f"  偏グループ {left_group} の処理中...")

                # このグループに属するサンプルを抽出
                mask = left_labels == left_group
                group_right_images = level_right_images[mask]
                group_labels = labels[mask]

                if len(group_right_images) == 0:
                    continue

                # 共通部ラベル
                right_common_labels = np.full(len(group_labels), left_group, dtype=np.int32)

                # 右側の認識器を作成
                right_recognizer = HierarchicalPatternRecognizer(
                    num_pyramid_levels=self.num_pyramid_levels,
                    lib_path=self.lib_path
                )
                right_recognizer.train(group_right_images, group_labels, right_common_labels)

                self.right_recognizers[level][left_group] = right_recognizer

        print("\n" + "="*70)
        print(" 学習完了")
        print("="*70)
        print(f"各レベルで左側認識器: {len(self.left_recognizers)}個")
        print(f"各レベルで右側認識器: {sum(len(r) for r in self.right_recognizers.values())}個")

    def predict(self, image):
        """
        階層構造的予測（マルチレベル統合）

        Args:
            image: 入力画像（32×32）

        Returns:
            予測クラスID
        """
        if not self.left_recognizers:
            raise RuntimeError("モデルが学習されていません")

        # 左右に分割
        left, right = self.split_left_right(image)

        # 各レベルで予測
        level_predictions = []

        for level in [4, 3, 2]:  # 優先度順（高解像度→低解像度）
            # このレベルにリサイズ
            level_left = cv2.resize(left, self.level_sizes[level])
            level_right = cv2.resize(right, self.level_sizes[level])

            # 左側（共通部）を認識
            left_group = self.left_recognizers[level].predict(level_left)

            # 右側（非共通部）を認識
            if left_group in self.right_recognizers[level]:
                right_recognizer = self.right_recognizers[level][left_group]
                class_id = right_recognizer.predict(level_right)
            else:
                # グループが見つからない場合は左グループをそのまま返す
                class_id = left_group

            level_predictions.append({
                'level': level,
                'class': class_id,
                'priority': level  # Level 4 > Level 3 > Level 2
            })

        # 統合判定
        final_class = self._integrate_predictions(level_predictions)
        return final_class

    def _integrate_predictions(self, level_predictions):
        """
        複数レベルの予測を統合して最終判定

        優先度付き多数決：
        - Level 4の結果に最大の重み
        - 一致しない場合は多数決
        - 同点の場合は高レベル優先

        Args:
            level_predictions: [{level, class, priority}, ...]

        Returns:
            最終クラスID
        """
        # 重み付き投票
        votes = Counter()

        for pred in level_predictions:
            level = pred['level']
            class_id = pred['class']

            # レベルに応じた重み
            if level == 4:
                weight = 3  # 32×32: 最も詳細
            elif level == 3:
                weight = 2  # 16×16: 中間
            else:  # level == 2
                weight = 1  # 8×8: 最も抽象的

            votes[class_id] += weight

        # 最多得票のクラスを返す
        if votes:
            return votes.most_common(1)[0][0]
        else:
            # フォールバック
            return level_predictions[0]['class']

    def evaluate(self, test_images, test_labels):
        """
        評価

        Args:
            test_images: テスト画像のリスト
            test_labels: 正解ラベル

        Returns:
            (認識率, 予測ラベル配列)
        """
        n_test = len(test_images)
        print(f"\n評価中: {n_test}サンプル")

        predictions = []
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
