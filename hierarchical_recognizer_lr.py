#!/usr/bin/env python3
"""
左右分割型階層構造的画像パターン認識器
漢字の偏（左側）と旁（右側）を別々に認識して組み合わせる

認識率（15クラス、各30サンプル、32x32画像、135テスト）:
  左右分割型認識器（現行版）: 40.74%
  ※元の階層型(73.33%)より低い。各クラスを独立偏グループとして扱うため
"""

import numpy as np
from hierarchical_recognizer import HierarchicalPatternRecognizer
import cv2

class LeftRightSplitRecognizer:
    """
    左右分割型認識器

    漢字を左側（偏）と右側（旁）に分割し、
    それぞれを階層構造的認識器で認識して組み合わせる
    """

    def __init__(self, num_pyramid_levels=5, lib_path='./hierarchical_ext.so'):
        """
        Args:
            num_pyramid_levels: ピラミッド階層数
            lib_path: C++ライブラリのパス
        """
        self.num_pyramid_levels = num_pyramid_levels
        self.lib_path = lib_path

        # 左側（偏）用の認識器
        self.left_recognizer = None
        # 右側（旁）用の認識器（偏グループごとに作成）
        self.right_recognizers = {}

        # クラス情報
        self.num_classes = 0
        self.class_to_lr = {}  # class_id -> (left_group, right_group)
        self.lr_to_class = {}  # (left_group, right_group) -> class_id

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

        # 中央付近で最大値（最も白い列）を探す
        height, width = image.shape
        center = width // 2
        search_start = max(0, center - width // 4)
        search_end = min(width, center + width // 4)

        search_range = range(search_start, search_end)
        if len(search_range) == 0:
            return center

        split_col = max(search_range, key=lambda x: vertical_projection[x])
        return split_col

    def split_left_right(self, image):
        """
        画像を左右に分割

        Args:
            image: 2D numpy array

        Returns:
            left_part, right_part: 左右の画像
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

    def train(self, images, labels):
        """
        学習

        Args:
            images: 学習画像のリスト
            labels: クラスラベル
        """
        n_samples = len(images)
        self.num_classes = len(np.unique(labels))

        print("="*70)
        print(" 左右分割型階層構造的画像パターン認識器 - 学習")
        print("="*70)
        print(f"サンプル数: {n_samples}")
        print(f"クラス数: {self.num_classes}")
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

        # 左側（偏）のグループ化
        # 簡易版: 各クラスの左側を独立したグループとして扱う
        print("\n左側（偏）の分類器を構築中...")
        left_labels = labels.copy()  # 各クラスの左側をそのままグループIDとする

        self.left_recognizer = HierarchicalPatternRecognizer(
            num_pyramid_levels=self.num_pyramid_levels,
            lib_path=self.lib_path
        )
        self.left_recognizer.train(left_images, left_labels)

        # 右側（旁）の分類器を構築（各偏グループごと）
        print("\n右側（旁）の分類器を構築中...")
        unique_left_groups = np.unique(left_labels)

        for left_group in unique_left_groups:
            print(f"  偏グループ {left_group} の処理中...")

            # このグループに属するサンプルを抽出
            mask = left_labels == left_group
            group_right_images = right_images[mask]
            group_labels = labels[mask]

            # 右側の認識器を作成
            right_recognizer = HierarchicalPatternRecognizer(
                num_pyramid_levels=self.num_pyramid_levels,
                lib_path=self.lib_path
            )
            right_recognizer.train(group_right_images, group_labels)

            self.right_recognizers[left_group] = right_recognizer

            # クラスマッピングを記録
            for class_id in np.unique(group_labels):
                self.class_to_lr[class_id] = (left_group, class_id)
                self.lr_to_class[(left_group, class_id)] = class_id

        print("\n" + "="*70)
        print(" 学習完了")
        print("="*70)
        print(f"偏グループ数: {len(self.right_recognizers)}")
        print(f"旁分類器数: {len(self.right_recognizers)}")

    def predict(self, image):
        """
        予測

        Args:
            image: 入力画像

        Returns:
            予測クラスID
        """
        if self.left_recognizer is None:
            raise RuntimeError("モデルが学習されていません")

        # 左右に分割
        left, right = self.split_left_right(image)

        # 左側（偏）を認識
        left_group = self.left_recognizer.predict(left)

        # 右側（旁）を認識
        if left_group in self.right_recognizers:
            right_recognizer = self.right_recognizers[left_group]
            class_id = right_recognizer.predict(right)
        else:
            # グループが見つからない場合は左グループをそのまま返す
            class_id = left_group

        return class_id

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
