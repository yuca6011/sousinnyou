#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
100クラス比較実験: C++版CS-PNN(生画素) vs Best Model(HOG+adaptive PNN)

C++版は pnn1.so を直接 ctypes で呼び出し、
生画素データ (64x64=4096次元) を [-1,1] 正規化して投入。
"""

import os
import sys
import time
import ctypes as ct
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

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
# ctypes 型定義（pnn1.py と同一）
# ============================================================================
c_int_p = ct.POINTER(ct.c_int)
c_double_p = ct.POINTER(ct.c_double)
_dp = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')


class SubNet(ct.Structure):
    _fields_ = (('cid', ct.c_int),
                ('nUnits', ct.c_int),
                ('nVecLen', ct.c_int),
                ('y', ct.c_double),
                ('c', ct.POINTER(c_double_p)),
                ('h', c_double_p),
                ('h_d2', c_double_p))


class PNN(ct.Structure):
    _fields_ = (('nSubNets', ct.c_int),
                ('subnet', ct.POINTER(SubNet)),
                ('r_den', ct.c_double))


_kept_refs = []  # GC防止用参照保持

def npMatrixToC(arr):
    """numpy配列をC++ポインタに変換（参照をグローバルに保持してGC防止）"""
    if len(arr.shape) == 1:
        _arr = np.ascontiguousarray(arr, dtype=np.float64)
        M = _arr.shape[0]
        N = 1
        retp = _arr.ctypes.data_as(c_double_p)
        _kept_refs.append(_arr)
    else:
        _arr = np.ascontiguousarray(arr, dtype=np.float64)
        M = _arr.shape[0]
        N = _arr.shape[1]
        retp = (_arr.__array_interface__['data'][0]
                + np.arange(M) * _arr.strides[0]).astype(np.uintp)
        _kept_refs.append(_arr)
        _kept_refs.append(retp)
    return retp, M, N


def user_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=(b != 0), casting='unsafe')


def normalize_minmax(train_x, test_x):
    """[-1, 1] 正規化（pnn1.py と同一方式）"""
    tr_max = np.max(train_x, axis=0)
    tr_min = np.min(train_x, axis=0)
    min_max_range = tr_max - tr_min
    train_x = (user_divide(train_x - tr_min, min_max_range) - 0.5) * 2
    test_x = (user_divide(test_x - tr_min, min_max_range) - 0.5) * 2
    return train_x, test_x


def make_split_data(x, y):
    """クラスごとに分割"""
    split_x, split_y = [], []
    for cls in range(max(y) + 1):
        idx = np.where(y == cls)
        split_x.append(x[idx])
        split_y.append(y[idx])
    return split_x, split_y


def make_data_even(tr_x, tr_y):
    """クラスをインターリーブ配置"""
    nXtr = [tr_x[i].shape[0] for i in range(len(tr_x))]
    max_n = max(nXtr)
    x_tr, y_tr = [], []
    for i in range(max_n):
        for j in range(len(nXtr)):
            if i < nXtr[j]:
                x_tr.append(tr_x[j][i])
                y_tr.append(tr_y[j][i])
    return np.array(x_tr), np.array(y_tr)


# ============================================================================
# データ読み込み
# ============================================================================
def load_etl8b_data(data_dir, target_classes):
    """ETL8Bデータ読み込み（生画素、平坦化）"""
    images, labels = [], []
    images_2d = []  # 64x64 形式（Best Model用）
    for label_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(data_dir, class_hex)
        if not os.path.exists(class_dir):
            print(f"  警告: {class_dir} が見つかりません")
            continue
        image_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
        for img_file in image_files:
            img = cv2.imread(os.path.join(class_dir, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))
                images_2d.append(img_resized)
                images.append(img_resized.flatten().astype(np.float64))
                labels.append(label_idx)
    return np.array(images), np.array(images_2d), np.array(labels)


def main():
    print("=" * 80)
    print(" 100クラス比較実験: C++版CS-PNN(生画素) vs Best Model(HOG+PNN)")
    print("=" * 80)

    # データ読み込み
    print("\nデータ読み込み中...")
    flat_images, images_2d, labels = load_etl8b_data("./ETL8B-img-full", TARGET_CLASSES_100)
    n_classes = len(np.unique(labels))
    print(f"  合計: {len(labels)}サンプル, {n_classes}クラス")
    print(f"  特徴次元: {flat_images.shape[1]} (生画素 64x64)")

    # 訓練・テスト分割（同一条件）
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=labels
    )

    train_flat = flat_images[train_idx]
    test_flat = flat_images[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    train_images_2d = images_2d[train_idx]
    test_images_2d = images_2d[test_idx]

    print(f"  訓練: {len(train_labels)}, テスト: {len(test_labels)}")

    results = []

    # =============================================
    # (A) C++ CS-PNN (生画素, [-1,1]正規化)
    # =============================================
    print("\n" + "=" * 80)
    print(" (A) C++ CS-PNN (生画素 4096次元, [-1,1]正規化)")
    print("=" * 80)

    # 正規化
    train_norm, test_norm = normalize_minmax(train_flat, test_flat)

    # クラスインターリーブ配置
    tr_x_split, tr_y_split = make_split_data(train_norm, train_labels)
    train_x_even, train_y_even = make_data_even(tr_x_split, tr_y_split)

    # C++ ライブラリロード
    lib = ct.CDLL('/home/hoyalab/pnn1.so')

    lib.constructCSPNN.argtypes = (_dp, c_double_p, ct.c_int, ct.c_int, ct.c_int, ct.c_short)
    lib.constructCSPNN.restype = PNN
    lib.testPNN.argtypes = (ct.POINTER(PNN), _dp, ct.c_int)
    lib.testPNN.restype = c_int_p
    lib.free1DInt.argtypes = (c_int_p,)
    lib.free1DInt.restype = None
    lib.freePNN.argtypes = (ct.POINTER(PNN),)
    lib.freePNN.restype = None

    # C++ ポインタ変換
    train_x_p, Mx, Nx = npMatrixToC(train_x_even)
    train_y_p, My, Ny = npMatrixToC(train_y_even.astype(np.float64))
    test_x_p, P, Q = npMatrixToC(test_norm)

    # 構築
    print(f"  CS-PNN 構築中... (サンプル={Mx}, 次元={Nx}, クラス={n_classes})")
    nRadiusDenType = 0  # r_den = 現在のクラス数
    start = time.time()
    pnn = lib.constructCSPNN(train_x_p, train_y_p, Mx, Nx, n_classes, nRadiusDenType)
    train_time_a = time.time() - start
    print(f"  構築完了: {train_time_a:.2f}秒, SubNets={pnn.nSubNets}, r_den={pnn.r_den:.1f}")

    # RBF数を表示
    total_units = 0
    for i in range(pnn.nSubNets):
        total_units += pnn.subnet[i].nUnits
    print(f"  総RBF数: {total_units}")

    # テスト
    print(f"  テスト中... ({P}サンプル)")
    start = time.time()
    _y = lib.testPNN(ct.pointer(pnn), test_x_p, P)
    eval_time_a = time.time() - start

    predictions_a = np.array([_y[i] for i in range(P)])
    accuracy_a = 100.0 * np.sum(predictions_a == test_labels) / len(test_labels)
    f1_a = f1_score(test_labels, predictions_a, average='macro')

    print(f"\n  認識率: {accuracy_a:.2f}% ({np.sum(predictions_a == test_labels)}/{len(test_labels)})")
    print(f"  F1-macro: {f1_a:.4f}")
    print(f"  学習時間: {train_time_a:.2f}秒, 評価時間: {eval_time_a:.2f}秒")

    lib.free1DInt(_y)

    results.append({
        'name': 'C++ CS-PNN (生画素)',
        'accuracy': accuracy_a,
        'f1_macro': f1_a,
        'train_time': train_time_a,
        'eval_time': eval_time_a,
        'total_rbfs': total_units,
    })

    # =============================================
    # (B) Best Model 旧版 (HOG + adaptive PNN)
    # =============================================
    print("\n" + "=" * 80)
    print(" (B) Best Model 旧版 (HOG + adaptive PNN)")
    print("=" * 80)

    from kanji_best_standalone import KanjiBestRecognizer

    model_b = KanjiBestRecognizer()

    start = time.time()
    model_b.train(train_images_2d, train_labels)
    train_time_b = time.time() - start

    start = time.time()
    accuracy_b, predictions_b = model_b.evaluate(test_images_2d, test_labels)
    eval_time_b = time.time() - start

    f1_b = f1_score(test_labels, predictions_b, average='macro')

    results.append({
        'name': 'Best Model (HOG+PNN)',
        'accuracy': accuracy_b,
        'f1_macro': f1_b,
        'train_time': train_time_b,
        'eval_time': eval_time_b,
    })

    # =============================================
    # 結果比較
    # =============================================
    print("\n" + "=" * 80)
    print(" 比較結果サマリー")
    print("=" * 80)

    print(f"\n{'モデル':<30} {'認識率':>10} {'F1-macro':>10} {'学習時間':>12} {'評価時間':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<30} {r['accuracy']:>9.2f}% {r['f1_macro']:>9.4f} "
              f"{r['train_time']:>10.2f}秒 {r['eval_time']:>10.2f}秒")

    diff = results[0]['accuracy'] - results[1]['accuracy']
    print(f"\n差分: {diff:+.2f}% (C++ CS-PNN - Best Model)")

    if 'total_rbfs' in results[0]:
        print(f"\nC++ CS-PNN 詳細:")
        print(f"  総RBF数: {results[0]['total_rbfs']}")
        print(f"  入力: 生画素 4096次元 [-1,1]正規化")
        print(f"  σ方式: 動的 max_d/k (k=クラス数)")

    print(f"\nBest Model 詳細:")
    print(f"  入力: HOG特徴 2204次元 + 階層ピラミッド + 左右分割")
    print(f"  σ方式: adaptive mean(NN距離)/√2")
    print(f"  投票: W:R=2:1 加重統合")

    # クラス別精度比較
    print("\n--- クラス別精度（上位/下位5クラス差分） ---")
    per_class_a, per_class_b = [], []
    for cls_idx in range(n_classes):
        mask = test_labels == cls_idx
        if mask.sum() > 0:
            per_class_a.append(100.0 * np.sum(predictions_a[mask] == cls_idx) / mask.sum())
            per_class_b.append(100.0 * np.sum(predictions_b[mask] == cls_idx) / mask.sum())
        else:
            per_class_a.append(0)
            per_class_b.append(0)

    diffs = [(i, per_class_a[i] - per_class_b[i], per_class_a[i], per_class_b[i])
             for i in range(n_classes)]
    diffs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'クラス':>6} {'C++CS-PNN':>12} {'BestModel':>12} {'差分':>10}")
    print("-" * 44)
    for i, d, a, b in diffs[:5]:
        print(f"  {TARGET_CLASSES_100[i]:>6} {a:>10.1f}% {b:>10.1f}% {d:>+9.1f}%")
    print("  ...")
    for i, d, a, b in diffs[-5:]:
        print(f"  {TARGET_CLASSES_100[i]:>6} {a:>10.1f}% {b:>10.1f}% {d:>+9.1f}%")

    wins_a = sum(1 for d in diffs if d[1] > 0)
    wins_b = sum(1 for d in diffs if d[1] < 0)
    ties = sum(1 for d in diffs if d[1] == 0)
    print(f"\nC++ CS-PNN勝利: {wins_a}クラス, Best Model勝利: {wins_b}クラス, 引き分け: {ties}クラス")

    print("\n" + "=" * 80)
    print(" 実験完了")
    print("=" * 80)


if __name__ == '__main__':
    main()
