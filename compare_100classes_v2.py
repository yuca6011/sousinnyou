#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
100クラス包括的比較実験 v2: 改良ベストモデル vs 旧ベストモデル vs MLP

改良点:
  - σ決定法: OrigPNN dmax/√Nc (CS-PNN式RBF)
  - 特徴正規化: 全体・左・右すべて個別にZ-score正規化
  - 投票戦略: 正規化スコア加重統合 (W:R=2:1)

比較項目:
  - 認識率（全体・偏グループ別・クラス別）
  - 学習時間・評価時間
  - F1スコア・適合率・再現率
  - 混同行列
  - モデル構造・パラメータ数
"""

import os
import sys
import time
import numpy as np
import cv2
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

# 偏グループ定義
NINBEN = {
    '304c', '304d', '322f', '323d', '323e', '323f', '3241', '3559',
    '3621', '3738', '376f', '3772', '3844', '3875', '3a6e', '3b45',
    '3b48', '3b77', '3c5a', '3d24', '3d3b', '3f2e', '3f4e', '417c',
    '4226', '422f', '423e', '424e', '4265', '4463', '4464', '4541',
    '462f', '4724', '475c', '4877', '4936', '4955', '4a29', '4a58',
    '4a5d', '4e63',
}
SANZUI = {
    '314b', '3155', '3169', '3239', '324f', '3324', '3368', '3441',
    '3525', '3579', '3768', '3769', '383a', '3850', '3941', '3a2e',
    '3a51', '3c23', '3c72', '3e43', '3f3c', '4036', '4075', '422c',
    '4353', '436d', '4572', '4748', '4749', '4b21', '4b7e', '4c7d',
    '4d4e', '4d61', '4e2e',
}
GONBEN = {
    '325d', '352d', '3544', '3576', '3731', '3757', '386c', '386d',
    '386e', '3956', '3b6c', '3b6d', '3b6e', '4f40',
}


def jis_to_unicode(jis_hex):
    try:
        jis_code = int(jis_hex, 16)
        j1 = (jis_code >> 8) & 0xFF
        j2 = jis_code & 0xFF
        if j1 % 2 == 0:
            s1 = ((j1 - 0x21) >> 1) + 0x81
            if s1 > 0x9f:
                s1 += 0x40
            s2 = j2 + 0x7e
        else:
            s1 = ((j1 - 0x21) >> 1) + 0x81
            if s1 > 0x9f:
                s1 += 0x40
            if j2 < 0x60:
                s2 = j2 + 0x1f
            else:
                s2 = j2 + 0x20
        return bytes([s1, s2]).decode('shift_jis')
    except Exception:
        return '?'


def get_radical_group(jis_hex):
    if jis_hex in NINBEN:
        return 'にんべん(亻)'
    elif jis_hex in SANZUI:
        return 'さんずい(氵)'
    elif jis_hex in GONBEN:
        return 'ごんべん(言)'
    else:
        return 'その他'


# ============================================================================
# データ読み込み・特徴抽出
# ============================================================================

def load_etl8b_data(data_dir, target_classes):
    images, labels, class_info = [], [], {}
    for idx, hex_code in enumerate(target_classes):
        d = os.path.join(data_dir, hex_code)
        if not os.path.isdir(d):
            print(f"  警告: {d} が見つかりません")
            continue
        class_images = []
        for f in sorted(os.listdir(d)):
            if f.endswith('.png'):
                img = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    class_images.append(cv2.resize(img, (64, 64)))
        images.extend(class_images)
        labels.extend([idx] * len(class_images))
        class_info[idx] = {
            'hex': hex_code,
            'char': jis_to_unicode(hex_code),
            'count': len(class_images),
            'radical': get_radical_group(hex_code),
        }
    return np.array(images), np.array(labels), class_info


def extract_all_features(images):
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


# ============================================================================
# 改良ベストモデル
# ============================================================================

class ImprovedBestModel:
    """
    改良ベストモデル:
      - σ = dmax/√Nc (OrigPNN方式)
      - RBF = exp(-d²/σ²) (CS-PNN式)
      - K-means代表点 (20%)
      - 正規化スコア加重統合 W:R=2:1
      - 全特徴Z-score正規化
    """

    def __init__(self):
        self.pnns = {}
        self.scalers = {}
        self.classes_ = None
        self.n_centroids = {'whole': 0, 'left': 0, 'right': 0}
        self.sigma_values = {}

    def fit(self, train_images, train_labels):
        """特徴抽出・正規化・PNN学習"""
        # 特徴抽出
        self.tr_w, self.tr_l, self.tr_r = extract_all_features(train_images)

        # 正規化
        for part, data in [('whole', self.tr_w), ('left', self.tr_l), ('right', self.tr_r)]:
            sc = StandardScaler()
            if part == 'whole':
                self.tr_w = sc.fit_transform(data)
            elif part == 'left':
                self.tr_l = sc.fit_transform(data)
            else:
                self.tr_r = sc.fit_transform(data)
            self.scalers[part] = sc

        self.classes_ = np.unique(train_labels)
        nc = len(self.classes_)

        # 各パートのPNN学習
        for part, tr_f in [('whole', self.tr_w), ('left', self.tr_l), ('right', self.tr_r)]:
            exemplars = {}
            for cls in self.classes_:
                X_cls = tr_f[train_labels == cls]
                n_ex = max(1, int(len(X_cls) * 0.2))
                km = KMeans(n_clusters=n_ex, random_state=42, n_init=10)
                km.fit(X_cls)
                exemplars[cls] = km.cluster_centers_

            # キャッシュ
            arrays, ranges = [], {}
            idx = 0
            for cls in sorted(exemplars.keys()):
                n = len(exemplars[cls])
                ranges[cls] = (idx, idx + n)
                arrays.append(exemplars[cls])
                idx += n
            all_centroids = np.vstack(arrays)

            # σ = dmax/√Nc (サブサンプル)
            n = len(tr_f)
            if n > 3000:
                sub_idx = np.random.RandomState(42).choice(n, 3000, replace=False)
                X_sub = tr_f[sub_idx]
            else:
                X_sub = tr_f
            d_max = np.max(pdist(X_sub, metric='euclidean'))
            sigma = d_max / np.sqrt(nc)

            self.pnns[part] = {
                'all_centroids': all_centroids,
                'class_ranges': ranges,
                'sigma': sigma,
            }
            self.n_centroids[part] = len(all_centroids)
            self.sigma_values[part] = sigma

    def predict(self, test_images):
        """特徴抽出・正規化・スコア統合予測"""
        ts_w, ts_l, ts_r = extract_all_features(test_images)
        ts_w = self.scalers['whole'].transform(ts_w)
        ts_l = self.scalers['left'].transform(ts_l)
        ts_r = self.scalers['right'].transform(ts_r)

        # 各パートのスコア計算
        scores = {}
        for part, ts_f in [('whole', ts_w), ('left', ts_l), ('right', ts_r)]:
            pnn = self.pnns[part]
            dists_sq = cdist(ts_f, pnn['all_centroids'], metric='sqeuclidean')
            s = pnn['sigma']
            acts = np.exp(-dists_sq / (s * s))  # CS-PNN式

            n_samples = len(ts_f)
            n_cls = len(self.classes_)
            sc = np.zeros((n_samples, n_cls))
            for c_idx, cls in enumerate(sorted(pnn['class_ranges'].keys())):
                start, end = pnn['class_ranges'][cls]
                sc[:, c_idx] = np.mean(acts[:, start:end], axis=1)
            scores[part] = sc

        # 正規化スコア加重統合 W:R=2:1
        sw_norm = scores['whole'] / (scores['whole'].sum(axis=1, keepdims=True) + 1e-30)
        sr_norm = scores['right'] / (scores['right'].sum(axis=1, keepdims=True) + 1e-30)
        combined = 2.0 * sw_norm + 1.0 * sr_norm

        return self.classes_[np.argmax(combined, axis=1)]

    def get_model_info(self):
        total_centroids = sum(self.n_centroids.values())
        dim = self.pnns['whole']['all_centroids'].shape[1] if 'whole' in self.pnns else 0
        return {
            'name': 'Improved Best Model',
            'sigma_method': 'OrigPNN: dmax/sqrt(Nc)',
            'rbf_formula': 'exp(-d^2/sigma^2) [CS-PNN]',
            'voting': 'Normalized Score W:R=2:1',
            'normalization': 'Z-score (all parts)',
            'exemplar_ratio': 0.2,
            'centroids_whole': self.n_centroids['whole'],
            'centroids_left': self.n_centroids['left'],
            'centroids_right': self.n_centroids['right'],
            'centroids_total': total_centroids,
            'feature_dim': dim,
            'sigma_whole': self.sigma_values.get('whole', 0),
            'sigma_left': self.sigma_values.get('left', 0),
            'sigma_right': self.sigma_values.get('right', 0),
            'n_params_equiv': total_centroids * dim,
        }


# ============================================================================
# 旧ベストモデル（kanji_best_standaloneを直接使用）
# ============================================================================

class OldBestModel:
    """旧ベストモデル（3者多数決、adaptiveσ、全体のみ正規化）"""

    def __init__(self):
        from kanji_best_standalone import KanjiBestRecognizer
        self.recognizer = KanjiBestRecognizer()
        self.recognizer.augmentation_factor = 1

    def fit(self, X, y):
        self.recognizer.train(X, y)

    def predict(self, X):
        predictions = []
        for i, img in enumerate(X):
            pred = self.recognizer.predict(img)
            predictions.append(pred)
            if (i + 1) % 1000 == 0:
                print(f"    予測: {i+1}/{len(X)}")
        return np.array(predictions)


# ============================================================================
# MLP
# ============================================================================

class MLP(nn.Module):
    def __init__(self, input_size=4096, hidden_sizes=None, num_classes=100):
        super(MLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [1024, 512, 256]
        layers = []
        in_size = input_size
        for hs in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, hs), nn.BatchNorm1d(hs),
                nn.ReLU(), nn.Dropout(0.3),
            ])
            in_size = hs
        layers.append(nn.Linear(in_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MLPWrapper:
    def __init__(self, num_classes=100):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_losses = []
        self.train_accs = []

    def fit(self, X, y, epochs=100, batch_size=64, lr=0.001):
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        X_t = torch.FloatTensor(X_flat)
        y_t = torch.LongTensor(y)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = MLP(input_size=X_flat.shape[1],
                         num_classes=self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss, correct, total = 0.0, 0, 0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                _, pred = torch.max(out, 1)
                total += by.size(0)
                correct += (pred == by).sum().item()
            scheduler.step()
            self.train_losses.append(epoch_loss / len(loader))
            self.train_accs.append(100.0 * correct / total)
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {self.train_losses[-1]:.4f}, "
                      f"Acc: {self.train_accs[-1]:.2f}%")

    def predict(self, X):
        self.model.eval()
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        X_t = torch.FloatTensor(X_flat).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)
            _, pred = torch.max(out, 1)
        return pred.cpu().numpy()

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def get_model_info(self):
        return {
            'name': 'MLP (DNN)',
            'architecture': '4096-1024-512-256-100',
            'activation': 'ReLU',
            'regularization': 'Dropout(0.3) + BatchNorm + L2(1e-4)',
            'optimizer': 'Adam(lr=0.001)',
            'scheduler': 'StepLR(step=30, gamma=0.5)',
            'epochs': 100,
            'batch_size': 64,
            'n_params': self.count_params(),
        }


# ============================================================================
# 可視化
# ============================================================================

def plot_all(results, class_info, output_dir, mlp_wrapper):
    os.makedirs(output_dir, exist_ok=True)

    methods = [r['method'] for r in results]
    accs = [r['accuracy'] for r in results]
    colors_map = {'Improved Best Model': '#2ecc71',
                  'Old Best Model': '#f39c12',
                  'MLP (DNN)': '#e74c3c'}
    colors = [colors_map.get(m, '#95a5a6') for m in methods]

    # 1. 認識率比較
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(methods)), accs, color=colors, alpha=0.85, width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., acc + 0.3,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Recognition Accuracy Comparison (100 classes)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(80, 100)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 処理時間比較
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    time_labels = [('train_time', 'Training Time (s)'),
                   ('eval_time', 'Evaluation Time (s)'),
                   ('total_time', 'Total Time (s)')]
    for ax, (key, title) in zip(axes, time_labels):
        if key == 'total_time':
            vals = [r['train_time'] + r['eval_time'] for r in results]
        else:
            vals = [r[key] for r in results]
        bars = ax.bar(range(len(methods)), vals, color=colors, alpha=0.85, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., v + max(vals)*0.02,
                    f'{v:.1f}s', ha='center', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m[:15] for m in methods], fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 混同行列（全モデル）
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array(r['confusion_matrix'])
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f"{r['method']}\n({r['accuracy']:.2f}%)", fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 偏グループ別精度（全モデル比較）
    radical_names = ['にんべん(亻)', 'さんずい(氵)', 'ごんべん(言)', 'その他']
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(radical_names))
    w = 0.25

    for m_idx, r in enumerate(results):
        cm = np.array(r['confusion_matrix'])
        group_accs = []
        for rn in radical_names:
            cls_accs = []
            for label_idx, info in class_info.items():
                if info['radical'] == rn and label_idx < cm.shape[0]:
                    total = cm[label_idx].sum()
                    if total > 0:
                        cls_accs.append(100.0 * cm[label_idx, label_idx] / total)
            group_accs.append(np.mean(cls_accs) if cls_accs else 0)

        offset = (m_idx - (len(results) - 1) / 2) * w
        bars = ax.bar(x + offset, group_accs, w, label=r['method'],
                      color=colors[m_idx], alpha=0.85)
        for bar, v in zip(bars, group_accs):
            ax.text(bar.get_x() + bar.get_width()/2., v + 0.5,
                    f'{v:.1f}', ha='center', fontsize=8, fontweight='bold')

    ax.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy by Radical Group (100 classes)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(radical_names, fontsize=11)
    ax.set_ylim(70, 102)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radical_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. MLP学習曲線
    if mlp_wrapper and mlp_wrapper.train_losses:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(mlp_wrapper.train_losses) + 1)
        ax1.plot(epochs, mlp_wrapper.train_losses, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('MLP Training Loss', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax2.plot(epochs, mlp_wrapper.train_accs, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('MLP Training Accuracy', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mlp_training_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 6. クラス別精度の分布
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array(r['confusion_matrix'])
        per_class = []
        for i in range(cm.shape[0]):
            total = cm[i].sum()
            if total > 0:
                per_class.append(100.0 * cm[i, i] / total)
        ax.hist(per_class, bins=20, color=colors_map.get(r['method'], '#95a5a6'),
                alpha=0.8, edgecolor='black')
        ax.axvline(np.mean(per_class), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={np.mean(per_class):.1f}%')
        ax.set_xlabel('Per-class Accuracy (%)')
        ax.set_ylabel('Number of classes')
        ax.set_title(f"{r['method']}\nmin={min(per_class):.1f}% max={max(per_class):.1f}%",
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_accuracy_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 7. 改善マップ（改良Best vs MLP のクラス別差分）
    improved = [r for r in results if r['method'] == 'Improved Best Model']
    mlp = [r for r in results if r['method'] == 'MLP (DNN)']
    if improved and mlp:
        cm_imp = np.array(improved[0]['confusion_matrix'])
        cm_mlp = np.array(mlp[0]['confusion_matrix'])
        diffs = []
        labels_list = []
        for i in range(min(cm_imp.shape[0], cm_mlp.shape[0])):
            t_imp = cm_imp[i].sum()
            t_mlp = cm_mlp[i].sum()
            if t_imp > 0 and t_mlp > 0:
                a_imp = 100.0 * cm_imp[i, i] / t_imp
                a_mlp = 100.0 * cm_mlp[i, i] / t_mlp
                diffs.append(a_imp - a_mlp)
                char = class_info[i]['char'] if i in class_info else '?'
                labels_list.append(f"{char}")

        fig, ax = plt.subplots(figsize=(16, 6))
        colors_diff = ['#2ecc71' if d >= 0 else '#e74c3c' for d in diffs]
        ax.bar(range(len(diffs)), diffs, color=colors_diff, alpha=0.8, width=0.7)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy Difference (%)')
        ax.set_title('Improved Best Model vs MLP: Per-class Accuracy Difference\n'
                     '(Green=Best wins, Red=MLP wins)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        n_win = sum(1 for d in diffs if d > 0)
        n_lose = sum(1 for d in diffs if d < 0)
        n_tie = sum(1 for d in diffs if d == 0)
        ax.text(0.02, 0.95, f'Best wins: {n_win} classes\nMLP wins: {n_lose} classes\nTie: {n_tie}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvement_map.png'), dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# レポート生成
# ============================================================================

def generate_report(results, class_info, output_dir, improved_info, mlp_info):
    total_samples = sum(info['count'] for info in class_info.values())
    n_train = int(total_samples * 0.7)
    n_test = total_samples - n_train
    nc = len(class_info)

    report_lines = []
    def w(s=''):
        report_lines.append(s)

    w("=" * 100)
    w(" 100クラス包括的比較実験レポート v2")
    w("=" * 100)
    w()

    # 1. 実験設定
    w("1. 実験設定")
    w("-" * 100)
    w(f"  対象クラス数: {nc}")
    w(f"  画像サイズ: 64x64")
    w(f"  訓練データ: {n_train}サンプル (70%)")
    w(f"  テストデータ: {n_test}サンプル (30%)")
    w(f"  合計: {total_samples}サンプル")
    w(f"  データ分割: stratified, random_state=42")
    w()
    w("  偏グループ構成:")
    w(f"    - にんべん(亻): {len(NINBEN)}クラス")
    w(f"    - さんずい(氵): {len(SANZUI)}クラス")
    w(f"    - ごんべん(言): {len(GONBEN)}クラス")
    w(f"    - その他: {nc - len(NINBEN) - len(SANZUI) - len(GONBEN)}クラス")
    w()

    # 2. モデル詳細
    w("2. モデル詳細")
    w("-" * 100)
    w("  (A) 改良ベストモデル (Improved Best Model):")
    w(f"    - HOG特徴抽出 (9方向, 8x8セル)")
    w(f"    - 階層ピラミッド: 5段, 特徴次元: {improved_info['feature_dim']}")
    w(f"    - 左右分割 (argmax projection)")
    w(f"    - σ決定: {improved_info['sigma_method']}")
    w(f"    - RBF: {improved_info['rbf_formula']}")
    w(f"    - 投票戦略: {improved_info['voting']}")
    w(f"    - 特徴正規化: {improved_info['normalization']}")
    w(f"    - 代表点: 全体{improved_info['centroids_whole']} + "
      f"左{improved_info['centroids_left']} + 右{improved_info['centroids_right']} "
      f"= {improved_info['centroids_total']}")
    w(f"    - σ値: 全体={improved_info['sigma_whole']:.4f}, "
      f"左={improved_info['sigma_left']:.4f}, 右={improved_info['sigma_right']:.4f}")
    w(f"    - 等価パラメータ数: {improved_info['n_params_equiv']:,}")
    w()
    w("  (B) 旧ベストモデル (Old Best Model):")
    w("    - HOG+PNN+左右分割+3者多数決投票")
    w("    - σ: adaptive (mean(NN距離)/√2)")
    w("    - RBF: exp(-d²/(2σ²))")
    w("    - 全体のみ正規化")
    w()
    w("  (C) MLP (DNN):")
    w(f"    - 構造: {mlp_info['architecture']}")
    w(f"    - 活性化: {mlp_info['activation']}")
    w(f"    - 正則化: {mlp_info['regularization']}")
    w(f"    - 最適化: {mlp_info['optimizer']}")
    w(f"    - パラメータ数: {mlp_info['n_params']:,}")
    w()

    # 3. 認識率比較
    w("3. 認識率比較")
    w("-" * 100)
    best_acc = max(r['accuracy'] for r in results)
    w(f"  {'モデル':<40} {'認識率':>10}   {'F1-macro':>10}   {'Precision':>10}   {'Recall':>10}")
    w(f"  {'-'*95}")
    for r in results:
        marker = " ★" if r['accuracy'] == best_acc else ""
        w(f"  {r['method']:<40} {r['accuracy']:>9.2f}%   "
          f"{r['f1_macro']:>9.4f}   {r['precision_macro']:>9.4f}   "
          f"{r['recall_macro']:>9.4f}{marker}")

    imp_r = next(r for r in results if r['method'] == 'Improved Best Model')
    old_r = next(r for r in results if r['method'] == 'Old Best Model')
    mlp_r = next(r for r in results if r['method'] == 'MLP (DNN)')

    w()
    w(f"  改良Best - MLP:  {imp_r['accuracy'] - mlp_r['accuracy']:+.2f}%")
    w(f"  改良Best - 旧Best: {imp_r['accuracy'] - old_r['accuracy']:+.2f}%")
    w(f"  旧Best - MLP:   {old_r['accuracy'] - mlp_r['accuracy']:+.2f}%")
    w()

    # 4. 処理時間
    w("4. 処理時間比較")
    w("-" * 100)
    w(f"  {'モデル':<40} {'学習時間':>12} {'評価時間':>12} {'合計':>12}")
    w(f"  {'-'*80}")
    for r in results:
        total = r['train_time'] + r['eval_time']
        w(f"  {r['method']:<40} {r['train_time']:>10.2f}秒 {r['eval_time']:>10.2f}秒 {total:>10.2f}秒")
    w()

    # 5. 偏グループ別精度
    w("5. 偏グループ別精度")
    w("-" * 100)
    radical_names = ['にんべん(亻)', 'さんずい(氵)', 'ごんべん(言)', 'その他']
    header = f"  {'偏グループ':<20}"
    for r in results:
        header += f" {r['method'][:18]:>18}"
    w(header)
    w(f"  {'-'*80}")
    for rn in radical_names:
        line = f"  {rn:<20}"
        for r in results:
            cm = np.array(r['confusion_matrix'])
            cls_accs = []
            for label_idx, info in class_info.items():
                if info['radical'] == rn and label_idx < cm.shape[0]:
                    total = cm[label_idx].sum()
                    if total > 0:
                        cls_accs.append(100.0 * cm[label_idx, label_idx] / total)
            mean_acc = np.mean(cls_accs) if cls_accs else 0
            line += f" {mean_acc:>17.2f}%"
        w(line)
    w()

    # 6. クラス別精度統計
    w("6. クラス別精度統計")
    w("-" * 100)
    for r in results:
        cm = np.array(r['confusion_matrix'])
        per_class = []
        for i in range(cm.shape[0]):
            total = cm[i].sum()
            if total > 0:
                per_class.append(100.0 * cm[i, i] / total)
        w(f"  {r['method']}:")
        w(f"    平均: {np.mean(per_class):.2f}%, 標準偏差: {np.std(per_class):.2f}%")
        w(f"    最小: {np.min(per_class):.2f}%, 最大: {np.max(per_class):.2f}%")
        w(f"    中央値: {np.median(per_class):.2f}%")
        w(f"    90%以上: {sum(1 for a in per_class if a >= 90)}/{len(per_class)}クラス")
        w(f"    80%以上: {sum(1 for a in per_class if a >= 80)}/{len(per_class)}クラス")
        w()

    # 7. 改良Best vs MLP 詳細
    w("7. 改良Best Model vs MLP: クラス別勝敗")
    w("-" * 100)
    cm_imp = np.array(imp_r['confusion_matrix'])
    cm_mlp = np.array(mlp_r['confusion_matrix'])
    n_win, n_lose, n_tie = 0, 0, 0
    worst_classes = []
    for i in range(min(cm_imp.shape[0], cm_mlp.shape[0])):
        t_imp = cm_imp[i].sum()
        t_mlp = cm_mlp[i].sum()
        if t_imp > 0 and t_mlp > 0:
            a_imp = 100.0 * cm_imp[i, i] / t_imp
            a_mlp = 100.0 * cm_mlp[i, i] / t_mlp
            diff = a_imp - a_mlp
            if diff > 0:
                n_win += 1
            elif diff < 0:
                n_lose += 1
                worst_classes.append((i, class_info[i]['char'], diff))
            else:
                n_tie += 1
    w(f"  改良Best勝ち: {n_win}クラス")
    w(f"  MLP勝ち: {n_lose}クラス")
    w(f"  引き分け: {n_tie}クラス")
    if worst_classes:
        worst_classes.sort(key=lambda x: x[2])
        w(f"\n  MLPに大きく負けているクラス（上位5）:")
        for idx, char, diff in worst_classes[:5]:
            w(f"    {char} (idx={idx}): {diff:+.1f}%")
    w()

    # 8. 15クラス実験との比較
    w("8. 15クラス実験との比較")
    w("-" * 100)
    w("  15クラス実験（参考）:")
    w("    - Best Model: 97.38% (2415サンプル)")
    w("    - MLP:        92.28% (2415サンプル)")
    w("    - 差分:       +5.10%")
    w()
    w(f"  100クラス実験（旧Best）:")
    w(f"    - Best Model: {old_r['accuracy']:.2f}% ({total_samples}サンプル)")
    w(f"    - MLP:        {mlp_r['accuracy']:.2f}% ({total_samples}サンプル)")
    w(f"    - 差分:       {old_r['accuracy'] - mlp_r['accuracy']:+.2f}%")
    w()
    w(f"  100クラス実験（改良Best）:")
    w(f"    - Best Model: {imp_r['accuracy']:.2f}% ({total_samples}サンプル)")
    w(f"    - MLP:        {mlp_r['accuracy']:.2f}% ({total_samples}サンプル)")
    w(f"    - 差分:       {imp_r['accuracy'] - mlp_r['accuracy']:+.2f}%")
    w()

    # 9. 改良のサマリ
    w("9. 改良サマリ")
    w("-" * 100)
    w("  旧Best → 改良Best の変更点:")
    w(f"    (1) σ: adaptive(mean(NN)/√2) → OrigPNN(dmax/√Nc)")
    w(f"    (2) RBF: exp(-d²/(2σ²)) → exp(-d²/σ²) [CS-PNN式]")
    w(f"    (3) 正規化: 全体のみ → 全体・左・右すべてZ-score")
    w(f"    (4) 投票: 3者多数決 → 正規化スコア加重(W:R=2:1)")
    w(f"    精度向上: {old_r['accuracy']:.2f}% → {imp_r['accuracy']:.2f}% ({imp_r['accuracy'] - old_r['accuracy']:+.2f}%)")
    w()

    # 10. 生成ファイル
    w("10. 生成ファイル")
    w("-" * 100)
    w("  - accuracy_comparison.png: 認識率比較グラフ")
    w("  - time_comparison.png: 処理時間比較グラフ")
    w("  - confusion_matrices.png: 混同行列比較")
    w("  - radical_accuracy.png: 偏グループ別精度比較")
    w("  - mlp_training_curve.png: MLP学習曲線")
    w("  - class_accuracy_distribution.png: クラス別精度分布")
    w("  - improvement_map.png: クラス別改善マップ")
    w("  - comparison_v2_results.json: 詳細結果(JSON)")
    w("  - comparison_v2_report.txt: このレポート")

    report_text = '\n'.join(report_lines) + '\n'
    report_path = os.path.join(output_dir, 'comparison_v2_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  レポートを {report_path} に保存しました")
    return report_text


# ============================================================================
# メイン
# ============================================================================

def main():
    print("=" * 100)
    print(" 100クラス包括的比較実験 v2")
    print(" 改良ベストモデル vs 旧ベストモデル vs MLP")
    print("=" * 100)

    nc = len(TARGET_CLASSES_100)
    print(f"\n対象: {nc}クラス, 3モデル比較")

    # データ読み込み
    print("\n[1/5] データ読み込み...")
    images, labels, class_info = load_etl8b_data("./ETL8B-img-full", TARGET_CLASSES_100)
    print(f"  {len(images)}サンプル, {nc}クラス")

    tr_img, ts_img, tr_lbl, ts_lbl = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"  訓練: {len(tr_img)}, テスト: {len(ts_img)}")

    results = []

    # ============================================================
    # (A) 改良ベストモデル
    # ============================================================
    print("\n" + "=" * 100)
    print("[2/5] (A) 改良ベストモデル")
    print("=" * 100)

    model_imp = ImprovedBestModel()

    t0 = time.time()
    model_imp.fit(tr_img, tr_lbl)
    train_time_imp = time.time() - t0
    print(f"  学習時間: {train_time_imp:.2f}秒")

    t0 = time.time()
    pred_imp = model_imp.predict(ts_img)
    eval_time_imp = time.time() - t0
    print(f"  評価時間: {eval_time_imp:.2f}秒")

    acc_imp = accuracy_score(ts_lbl, pred_imp) * 100
    cm_imp = confusion_matrix(ts_lbl, pred_imp)
    f1_imp = f1_score(ts_lbl, pred_imp, average='macro')
    prec_imp = precision_score(ts_lbl, pred_imp, average='macro', zero_division=0)
    rec_imp = recall_score(ts_lbl, pred_imp, average='macro', zero_division=0)
    print(f"  認識率: {acc_imp:.2f}%")
    print(f"  F1-macro: {f1_imp:.4f}, Precision: {prec_imp:.4f}, Recall: {rec_imp:.4f}")

    improved_info = model_imp.get_model_info()

    results.append({
        'method': 'Improved Best Model',
        'accuracy': acc_imp,
        'train_time': train_time_imp,
        'eval_time': eval_time_imp,
        'f1_macro': f1_imp,
        'precision_macro': prec_imp,
        'recall_macro': rec_imp,
        'predictions': pred_imp.tolist(),
        'confusion_matrix': cm_imp.tolist(),
    })

    # ============================================================
    # (B) 旧ベストモデル
    # ============================================================
    print("\n" + "=" * 100)
    print("[3/5] (B) 旧ベストモデル")
    print("=" * 100)

    model_old = OldBestModel()

    t0 = time.time()
    model_old.fit(tr_img, tr_lbl)
    train_time_old = time.time() - t0
    print(f"  学習時間: {train_time_old:.2f}秒")

    t0 = time.time()
    pred_old = model_old.predict(ts_img)
    eval_time_old = time.time() - t0
    print(f"  評価時間: {eval_time_old:.2f}秒")

    acc_old = accuracy_score(ts_lbl, pred_old) * 100
    cm_old = confusion_matrix(ts_lbl, pred_old)
    f1_old = f1_score(ts_lbl, pred_old, average='macro')
    prec_old = precision_score(ts_lbl, pred_old, average='macro', zero_division=0)
    rec_old = recall_score(ts_lbl, pred_old, average='macro', zero_division=0)
    print(f"  認識率: {acc_old:.2f}%")
    print(f"  F1-macro: {f1_old:.4f}, Precision: {prec_old:.4f}, Recall: {rec_old:.4f}")

    results.append({
        'method': 'Old Best Model',
        'accuracy': acc_old,
        'train_time': train_time_old,
        'eval_time': eval_time_old,
        'f1_macro': f1_old,
        'precision_macro': prec_old,
        'recall_macro': rec_old,
        'predictions': pred_old.tolist(),
        'confusion_matrix': cm_old.tolist(),
    })

    # ============================================================
    # (C) MLP
    # ============================================================
    print("\n" + "=" * 100)
    print("[4/5] (C) MLP (DNN)")
    print("=" * 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  デバイス: {device}")

    mlp_wrapper = MLPWrapper(num_classes=nc)

    t0 = time.time()
    mlp_wrapper.fit(tr_img, tr_lbl, epochs=100, batch_size=64, lr=0.001)
    train_time_mlp = time.time() - t0
    print(f"  学習時間: {train_time_mlp:.2f}秒")

    t0 = time.time()
    pred_mlp = mlp_wrapper.predict(ts_img)
    eval_time_mlp = time.time() - t0
    print(f"  評価時間: {eval_time_mlp:.2f}秒")

    acc_mlp = accuracy_score(ts_lbl, pred_mlp) * 100
    cm_mlp = confusion_matrix(ts_lbl, pred_mlp)
    f1_mlp = f1_score(ts_lbl, pred_mlp, average='macro')
    prec_mlp = precision_score(ts_lbl, pred_mlp, average='macro', zero_division=0)
    rec_mlp = recall_score(ts_lbl, pred_mlp, average='macro', zero_division=0)
    print(f"  認識率: {acc_mlp:.2f}%")
    print(f"  F1-macro: {f1_mlp:.4f}, Precision: {prec_mlp:.4f}, Recall: {rec_mlp:.4f}")

    mlp_info = mlp_wrapper.get_model_info()

    results.append({
        'method': 'MLP (DNN)',
        'accuracy': acc_mlp,
        'train_time': train_time_mlp,
        'eval_time': eval_time_mlp,
        'f1_macro': f1_mlp,
        'precision_macro': prec_mlp,
        'recall_macro': rec_mlp,
        'predictions': pred_mlp.tolist(),
        'confusion_matrix': cm_mlp.tolist(),
    })

    # ============================================================
    # 結果まとめ
    # ============================================================
    print("\n" + "=" * 100)
    print("[5/5] 結果まとめ・可視化・レポート生成")
    print("=" * 100)

    print(f"\n{'モデル':<25} {'認識率':>8} {'F1':>8} {'学習':>8} {'評価':>8} {'合計':>8}")
    print("-" * 80)
    for r in results:
        total = r['train_time'] + r['eval_time']
        print(f"{r['method']:<25} {r['accuracy']:>7.2f}% {r['f1_macro']:>7.4f} "
              f"{r['train_time']:>7.1f}s {r['eval_time']:>7.1f}s {total:>7.1f}s")

    output_dir = 'results/comparison_100_v2'
    os.makedirs(output_dir, exist_ok=True)

    # JSON保存
    save_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k not in ['predictions', 'confusion_matrix']}
        save_results.append(sr)
    # 大きなデータも含む完全版
    with open(os.path.join(output_dir, 'comparison_v2_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # サマリのみ
    with open(os.path.join(output_dir, 'comparison_v2_summary.json'), 'w') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    # 可視化
    plot_all(results, class_info, output_dir, mlp_wrapper)

    # レポート
    report = generate_report(results, class_info, output_dir, improved_info, mlp_info)

    # コンソールにもレポート出力
    print("\n" + report)

    print(f"\n結果は {output_dir}/ に保存されました")
    print("=" * 100)


if __name__ == '__main__':
    main()
