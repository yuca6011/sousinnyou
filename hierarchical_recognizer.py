"""
hierarchical_recognizer.py
階層構造的画像パターン認識器 - Pythonラッパー
論文の3.1節～3.6節の実装（計算はC++で実行）

認識率（15クラス、各30サンプル、32x32画像、135テスト）:
  階層型パターン認識器: 73.33%（学習0.009秒、RBF数130）

使用方法:
    python hierarchical_recognizer.py
"""

import ctypes as ct
import numpy as np
import cv2
import time
from typing import List, Tuple, Optional, Dict
from enum import IntEnum

#==============================================================
# 設定用のEnum（論文3.4節）
#==============================================================

class ClassNumMethod(IntEnum):
    """クラス数設定法"""
    INCREMENTAL = 0  # 法1: 動的増加
    FIXED = 1        # 法2: 固定

class RadiusMethod(IntEnum):
    """半径値設定法"""
    METHOD1 = 0  # σ = d_max,M / q
    METHOD2 = 1  # σ = d'_max,M / N_c
    METHOD3 = 2  # 任意の値

class WeightUpdateMethod(IntEnum):
    """結合係数更新法"""
    NO_UPDATE = 0  # 更新なし
    AVERAGE = 1    # 平均値で更新

#==============================================================
# ctypes型定義
#==============================================================

c_int_p = ct.POINTER(ct.c_int)
c_double_p = ct.POINTER(ct.c_double)
_dp = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')

# C++構造体（必要に応じて定義）
class CHierarchicalRecognizer(ct.Structure):
    pass

#==============================================================
# ユーティリティ関数
#==============================================================

def numpy_to_c_2d(arr: np.ndarray):
    """2次元numpy配列をC++用に変換"""
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    M, N = arr.shape
    ptr = (arr.__array_interface__['data'][0] + 
           np.arange(M) * arr.strides[0]).astype(np.uintp)
    return ptr, M, N

def numpy_to_c_1d(arr: np.ndarray):
    """1次元numpy配列をC++用に変換"""
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    return arr.ctypes.data_as(c_double_p), len(arr)

def int_array_to_c(arr: np.ndarray):
    """int配列をC++用に変換"""
    arr_c = arr.astype(np.int32)
    return arr_c.ctypes.data_as(c_int_p)

def tic():
    """タイマー開始"""
    global start_time_tictoc
    start_time_tictoc = time.time()

def toc(tag="elapsed time"):
    """タイマー終了"""
    if "start_time_tictoc" in globals():
        print("{}: {:.4f} [sec]".format(tag, time.time() - start_time_tictoc))
    else:
        print("tic has not been called")

#==============================================================
# 画像前処理（論文3.1節 Step 1）
#==============================================================

class ImagePreprocessor:
    """
    画像の前処理
    - グレースケール化
    - 2値化（大津の方法）
    - 余白トリミング（射影法）
    - 正規化
    """
    
    @staticmethod
    def preprocess(img: np.ndarray, target_size: int = 32) -> np.ndarray:
        """画像の前処理を実行"""
        # グレースケール化
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # float64の場合はuint8に変換
        if gray.dtype == np.float64 or gray.dtype == np.float32:
            gray = (gray * 255).astype(np.uint8)

        # 2値化（大津の方法）
        _, binary = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 射影法による余白トリミング
        x_proj = np.sum(binary, axis=0)
        y_proj = np.sum(binary, axis=1)
        
        x_nonzero = np.where(x_proj > 0)[0]
        y_nonzero = np.where(y_proj > 0)[0]
        
        if len(x_nonzero) > 0 and len(y_nonzero) > 0:
            x_min, x_max = x_nonzero[0], x_nonzero[-1]
            y_min, y_max = y_nonzero[0], y_nonzero[-1]
            trimmed = binary[y_min:y_max+1, x_min:x_max+1]
        else:
            trimmed = binary
        
        # 正方形化
        h, w = trimmed.shape
        size = max(h, w)
        canvas = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = trimmed
        
        # リサイズと正規化
        resized = cv2.resize(canvas, (target_size, target_size))
        normalized = resized.astype(np.float64) / 255.0
        
        return normalized

#==============================================================
# 階層構造的パターン認識器（メインクラス）
#==============================================================

class HierarchicalPatternRecognizer:
    """
    階層構造的画像パターン認識器
    計算処理はC++で実行
    """
    
    def __init__(self,
                 lib_path: str = './hierarchical_ext.so',
                 num_pyramid_levels: int = 5,
                 target_image_size: int = 32,
                 class_num_method: ClassNumMethod = ClassNumMethod.INCREMENTAL,
                 radius_method: RadiusMethod = RadiusMethod.METHOD1,
                 weight_update_method: WeightUpdateMethod = WeightUpdateMethod.NO_UPDATE,
                 custom_radius: float = 1.0):
        """
        Args:
            lib_path: C++ライブラリのパス
            num_pyramid_levels: ピラミッド階層数
            target_image_size: 正規化画像サイズ
            class_num_method: クラス数設定法
            radius_method: 半径値設定法
            weight_update_method: 結合係数更新法
            custom_radius: カスタム半径値
        """
        # C++ライブラリのロード
        self.lib = ct.CDLL(lib_path)
        self._setup_lib_functions()
        
        # パラメータ
        self.num_pyramid_levels = num_pyramid_levels
        self.target_image_size = target_image_size
        self.class_num_method = int(class_num_method)
        self.radius_method = int(radius_method)
        self.weight_update_method = int(weight_update_method)
        self.custom_radius = custom_radius
        
        # 前処理器
        self.preprocessor = ImagePreprocessor()
        
        # C++側の認識器オブジェクト
        self.c_recognizer = None
        self.total_classes = 0
    
    def _setup_lib_functions(self):
        """C++ライブラリの関数定義"""
        # 認識器の作成
        self.lib.createHierarchicalRecognizer.argtypes = [ct.c_int]
        self.lib.createHierarchicalRecognizer.restype = ct.POINTER(CHierarchicalRecognizer)
        
        # 学習
        self.lib.trainHierarchicalRecognizer.argtypes = [
            ct.POINTER(CHierarchicalRecognizer),  # hr
            _dp,                                   # images
            c_int_p,                              # widths
            c_int_p,                              # heights
            c_int_p,                              # class_labels
            c_int_p,                              # common_labels
            ct.c_int,                             # n_samples
            ct.c_int,                             # total_classes
            ct.c_int,                             # class_num_method
            ct.c_int,                             # radius_method
            ct.c_int,                             # weight_update_method
            ct.c_double                           # custom_radius
        ]
        self.lib.trainHierarchicalRecognizer.restype = None
        
        # 予測
        self.lib.predictHierarchicalRecognizer.argtypes = [
            ct.POINTER(CHierarchicalRecognizer),
            c_double_p,
            ct.c_int,
            ct.c_int
        ]
        self.lib.predictHierarchicalRecognizer.restype = ct.c_int
        
        # バッチ予測
        self.lib.batchPredictHierarchical.argtypes = [
            ct.POINTER(CHierarchicalRecognizer),
            _dp,
            c_int_p,
            c_int_p,
            ct.c_int
        ]
        self.lib.batchPredictHierarchical.restype = c_int_p
        
        # メモリ解放
        self.lib.freeHierarchicalRecognizer.argtypes = [ct.POINTER(CHierarchicalRecognizer)]
        self.lib.freeHierarchicalRecognizer.restype = None
        
        self.lib.freeIntArray.argtypes = [c_int_p]
        self.lib.freeIntArray.restype = None
    
    def train(self, 
              images: List[np.ndarray],
              labels: np.ndarray,
              common_labels: Optional[np.ndarray] = None):
        """
        学習（論文3.1節）
        
        Args:
            images: 学習画像のリスト
            labels: クラスラベル
            common_labels: 共通部グループラベル（省略時は各クラスを独立した共通部として扱う）
        """
        n_samples = len(images)
        self.total_classes = len(np.unique(labels))
        
        print("=" * 70)
        print(" 階層構造的画像パターン認識器 - 学習")
        print("=" * 70)
        print(f"サンプル数: {n_samples}")
        print(f"クラス数: {self.total_classes}")
        print(f"ピラミッド階層数: {self.num_pyramid_levels}")
        print(f"画像サイズ: {self.target_image_size}x{self.target_image_size}")
        print(f"クラス数設定法: {ClassNumMethod(self.class_num_method).name}")
        print(f"半径値設定法: {RadiusMethod(self.radius_method).name}")
        print(f"結合係数更新法: {WeightUpdateMethod(self.weight_update_method).name}")
        print()
        
        # 共通部ラベルの設定
        if common_labels is None:
            common_labels = labels.copy()
        
        # 前処理
        print("画像の前処理中...")
        tic()
        processed_images = []
        widths = []
        heights = []
        
        for i, img in enumerate(images):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  処理中: {i+1}/{n_samples}")
            
            processed = self.preprocessor.preprocess(img, self.target_image_size)
            processed_images.append(processed.flatten())
            heights.append(self.target_image_size)
            widths.append(self.target_image_size)
        
        toc("前処理")
        
        # numpy配列に変換
        processed_array = np.array(processed_images, dtype=np.float64)
        widths_array = np.array(widths, dtype=np.int32)
        heights_array = np.array(heights, dtype=np.int32)
        labels_array = labels.astype(np.int32)
        common_array = common_labels.astype(np.int32)
        
        # C++用に変換
        images_ptr, _, _ = numpy_to_c_2d(processed_array)
        widths_ptr = widths_array.ctypes.data_as(c_int_p)
        heights_ptr = heights_array.ctypes.data_as(c_int_p)
        labels_ptr = labels_array.ctypes.data_as(c_int_p)
        common_ptr = common_array.ctypes.data_as(c_int_p)
        
        # 認識器の作成
        self.c_recognizer = self.lib.createHierarchicalRecognizer(self.num_pyramid_levels)
        
        # 学習実行（C++側で処理）
        print("\nC++側で学習を実行中...")
        tic()
        self.lib.trainHierarchicalRecognizer(
            self.c_recognizer,
            images_ptr,
            widths_ptr,
            heights_ptr,
            labels_ptr,
            common_ptr,
            n_samples,
            self.total_classes,
            self.class_num_method,
            self.radius_method,
            self.weight_update_method,
            self.custom_radius
        )
        toc("学習")
        
        print("\n" + "=" * 70)
        print(" 学習完了")
        print("=" * 70)
    
    def predict(self, img: np.ndarray) -> int:
        """
        予測（論文3.6節）
        
        Args:
            img: 入力画像
        
        Returns:
            予測クラスID
        """
        if self.c_recognizer is None:
            raise RuntimeError("モデルが学習されていません。先にtrain()を実行してください。")
        
        # 前処理
        processed = self.preprocessor.preprocess(img, self.target_image_size)
        processed_flat = processed.flatten()
        
        # C++用に変換
        img_ptr, img_len = numpy_to_c_1d(processed_flat)
        
        # 予測実行（C++側）
        class_id = self.lib.predictHierarchicalRecognizer(
            self.c_recognizer,
            img_ptr,
            self.target_image_size,
            self.target_image_size
        )
        
        return class_id
    
    def evaluate(self,
                test_images: List[np.ndarray],
                test_labels: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        評価
        
        Args:
            test_images: テスト画像のリスト
            test_labels: 正解ラベル
        
        Returns:
            (認識率, 予測ラベル配列)
        """
        if self.c_recognizer is None:
            raise RuntimeError("モデルが学習されていません。")
        
        n_test = len(test_images)
        print(f"\n評価中: {n_test}サンプル")
        
        # 前処理
        processed_images = []
        for i, img in enumerate(test_images):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  前処理: {i+1}/{n_test}")
            processed = self.preprocessor.preprocess(img, self.target_image_size)
            processed_images.append(processed.flatten())
        
        # numpy配列に変換
        processed_array = np.array(processed_images, dtype=np.float64)
        widths_array = np.full(n_test, self.target_image_size, dtype=np.int32)
        heights_array = np.full(n_test, self.target_image_size, dtype=np.int32)
        
        # C++用に変換
        images_ptr, _, _ = numpy_to_c_2d(processed_array)
        widths_ptr = widths_array.ctypes.data_as(c_int_p)
        heights_ptr = heights_array.ctypes.data_as(c_int_p)
        
        # バッチ予測実行（C++側）
        tic()
        predictions_ptr = self.lib.batchPredictHierarchical(
            self.c_recognizer,
            images_ptr,
            widths_ptr,
            heights_ptr,
            n_test
        )
        toc("予測")
        
        # 結果を取得
        predictions = np.array([predictions_ptr[i] for i in range(n_test)])
        
        # メモリ解放
        self.lib.freeIntArray(predictions_ptr)
        
        # 認識率計算
        correct = np.sum(predictions == test_labels)
        accuracy = 100.0 * correct / n_test
        
        print(f"\n認識率: {accuracy:.2f}% ({correct}/{n_test})")
        
        return accuracy, predictions
    
    def __del__(self):
        """デストラクタ"""
        if hasattr(self, 'c_recognizer') and self.c_recognizer:
            self.lib.freeHierarchicalRecognizer(self.c_recognizer)

#==============================================================
# サンプルデータ生成
#==============================================================

def create_sample_data(n_samples_per_class: int = 100) -> Tuple[List, np.ndarray]:
    """
    サンプルデータ生成（論文の例に基づく）
    """
    images = []
    labels = []
    img_size = 32
    
    # クラス0: 「門構え」風（上部が開いた四角）
    for _ in range(n_samples_per_class):
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.rectangle(img, (8, 8), (24, 24), 255, 2)
        img[8:10, 8:24] = 0  # 上部を消す
        noise = np.random.randint(-30, 30, img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(img)
        labels.append(0)
    
    # クラス1: 「門構え」+「口」→「問」風
    for _ in range(n_samples_per_class):
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.rectangle(img, (8, 8), (24, 24), 255, 2)
        img[8:10, 8:24] = 0
        cv2.rectangle(img, (12, 14), (20, 20), 255, 2)
        noise = np.random.randint(-30, 30, img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(img)
        labels.append(1)
    
    # クラス2: 「門構え」+「日」→「間」風
    for _ in range(n_samples_per_class):
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.rectangle(img, (8, 8), (24, 24), 255, 2)
        img[8:10, 8:24] = 0
        cv2.rectangle(img, (12, 14), (20, 20), 255, 2)
        cv2.line(img, (12, 17), (20, 17), 255, 2)
        noise = np.random.randint(-30, 30, img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(img)
        labels.append(2)
    
    return images, np.array(labels)

#==============================================================
# ETL8B漢字データ読み込み
#==============================================================

def load_etl8b_data(base_path: str, 
                    target_classes: List[str],
                    max_samples: int = 50,
                    target_size: int = 32) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    ETL8Bデータセットから漢字画像を読み込む
    
    Args:
        base_path: ETL8B画像のベースパス（例: "./ETL8B-img-full"）
        target_classes: 読み込む漢字のHexコードリスト（例: ['3a2c', '3a2e', ...]）
        max_samples: 各クラスから読み込む最大サンプル数
        target_size: リサイズ後の画像サイズ
    
    Returns:
        images: 画像配列 [n_samples, height, width]
        labels: クラスラベル配列 [n_samples]
        class_info: クラス情報辞書 {class_id: char}
    """
    import os
    
    images = []
    labels = []
    class_info = {}
    
    print("ETL8Bデータ読み込み中...")
    print(f"ベースパス: {base_path}")
    print(f"対象クラス数: {len(target_classes)}")
    print("-" * 70)
    
    for class_idx, class_hex in enumerate(target_classes):
        class_dir = os.path.join(base_path, class_hex)
        
        # ディレクトリ存在チェック
        if not os.path.exists(class_dir):
            print(f"警告: ディレクトリが見つかりません: {class_dir}")
            continue
        
        # 画像ファイル取得
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        
        # サンプル数制限
        if len(image_files) > max_samples:
            np.random.seed(42)
            image_files = np.random.choice(image_files, max_samples, replace=False)
        
        # 画像読み込み
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # リサイズ
                if img.shape != (target_size, target_size):
                    img = cv2.resize(img, (target_size, target_size))
                
                # 正規化
                img = img.astype(np.float64) / 255.0
                
                images.append(img)
                labels.append(class_idx)
        
        # 文字情報
        try:
            decimal = int(class_hex, 16)
            char = chr(decimal) if decimal < 0x10000 else f'U+{class_hex}'
            class_info[class_idx] = char
        except:
            char = f'[{class_hex}]'
            class_info[class_idx] = char
        
        sample_count = len([l for l in labels if l == class_idx])
        print(f"  クラス {class_idx}: {class_hex} → {char} ({sample_count}サンプル)")
    
    print("-" * 70)
    print(f"読み込み完了: {len(images)}サンプル, {len(set(labels))}クラス")
    print()
    
    return np.array(images), np.array(labels), class_info

#==============================================================
# メインプログラム
#==============================================================

def test_with_synthetic_data():
    """合成データでのテスト"""
    print("=" * 70)
    print(" 階層構造的画像パターン認識器 - 合成データテスト")
    print("=" * 70)
    print()
    
    # サンプルデータ生成
    print("合成データ生成中...")
    train_images, train_labels = create_sample_data(n_samples_per_class=150)
    test_images, test_labels = create_sample_data(n_samples_per_class=50)
    
    # データのシャッフル
    train_indices = np.random.permutation(len(train_images))
    train_images = [train_images[i] for i in train_indices]
    train_labels = train_labels[train_indices]
    
    test_indices = np.random.permutation(len(test_images))
    test_images = [test_images[i] for i in test_indices]
    test_labels = test_labels[test_indices]
    
    print(f"訓練データ: {len(train_images)}サンプル")
    print(f"テストデータ: {len(test_images)}サンプル")
    print()
    
    # 共通部ラベル（全て「門構え」グループ）
    common_train = np.zeros_like(train_labels)
    
    # 認識器の構築
    recognizer = HierarchicalPatternRecognizer(
        lib_path='./hierarchical_ext.so',
        num_pyramid_levels=5,
        target_image_size=32,
        class_num_method=ClassNumMethod.INCREMENTAL,
        radius_method=RadiusMethod.METHOD1,
        weight_update_method=WeightUpdateMethod.NO_UPDATE
    )
    
    # 学習
    recognizer.train(train_images, train_labels, common_train)
    
    # 評価
    accuracy, predictions = recognizer.evaluate(test_images, test_labels)
    
    # 混同行列
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_labels, predictions)
        print("\n混同行列:")
        print(cm)
    except ImportError:
        pass
    
    print("\n" + "=" * 70)
    print(" テスト完了")
    print("=" * 70)
    
    return accuracy

def test_with_etl8b_data(base_path='./ETL8B-img-full', max_samples=50):
    """ETL8B漢字データでのテスト"""
    print("=" * 70)
    print(" 階層構造的画像パターン認識器 - ETL8B漢字認識テスト")
    print(" 論文実装: 門構え、しんにょう、りっしんべんの認識")
    print("=" * 70)
    print()
    
    # 論文の例に基づく漢字選択
    # グループ0: 門構え
    kanji_groups = {
        'group0_monkamae': {
            'name': '門構え',
            'classes': ['9580', '9593', '9580']  # 間、問、開など
        },
        'group1_shinnyou': {
            'name': 'しんにょう',
            'classes': ['8fd4', '8fba', '9042']  # 返、辺、遂など  
        },
        'group2_risshinben': {
            'name': 'りっしんべん',
            'classes': ['6027', '5feb', '6094']  # 性、快、悔など
        }
    }
    
    # または汎用的な12クラス
    target_classes = [
        '3a2c',  # 㨬
        '3a2e',  # 㨮
        '3a4e',  # 㩎
        '3a59',  # 㩙
        '3a6e',  # 㩮
        '436d',  # 䍭
        '3c23',  # 㰣
        '3c72',  # 㱲
        '3c78',  # 㱸
        '3d26',  # 㴦
        '3d3b',  # 㴻
        '3d3e'   # 㴾
    ]
    
    # データ読み込み
    images, labels, class_info = load_etl8b_data(
        base_path, 
        target_classes,
        max_samples=max_samples
    )
    
    if len(images) == 0:
        print("エラー: データが読み込めませんでした")
        print(f"ディレクトリを確認してください: {base_path}")
        return None
    
    # 訓練・テスト分割
    from sklearn.model_selection import train_test_split
    
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, 
        test_size=0.3, 
        random_state=42, 
        stratify=labels
    )
    
    print(f"データ分割:")
    print(f"  訓練データ: {len(train_images)}サンプル")
    print(f"  テストデータ: {len(test_images)}サンプル")
    print()
    
    # 共通部ラベルの設定（簡易版：各クラスを独立した共通部として扱う）
    common_train = train_labels.copy()
    
    # 複数の設定で評価
    configs = [
        {
            'name': '設定1: 動的クラス数、半径法1、更新なし',
            'class_num_method': ClassNumMethod.INCREMENTAL,
            'radius_method': RadiusMethod.METHOD1,
            'weight_update_method': WeightUpdateMethod.NO_UPDATE,
            'num_levels': 5,
            'image_size': 32
        },
        {
            'name': '設定2: 固定クラス数、半径法2、平均更新',
            'class_num_method': ClassNumMethod.FIXED,
            'radius_method': RadiusMethod.METHOD2,
            'weight_update_method': WeightUpdateMethod.AVERAGE,
            'num_levels': 5,
            'image_size': 32
        },
        {
            'name': '設定3: 動的クラス数、半径法1、平均更新、階層7',
            'class_num_method': ClassNumMethod.INCREMENTAL,
            'radius_method': RadiusMethod.METHOD1,
            'weight_update_method': WeightUpdateMethod.AVERAGE,
            'num_levels': 7,
            'image_size': 32
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'=' * 70}")
        print(config['name'])
        print('=' * 70)
        
        try:
            recognizer = HierarchicalPatternRecognizer(
                lib_path='./hierarchical_ext.so',
                num_pyramid_levels=config['num_levels'],
                target_image_size=config['image_size'],
                class_num_method=config['class_num_method'],
                radius_method=config['radius_method'],
                weight_update_method=config['weight_update_method']
            )
            
            # 学習
            recognizer.train(train_images, train_labels, common_train)
            
            # 評価
            accuracy, predictions = recognizer.evaluate(test_images, test_labels)
            
            results.append({
                'config': config['name'],
                'accuracy': accuracy,
                'predictions': predictions
            })
            
            # クラス別認識率
            print("\nクラス別認識率:")
            for class_id, char in sorted(class_info.items()):
                mask = test_labels == class_id
                if np.sum(mask) > 0:
                    class_acc = 100.0 * np.sum(
                        predictions[mask] == class_id
                    ) / np.sum(mask)
                    print(f"  クラス{class_id} ({char}): {class_acc:.2f}%")
            
        except Exception as e:
            print(f"\nエラー: {e}")
            import traceback
            traceback.print_exc()
    
    # 結果サマリー
    print("\n" + "=" * 70)
    print(" 結果サマリー")
    print("=" * 70)
    
    for result in results:
        print(f"{result['config']}: {result['accuracy']:.2f}%")
    
    print("\n" + "=" * 70)
    print(" テスト完了")
    print("=" * 70)
    
    return results

def main():
    """メインプログラム"""
    import sys
    import os
    
    print("=" * 70)
    print(" 階層構造的画像パターン認識器")
    print("=" * 70)
    print()
    
    # コマンドライン引数チェック
    if len(sys.argv) > 1:
        if sys.argv[1] == '--synthetic':
            # 合成データでテスト
            test_with_synthetic_data()
            return
        elif sys.argv[1] == '--etl8b':
            # ETL8Bデータでテスト
            base_path = sys.argv[2] if len(sys.argv) > 2 else './ETL8B-img-full'
            max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 50
            test_with_etl8b_data(base_path, max_samples)
            return
        elif sys.argv[1] == '--help':
            print("使用方法:")
            print("  python hierarchical_recognizer.py                    # 自動判定")
            print("  python hierarchical_recognizer.py --synthetic        # 合成データ")
            print("  python hierarchical_recognizer.py --etl8b [path] [n] # ETL8Bデータ")
            print("  python hierarchical_recognizer.py --help             # ヘルプ")
            print()
            print("例:")
            print("  python hierarchical_recognizer.py --etl8b ./ETL8B-img-full 30")
            return
    
    # 自動判定: ETL8Bデータがあればそれを使用、なければ合成データ
    etl8b_path = './ETL8B-img-full'
    if os.path.exists(etl8b_path) and os.path.isdir(etl8b_path):
        print("ETL8Bデータディレクトリを検出しました")
        print(f"パス: {etl8b_path}")
        print()
        test_with_etl8b_data(etl8b_path)
    else:
        print("ETL8Bデータが見つかりません")
        print("合成データでテストを実行します")
        print()
        print("【注意】実際の漢字データを使用する場合:")
        print(f"  1. ETL8B画像を {etl8b_path} に配置")
        print(f"  2. または: python hierarchical_recognizer.py --etl8b <パス>")
        print()
        
        try:
            test_with_synthetic_data()
        except Exception as e:
            print(f"\nエラー: {e}")
            print("\n【重要】C++ライブラリのコンパイルが必要です")
            print("コンパイル方法:")
            print("  make")
            print("または:")
            print("  g++ -shared -fPIC -O3 -o hierarchical_ext.so hierarchical_recognizer.cpp")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()