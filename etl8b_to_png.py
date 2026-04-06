"""
ETL8B バイナリファイル → PNG画像変換ツール

ETL8Bのバイナリフォーマットを読み込み、PNG画像に変換します。

使用方法:
python etl8b_to_png.py \
    --input ./ETL8B/ETL8B2C1 \
    --output ./etl8b_png \
    --structure flat \
    --target_chars 聞,間,問,返,辺,性,快
"""

import numpy as np
import cv2
import struct
from pathlib import Path
from typing import List, Tuple, Optional, Set
import argparse
from tqdm import tqdm

#==============================================================
# ETL8B バイナリリーダー
#==============================================================
class ETL8BBinaryReader:
    """ETL8Bバイナリファイルの読み込み"""
    
    RECORD_SIZE = 8199
    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 63
    IMAGE_START = 0x30
    IMAGE_BYTES = 2016
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    
    def _unpack_image(self, data: bytes) -> np.ndarray:
        """4ビットパッキングされた画像を展開"""
        img_data = np.zeros(self.IMAGE_WIDTH * self.IMAGE_HEIGHT, dtype=np.uint8)
        
        for i in range(len(data)):
            if i * 2 < len(img_data):
                img_data[i * 2] = (data[i] >> 4) * 16
            if i * 2 + 1 < len(img_data):
                img_data[i * 2 + 1] = (data[i] & 0x0F) * 16
        
        img = img_data[:self.IMAGE_WIDTH * self.IMAGE_HEIGHT].reshape(
            self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
        
        return img
    
    def _jis_to_unicode(self, jis_code: int) -> Optional[str]:
        """JIS X 0208コードをUnicode文字に変換"""
        try:
            # JIS X 0208 → Shift_JIS → Unicode
            j1 = (jis_code >> 8) & 0xFF
            j2 = jis_code & 0xFF
            
            # JIS X 0208 → Shift_JIS変換
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
            
            sjis_bytes = bytes([s1, s2])
            char = sjis_bytes.decode('shift_jis')
            return char
        except:
            return None
    
    def read_all(self, target_chars: Optional[Set[str]] = None) -> List[Tuple[np.ndarray, str]]:
        """
        全レコードを読み込み
        
        Args:
            target_chars: 対象文字のセット（Noneの場合は全文字）
        
        Returns:
            (画像, 文字)のタプルリスト
        """
        results = []
        
        with open(self.file_path, 'rb') as f:
            file_size = self.file_path.stat().st_size
            total_records = file_size // self.RECORD_SIZE
            
            with tqdm(total=total_records, desc="読み込み中") as pbar:
                while True:
                    record = f.read(self.RECORD_SIZE)
                    
                    if len(record) < self.RECORD_SIZE:
                        break
                    
                    # 文字コード取得
                    char_code = struct.unpack('>H', record[2:4])[0]
                    char = self._jis_to_unicode(char_code)
                    
                    if char is None:
                        pbar.update(1)
                        continue
                    
                    # 対象文字のフィルタ
                    if target_chars is not None and char not in target_chars:
                        pbar.update(1)
                        continue
                    
                    # 画像データ展開
                    img_data = record[self.IMAGE_START:self.IMAGE_START + self.IMAGE_BYTES]
                    img = self._unpack_image(img_data)
                    
                    results.append((img, char))
                    pbar.update(1)
        
        return results

#==============================================================
# PNG変換・保存
#==============================================================
class PNGConverter:
    """画像データをPNG形式で保存"""
    
    @staticmethod
    def save_flat_structure(output_dir: Path, 
                           data: List[Tuple[np.ndarray, str]],
                           invert: bool = False):
        """
        フラット構造で保存
        
        出力: output_dir/漢字_番号.png
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 文字ごとにカウント
        char_count = {}
        
        print(f"\nフラット構造で保存中: {output_dir}")
        for img, char in tqdm(data, desc="保存中"):
            count = char_count.get(char, 0) + 1
            char_count[char] = count
            
            # 画像の反転（白黒反転）
            if invert:
                img = 255 - img
            
            # ファイル名生成
            filename = f"{char}_{count:03d}.png"
            filepath = output_dir / filename
            
            # 保存
            cv2.imwrite(str(filepath), img)
        
        # 統計情報
        print(f"\n保存完了:")
        print(f"  総画像数: {len(data)}")
        print(f"  文字種類数: {len(char_count)}")
        print(f"  各文字のサンプル数:")
        for char, count in sorted(char_count.items()):
            print(f"    {char}: {count}枚")
    
    @staticmethod
    def save_hierarchical_structure(output_dir: Path,
                                   data: List[Tuple[np.ndarray, str]],
                                   invert: bool = False):
        """
        階層構造で保存
        
        出力: output_dir/漢字/番号.png
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 文字ごとにグループ化
        char_groups = {}
        for img, char in data:
            if char not in char_groups:
                char_groups[char] = []
            char_groups[char].append(img)
        
        print(f"\n階層構造で保存中: {output_dir}")
        for char, images in tqdm(char_groups.items(), desc="保存中"):
            # 文字ごとのディレクトリ作成
            char_dir = output_dir / char
            char_dir.mkdir(parents=True, exist_ok=True)
            
            # 画像を保存
            for i, img in enumerate(images, 1):
                if invert:
                    img = 255 - img
                
                filename = f"{i:03d}.png"
                filepath = char_dir / filename
                cv2.imwrite(str(filepath), img)
        
        # 統計情報
        print(f"\n保存完了:")
        print(f"  総画像数: {len(data)}")
        print(f"  文字種類数: {len(char_groups)}")
        print(f"  各文字のサンプル数:")
        for char, images in sorted(char_groups.items()):
            print(f"    {char}: {len(images)}枚")
    
    @staticmethod
    def save_common_grouped_structure(output_dir: Path,
                                     data: List[Tuple[np.ndarray, str]],
                                     kanji_groups: dict,
                                     common_parts: dict,
                                     invert: bool = False):
        """
        共通部グループ構造で保存
        
        出力: output_dir/共通部名/漢字_番号.png
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 共通部ごとにグループ化
        common_data = {cid: [] for cid in common_parts.keys()}
        
        for img, char in data:
            if char in kanji_groups:
                common_id = kanji_groups[char]
                common_data[common_id].append((img, char))
        
        print(f"\n共通部グループ構造で保存中: {output_dir}")
        
        # 各共通部ごとに保存
        for common_id, items in common_data.items():
            if len(items) == 0:
                continue
            
            common_name = common_parts[common_id]
            common_dir = output_dir / common_name
            common_dir.mkdir(parents=True, exist_ok=True)
            
            # 文字ごとにカウント
            char_count = {}
            
            for img, char in tqdm(items, desc=f"{common_name}", leave=False):
                count = char_count.get(char, 0) + 1
                char_count[char] = count
                
                if invert:
                    img = 255 - img
                
                filename = f"{char}_{count:03d}.png"
                filepath = common_dir / filename
                cv2.imwrite(str(filepath), img)
            
            print(f"  {common_name}: {len(items)}枚")
        
        print(f"\n保存完了:")
        print(f"  総画像数: {len(data)}")

#==============================================================
# メイン
#==============================================================
def main():
    parser = argparse.ArgumentParser(
        description='ETL8B バイナリファイル → PNG画像変換')
    parser.add_argument('--input', type=str, required=True,
                       help='ETL8Bバイナリファイルのパス')
    parser.add_argument('--output', type=str, required=True,
                       help='出力ディレクトリ')
    parser.add_argument('--structure', type=str, default='flat',
                       choices=['flat', 'hierarchical', 'common_grouped'],
                       help='出力ディレクトリ構造')
    parser.add_argument('--target_chars', type=str, default=None,
                       help='対象文字（カンマ区切り）例: 聞,間,問,返')
    parser.add_argument('--invert', action='store_true',
                       help='白黒反転（白背景に黒文字にする）')
    parser.add_argument('--preview', action='store_true',
                       help='変換前にサンプル画像をプレビュー')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ETL8B → PNG 変換ツール")
    print("=" * 70)
    
    # 対象文字の解析
    target_chars = None
    if args.target_chars:
        target_chars = set(args.target_chars.split(','))
        print(f"\n対象文字: {', '.join(sorted(target_chars))}")
    else:
        print(f"\n対象文字: 全て")
    
    # バイナリ読み込み
    print(f"\n[ステップ1: バイナリ読み込み]")
    print(f"  入力ファイル: {args.input}")
    
    reader = ETL8BBinaryReader(args.input)
    data = reader.read_all(target_chars)
    
    print(f"\n  読み込み完了: {len(data)}サンプル")
    
    if len(data) == 0:
        print("エラー: 画像が読み込めませんでした")
        return
    
    # プレビュー
    if args.preview:
        print(f"\n[プレビュー]")
        preview_samples(data[:10], args.invert)
    
    # PNG変換・保存
    print(f"\n[ステップ2: PNG変換・保存]")
    print(f"  出力ディレクトリ: {args.output}")
    print(f"  構造: {args.structure}")
    print(f"  白黒反転: {args.invert}")
    
    output_dir = Path(args.output)
    
    if args.structure == 'flat':
        PNGConverter.save_flat_structure(output_dir, data, args.invert)
    
    elif args.structure == 'hierarchical':
        PNGConverter.save_hierarchical_structure(output_dir, data, args.invert)
    
    elif args.structure == 'common_grouped':
        # デフォルトの共通部定義を使用
        kanji_groups = {
            "聞": 0, "間": 0, "問": 0, "開": 0, "閉": 0, "閑": 0,
            "関": 0, "閣": 0, "闇": 0, "闘": 0,
            "返": 1, "辺": 1, "進": 1, "道": 1, "近": 1, "迷": 1,
            "送": 1, "遠": 1, "速": 1, "通": 1, "達": 1, "遂": 1,
            "運": 1, "過": 1, "遅": 1, "連": 1,
            "性": 2, "快": 2, "悔": 2, "情": 2, "慣": 2, "悲": 2,
            "怖": 2, "恐": 2, "恥": 2, "悩": 2, "怪": 2, "悦": 2,
            "恨": 2, "憎": 2, "慌": 2, "憧": 2,
        }
        common_parts = {
            0: "門構え",
            1: "しんにょう",
            2: "りっしんべん"
        }
        
        PNGConverter.save_common_grouped_structure(
            output_dir, data, kanji_groups, common_parts, args.invert)
    
    print("\n" + "=" * 70)
    print("変換完了")
    print("=" * 70)

def preview_samples(data: List[Tuple[np.ndarray, str]], invert: bool = False):
    """サンプル画像のプレビュー"""
    import matplotlib.pyplot as plt
    
    n_samples = min(len(data), 10)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('ETL8B サンプルプレビュー', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        if idx < n_samples:
            img, char = data[idx]
            
            if invert:
                img = 255 - img
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{char}', fontsize=14)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('etl8b_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  プレビュー画像: etl8b_preview.png")

if __name__ == "__main__":
    main()