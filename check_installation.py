"""
インストール確認スクリプト
階層構造的画像パターン認識器に必要なライブラリの確認

使用方法:
    python check_installation.py
"""

import sys
import platform

def check_python_version():
    """Pythonバージョンの確認"""
    print("=" * 60)
    print("システム情報")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Pythonバージョン: {sys.version}")
    print(f"Python実行パス: {sys.executable}")
    
    # バージョンチェック
    version_info = sys.version_info
    if version_info < (3, 7):
        print("\n⚠ 警告: Python 3.7以上が推奨されています")
        return False
    elif version_info >= (3, 12):
        print("\n⚠ 注意: Python 3.12は一部ライブラリのサポートが限定的です")
        return True
    else:
        print("\n✓ Pythonバージョン: OK")
        return True

def check_library(module_name, package_name, required=True):
    """ライブラリの確認"""
    try:
        lib = __import__(module_name)
        version = getattr(lib, '__version__', 'バージョン不明')
        
        # バージョン警告
        warnings = []
        if module_name == 'numpy':
            major = int(version.split('.')[0])
            if major < 1:
                warnings.append("バージョンが古すぎます（1.19.0以上推奨）")
        elif module_name == 'cv2':
            major, minor = map(int, version.split('.')[:2])
            if major < 4 or (major == 4 and minor < 5):
                warnings.append("バージョンが古すぎます（4.5.0以上推奨）")
        
        status = "✓"
        if warnings:
            status = "⚠"
        
        print(f"  {status} {package_name:20s}: {version}")
        for warning in warnings:
            print(f"      → {warning}")
        
        return True, version
    except ImportError:
        if required:
            print(f"  ✗ {package_name:20s}: インストールされていません（必須）")
        else:
            print(f"  - {package_name:20s}: インストールされていません（オプション）")
        return False, None

def check_all_libraries():
    """全ライブラリのチェック"""
    results = {}
    
    # 必須ライブラリ
    print("\n" + "=" * 60)
    print("必須ライブラリ（コア機能）")
    print("=" * 60)
    
    required_libs = [
        ('numpy', 'numpy', True),
        ('cv2', 'opencv-python', True),
        ('matplotlib', 'matplotlib', True),
    ]
    
    for module, package, req in required_libs:
        success, version = check_library(module, package, req)
        results[package] = success
    
    # 推奨ライブラリ
    print("\n" + "=" * 60)
    print("推奨ライブラリ（結果分析・可視化）")
    print("=" * 60)
    
    recommended_libs = [
        ('sklearn', 'scikit-learn', False),
        ('seaborn', 'seaborn', False),
        ('tqdm', 'tqdm', False),
    ]
    
    for module, package, req in recommended_libs:
        success, version = check_library(module, package, req)
        results[package] = success
    
    # オプションライブラリ
    print("\n" + "=" * 60)
    print("オプションライブラリ（拡張機能）")
    print("=" * 60)
    
    optional_libs = [
        ('pandas', 'pandas', False),
        ('PIL', 'Pillow', False),
    ]
    
    for module, package, req in optional_libs:
        success, version = check_library(module, package, req)
        results[package] = success
    
    return results

def run_functionality_tests():
    """簡易機能テスト"""
    print("\n" + "=" * 60)
    print("簡易機能テスト")
    print("=" * 60)
    
    all_passed = True
    
    # NumPyテスト
    try:
        import numpy as np
        arr = np.array([[1, 2], [3, 4]])
        assert arr.shape == (2, 2)
        print("  ✓ NumPy配列作成: OK")
    except Exception as e:
        print(f"  ✗ NumPy配列作成: 失敗 ({e})")
        all_passed = False
    
    # OpenCVテスト
    try:
        import cv2
        import numpy as np
        img = np.zeros((64, 64), dtype=np.uint8)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        assert binary.shape == (64, 64)
        print("  ✓ OpenCV画像処理: OK")
    except Exception as e:
        print(f"  ✗ OpenCV画像処理: 失敗 ({e})")
        all_passed = False
    
    # Matplotlibテスト
    try:
        import matplotlib
        matplotlib.use('Agg')  # GUIなしバックエンド
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.close(fig)
        print("  ✓ Matplotlib描画: OK")
    except Exception as e:
        print(f"  ✗ Matplotlib描画: 失敗 ({e})")
        all_passed = False
    
    # scikit-learnテスト（オプション）
    try:
        from sklearn.metrics import confusion_matrix
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
        print("  ✓ scikit-learn: OK")
    except ImportError:
        print("  - scikit-learn: スキップ（未インストール）")
    except Exception as e:
        print(f"  ⚠ scikit-learn: エラー ({e})")
    
    # seabornテスト（オプション）
    try:
        import seaborn as sns
        print("  ✓ seaborn: OK")
    except ImportError:
        print("  - seaborn: スキップ（未インストール）")
    except Exception as e:
        print(f"  ⚠ seaborn: エラー ({e})")
    
    return all_passed

def print_installation_instructions(results):
    """インストール手順の表示"""
    missing_required = [pkg for pkg, installed in results.items() 
                       if not installed and pkg in ['numpy', 'opencv-python', 'matplotlib']]
    missing_recommended = [pkg for pkg, installed in results.items()
                          if not installed and pkg in ['scikit-learn', 'seaborn', 'tqdm']]
    
    if missing_required:
        print("\n" + "=" * 60)
        print("必須ライブラリが不足しています")
        print("=" * 60)
        print("\n以下のコマンドでインストールしてください：")
        print(f"\npip install {' '.join(missing_required)}")
    
    if missing_recommended:
        print("\n" + "=" * 60)
        print("推奨ライブラリが不足しています")
        print("=" * 60)
        print("\n結果の可視化・分析機能を使用する場合は以下をインストール：")
        print(f"\npip install {' '.join(missing_recommended)}")

def print_summary(python_ok, results, tests_passed):
    """結果サマリーの表示"""
    print("\n" + "=" * 60)
    print("確認結果サマリー")
    print("=" * 60)
    
    required_ok = all(results.get(pkg, False) for pkg in ['numpy', 'opencv-python', 'matplotlib'])
    
    if python_ok and required_ok and tests_passed:
        print("\n✓ 全ての必須要件を満たしています！")
        print("  システムは正常に動作します。")
        
        missing_opt = sum(1 for pkg, installed in results.items() 
                         if not installed and pkg not in ['numpy', 'opencv-python', 'matplotlib'])
        if missing_opt > 0:
            print(f"\n⚠ {missing_opt}個のオプションライブラリが未インストールです")
            print("  基本機能は使用できますが、一部機能が制限されます。")
    else:
        print("\n✗ 要件を満たしていません")
        if not python_ok:
            print("  - Pythonバージョンを確認してください")
        if not required_ok:
            print("  - 必須ライブラリをインストールしてください")
        if not tests_passed:
            print("  - ライブラリに問題がある可能性があります")
    
    print("\n次のステップ:")
    if not required_ok:
        print("  1. 上記のインストールコマンドを実行")
        print("  2. このスクリプトを再実行して確認")
    else:
        print("  1. ETL8Bデータセットを配置")
        print("  2. python etl8b_loader.py でデータセット確認")
        print("  3. python etl8b_recognition_main.py で実行")
    
    print("\n詳細なインストール手順:")
    print("  README.md または INSTALL.md を参照")

def main():
    """メイン処理"""
    print("\n階層構造的画像パターン認識器")
    print("インストール確認スクリプト")
    
    # Pythonバージョン確認
    python_ok = check_python_version()
    
    # ライブラリ確認
    results = check_all_libraries()
    
    # 機能テスト
    tests_passed = run_functionality_tests()
    
    # インストール手順
    print_installation_instructions(results)
    
    # サマリー
    print_summary(python_ok, results, tests_passed)
    
    print("\n" + "=" * 60)
    print("確認完了")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n確認を中断しました")
    except Exception as e:
        print(f"\n予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()