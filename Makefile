# Makefile for Hierarchical Pattern Recognizer
# 階層構造的画像パターン認識器のビルドシステム
#
# 構造: pnn1.cpp/pyと同じく、C++で計算、Pythonでインターフェース

CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -fPIC
LDFLAGS = -shared

# ターゲット
TARGET = hierarchical_ext.so
SRC = hierarchical_recognizer.cpp
PYTHON_WRAPPER = hierarchical_recognizer.py

# デフォルトターゲット
all: $(TARGET)
	@echo ""
	@echo "=========================================="
	@echo " ビルド完了"
	@echo "=========================================="
	@echo "使用方法:"
	@echo "  make test    - デモプログラムを実行"
	@echo "  make clean   - ビルド成果物を削除"
	@echo ""

# 共有ライブラリのビルド
$(TARGET): $(SRC)
	@echo "階層構造的認識器をコンパイル中..."
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<
	@echo "コンパイル完了: $(TARGET)"

# クリーンアップ
clean:
	@echo "クリーンアップ中..."
	rm -f $(TARGET) *.o
	rm -rf __pycache__
	@echo "完了"

# テスト実行
test: $(TARGET)
	@echo ""
	@echo "=========================================="
	@echo " テストプログラム実行"
	@echo "=========================================="
	@echo ""
	python3 $(PYTHON_WRAPPER)

# ETL8Bテスト
test-etl8b: $(TARGET)
	@echo ""
	@echo "=========================================="
	@echo " ETL8Bシンプルテスト"
	@echo "=========================================="
	@echo ""
	python3 test_etl8b.py simple ./ETL8B-img-full 30

test-etl8b-full: $(TARGET)
	@echo ""
	@echo "=========================================="
	@echo " ETL8Bフルテスト"
	@echo "=========================================="
	@echo ""
	python3 test_etl8b.py full ./ETL8B-img-full 50

test-etl8b-compare: $(TARGET)
	@echo ""
	@echo "=========================================="
	@echo " ETL8B比較テスト"
	@echo "=========================================="
	@echo ""
	python3 test_etl8b.py compare ./ETL8B-img-full

# ETL8B 15クラステスト
test-etl8b-15-basic: $(TARGET)
	@echo ""
	@echo "=========================================="
	@echo " ETL8B 15クラス - 基本テスト"
	@echo "=========================================="
	@echo ""
	python3 test_etl8b_15classes.py basic ./ETL8B-img-full

test-etl8b-15-compare: $(TARGET)
	@echo ""
	@echo "=========================================="
	@echo " ETL8B 15クラス - 複数設定比較"
	@echo "=========================================="
	@echo ""
	python3 test_etl8b_15classes.py compare ./ETL8B-img-full

test-etl8b-15-sample: $(TARGET)
	@echo ""
	@echo "=========================================="
	@echo " ETL8B 15クラス - サンプル数変動"
	@echo "=========================================="
	@echo ""
	python3 test_etl8b_15classes.py sample ./ETL8B-img-full

test-etl8b-15-all: $(TARGET)
	@echo ""
	@echo "=========================================="
	@echo " ETL8B 15クラス - 完全実験"
	@echo "=========================================="
	@echo ""
	python3 test_etl8b_15classes.py all ./ETL8B-img-full

# C++のデバッグビルド
debug: $(SRC)
	@echo "デバッグ版をコンパイル中..."
	$(CXX) -g -O0 -std=c++11 -Wall -fPIC $(LDFLAGS) -o $(TARGET) $<
	@echo "デバッグ版コンパイル完了"

# 最適化ビルド（より積極的）
optimize: $(SRC)
	@echo "最適化版をコンパイル中..."
	$(CXX) -O3 -march=native -ffast-math -funroll-loops \
	       -std=c++11 -Wall -fPIC $(LDFLAGS) -o $(TARGET) $<
	@echo "最適化版コンパイル完了"

# ヘルプ
help:
	@echo "使用可能なターゲット:"
	@echo "  make                      - ライブラリをビルド"
	@echo "  make test                 - デモプログラムを実行（合成データ）"
	@echo ""
	@echo "ETL8B 4クラステスト:"
	@echo "  make test-etl8b           - ETL8Bシンプルテスト（4クラス、30サンプル）"
	@echo "  make test-etl8b-full      - ETL8Bフルテスト（12クラス、複数設定）"
	@echo "  make test-etl8b-compare   - ETL8B比較テスト（サンプル数変動）"
	@echo ""
	@echo "ETL8B 15クラステスト:"
	@echo "  make test-etl8b-15-basic  - 15クラス基本テスト"
	@echo "  make test-etl8b-15-compare - 15クラス複数設定比較"
	@echo "  make test-etl8b-15-sample - 15クラスサンプル数変動"
	@echo "  make test-etl8b-15-all    - 15クラス完全実験（すべて実行）"
	@echo ""
	@echo "その他:"
	@echo "  make clean                - ビルド成果物を削除"
	@echo "  make debug                - デバッグ版をビルド"
	@echo "  make optimize             - 最適化版をビルド"
	@echo "  make help                 - このヘルプを表示"
	@echo ""
	@echo "ファイル構成:"
	@echo "  $(SRC)               - C++実装（計算コア）"
	@echo "  $(PYTHON_WRAPPER)    - Pythonラッパー"
	@echo "  test_etl8b.py        - ETL8Bテストスクリプト"
	@echo "  test_etl8b_15classes.py - ETL8B 15クラステストスクリプト"
	@echo "  $(TARGET)         - 共有ライブラリ"

# インストール（オプション）
install: $(TARGET)
	@echo "ライブラリをインストール中..."
	sudo cp $(TARGET) /usr/local/lib/
	sudo ldconfig
	@echo "インストール完了"

.PHONY: all clean test test-etl8b test-etl8b-full test-etl8b-compare \
        test-etl8b-15-basic test-etl8b-15-compare test-etl8b-15-sample test-etl8b-15-all \
        debug optimize help install

#==============================================================
# 使用方法とセットアップ
#==============================================================
#
# 1. 必要なパッケージのインストール
#    Ubuntu/Debian:
#      sudo apt-get install build-essential python3-dev python3-numpy python3-opencv
#
#    macOS:
#      brew install python3
#      pip3 install numpy opencv-python
#
# 2. ビルド
#      make
#
# 3. テスト実行
#      make test
#
#==============================================================
# アーキテクチャ
#==============================================================
#
# このシステムは pnn1.cpp/py と同じ構造を採用しています:
#
# [Python側]
# - データの読み込み、前処理（OpenCV）
# - 画像の2値化、トリミング、正規化
# - 結果の表示、可視化
# - インターフェース、API
#
# [C++側]
# - 階層ピラミッド構造の生成
# - カーネルメモリー分類器の学習・予測
# - 共通部・非共通部の処理
# - すべての計算集約的処理
#
# Python -(ctypes)-> C++ Shared Library
#         ↓
#    高速な計算処理
#
#==============================================================