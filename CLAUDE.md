# CLAUDE.md

## 言語設定
- 常に日本語で会話する
- コメントも日本語で記述する
- エラーメッセージの説明も日本語で行う
- ドキュメントも日本語で生成する

## プロジェクト概要

ETL8Bデータセットを用いた漢字認識研究プロジェクト。階層構造的パターン認識器（HOG特徴+PNN分類器）とMLPディープニューラルネットワークを比較し、提案手法が**認識率97.38%**（15クラス、2415サンプル）を達成。MLPの92.28%を+5.10%上回る。

### 主要コンポーネント
- `kanji_best_standalone.py` - 最良モデル（HOG+PNN+左右分割+多数決投票）
- `hierarchical_recognizer_optimized.py` - 階層パターン認識器（C++拡張）
- `hierarchical_recognizer.cpp` + `hierarchical_ext.so` - C++計算コア
- `compare_full_dataset.py` - フルデータセット比較実験
- `kanji_jikkou.py` - 漢字認識実験メイン

### アーキテクチャ
```
入力(64×64) → 左右分割(argmax projection) → 階層ピラミッド(5段)
  → HOG特徴抽出(9方向,8×8セル) → Z-score正規化
  → PNN分類器(適応的σ, K-means代表点) × 3（全体・偏・旁）
  → 多数決投票 → 最終予測
```

## ビルド・テストコマンド

```bash
# C++拡張ビルド
make                      # hierarchical_ext.so をビルド
make clean                # ビルド成果物を削除

# テスト実行
make test-etl8b-15-basic  # 15クラス基本テスト
make test-etl8b-15-all    # 15クラス完全実験

# ヘルパースクリプト
python tools/run_experiment.py --script kanji_jikkou.py --name "実験名"
python tools/analyze_results.py --all
python tools/paper_checker.py --check all
```

## カスタムスキル一覧

| コマンド | 説明 | 許可ツール |
|---------|------|-----------|
| `/experiment-runner` | 実験実行・ログ記録 | Bash(python3, make, ls), Write |
| `/result-analyzer` | 結果集計・分析 | Read, Bash(python3, ls) |
| `/paper-checker` | 論文整合性チェック | Read, Grep, Glob |
| `/debugger` | エラー原因特定 | Read, Bash(python3, nm, ldd, file), Grep, Glob |

### 使用例
```
/experiment-runner kanji_jikkou.py
/result-analyzer all
/paper-checker all
/debugger hierarchical_ext.so undefined symbol
```

## ヘルパースクリプト (tools/)

| スクリプト | 説明 |
|-----------|------|
| `tools/run_experiment.py` | 実験実行ヘルパー（タイムスタンプ付きログ保存） |
| `tools/analyze_results.py` | 結果集計・比較（認識率テーブル生成） |
| `tools/paper_checker.py` | 論文整合性チェック（[OK]/[WARNING]/[NG]形式） |

## ディレクトリ構造

```
ETL8B-img-full/        # ETL8B画像データ
docs/                  # 論文・アルゴリズム詳細
  ALGORITHM_DETAILS.txt
  EXPERIMENT_RESULTS.txt
results/               # 実験結果
  full_dataset/        # フルデータセット結果（JSON/PNG）
  kanji_jikkou/        # 漢字実行テスト結果
  experiment_logs/     # 実験ログ（run_experiment.pyが生成）
models/                # 保存済みモデル
tools/                 # ヘルパースクリプト
.claude/commands/      # カスタムスラッシュコマンド
```

## 重要なパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| num_pyramid_levels | 5 | 階層ピラミッド段数 |
| HOG orientations | 9 | 勾配方向ビン数 |
| HOG pixels_per_cell | (8,8) | セルサイズ |
| HOG cells_per_block | (2,2) | ブロックサイズ |
| exemplar_ratio | 0.2 | K-means代表点比率 |
| sigma_method | adaptive | PNNのσ決定方法 |
| split_method | projection | 左右分割方式（argmax） |
| random_state | 42 | 乱数シード |

## ワークフロー・オーケストレーション

### 1. プランモードの原則適用
- 3ステップ以上の作業や、アーキテクチャ上の決定を伴う重要なタスクでは、必ず「プランモード」に入ること。
- 作業が停滞したり予期せぬ方向へ進んだりした場合は、即座に中断して計画を練り直すこと。
- 構築だけでなく、検証ステップにおいてもプランモードを活用すること。
- 曖昧さを排除するため、事前に詳細な仕様を記述すること。

### 2. サブエージェントの活用戦略
- メインのコンテキストウィンドウをクリーンに保つため、サブエージェントを積極的に活用すること。
- 調査、探索、並列分析などはサブエージェントに任せること。
- 集中して実行できるよう、1つのサブエージェントにつき1つのタスクを割り当てること。

### 3. 自己改善ループ
- ユーザーから修正指示を受けた後は、そのパターンを `tasks/lessons.md` に記録すること。
- 同じミスを繰り返さないよう、自分自身に対するルールを策定すること。

### 4. 完了前の検証
- 動作の証明ができるまで、タスクを完了と見なさないこと。
- 必要に応じて、変更前後の挙動の差異（diff）を確認すること。
- テストを実行し、ログを確認し、正当性を実証すること。

### 5. エレガントさの追求（バランス重視）
- 重要な変更を加える際は、一度立ち止まって「よりエレガントな方法はないか？」を考えること。
- 対処療法的な修正だと感じた場合は、洗練された解決策を再実装すること。
- 単純で明白な修正については、過剰なエンジニアリングを避けること。

### 6. 自律的なバグ修正
- バグ報告を受けた際は、手取り足取りの指示を待たずに自力で修正すること。
- ログやエラー、失敗したテストを特定し、それらを解決すること。

## タスク管理
1. **計画優先**: チェックリスト形式で計画を書き出す。
2. **計画の確認**: 実装を開始する前に、方針に間違いがないか確認する。
3. **進捗の追跡**: 完了した項目は随時チェックを入れる。
4. **変更の説明**: 各ステップで、何を行ったかハイレベルな要約を提示する。

## コア原則
- **シンプルさの追求**: あらゆる変更を可能な限りシンプルに保つこと。コードへの影響範囲を最小限に抑える。
- **妥協の排除**: 根本原因を突き止めること。一時しのぎの修正は行わない。
- **影響の最小化**: 必要な箇所のみを変更すること。不必要な変更によって新たなバグを作り込まない。
