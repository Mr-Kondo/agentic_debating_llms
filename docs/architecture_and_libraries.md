## 使用フレームワーク / ライブラリ

このリポジトリは、ローカル LLM の議論オーケストレーションを「状態遷移」「構造化データ」「観測」「外部検索」を分離して実装しています。

### LangGraph

- 目的: 議論ワークフローの状態遷移を明示的に管理する
- 主な利用箇所: `app/graph.py`
- 役割:
  - `START -> facilitator` から開始
  - `next_action` による条件分岐
  - `debater -> validator -> summarizer` と `search -> summarizer` のループ
  - continuation 有効時は `finalize -> continuation_facilitator` へ遷移
  - `finalize_continuation -> END` で終了

### Pydantic v2

- 目的: LLM の構造化出力を強制し、実行時に厳密バリデーションする
- 主な利用箇所: `app/schemas.py`
- 役割:
  - `FacilitatorDecision`
  - `ContinuationDecision`
  - `DebaterResponse`
  - `ValidatorFeedback`
  - `SearchResult`
  - `DiscussionTurn`

### Ollama + httpx

- 目的: ローカルモデルへの推論リクエストを HTTP 経由で実行する
- 主な利用箇所: `app/llm/ollama_client.py`, `app/llm/model_manager.py`
- 役割:
  - `/api/generate` で structured / text generation
  - `/api/ps` でロード状態確認
  - keep_alive / preload / warmup / unload の制御
  - qwen 系モデルの structured 崩れ対策として `think=false` を付与
  - `<think>` 除去、コードフェンス抽出、JSON抽出のフォールバックパース

### Langfuse Python SDK

- 目的: セッション単位でトレースし、ノード実行・生成・エラーを可観測化する
- 主な利用箇所: `app/services/langfuse_service.py`
- 役割:
  - 1セッション = 1 trace
  - ノード単位 = span
  - LLM 呼び出し = generation
  - 障害時 = error event
  - 接続失敗時は警告を出しつつ無効化して実行継続

### Validator Node（rnj-1:latest）

- 目的: Debater の直前主張を品質評価し、議論の自己修正を促進する
- 主な利用箇所: `app/nodes/validator.py`
- 役割:
  - debater_a / debater_b の直後に実行
  - `ValidatorFeedback` を state と markdown ログに記録
  - 議論停止は行わないアドバイザリ運用

### subprocess（標準ライブラリ）

- 目的: 外部検索 CLI を直接実行する
- 主な利用箇所: `app/services/search_service.py`
- 役割:
  - `SEARCH_COMMAND_TEMPLATE` からコマンド構築
  - timeout / returncode / stdout / stderr 管理
  - 次ターン向け digest 作成
  - CLI 未導入（returncode 127）時は例外を記録しつつ議論本体は継続

### Input Service（Markdown 読み込み）

- 目的: CLI 引数がない場合でも `in/` の Markdown から議論を開始できるようにする
- 主な利用箇所: `app/services/input_service.py`
- 役割:
  - `in/*.md` を全件読み込んで連結（ファイル名昇順）
  - 議題（topic）を自動決定
  - 入力ソース一覧を state に保持

### Output Snapshot（最終成果 Markdown）

- 目的: 実行結果をユーザー向けに `out/` に保存する
- 主な利用箇所: `app/utils/markdown_logger.py`, `app/nodes/finalize.py`
- 役割:
  - `logs/` のイベントログとは別に最終成果物を出力
  - topic / input sources / final summary を1ファイルに集約

### python-dotenv + pydantic-settings

- 目的: 実行設定を `.env` から読み込み、型付き設定として扱う
- 主な利用箇所: `app/config.py`
- 役割:
  - Ollama URL、モデル名、max_turns、検索CLI、Langfuse設定を集約
  - 設定の妥当性チェック（例: `{query}` プレースホルダ必須）

### pytest

- 目的: 主要機能の回帰防止
- 主な利用箇所: `tests/`
- 役割:
  - schema validation
  - routing
  - retry 挙動
  - summarizer の圧縮ロジック

## 設計上の意図

- LangGraph は「遷移制御」に限定
- LLM 呼び出し、検索、観測、ログを別レイヤーへ分離
- 将来 DSPy を入れるときは `app/llm/interfaces.py` で差し替え可能な形を維持
