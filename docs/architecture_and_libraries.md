## 使用フレームワーク / ライブラリ

このリポジトリは、ローカル LLM の議論オーケストレーションを「状態遷移」「構造化データ」「観測」「外部検索」を分離して実装しています。

### LangGraph

- 目的: 議論ワークフローの状態遷移を明示的に管理する
- 主な利用箇所: `app/graph.py`
- 役割:
  - `START -> facilitator` から開始
  - `next_action` による条件分岐
  - `debater/search -> summarizer -> facilitator` ループ
  - `finish -> finalize -> END` で終了

### Pydantic v2

- 目的: LLM の構造化出力を強制し、実行時に厳密バリデーションする
- 主な利用箇所: `app/schemas.py`
- 役割:
  - `FacilitatorDecision`
  - `DebaterResponse`
  - `SearchResult`
  - `DiscussionTurn`

### Ollama + httpx

- 目的: ローカルモデルへの推論リクエストを HTTP 経由で実行する
- 主な利用箇所: `app/llm/ollama_client.py`, `app/llm/model_manager.py`
- 役割:
  - `/api/generate` で structured / text generation
  - `/api/ps` でロード状態確認
  - keep_alive / preload / warmup / unload の制御

### Langfuse Python SDK

- 目的: セッション単位でトレースし、ノード実行・生成・エラーを可観測化する
- 主な利用箇所: `app/services/langfuse_service.py`
- 役割:
  - 1セッション = 1 trace
  - ノード単位 = span
  - LLM 呼び出し = generation
  - 障害時 = error event

### subprocess（標準ライブラリ）

- 目的: 外部検索 CLI を直接実行する
- 主な利用箇所: `app/services/search_service.py`
- 役割:
  - `SEARCH_COMMAND_TEMPLATE` からコマンド構築
  - timeout / returncode / stdout / stderr 管理
  - 次ターン向け digest 作成

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
