## Local LLM Debate App

Ollama 上の複数ローカル LLM を使って、特定テーマを自律的に議論する LangGraph アプリです。

- Facilitator: `llama3.1:8b`
- Debater A: `gemma4:8b`
- Debater B: `qwen3.5:8b`
- Structured output: Pydantic v2
- Trace/Observability: Langfuse
- Search: subprocess で外部 CLI 実行
- Session logging: Markdown 逐次追記

## アーキテクチャ概要

責務分離を前提に、以下の構成で実装しています。

- `app/graph.py`: LangGraph の状態遷移のみを定義
- `app/nodes/`: facilitator / debater / search / summarizer / finish / finalize ノード
- `app/llm/`: Ollama API クライアントとモデル管理
- `app/services/`: Langfuse、検索サービス、セッション初期化
- `app/utils/`: retry 戦略、Markdown logger、時刻ユーティリティ
- `app/schemas.py`: 構造化出力スキーマ
- `app/state.py`: LangGraph state

## セットアップ

1. Python 3.11+ と `uv` を用意
2. 依存関係を同期

```bash
uv sync --all-extras
```

3. 環境変数ファイル作成

```bash
cp .env.example .env
```

## .env.example の説明

- `OLLAMA_BASE_URL`: Ollama のベースURL
- `FACILITATOR_MODEL`: 司会モデル
- `DEBATER_A_MODEL`: A 側モデル
- `DEBATER_B_MODEL`: B 側モデル
- `MODEL_KEEP_ALIVE`: Ollama keep_alive
- `MAX_TURNS`: 最大ターン数
- `SEARCH_COMMAND_TEMPLATE`: 検索 CLI テンプレート（`{query}` 必須）
- `MARKDOWN_LOG_DIR`: ログ保存ディレクトリ
- `LANGFUSE_ENABLED`: Langfuse 利用フラグ
- `LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY`: Langfuse 接続情報

## Ollama で必要なモデル

例:

```bash
ollama pull llama3.1:8b
ollama pull gemma4:8b
ollama pull qwen3.5:8b
```

実環境でタグ名が異なる場合は `.env` のモデル名を差し替えてください。

## 実行方法

```bash
uv run python -m app.main --topic "ローカルLLMは企業内ナレッジ活用をどこまで改善できるか"
```

オプション:

- `--max-turns 10`: 最大ターン上書き
- `--no-preload`: セッション開始時の preload/warmup 無効化

## Langfuse を使う場合

`.env` で以下を設定します。

```env
LANGFUSE_ENABLED=true
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
```

1セッション1traceで記録され、各ノードは span、Ollama 呼び出しは generation として記録されます。

## 検索 CLI の設定

既定例は `ddgs` です。

```env
SEARCH_COMMAND_TEMPLATE=ddgs text "{query}" --max-results 5
```

他 CLI に変更する場合も `{query}` プレースホルダを必ず含めてください。

## テスト実行方法

```bash
uv run pytest
```

含まれる主なテスト:

- `tests/test_schemas.py`
- `tests/test_routing.py`
- `tests/test_retry.py`
- `tests/test_summarizer.py`

## 今後の拡張候補

- DSPy ベースの Facilitator/Summarizer 差し替え
- Search digest の高度化（抽出要約、重複除去）
- 評価用メトリクス（議論収束度、主張多様性）
- 非同期実行と並列検索

## 詳細ドキュメント

- [使用フレームワーク / ライブラリ](docs/architecture_and_libraries.md)
- [処理フロー + フロー図 (Mermaid)](docs/repository_flow.md)
