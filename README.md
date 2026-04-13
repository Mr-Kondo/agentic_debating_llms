## Local LLM Debate App

Ollama 上の複数ローカル LLM を使って、特定テーマを自律的に議論する LangGraph アプリです。

- Facilitator: `llama3.1:latest`
- Debater A: `gemma4:latest`
- Debater B: `qwen3.5:latest`
- Validator: `rnj-1:latest`
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
- `VALIDATOR_MODEL`: Debater 発言の品質検証モデル
- `MODEL_KEEP_ALIVE`: Ollama keep_alive
- `MAX_TURNS`: 最大ターン数
- `CONTINUATION_ROUNDS`: 結論後の継続議論ラウンド数（0 = 無効、既定値）
- `SEARCH_COMMAND_TEMPLATE`: 検索 CLI テンプレート（`{query}` 必須）
- `MARKDOWN_LOG_DIR`: ログ保存ディレクトリ
- `INPUT_DIR`: `--topic` 未指定時に読む Markdown 入力ディレクトリ
- `OUTPUT_DIR`: 議論結果の最終 Markdown 出力ディレクトリ
- `LANGFUSE_ENABLED`: Langfuse 利用フラグ
- `LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY`: Langfuse 接続情報

## Ollama で必要なモデル

例:

```bash
ollama pull llama3.1:latest
ollama pull gemma4:latest
ollama pull qwen3.5:latest
ollama pull rnj-1:latest
```

モデルタグは `.env` の値が優先されます。実環境でタグ名が異なる場合は `.env` のモデル名を差し替えてください。

## 実行方法

```bash
uv run python -m app.main --topic "ローカルLLMは企業内ナレッジ活用をどこまで改善できるか"
```

`--topic` を省略した場合、既定で `in/*.md` を全件読み込み（ファイル名昇順で連結）して議題・背景コンテキストとして使用します。

オプション:

- `--max-turns 10`: 最大ターン上書き
- `--continuation-rounds 3`: 結論後の継続議論ラウンド数（0 でデフォルト無効）
- `--no-preload`: セッション開始時の preload/warmup 無効化

### in/ からの自動入力

`--topic` を指定しない場合は `INPUT_DIR`（既定 `./in`）配下の `.md` を読み込みます。

```bash
mkdir -p in
cat > in/topic.md << 'EOF'
# アークナイツの魅力はどこにあるか
キャラクター性とゲームデザインの観点から比較したい。
EOF

uv run python -m app.main
```

入力ファイルが1件もない場合は、`--topic` を指定するか `in/` に `.md` を配置してください。

### out/ への成果物出力

実行後、以下の2種類のMarkdownが生成されます。

- `logs/`: ノードごとの詳細イベントログ
- `out/`: ユーザー向け最終成果（topic / input sources / final summary）

## 起動トラブルシューティング

### 1. モデル未取得エラー

起動時にモデル不足で停止した場合は、表示されたモデルを pull してください。

```bash
ollama pull llama3.1:latest
ollama pull gemma4:latest
ollama pull qwen3.5:latest
ollama pull rnj-1:latest
```

### 1.1 qwen 系で structured output が崩れる場合

このアプリは `qwen3.5:latest` の structured output で thinking 由来の崩れを避けるため、Ollama リクエストで `think=false`（および `options.thinking=false`）を明示しています。

それでも失敗する場合は、まず Ollama 側のモデル更新と再 pull を行ってください。

```bash
ollama pull qwen3.5:latest
```

### 2. Ollama 404 / endpoint エラー

`OLLAMA_BASE_URL` を確認し、`ollama serve` が起動していることを確認してください。

```bash
ollama serve
```

```env
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. 初期診断の一時回避

preload をスキップしてアプリ本体の動作確認をする場合:

```bash
uv run python -m app.main --topic "..." --no-preload
```

## Langfuse を使う場合

`.env` で以下を設定します。

```env
LANGFUSE_ENABLED=true
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
```

1セッション1traceで記録され、各ノードは span、Ollama 呼び出しは generation として記録されます。

Langfuse 接続に失敗した場合、アプリは停止せずに Langfuse を無効化して継続します。失敗理由は標準エラーに `[langfuse] ...` として表示されます。

## 検索設定（API-first）

既定では `ddgs` Python API を使います。

```env
SEARCH_BACKEND=api
SEARCH_MAX_RESULTS=5
SEARCH_QUERY_OPTIMIZER=none
SEARCH_COMMAND_TEMPLATE=ddgs text -q "{query}" --max-results 5
```

- `SEARCH_BACKEND=api`: ddgs Python API を利用（推奨）
- `SEARCH_BACKEND=cli`: 外部 CLI を利用（互換モード）。この場合のみ `SEARCH_COMMAND_TEMPLATE` を使用
- `SEARCH_QUERY_OPTIMIZER=dspy`: DSPy が利用可能な場合に検索クエリを最適化（失敗時は元クエリで継続）

CLI モードを使う場合は `ddgs` で `-q`（または `--query`）が必須です。非0終了が続く場合はテンプレートの引数順と必須オプションを確認してください。

## テスト実行方法

```bash
uv run pytest
```

含まれる主なテスト:

- `tests/test_schemas.py`
- `tests/test_routing.py`
- `tests/test_retry.py`
- `tests/test_summarizer.py`
- `tests/test_ollama_client.py`
- `tests/test_validator.py`
- `tests/test_input_service.py`
- `tests/test_output_writer.py`

## 今後の拡張候補

- DSPy ベースの継続議論判断の自動最適化（`app/dspy_modules/continuation_decider.py`）
- Search digest の高度化（抽出要約、重複除去）
- 評価用メトリクス（議論収束度、主張多様性、継続ラウンドの新規性スコア）
- 非同期実行と並列検索

## 詳細ドキュメント

- [使用フレームワーク / ライブラリ](docs/architecture_and_libraries.md)
- [処理フロー + フロー図 (Mermaid)](docs/repository_flow.md)
- [サードパーティライセンス一覧](docs/third_party_licenses.md)

## ライセンス表示の再生成

`docs/third_party_licenses.md` は `uv.lock` と仮想環境内の配布メタデータから自動生成されます。
依存関係を更新したあとは以下のコマンドで再生成してください。

```bash
uv run python -m app.licenses
```
