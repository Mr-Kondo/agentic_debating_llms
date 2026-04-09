## リポジトリの処理フロー

このアプリは CLI 起動から終了まで、以下の順に処理されます。

1. `app.main` が引数（`--topic`, `--max-turns`, `--no-preload`）を受け取る
2. `--topic` 未指定時は `in/*.md` を全件読込し、議題と背景コンテキストを組み立てる
3. `initialize_session()` が設定・各サービス・初期 state を構築
4. `build_graph()` で LangGraph を構築して実行
5. `facilitator_node()` が次アクションを決定（structured output）
6. アクションに応じて `debater_a` / `debater_b` / `search` へ分岐
7. debater 系のときは `validator_node()` が直後に主張品質を評価
8. `summarizer_node()` が context を圧縮して facilitator に戻す
9. `finish` 判定後、`finalize_node()` で最終要約を生成し、`out/` に成果Markdownを書き出す
10. `logs/`（詳細イベント）と `out/`（最終成果）を出力して終了

## Mermaid フロー図

```mermaid
flowchart TD
    A[CLI: uv run python -m app.main --topic ...] --> B[load_config]
    B --> B2{topic provided?}
    B2 -->|yes| C[initialize_session]
    B2 -->|no| BI[load in/*.md and build topic/context]
    BI --> C
    C --> D[create markdown session file]
    C --> E[start langfuse trace]
    C --> F[optional model preload/warmup]
    D --> G[build_graph]
    E --> G
    F --> G

    G --> H[facilitator_node]
    H -->|next_action=speak_a| I[debater_a_node]
    H -->|next_action=speak_b| J[debater_b_node]
    H -->|next_action=search| K[search_node]
    H -->|next_action=finish| L[finish_node]

    I --> V[validator_node]
    J --> V
    V --> M[summarizer_node]
    K --> M
    M --> H

    L --> N[finalize_node]
    N --> O[end_trace + logs final append]
    O --> O2[write result snapshot to out/]
    O2 --> P[END]
```

## 補足: 失敗時の扱い

- Facilitator / Debater は失敗種別ごとに retry 戦略を分離
- model not loaded は `ModelManager.ensure_loaded()` で回復を試行
- search CLI の timeout / 非0終了は分離して記録
- 回復不能時は `finish` 側に寄せる安全フォールバックを採用
