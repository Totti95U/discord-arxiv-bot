# Decision 0001: プロバイダー非依存の単一 poll 設計

Status: Accepted (2026-11-19)

## 背景

2026-11 時点で GitHub Actions 4 workflow すべてが失敗していた。直接原因は `actions/setup-python` の
`python-version: '3.10.5'` ピン留め (現行 ubuntu ランナーに 3.10.5 のビルドが存在しない)。
副次的に、失敗した run の "Save state file" step (`if: always()`) が空の state を push してしまい、
2026-11-19 に state ブランチの実データが初期化された (コミット `ddc2410e8f`。旧 state は
`165afb081c` 以前から復元可能)。

## 決定

1. **LLM を OpenAI 互換 API に限定** (`openai` SDK, base_url 上書き)。
   役割 (interest / summary / reading / qa) ごとに `*_API_BASE` / `*_API_KEY` / `*_MODEL` を指定でき、
   未指定なら `LLM_API_BASE` / `LLM_API_KEY` と後続役割のモデルにフォールバックする。
   Gemini batch API (専用 SDK, ファイル upload, batch state machine) を廃止した理由:
   - batch を使わないと複数プロバイダーで同じ費用/遅延特性が得られない
   - batch 完了待ちの poll×2 本・フォールバック状態機械がコードの大部分を占めていた
2. **4 workflow を 5 分間隔の単一 poll 1 本に統合**。全 phase が state を都度保存する冪等設計なので、
   cron の再実行がそのままリカバリーになる。
3. **興味判定は decision 系モデルを指定可能** (`INTEREST_MODEL` + `INTEREST_API_BASE`)。
4. **全文リーディングと Q&A は pypdf によるテキスト抽出**でプロバイダー非依存にした。
   数式が崩れるのは既知の制限で、プロンプト側で「崩れの明示」を指示している。
5. **Q&A は gateway ではなく REST polling**: `GET /guilds/{id}/threads/active` でフォーラムの
   active スレッドを列挙し、`after=<last_read>` の差分質問にのみ回答する。
   遅延 5〜15 分を許容できない場合は gateway bot への移行が必要。

## 既知の天井 (ponytail 債務)

- active スレッド列挙 API は guild 全体で最大 100 件を返しさらにページングがない
  (`src/main.py` の `list_active_forum_threads` に注記)。フォーラムの未アーカイブ投稿が
  100 を超える運用では取りこぼしうる。
- 論文テキストは質問ごとに arXiv から再ダウンロードする。arXiv 側の負荷が問題になったら
  state へのキャッシュを足す。
- `/5` cron は GitHub Actions のスケジュール遅延 (数分) の影響を受ける。

## 反映先

- `src/main.py`, `.github/workflows/arxiv-bot.yml`, `README.md`
- 検証: `test/test_main.py` (29 unit tests) + fake LLM/Discord サーバによる E2E
  (`.scratch/e2e/run_e2e.py`, 実 arXiv 検索込み)
