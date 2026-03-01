# ARXIV RECOMMENDER

毎日定時に事前に設定した興味ある分野に合致する新着プレプリントを [arXiv](https://arxiv.org) から取得して, 要約・日本語訳したものを [Discord](https://discord.com) に投稿する webhook です.

Google の [gemini api](https://ai.google.dev/gemini-api/docs?hl=ja) を使用します.

## 使用例

![discord にプレプリント情報が投稿される様子](https://raw.githubusercontent.com/Totti95U/discord-arxiv-bot/refs/heads/main/img/sample1.png)

## 使い方

Gemini api および Discord の Webhook URL を使用する都合上, このリポジトリをフォーク, もしくはお使いのコンピュータ上で動かす必要があります.

ここではリポジトリをフォークする場合で説明します. ローカルなどで動かす場合は適宜読み替えてください.

ローカルで動かす場合は **python 3.10.5** のインストールおよび `requirements.txt` にあるライブラリが必要です.

### 手順

1. [Google AI Studio](https://aistudio.google.com) に行き, Gemini API の API key を取得してください
    - 無料枠でも使用可能です
2. arXiv の情報を流すための Discord のチャンネルを作り, サーバー設定の連携アプリから webhook URL を取得してください
3. このリポジトリをフォークしてください. GItHub Actions を使用するため, GitHub アカウントが Free プランの場合はフォークしたリポジトリは Public にしてください
4. フォークしたリポジトリの secret として gemini api key と discord webhook url を登録してください
    - Gemini api key は `GEMINI_API_KEY` として登録してください
    - Discord webhook url は `ARXIV_RECOMMENDER_WEBHOOK_URL` として登録してください
5. `src/prompt_check_interest.txt` および `src/prompt_summarize.txt` にある `## 興味ある分野` の部分に自身が興味ある分野名やキーワードを入れてください
    - **ヒント:** これら2つのファイルは Gemini が読み取ります.
6. `src/main.py` の27行目にある `(cat:math.DS OR cat:math.CO OR cat:math.GR OR cat:cs.LO OR cat:cs.FL OR cat:cs.DM)` の部分を自身が興味のある arXiv のカテゴリに変えてください
    - 単に削除することで全てのカテゴリからプレプリントを取得するようになりますが, Gemini api のリクエスト回数が大幅に増加する可能性があります
7. (Gemini api 無料枠の場合) Gemini api の無料枠を使用する場合は, 対象カテゴリや検索対象日数を減らして1日の処理件数を抑えてください
    - `src/main.py` の `search_papers()` 内の `max_results` と日付レンジ (`days`) を調整してください

通常では次の 3 つの workflow が動作します.

- `.github/workflows/arxiv-summarizer.yml`
  - 毎日1回, arXiv 検索と興味判定 batch の submit を実行
- `.github/workflows/arxiv-poll-interest-submit-summary.yml`
  - 30分ごとに, 興味判定 batch を poll して完了分の要約 batch を submit
- `.github/workflows/arxiv-poll-summary-send.yml`
  - 30分ごとに, 要約 batch を poll して完了分を Discord に送信

この構成により, Gemini Batch API の完了待ちが長引いても単一ジョブがタイムアウトしにくくなります.

さらに, batch の応答が長時間返らない場合はフォールバック処理が自動で動きます.

- 既定では 48 時間以上 batch が未完了の場合, `batches.cancel` を試行
- cancel の成否にかかわらず `client.models.generate_content` に切り替えて逐次処理
- 興味判定・要約の両方でフォールバック対応

タイムアウト閾値は環境変数 `BATCH_TIMEOUT_HOURS` で変更できます（既定値: `48`）.

### state 管理ブランチについて

`pending_jobs.json` は `bot/manage-pending-jobs` ブランチ上で管理します.

- 保存場所: `state/pending_jobs.json`
- 形式: `{ "schema_version": 1, "jobs": [...] }`
- 各 workflow は実行前に state を読み込み, 実行後に更新内容を同ブランチへ push します

初回実行時にブランチが存在しない場合でも workflow が自動作成します.

### `pending_jobs.json` の status 一覧

`jobs` 配列の各要素は次の status を取ります.

- `interest_submitted`
  - 興味判定 batch を submit 済み, poll 待ち
- `interest_running`
  - 興味判定 batch が未完了
- `interest_fallback_running`
  - 興味判定 batch がタイムアウトし, `generate_content` 逐次処理へ切替中
- `summarize_submitted`
  - 要約 batch を submit 済み, poll 待ち
- `summarize_running`
  - 要約 batch が未完了
- `summary_fallback_running`
  - 要約 batch がタイムアウトし, `generate_content` 逐次処理へ切替中
- `send_failed`
  - Discord 送信に失敗（次回 poll で再送を試行）
- `completed_no_interests`
  - 興味あり論文が 0 件で正常終了
- `completed`
  - 全送信完了で正常終了
- `failed`
  - batch の失敗/期限切れ/キャンセルなどで処理停止

状態確認時は `status` に加えて `last_error`, `retry_count`, `updated_at`, `finalized_at` を見ると原因追跡しやすいです.

### 異常時の復旧手順

1. `bot/manage-pending-jobs` の `state/pending_jobs.json` を開き, 対象 job の `status` と `last_error` を確認
2. secret/外部設定を確認
   - `GEMINI_API_KEY`
   - `ARXIV_RECOMMENDER_WEBHOOK_URL`
   - 必要なら `BATCH_TIMEOUT_HOURS`
3. 該当 stage の workflow を `workflow_dispatch` で再実行
   - 興味判定側: `arxiv-poll-interest-submit-summary.yml`
   - 要約/送信側: `arxiv-poll-summary-send.yml`
4. Discord 送信失敗時 (`send_failed`) は, Webhook を修正後に再実行
5. 長期間停滞した job を手動で終了する場合は `status` を `failed` に変更して保存

重複送信は `sent_paper_ids` で抑制されるため, 再実行しても原則として未送信分のみ送られます.

cron を変更したい場合は上記 3 つの workflow の `schedule` を調整してください.

興味がありそうなプレプリントがなかった場合, その旨の通知は来ません.
