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

### state 管理ブランチについて

`pending_jobs.json` は `bot/manage-pending-jobs` ブランチ上で管理します.

- 保存場所: `state/pending_jobs.json`
- 形式: `{ "schema_version": 1, "jobs": [...] }`
- 各 workflow は実行前に state を読み込み, 実行後に更新内容を同ブランチへ push します

初回実行時にブランチが存在しない場合でも workflow が自動作成します.

cron を変更したい場合は上記 3 つの workflow の `schedule` を調整してください.

興味がありそうなプレプリントがなかった場合, その旨の通知は来ません.