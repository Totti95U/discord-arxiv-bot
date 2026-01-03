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
7. (Gemini api 無料枠の場合) Gemini api の無料枠を使用する場合, `src/main.py` に以下の変更を加えてください
    - 28行目の `None` を `20` にしてください (Gemini api 無料枠の一日の呼び出し回数が20回なことに対応)
    - 232行目を `interests = check_interest_sequential(search_results)` にしてください
    - 243行目を `summaries = summarize_paper_sequential(results)` にしてください

通常では毎日午前11時頃に新着プレプリントの情報を投稿します. 
変更したい場合は `.github/workflows/arxiv-summarizer.yml` を調整してみてください.

興味がありそうなプレプリントがなかった場合, その旨の通知は来ません.