# 興味判定プロンプトの再構成に使ったデータ分析 (2026-11-19/20)

Status: Completed (メンテナンス系タスクなので継続運用はしない)

## 出典

- `bot/manage-pending-jobs` ブランチのコミット `165afb081c` (2026-10-21) の `state/pending_jobs.json`。
  2026-07-16 〜 2026-10-21 の 88 job 分。興味判定は旧プロンプト (分野リスト 8 項目) + gemini 3.5-flash-lite batch によるもの。
- 直近の state は旧ワークフローの失敗時に空 push されていたため、上記コミットから復元した。

## 方法

1. **第一段階**: 興味判定結果 4403 件 (interested 592 / rejected 3811) を、interested 4 シャード +
   rejected ニアミス 170 件 (興味語彙を含むもの) に分け、計 5 エージェントでテーマ・除外パターンを集計。
2. **第二段階 (最重要)**: ユーザーの指摘により、真の興味の基準を「model が interested と判定したもの」
   ではなく「ユーザーが実際に 📖 を付けて Forum 詳細解説を頼んだ論文」に変更した。該当 26 件を
   `reading_memos` から抽出し全件精読して核を再定義した。
   - 抽出先: 各 job の `reading_memos` (📖 による Forum post 作成済みのもの)
3. 報告と中間データは `.scratch/prompts/` (git 管理外)。state ブランチの旧コミットから再現できる。

## 第一段階の知見 (参考)

- interested 判定の多義語誤発火パターン (測度論的エルゴード、滑らか/斉次力学系、C*-代数、
  フラクタル次元、汎用論理、有限群計算、グラフ本体、PDE の self-similar、量子、工学 dynamics)。
  これらは除外パターンとして意味が変わらないため現在も採用。

## 第二段階の知見 (26 件の実深掘りの分布)

- 語の組合せ論 8 件 (反復回避、complexity、string attractor、Lyndon、語アルゴリズム)
- 複素・1 次元実力学系 7 件 (Julia/Fatou、Hénon、単峰写像・kneading、分岐、可積分との境界、
  emergence・エントロピーなどの力学系不変量) ← ユーザーが「かなり注目」と明言
- 記号力学系 6 件 (SFT・強シフト同値、動的ゼータ、シフトの自己同型、回転の符号化)
- 算術力学系・数論×力学系 3 件 (arboreal Galois、preperiodic、p 進証明、素数の動的表現)
- 計算機援用証明 5 件 (Walnut、区間演算による分岐検証、厳密全数探索、Lean による形式化)
  → **validated numerics は第一段階で「実質 2 件しか発火していない」と扱ったが、
    実深掘りでは中心的手法として現れたため、核に明示的に戻した**
- aperiodic order / CA / 離散パターン 3 件 (monotile、mod ラプラシアン、SCA と braid)
- 語・言語の代数 2 件 (書換え系サーベイ、半環の有限基底問題)
- 列挙組合せ 2 件 (駐車関数、Permutation Wordle)、無限群 1 件 (Thompson F の歪み — 語の距離が主題)
- **27 件中 ML×formal languages は 0 件**。第一段階で「リスト外の反復要素」として追加したが、
  実深掘りデータでは支持されず、ユーザーも「あまり興味がない」と明言したため除外に回した。
- 無限群の具体族 (Thompson, braid, Coxeter, Grigorchuk, sofic) の取り上げも
  ユーザーの明言により「控えめ」に格下げした。

## プロンプトへの反映

- `src/prompt_check_interest.txt`: 核 8 分野 (上記の順) + 補足 (無限群は条件付き) + 正シグナル
  (予想決着・完全分類・計算機援用・サーベイ) + 除外パターン 10 条 + 例外条項 + 判定手順。
- `src/prompt_summarize.txt`, `src/prompt_reading_memo.txt`: 末尾の興味ある分野リストを同一内容に更新。
- 今後のチューニングは state の `interest_reasons` (判定理由) と 📖 反応の有無を突き合わせて行う。
