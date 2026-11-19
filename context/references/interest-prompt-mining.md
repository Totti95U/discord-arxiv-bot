# 興味判定プロンプトの再構成に使ったデータ分析 (2026-11-19)

Status: Completed (メンテナンス系タスクなので継続運用はしない)

## 出典

- `bot/manage-pending-jobs` ブランチのコミット `165afb081c` (2026-10-21) の `state/pending_jobs.json`。
  2026-07-16 〜 2026-10-21 の 88 job 分で、興味判定結果 4403 件 (interested 592 / rejected 3811)。
  当時の判定は旧プロンプト (分野リスト 8 項目) + gemini 3.5-flash-lite batch によるもの。
- 直近の state は failure 時の "Save state" step により空になっているため、上記コミットから復元した。

## 方法

1. interested 592 件を 4 シャード (各 148 件) に分け、それぞれ別エージェントがタイトル・アブストラクト・
   キーワード・appendix note を精読してテーマ集計・方法論傾向・リスト外要素を報告した。
2. rejected 3811 件から興味語彙 (symbolic dynamics/automata/self-similar/tiling/infinite group 等) を
   含むニアミス 170 件を抽出し、別エージェントが除外パターンを帰納した。
3. 報告 `.scratch/prompts/report_interested_{1..4}.md`, `report_rejected.md` (生データは
   `.scratch/prompts/` 以下で git 管理外。Git 履歴から消えた場合は state ブランチの旧コミットから
   同じ抽出方法で再現できる)。

## 主な発見 (プロンプトに反映済み)

- 5 本柱: 無限群 (とくに具体族: Thompson/braid/Coxeter/Grigorchuk/sofic)、オートマトン・形式言語、
  複素・算術力学系、記号力学系 (thermodynamic formalism など)、語の組合せ論。
- 横断する嗜好: 具名予想の決着、完全分類・iff、計算量/決定可能性の確定、sharp・exact な値。
  理論対実験は約 130:15。
- リスト外で反復していた要素 (プロンプトに新規追加): ML×形式言語 (transformer 表現力、
  grammar-constrained decoding、時相論理エージェント)、検証・合成 (bisimulation、parity/Streett games)、
  2 次元語・CA・タイリングに由来する離散構造、Büchi/Presburger 等の算術的決定問題、
  計算機支援証明 (Walnut/Lean/CAS-verified)、群の作用素環的不変量 (個別群に限る)、数え上げ組合せの一部。
- **「validated numerics」は interested 592 件中実質 2 件しか中核になっていなかった。**
  実態は「数学の定理の証明への計算・検証の組み込み」なので、プロンプトではこの広義の形に置き換えた。
  区間演算そのものへの関心が残るなら「興味ある分野の核」への再追加で戻せる。
- rejected ニアミスの大半は多義語の誤発火: 測度論的エルゴード理論、滑らか/斉次力学系、C*-crossed product、
  フラクタル次元論・Delone、並進タイリング本体、汎用論理・SMT、有限群論・確率 GGT、グラフ本体、
  PDE のスケール不変性、量子文脈、工学応用の一般 dynamics 語彙。
  これらは除外パターンとして明文化した (例外条項つき)。

## プロンプトへの反映

- `src/prompt_check_interest.txt`: 核 8 分野、正シグナル、除外パターン 10 条、例外条項、判定手順。
  理由 (reason) を state に記録するため、今後のチューニングは `interest_reasons` を見て行える。
- `src/prompt_summarize.txt`, `src/prompt_reading_memo.txt`: 末尾の興味ある分野リストを同一の内容に更新。
