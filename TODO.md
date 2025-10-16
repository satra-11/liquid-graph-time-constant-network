# TODO
* [ ] `train_driving.py`のリファクタリング
    * [ ] `main`関数の責務を分離する。プロットや結果保存のロジックを別のヘルパー関数に切り出し、`main`は全体のワークフロー管理に集中させる。
    * [ ] モデルごとの冗長な処理を共通化する。モデルをリストや辞書で管理し、ループ処理に書き換えることで、将来的なモデル追加の拡張性を向上させる。
    * [ ] `evaluate_networks`内のハードコーディングを解消する。`corruption_levels`などの設定値をコマンドライン引数から渡せるようにし、再利用性を高める。
    * [ ] 進捗表示を改善する。`print`文によるロギングを`tqdm`ライブラリなどに置き換え、よりクリーンで分かりやすい進捗表示を実現する。
* [ ] 学習が進まない
    * [x] 損失計算の誤り。`train_driving.py`の損失計算で、`predictions`と`targets`の形状が不一致
    * [x] 勾配クリッピングの欠如。
    * [x] targetではなくsensorに変更(検証中)
    * [ ] ランダムウォーク行列、ラプラシアン行列、などを選べるように
    * [ ] outputからGPS情報を外す
    * [ ] flocking も入れてみる
    * [ ] Loss関数にペナルティ項を入れる
    * [ ] 層を増やしてみる
    * [ ] 学習率スケジューラの欠如。学習の停滞を避けるため、`torch.optim.lr_scheduler`（例：`StepLR`や`ReduceLROnPlateau`）を訓練ループに追加する。
    * [ ] データ正規化の不足。`DrivingDataset`で、`ToTensor`による[0, 1]スケーリングに加え、ImageNet等の平均と標準偏差を用いた正規化を追加する。
    * [ ] LTCNモデルの入力処理。`LTCNController`で空間情報を集約する`mean(dim=1)`が情報ボトルネックになっている可能性があるため、より多くの情報を保持する別の集約方法を検討する。
* [ ] グラフを時系列で求める
* [ ] グラフ処理をClosed Formに変更


---
* [x] フロー図を生成する
* [x] Vmasタスクから画像欠損識別タスクに変える
* [x] 新しい自律走行タスクでの実験実行
* [x] ディレクトリツリーを追加
* [x] 評価指標の妥当性を検討
* [x] リアルなドライブ映像を使用する
* [x] 画像へのグラフの適用が適切か検討
* [x] 最適なバッチサイズを求める
* [x] 自動運転タスクであることをREADMEに記述
* [x] src/tasks/autonomous_driving.pyの確認
* [x] src/utils/graph.pyの確認

### 論文読み
* [x] Section I (INTRODUCTION) を読む — 新しい概念をObsidianにメモ
* [x] Section II (PRELIMINARIES) を読む — 新しい概念をObsidianにメモ
* [x] Section III (LIQUID-GRAPH TIME CONSTANT NETWORK) を読む — 新しい概念をObsidianにメモ
* [x] Section IV (CLOSED-FORM APPROXIMATION) を読む — 新しい概念をObsidianにメモ
* [x] Section V (VALIDATION EXAMPLE) を読む — 新しい概念をObsidianにメモ
* [x] Section VI (CONCLUSION) を読む — 新しい概念をObsidianにメモ

### Obsidian (研究ノート)
* [x] 調査・要約: ISS (入力対状態安定性)
* [x] 調査・要約: incremental ISS (δISS)
* [x] 調査・要約: GNN