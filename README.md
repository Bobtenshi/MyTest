# フォルダ構成

```
.
├── docker
│   ├── Dockerfile
│   ├── Makefile
│   ├── entrypoint.sh
│   └── requirements.txt
├── outputs
├── src
│   ├── Tool_
│   │   ├── cites
│   │   │   ├── lili19-3asj201-204.pdf
│   │   │   └── si_sdr.pdf
│   │   ├── FastMVAE2.py
│   │   ├── FastMVAE2_.py
│   │   ├── calc_cost.py
│   │   ├── calc_sdr_by_tmp_dnn.py
│   │   ├── ilrma.py
│   │   ├── initalizers.py
│   │   ├── linenotify.py
│   │   ├── main_ilrma.py
│   │   ├── metrics.py
│   │   ├── separation.py
│   │   ├── stft_or.py
│   │   ├── update_models.py
│   │   ├── util.py
│   │   └── utils.py
│   ├── Tools
│   │   ├── ILRMA_tools
│   │   │   ├── ilrma.py
│   │   │   ├── initalizers.py
│   │   │   ├── main_ilrma.py
│   │   │   ├── update_models.py
│   │   │   └── util.py
│   │   ├── FastMVAE2.py
│   │   ├── calc_cost.py
│   │   ├── calc_sdr_by_tmp_dnn.py
│   │   ├── data.py
│   │   ├── metrics.py
│   │   ├── stft_tools.py
│   │   └── utils.py
│   ├── constant.py
│   ├── evaluation.py
│   ├── hydras.yaml
│   ├── main.py
│   ├── net_dev.py
│   ├── retrain_encoder.py
│   ├── show_glaph.py
│   ├── simurate_pyroom.py
│   ├── source_separation.py
│   ├── trim_wav_wsj0.py
│   ├── trimming_wav.py
│   └── utils.py
├── README.md
└── pyproject.toml
```

# 仮想環境の準備

1. doker フォルダに移動
   ```
   cd docker
   ```
1. docker コンテナを起動
   ```
   make build
   ```
1. requirements.txtから必要なパッケージをインストール
   ```
   pip install -r requirements.txt
   ```

# DNN モデルの学習
## データセットの準備
1. datanet にあるjvs コーパスの音声データを`data/`以下に配置
## 学習用データの作成
-  `trimming_wav.py`で jvs0 の音声データを結合し，10 秒ごとに分割して保存する．
``` python3 src/trimming_wav.py```
- `simurate_pyroom.py`で残響のある混合信号をシミュレートし，保存する．
``` python3 src/simurate_pyroom.py```
## ネットワークの学習
- `retrain_encoder.py`によりシミュレート音声を用いて DNN モデルを学習
``` python3 src/retrain_encoder.py```

# main.py の仕様
main.py では，
- 音源分離（`source_separation.py`）
- SDR 値による性能評価（`evaluation.py`）
- 分離性能のグラフ表示（`show_glaph.py`）
  を実行できます．

## 各パラメータの説明（hydra.yaml）

音源分離=1，SDR評価=2，プロット=3の指定
```
hydra.const.stage: 1
hydra.const.stop_stage: 2
```

ILRMA,FastMVAE2を実行する場合の指定（デフォルトはNone）
```
params.modeling_method: ILRMA
params.modeling_method: ChimeraACVAE
```


## 実行コマンド例
- 従来法: FastMVAE2法における学習済みモデルで音源分離＆評価
```
python3 src/main.py const.stage=1 const.stopstage=2 params.modeling_method="ChimeraACVAE"
```

- 提案法: 提案法での学習済みモデルで音源分離＆評価
```
python3 src/main.py const.stage=1 const.stopstage=2 params.retrain_model_type="AE"  params.retrain_loss="ISdiv"
```

- 実験結果からグラフ作成
```
python3 src/main.py const.stage=3 const.stopstage=3
```


# todo
- ~~いらないコメント文を削除~~
  - ~~`hydras.yaml`~~
  - ~~`src/simurate_pyroom.py`~~
~~- hydra のパラメタを説明~~
~~- 事前に `pip install` が必要なパッケージの説明~~
~~- 実行コマンド~~
~~  - 音響学会~~
~~  - APSIPA~~
~~- 実験結果の可視化コマンド~~
~~- 学習データの生成コマンド~~
- `src/Tools`, `src/Tool_` の違い？いらないものがあるなら削除
