"""Reproduce the synthetic data, estimates, figures and README."""

import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
import numpy as np
import pandas as pd
import scipy

from mmm_demo.analysis import (
    Settings, draw_parameters, evaluate, fit, generate_data, parameter_summary, predict,
)
from mmm_demo.plots import save_figures


def write_readme(metrics: dict, parameters: pd.DataFrame) -> None:
    outside = int(((parameters.truth < parameters.credible_low) |
                   (parameters.truth > parameters.credible_high)).sum())
    text = f"""# 人工データによる広告効果のベイズ推定

2つの広告媒体の週次支出と成果を人工的に生成し，マーケティング・ミックス・モデル（MMM）で効果を推定する．既存文献に基づく基本モデルを題材に，データ生成，推定，不確実性の可視化，未使用期間での予測評価を一通り実装した．

## 分析の流れ

1. 支出，効果の持続，飽和を含む156週の人工データを生成する．
2. 先頭124週を使い，効果の持続と媒体別の反応係数，切片を推定する．
3. 真値と事後分布を比較し，パラメータの推定幅を確認する．
4. 残り32週の実際の支出を入力し，成果の予測誤差と予測区間を評価する．

成果の検証期間は推定に使わない．検証期間の支出は既知として扱うため，これは支出を所与とした条件付き予測であり，未知の将来支出まで予測するものではない．各週の計算は，その週までの支出だけを使う．

![人工データ](reports/figures/01_data.png)

破線より左を学習期間，右を検証期間とする．単位は任意であり，実際の広告費や売上額ではない．

## 結果

![パラメータの推定](reports/figures/02_parameters.png)

青は事後平均と90%信用区間，橙はデータ生成に使った真値を表す．係数は持続効果や切片と同時に推定するため，個々の値だけでなく推定幅も確認する．切片を含む5パラメータのうち，この実行では{outside}個の真値が90%信用区間の外にあった．1回の分析で真値を必ず回収できるわけではない．

| 検証期間の指標 | 値 |
|---|---:|
| MMMのRMSE | {metrics['test_rmse']:.3f} |
| MMMのMAE | {metrics['test_mae']:.3f} |
| 学習期間の成果平均を常に予測する場合のRMSE | {metrics['train_mean_baseline_rmse']:.3f} |
| 90%予測区間に入った観測の割合 | {metrics['test_predictive_90_coverage']:.1%} |
| 90%予測区間の平均幅 | {metrics['test_predictive_90_mean_width']:.3f} |

RMSEとMAEは小さいほど予測誤差が小さい．この例では，学習期間の平均値による予測よりMMMのRMSEが小さかった．ただし，この比較は1回のデータ生成に限られる．

90%予測区間に入った割合は{metrics['test_predictive_90_coverage']:.1%}であり，名目の90%を下回った．32週で計算した記述値であり，区間が適切に校正されているとは結論付けない．再実行で事後標本数を変えると，区間や被覆割合は変わり得る．

![検証期間の予測と残差](reports/figures/03_prediction.png)

予測区間には，パラメータの不確実性に加え，観測ノイズを含めた．パラメータの信用区間と，成果の予測区間は区別する．推定値・指標は [results](results/) に保存した．

## モデルと実装

```text
adstock[t,m] = spend[t,m] + alpha[m] * adstock[t-1,m]
response[t,m] = adstock[t,m] / (adstock[t,m] + k[m])
outcome[t] = intercept + sum_m beta[m] * response[t,m] + noise[t]
```

alphaは効果の持続，kは半飽和点，betaは飽和後の反応水準を決める．betaは支出1単位当たりの一定の効果ではない．初期adstockは0とする．

alphaには媒体ごとに独立な一様事前分布を置き，0から1の間の51点ずつの中点格子で近似する．切片とbetaには独立なNormal(0, 50²)事前分布を置く．alphaを固定すると線形Gaussianモデルになるため，係数を解析的に積分して格子の事後確率を求める．MCMCは使わない．

予測平均は格子上で積分する．信用区間と予測区間には，格子と条件付き係数分布から得た4,000個の同時事後標本を使う．データ生成の真値は評価にだけ使い，alpha・beta・切片の真値を推定処理には渡さない．ただし，以下のとおりkとノイズ標準偏差は既知と仮定する．

### 前提と限界

- 人工データ1条件・1回の分析であり，実市場での効果や，他の条件での性能を示すものではない．
- 推定モデルとデータ生成モデルの形を揃えているため，モデルの誤指定に対する強さは評価していない．
- 半飽和点kと観測ノイズ標準偏差は生成時の値に固定する．実務では通常これらも未知となる．
- 季節性，価格，競合，交絡，相互作用は含めない．観測データからの因果効果の識別を実証したものではない．
- betaに正値制約は課さない．負の係数も許すGaussianモデルであり，実務的な正値制約付きモデルとは異なる．
- 格子による近似と事後標本による誤差が残る．高次元の媒体数への拡張や数値収束の網羅的な検証は扱わない．

## 読む順序と構成

- [出力付きNotebook](notebooks/estimation.ipynb)：データ，推定，検証の手順
- [分析コード](src/mmm_demo/analysis.py)：生成・推定・予測・評価
- [可視化コード](src/mmm_demo/plots.py)：図の作成
- [実行スクリプト](scripts/run_analysis.py)：結果・図・READMEの再生成
- [Notebook生成スクリプト](scripts/build_notebook.py)：セルの実行と出力の保存
- [テスト](tests/test_analysis.py)：持続効果の計算，過去の値への将来支出の混入防止，事後確率など
- [出力確認スクリプト](scripts/validate_outputs.py)：保存値・図・Notebook・READMEの整合性

## 再現方法

Python 3.11以上を用いる．保存済み結果を作った環境は [実行記録](results/environment.json)，設定は [settings.json](results/settings.json) に記載した．

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python -m pytest -q
python scripts/run_analysis.py
python scripts/build_notebook.py
python scripts/validate_outputs.py
```

Notebook生成スクリプトはセルを順に同じPythonプロセスで実行し，標準出力・表・PNGを保存する．別のJupyterカーネルは起動しない．Notebookを対話的に編集する場合は，別途Jupyter環境が必要となる．保存済みNotebookの閲覧だけならPython環境は不要となる．

`requirements-lock.txt` は使用ライブラリのバージョンを固定した一覧であり，OSや全ての推移的依存を固定するものではない．再実行は保存済み結果・図・READMEを更新する．解釈の文章は，実装を変更した際に併せて確認する必要がある．

## 参考文献

- Jin, Y., Wang, Y., Sun, Y., Chan, D., and Koehler, J. (2017). [Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects](https://research.google/pubs/bayesian-methods-for-media-mix-modeling-with-carryover-and-shape-effects/).

持続効果と飽和を含むMMMの背景文献である．本分析は同論文の完全な再現実装ではない．
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    japan_time = timezone(timedelta(hours=9))
    started = datetime.now(japan_time).isoformat()
    settings = Settings()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    data = generate_data(settings)
    spend = data[["spend_a", "spend_b"]].to_numpy()
    n_train = settings.train_weeks
    posterior = fit(spend[:n_train], data.outcome.to_numpy()[:n_train], settings)
    draws = draw_parameters(posterior, settings)
    predictions = predict(spend, posterior, draws, settings)
    parameters = parameter_summary(draws, settings)
    metrics = evaluate(data, predictions, settings)
    metrics["negative_beta_fraction"] = float(np.mean((draws[:, 3:] < 0).any(axis=1)))
    data.to_csv(output / "synthetic_data.csv", index=False)
    parameters.to_csv(output / "parameters.csv", index=False)
    pd.concat([data[["week", "split", "outcome"]], predictions], axis=1).to_csv(
        output / "predictions.csv", index=False
    )
    for name, content in [("settings.json", asdict(settings)), ("metrics.json", metrics)]:
        (output / name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    save_figures(data, predictions, parameters, ROOT / "reports" / "figures")
    write_readme(metrics, parameters)
    environment = {
        "started_at": started,
        "finished_at": datetime.now(japan_time).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
