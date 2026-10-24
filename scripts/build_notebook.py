"""Create and execute the walkthrough notebook in the current Python process."""

import base64
import contextlib
import io
import os
import platform
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def build() -> None:
    notebook = nbformat.v4.new_notebook()
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    notebook.cells = [
        markdown("# 人工データによる広告効果のベイズ推定\n\n"
                 "2媒体の基本的なMMMを使い，推定と未使用期間での予測を確認する．"
                 "全セルは単一の設定で実行した結果を含む．"),
        code("from pathlib import Path\nimport sys\nimport pandas as pd\n"
             "from IPython.display import Image\n\n"
             "ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\n"
             "sys.path.insert(0, str(ROOT / 'src'))\n"
             "from mmm_demo.analysis import (\n"
             "    Settings, generate_data, fit, draw_parameters, predict, parameter_summary, evaluate,\n)\n"
             "from mmm_demo.plots import save_figures\nsettings = Settings()\nprint(settings)"),
        markdown("## 1．データと分割\n\n156週のうち先頭124週で推定し，残り32週で評価する．"
                 "true_meanは評価用の真値であり，推定には渡さない．"),
        code("data = generate_data(settings)\ndisplay(data.head())\n"
             "print(data.groupby('split', sort=False).size().to_string())"),
        markdown("## 2．事後分布の計算\n\nalphaは一様事前分布の格子近似，切片とbetaは"
                 "独立なNormal(0, 50²)とする．半飽和点とノイズ標準偏差は既知と仮定する．"
                 "MCMCは使わず，係数の条件付き事後分布を解析的に計算する．"),
        code("spend = data[['spend_a', 'spend_b']].to_numpy()\nn_train = settings.train_weeks\n"
             "posterior = fit(spend[:n_train], data.outcome.to_numpy()[:n_train], settings)\n"
             "draws = draw_parameters(posterior, settings)\n"
             "parameters = parameter_summary(draws, settings)\ndisplay(parameters.round(3))\n"
             "print('Posterior mass:', round(float(posterior['weights'].sum()), 8))"),
        markdown("## 3．検証期間の予測\n\n検証期間の成果は推定に使わない．支出は既知として扱う．"
                 "予測には，学習期間から引き継いだ持続効果を含める．予測区間には観測ノイズも含める．"),
        code("predictions = predict(spend, posterior, draws, settings)\n"
             "metrics = evaluate(data, predictions, settings)\n"
             "display(pd.Series(metrics, name='value').to_frame().round(3))"),
        code("save_figures(data, predictions, parameters, ROOT / 'reports' / 'figures')\n"
             "display(Image(filename=str(ROOT / 'reports' / 'figures' / '01_data.png')))\n"
             "display(Image(filename=str(ROOT / 'reports' / 'figures' / '02_parameters.png')))\n"
             "display(Image(filename=str(ROOT / 'reports' / 'figures' / '03_prediction.png')))"),
        markdown("## 解釈の範囲\n\nこの分析は，生成モデルと推定モデルの形を揃えた人工データ1条件の例である．"
                 "真値と推定幅の比較，予測と残差の確認を示すものであり，実市場の因果効果を検証してはいない．"
                 "半飽和点とノイズ標準偏差は既知，係数は負値も許す．季節性・交絡・モデル誤指定は扱わない．"
                 "32週の被覆割合だけでは，予測区間の校正を十分に評価できない．"
                 "モデルの定義と背景文献はREADMEに記載した．"),
    ]
    namespace = {"__name__": "__main__"}
    active_outputs = []

    def display(value):
        if isinstance(value, pd.DataFrame):
            content = {"text/plain": value.to_string(), "text/html": value.to_html()}
        elif isinstance(value, Image):
            content = {"image/png": base64.b64encode(value.data).decode("ascii")}
        else:
            content = {"text/plain": str(value)}
        active_outputs.append(nbformat.v4.new_output("display_data", data=content))

    from IPython.display import Image
    import pandas as pd

    namespace["display"] = display
    old_directory = Path.cwd()
    try:
        os.chdir(ROOT)
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            active_outputs = []
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exec(compile(cell.source, f"notebook_cell_{count}", "exec"), namespace)
            if stream.getvalue():
                active_outputs.append(nbformat.v4.new_output("stream", name="stdout", text=stream.getvalue()))
            cell.outputs = active_outputs
            cell.execution_count = count
    finally:
        os.chdir(old_directory)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": platform.python_version()},
    }
    target = ROOT / "notebooks" / "estimation.ipynb"
    target.parent.mkdir(exist_ok=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, target)
    print(f"Executed {count} cells and saved {target.name}")


if __name__ == "__main__":
    build()
