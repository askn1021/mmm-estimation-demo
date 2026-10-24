"""Check saved metrics, notebook outputs and local README links."""

import json
import re
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmm_demo.analysis import Settings, evaluate, generate_data


def main() -> None:
    settings = Settings(**json.loads((ROOT / "results/settings.json").read_text(encoding="utf-8")))
    data = pd.read_csv(ROOT / "results/synthetic_data.csv")
    predictions = pd.read_csv(ROOT / "results/predictions.csv")
    metrics = json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))
    pd.testing.assert_frame_equal(data, generate_data(settings), check_exact=False, rtol=1e-12, atol=1e-12)
    actual = evaluate(data, predictions, settings)
    for name, value in actual.items():
        assert np.isclose(value, metrics[name], rtol=1e-10, atol=1e-10), name
    assert predictions.week.equals(data.week)
    assert predictions.split.equals(data.split)
    np.testing.assert_allclose(predictions.outcome, data.outcome)
    assert (predictions.predictive_low < predictions.predictive_high).all()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", readme):
        if not target.startswith("https://"):
            assert (ROOT / target).exists(), target
    assert f"{metrics['test_rmse']:.3f}" in readme
    assert f"{metrics['test_predictive_90_coverage']:.1%}" in readme
    notebook = nbformat.read(ROOT / "notebooks/estimation.ipynb", as_version=4)
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in cells] == list(range(1, len(cells) + 1))
    images = []
    tables = []
    for cell in cells:
        for output in cell.outputs:
            assert output.output_type != "error"
            if "image/png" in output.get("data", {}):
                images.append(output.data["image/png"])
            if "text/html" in output.get("data", {}):
                tables.append(output.data["text/html"])
    assert len(images) == 3
    assert len(tables) == 3
    expected = ["01_data.png", "02_parameters.png", "03_prediction.png"]
    import base64

    for image, name in zip(images, expected):
        assert base64.b64decode(image) == (ROOT / "reports/figures" / name).read_bytes()
    print(f"Verified saved metrics, README links, {len(cells)} executed cells and 3 figures.")


if __name__ == "__main__":
    main()
