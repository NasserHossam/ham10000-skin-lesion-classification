"""End-to-end smoke test on a tiny synthetic dataset (about 2 minutes). Run with:  pytest -m slow"""
import subprocess, sys
from pathlib import Path
import pytest
from ham10000.synthetic import make_synthetic_dataset

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.slow
def test_prepare_select_and_evaluate(tmp_path):
    data = make_synthetic_dataset(tmp_path / "raw", n_lesions=300)
    common = ["--data-dir", str(data), "--work-dir", str(tmp_path / "work"), "--results-dir", str(tmp_path / "results"),
              "--figures-dir", str(tmp_path / "figures"), "--fast"]
    for script in ["01_prepare_data.py", "05_feature_selection.py", "06_final_models.py"]:
        r = subprocess.run([sys.executable, str(SCRIPTS / script), *common], capture_output=True, text=True)
        assert r.returncode == 0, f"{script} failed:\n{r.stderr[-2000:]}"
    res = tmp_path / "results"
    for f in ["res_selection.csv", "selected_features.json", "res_final_models.csv"]:
        assert (res / f).exists(), f
    assert (tmp_path / "figures" / "final_eval.png").exists()
