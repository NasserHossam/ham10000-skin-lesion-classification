"""Run the whole pipeline in order. Example:  python scripts/run_all.py --data-dir "..\\Skin Cancer"

Use --fast for a quick test run, --deep to include the PyTorch part, --bonus to include the bonus experiments."""
import argparse, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_prepare_data.py", "02_eda.py", "03_deep_learning.py", "04_dimensionality_reduction.py",
         "05_feature_selection.py", "06_final_models.py", "07_bonus_experiments.py"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir"); ap.add_argument("--work-dir"); ap.add_argument("--results-dir"); ap.add_argument("--figures-dir")
    ap.add_argument("--fast", action="store_true"); ap.add_argument("--deep", action="store_true"); ap.add_argument("--bonus", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these steps, e.g. --only 05 06")
    a = ap.parse_args()
    common = [x for k in ("data_dir", "work_dir", "results_dir", "figures_dir") if getattr(a, k) for x in (f"--{k.replace('_', '-')}", getattr(a, k))] + (["--fast"] if a.fast else [])
    for step in STEPS:
        num = step[:2]
        if a.only and num not in a.only: continue
        if not a.only and ((num == "03" and not a.deep) or (num == "07" and not a.bonus)): continue
        print(f"\n===== {step} =====", flush=True)
        subprocess.run([sys.executable, str(HERE / step), *common], check=True)


if __name__ == "__main__":
    main()
