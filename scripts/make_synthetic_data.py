"""Create a tiny fake HAM10000-like dataset so you can try the whole pipeline without the real images.

    python scripts/make_synthetic_data.py --out data/synthetic
    python scripts/run_all.py --data-dir data/synthetic --fast
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ham10000.synthetic import make_synthetic_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic"); ap.add_argument("--lesions", type=int, default=400)
    a = ap.parse_args(); print("written to", make_synthetic_dataset(a.out, a.lesions))
