import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ham10000.config import add_common_args, config_from_args, CLASSES
import pandas as pd
from ham10000.data import load_processed
from ham10000.plots import eda_figures


def main(argv=None):
    """Exploratory data analysis: missing values, class balance, images per lesion, and the EDA figures."""
    args = add_common_args(argparse.ArgumentParser(description=main.__doc__)).parse_args(argv); cfg = config_from_args(args)
    raw = pd.read_csv(cfg.data_dir / "HAM10000_metadata.csv"); meta, imgs = load_processed(cfg)
    print("Missing values:\n", raw.isna().sum().to_string()); print("\nClass counts:\n", raw["dx"].value_counts().to_string())
    per = raw.groupby("lesion_id").size()
    print("\nImages per lesion:\n", per.value_counts().sort_index().to_string())
    print(f"Lesions with more than one image: {(per > 1).sum()} of {len(per)}")
    eda_figures(raw, imgs, meta, CLASSES, cfg.figures_dir)


if __name__ == "__main__":
    main()
