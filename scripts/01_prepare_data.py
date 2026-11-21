import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ham10000.config import add_common_args, config_from_args, CLASSES
from ham10000.data import prepare


def main(argv=None):
    """Read the metadata and images, split by lesion (70/15/15 and 80/20) and save the resized image array."""
    args = add_common_args(argparse.ArgumentParser(description=main.__doc__)).parse_args(argv); cfg = config_from_args(args)
    meta, imgs = prepare(cfg, cfg.seed)
    print("images:", imgs.shape, "| lesions:", meta["lesion_id"].nunique())
    print("70/15/15 split:\n", meta["split"].value_counts().to_string())
    print("80/20 split:\n", meta["split80"].value_counts().to_string())
    print(meta.groupby("split80")["dx"].value_counts(normalize=True).unstack().round(3))


if __name__ == "__main__":
    main()
