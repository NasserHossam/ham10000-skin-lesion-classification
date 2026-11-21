import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ham10000.config import add_common_args, config_from_args, CLASSES
from ham10000.data import load_processed
from ham10000.deep import train_and_evaluate
from ham10000.pipeline import save_table


def main(argv=None):
    """Deep-learning reference study (needs PyTorch): ResNet18 and an image + metadata fusion network."""
    ap = add_common_args(argparse.ArgumentParser(description=main.__doc__))
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="use only this many images per split (quick test)")
    args = ap.parse_args(argv); cfg = config_from_args(args)
    meta, imgs = load_processed(cfg)
    res = train_and_evaluate(imgs, meta, cfg, epochs=1 if cfg.fast else args.epochs, batch_size=args.batch_size, limit=args.limit or (60 if cfg.fast else None))
    print(res.to_string(index=False)); save_table(res, cfg.results_dir / "res_deep_learning.csv")


if __name__ == "__main__":
    main()
