import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ham10000.config import add_common_args, config_from_args, CLASSES
import json
import numpy as np
from ham10000.data import load_processed
from ham10000.features import flat_features
from ham10000.pipeline import save_table
from ham10000.plots import linear_figures, nonlinear_figure
from ham10000.reduction import linear_reduction, kmeans_report, nonlinear_reduction


def main(argv=None):
    """Steps 2 and 3: PCA, LDA, SVD, K-Means and eight non-linear methods on the raw pixels."""
    args = add_common_args(argparse.ArgumentParser(description=main.__doc__)).parse_args(argv); cfg = config_from_args(args)
    meta, imgs = load_processed(cfg); is_train = (meta["split80"] == "train").values; y = meta["label"].values
    cache = cfg.processed_dir / "features_28g.npy"
    if cache.exists(): X = np.load(cache)
    else: X = flat_features(imgs, 28, gray=True); np.save(cache, X)
    res_lin, info = linear_reduction(X, y, is_train)
    print(res_lin.to_string(index=False)); print(f"components for 90% variance: {info['n90']} | for 95%: {info['n95']}")
    save_table(res_lin, cfg.results_dir / "res_linear.csv")
    km, kres = kmeans_report(info["Ztr50"], info["ytr"], seed=cfg.seed); print("K-Means vs true classes:", {k: round(v, 3) for k, v in kres.items()})
    (cfg.results_dir / "clustering.json").write_text(json.dumps({**kres, "n90": info["n90"], "n95": info["n95"]}, indent=1))
    linear_figures(info, X[is_train], info["ytr"], CLASSES, cfg.figures_dir, km)
    res_nl, emb, yvis = nonlinear_reduction(info["Ztr50"], info["Zte50"], info["ytr"], info["yte"], cfg.nl_fit_n, cfg.nl_vis_n)
    print(res_nl.to_string(index=False)); save_table(res_nl, cfg.results_dir / "res_nonlinear.csv")
    nonlinear_figure(emb, yvis, CLASSES, cfg.figures_dir)


if __name__ == "__main__":
    main()
