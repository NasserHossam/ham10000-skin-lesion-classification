import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ham10000.config import add_common_args, config_from_args, CLASSES
import json
import numpy as np, pandas as pd
from ham10000 import bonus
from ham10000.pipeline import handcrafted_data
from ham10000.plots import bonus_figures


def main(argv=None):
    """Bonus: repeated lesion-level splits, GA vs binary PSO vs NSGA-II, stability, feature-group ablation. Resumes after an interruption."""
    ap = add_common_args(argparse.ArgumentParser(description=main.__doc__)); ap.add_argument("--seeds", type=int, default=5, help="number of repeated splits (use 10 for the paper)")
    args = ap.parse_args(argv); cfg = config_from_args(args); n_seeds = 2 if cfg.fast else args.seeds
    d = handcrafted_data(cfg); X, y, groups, names = d["X"], d["y"], d["groups"], d["names"]
    state = cfg.processed_dir / "bonus"; state.mkdir(exist_ok=True)
    f_res, f_mask, f_hist = state / "res_multiseed.csv", state / "selected_masks.json", state / "meta_hist.json"
    rows = pd.read_csv(f_res).to_dict("records") if f_res.exists() else []
    masks = json.loads(f_mask.read_text()) if f_mask.exists() else {}; hists = json.loads(f_hist.read_text()) if f_hist.exists() else {}
    for seed in range(n_seeds):
        if seed in {int(r["seed"]) for r in rows}: continue
        r, m, h = bonus.run_seed(X, y, groups, seed, cfg); rows += r; masks[str(seed)] = m; hists[str(seed)] = h
        pd.DataFrame(rows).to_csv(f_res, index=False); f_mask.write_text(json.dumps(masks)); f_hist.write_text(json.dumps(hists))
        print(f"seed {seed} done:", {x["method"]: round(x["test_f1"], 3) for x in r})
    R = pd.DataFrame(rows); seeds = sorted(R.seed.unique()); P, S, W, fried = bonus.summarise(R)
    print(S.round(3).to_string()); print("Friedman:", fried); print(W.round(4).to_string(index=False))
    stab = bonus.stability(masks, seeds); print(stab.round(3).to_string())
    ab = bonus.ablation(X, y, groups, names, seeds, cfg)
    A = ab.groupby("variant").agg(n=("n_features", "first"), f1=("test_f1", "mean"), f1s=("test_f1", "std"), acc=("test_acc", "mean")); A["delta"] = A["f1"] - A.loc["All features", "f1"]
    print(A.sort_values("f1", ascending=False).round(3).to_string())
    tex = cfg.results_dir / "bonus_tex"; tex.mkdir(exist_ok=True)
    R.round(4).to_csv(cfg.results_dir / "res_multiseed.csv", index=False); S.round(4).to_csv(cfg.results_dir / "summary_multiseed.csv")
    W.round(5).to_csv(cfg.results_dir / "wilcoxon_vs_all.csv", index=False); stab.round(4).to_csv(cfg.results_dir / "stability.csv"); ab.round(4).to_csv(cfg.results_dir / "res_ablation.csv", index=False)
    bonus.export_latex(S, W, A, tex, len(seeds)); bonus_figures(P, S, hists, masks, seeds, names, cfg.figures_dir, bonus.ORDER)


if __name__ == "__main__":
    main()
