import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ham10000.config import add_common_args, config_from_args, CLASSES
import json
import pandas as pd
from ham10000.evaluation import evaluate_subset
from ham10000.metaheuristics import make_fitness, penalised, run_ga
from ham10000.pipeline import handcrafted_data, save_table
from ham10000.plots import filter_figure, embedded_figure, ga_figure, selection_summary_figure
from ham10000.selection import filter_scores, top_k, wrapper_select, embedded_select, l2_top_k, rf_importance


def main(argv=None):
    """Steps 4 and 5: build the 47 features, then filter, wrapper, embedded and genetic-algorithm selection."""
    args = add_common_args(argparse.ArgumentParser(description=main.__doc__)).parse_args(argv); cfg = config_from_args(args)
    d = handcrafted_data(cfg); names, Xs, y, is_tr, groups = d["names"], d["Xs"], d["y"], d["is_train"], d["groups"]
    Xtr, Xte, ytr, yte, gtr = Xs[is_tr], Xs[~is_tr], y[is_tr], y[~is_tr], groups[is_tr]
    ev = lambda cols, name: evaluate_subset(Xtr, ytr, Xte, yte, list(cols), name, cfg)
    rows, chosen = [ev(range(len(names)), "All features")], {"All features": names}
    print(rows[0])
    mi, fs = filter_scores(Xtr, ytr, cfg.seed); filter_figure(mi, fs, names, cfg.figures_dir)
    for k in [5, 10, 15, 20, 30]:
        for tag, sc in (("MI", mi), ("ANOVA-F", fs)): rows.append(ev(top_k(sc, k), f"Filter: {tag} top-{k}"))
    for direction in ["forward", "backward"]:
        cols = wrapper_select(Xtr, ytr, gtr, cfg.wrapper_k, direction, cfg.wrapper_sub); rows.append(ev(cols, f"Wrapper: {direction} ({cfg.wrapper_k})"))
        chosen[f"Wrapper {direction}"] = [names[i] for i in cols]; print(direction, chosen[f"Wrapper {direction}"])
    C1, m1, nz1 = embedded_select(Xtr, ytr, "L1"); rows.append(ev(nz1, f"Embedded: L1 (C={C1})")); chosen["L1"] = [names[i] for i in nz1]
    Ce, me, nze = embedded_select(Xtr, ytr, "EN"); rows.append(ev(nze, f"Embedded: Elastic Net (C={Ce})")); chosen["Elastic Net"] = [names[i] for i in nze]
    k = len(nz1); top2, m2 = l2_top_k(Xtr, ytr, k); rows.append(ev(top2, f"Embedded: L2 top-{k}"))
    toprf, rf = rf_importance(Xtr, ytr, k, trees=cfg.rf_trees); rows.append(ev(toprf, f"Embedded: RF importance top-{k}"))
    embedded_figure([("L1 |coef|", abs(m1.coef_).mean(0)), ("Elastic Net |coef|", abs(me.coef_).mean(0)), ("RF importance", rf.feature_importances_)], names, cfg.figures_dir)
    fit = penalised(make_fitness(Xtr, ytr, gtr, seed=0, n_trees=cfg.fit_trees))
    best, hist, mean_hist = run_ga(len(names), fit, cfg.ga_pop, cfg.ga_gens, seed=0, verbose=True)
    ga_cols = [int(i) for i in best.nonzero()[0]]; chosen["GA"] = [names[i] for i in ga_cols]
    rows.append(ev(ga_cols, f"Metaheuristic: GA ({len(ga_cols)})")); print("GA selected", len(ga_cols), "features:", chosen["GA"]); ga_figure(hist, mean_hist, cfg.figures_dir)
    res = pd.DataFrame(rows).round(3); res["Gap (train-test F1)"] = (res["Train macro-F1"] - res["Test macro-F1"]).round(3)
    res = res.sort_values("Test macro-F1", ascending=False); print(res.to_string(index=False))
    save_table(res, cfg.results_dir / "res_selection.csv"); selection_summary_figure(res, cfg.figures_dir)
    (cfg.results_dir / "selected_features.json").write_text(json.dumps(chosen, indent=1)); print("saved", cfg.results_dir / "selected_features.json")


if __name__ == "__main__":
    main()
