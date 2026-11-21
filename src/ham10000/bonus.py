"""Bonus experiments: repeated lesion-level splits, metaheuristic comparison, stability and feature-group ablation."""
import itertools, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import Config
from .features import feature_group
from .metaheuristics import make_fitness, penalised, run_ga, run_bpso, run_nsga2, knee
from .selection import filter_scores, top_k, wrapper_select, embedded_select, rf_importance

K_SUBSET = 20
ORDER = ["All features", "Filter: MI", "Filter: ANOVA-F", "Wrapper: forward", "Wrapper: backward", "Embedded: L1",
         "Embedded: Elastic Net", "Embedded: RF importance", "GA", "BPSO", "NSGA-II (knee)"]


def lesion_split(groups, y, seed, test_size=0.20):
    first = pd.DataFrame({"g": groups, "y": y}).drop_duplicates("g")
    tr_g, _ = train_test_split(first["g"].values, test_size=test_size, stratify=first["y"].values, random_state=seed)
    is_tr = np.isin(groups, tr_g); assert set(groups[is_tr]).isdisjoint(set(groups[~is_tr])); return is_tr


def evaluate(Xtr, ytr, Xte, yte, cols, seed, trees):
    rf = RandomForestClassifier(trees, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=seed).fit(Xtr[:, cols], ytr)
    p, pr = rf.predict(Xte[:, cols]), rf.predict_proba(Xte[:, cols])
    ok = len(rf.classes_) == 7 and len(np.unique(yte)) == 7
    return dict(test_f1=f1_score(yte, p, average="macro"), test_acc=accuracy_score(yte, p),
                auc=roc_auc_score(yte, pr, multi_class="ovr", average="macro", labels=rf.classes_) if ok else np.nan,
                train_f1=f1_score(ytr, rf.predict(Xtr[:, cols]), average="macro"))


def run_seed(X_all, y, groups, seed, cfg: Config):
    """One repeated split: every selector runs on the training lesions only; every subset is scored once on the test lesions."""
    n = X_all.shape[1]; is_tr = lesion_split(groups, y, seed)
    Xs = StandardScaler().fit(X_all[is_tr]).transform(X_all)
    Xtr, Xte, ytr, yte, gtr = Xs[is_tr], Xs[~is_tr], y[is_tr], y[~is_tr], groups[is_tr]
    f1_of = make_fitness(Xtr, ytr, gtr, seed, n_trees=10 if cfg.fast else 30, max_samples=0.6, sub=4000); fit = penalised(f1_of)
    mi, fs = filter_scores(Xtr, ytr, seed); sub = {"All features": list(range(n)), "Filter: MI": top_k(mi, K_SUBSET), "Filter: ANOVA-F": top_k(fs, K_SUBSET)}
    wk, ws = (5, 400) if cfg.fast else (K_SUBSET, 1500)
    sub["Wrapper: forward"] = wrapper_select(Xtr, ytr, gtr, wk, "forward", ws, seed)
    sub["Wrapper: backward"] = wrapper_select(Xtr, ytr, gtr, wk, "backward", ws, seed)
    sub["Embedded: L1"] = [int(i) for i in embedded_select(Xtr, ytr, "L1", seed=seed, verbose=False)[2]]
    sub["Embedded: Elastic Net"] = [int(i) for i in embedded_select(Xtr, ytr, "EN", seed=seed, verbose=False)[2]]
    sub["Embedded: RF importance"] = rf_importance(Xtr, ytr, K_SUBSET, trees=50 if cfg.fast else 200, seed=seed)[0]
    g, hg, _ = run_ga(n, fit, cfg.ga_pop, cfg.ga_gens, seed=seed); sub["GA"] = list(np.where(g)[0])
    b, hb = run_bpso(n, fit, cfg.ga_pop, cfg.ga_gens, seed=seed); sub["BPSO"] = list(np.where(b)[0])
    M, F1s, sizes, hn = run_nsga2(n, f1_of, cfg.ga_pop, cfg.ga_gens, seed=seed); sub["NSGA-II (knee)"] = list(np.where(knee(M, F1s, sizes))[0])
    rows = []
    for name, cols in sub.items():
        r = evaluate(Xtr, ytr, Xte, yte, [int(c) for c in cols], seed, cfg.rf_trees); r.update(seed=seed, method=name, n_features=len(cols)); rows.append(r)
    hist = {"GA": hg, "BPSO": hb, "NSGA-II": hn, "pareto": {"f1": [float(v) for v in F1s], "size": [int(v) for v in sizes]}}
    return rows, {k: [int(i) for i in v] for k, v in sub.items()}, hist


def summarise(R: pd.DataFrame):
    """Mean, std and rank per method, the Friedman test and Wilcoxon tests against 'All features' (Holm-corrected)."""
    P = R.pivot(index="seed", columns="method", values="test_f1")[ORDER]
    S = R.groupby("method").agg(n=("n_features", "mean"), f1=("test_f1", "mean"), f1s=("test_f1", "std"), acc=("test_acc", "mean"),
                                auc=("auc", "mean"), train=("train_f1", "mean")).loc[ORDER]
    S["gap"] = S["train"] - S["f1"]; S["rank"] = P.rank(axis=1, ascending=False).mean()
    fried = stats.friedmanchisquare(*[P[m].values for m in ORDER]) if len(P) >= 3 else None
    W = []
    for m in ORDER[1:]:
        d = P[m] - P["All features"]
        try: p = stats.wilcoxon(d, zero_method="wilcox").pvalue if (d != 0).any() else 1.0
        except ValueError: p = 1.0
        W.append((m, d.mean(), p))
    W = pd.DataFrame(W, columns=["method", "mean_diff_vs_all", "p"]).sort_values("p")
    W["p_holm"] = np.maximum.accumulate(np.minimum(1, W["p"].values * (len(W) - np.arange(len(W)))))
    return P, S, W, fried


def stability(masks: dict, seeds) -> pd.Series:
    """Mean pairwise Jaccard index of the subsets chosen on different splits."""
    jac = lambda a, b: len(set(a) & set(b)) / len(set(a) | set(b))
    out = {}
    for m in ORDER[1:]:
        sets = [masks[str(s)][m] for s in seeds if str(s) in masks]
        pairs = [jac(a, b) for a, b in itertools.combinations(sets, 2)]; out[m] = np.mean(pairs) if pairs else np.nan
    return pd.Series(out, name="stability")


def ablation(X_all, y, groups, names, seeds, cfg: Config) -> pd.DataFrame:
    """Leave-one-group-out and single-group-only Random Forests over the same splits."""
    gof = np.array([feature_group(n) for n in names]); glist = list(dict.fromkeys(gof)); rows = []
    for seed in seeds:
        is_tr = lesion_split(groups, y, seed); Xs = StandardScaler().fit(X_all[is_tr]).transform(X_all)
        Xtr, Xte, ytr, yte = Xs[is_tr], Xs[~is_tr], y[is_tr], y[~is_tr]
        variants = {"All features": np.arange(X_all.shape[1])}
        for g in glist: variants[f"without {g}"] = np.where(gof != g)[0]; variants[f"only {g}"] = np.where(gof == g)[0]
        for name, cols in variants.items():
            r = evaluate(Xtr, ytr, Xte, yte, [int(c) for c in cols], seed, 20 if cfg.fast else 150); r.update(seed=seed, variant=name, n_features=len(cols)); rows.append(r)
    return pd.DataFrame(rows)


def _esc(s): return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def export_latex(S, W, A, tex_dir, n_seeds):
    """Write the LaTeX tables and macros that the paper includes automatically."""
    pm = lambda m, s: f"{m:.3f}$\\pm${s:.3f}"; row_end = " \\\\"
    L = [r"\begin{table}[t]", r"\caption{Repeated lesion-level 80/20 splits (%d seeds): mean $\pm$ std of the test scores; rank over seeds (1 = best).}" % n_seeds,
         r"\label{tab:multiseed}", r"\centering\footnotesize", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule", r"Method & \# feat. & Macro-F1 & Accuracy & AUC & Rank \\ \midrule"]
    L += [f"{_esc(m)} & {r['n']:.1f} & {pm(r['f1'], r['f1s'])} & {r['acc']:.3f} & {r['auc']:.3f} & {r['rank']:.1f}{row_end}" for m, r in S.iterrows()]
    (tex_dir / "tab_multiseed.tex").write_text("\n".join(L + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]))
    L = [r"\begin{table}[t]", r"\caption{Wilcoxon signed-rank tests against \emph{All features} (Holm-adjusted p-values).}", r"\label{tab:wilcoxon}",
         r"\centering\footnotesize", r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"Method & Mean $\Delta$F1 & $p_{\mathrm{Holm}}$ \\ \midrule"]
    L += [f"{_esc(r['method'])} & {r['mean_diff_vs_all']:+.3f} & {r['p_holm']:.3f}{row_end}" for _, r in W.iterrows()]
    (tex_dir / "tab_wilcoxon.tex").write_text("\n".join(L + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]))
    L = [r"\begin{table}[t]", r"\caption{Feature-group ablation (test macro-F1, mean $\pm$ std over seeds).}", r"\label{tab:ablation}",
         r"\centering\footnotesize", r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Variant & \# feat. & Macro-F1 & $\Delta$ vs all \\ \midrule"]
    L += [f"{_esc(v)} & {int(r['n'])} & {pm(r['f1'], r['f1s'])} & {r['delta']:+.3f}{row_end}" for v, r in A.sort_values("f1", ascending=False).iterrows()]
    (tex_dir / "tab_ablation.tex").write_text("\n".join(L + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]))
    best = S["f1"].idxmax()
    macros = {"NSeeds": n_seeds, "BestMethod": _esc(best), "BestFscore": f"{S.loc[best, 'f1']:.3f}", "AllFscore": f"{S.loc['All features', 'f1']:.3f}",
              "GAFscore": f"{S.loc['GA', 'f1']:.3f}", "BPSOFscore": f"{S.loc['BPSO', 'f1']:.3f}", "NSGAFscore": f"{S.loc['NSGA-II (knee)', 'f1']:.3f}",
              "GASize": f"{S.loc['GA', 'n']:.1f}", "NSGASize": f"{S.loc['NSGA-II (knee)', 'n']:.1f}"}
    (tex_dir / "bonus_macros.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()))
