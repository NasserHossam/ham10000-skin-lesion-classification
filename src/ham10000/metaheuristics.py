"""Metaheuristic feature selection: genetic algorithm, binary PSO and NSGA-II.

A subset is a boolean mask over the features. Every algorithm scores a mask with the same inner validation
(a lesion-grouped hold-out inside the TRAINING set), so the test set is never touched.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

ALPHA = 0.002   # penalty per selected feature in the single-objective fitness


def make_fitness(Xtr, ytr, gtr, seed=0, n_trees=40, max_samples=None, sub=None):
    """Return f1_of(mask): macro-F1 on a 25% lesion-grouped hold-out of the training set (cached)."""
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(Xtr, ytr, gtr))
    if sub and len(tr_i) > sub: tr_i = np.random.default_rng(seed).choice(tr_i, sub, replace=False)
    Xa, ya, Xb, yb = Xtr[tr_i], ytr[tr_i], Xtr[va_i], ytr[va_i]; cache = {}

    def f1_of(mask):
        key = mask.tobytes()
        if key not in cache:
            if not mask.any(): cache[key] = 0.0
            else:
                rf = RandomForestClassifier(n_trees, max_depth=8, max_samples=max_samples, class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=seed).fit(Xa[:, mask], ya)
                cache[key] = f1_score(yb, rf.predict(Xb[:, mask]), average="macro")
        return cache[key]
    return f1_of


def penalised(f1_of, alpha=ALPHA):
    return lambda m: f1_of(m) - alpha * m.sum()


def run_ga(n, fit, pop=20, gens=15, p_mut=0.03, seed=0, verbose=False):
    """Genetic algorithm: tournament selection (3), uniform crossover, bit-flip mutation, elitism of two."""
    r = np.random.default_rng(seed); P = r.random((pop, n)) < 0.5; hist, mean_hist = [], []
    for g in range(gens):
        f = np.array([fit(x) for x in P]); hist.append(float(f.max())); mean_hist.append(float(f.mean()))
        if verbose: print(f"gen {g + 1}: best {f.max():.4f}  mean {f.mean():.4f}")
        o = np.argsort(f)[::-1]; new = [P[o[0]].copy(), P[o[1]].copy()]
        pick = lambda: P[(lambda c: c[np.argmax(f[c])])(r.choice(pop, 3, replace=False))]
        while len(new) < pop:
            c = np.where(r.random(n) < 0.5, pick(), pick()) ^ (r.random(n) < p_mut)
            if not c.any(): c[r.integers(n)] = True
            new.append(c)
        P = np.array(new)
    f = np.array([fit(x) for x in P]); return P[np.argmax(f)], hist, mean_hist


def run_bpso(n, fit, pop=20, iters=15, seed=0, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax=4.0):
    """Binary PSO (Kennedy and Eberhart, 1997): a sigmoid of the velocity gives the probability of a bit being 1."""
    r = np.random.default_rng(seed); X = r.random((pop, n)) < 0.5; V = r.uniform(-1, 1, (pop, n))
    pf = np.array([fit(x) for x in X]); pb = X.copy(); gb = pb[pf.argmax()].copy(); gf = pf.max(); hist = [float(gf)]
    for t in range(1, iters):
        w = w_max - (w_max - w_min) * t / max(1, iters - 1); Xf = X.astype(float)
        V = np.clip(w * V + c1 * r.random((pop, n)) * (pb.astype(float) - Xf) + c2 * r.random((pop, n)) * (gb.astype(float) - Xf), -vmax, vmax)
        X = r.random((pop, n)) < 1 / (1 + np.exp(-V))
        for i in range(pop):
            if not X[i].any(): X[i, r.integers(n)] = True
        f = np.array([fit(x) for x in X]); b = f > pf; pb[b], pf[b] = X[b], f[b]
        if pf.max() > gf: gf = pf.max(); gb = pb[pf.argmax()].copy()
        hist.append(float(gf))
    return gb, hist


def nd_sort(F):
    """Fast non-dominated sorting (all objectives minimised). Returns a list of fronts (index lists)."""
    n = len(F); S = [[] for _ in range(n)]; cnt = np.zeros(n, int); fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q: continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]): S[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]): cnt[p] += 1
        if cnt[p] == 0: fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0: nxt.append(q)
        i += 1; fronts.append(nxt)
    return fronts[:-1]


def crowding(F, idx):
    d = np.zeros(len(idx))
    for m in range(F.shape[1]):
        o = np.argsort(F[idx, m]); d[o[0]] = d[o[-1]] = np.inf
        rg = F[idx[o[-1]], m] - F[idx[o[0]], m]
        if rg == 0: continue
        for k in range(1, len(idx) - 1): d[o[k]] += (F[idx[o[k + 1]], m] - F[idx[o[k - 1]], m]) / rg
    return d


def run_nsga2(n, f1_of, pop=20, gens=15, p_mut=0.03, seed=0):
    """NSGA-II (Deb et al., 2002). Objectives: maximise macro-F1, minimise the number of features.

    Returns (masks, f1 values, sizes of the final Pareto front, best-F1 history).
    """
    r = np.random.default_rng(seed); P = r.random((pop, n)) < 0.5
    obj = lambda M: np.array([[-f1_of(m), m.sum()] for m in M])
    F = obj(P); hist = []
    for _ in range(gens):
        fronts = nd_sort(F); rank = np.zeros(len(P), int); cd = np.zeros(len(P))
        for k, fr in enumerate(fronts): rank[fr] = k; cd[fr] = crowding(F, np.array(fr))

        def tour():
            a, b = r.choice(len(P), 2, replace=False)
            return P[a] if (rank[a], -cd[a]) < (rank[b], -cd[b]) else P[b]
        Q = []
        while len(Q) < pop:
            c = np.where(r.random(n) < 0.5, tour(), tour()) ^ (r.random(n) < p_mut)
            if not c.any(): c[r.integers(n)] = True
            Q.append(c)
        Q = np.array(Q); R = np.vstack([P, Q]); FR = np.vstack([F, obj(Q)]); sel = []
        for fr in nd_sort(FR):
            if len(sel) + len(fr) <= pop: sel += fr
            else:
                d = crowding(FR, np.array(fr)); sel += list(np.array(fr)[np.argsort(-d)[: pop - len(sel)]]); break
        P, F = R[sel], FR[sel]; hist.append(float(-F[:, 0].max()))
    front = np.array(nd_sort(F)[0]); return P[front], -F[front, 0], F[front, 1].astype(int), hist


def knee(masks, f1s, sizes, tol=0.01):
    """Smallest subset on the Pareto front whose validation F1 is within `tol` of the best."""
    ok = np.where(f1s >= f1s.max() - tol)[0]; return masks[ok[np.argmin(sizes[ok])]]
