# Skin Lesion Classification on HAM10000

Course project for **AID427 New Trends in Data Science**, Egypt-Japan University of Science and Technology (E-JUST).
**Team:** Nasser Hossam Hamed (320230051), Mariam Ahmed Shawky (320230056) and Ammar Yasser (320230031), Section 1.

Seven-class classification of dermoscopic skin lesions with classical machine learning: dimensionality reduction of raw pixels,
47 interpretable features, filter / wrapper / embedded / metaheuristic feature selection, and a lesion-level evaluation protocol.
A deep-learning reference (ResNet18, image + metadata fusion) is included for comparison.

## Dataset

Not included. Download HAM10000 from Kaggle (`kmader/skin-cancer-mnist-ham10000`) and keep `HAM10000_metadata.csv` and the two image
folders together in one folder.

> P. Tschandl, C. Rosendahl and H. Kittler, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common
> pigmented skin lesions," *Scientific Data*, vol. 5, 180161, 2018.

Check the dataset's licence before reusing the images.

## Evaluation protocol

- Every split is made **by lesion** (`lesion_id`); about 26% of lesions have several images, so an image-level split would leak.
- Scalers, imputation, reducers, feature selectors and classifiers are fitted on **training data only**.
- `dx_type` (how the diagnosis was confirmed) is never used as a feature.
- Macro-F1 is the main metric (67% of the images are nevi); per-class recall and ROC-AUC are reported too.

## Project structure

```
.
├── src/ham10000/            the reusable code
│   ├── config.py            paths, constants, run sizes (--fast)
│   ├── data.py              metadata, image paths, lesion-level splits, resized image array
│   ├── features.py          flattened pixels, 28 colour/texture features, metadata features
│   ├── reduction.py         PCA, LDA, SVD, Kernel PCA, Isomap, LLE, UMAP, t-SNE, SNE, MDS, K-Means
│   ├── selection.py         filter, wrapper (forward/backward), embedded (L1, L2, Elastic Net, RF importance)
│   ├── metaheuristics.py    genetic algorithm, binary PSO, NSGA-II
│   ├── models.py            the five classifiers
│   ├── evaluation.py        metrics, grouped cross-validation, overfitting table
│   ├── plots.py             all figures
│   ├── deep.py              ResNet18 and fusion model (needs PyTorch)
│   ├── bonus.py             repeated splits, statistical tests, stability, ablation, LaTeX export
│   ├── pipeline.py          shared loading helpers
│   └── synthetic.py         tiny fake dataset for testing
├── scripts/                 one command-line script per stage (see below) and run_all.py
├── tests/                   unit tests and an end-to-end smoke test
├── notebooks/               the original Colab notebooks (the record of how the work was done)
├── results/                 result tables (CSV, JSON), written by the scripts
├── figures/                 figures (PNG), written by the scripts
├── docs/                    project summary
├── .vscode/                 VS Code settings and launch configurations
├── requirements.txt
└── requirements-dev.txt
```

## Quick start (VS Code, Windows)

```powershell
cd ham10000-skin-lesion-classification
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
code .
```

In VS Code pick the interpreter (**Ctrl+Shift+P**, *Python: Select Interpreter*, choose `.venv`). Then run the whole pipeline.
If the dataset sits next to this folder in `..\Skin Cancer`:

```powershell
python scripts/run_all.py --data-dir "..\Skin Cancer"
```

Large intermediate files go to `outputs/` (git-ignored); result tables go to `results/` and figures to `figures/`.
Use `--work-dir`, `--results-dir` and `--figures-dir` to change those, or the environment variables `HAM_DATA_DIR` and `HAM_WORK_DIR`.

**Try it without the dataset** (fake images, takes a few minutes):

```powershell
python scripts/make_synthetic_data.py --out data/synthetic
python scripts/run_all.py --data-dir data/synthetic --work-dir outputs_test --results-dir outputs_test/results --figures-dir outputs_test/figures --fast
```

## The pipeline

| Script | Stage | Writes |
|---|---|---|
| `01_prepare_data.py` | metadata, lesion-level 70/15/15 and 80/20 splits, resized 128x128 image array | `outputs/processed/` |
| `02_eda.py` | missing values, class balance, images per lesion, EDA figures | `figures/eda_*.png` |
| `03_deep_learning.py` | ResNet18 and fusion model (optional, needs PyTorch; use `--deep` with `run_all.py`) | `results/res_deep_learning.csv` |
| `04_dimensionality_reduction.py` | PCA, LDA, SVD, K-Means and eight non-linear methods | `results/res_linear.csv`, `res_nonlinear.csv`, `clustering.json`, figures |
| `05_feature_selection.py` | 47 features; filter, wrapper, embedded and genetic-algorithm selection | `results/res_selection.csv`, `selected_features.json`, figures |
| `06_final_models.py` | five classifiers x three feature sets, 80/20 test metrics, grouped 5-fold CV, confusion matrix, ROC, overfitting | `results/res_final_models.csv`, figures |
| `07_bonus_experiments.py` | repeated splits, GA vs binary PSO vs NSGA-II, Wilcoxon tests, stability, ablation (optional; `--bonus`) | `results/res_multiseed.csv`, ..., `results/bonus_tex/`, figures |

Every script can be run on its own and has `--help`. Scripts 05 and 06 need 01 first; 06 needs 05. Step 07 resumes if interrupted.
Use `--seeds 10` for the bonus experiments (the default is 5), and `--fast` for tiny settings to check that everything runs.

## Key results (80/20 lesion-level split, 1,995 test images)

| Experiment | Test macro-F1 |
|---|---|
| Raw-pixel methods (PCA, LDA, SVD, Kernel PCA, Isomap, LLE, UMAP) | about 0.20 to 0.24 |
| 47 interpretable features + Random Forest | 0.457 |
| Final model: gradient boosting on 21 genetic-algorithm features | **0.463** |

Final model: accuracy 0.723, macro ROC-AUC 0.906, cross-validated macro-F1 0.509; melanoma recall is only **0.51**.
Deep-learning reference (a different 70/15/15 split, so only indicative): ResNet18 0.515, image + metadata fusion 0.615.
Two Random Forests are used on purpose: step 05 scores every feature subset with one lightly regularised forest (`min_samples_leaf=3`), while step 06 uses a more regularised one (`min_samples_leaf=5`, `max_depth=12`). So the same GA (21) set scores 0.438 in `res_selection.csv` and 0.464 in `res_final_models.csv`; compare numbers only within one table.
Differences below about 0.03 macro-F1 are within noise, because the rare classes have only 20 to 60 test images.

These numbers come from the Colab notebooks in `notebooks/`. The scripts implement the same pipeline; small differences in the third
decimal are possible (see below), so re-run the scripts and compare with `results/` before quoting numbers.

## Differences between the scripts and the original notebooks

- Age imputation uses the median of the training rows of the split being used (the notebooks used the 70/15/15 training median for every split).
- Logistic regression uses `max_iter=3000` (the notebooks used 500 and printed convergence warnings for the raw-pixel model).
- LLE and non-metric MDS use the corrected settings directly (30 neighbours with regularisation 0.1; non-metric MDS started from the metric solution).
- UMAP is optional: it is skipped with a message if `umap-learn` is not installed.
- Nothing needs a GPU or Google Drive; only `03_deep_learning.py` needs PyTorch.

## Tests

```powershell
pytest -m "not slow"     # unit tests, a few seconds
pytest                   # also the end-to-end smoke test, about 2 minutes
```

## Limitations

- Rare classes have few test images, so their scores are unstable.
- Images are reduced to 128x128 (and smaller for some methods); heavy reduction methods use subsamples.
- HAM10000 comes from two sites and mostly light skin tones, so the results may not generalise.
- A course project, not a clinical tool.
