"""Paths, constants and run settings shared by every module and script."""
from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]      # alphabetical = label order
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
MEL = CLASS_TO_IDX["mel"]
IMAGE_DIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]


@dataclass
class Config:
    """Where the data lives and how big the experiments are.

    data_dir  folder that holds HAM10000_metadata.csv and the two image folders (never committed)
    work_dir  big intermediate files (image array, feature tables, model weights); git-ignored
    results_dir / figures_dir  small outputs that are committed to the repository
    fast      tiny settings used by the smoke test and for quick experiments
    """
    data_dir: Path = Path(os.environ.get("HAM_DATA_DIR", ROOT / "data" / "raw"))
    work_dir: Path = Path(os.environ.get("HAM_WORK_DIR", ROOT / "outputs"))
    results_dir: Path = ROOT / "results"
    figures_dir: Path = ROOT / "figures"
    seed: int = 42
    img_size: int = 128
    fast: bool = False

    def __post_init__(self):
        self.data_dir, self.work_dir = Path(self.data_dir), Path(self.work_dir)
        self.results_dir, self.figures_dir = Path(self.results_dir), Path(self.figures_dir)

    @property
    def processed_dir(self) -> Path:
        return self.work_dir / "processed"

    def ensure_dirs(self) -> None:
        for d in (self.processed_dir, self.results_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---- sizes that shrink in fast mode ----
    @property
    def rf_trees(self) -> int: return 30 if self.fast else 300
    @property
    def ga_pop(self) -> int: return 6 if self.fast else 20
    @property
    def ga_gens(self) -> int: return 3 if self.fast else 15
    @property
    def fit_trees(self) -> int: return 10 if self.fast else 40
    @property
    def nl_fit_n(self) -> int: return 300 if self.fast else 3000
    @property
    def nl_vis_n(self) -> int: return 200 if self.fast else 1500
    @property
    def wrapper_sub(self) -> int: return 400 if self.fast else 2500
    @property
    def wrapper_k(self) -> int: return 5 if self.fast else 15
    @property
    def cv_folds(self) -> int: return 3 if self.fast else 5


def add_common_args(parser):
    """--data-dir, --work-dir, --fast for every script."""
    parser.add_argument("--data-dir", default=None, help="folder with HAM10000_metadata.csv and the image folders")
    parser.add_argument("--work-dir", default=None, help="folder for large intermediate files (default: ./outputs)")
    parser.add_argument("--results-dir", default=None, help="where result tables are written (default: ./results)")
    parser.add_argument("--figures-dir", default=None, help="where figures are written (default: ./figures)")
    parser.add_argument("--fast", action="store_true", help="tiny settings for a quick test run")
    return parser


def config_from_args(args) -> Config:
    kw = {"fast": args.fast}
    if args.data_dir: kw["data_dir"] = Path(args.data_dir)
    if args.work_dir: kw["work_dir"] = Path(args.work_dir)
    if args.results_dir: kw["results_dir"] = Path(args.results_dir)
    if args.figures_dir: kw["figures_dir"] = Path(args.figures_dir)
    cfg = Config(**kw); cfg.ensure_dirs(); return cfg
