import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
ARTIFACT_DIR = RESULTS_DIR / "artifacts"

TRANSACTION_FILE = DATA_DIR / "transaction_data.csv"
PRODUCT_FILE = DATA_DIR / "product.csv"
DEMOGRAPHIC_FILE = DATA_DIR / "hh_demographic.csv"
CAMPAIGN_TABLE_FILE = DATA_DIR / "campaign_table.csv"
CAMPAIGN_DESC_FILE = DATA_DIR / "campaign_desc.csv"
COUPON_FILE = DATA_DIR / "coupon.csv"
REDEMPTION_FILE = DATA_DIR / "coupon_redempt.csv"
EXPOSURE_FILE = DATA_DIR / "causal_data.csv"

for _d in (CACHE_DIR, FIGURE_DIR, TABLE_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False

RANDOM_SEED = 25941494
REPEAT_SEEDS = (
    RANDOM_SEED,
    RANDOM_SEED - RANDOM_SEED + 42,
    RANDOM_SEED + RANDOM_SEED + 2,
)

TREATMENT_CAMPAIGN = 18
COVARIATE_DAYS = 84
OUTCOME_DAYS = 56
PRIOR_TYPE_A = (8, 13)
BOOTSTRAP_REPS = 300
CROSS_FIT_FOLDS = 5
PROPENSITY_CLIP = (0.01, 0.99)
TRIM_LEVELS = (0.0, 0.01, 0.025, 0.05, 0.1)
TRIM_PRIMARY = 0.05
CALIPER_SD = 0.2

TRAIN_LAST_WEEK = 86
VAL_LAST_WEEK = 94
TEST_LAST_WEEK = 102
MIN_ITEM_HOUSEHOLDS = 30

GRAPH_K = 10
GRAPH_MIN_COOCCURRENCE = 3
MIN_CLASS_NODES = 50
GNN_HIDDEN = 128
GNN_LAYERS = 2
GNN_DROPOUT = 0.2
GNN_LR = 0.01
GNN_WEIGHT_DECAY = 5e-4
GNN_MAX_EPOCHS = 300
GNN_PATIENCE = 40
GAT_HEADS = 4
DEPTH_SWEEP = (1, 2, 3, 4, 6, 8)
AGGREGATIONS = ("mean", "max", "sum")
K_SWEEP = (3, 5, 10, 20, 40)
EDGE_DROP_LEVELS = (0.0, 0.25, 0.5, 0.75, 0.9)
EDGE_NOISE_LEVELS = (0.0, 0.25, 0.5, 1.0, 2.0)

TOP_K = (10, 20)
EMBED_DIM = 64
LIGHTGCN_LAYERS = 3
REC_LR = 0.005
REC_WEIGHT_DECAY = 1e-2
REC_BATCH = 8192
REC_EPOCHS = 30
REC_EVAL_EVERY = 2
ITEMKNN_NEIGHBOURS = 200
IPS_POWER = 0.5
IPS_CLIP = 0.05
