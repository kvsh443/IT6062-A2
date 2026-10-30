# IT6062 Advanced Machine Learning, Assignment 02

Causal inference, graph neural networks, and recommender systems on dunnhumby The Complete Journey.

## 1. Environment

```bash
uv sync
uv run python -m ipykernel install --user --name it6062-uv --display-name "Python 3 (uv)"
```

Python 3.13, set by `.python-version`. `uv sync` installs the exact versions locked in
`uv.lock`. `requirements.txt` lists the same dependencies as minimum bounds, for pip users.

The `ipykernel install` step registers a Jupyter kernel bound to `.venv`.

On Windows and Linux `torch` is installed from the PyTorch CUDA 12.8 index, set in `pyproject.toml`.
The same build runs on the CPU when no CUDA device is present.

Prefix every later command with `uv run`.

To rebuild the environment from a clean state, delete `.venv` and sync again:

```bash
rm -rf .venv
uv sync
uv run python -m ipykernel install --user --name it6062-uv --display-name "Python 3 (uv)"
```

On Windows, close VS Code first. Its Python extensions keep a process running from `.venv`, which
blocks the delete.

## 2. Data

Place a Kaggle API token at `~/.kaggle/kaggle.json`, then from the repository root:

```bash
mkdir -p data
uv run kaggle datasets download -d frtgnn/dunnhumby-the-complete-journey -p data --unzip
```

Required afterwards in `data/`: `transaction_data.csv`, `product.csv`, `hh_demographic.csv`,
`campaign_table.csv`, `campaign_desc.csv`, `coupon.csv`, `coupon_redempt.csv`, `causal_data.csv`.

The first run writes intermediate caches to `data/cache/`.

## 3. Run

```bash
uv run jupyter lab src/MS25941494.ipynb
```

Select the **Python 3 (uv)** kernel, then run all cells. The last cell writes the stage
transcript to `results/artifacts/run_all_log_.txt`.

Or run the tasks directly, in this order:

```bash
cd src
uv run --project .. python task1_setup.py
uv run --project .. python task2_causal.py
uv run --project .. python task3_gnn.py
uv run --project .. python task4_recsys.py
```

Runtime: about 14 minutes on a laptop RTX 5060 GPU and about 70 to 75 minutes on a CPU. The code uses CUDA when available and falls back to the CPU otherwise.

## 4. Configuration

Edit `src/config.py`:

| Setting | Value |
| --- | --- |
| `RANDOM_SEED` | 25941494 |
| `REPEAT_SEEDS` | 25941494, 42, 51882990 |
| `TREATMENT_CAMPAIGN`, `COVARIATE_DAYS`, `OUTCOME_DAYS` | 18, 84, 56 |
| `TRIM_PRIMARY`, `BOOTSTRAP_REPS` | 0.05, 300 |
| `GRAPH_K`, `MIN_ITEM_HOUSEHOLDS` | 10, 30 |
| `TRAIN_LAST_WEEK`, `VAL_LAST_WEEK`, `TEST_LAST_WEEK` | 86, 94, 102 |

## 5. Outputs

| Path | Contents |
| --- | --- |
| `results/figures/` | 17 figures |
| `results/tables/` | 30 tables |
| `results/artifacts/` | run log `run_all_log_.txt` |

## Layout

```
data/                       dunnhumby files from Kaggle
src/config.py               paths, seeds, windows, hyperparameters
src/data.py                 loading, household covariates, product features, graph, interaction splits
src/evaluation.py           classification and ranking metrics, aggregation, table writing
src/viz.py                  figure style and colour palette
src/task1_setup.py          dataset description, time windows, conventional baselines
src/task2_causal.py         DAG, propensity, matching, IPW, AIPW, placebo, sensitivity
src/task3_gnn.py            co-purchase graph, MLP, GCN, GraphSAGE, GAT, depth, density, robustness
src/task4_recsys.py         popularity, repeat, content, ItemKNN, BPR-MF, LightGCN, IPS LightGCN
src/MS25941494.ipynb        notebook entry point for all four tasks
results/                    generated figures, tables, artifacts
```
