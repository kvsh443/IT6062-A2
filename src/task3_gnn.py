import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, LabelPropagation, SAGEConv
from torch_geometric.utils import dropout_edge

import viz
from config import (
    AGGREGATIONS,
    DEVICE,
    DEPTH_SWEEP,
    EDGE_DROP_LEVELS,
    EDGE_NOISE_LEVELS,
    FIGURE_DIR,
    GAT_HEADS,
    GNN_DROPOUT,
    GNN_HIDDEN,
    GNN_LAYERS,
    GNN_LR,
    GNN_MAX_EPOCHS,
    GNN_PATIENCE,
    GNN_WEIGHT_DECAY,
    GRAPH_K,
    K_SWEEP,
    MIN_CLASS_NODES,
    RANDOM_SEED,
    REPEAT_SEEDS,
    TABLE_DIR,
)
from data import catalogue, knn_graph, load_products, pmi_edges, product_features
from evaluation import aggregate, classification_scores, per_class_f1, write_table

warnings.filterwarnings("ignore")
torch.set_num_threads(16)

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
COLOURS = {
    "Logistic regression": viz.SERIES[5],
    "MLP": viz.SERIES[1],
    "Label propagation": viz.SERIES[3],
    "GCN": viz.SERIES[0],
    "GraphSAGE": viz.SERIES[2],
    "GAT": viz.SERIES[4],
}


def node_labels(items):
    departments = load_products().reindex(items)["DEPARTMENT"].to_numpy()
    counts = pd.Series(departments).value_counts()
    classes = counts[counts >= MIN_CLASS_NODES].index.tolist()
    lookup = {c: i for i, c in enumerate(classes)}
    y = np.array([lookup.get(d, -1) for d in departments], dtype=np.int64)
    return y, classes, departments


def graph_dataset():
    items = catalogue()
    feats = product_features(items)
    edges = pmi_edges(items)
    y, classes, departments = node_labels(items)
    return {
        "items": items,
        "features": feats,
        "edges": edges,
        "y": y,
        "classes": classes,
        "departments": departments,
    }


def graph_statistics(edge_index, n, y, departments):
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(n, n))
    degree = np.asarray(adj.sum(axis=1)).ravel()
    n_comp, comp = connected_components(adj, directed=False)
    giant = np.bincount(comp).max()
    src, dst = edge_index
    labelled = (y[src] >= 0) & (y[dst] >= 0)
    edge_h = float((departments[src] == departments[dst]).mean())
    same = y[src][labelled] == y[dst][labelled]
    k = y.max() + 1
    adjusted = 0.0
    for c in range(k):
        rows = labelled & (y[src] == c)
        if rows.sum() == 0:
            continue
        hc = (y[dst][rows] == c).mean()
        adjusted += max(hc - (y == c).sum() / (y >= 0).sum(), 0.0)
    return {
        "nodes": n,
        "undirected_edges": edge_index.shape[1] // 2,
        "density": edge_index.shape[1] / (n * (n - 1)),
        "mean_degree": degree.mean(),
        "median_degree": float(np.median(degree)),
        "max_degree": int(degree.max()),
        "isolated_nodes": int((degree == 0).sum()),
        "components": int(n_comp),
        "giant_component_share": giant / n,
        "edge_homophily_department": edge_h,
        "edge_homophily_labelled": float(same.mean()),
        "class_insensitive_homophily": adjusted / (k - 1),
    }


def node_split(y, seed):
    labelled = np.flatnonzero(y >= 0)
    train, rest = train_test_split(labelled, test_size=0.4, stratify=y[labelled], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    return train, val, test


def scaled_features(features, train):
    scaler = StandardScaler().fit(features[train])
    return np.clip(scaler.transform(features), -6, 6).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, n_in, n_out, layers, hidden, dropout):
        super().__init__()
        sizes = [n_in] + [hidden] * (layers - 1) + [n_out]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        hidden = x
        for i, layer in enumerate(self.layers):
            x = F.dropout(x, self.dropout, self.training)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                hidden = x
        return x, hidden


class GNN(nn.Module):
    def __init__(self, kind, n_in, n_out, layers, hidden, dropout, aggr="mean"):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for i in range(layers):
            a = n_in if i == 0 else hidden
            b = n_out if i == layers - 1 else hidden
            if kind == "gcn":
                self.convs.append(GCNConv(a, b, cached=False))
            elif kind == "sage":
                self.convs.append(SAGEConv(a, b, aggr=aggr))
            elif kind == "gat":
                last = i == layers - 1
                heads = 1 if last else GAT_HEADS
                self.convs.append(GATConv(a, b // heads if not last else b, heads=heads, dropout=0.2))

    def forward(self, x, edge_index):
        hidden = x
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x) if isinstance(conv, GATConv) else F.relu(x)
                hidden = x
        return x, hidden


def build(kind, n_in, n_out, layers=GNN_LAYERS, aggr="mean", dropout=GNN_DROPOUT):
    if kind == "mlp":
        return MLP(n_in, n_out, layers, GNN_HIDDEN, dropout)
    return GNN(kind, n_in, n_out, layers, GNN_HIDDEN, dropout, aggr=aggr)


def class_weights(y, train, n_classes):
    counts = np.bincount(y[train], minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights)


def synchronise():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def train_model(kind, x, edge_index, y, split, seed, layers=GNN_LAYERS, aggr="mean", dropout=GNN_DROPOUT):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, val, test = split
    n_classes = int(y.max()) + 1
    model = build(kind, x.shape[1], n_classes, layers=layers, aggr=aggr, dropout=dropout).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=GNN_WEIGHT_DECAY)
    xt = torch.tensor(x, device=DEVICE)
    ei = torch.tensor(edge_index, device=DEVICE)
    yt = torch.tensor(y, device=DEVICE)
    weights = class_weights(y, train, n_classes).to(DEVICE)
    tr = torch.tensor(train, device=DEVICE)
    best, best_state, best_epoch, waited = -1.0, None, 0, 0
    start = time.perf_counter()
    epochs = 0
    for epoch in range(GNN_MAX_EPOCHS):
        epochs = epoch + 1
        model.train()
        optimiser.zero_grad()
        logits, _ = model(xt, ei)
        loss = F.cross_entropy(logits[tr], yt[tr], weight=weights)
        loss.backward()
        optimiser.step()
        model.eval()
        with torch.no_grad():
            pred = model(xt, ei)[0].argmax(1).cpu().numpy()
        score = macro(y[val], pred[val])
        if score > best:
            best, best_epoch, waited = score, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= GNN_PATIENCE:
                break
    synchronise()
    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        logits, hidden = model(xt, ei)
    synchronise()
    infer_seconds = time.perf_counter() - start
    pred = logits.argmax(1).cpu().numpy()
    return {
        "pred": pred,
        "hidden": hidden.cpu().numpy(),
        "val_macro_f1": best,
        "best_epoch": best_epoch + 1,
        "epochs": epochs,
        "train_seconds": train_seconds,
        "seconds_per_epoch": train_seconds / epochs,
        "infer_seconds": infer_seconds,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def logistic_baseline(x, y, split, seed):
    train, val, test = split
    start = time.perf_counter()
    model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    model.fit(x[train], y[train])
    seconds = time.perf_counter() - start
    pred = np.full(len(y), -1)
    pred[y >= 0] = model.predict(x[y >= 0])
    labelled = np.flatnonzero(y >= 0)
    full = np.zeros(len(y), dtype=np.int64)
    full[labelled] = pred[labelled]
    return {"pred": model.predict(x), "train_seconds": seconds, "epochs": 1, "best_epoch": 1,
            "seconds_per_epoch": seconds, "infer_seconds": 0.0,
            "parameters": model.coef_.size + model.intercept_.size,
            "val_macro_f1": macro(y[val], model.predict(x[val]))}


def label_propagation(edge_index, y, split):
    train, val, test = split
    start = time.perf_counter()
    model = LabelPropagation(num_layers=30, alpha=0.9)
    mask = torch.zeros(len(y), dtype=torch.bool, device=DEVICE)
    mask[torch.tensor(train, device=DEVICE)] = True
    yt = torch.tensor(np.where(y >= 0, y, 0), device=DEVICE)
    out = model(yt, torch.tensor(edge_index, device=DEVICE), mask=mask)
    synchronise()
    seconds = time.perf_counter() - start
    pred = out.argmax(1).cpu().numpy()
    return {"pred": pred, "train_seconds": seconds, "epochs": 30, "best_epoch": 30,
            "seconds_per_epoch": seconds / 30, "infer_seconds": seconds, "parameters": 0,
            "val_macro_f1": macro(y[val], pred[val])}


def mean_average_distance(hidden, edge_index, seed, sample=3000):
    rng = np.random.default_rng(seed)
    h = hidden / np.maximum(np.linalg.norm(hidden, axis=1, keepdims=True), 1e-9)
    idx = rng.choice(len(h), size=min(sample, len(h)), replace=False)
    sub = h[idx]
    global_mad = float(1.0 - (sub @ sub.T)[np.triu_indices(len(sub), 1)].mean())
    src, dst = edge_index
    neighbour_mad = float(1.0 - (h[src] * h[dst]).sum(axis=1).mean())
    return global_mad, neighbour_mad


def evaluate(result, y, split, names, extra):
    test = split[2]
    row = dict(extra)
    row.update(classification_scores(y[test], result["pred"][test]))
    row.update(per_class_f1(y[test], result["pred"][test], list(range(len(names))), names))
    for key in ("val_macro_f1", "best_epoch", "epochs", "train_seconds", "seconds_per_epoch",
                "infer_seconds", "parameters"):
        row[key] = result[key]
    return row


def drop_edges(edge_index, p, seed):
    if p <= 0:
        return edge_index
    torch.manual_seed(seed)
    ei = torch.tensor(edge_index)
    upper = ei[:, ei[0] < ei[1]]
    kept, _ = dropout_edge(upper, p=p, force_undirected=False, training=True)
    return torch.cat([kept, kept.flip(0)], dim=1).numpy()


def add_noise_edges(edge_index, ratio, n, seed):
    if ratio <= 0:
        return edge_index
    rng = np.random.default_rng(seed)
    count = int(ratio * edge_index.shape[1] / 2)
    src = rng.integers(0, n, count)
    dst = rng.integers(0, n, count)
    keep = src != dst
    noise = np.stack([src[keep], dst[keep]])
    combined = np.concatenate([edge_index, noise, noise[::-1]], axis=1)
    return np.unique(combined, axis=1)


def short(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def plot_comparison(table, names):
    fig, axes = viz.plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    order = table.sort_values("macro_f1_mean")
    colours = [COLOURS[m] for m in order["model"]]
    bars = ax.barh(order["model"], order["macro_f1_mean"], xerr=order["macro_f1_std"], color=colours,
                   ecolor=viz.INK_MUTED, capsize=3)
    for bar, value in zip(bars, order["macro_f1_mean"]):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center",
                fontsize=7.4, color=viz.INK_SECONDARY)
    ax.set_xlabel("test macro-F1, mean of three seeds")
    ax.set_xlim(0, 1)
    ax.set_title("Features only, structure only, and both")
    ax.grid(axis="y", visible=False)
    viz.strip_spines(ax)

    ax = axes[1]
    keep = ["Logistic regression", "MLP", "Label propagation", "GCN", "GraphSAGE", "GAT"]
    sub = table.set_index("model").loc[[m for m in keep if m in set(table["model"])]]
    width = 0.8 / len(sub)
    x = np.arange(len(names))
    for i, (model, row) in enumerate(sub.iterrows()):
        values = [row[f"f1_{short(n)}_mean"] for n in names]
        ax.bar(x + i * width - 0.4 + width / 2, values, width, color=COLOURS[model], label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("test F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-department F1")
    ax.legend(ncol=3, fontsize=7, loc="upper right")
    ax.grid(axis="x", visible=False)
    viz.strip_spines(ax)
    fig.tight_layout()
    return viz.save(fig, FIGURE_DIR / "t3_model_comparison.png")


def plot_depth(depth):
    fig, axes = viz.plt.subplots(1, 2, figsize=(10.0, 3.5))
    for kind in ["GCN", "GraphSAGE"]:
        sub = depth[depth["model"] == kind]
        axes[0].errorbar(sub["layers"], sub["macro_f1_mean"], yerr=sub["macro_f1_std"], marker="o",
                         color=COLOURS[kind], capsize=3, label=kind)
        sub = sub[sub["layers"] > 1]
        axes[1].plot(sub["layers"], sub["global_mad_mean"], marker="o", color=COLOURS[kind], label=f"{kind}, all pairs")
        axes[1].plot(sub["layers"], sub["neighbour_mad_mean"], marker="s", linestyle="--", color=COLOURS[kind],
                     label=f"{kind}, neighbours")
    mlp = depth[depth["model"] == "MLP"]
    axes[0].errorbar(mlp["layers"], mlp["macro_f1_mean"], yerr=mlp["macro_f1_std"], marker="^",
                     color=COLOURS["MLP"], capsize=3, label="MLP")
    axes[0].set_xlabel("layers")
    axes[0].set_ylabel("test macro-F1")
    axes[0].set_title("Depth and test macro-F1")
    axes[0].legend()
    axes[1].set_xlabel("layers (two or more, so a hidden layer exists)")
    axes[1].set_ylabel("mean cosine distance of final hidden layer")
    axes[1].set_title("Representation spread shrinks with depth")
    axes[1].legend(fontsize=7)
    axes[0].set_xticks(sorted(depth["layers"].unique()))
    axes[1].set_xticks(sorted(l for l in depth["layers"].unique() if l > 1))
    for ax in axes:
        viz.strip_spines(ax)
    fig.tight_layout()
    return viz.save(fig, FIGURE_DIR / "t3_depth.png")


def plot_degree(degree_table):
    fig, ax = viz.plt.subplots(figsize=(6.6, 3.4))
    models = degree_table["model"].unique()
    buckets = degree_table["bucket"].unique()
    x = np.arange(len(buckets))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        sub = degree_table[degree_table["model"] == model].set_index("bucket").loc[buckets]
        ax.bar(x + i * width - 0.4 + width / 2, sub["macro_f1"], width, color=COLOURS[model], label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, fontsize=7.4)
    ax.set_xlabel("node degree quartile")
    ax.set_ylabel("test macro-F1, pooled seeds")
    ax.set_title("Accuracy by node degree")
    ax.legend(ncol=4, fontsize=7)
    ax.grid(axis="x", visible=False)
    viz.strip_spines(ax)
    return viz.save(fig, FIGURE_DIR / "t3_degree.png")


def plot_robustness(robust, mlp_score):
    fig, axes = viz.plt.subplots(1, 2, figsize=(10.0, 3.4))
    for ax, kind, xlabel, title in (
        (axes[0], "drop", "share of edges removed", "Sparser graph"),
        (axes[1], "noise", "random edges added per real edge", "Noisier graph"),
    ):
        sub = robust[robust["perturbation"] == kind]
        for i, model in enumerate(sub["model"].unique()):
            part = sub[sub["model"] == model]
            ax.errorbar(part["level"], part["macro_f1_mean"], yerr=part["macro_f1_std"], marker="o",
                        color=COLOURS[model], capsize=3, label=model)
        ax.axhline(mlp_score, color=COLOURS["MLP"], linestyle="--", linewidth=1.2, label="MLP, no graph")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("test macro-F1")
        ax.set_title(title)
        ax.legend(fontsize=7)
        viz.strip_spines(ax)
    fig.tight_layout()
    return viz.save(fig, FIGURE_DIR / "t3_robustness.png")


def plot_density(density):
    fig, ax = viz.plt.subplots(figsize=(6.4, 3.3))
    for i, model in enumerate(density["model"].unique()):
        part = density[density["model"] == model]
        ax.errorbar(part["k"], part["macro_f1_mean"], yerr=part["macro_f1_std"], marker="o",
                    color=COLOURS[model], capsize=3, label=model,
                    linestyle="--" if model == "MLP" else "-")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(sorted(density["k"].unique()))
    ax.set_xticklabels(sorted(density["k"].unique()))
    ax.set_xlabel("neighbours kept per product (k)")
    ax.set_ylabel("test macro-F1")
    ax.set_title("Graph density and node classification")
    ax.legend(fontsize=7)
    viz.strip_spines(ax)
    return viz.save(fig, FIGURE_DIR / "t3_density.png")


def run():
    ds = graph_dataset()
    y, names, departments = ds["y"], ds["classes"], ds["departments"]
    features = ds["features"].to_numpy()
    n = len(y)
    base_edges = knn_graph(ds["edges"], n, GRAPH_K)

    stats_rows = []
    for k in K_SWEEP:
        stats = graph_statistics(knn_graph(ds["edges"], n, k), n, y, departments)
        stats["k"] = k
        stats_rows.append(stats)
    graph_stats = pd.DataFrame(stats_rows)
    write_table(graph_stats, TABLE_DIR / "t3_graph_statistics.csv")

    class_table = pd.DataFrame({"department": names,
                                "nodes": [(y == i).sum() for i in range(len(names))]})
    class_table["share"] = class_table["nodes"] / (y >= 0).sum()
    write_table(class_table, TABLE_DIR / "t3_classes.csv")

    rows, predictions = [], {}
    for seed in REPEAT_SEEDS:
        split = node_split(y, seed)
        x = scaled_features(features, split[0])
        runs = {
            "Logistic regression": logistic_baseline(x, y, split, seed),
            "MLP": train_model("mlp", x, base_edges, y, split, seed),
            "Label propagation": label_propagation(base_edges, y, split),
            "GCN": train_model("gcn", x, base_edges, y, split, seed),
            "GraphSAGE": train_model("sage", x, base_edges, y, split, seed),
            "GAT": train_model("gat", x, base_edges, y, split, seed),
        }
        for model, result in runs.items():
            rows.append(evaluate(result, y, split, [short(c) for c in names], {"model": model, "seed": seed}))
            predictions[(model, seed)] = (result["pred"], split)
    raw = pd.DataFrame(rows)
    write_table(raw, TABLE_DIR / "t3_raw_runs.csv")
    values = METRICS + [c for c in raw.columns if c.startswith("f1_")] + [
        "val_macro_f1", "best_epoch", "epochs", "train_seconds", "seconds_per_epoch", "infer_seconds", "parameters"]
    comparison = aggregate(raw, ["model"], values)
    write_table(comparison, TABLE_DIR / "t3_model_comparison.csv")
    plot_comparison(comparison, names)
    mlp_score = comparison.loc[comparison["model"] == "MLP", "macro_f1_mean"].iloc[0]

    degree = np.bincount(base_edges[0], minlength=n)
    labelled_degree = degree[y >= 0]
    edges_q = np.quantile(labelled_degree, [0.25, 0.5, 0.75])
    bucket_names = [f"Q1 (<= {edges_q[0]:.0f})", f"Q2 (<= {edges_q[1]:.0f})",
                    f"Q3 (<= {edges_q[2]:.0f})", f"Q4 (> {edges_q[2]:.0f})"]
    bucket = np.digitize(degree, edges_q, right=True)
    degree_rows = []
    for model in ("MLP", "Label propagation", "GCN", "GraphSAGE"):
        for b, label in enumerate(bucket_names):
            truth, pred = [], []
            for seed in REPEAT_SEEDS:
                p, split = predictions[(model, seed)]
                test = split[2]
                pick = test[bucket[test] == b]
                truth.append(y[pick])
                pred.append(p[pick])
            truth, pred = np.concatenate(truth), np.concatenate(pred)
            row = {"model": model, "bucket": label, "nodes": len(truth) // len(REPEAT_SEEDS)}
            row.update(classification_scores(truth, pred))
            degree_rows.append(row)
    degree_table = pd.DataFrame(degree_rows)
    write_table(degree_table, TABLE_DIR / "t3_degree.csv")
    plot_degree(degree_table)

    depth_rows = []
    for seed in REPEAT_SEEDS:
        split = node_split(y, seed)
        x = scaled_features(features, split[0])
        for layers in DEPTH_SWEEP:
            for kind, label in (("gcn", "GCN"), ("sage", "GraphSAGE"), ("mlp", "MLP")):
                if kind == "mlp" and layers > 4:
                    continue
                result = train_model(kind, x, base_edges, y, split, seed, layers=layers)
                row = evaluate(result, y, split, [short(c) for c in names],
                               {"model": label, "layers": layers, "seed": seed})
                row["global_mad"], row["neighbour_mad"] = mean_average_distance(result["hidden"], base_edges, seed)
                depth_rows.append(row)
    depth = aggregate(pd.DataFrame(depth_rows), ["model", "layers"],
                      METRICS + ["global_mad", "neighbour_mad", "train_seconds", "parameters"])
    write_table(depth, TABLE_DIR / "t3_depth.csv")
    plot_depth(depth)

    agg_rows = []
    for seed in REPEAT_SEEDS:
        split = node_split(y, seed)
        x = scaled_features(features, split[0])
        for aggr in AGGREGATIONS:
            result = train_model("sage", x, base_edges, y, split, seed, aggr=aggr)
            agg_rows.append(evaluate(result, y, split, [short(c) for c in names],
                                     {"aggregation": aggr, "seed": seed}))
    aggregation = aggregate(pd.DataFrame(agg_rows), ["aggregation"], METRICS + ["train_seconds"])
    write_table(aggregation, TABLE_DIR / "t3_aggregation.csv")

    density_rows = []
    for k in K_SWEEP:
        edge_index = knn_graph(ds["edges"], n, k)
        for seed in REPEAT_SEEDS:
            split = node_split(y, seed)
            x = scaled_features(features, split[0])
            for kind, label in (("gcn", "GCN"), ("sage", "GraphSAGE")):
                result = train_model(kind, x, edge_index, y, split, seed)
                density_rows.append(evaluate(result, y, split, [short(c) for c in names],
                                             {"model": label, "k": k, "seed": seed}))
    density = aggregate(pd.DataFrame(density_rows), ["model", "k"], METRICS + ["train_seconds"])
    density = pd.concat([density, pd.DataFrame([{"model": "MLP", "k": k, "macro_f1_mean": mlp_score,
                                                  "macro_f1_std": 0.0} for k in K_SWEEP])], ignore_index=True)
    write_table(density, TABLE_DIR / "t3_density.csv")
    plot_density(density)

    robust_rows = []
    for seed in REPEAT_SEEDS:
        split = node_split(y, seed)
        x = scaled_features(features, split[0])
        for kind, label in (("gcn", "GCN"), ("sage", "GraphSAGE")):
            for level in EDGE_DROP_LEVELS:
                result = train_model(kind, x, drop_edges(base_edges, level, seed), y, split, seed)
                robust_rows.append(evaluate(result, y, split, [short(c) for c in names],
                                            {"model": label, "perturbation": "drop", "level": level, "seed": seed}))
            for level in EDGE_NOISE_LEVELS[1:]:
                result = train_model(kind, x, add_noise_edges(base_edges, level, n, seed), y, split, seed)
                robust_rows.append(evaluate(result, y, split, [short(c) for c in names],
                                            {"model": label, "perturbation": "noise", "level": level, "seed": seed}))
    robust = pd.DataFrame(robust_rows)
    clean = robust[(robust["perturbation"] == "drop") & (robust["level"] == 0)].copy()
    clean["perturbation"] = "noise"
    robust = pd.concat([robust, clean], ignore_index=True)
    robust = aggregate(robust, ["model", "perturbation", "level"], METRICS).sort_values(
        ["model", "perturbation", "level"])
    write_table(robust, TABLE_DIR / "t3_robustness.csv")
    plot_robustness(robust, mlp_score)

    print("graph statistics")
    print(graph_stats.round(4).to_string(index=False))
    print()
    print("model comparison")
    print(comparison[["model"] + [f"{m}_mean" for m in METRICS] + ["macro_f1_std", "train_seconds_mean",
                                                                     "parameters_mean"]].round(4).to_string(index=False))
    print()
    print("degree")
    print(degree_table.round(4).to_string(index=False))
    print()
    print("depth")
    print(depth[["model", "layers", "macro_f1_mean", "macro_f1_std", "global_mad_mean",
                 "neighbour_mad_mean"]].round(4).to_string(index=False))
    print()
    print("aggregation")
    print(aggregation[["aggregation", "macro_f1_mean", "macro_f1_std", "accuracy_mean"]].round(4).to_string(index=False))
    print()
    print("density")
    print(density[["model", "k", "macro_f1_mean", "macro_f1_std"]].round(4).to_string(index=False))
    print()
    print("robustness")
    print(robust[["model", "perturbation", "level", "macro_f1_mean", "macro_f1_std"]].round(4).to_string(index=False))
    return {"comparison": comparison, "depth": depth, "robust": robust}


if __name__ == "__main__":
    run()
