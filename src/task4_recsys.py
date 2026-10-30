import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch import nn

import viz
from config import (
    DEVICE,
    EMBED_DIM,
    FIGURE_DIR,
    IPS_CLIP,
    IPS_POWER,
    ITEMKNN_NEIGHBOURS,
    LIGHTGCN_LAYERS,
    RANDOM_SEED,
    REC_BATCH,
    REC_EPOCHS,
    REC_EVAL_EVERY,
    REC_LR,
    REC_WEIGHT_DECAY,
    REPEAT_SEEDS,
    TABLE_DIR,
    TOP_K,
    TRAIN_LAST_WEEK,
)
from data import catalogue, exposure_rates, load_products, recommendation_splits
from evaluation import aggregate, per_user_recall, ranking_scores, write_table

warnings.filterwarnings("ignore")
torch.set_num_threads(16)

PROTOCOLS = ("all items", "new items")
COLOURS = {
    "Popularity": viz.SERIES[0],
    "Repeat purchase": viz.SERIES[1],
    "Content-based (commodity)": viz.SERIES[2],
    "ItemKNN": viz.SERIES[3],
    "BPR-MF": viz.SERIES[4],
    "LightGCN": viz.SERIES[5],
    "LightGCN + IPS": viz.SERIES[6],
}
MAX_K = max(TOP_K)


def binary(matrix):
    out = matrix.copy().tocsr()
    out.data[:] = 1.0
    return out


def popularity_scores(train):
    pop = np.asarray(binary(train).sum(axis=0)).ravel().astype(np.float32)
    return np.tile(pop, (train.shape[0], 1))


def repeat_scores(train):
    pop = np.asarray(binary(train).sum(axis=0)).ravel()
    tie = pop / (pop.max() * 10.0)
    dense = train.toarray().astype(np.float32)
    return dense + tie[None, :] * (dense > 0) + tie[None, :] * 1e-3


def itemknn_scores(train, neighbours=ITEMKNN_NEIGHBOURS):
    x = binary(train)
    xn = normalize(x.T.tocsr(), axis=1)
    sim = (xn @ xn.T).toarray().astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    cut = np.argpartition(-sim, neighbours, axis=1)[:, neighbours:]
    np.put_along_axis(sim, cut, 0.0, axis=1)
    return np.asarray(x @ sim, dtype=np.float32)


def content_scores(train, commodity):
    x = binary(train)
    codes, uniques = pd.factorize(commodity)
    onehot = sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)),
                           shape=(len(codes), len(uniques)))
    profile = normalize(x @ onehot, norm="l1", axis=1)
    pop = np.asarray(x.sum(axis=0)).ravel()
    within = pd.Series(pop).groupby(codes).transform(lambda s: s / s.max()).to_numpy()
    scores = (profile @ onehot.T).toarray()
    return (scores * (0.5 + 0.5 * within[None, :])).astype(np.float32)


def propensities(train):
    pop = np.asarray(binary(train).sum(axis=0)).ravel().astype(np.float64)
    p = (pop / pop.max()) ** IPS_POWER
    return np.clip(p, IPS_CLIP, 1.0)


class SymmetricPropagation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, adjacency, h):
        ctx.adjacency = adjacency
        return adjacency @ h

    @staticmethod
    def backward(ctx, grad):
        return None, ctx.adjacency @ grad


class Embeddings(nn.Module):
    def __init__(self, n_users, n_items, dim, adjacency=None, layers=0):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)
        self.adjacency = adjacency
        self.layers = layers
        self.n_users = n_users

    def propagate(self):
        base = torch.cat([self.user.weight, self.item.weight])
        if self.layers == 0:
            return self.user.weight, self.item.weight
        out, h = [base], base
        for _ in range(self.layers):
            h = SymmetricPropagation.apply(self.adjacency, h)
            out.append(h)
        final = torch.stack(out).mean(0)
        return final[: self.n_users], final[self.n_users:]


def normalised_adjacency(train):
    x = binary(train).tocoo()
    n_users, n_items = x.shape
    rows = np.concatenate([x.row, x.col + n_users])
    cols = np.concatenate([x.col + n_users, x.row])
    degree = np.bincount(rows, minlength=n_users + n_items).astype(np.float32)
    inv = 1.0 / np.sqrt(np.maximum(degree, 1))
    values = inv[rows] * inv[cols]
    index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(index, torch.tensor(values, dtype=torch.float32),
                                   (n_users + n_items, n_users + n_items)).coalesce().to_sparse_csr().to(DEVICE)


def train_embeddings(train, kind, seed, epochs, weights=None, monitor=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = binary(train).tocoo()
    users, items = x.row.astype(np.int64), x.col.astype(np.int64)
    n_users, n_items = train.shape
    adjacency = normalised_adjacency(train) if kind == "lightgcn" else None
    layers = LIGHTGCN_LAYERS if kind == "lightgcn" else 0
    model = Embeddings(n_users, n_items, EMBED_DIM, adjacency, layers).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=REC_LR)
    pair_weight = np.ones(len(users), dtype=np.float32) if weights is None else weights[items].astype(np.float32)
    pair_weight = pair_weight / pair_weight.mean()
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(users))
        for begin in range(0, len(order), REC_BATCH):
            batch = order[begin: begin + REC_BATCH]
            u = torch.tensor(users[batch], device=DEVICE)
            i = torch.tensor(items[batch], device=DEVICE)
            j = torch.tensor(rng.integers(0, n_items, len(batch)), device=DEVICE)
            w = torch.tensor(pair_weight[batch], device=DEVICE)
            user_emb, item_emb = model.propagate()
            eu, ei, ej = user_emb[u], item_emb[i], item_emb[j]
            margin = (eu * ei).sum(1) - (eu * ej).sum(1)
            loss = (w * F.softplus(-margin)).mean()
            reg = (model.user.weight[u].pow(2).sum(1) + model.item.weight[i].pow(2).sum(1)
                   + model.item.weight[j].pow(2).sum(1)).mean()
            loss = loss + REC_WEIGHT_DECAY * reg
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        if monitor is not None and epoch % REC_EVAL_EVERY == 0:
            history.append({"epoch": epoch, "score": monitor(embedding_scores(model))})
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return model, seconds, history


def embedding_scores(model):
    with torch.no_grad():
        user_emb, item_emb = model.propagate()
        return (user_emb @ item_emb.T).cpu().numpy()


def top_items(scores, history, protocol, k=MAX_K):
    scores = scores.copy()
    if protocol == "new items":
        scores[history.nonzero()] = -np.inf
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(top, order, axis=1)


def truth_sets(target, history, protocol):
    truth = []
    for u in range(target.shape[0]):
        bought = set(target.indices[target.indptr[u]: target.indptr[u + 1]])
        if protocol == "new items":
            bought -= set(history.indices[history.indptr[u]: history.indptr[u + 1]])
        truth.append(bought)
    return truth


def beyond_accuracy(top, k, pop, commodity, mailer, p):
    rec = top[:, :k]
    n_items = len(pop)
    users = pop.max()
    counts = np.bincount(rec.ravel(), minlength=n_items).astype(float)
    sorted_counts = np.sort(counts)
    cum = np.cumsum(sorted_counts)
    gini = 1 - 2 * (cum / cum[-1]).sum() / n_items + 1 / n_items
    head = pop >= np.quantile(pop, 0.8)
    distinct = np.array([len(set(commodity[row])) for row in rec]) / k
    return {
        f"coverage@{k}": (counts > 0).mean(),
        f"novelty@{k}": float(-np.log2(pop[rec] / users).mean()),
        f"diversity@{k}": float(distinct.mean()),
        f"long_tail_share@{k}": float((~head[rec]).mean()),
        f"exposure_gini@{k}": float(gini),
        f"mailer_rate@{k}": float(mailer[rec].mean()),
    }


def ips_recall(top, truth, k, p):
    weights = 1.0 / p
    values = []
    for row, t in zip(top, truth):
        if not t:
            continue
        t_list = np.fromiter(t, dtype=int)
        hit = np.intersect1d(row[:k], t_list)
        values.append(weights[hit].sum() / weights[t_list].sum())
    return float(np.mean(values))


def evaluate_scores(name, scores, history, target, users, protocol, extras, seed, seconds):
    top = top_items(scores[users], history[users], protocol)
    truth = truth_sets(target[users], history[users], protocol)
    keep = np.array([len(t) > 0 for t in truth])
    top, truth = top[keep], [t for t in truth if t]
    row = {"model": name, "protocol": protocol, "seed": seed, "users": len(truth), "fit_seconds": seconds}
    for k in TOP_K:
        row.update(ranking_scores(top, truth, k))
    row.update(beyond_accuracy(top, 10, extras["pop"], extras["commodity"], extras["mailer"], extras["p"]))
    row["ips_recall@10"] = ips_recall(top, truth, 10, extras["p"])
    return row, top, truth, np.asarray(users)[keep]


def val_monitor(history, target, users):
    def score(scores):
        top = top_items(scores[users], history[users], "all items", k=10)
        truth = truth_sets(target[users], history[users], "all items")
        keep = [i for i, t in enumerate(truth) if t]
        return ranking_scores(top[keep], [truth[i] for i in keep], 10)["ndcg@10"]
    return score


def plot_main(table):
    fig, axes = viz.plt.subplots(1, 2, figsize=(10.4, 3.8))
    for ax, protocol in zip(axes, PROTOCOLS):
        sub = table[table["protocol"] == protocol].sort_values("ndcg@10_mean")
        y = np.arange(len(sub))
        ax.barh(y - 0.2, sub["ndcg@10_mean"], 0.4, xerr=sub["ndcg@10_std"], color=viz.SERIES[0],
                ecolor=viz.INK_MUTED, capsize=2, label="NDCG@10")
        ax.barh(y + 0.2, sub["recall@20_mean"], 0.4, xerr=sub["recall@20_std"], color=viz.SERIES[2],
                ecolor=viz.INK_MUTED, capsize=2, label="Recall@20")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["model"], fontsize=7.4)
        ax.set_title(f"Test weeks 95 to 102, {protocol} protocol")
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(axis="y", visible=False)
        viz.strip_spines(ax)
    fig.tight_layout()
    return viz.save(fig, FIGURE_DIR / "t4_ranking.png")


def plot_tradeoff(table):
    fig, axes = viz.plt.subplots(1, 2, figsize=(10.4, 3.7))
    sub = table[table["protocol"] == "new items"]
    for ax, metric, label in ((axes[0], "coverage@10_mean", "catalogue coverage of top-10 lists"),
                              (axes[1], "novelty@10_mean", "novelty of top-10 (bits)")):
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.scatter(row[metric], row["ndcg@10_mean"], color=COLOURS[row["model"]], s=46,
                       zorder=3, label=row["model"])
        ax.set_xlabel(label)
        ax.set_ylabel("NDCG@10, new items")
        viz.strip_spines(ax)
    axes[1].legend(fontsize=7, loc="center right")
    axes[0].set_title("Accuracy against coverage")
    axes[1].set_title("Accuracy against novelty")
    fig.tight_layout()
    return viz.save(fig, FIGURE_DIR / "t4_tradeoff.png")


def plot_coldstart(cold):
    fig, axes = viz.plt.subplots(1, 2, figsize=(10.4, 3.5))
    for ax, protocol in zip(axes, PROTOCOLS):
        sub = cold[cold["protocol"] == protocol]
        groups = sub["bucket"].unique()
        models = sub["model"].unique()
        x = np.arange(len(groups))
        width = 0.8 / len(models)
        for i, model in enumerate(models):
            part = sub[sub["model"] == model].set_index("bucket").loc[groups]
            ax.bar(x + i * width - 0.4 + width / 2, part["recall@10"], width,
                   color=COLOURS[model], label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=7)
        ax.set_xlabel("training history size (distinct products)")
        ax.set_ylabel("Recall@10")
        ax.set_title(f"Household history and accuracy, {protocol}")
        ax.grid(axis="x", visible=False)
        viz.strip_spines(ax)
    axes[1].text(0.98, 0.95, "repeat purchase is undefined\nwhen history is excluded",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=7, color=viz.INK_MUTED)
    handles = [viz.plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOURS.values()]
    fig.legend(handles, list(COLOURS), loc="lower center", ncol=len(COLOURS), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return viz.save(fig, FIGURE_DIR / "t4_coldstart.png")


def plot_convergence(curves):
    fig, ax = viz.plt.subplots(figsize=(6.4, 3.2))
    for i, model in enumerate(curves["model"].unique()):
        part = curves[curves["model"] == model]
        ax.plot(part["epoch"], part["score"], marker="o", color=COLOURS[model], label=model)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NDCG@10")
    ax.set_title("Validation curves, weeks 87 to 94")
    ax.legend(fontsize=7)
    viz.strip_spines(ax)
    return viz.save(fig, FIGURE_DIR / "t4_convergence.png")


def run():
    items = catalogue()
    data = recommendation_splits(items)
    products = load_products().reindex(items)
    commodity = products["COMMODITY_DESC"].to_numpy()
    mailer = exposure_rates(items, TRAIN_LAST_WEEK)["mailer_rate"].to_numpy()

    train, val, train_full, test = data["train"], data["val"], data["train_full"], data["test"]
    val_users = np.flatnonzero((np.diff(train.indptr) > 0) & (np.diff(val.indptr) > 0))
    test_users = np.flatnonzero((np.diff(train_full.indptr) > 0) & (np.diff(test.indptr) > 0))

    stats = pd.DataFrame([{
        "households": train.shape[0], "items": train.shape[1],
        "train_pairs": binary(train).nnz, "train_full_pairs": binary(train_full).nnz,
        "val_pairs": binary(val).nnz, "test_pairs": binary(test).nnz,
        "train_density": binary(train_full).nnz / (train.shape[0] * train.shape[1]),
        "val_users": len(val_users), "test_users": len(test_users),
        "test_repeat_share": binary(test).multiply(binary(train_full)).nnz / binary(test).nnz,
        "median_history": float(np.median(np.diff(binary(train_full).indptr)[test_users])),
    }])
    write_table(stats, TABLE_DIR / "t4_split_statistics.csv")

    curves = []
    chosen = {}
    for kind, label, weights in (("mf", "BPR-MF", None), ("lightgcn", "LightGCN", None),
                                 ("lightgcn", "LightGCN + IPS", propensities(train))):
        w = None if weights is None else 1.0 / weights
        _, _, history = train_embeddings(train, kind, RANDOM_SEED, REC_EPOCHS, weights=w,
                                         monitor=val_monitor(train, val, val_users))
        frame = pd.DataFrame(history)
        frame["model"] = label
        curves.append(frame)
        chosen[label] = int(frame.loc[frame["score"].idxmax(), "epoch"])
    curves = pd.concat(curves, ignore_index=True)
    write_table(curves, TABLE_DIR / "t4_convergence.csv")
    plot_convergence(curves)

    pop_full = np.asarray(binary(train_full).sum(axis=0)).ravel().astype(float)
    extras = {"pop": np.maximum(pop_full, 1), "commodity": commodity, "mailer": mailer,
              "p": propensities(train_full)}

    rows, tops = [], {}
    fixed = {}
    for name, fn in (("Popularity", lambda: popularity_scores(train_full)),
                     ("Repeat purchase", lambda: repeat_scores(train_full)),
                     ("Content-based (commodity)", lambda: content_scores(train_full, commodity)),
                     ("ItemKNN", lambda: itemknn_scores(train_full))):
        start = time.perf_counter()
        fixed[name] = fn()
        fixed[name + "_seconds"] = time.perf_counter() - start
    for seed in REPEAT_SEEDS:
        learned = {}
        for kind, label, weights in (("mf", "BPR-MF", None), ("lightgcn", "LightGCN", None),
                                     ("lightgcn", "LightGCN + IPS", 1.0 / extras["p"])):
            model, seconds, _ = train_embeddings(train_full, kind, seed, chosen[label], weights=weights)
            learned[label] = (embedding_scores(model), seconds)
        candidates = {name: (fixed[name], fixed[name + "_seconds"]) for name in
                      ("Popularity", "Repeat purchase", "Content-based (commodity)", "ItemKNN")}
        candidates.update(learned)
        for name, (scores, seconds) in candidates.items():
            for protocol in PROTOCOLS:
                if name == "Repeat purchase" and protocol == "new items":
                    continue
                row, top, truth, kept = evaluate_scores(name, scores, train_full, test, test_users,
                                                        protocol, extras, seed, seconds)
                rows.append(row)
                if seed == RANDOM_SEED:
                    tops[(name, protocol)] = (top, truth, kept)
    raw = pd.DataFrame(rows)
    write_table(raw, TABLE_DIR / "t4_raw_runs.csv")
    metrics = [c for c in raw.columns if "@" in c] + ["fit_seconds"]
    table = aggregate(raw, ["model", "protocol"], metrics)
    write_table(table, TABLE_DIR / "t4_model_performance.csv")
    plot_main(table)
    plot_tradeoff(table)

    history_size = np.diff(binary(train_full).indptr)
    edges = np.quantile(history_size[test_users], [0.25, 0.5, 0.75])
    labels = [f"<= {edges[0]:.0f}", f"{edges[0]:.0f} to {edges[1]:.0f}",
              f"{edges[1]:.0f} to {edges[2]:.0f}", f"> {edges[2]:.0f}"]
    cold_rows = []
    for (name, protocol), (top, truth, kept) in tops.items():
        recall = per_user_recall(top, truth, 10)
        bucket = np.digitize(history_size[kept], edges, right=True)
        for b, label in enumerate(labels):
            mask = bucket == b
            cold_rows.append({"model": name, "protocol": protocol, "bucket": label,
                              "users": int(mask.sum()), "recall@10": recall[mask].mean()})
    cold = pd.DataFrame(cold_rows)
    write_table(cold, TABLE_DIR / "t4_coldstart.csv")
    plot_coldstart(cold)

    item_rows = []
    test_pairs = binary(test)[test_users].tocoo()
    item_pop = pop_full[test_pairs.col]
    tail_cut = np.quantile(pop_full, 0.8)
    for (name, protocol), (top, truth, kept) in tops.items():
        head_hits = tail_hits = head_total = tail_total = 0
        for row, t in zip(top, truth):
            rec = set(row[:10])
            for item in t:
                if pop_full[item] >= tail_cut:
                    head_total += 1
                    head_hits += item in rec
                else:
                    tail_total += 1
                    tail_hits += item in rec
        item_rows.append({"model": name, "protocol": protocol,
                          "head_recall_share": head_hits / max(head_total, 1),
                          "tail_recall_share": tail_hits / max(tail_total, 1),
                          "tail_share_of_test": tail_total / max(head_total + tail_total, 1)})
    items_table = pd.DataFrame(item_rows)
    write_table(items_table, TABLE_DIR / "t4_item_popularity.csv")

    chosen_table = pd.DataFrame([{"model": k, "epochs": v} for k, v in chosen.items()])
    write_table(chosen_table, TABLE_DIR / "t4_selected_epochs.csv")

    print("split statistics")
    print(stats.T.round(4).to_string())
    print()
    print("selected epochs", chosen)
    print()
    cols = ["model", "protocol", "precision@10_mean", "recall@10_mean", "ndcg@10_mean", "map@10_mean",
            "recall@20_mean", "ndcg@20_mean", "coverage@10_mean", "novelty@10_mean", "diversity@10_mean",
            "mailer_rate@10_mean", "ips_recall@10_mean", "fit_seconds_mean"]
    print(table[cols].round(4).to_string(index=False))
    print()
    print(cold.round(4).to_string(index=False))
    print()
    print(items_table.round(4).to_string(index=False))
    return {"table": table, "cold": cold}


if __name__ == "__main__":
    run()
