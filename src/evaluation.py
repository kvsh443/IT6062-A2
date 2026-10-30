import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def classification_scores(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def per_class_f1(y_true, y_pred, classes, names):
    values = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    return {f"f1_{n}": v for n, v in zip(names, values)}


def ranking_scores(top, truth, k):
    hits = np.array([[item in t for item in row[:k]] for row, t in zip(top, truth)], dtype=float)
    sizes = np.array([len(t) for t in truth], dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(axis=1)
    ideal = np.array([discounts[: int(min(s, k))].sum() for s in sizes])
    cumulative = np.cumsum(hits, axis=1)
    precision_at = cumulative / np.arange(1, k + 1)
    ap = (precision_at * hits).sum(axis=1) / np.minimum(sizes, k)
    return {
        f"precision@{k}": hits.sum(axis=1).mean() / k,
        f"recall@{k}": (hits.sum(axis=1) / sizes).mean(),
        f"ndcg@{k}": (dcg / ideal).mean(),
        f"map@{k}": ap.mean(),
        f"hit_rate@{k}": (hits.sum(axis=1) > 0).mean(),
    }


def per_user_recall(top, truth, k):
    return np.array(
        [len(set(row[:k]) & t) / len(t) for row, t in zip(top, truth)], dtype=float
    )


def aggregate(frame, group_keys, value_keys):
    grouped = frame.groupby(group_keys, sort=False)[value_keys]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).fillna(0.0).add_suffix("_std")
    return mean.join(std).reset_index()


def mean_confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = values.size
    mean = values.mean()
    if n < 2:
        return mean, 0.0
    sem = values.std(ddof=1) / np.sqrt(n)
    critical = stats.t.ppf(0.5 + confidence / 2.0, n - 1)
    return mean, critical * sem


def write_table(frame, path, float_format="%.4f"):
    frame.to_csv(path, index=False, float_format=float_format)
    return path
