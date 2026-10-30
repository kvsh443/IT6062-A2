import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import viz
from config import (
    COVARIATE_DAYS,
    EXPOSURE_FILE,
    FIGURE_DIR,
    GRAPH_K,
    OUTCOME_DAYS,
    PRIOR_TYPE_A,
    RANDOM_SEED,
    REPEAT_SEEDS,
    TABLE_DIR,
    TEST_LAST_WEEK,
    TRAIN_LAST_WEEK,
    TREATMENT_CAMPAIGN,
    VAL_LAST_WEEK,
)
from data import (
    campaign_window,
    catalogue,
    causal_frame,
    knn_graph,
    load_campaigns,
    load_coupons,
    load_demographics,
    load_products,
    load_transactions,
    recommendation_splits,
)
from evaluation import aggregate, classification_scores, write_table


def exposure_rows():
    with open(EXPOSURE_FILE, "rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 24), b"")) - 1


def dataset_summary():
    tx = load_transactions()
    products = load_products()
    demo = load_demographics()
    table, desc = load_campaigns()
    coupons, redemptions = load_coupons()
    rows = [
        {"file": "transaction_data", "rows": len(tx), "columns": tx.shape[1],
         "content": "line items with household, basket, day, product, quantity, spend, discounts"},
        {"file": "product", "rows": len(products), "columns": products.shape[1] + 1,
         "content": "department, commodity, brand type, manufacturer, pack size"},
        {"file": "hh_demographic", "rows": len(demo), "columns": demo.shape[1] + 1,
         "content": "age, income, household size, composition, home ownership"},
        {"file": "campaign_table", "rows": len(table), "columns": table.shape[1],
         "content": "household to campaign assignments"},
        {"file": "campaign_desc", "rows": len(desc), "columns": desc.shape[1],
         "content": "campaign type and start and end day"},
        {"file": "coupon", "rows": len(coupons), "columns": coupons.shape[1],
         "content": "coupon to product links per campaign"},
        {"file": "coupon_redempt", "rows": len(redemptions), "columns": redemptions.shape[1],
         "content": "coupon redemptions by household and day"},
        {"file": "causal_data", "rows": exposure_rows(), "columns": 5,
         "content": "weekly in-store display and mailer placement per product and store"},
    ]
    summary = pd.DataFrame(rows)
    write_table(summary, TABLE_DIR / "t1_dataset_summary.csv", float_format="%.0f")

    quality = pd.DataFrame([{
        "households": tx["household_key"].nunique(),
        "products_sold": tx["PRODUCT_ID"].nunique(),
        "baskets": tx["BASKET_ID"].nunique(),
        "stores": tx["STORE_ID"].nunique(),
        "first_day": int(tx["DAY"].min()), "last_day": int(tx["DAY"].max()),
        "weeks": int(tx["WEEK_NO"].max()),
        "missing_values_transactions": int(tx.isna().sum().sum()),
        "non_positive_sales_share": float((tx["SALES_VALUE"] <= 0).mean()),
        "zero_quantity_share": float((tx["QUANTITY"] <= 0).mean()),
        "quantity_over_100_share": float((tx["QUANTITY"] > 100).mean()),
        "demographic_coverage": len(demo) / tx["household_key"].nunique(),
        "households_in_any_campaign": table["household_key"].nunique(),
        "campaigns": len(desc),
        "type_a_campaigns": int((desc["DESCRIPTION"] == "TypeA").sum()),
        "mean_basket_lines": float(tx.groupby("BASKET_ID").size().mean()),
        "median_baskets_per_household": float(tx.groupby("household_key")["BASKET_ID"].nunique().median()),
        "unmapped_commodity_products": int((products["COMMODITY_DESC"] == "NO COMMODITY DESCRIPTION").sum()),
    }])
    write_table(quality, TABLE_DIR / "t1_data_quality.csv")
    return summary, quality


def plot_timeline():
    start, end, _ = campaign_window(TREATMENT_CAMPAIGN)
    to_week = lambda day: day / 7.0
    fig, ax = viz.plt.subplots(figsize=(10.4, 2.6))
    lanes = [
        ("Causal", [
            (to_week(start - COVARIATE_DAYS), to_week(start), viz.SERIES[0], "covariates"),
            (to_week(start), to_week(start + OUTCOME_DAYS), viz.SERIES[1], "outcome"),
        ]),
        ("Placebo", [
            (to_week(start - OUTCOME_DAYS - COVARIATE_DAYS), to_week(start - OUTCOME_DAYS), viz.SERIES[0], "covariates"),
            (to_week(start - OUTCOME_DAYS), to_week(start), viz.SERIES[3], "placebo"),
        ]),
        ("Graph", [(0, TRAIN_LAST_WEEK, viz.SERIES[2], "edges and node features from weeks 1 to 86")]),
        ("Recommender", [
            (0, TRAIN_LAST_WEEK, viz.SERIES[2], "train"),
            (TRAIN_LAST_WEEK, VAL_LAST_WEEK, viz.SERIES[3], "val"),
            (VAL_LAST_WEEK, TEST_LAST_WEEK, viz.SERIES[4], "test"),
        ]),
    ]
    for i, (lane, blocks) in enumerate(lanes):
        y = len(lanes) - 1 - i
        for a, b, colour, label in blocks:
            ax.add_patch(Rectangle((a, y - 0.32), b - a, 0.64, color=colour, alpha=0.85, linewidth=0))
            ax.text((a + b) / 2, y, label, ha="center", va="center", fontsize=6.8, color="white")
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([l for l, _ in lanes][::-1])
    ax.set_xlim(0, TEST_LAST_WEEK)
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.set_xlabel("week of the 102-week panel")
    ax.set_title("Time windows that keep every model blind to its own future")
    ax.grid(axis="y", visible=False)
    viz.strip_spines(ax, keep=("bottom",))
    return viz.save(fig, FIGURE_DIR / "t1_timeline.png")


def plot_preprocessing_flow():
    fig, ax = viz.plt.subplots(figsize=(10.4, 3.0))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3)
    ax.axis("off")
    source = (0.1, 1.05, 1.9, 0.9, "dunnhumby\nThe Complete Journey\n8 relational files", "#e8f0fb")
    branches = [
        (2.7, 2.1, "Household table\npre-period covariates,\ncampaign 18 flag, spend", "#fdece3"),
        (2.7, 1.05, "Product catalogue\nbought by 30+ households\nin weeks 1 to 86", "#fdece3"),
        (2.7, 0.0, "Household by product\nmatrix, weekly splits\n1-86, 87-94, 95-102", "#fdece3"),
    ]
    finals = [
        (5.7, 2.1, "Log transform, overlap\ntrimming, cross-fitted\nnuisance models", "#e6f6ef"),
        (5.7, 1.05, "Basket PMI kNN graph,\nscaler fit on train nodes,\nstratified node split", "#e6f6ef"),
        (5.7, 0.0, "Positive lines only,\nbinary implicit feedback,\nhistory masked for new items", "#e6f6ef"),
    ]
    tasks = [(8.6, 2.1, "Task 2\ncausal effect"), (8.6, 1.05, "Task 3\nGNN"), (8.6, 0.0, "Task 4\nrecommender")]

    def box(x, y, w, h, text, fill):
        ax.add_patch(FancyBboxPatch((x, y + 0.05), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                                    linewidth=1.0, edgecolor=viz.GRID, facecolor=fill))
        ax.text(x + w / 2, y + 0.05 + h / 2, text, ha="center", va="center", fontsize=7.2, color=viz.INK_PRIMARY)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color=viz.INK_MUTED))

    box(*source)
    for (x, y, text, fill), (fx, fy, ftext, ffill), (tx, ty, ttext) in zip(branches, finals, tasks):
        box(x, y, 2.4, 0.8, text, fill)
        box(fx, fy, 2.4, 0.8, ftext, ffill)
        box(tx, ty, 1.6, 0.8, ttext, "#f3f3f1")
        arrow((2.03, 1.5), (x - 0.05, y + 0.45))
        arrow((x + 2.45, y + 0.45), (fx - 0.05, fy + 0.45))
        arrow((fx + 2.45, fy + 0.45), (tx - 0.05, ty + 0.45))
    ax.set_title("One dataset, three derived structures, leakage controls applied before fitting", pad=6)
    return viz.save(fig, FIGURE_DIR / "t1_preprocessing_flow.png")


def baselines():
    import task2_causal
    import task3_gnn
    import task4_recsys

    rows = []
    full, _ = causal_frame(TREATMENT_CAMPAIGN, COVARIATE_DAYS, OUTCOME_DAYS, PRIOR_TYPE_A)
    diff, se = task2_causal.naive(full["out_spend"].to_numpy(float), full["treated"].to_numpy(int))
    rows.append({"component": "causal", "model": "Naive difference in means", "metric": "effect on 8-week spend (USD)",
                 "value": diff, "spread": se, "spread_kind": "standard error"})
    coef, coef_se, _ = task2_causal.ols(full["out_spend"].to_numpy(float), full["treated"].to_numpy(float),
                                        np.empty((len(full), 0)))
    rows.append({"component": "causal", "model": "OLS on treatment only", "metric": "effect on 8-week spend (USD)",
                 "value": coef, "spread": coef_se, "spread_kind": "HC3 standard error"})

    ds = task3_gnn.graph_dataset()
    y = ds["y"]
    edges = knn_graph(ds["edges"], len(y), GRAPH_K)
    graph_rows = []
    for seed in REPEAT_SEEDS:
        split = task3_gnn.node_split(y, seed)
        x = task3_gnn.scaled_features(ds["features"].to_numpy(), split[0])
        for name, result in (("Logistic regression on node features", task3_gnn.logistic_baseline(x, y, split, seed)),
                             ("MLP on node features", task3_gnn.train_model("mlp", x, edges, y, split, seed))):
            test = split[2]
            row = {"model": name, "seed": seed}
            row.update(classification_scores(y[test], result["pred"][test]))
            graph_rows.append(row)
    graph = aggregate(pd.DataFrame(graph_rows), ["model"], ["macro_f1", "accuracy"])
    for _, r in graph.iterrows():
        rows.append({"component": "graph", "model": r["model"], "metric": "test macro-F1",
                     "value": r["macro_f1_mean"], "spread": r["macro_f1_std"], "spread_kind": "std over seeds"})

    items = catalogue()
    data = recommendation_splits(items)
    train_full, test = data["train_full"], data["test"]
    users = np.flatnonzero((np.diff(train_full.indptr) > 0) & (np.diff(test.indptr) > 0))
    scores = task4_recsys.popularity_scores(train_full)
    pop = np.asarray(task4_recsys.binary(train_full).sum(axis=0)).ravel().astype(float)
    extras = {"pop": np.maximum(pop, 1), "commodity": load_products().reindex(items)["COMMODITY_DESC"].to_numpy(),
              "mailer": np.zeros(len(items)), "p": task4_recsys.propensities(train_full)}
    for protocol in task4_recsys.PROTOCOLS:
        row, *_ = task4_recsys.evaluate_scores("Popularity", scores, train_full, test, users, protocol,
                                               extras, RANDOM_SEED, 0.0)
        rows.append({"component": "recommender", "model": f"Popularity, {protocol}", "metric": "NDCG@10",
                     "value": row["ndcg@10"], "spread": np.nan, "spread_kind": "deterministic"})
    table = pd.DataFrame(rows)
    write_table(table, TABLE_DIR / "t1_baselines.csv")
    return table


def run():
    summary, quality = dataset_summary()
    plot_timeline()
    plot_preprocessing_flow()
    table = baselines()
    print("dataset files")
    print(summary[["file", "rows", "columns"]].to_string(index=False))
    print()
    print("data quality")
    print(quality.T.round(4).to_string())
    print()
    print("baselines")
    print(table.round(4).to_string(index=False))
    return {"summary": summary, "quality": quality, "baselines": table}


if __name__ == "__main__":
    run()
