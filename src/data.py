import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import (
    CACHE_DIR,
    CAMPAIGN_DESC_FILE,
    CAMPAIGN_TABLE_FILE,
    COUPON_FILE,
    DEMOGRAPHIC_FILE,
    EXPOSURE_FILE,
    GRAPH_MIN_COOCCURRENCE,
    MIN_ITEM_HOUSEHOLDS,
    PRODUCT_FILE,
    REDEMPTION_FILE,
    TEST_LAST_WEEK,
    TRAIN_LAST_WEEK,
    TRANSACTION_FILE,
    VAL_LAST_WEEK,
)

TRANSACTION_TYPES = {
    "household_key": "int32",
    "BASKET_ID": "int64",
    "DAY": "int16",
    "PRODUCT_ID": "int32",
    "QUANTITY": "int32",
    "SALES_VALUE": "float32",
    "STORE_ID": "int32",
    "RETAIL_DISC": "float32",
    "TRANS_TIME": "int16",
    "WEEK_NO": "int16",
    "COUPON_DISC": "float32",
    "COUPON_MATCH_DISC": "float32",
}

_CACHE = {}


def load_transactions():
    if "tx" not in _CACHE:
        _CACHE["tx"] = pd.read_csv(TRANSACTION_FILE, dtype=TRANSACTION_TYPES)
    return _CACHE["tx"]


def purchase_lines(tx):
    return tx[(tx["QUANTITY"] > 0) & (tx["SALES_VALUE"] > 0)]


def load_products():
    frame = pd.read_csv(PRODUCT_FILE)
    for column in ("DEPARTMENT", "BRAND", "COMMODITY_DESC", "SUB_COMMODITY_DESC"):
        frame[column] = frame[column].astype(str).str.strip()
    return frame.set_index("PRODUCT_ID")


def load_demographics():
    return pd.read_csv(DEMOGRAPHIC_FILE).set_index("household_key")


def load_campaigns():
    table = pd.read_csv(CAMPAIGN_TABLE_FILE)
    desc = pd.read_csv(CAMPAIGN_DESC_FILE)
    return table, desc


def load_coupons():
    return pd.read_csv(COUPON_FILE), pd.read_csv(REDEMPTION_FILE)


def exposure_by_product():
    path = CACHE_DIR / "exposure_weeks.pkl"
    if path.exists():
        return pd.read_pickle(path)
    frame = pd.read_csv(
        EXPOSURE_FILE,
        dtype={
            "PRODUCT_ID": "int32",
            "STORE_ID": "int32",
            "WEEK_NO": "int16",
            "display": "category",
            "mailer": "category",
        },
    )
    frame["on_display"] = frame["display"].astype(str) != "0"
    frame["on_mailer"] = frame["mailer"].astype(str) != "0"
    weekly = frame.groupby(["PRODUCT_ID", "WEEK_NO"])[["on_display", "on_mailer"]].any()
    weekly = weekly.reset_index()
    weekly.to_pickle(path)
    return weekly


def exposure_rates(items, last_week):
    weekly = exposure_by_product()
    window = weekly[weekly["WEEK_NO"] <= last_week]
    weeks = float(window["WEEK_NO"].nunique())
    rates = window.groupby("PRODUCT_ID")[["on_display", "on_mailer"]].sum() / weeks
    rates = rates.reindex(items).fillna(0.0)
    return rates.rename(columns={"on_display": "display_rate", "on_mailer": "mailer_rate"})


def catalogue():
    tx = purchase_lines(load_transactions())
    train = tx[tx["WEEK_NO"] <= TRAIN_LAST_WEEK]
    reach = train.groupby("PRODUCT_ID")["household_key"].nunique()
    return np.sort(reach[reach >= MIN_ITEM_HOUSEHOLDS].index.to_numpy())


def product_features(items):
    tx = purchase_lines(load_transactions())
    tx = tx[(tx["WEEK_NO"] <= TRAIN_LAST_WEEK) & tx["PRODUCT_ID"].isin(items)].copy()
    basket_lines = tx.groupby("BASKET_ID")["PRODUCT_ID"].transform("size")
    basket_value = tx.groupby("BASKET_ID")["SALES_VALUE"].transform("sum")
    tx["basket_lines"] = basket_lines.astype(np.float32)
    tx["basket_value"] = basket_value.astype(np.float32)
    shelf = tx["SALES_VALUE"] - tx["RETAIL_DISC"]
    tx["unit_price"] = np.where(tx["QUANTITY"] <= 100, shelf / tx["QUANTITY"], np.nan)
    tx["discounted"] = (tx["RETAIL_DISC"] < 0).astype(np.float32)
    tx["depth"] = (-tx["RETAIL_DISC"] / shelf.clip(lower=0.01)).clip(0, 1)
    tx["couponed"] = (tx["COUPON_DISC"] < 0).astype(np.float32)
    tx["hour"] = (tx["TRANS_TIME"] // 100).astype(np.float32)
    tx["evening"] = (tx["hour"] >= 17).astype(np.float32)
    tx["late_period"] = (tx["WEEK_NO"] > TRAIN_LAST_WEEK / 2).astype(np.float32)
    tx["multi_unit"] = (tx["QUANTITY"] > 1).astype(np.float32)
    for d in range(7):
        tx[f"weekday_{d}"] = (tx["DAY"] % 7 == d).astype(np.float32)
    for q in range(4):
        tx[f"season_{q}"] = (((tx["WEEK_NO"] - 1) % 52) // 13 == q).astype(np.float32)

    grouped = tx.groupby("PRODUCT_ID")
    per_household = tx.groupby(["PRODUCT_ID", "household_key"])["BASKET_ID"].nunique()
    repeat = (per_household >= 2).groupby(level=0).mean()

    feats = pd.DataFrame(index=pd.Index(items, name="PRODUCT_ID"))
    feats["log_households"] = np.log1p(grouped["household_key"].nunique())
    feats["log_baskets"] = np.log1p(grouped["BASKET_ID"].nunique())
    feats["log_units"] = np.log1p(grouped["QUANTITY"].sum().clip(upper=1e6))
    feats["log_revenue"] = np.log1p(grouped["SALES_VALUE"].sum())
    feats["log_unit_price"] = np.log1p(grouped["unit_price"].median())
    feats["price_dispersion"] = grouped["unit_price"].std() / grouped["unit_price"].mean()
    feats["discount_rate"] = grouped["discounted"].mean()
    feats["discount_depth"] = grouped["depth"].mean()
    feats["coupon_rate"] = grouped["couponed"].mean()
    feats["repeat_rate"] = repeat
    feats["log_basket_lines"] = np.log1p(grouped["basket_lines"].mean())
    feats["log_basket_value"] = np.log1p(grouped["basket_value"].mean())
    feats["mean_hour"] = grouped["hour"].mean()
    feats["evening_share"] = grouped["evening"].mean()
    feats["late_period_share"] = grouped["late_period"].mean()
    feats["log_stores"] = np.log1p(grouped["STORE_ID"].nunique())
    feats["multi_unit_share"] = grouped["multi_unit"].mean()
    for d in range(7):
        feats[f"weekday_{d}_share"] = grouped[f"weekday_{d}"].mean()
    for q in range(4):
        feats[f"season_{q}_share"] = grouped[f"season_{q}"].mean()
    products = load_products()
    feats["private_label"] = (
        products.reindex(items)["BRAND"].eq("Private").astype(np.float32).to_numpy()
    )
    feats = feats.join(exposure_rates(items, TRAIN_LAST_WEEK))
    return feats.fillna(0.0).astype(np.float32)


def basket_item_matrix(items):
    tx = purchase_lines(load_transactions())
    tx = tx[(tx["WEEK_NO"] <= TRAIN_LAST_WEEK) & tx["PRODUCT_ID"].isin(items)]
    pairs = tx[["BASKET_ID", "PRODUCT_ID"]].drop_duplicates()
    rows = pd.factorize(pairs["BASKET_ID"])[0]
    cols = np.searchsorted(items, pairs["PRODUCT_ID"].to_numpy())
    data = np.ones(len(pairs), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(rows.max() + 1, len(items)))


def pmi_edges(items):
    path = CACHE_DIR / f"pmi_edges_{len(items)}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    baskets = basket_item_matrix(items)
    cooc = (baskets.T @ baskets).tocsr()
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    cooc = cooc.tocoo()
    keep = cooc.data >= GRAPH_MIN_COOCCURRENCE
    src, dst, count = cooc.row[keep], cooc.col[keep], cooc.data[keep]
    freq = np.asarray(baskets.sum(axis=0)).ravel()
    pmi = np.log(count * baskets.shape[0] / (freq[src] * freq[dst]))
    frame = pd.DataFrame({"src": src, "dst": dst, "count": count, "pmi": pmi})
    frame = frame[frame["pmi"] > 0].reset_index(drop=True)
    frame.to_pickle(path)
    return frame


def knn_graph(edges, n_nodes, k):
    top = edges.sort_values(["src", "pmi"], ascending=[True, False]).groupby("src").head(k)
    src = np.concatenate([top["src"].to_numpy(), top["dst"].to_numpy()])
    dst = np.concatenate([top["dst"].to_numpy(), top["src"].to_numpy()])
    pairs = np.unique(np.stack([src, dst]), axis=1)
    pairs = pairs[:, pairs[0] != pairs[1]]
    return pairs.astype(np.int64)


def interaction_matrix(items, first_week, last_week, households):
    tx = purchase_lines(load_transactions())
    tx = tx[
        (tx["WEEK_NO"] >= first_week)
        & (tx["WEEK_NO"] <= last_week)
        & tx["PRODUCT_ID"].isin(items)
        & tx["household_key"].isin(households)
    ]
    counts = tx.groupby(["household_key", "PRODUCT_ID"])["BASKET_ID"].nunique()
    rows = np.searchsorted(households, counts.index.get_level_values(0).to_numpy())
    cols = np.searchsorted(items, counts.index.get_level_values(1).to_numpy())
    return sp.csr_matrix(
        (counts.to_numpy().astype(np.float32), (rows, cols)),
        shape=(len(households), len(items)),
    )


def recommendation_splits(items):
    households = np.sort(load_transactions()["household_key"].unique())
    return {
        "households": households,
        "items": items,
        "train": interaction_matrix(items, 1, TRAIN_LAST_WEEK, households),
        "val": interaction_matrix(items, TRAIN_LAST_WEEK + 1, VAL_LAST_WEEK, households),
        "train_full": interaction_matrix(items, 1, VAL_LAST_WEEK, households),
        "test": interaction_matrix(items, VAL_LAST_WEEK + 1, TEST_LAST_WEEK, households),
    }


def campaign_window(campaign):
    _, desc = load_campaigns()
    row = desc[desc["CAMPAIGN"] == campaign].iloc[0]
    return int(row["START_DAY"]), int(row["END_DAY"]), str(row["DESCRIPTION"])


def window_activity(tx, first_day, last_day, prefix):
    window = tx[(tx["DAY"] >= first_day) & (tx["DAY"] <= last_day)]
    grouped = window.groupby("household_key")
    out = pd.DataFrame(
        {
            f"{prefix}spend": grouped["SALES_VALUE"].sum(),
            f"{prefix}baskets": grouped["BASKET_ID"].nunique(),
        }
    )
    return out


def household_covariates(start_day, covariate_days):
    tx = load_transactions()
    products = load_products()
    first = start_day - covariate_days
    window = tx[(tx["DAY"] >= first) & (tx["DAY"] < start_day)].copy()
    window = window.join(products[["DEPARTMENT", "BRAND", "COMMODITY_DESC"]], on="PRODUCT_ID")
    grouped = window.groupby("household_key")
    cov = pd.DataFrame(
        {
            "pre_spend": grouped["SALES_VALUE"].sum(),
            "pre_baskets": grouped["BASKET_ID"].nunique(),
            "pre_products": grouped["PRODUCT_ID"].nunique(),
            "pre_commodities": grouped["COMMODITY_DESC"].nunique(),
            "pre_stores": grouped["STORE_ID"].nunique(),
            "pre_discount_share": grouped["RETAIL_DISC"].sum()
            / grouped["SALES_VALUE"].sum().clip(lower=0.01),
            "pre_coupon_lines": grouped["COUPON_DISC"].apply(lambda s: float((s < 0).sum())),
            "pre_last_day": grouped["DAY"].max(),
        }
    )
    cov["pre_discount_share"] = -cov["pre_discount_share"]
    cov["pre_basket_value"] = cov["pre_spend"] / cov["pre_baskets"]
    cov["pre_recency"] = start_day - cov.pop("pre_last_day")
    private = window.assign(private=window["BRAND"].eq("Private") * window["SALES_VALUE"])
    cov["pre_private_share"] = (
        private.groupby("household_key")["private"].sum() / cov["pre_spend"].clip(lower=0.01)
    )
    for dept, column in (
        ("PRODUCE", "pre_produce_share"),
        ("MEAT", "pre_meat_share"),
        ("DRUG GM", "pre_druggm_share"),
    ):
        part = window[window["DEPARTMENT"] == dept].groupby("household_key")["SALES_VALUE"].sum()
        cov[column] = (part / cov["pre_spend"].clip(lower=0.01)).reindex(cov.index).fillna(0.0)

    history = tx[tx["DAY"] < first].groupby("household_key")
    cov["hist_spend"] = history["SALES_VALUE"].sum().reindex(cov.index).fillna(0.0)
    cov["hist_baskets"] = history["BASKET_ID"].nunique().reindex(cov.index).fillna(0.0)
    cov["tenure_days"] = (start_day - tx.groupby("household_key")["DAY"].min()).reindex(cov.index)
    return cov


def campaign_history(households, start_day, campaign, prior_type_a):
    table, desc = load_campaigns()
    starts = desc.set_index("CAMPAIGN")["START_DAY"]
    table = table.assign(start=table["CAMPAIGN"].map(starts))
    earlier = table[(table["start"] < start_day) & (table["CAMPAIGN"] != campaign)]
    out = pd.DataFrame(index=pd.Index(households, name="household_key"))
    out["prior_campaigns"] = earlier.groupby("household_key").size().reindex(households).fillna(0)
    out["prior_type_a"] = (
        earlier[earlier["CAMPAIGN"].isin(prior_type_a)]
        .groupby("household_key")
        .size()
        .reindex(households)
        .fillna(0)
    )
    _, redemptions = load_coupons()
    redeemed = redemptions[redemptions["DAY"] < start_day].groupby("household_key").size()
    out["prior_redemptions"] = redeemed.reindex(households).fillna(0)
    return out


def concurrent_campaigns(households, start_day, end_day, campaign):
    table, desc = load_campaigns()
    desc = desc.set_index("CAMPAIGN")
    overlap = desc[
        (desc["START_DAY"] <= end_day) & (desc["END_DAY"] >= start_day) & (desc.index != campaign)
    ].index
    hits = table[table["CAMPAIGN"].isin(overlap)].groupby("household_key").size()
    return hits.reindex(households).fillna(0).rename("concurrent_campaigns")


def demographic_covariates(households):
    demo = load_demographics().reindex(households)
    income_order = [
        "Under 15K", "15-24K", "25-34K", "35-49K", "50-74K", "75-99K",
        "100-124K", "125-149K", "150-174K", "175-199K", "200-249K", "250K+",
    ]
    age_order = ["19-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    out = pd.DataFrame(index=demo.index)
    out["income_rank"] = demo["INCOME_DESC"].map({v: i for i, v in enumerate(income_order)})
    out["age_rank"] = demo["AGE_DESC"].map({v: i for i, v in enumerate(age_order)})
    out["household_size"] = demo["HOUSEHOLD_SIZE_DESC"].astype(str).str.replace("+", "").astype(float)
    out["has_kids"] = (demo["KID_CATEGORY_DESC"] != "None/Unknown").astype(float)
    out["homeowner"] = demo["HOMEOWNER_DESC"].isin(["Homeowner", "Probable Owner"]).astype(float)
    out["married"] = (demo["MARITAL_STATUS_CODE"] == "A").astype(float)
    return out


def causal_frame(campaign, covariate_days, outcome_days, prior_type_a, shift_days=0):
    start, end, kind = campaign_window(campaign)
    anchor = start - shift_days
    cov = household_covariates(anchor, covariate_days)
    households = cov.index.to_numpy()
    table, _ = load_campaigns()
    treated = set(table.loc[table["CAMPAIGN"] == campaign, "household_key"])
    frame = cov.copy()
    frame["treated"] = frame.index.isin(treated).astype(int)
    frame = frame.join(campaign_history(households, anchor, campaign, prior_type_a))
    frame = frame.join(concurrent_campaigns(households, start, end, campaign))
    outcome = window_activity(load_transactions(), anchor, anchor + outcome_days - 1, "out_")
    frame = frame.join(outcome).fillna({"out_spend": 0.0, "out_baskets": 0})
    tx = load_transactions()
    coupon_window = tx[(tx["DAY"] >= anchor) & (tx["DAY"] < anchor + outcome_days)]
    coupon_cost = -coupon_window.groupby("household_key")["COUPON_DISC"].sum()
    frame["out_coupon_cost"] = coupon_cost.reindex(frame.index).fillna(0.0)
    coupons, _ = load_coupons()
    promoted = set(coupons.loc[coupons["CAMPAIGN"] == campaign, "PRODUCT_ID"])
    targeted_lines = coupon_window[coupon_window["PRODUCT_ID"].isin(promoted)]
    promoted_spend = targeted_lines.groupby("household_key")["SALES_VALUE"].sum()
    frame["out_promoted_spend"] = promoted_spend.reindex(frame.index).fillna(0.0)
    frame["out_other_spend"] = frame["out_spend"] - frame["out_promoted_spend"]
    frame["has_demographics"] = frame.index.isin(load_demographics().index).astype(int)
    return frame, {"start": start, "end": end, "kind": kind, "anchor": anchor}
