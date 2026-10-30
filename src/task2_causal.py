import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import viz
from config import (
    BOOTSTRAP_REPS,
    CALIPER_SD,
    COVARIATE_DAYS,
    CROSS_FIT_FOLDS,
    FIGURE_DIR,
    OUTCOME_DAYS,
    PRIOR_TYPE_A,
    PROPENSITY_CLIP,
    RANDOM_SEED,
    TABLE_DIR,
    TREATMENT_CAMPAIGN,
    TRIM_LEVELS,
    TRIM_PRIMARY,
)
from data import causal_frame, demographic_covariates
from evaluation import write_table

warnings.filterwarnings("ignore")

LOG_COLUMNS = [
    "pre_spend", "pre_baskets", "pre_products", "pre_commodities", "pre_coupon_lines",
    "pre_basket_value", "pre_recency", "hist_spend", "hist_baskets", "prior_campaigns",
    "prior_redemptions", "pre_stores",
]
RFM = ["pre_spend", "pre_baskets", "pre_recency"]
BEHAVIOURAL = RFM + [
    "pre_products", "pre_commodities", "pre_stores", "pre_discount_share", "pre_coupon_lines",
    "pre_basket_value", "pre_private_share", "pre_produce_share", "pre_meat_share",
    "pre_druggm_share", "hist_spend", "hist_baskets", "tenure_days", "prior_campaigns",
    "prior_type_a", "prior_redemptions",
]
DEMOGRAPHIC = ["income_rank", "age_rank", "household_size", "has_kids", "homeowner", "married"]

COVARIATE_SETS = {
    "none": [],
    "recency-frequency-monetary": RFM,
    "behavioural (primary)": BEHAVIOURAL,
    "behavioural + concurrent campaigns": BEHAVIOURAL + ["concurrent_campaigns"],
}


def design(frame, columns):
    x = frame[columns].astype(float).copy()
    for column in columns:
        if column in LOG_COLUMNS:
            x[column] = np.log1p(x[column].clip(lower=0))
    return x


def naive(y, t):
    a, b = y[t == 1], y[t == 0]
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return diff, se


def ols(y, t, x):
    exog = sm.add_constant(np.column_stack([t, x]) if x.shape[1] else t[:, None])
    fit = sm.OLS(y, exog).fit(cov_type="HC3")
    plain = sm.OLS(y, exog).fit()
    return fit.params[1], fit.bse[1], plain


def propensity(t, x, train=None, test=None):
    scaler = StandardScaler()
    model = LogisticRegression(C=1.0, max_iter=5000)
    train = np.arange(len(t)) if train is None else train
    test = np.arange(len(t)) if test is None else test
    model.fit(scaler.fit_transform(x[train]), t[train])
    return model.predict_proba(scaler.transform(x[test]))[:, 1]


def ipw(y, t, e, estimand="att"):
    e = np.clip(e, *PROPENSITY_CLIP)
    if estimand == "att":
        w = np.where(t == 1, 1.0, e / (1 - e))
        return y[t == 1].mean() - np.average(y[t == 0], weights=w[t == 0])
    w1, w0 = t / e, (1 - t) / (1 - e)
    return np.average(y, weights=w1) - np.average(y, weights=w0)


def overlap_weighting(y, t, e):
    w = np.where(t == 1, 1 - e, e)
    return np.average(y[t == 1], weights=w[t == 1]) - np.average(y[t == 0], weights=w[t == 0])


def overlap_sample(frame, columns, level=TRIM_PRIMARY):
    t = frame["treated"].to_numpy(int)
    e = propensity(t, design(frame, columns).to_numpy())
    keep = (e >= level) & (e <= 1 - level)
    return frame.iloc[np.flatnonzero(keep)].copy(), e


def match(t, e, caliper_sd=CALIPER_SD):
    logit = np.log(np.clip(e, 1e-6, 1 - 1e-6) / np.clip(1 - e, 1e-6, 1))
    caliper = caliper_sd * logit.std()
    treated, control = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    order = np.argsort(logit[control])
    sorted_logit = logit[control][order]
    pos = np.clip(np.searchsorted(sorted_logit, logit[treated]), 1, len(order) - 1)
    left, right = sorted_logit[pos - 1], sorted_logit[pos]
    pick = np.where(
        np.abs(logit[treated] - left) <= np.abs(logit[treated] - right), pos - 1, pos
    )
    partner = control[order[pick]]
    gap = np.abs(logit[treated] - logit[partner])
    keep = gap <= caliper
    return treated[keep], partner[keep]


def psm(y, t, e):
    a, b = match(t, e)
    return (y[a] - y[b]).mean(), len(a)


def aipw(y, t, x, seed, folds=CROSS_FIT_FOLDS):
    n = len(y)
    e, mu0, mu1 = np.zeros(n), np.zeros(n), np.zeros(n)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(x, t):
        e[test] = propensity(t, x, train, test)
        for arm, store in ((0, mu0), (1, mu1)):
            rows = train[t[train] == arm]
            model = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=20, random_state=seed,
            )
            model.fit(x[rows], y[rows])
            store[test] = model.predict(x[test])
    e = np.clip(e, *PROPENSITY_CLIP)
    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    ate = psi.mean()
    ate_se = psi.std(ddof=1) / np.sqrt(n)
    p = t.mean()
    att_terms = t * (y - mu0) - (1 - t) * e / (1 - e) * (y - mu0)
    att = att_terms.sum() / t.sum()
    influence = (att_terms - t * att) / p
    att_se = influence.std(ddof=1) / np.sqrt(n)
    return {"ate": ate, "ate_se": ate_se, "att": att, "att_se": att_se, "psi": psi, "e": e}


def bootstrap(y, t, x, seed, reps=BOOTSTRAP_REPS):
    rng = np.random.default_rng(seed)
    draws = {"psm": [], "ipw_att": [], "ipw_ate": [], "ato": []}
    n = len(y)
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        yb, tb, xb = y[idx], t[idx], x[idx]
        eb = propensity(tb, xb)
        draws["psm"].append(psm(yb, tb, eb)[0])
        draws["ipw_att"].append(ipw(yb, tb, eb, "att"))
        draws["ipw_ate"].append(ipw(yb, tb, eb, "ate"))
        draws["ato"].append(overlap_weighting(yb, tb, eb))
    return {k: np.std(v, ddof=1) for k, v in draws.items()}, draws


def smd(x, t, weights=None):
    weights = np.ones(len(t)) if weights is None else weights
    out = {}
    for j, column in enumerate(x.columns):
        v = x[column].to_numpy()
        a, wa = v[t == 1], weights[t == 1]
        b, wb = v[t == 0], weights[t == 0]
        ma, mb = np.average(a, weights=wa), np.average(b, weights=wb)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        out[column] = (ma - mb) / pooled if pooled > 0 else 0.0
    return pd.Series(out)


def estimate_all(frame, columns, outcome, seed, with_bootstrap=True):
    y = frame[outcome].to_numpy(float)
    t = frame["treated"].to_numpy(int)
    xd = design(frame, columns)
    x = xd.to_numpy()
    rows = []
    diff, se = naive(y, t)
    rows.append(("Naive difference in means", "ATE = ATT", diff, se))
    if not columns:
        return pd.DataFrame(rows, columns=["estimator", "estimand", "estimate", "se"]), {}
    coef, coef_se, _ = ols(y, t, x)
    rows.append(("Regression adjustment (OLS)", "ATE", coef, coef_se))
    e = propensity(t, x)
    att_psm, matched = psm(y, t, e)
    boot_se = {"psm": np.nan, "ipw_att": np.nan, "ipw_ate": np.nan, "ato": np.nan}
    if with_bootstrap:
        boot_se, _ = bootstrap(y, t, x, seed)
    rows.append(("Propensity score matching", "ATT", att_psm, boot_se["psm"]))
    rows.append(("Inverse probability weighting", "ATT", ipw(y, t, e, "att"), boot_se["ipw_att"]))
    rows.append(("Inverse probability weighting", "ATE", ipw(y, t, e, "ate"), boot_se["ipw_ate"]))
    rows.append(("Overlap weighting", "ATO", overlap_weighting(y, t, e), boot_se["ato"]))
    dr = aipw(y, t, x, seed)
    rows.append(("Doubly robust AIPW", "ATT", dr["att"], dr["att_se"]))
    rows.append(("Doubly robust AIPW", "ATE", dr["ate"], dr["ate_se"]))
    table = pd.DataFrame(rows, columns=["estimator", "estimand", "estimate", "se"])
    extras = {"e": e, "matched": matched, "dr": dr, "x": xd, "y": y, "t": t}
    return table, extras


def add_intervals(table):
    table = table.copy()
    table["ci_low"] = table["estimate"] - 1.959964 * table["se"]
    table["ci_high"] = table["estimate"] + 1.959964 * table["se"]
    table["p_value"] = 2 * stats.norm.sf(np.abs(table["estimate"] / table["se"]))
    return table


def robustness_value(t_stat, df, q=1.0, alpha=None):
    f = q * abs(t_stat) / np.sqrt(df)
    if alpha is not None:
        f = f - stats.t.ppf(1 - alpha / 2, df - 1) / np.sqrt(df - 1)
        if f <= 0:
            return 0.0
    return 0.5 * (np.sqrt(f**4 + 4 * f**2) - f**2)


def partial_r2(t_stat, df):
    return t_stat**2 / (t_stat**2 + df)


def adjusted_estimate(estimate, se, df, r2dz, r2yz):
    bias = np.sqrt(r2yz * r2dz / (1 - r2dz)) * se * np.sqrt(df)
    return estimate - np.sign(estimate) * bias


def sensitivity_analysis(frame, columns, outcome):
    y = frame[outcome].to_numpy(float)
    t = frame["treated"].to_numpy(float)
    x = design(frame, columns)
    exog = sm.add_constant(np.column_stack([t, x.to_numpy()]))
    fit = sm.OLS(y, exog).fit()
    est, se, df = fit.params[1], fit.bse[1], fit.df_resid
    t_stat = est / se
    treat_fit = sm.OLS(t, sm.add_constant(x.to_numpy())).fit()
    summary = {
        "estimate": est,
        "se": se,
        "t_stat": t_stat,
        "df": df,
        "partial_r2_treatment": partial_r2(t_stat, df),
        "robustness_value": robustness_value(t_stat, df),
        "robustness_value_alpha05": robustness_value(t_stat, df, alpha=0.05),
    }
    bench_rows = []
    for j, column in enumerate(columns):
        r2dx = partial_r2(treat_fit.tvalues[j + 1], treat_fit.df_resid)
        r2yx = partial_r2(fit.tvalues[j + 2], df)
        for k in (1, 2, 3):
            r2dz = k * r2dx / (1 - r2dx)
            if r2dz >= 1:
                continue
            r2zx = k * r2dx**2 / ((1 - k * r2dx) * (1 - r2dx))
            r2yz = ((np.sqrt(k) + np.sqrt(r2zx)) / np.sqrt(1 - r2zx)) ** 2 * (r2yx / (1 - r2yx))
            r2yz = min(r2yz, 1.0)
            bench_rows.append(
                {
                    "benchmark": column,
                    "multiple": k,
                    "r2_treatment": r2dz,
                    "r2_outcome": r2yz,
                    "adjusted_estimate": adjusted_estimate(est, se, df, r2dz, r2yz),
                }
            )
    bench = pd.DataFrame(bench_rows)
    return summary, bench


def plot_dag():
    nodes = {
        "U": (0.50, 0.92, "Unobserved needs\nand store loyalty", "#f3f3f1"),
        "B": (0.18, 0.62, "Pre-period purchasing\n(spend, visits, mix)", "#e8f0fb"),
        "H": (0.18, 0.08,"Campaign and coupon\nhistory", "#e8f0fb"),
        "D": (0.50, 0.62, "Demographics\n(801 of 2,500 known)", "#e8f0fb"),
        "T": (0.42, 0.28, "Targeted by\nTypeA campaign 18", "#fdece3"),
        "M": (0.63, 0.06,"Coupon redemption\n(mediator)", "#fff4d6"),
        "Y": (0.84, 0.28, "Spend during the\n8-week window", "#e6f6ef"),
        "S": (0.84, 0.62, "Active in window\n(collider, not\nconditioned on)", "#fbe4e4"),
    }
    edges = [
        ("U", "B", True), ("U", "Y", True), ("U", "S", True), ("D", "B", False),
        ("D", "Y", False), ("B", "T", False), ("B", "Y", False), ("H", "T", False),
        ("H", "Y", False), ("B", "H", False), ("T", "M", False), ("M", "Y", False),
        ("T", "Y", False), ("T", "S", False), ("Y", "S", False),
    ]
    fig, ax = viz.plt.subplots(figsize=(8.6, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.02)
    ax.axis("off")
    w, h = 0.19, 0.13
    for key, (x, y, text, fill) in nodes.items():
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=1.1, edgecolor=viz.INK_MUTED, facecolor=fill,
                linestyle="--" if key == "U" else "-",
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=7.6, color=viz.INK_PRIMARY)
    for a, b, latent in edges:
        xa, ya = nodes[a][:2]
        xb, yb = nodes[b][:2]
        vec = np.array([xb - xa, yb - ya])
        unit = vec / np.linalg.norm(vec)
        scale_a = min((w / 2) / max(abs(unit[0]), 1e-6), (h / 2) / max(abs(unit[1]), 1e-6))
        start = np.array([xa, ya]) + unit * (scale_a + 0.005)
        end = np.array([xb, yb]) - unit * (scale_a + 0.008)
        colour = viz.SERIES[4] if (a, b) == ("T", "Y") else viz.INK_SECONDARY
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=10,
                linewidth=1.8 if (a, b) == ("T", "Y") else 1.0, color=colour,
                linestyle=(0, (4, 3)) if latent else "-",
            )
        )
    ax.set_title("Assumed causal diagram for the campaign effect on household spend", pad=4)
    return viz.save(fig, FIGURE_DIR / "t2_dag.png")


def plot_overlap(e, t, trimmed_share):
    fig, ax = viz.plt.subplots(figsize=(6.4, 3.4))
    bins = np.linspace(0, 1, 41)
    ax.hist(e[t == 0], bins=bins, alpha=0.75, color=viz.SERIES[0], label="not targeted")
    ax.hist(e[t == 1], bins=bins, alpha=0.65, color=viz.SERIES[1], label="targeted")
    for edge in (0.05, 0.95):
        ax.axvline(edge, color=viz.INK_MUTED, linestyle="--", linewidth=1.0)
    ax.set_xlabel("estimated propensity of being targeted")
    ax.set_ylabel("households")
    ax.set_title(f"Propensity overlap, {trimmed_share:.1%} of households outside [0.05, 0.95]")
    ax.legend(loc="upper center")
    ax.grid(axis="x", visible=False)
    viz.strip_spines(ax)
    return viz.save(fig, FIGURE_DIR / "t2_overlap.png")


def plot_balance(balance):
    order = balance.sort_values("raw_all_abs").index
    fig, ax = viz.plt.subplots(figsize=(6.4, 5.0))
    ypos = np.arange(len(order))
    ax.scatter(balance.loc[order, "raw_all_abs"], ypos, color=viz.SERIES[1],
               label="unadjusted, all households", zorder=3)
    ax.scatter(balance.loc[order, "raw_abs"], ypos, color=viz.SERIES[3], marker="s", s=16,
               label="unadjusted, overlap population", zorder=3)
    ax.scatter(balance.loc[order, "ipw_abs"], ypos, color=viz.SERIES[0], label="IPW, ATT weights", zorder=3)
    ax.scatter(balance.loc[order, "matched_abs"], ypos, color=viz.SERIES[2], marker="D", s=16,
               label="matched pairs", zorder=3)
    ax.axvline(0.1, color=viz.INK_MUTED, linestyle="--", linewidth=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("absolute standardised mean difference")
    ax.set_title("Covariate balance before and after adjustment")
    ax.legend(loc="lower right")
    viz.strip_spines(ax)
    return viz.save(fig, FIGURE_DIR / "t2_balance.png")


SHORT = {
    "Naive difference in means, all households": "Naive, all households",
    "Naive difference in means": "Naive, overlap population",
    "Regression adjustment (OLS)": "OLS adjustment",
    "Propensity score matching": "PS matching",
    "Inverse probability weighting": "IPW",
    "Overlap weighting": "Overlap weights",
    "Doubly robust AIPW": "AIPW",
}


def plot_estimates(estimates, placebo):
    fig, axes = viz.plt.subplots(1, 2, figsize=(10.4, 3.6))
    for ax, frame, title in (
        (axes[0], estimates, "Campaign window outcome"),
        (axes[1], placebo, "Placebo outcome before the campaign"),
    ):
        labels = [f"{SHORT.get(a, a)}, {b}" for a, b in zip(frame["estimator"], frame["estimand"])]
        ypos = np.arange(len(frame))[::-1]
        colours = [viz.SERIES[1] if s.startswith("Naive") else viz.SERIES[0] for s in frame["estimator"]]
        ax.errorbar(frame["estimate"], ypos,
                    xerr=[frame["estimate"] - frame["ci_low"], frame["ci_high"] - frame["estimate"]],
                    fmt="none", ecolor=viz.INK_MUTED, elinewidth=1.2, capsize=3)
        ax.scatter(frame["estimate"], ypos, color=colours, zorder=3)
        ax.axvline(0, color=viz.INK_SECONDARY, linewidth=0.9)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("effect on 8-week spend (USD)")
        ax.set_title(title)
        viz.strip_spines(ax)
    fig.tight_layout()
    return viz.save(fig, FIGURE_DIR / "t2_estimates.png")


def plot_sensitivity(summary, bench, sets):
    fig, axes = viz.plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axes[0]
    grid = np.linspace(0.0, 0.3, 61)
    r2d, r2y = np.meshgrid(grid, grid)
    adj = adjusted_estimate(summary["estimate"], summary["se"], summary["df"],
                            np.clip(r2d, 0, 0.99), r2y)
    levels = np.linspace(adj.min(), adj.max(), 12)
    contour = ax.contour(r2d, r2y, adj, levels=levels, cmap=viz.SEQUENTIAL, linewidths=1.0)
    ax.clabel(contour, fontsize=6.5, fmt="%.0f")
    ax.contour(r2d, r2y, adj, levels=[0], colors=[viz.SERIES[4]], linewidths=2.0)
    first = bench[bench["multiple"] == 1].sort_values("r2_outcome", ascending=False).head(3)
    for marker, name in zip(("o", "s", "^"), first["benchmark"]):
        part = bench[(bench["benchmark"] == name) & (bench["r2_treatment"] <= 0.3) & (bench["r2_outcome"] <= 0.3)]
        ax.scatter(part["r2_treatment"], part["r2_outcome"], color=viz.SERIES[1], marker=marker, s=22,
                   zorder=3, label=f"{name} at 1x, 2x, 3x")
    ax.legend(loc="center right", fontsize=6.8, frameon=True, facecolor=viz.SURFACE, edgecolor=viz.GRID)
    ax.set_xlabel("partial R2 of confounder with treatment")
    ax.set_ylabel("partial R2 of confounder with outcome")
    ax.set_title("OLS estimate under a hypothetical confounder")
    ax.grid(False)

    ax = axes[1]
    order = sets.reset_index(drop=True)
    ypos = np.arange(len(order))[::-1]
    ax.errorbar(order["estimate"], ypos, xerr=1.959964 * order["se"], fmt="o",
                color=viz.SERIES[0], ecolor=viz.INK_MUTED, capsize=3)
    ax.axvline(0, color=viz.INK_SECONDARY, linewidth=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order["label"], fontsize=7)
    ax.set_xlabel("AIPW ATT on 8-week spend (USD)")
    ax.set_title("Estimate across confounder sets")
    viz.strip_spines(ax)
    fig.tight_layout()
    return viz.save(fig, FIGURE_DIR / "t2_sensitivity.png")


def plot_heterogeneity(table):
    fig, ax = viz.plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(table))
    ax.bar(x, table["cate"], color=viz.SERIES[0], width=0.6)
    ax.errorbar(x, table["cate"], yerr=1.959964 * table["se"], fmt="none", ecolor=viz.INK_SECONDARY, capsize=4)
    ax.axhline(0, color=viz.INK_SECONDARY, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(table["group"])
    ax.set_ylabel("effect on 8-week spend (USD)")
    ax.set_title("Doubly robust effect by pre-period spend quartile")
    ax.grid(axis="x", visible=False)
    viz.strip_spines(ax)
    return viz.save(fig, FIGURE_DIR / "t2_heterogeneity.png")


def pick(table, estimator, estimand):
    return table[(table["estimator"] == estimator) & (table["estimand"] == estimand)].iloc[0]


def run():
    full, window = causal_frame(TREATMENT_CAMPAIGN, COVARIATE_DAYS, OUTCOME_DAYS, PRIOR_TYPE_A)
    placebo_full, placebo_window = causal_frame(
        TREATMENT_CAMPAIGN, COVARIATE_DAYS, OUTCOME_DAYS, PRIOR_TYPE_A, shift_days=OUTCOME_DAYS
    )
    plot_dag()
    frame, e_full = overlap_sample(full, BEHAVIOURAL)
    placebo_frame, _ = overlap_sample(placebo_full, BEHAVIOURAL)
    t_full = full["treated"].to_numpy(int)

    profile = ["pre_spend", "pre_baskets", "hist_spend", "prior_type_a", "prior_campaigns",
               "concurrent_campaigns", "out_spend", "out_baskets", "out_promoted_spend",
               "out_coupon_cost", "has_demographics"]
    population = []
    for label, data in (("all active households", full), ("overlap population", frame)):
        part = data.groupby("treated")[profile].mean()
        part.insert(0, "households", data.groupby("treated").size())
        part.insert(0, "sample", label)
        population.append(part.reset_index())
    population = pd.concat(population, ignore_index=True)
    write_table(population, TABLE_DIR / "t2_population.csv")

    estimates, extras = estimate_all(frame, BEHAVIOURAL, "out_spend", RANDOM_SEED)
    full_diff, full_se = naive(full["out_spend"].to_numpy(float), t_full)
    estimates = pd.concat([
        pd.DataFrame([{"estimator": "Naive difference in means, all households",
                       "estimand": "ATE = ATT", "estimate": full_diff, "se": full_se}]),
        estimates,
    ], ignore_index=True)
    estimates = add_intervals(estimates)
    estimates.insert(0, "outcome", "spend")
    secondary = []
    for outcome, label in (("out_baskets", "baskets"), ("out_promoted_spend", "promoted spend"),
                           ("out_other_spend", "other spend")):
        part, _ = estimate_all(frame, BEHAVIOURAL, outcome, RANDOM_SEED, with_bootstrap=False)
        part = add_intervals(part)
        part.insert(0, "outcome", label)
        secondary.append(part)
    visits = pd.concat(secondary, ignore_index=True)
    write_table(pd.concat([estimates, visits]), TABLE_DIR / "t2_estimates.csv")

    e, t, xd = extras["e"], extras["t"], extras["x"]
    ec = np.clip(e, *PROPENSITY_CLIP)
    w_att = np.where(t == 1, 1.0, ec / (1 - ec))
    w_ato = np.where(t == 1, 1 - e, e)
    a, b = match(t, e)
    matched_rows = np.concatenate([a, b])
    balance = pd.DataFrame({
        "raw_all": smd(design(full, BEHAVIOURAL), t_full),
        "raw": smd(xd, t),
        "ipw": smd(xd, t, w_att),
        "overlap": smd(xd, t, w_ato),
        "matched": smd(xd.iloc[matched_rows].reset_index(drop=True), t[matched_rows]),
    })
    for column in ("raw_all", "raw", "ipw", "overlap", "matched"):
        balance[f"{column}_abs"] = balance[column].abs()
    plot_balance(balance)
    write_table(balance.reset_index(names="covariate"), TABLE_DIR / "t2_balance.csv")

    outside = float(((e_full < TRIM_PRIMARY) | (e_full > 1 - TRIM_PRIMARY)).mean())
    plot_overlap(e_full, t_full, outside)
    ef = np.clip(e_full, *PROPENSITY_CLIP)
    wf = np.where(t_full == 1, 1.0, ef / (1 - ef))
    overlap = pd.DataFrame([{
        "propensity_auc": roc_auc_score(t_full, e_full),
        "max_e_control": e_full[t_full == 0].max(),
        "share_outside": outside,
        "households_full": len(full), "treated_full": int(t_full.sum()),
        "households_overlap": len(frame), "treated_overlap": int(t.sum()),
        "controls_overlap": int((t == 0).sum()),
        "ess_controls_full": wf[t_full == 0].sum() ** 2 / (wf[t_full == 0] ** 2).sum(),
        "ess_controls_overlap": w_att[t == 0].sum() ** 2 / (w_att[t == 0] ** 2).sum(),
        "matched_treated": len(a), "unique_controls_used": len(np.unique(b)),
        "max_abs_smd_raw_all": balance["raw_all_abs"].max(),
        "max_abs_smd_raw": balance["raw_abs"].max(),
        "max_abs_smd_ipw": balance["ipw_abs"].max(),
        "max_abs_smd_overlap": balance["overlap_abs"].max(),
        "max_abs_smd_matched": balance["matched_abs"].max(),
    }])
    write_table(overlap, TABLE_DIR / "t2_overlap.csv")

    placebo, _ = estimate_all(placebo_frame, BEHAVIOURAL, "out_spend", RANDOM_SEED)
    placebo = add_intervals(placebo)
    write_table(placebo, TABLE_DIR / "t2_placebo.csv")
    plot_estimates(estimates, placebo)

    set_rows = []
    for label, columns in COVARIATE_SETS.items():
        table, _ = estimate_all(frame, columns, "out_spend", RANDOM_SEED, with_bootstrap=False)
        row = pick(table, "Doubly robust AIPW", "ATT") if columns else table.iloc[0]
        set_rows.append({"label": label, "sample": "overlap population", "households": len(frame),
                         "estimator": row["estimator"], "estimate": row["estimate"], "se": row["se"]})
    known = frame.index[frame["has_demographics"] == 1]
    demo = frame.loc[known].join(demographic_covariates(known))
    for label, columns in (("behavioural, demographic subset", BEHAVIOURAL),
                           ("behavioural + demographics", BEHAVIOURAL + DEMOGRAPHIC)):
        table, _ = estimate_all(demo, columns, "out_spend", RANDOM_SEED, with_bootstrap=False)
        row = pick(table, "Doubly robust AIPW", "ATT")
        set_rows.append({"label": label, "sample": "demographic subset", "households": len(demo),
                         "estimator": row["estimator"], "estimate": row["estimate"], "se": row["se"]})
    sets = pd.DataFrame(set_rows)
    write_table(sets, TABLE_DIR / "t2_confounder_sets.csv")

    trim_rows = []
    y_full = full["out_spend"].to_numpy(float)
    for level in TRIM_LEVELS:
        keep = (e_full >= level) & (e_full <= 1 - level) if level > 0 else np.ones(len(e_full), bool)
        sub = full.iloc[np.flatnonzero(keep)]
        dr = aipw(sub["out_spend"].to_numpy(float), sub["treated"].to_numpy(int),
                  design(sub, BEHAVIOURAL).to_numpy(), RANDOM_SEED)
        trim_rows.append({"trim": level, "households": int(keep.sum()),
                          "treated": int(t_full[keep].sum()),
                          "ipw_att": ipw(y_full[keep], t_full[keep], e_full[keep], "att"),
                          "aipw_att": dr["att"], "aipw_att_se": dr["att_se"]})
    trimming = pd.DataFrame(trim_rows)
    write_table(trimming, TABLE_DIR / "t2_trimming.csv")

    summary, bench = sensitivity_analysis(frame, BEHAVIOURAL, "out_spend")
    write_table(pd.DataFrame([summary]), TABLE_DIR / "t2_sensitivity.csv")
    write_table(bench, TABLE_DIR / "t2_sensitivity_benchmarks.csv")
    plot_sensitivity(summary, bench, sets)

    psi = extras["dr"]["psi"]
    quartile = pd.qcut(frame["pre_spend"].rank(method="first"), 4,
                       labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    het_rows = []
    for group in quartile.cat.categories:
        mask = (quartile == group).to_numpy()
        het_rows.append({"group": group, "households": int(mask.sum()),
                         "treated_share": t[mask].mean(),
                         "pre_spend_median": frame.loc[mask, "pre_spend"].median(),
                         "cate": psi[mask].mean(),
                         "se": psi[mask].std(ddof=1) / np.sqrt(mask.sum())})
    het = pd.DataFrame(het_rows)
    write_table(het, TABLE_DIR / "t2_heterogeneity.csv")
    plot_heterogeneity(het)

    dr_att = pick(estimates, "Doubly robust AIPW", "ATT")
    treated_mean = frame.loc[frame["treated"] == 1, "out_spend"].mean()
    counterfactual = treated_mean - dr_att["estimate"]
    coupon_gap = (frame.loc[frame["treated"] == 1, "out_coupon_cost"].mean()
                  - frame.loc[frame["treated"] == 0, "out_coupon_cost"].mean())
    practical = pd.DataFrame([{
        "campaign": TREATMENT_CAMPAIGN, "campaign_type": window["kind"],
        "start_day": window["start"], "end_day": window["end"],
        "covariate_first_day": window["anchor"] - COVARIATE_DAYS,
        "placebo_first_day": placebo_window["anchor"],
        "placebo_covariate_first_day": placebo_window["anchor"] - COVARIATE_DAYS,
        "households": len(frame), "treated": int(frame["treated"].sum()),
        "treated_mean_spend": treated_mean,
        "counterfactual_mean_spend": counterfactual,
        "att": dr_att["estimate"], "att_low": dr_att["ci_low"], "att_high": dr_att["ci_high"],
        "att_per_week": dr_att["estimate"] / (OUTCOME_DAYS / 7),
        "att_relative": dr_att["estimate"] / counterfactual,
        "att_relative_low": dr_att["ci_low"] / counterfactual,
        "att_relative_high": dr_att["ci_high"] / counterfactual,
        "coupon_cost_gap": coupon_gap,
        "total_incremental_spend": dr_att["estimate"] * frame["treated"].sum(),
    }])
    write_table(practical, TABLE_DIR / "t2_practical.csv")

    print("population")
    print(population.round(2).to_string(index=False))
    print()
    print("overlap")
    print(overlap.T.round(4).to_string())
    print()
    print("estimates")
    print(pd.concat([estimates, visits])[["outcome", "estimator", "estimand", "estimate", "se",
                                          "ci_low", "ci_high"]].round(3).to_string(index=False))
    print()
    print("placebo")
    print(placebo[["estimator", "estimand", "estimate", "ci_low", "ci_high"]].round(3).to_string(index=False))
    print()
    print("confounder sets")
    print(sets.round(3).to_string(index=False))
    print()
    print("trimming")
    print(trimming.round(3).to_string(index=False))
    print()
    print("sensitivity")
    print(pd.Series(summary).round(4).to_string())
    print()
    print("heterogeneity")
    print(het.round(3).to_string(index=False))
    print()
    print("practical")
    print(practical.T.round(4).to_string())
    return {"estimates": estimates, "placebo": placebo, "sets": sets, "practical": practical}


if __name__ == "__main__":
    run()
