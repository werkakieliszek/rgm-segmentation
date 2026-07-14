"""Reusable plots for cluster profile tables (from clustering.py's profile()).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

from features import build_retailer_weekly_panel,  _clean

GOLD = "#E6D389"
RUST = "#914E56"
SLATE = "#4D303F"
INK = "#2F2A35"
MUTED = "#C1A980"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#D8D5CC",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _label_for(cluster_id, labels: dict | None) -> str:
    if labels and cluster_id in labels:
        return labels[cluster_id]
    return f"Cluster {cluster_id}"


def find_extreme(profile: pd.DataFrame, metric: str, mode: str = "max"):
    """Which cluster is the interesting extreme on a given metric"""
    return profile[metric].idxmax() if mode == "max" else profile[metric].idxmin()


def plot_growth_vs_erosion(profile: pd.DataFrame, labels: dict | None = None, ax=None):
    """Quadrant scatter: x=acv_trend (distribution trajectory), y=mat_growth, dot size=avg_weekly_sales"""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.5))

    growth_id = find_extreme(profile, "mat_growth", "max")
    erosion_id = find_extreme(profile, "acv_trend", "min")

    sizes = 200 + 1800 * (profile["avg_weekly_sales"] / profile["avg_weekly_sales"].max())
    texts = []
    for cid, row in profile.iterrows():
        color = GOLD if cid == growth_id else RUST if cid == erosion_id else SLATE
        alpha = 0.85 if cid in (growth_id, erosion_id) else 0.45
        ax.scatter(row["acv_trend"], row["mat_growth"], s=sizes.loc[cid], color=color, alpha=alpha,
                   edgecolors="white", linewidths=1.2, zorder=3 if cid in (growth_id, erosion_id) else 2)
        label = _label_for(cid, labels)
        texts.append(ax.text(row["acv_trend"], row["mat_growth"], f"{label}\nn={int(row['n_ppgs'])}",
                              fontsize=9.5, color=INK))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))

    ax.axvline(0, color="#D8D5CC", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("← distribution shrinking     acv trend     distribution expanding →")
    ax.set_ylabel("MAT growth (trailing 52wk vs. prior 52wk)")
    ax.set_title("Where each segment sits: growth vs. distribution trajectory", fontsize=13, color=INK, pad=14)
    return ax


def find_quadrant_extreme(profile: pd.DataFrame, x_col: str, y_col: str, x_side: str, y_mode: str):
    """Among clusters on one side of `x_col`'s median, return the one with the highest or lowest value of `y_col`. This helps identify clusters with unusual combinations of related metrics.
    """
    x_median = profile[x_col].median()
    subset = profile[profile[x_col] >= x_median] if x_side == "above" else profile[profile[x_col] < x_median]
    if subset.empty:
        return None
    return subset[y_col].idxmax() if y_mode == "max" else subset[y_col].idxmin()


def plot_categorical_mix(feats: pd.DataFrame, cluster_labels: pd.Series, category_col: str,
                          labels: dict | None = None, ax=None):
    """100%-stacked horizontal bar: category_col composition per cluster"""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.6 * cluster_labels.nunique() + 1.5))

    mix = pd.crosstab(cluster_labels, feats[category_col], normalize="index")
    mix.index = [_label_for(cid, labels) for cid in mix.index]

    palette = plt.cm.Set3(np.linspace(0, 1, mix.shape[1]))
    left = np.zeros(len(mix))
    for i, col in enumerate(mix.columns):
        ax.barh(mix.index, mix[col], left=left, color=palette[i], label=col, height=0.6)
        left += mix[col].values

    ax.set_xlim(0, 1)
    ax.set_xlabel(f"share of {category_col}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, frameon=False)
    ax.set_title(f"{category_col} composition by segment", fontsize=13, color=INK, pad=12)
    return ax


def plot_promo_efficiency(profile: pd.DataFrame, labels: dict | None = None, ax=None):
    """Scatter: promo_week_share (x) vs. promo_lift (y, log scale"""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.5))

    over_id = find_quadrant_extreme(profile, "promo_week_share", "promo_lift", "above", "min")
    under_id = find_quadrant_extreme(profile, "promo_week_share", "promo_lift", "below", "max")

    texts = []
    for cid, row in profile.iterrows():
        color = RUST if cid == over_id else GOLD if cid == under_id else SLATE
        alpha = 0.85 if cid in (over_id, under_id) else 0.45
        ax.scatter(row["promo_week_share"], row["promo_lift"], s=500, color=color, alpha=alpha,
                   edgecolors="white", linewidths=1.2, zorder=3 if cid in (over_id, under_id) else 2)
        label = _label_for(cid, labels)
        texts.append(ax.text(row["promo_week_share"], row["promo_lift"], label, fontsize=9.5, color=INK))

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.axvline(profile["promo_week_share"].median(), color="#D8D5CC", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("share of weeks promoted")
    ax.set_ylabel("promo lift (log scale)")
    ax.set_title("Promotion frequency vs. effectiveness", fontsize=13, color=INK, pad=14)
    return ax


def plot_ppg_trend(weekly: pd.DataFrame, ppg_ids: dict, ax=None):
    """
    Stacked area version of plot_ppg_trend
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.5))

    mask = weekly["rgm_ppg"].isin(ppg_ids.values())
    sub = weekly.loc[mask, ["rgm_ppg", "period_id", "sales"]].copy()
    sub = sub.sort_values(["period_id", "rgm_ppg"])

    name_by_ppg = {ppg_id: name for name, ppg_id in ppg_ids.items()}
    sub["display_name"] = sub["rgm_ppg"].map(name_by_ppg)
    wide = (
        sub.pivot(index="period_id", columns="display_name", values="sales")
        .sort_index()
        .fillna(0.0)
    )

    dates = wide.index
    series_names = list(wide.columns)
    values = wide.to_numpy().T  # shape (n_series, n_time)

    palette = plt.cm.tab10(np.linspace(0, 1, len(series_names)))

    ax.stackplot(dates, *values, labels=series_names, colors=palette, linewidth=0)

    ax.set_xlabel("week")
    ax.set_ylabel("weekly sales")
    ax.set_title(
        "Weekly sales (stacked) — one representative product per segment",
        fontsize=13,
        color=INK,
        pad=14,
    )
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    return ax


def plot_promo_response(weekly: pd.DataFrame, ppg_id: str, title: str | None = None, ax=None):
    """One product's weekly sales with promoted weeks shaded"""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))

    series = weekly[weekly["rgm_ppg"] == ppg_id].sort_values("period_id")
    ax.plot(series["period_id"], series["sales"], color=INK, linewidth=1.3, zorder=2)
    promo_weeks = series[series["on_promo"] == 1]
    for _, row in promo_weeks.iterrows():
        ax.axvspan(row["period_id"] - pd.Timedelta(days=3), row["period_id"] + pd.Timedelta(days=3),
                   color=GOLD, alpha=0.25, zorder=1)

    ax.set_xlabel("week")
    ax.set_ylabel("weekly sales")
    ax.set_title(title or "Promo weeks (shaded) vs. baseline", fontsize=13, color=INK, pad=14)
    return ax


def plot_cluster_heatmap(profile: pd.DataFrame, feature_cols: list, labels: dict | None = None,
                          title: str = "Cluster profiles on key business features", ax=None):
    """z-scored heatmap of every clustering feature"""
    if ax is None:
        _, ax = plt.subplots(figsize=(16, 0.6 * len(profile) + 2))

    z = (profile[feature_cols] - profile[feature_cols].mean()) / profile[feature_cols].std(ddof=0)
    z.index = [_label_for(cid, labels) for cid in z.index]
    sns.heatmap(z, cmap="RdBu_r", center=0, annot=True, fmt=".1f", annot_kws={"size": 7},
                cbar_kws={"label": "z-score vs. cross-cluster mean"}, ax=ax)
    ax.set_title(title, fontsize=13, color=INK, pad=12)
    ax.set_xlabel(""); ax.set_ylabel("cluster")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    return ax


def plot_segment_sales_trend(weekly_by_cluster: pd.DataFrame, labels: dict | None = None, ax=None):
    """Stacked area of total weekly sales by cluster over time"""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5.5))

    dates = weekly_by_cluster.index
    cols = list(weekly_by_cluster.columns)
    names = [_label_for(c, labels) for c in cols]
    palette = plt.cm.tab10(np.linspace(0, 1, len(cols)))

    ax.stackplot(dates, *[weekly_by_cluster[c] for c in cols], labels=names, colors=palette, linewidth=0)
    ax.set_xlabel("week"); ax.set_ylabel("weekly sales")
    ax.set_title("Weekly sales by segment (stacked)", fontsize=13, color=INK, pad=14)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    return ax

def product_weekly_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean(df)
    pr_weekly = build_retailer_weekly_panel(df)
    return (
        pr_weekly.groupby(["rgm_ppg", "period_id"])
        .agg(sales=("sales", "sum"), on_promo=("on_promo", "max"))
        .reset_index()
    )


def plot_ppg_trend_grid(weekly: pd.DataFrame, ppg_ids: dict, feats: pd.DataFrame, ncols: int = 3,
                         axes=None, shared_axes: bool = True):
    n = len(ppg_ids)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    if axes is None:
        _, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.4 * nrows), squeeze=False)
    axes_flat = np.ravel(axes)

    if shared_axes:
        x_min, x_max = weekly["period_id"].min(), weekly["period_id"].max()
        y_max = weekly.loc[weekly["rgm_ppg"].isin(ppg_ids.values()), "sales"].max()

    for ax, (label, ppg) in zip(axes_flat, ppg_ids.items()):
        series = weekly[weekly["rgm_ppg"] == ppg].sort_values("period_id")
        ax.plot(series["period_id"], series["sales"], color=SLATE, linewidth=1.2, zorder=2)
        if "on_promo" in series.columns:
            for _, row in series[series["on_promo"] == 1].iterrows():
                ax.axvspan(row["period_id"] - pd.Timedelta(days=3), row["period_id"] + pd.Timedelta(days=3),
                           color=GOLD, alpha=0.2, zorder=1)

        row = feats.loc[ppg]
        line1 = f"{row.brand_nm} · {row.subcategory_nm}"
        line2 = f"{row.form} · {row.price_tier} · {row.benefit_type}"
        ax.set_title(f"{label}\n{line1}\n{line2}", fontsize=8.5, color=INK, linespacing=1.4)
        ax.set_ylabel("weekly sales", fontsize=9)
        ax.tick_params(labelsize=7.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        if shared_axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max * 1.05)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    return axes